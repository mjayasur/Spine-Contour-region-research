"""Fixed measurement routes; localization is independent of reader annotations.

Distinct measurement geometries are intentional: PT and global SVA must not
silently inherit the forced-upward crop used for LL, SS and PI.
"""
from copy import deepcopy
import numpy as np
from . import framing, runtime
from .models import models, full_spine, cervical_anatomical
from .pipeline_stages import production_stages

REVISION = 'seven-measurements-20261023'
ROUTES = {
    'cervical': {'model': 'cervical_slope_hrnet', 'epoch': 12, 'decoder': 'local9-temperature12', 'crop': 'automatic_neck'},
    'global_sva': {'model': 'cervical_anatomical_hrnet+hrnet', 'decoder': 'original', 'crop': 'original_plus_up15'},
    'lumbar': {'model': 'lumbar_heatmaps', 'epoch': 45, 'decoder': 'local9-temperature12', 'crop': 'all_up15'},
    'pelvic_tilt': {'model': 'lumbar_heatmaps', 'epoch': 45, 'decoder': 'local9-temperature12', 'crop': 'original_plus_up15'},
}


def crop_windows(window, shape, policy):
    original = full_spine.lumbar_windows(window, shape)
    extended = [(l, max(0, round(t-.15*(b-t))), r, b) for l, t, r, b in original]
    if policy not in ('all_up15', 'original_plus_up15'):
        raise ValueError('Unknown fixed crop policy')
    return sorted(set(extended if policy == 'all_up15' else original+extended))


def cervical_geometry(raw, window, side, kind, decoder):
    if window is None:
        return {'vertebrae': {}, 'c2_centroid': None, 'c7_centroid': None}
    l, t, r, b = map(int, window)
    crop = raw[t:b, l:r]
    image = crop if crop.dtype == np.uint8 else models._robust_rescale(crop)
    tensor, transform = cervical_anatomical.landmark_input(image)
    heat = models._infer(kind, lambda session: session.run(None, {'image': tensor})[0], 'Predicting cervical landmarks')
    q = decoder(heat)
    points = transform.restore(q)
    geometry = cervical_anatomical.anatomical_contract(points, crop.shape, side)
    for body in geometry['vertebrae'].values():
        for key, value in body.items():
            if value is not None:
                body[key] = (np.asarray(value)+[l, t]).tolist()
    if geometry.get('c2_centroid') is not None:
        geometry['c2_centroid'] = (np.asarray(geometry['c2_centroid'])+[l, t]).tolist()
    c7 = geometry['vertebrae'].get('C7', {}).get('quadrilateral')
    geometry['c7_centroid'] = np.mean(c7, axis=0).tolist() if c7 else None
    return geometry


def lumbar_geometries(raw, located, side, stages=None):
    """Share subpixel heatmaps; retain the original graph for global SVA."""
    stages = stages or production_stages()
    width = raw.shape[1]
    canonical = np.ascontiguousarray(raw[:, ::-1]) if side == 'right' else raw
    if located is None:
        return {key: ({'vertebrae': {}, 's1_superior': None, 'femoral_circles': []}, {})
                for key in ('lumbar', 'global_sva', 'pelvic_tilt')}
    both = crop_windows(located['window'], raw.shape, 'original_plus_up15')
    forced = set(crop_windows(located['window'], raw.shape, 'all_up15'))
    prepared = [framing.prepare_crop(canonical, window) for window in both]
    detections = stages.sacrum_detector([canvas for canvas, _ in prepared])
    inputs = np.concatenate([models._segmentation_input(canvas) for canvas, _ in prepared])
    heat = models._infer(stages.lumbar_graph, lambda session: session.run(None, {'image': inputs})[0],
                         'Predicting lumbar crop candidates')
    # Original graph includes its original decoder. Tiny numerical changes from
    # a re-export can move a near-tied crop medoid, so retain its full contract.
    original = models._infer(stages.global_lumbar_graph,
        lambda session: np.concatenate([session.run(None, {'image': item[None]})[0] for item in inputs]),
        'Predicting original global-SVA landmarks')
    decoded = {'original': original, 'subpixel': stages.lumbar_decoder(heat)}
    results, femoral_cache = {}, {}
    source = lambda p: full_spine._source_points(p, width, side == 'right').tolist()
    for route, decoder, windows in [('lumbar', 'subpixel', forced),
                                     ('global_sva', 'original', set(both)),
                                     ('pelvic_tilt', 'subpixel', set(both))]:
        candidates = []
        for i, (window, (_, transform), (score, detected)) in enumerate(zip(both, prepared, detections)):
            if window not in windows or detected is None or score < .5:
                continue
            detected = transform.restore_points(detected)
            points = transform.restore_points(decoded[decoder][i])
            if not full_spine._inside(detected, window) or not full_spine._inside(points, window):
                continue
            body_scale = full_spine._body_chain(points[:20])
            plate = points[20:22]
            scale = float(np.linalg.norm(plate[1]-plate[0]))
            if (body_scale is None or scale < 3 or not .5 < scale/body_scale < 2.5
                    or plate[1, 0] <= plate[0, 0] or plate[:, 1].mean() <= points[16:20, 1].mean()
                    or detected[1, 0] <= detected[0, 0]):
                continue
            discrepancy = float(np.linalg.norm(plate-detected, axis=1).max()/scale)
            if discrepancy <= .75:
                candidates.append({'anchor': plate[1], 'scale': scale, 'points': points,
                                   'endplate': plate, 'window': window, 'score': float(score)})
        pelvis, qc = full_spine.select_consensus(candidates)
        g = {'vertebrae': {}, 's1_superior': None, 'femoral_circles': []}
        if pelvis is not None:
            g['s1_superior'] = source(pelvis['endplate'])
            for level, body in enumerate(np.asarray(source(pelvis['points'][:20])).reshape(5, 4, 2), 1):
                g['vertebrae'][f'L{level}'] = {'superior': body[:2].tolist(), 'inferior': body[2:].tolist(),
                                              'quadrilateral': body[[0, 1, 3, 2]].tolist()}
            if route != 'global_sva':
                key = tuple(pelvis['window'])
                if key not in femoral_cache:
                    _, circles, fqc = stages.femoral_segmenter(canonical, pelvis)
                    circles = [c for c in circles if len(c) == 3 and np.isfinite(c).all()
                               and 0 <= c[0] < width and 0 <= c[1] < raw.shape[0] and c[2] > 0]
                    femoral_cache[key] = ([[*source(c[:2]), c[2]] for c in circles], fqc)
                g['femoral_circles'], qc['femoral'] = deepcopy(femoral_cache[key])
        qc.update(route=ROUTES[route], selected_window=None if pelvis is None else list(pelvis['window']))
        results[route] = (g, qc)
    return results


def localize(raw, anterior_side, stages, detection_evidence=None, region='full_spine'):
    searches = dict(detection_evidence or {})
    sides = ('left', 'right') if anterior_side not in ('left', 'right') else (anterior_side,)
    for side in sides:
        if side in searches:
            continue
        canonical = np.ascontiguousarray(raw[:, ::-1]) if side == 'right' else raw
        necks, neck_search = stages.cervical_candidates(canonical) if region != 'lumbar' else ([], {})
        located = stages.lumbar_locator(canonical)
        pelves, pelvis_search = full_spine._lumbar_candidates(canonical, located=located)
        neck, neck_qc = full_spine.select_consensus(necks)
        pelvis, pelvis_qc = full_spine.select_consensus(pelves)
        searches[side] = dict(anterior_side=side, neck=neck, pelvis=pelvis, neck_qc=neck_qc,
                             pelvis_qc=pelvis_qc, neck_search=neck_search, pelvis_search=pelvis_search,
                             neck_candidates=necks)
    if region != 'lumbar':
        searches = full_spine.reconcile_cervical_searches(raw, searches)
    side = anterior_side
    if side not in ('left', 'right'):
        side, _ = full_spine.select_orientation(searches)
        if side is None:
            side, _ = full_spine.anatomical_orientation(raw, searches)
    if side is None:
        raise ValueError('Automatic orientation is uncertain; choose anterior left or right.')
    selected = searches[side]
    neck = selected['neck']
    window = None if neck is None else list(neck['window'])
    if window is not None and side == 'right':
        l, t, r, b = window
        window = [raw.shape[1]-r, t, raw.shape[1]-l, b]
    return dict(anterior_side=side, cervical_window=window,
                lumbar_localizer=selected['pelvis_search'].get('localizer'))


def predict(raw, anterior_side=None, *, localization=None, detection_evidence=None, stages=None, region='full_spine'):
    """Run the fixed pipeline. Frozen localization is an internal benchmark seam.

    The HTTP API does not accept localization; application calls always infer it.
    """
    stages = stages or production_stages()
    raw = np.asarray(raw)
    if raw.ndim != 2 or not raw.size or not np.isfinite(raw).all():
        raise ValueError('Expected a finite grayscale radiograph')
    if region == 'cervical':
        window = stages.cervical_locator(raw)
        neck = cervical_geometry(raw, window, anterior_side, stages.cervical_graph, stages.cervical_decoder)
        side = anterior_side if anterior_side in ('left', 'right') else neck.get('anterior_side')
        if side is None:
            raise ValueError('Cervical orientation is unavailable; choose anterior left or right.')
        g = {**neck, 'region': 'cervical', 'anterior_side': side, 'pipeline_revision': REVISION}
        return {'image': raw.copy() if raw.dtype == np.uint8 else models._robust_rescale(raw),
                'mask': np.zeros(raw.shape, np.uint8), 'femoral_mask': np.zeros(raw.shape, np.uint8),
                'landmarks': g, 'models': {'vertebrae': stages.cervical_graph, 'detector': 'cervical_detr'},
                'framing': {'window': window}, 'provenance': {'pipeline': REVISION, 'routes': {'cervical': ROUTES['cervical']}}}
    if localization is None:
        evidence = localize(raw, anterior_side, stages, detection_evidence, region)
        side, window, located = evidence['anterior_side'], evidence['cervical_window'], evidence['lumbar_localizer']
    else:
        side, window = localization['anterior_side'], localization['cervical_window']
        located = deepcopy(localization['lumbar_localizer'])
    regional = lumbar_geometries(raw, located, side, stages)
    neck = cervical_geometry(raw, window, side, stages.cervical_graph, stages.cervical_decoder)
    original_neck = cervical_geometry(raw, window, side, stages.global_cervical_graph, stages.global_cervical_decoder)
    g = {'region': region, 'anterior_side': side, 'coordinate_space': 'original_image',
         'image_width': int(raw.shape[1]), 'image_height': int(raw.shape[0]),
         **regional['lumbar'][0], 'c2_centroid': neck.get('c2_centroid'),
         'c7_centroid': original_neck.get('c7_centroid')}
    g['vertebrae'].update(neck['vertebrae'])
    g['measurement_sources'] = {
        'global_sva': {'c7_centroid': original_neck.get('c7_centroid'),
                       's1_superior': regional['global_sva'][0]['s1_superior']},
        'pelvic_tilt': regional['pelvic_tilt'][0],
    }
    # Baseline geometry lets manual correction invalidate only affected routes.
    g['pipeline_revision'] = REVISION
    g['source_geometry'] = deepcopy({k: v for k, v in g.items() if k != 'measurement_sources'})
    return {'image': raw.copy() if raw.dtype == np.uint8 else models._robust_rescale(raw),
            'mask': np.zeros(raw.shape, np.uint8), 'femoral_mask': np.zeros(raw.shape, np.uint8),
            'landmarks': g, 'models': {'vertebrae': REVISION, 'cervical': 'cervical_slope_hrnet',
                                      'lumbar': 'lumbar_heatmaps', 'femoral': 'unet', 's1': 'keypointrcnn'},
            'framing': {'cervical_window': window, 'lumbar_localizer': located,
                        'measurement_routes': {k: qc for k, (_, qc) in regional.items()}},
            'provenance': {'pipeline': REVISION, 'routes': ROUTES, 'reference_landmarks_used': False,
                           'localization': 'fresh' if localization is None else 'frozen_image_derived'},
            'warnings': []}
