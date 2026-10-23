"""Fail-closed DirectML qualification. Test tensors are numerical probes, not scans.

No accelerator fallback is permitted here. Reports contain metrics/hashes, never
radiograph pixels, filenames, or model output tensors. Verdicts live only for the
backend lifetime and are invalidated by driver, runtime, adapter or model changes.
"""
from functools import lru_cache
import gc
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np
import onnxruntime as ort

try:
    from . import processors, runtime
    from .models import models, cervical
except ImportError:  # Support uvicorn server:app from backend/.
    import processors
    import runtime
    from models import models, cervical

TOLERANCE = {'rtol': 2e-3, 'atol': 2e-3, 'landmark_px': .25, 'measurement': .1}
_VERDICTS = {}


def compare(reference, candidate):
    result = {'passed': False, 'elements': int(reference.size)}
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        return {**result, 'reason': 'shape_or_dtype'}
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        return {**result, 'reason': 'nonfinite'}
    delta = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    ratio = delta / (TOLERANCE['atol'] + TOLERANCE['rtol'] * np.abs(reference))
    worst = np.unravel_index(int(ratio.argmax()), ratio.shape) if ratio.size else ()
    return {**result, 'passed': bool((ratio <= 1).all()),
            'max_abs': float(delta.max(initial=0)),
            'max_rel': float((delta / np.maximum(np.abs(reference), 1e-30)).max(initial=0)),
            'tolerance_ratio': float(ratio.max(initial=0)), 'violations': int((ratio > 1).sum()),
            'worst_index': [int(i) for i in worst],
            'cpu_value': float(reference[worst]) if ratio.size else None}


def outputs(reference, candidate):
    if len(reference) != len(candidate):
        return {'passed': False, 'reason': 'output_count'}
    values = [compare(a, b) for a, b in zip(reference, candidate)]
    return {'passed': all(v['passed'] for v in values), 'outputs': values}


def decoded(kind, cpu, gpu, feeds):
    if any(a.shape != b.shape for a, b in zip(cpu, gpu)):
        return False
    if kind == 'hrnet':
        return bool(np.max(np.abs(cpu[0] - gpu[0]), initial=0) <= .05)
    if kind in ('lumbar_heatmaps', 'cervical_slope_hrnet'):
        from .decoders import local_coordinates
        a, b = [local_coordinates(v[0], unique_boundary=kind == 'cervical_slope_hrnet') for v in (cpu, gpu)]
        return bool(np.max(np.abs(a-b), initial=0) <= .0125)
    if kind == 's1':
        return bool(np.max(np.abs(cpu[1][..., :2] - gpu[1][..., :2]), initial=0) <= .05)
    if kind in ('cervical_hrnet', 'cervical_anatomical_hrnet'):
        return bool(np.array_equal(cpu[0].reshape(23, -1).argmax(1), gpu[0].reshape(23, -1).argmax(1)))
    if kind == 'vertebra':
        # Match production argmax labels before extracting extremal corners.
        a, b = cpu[0][0].argmax(0).astype(np.uint8), gpu[0][0].argmax(0).astype(np.uint8)
        try:
            from . import landmarks
        except ImportError:
            import landmarks
        values = dict(zip(models.LUMBAR_LEVELS, range(1, 6)))
        return (float(np.mean(a == b)) >= .9999 and
                all(tree_close(landmarks.corners_from_label_map(a, values, side),
                               landmarks.corners_from_label_map(b, values, side), 0)
                    for side in (np.array([-1., 0.]), np.array([1., 0.]))))
    if kind == 'femoral':
        masks = [np.asarray(1 / (1 + np.exp(-np.clip(v[0][0, 0], -80, 80))) >= models.FEMORAL_THRESHOLD, np.uint8)
                 for v in (cpu, gpu)]
        return float(np.mean(masks[0] == masks[1])) >= .9999 and cv2.connectedComponents(masks[0])[0] == cv2.connectedComponents(masks[1])[0]
    if kind == 'cervical_detr':
        shape = feeds['pixel_values'].shape[-2:]
        a, b = [cervical.detection_from_output(*v, shape) for v in (cpu, gpu)]
        # Query selection is checked even when the detector rejects the box.
        def query(v):
            z = v[0][0]; p = np.exp(z - z.max(1, keepdims=True)); return int((p[:, 0] / p.sum(1)).argmax())
        return query(cpu) == query(gpu) and ((a is None and b is None) or
            (a is not None and b is not None and abs(a['score'] - b['score']) <= .002 and
             np.max(np.abs(a['bbox'] - b['bbox'])) <= .25))
    return True


def tree_close(a, b, tolerance):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(tree_close(v, b[k], tolerance) for k, v in a.items())
    if isinstance(a, (list, tuple, np.ndarray)):
        return isinstance(b, (list, tuple, np.ndarray)) and len(a) == len(b) and all(tree_close(x, y, tolerance) for x, y in zip(a, b))
    if isinstance(a, (float, int, np.number)) and not isinstance(a, bool):
        return isinstance(b, (float, int, np.number)) and np.isfinite(a) and np.isfinite(b) and abs(a - b) <= tolerance
    return a == b


def probes(kind):
    size = models.FEMORAL_IMAGE_SIZE if kind == 'femoral' else models.MODEL_IMAGE_SIZE
    shape = (1, 3 if kind == 's1' else 1, size, size)
    if kind == 'cervical_detr':
        yield 'zeros', {'pixel_values': np.zeros((1, 3, 800, 800), np.float32), 'pixel_mask': np.ones((1, 800, 800), np.int64)}
    else:
        if kind == 'cervical_hrnet': shape = (1, 3, 384, 384)
        if kind in ('cervical_anatomical_hrnet', 'cervical_slope_hrnet'): shape = (1, 3, 512, 384)
        yield 'zeros', {'image': np.zeros(shape, np.float32)}
    rng = np.random.default_rng(40)
    y, x = np.mgrid[:768, :768]
    images = [np.zeros((768, 768), np.uint8), rng.integers(0, 256, (768, 768), dtype=np.uint8),
              (255 * np.exp(-((x-384)**2 + (y-384)**2) / 30000)).astype(np.uint8)]
    for name, image in zip(('blank_preprocessed', 'noise_preprocessed', 'smooth_preprocessed'), images):
        if kind == 'cervical_detr':
            feed = cervical.detector_input(image)
        elif kind in ('cervical_anatomical_hrnet', 'cervical_slope_hrnet'):
            from .models.cervical_anatomical import landmark_input
            feed = {'image': landmark_input(image)[0]}
        elif kind == 'cervical_hrnet':
            feed = {'image': cervical.landmark_input(image, [0, 0, 768, 768])[0]}
        elif kind == 'femoral': feed = {'image': models._femoral_input(image)}
        elif kind == 's1': feed = {'image': models._detection_input(image)}
        else: feed = {'image': models._segmentation_input(image)}
        yield name, feed
    if kind == 'cervical_detr':
        for h, w in ((800, 1067), (1067, 800)):
            yield f'aspect_{h}_{w}', cervical.detector_input(rng.integers(0, 256, (h, w), dtype=np.uint8))



def femoral_flip_check(cpu, gpu, feeds, reference, actual):
    flipped = {'image': np.ascontiguousarray(feeds['image'][..., ::-1])}
    a, b = cpu.run(None, flipped), gpu.run(None, flipped)
    masks = []
    for first, second in ((reference, a), (actual, b)):
        sigmoid = lambda z: 1 / (1 + np.exp(-np.clip(z, -80., 80.)))
        probability = (sigmoid(first[0][0, 0]) + sigmoid(second[0][0, 0])[:, ::-1]) * .5
        probability = cv2.resize(probability, (models.MODEL_IMAGE_SIZE, models.MODEL_IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        masks.append(np.asarray(probability >= models.FEMORAL_THRESHOLD, np.uint8))
    return (outputs(a, b)['passed'] and np.mean(masks[0] == masks[1]) >= .9999 and
            cv2.connectedComponents(masks[0])[0] == cv2.connectedComponents(masks[1])[0])


def new_session(kind, policy, gpu=False, profile=None):
    options = models.session_options(policy)
    if profile:
        options.enable_profiling = True
        options.profile_file_prefix = profile
    providers = ['CPUExecutionProvider']
    if gpu:
        providers.insert(0, (processors.DIRECTML, {'device_id': str(policy[2]), 'disable_metacommands': 'True'}))
    session = ort.InferenceSession(str(models.ONNX_DIRECTORY / f'{kind}.onnx'), sess_options=options,
                                  providers=providers, enable_fallback=0)
    if gpu and session.get_providers()[0] != processors.DIRECTML:
        raise RuntimeError('DirectML was not registered as the primary provider')
    return session


@lru_cache(maxsize=24)
def _hash(path, size, mtime):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''): digest.update(chunk)
    return digest.hexdigest()


def fingerprint(gpu):
    hashes = {}
    for kind in models.MODEL_NAMES:
        if kind == 's1_batch':
            continue  # CUDA-only throughput graph; DirectML uses the single detector.
        path = models.ONNX_DIRECTORY / f'{kind}.onnx'; stat = path.stat()
        hashes[kind] = _hash(str(path), stat.st_size, stat.st_mtime_ns)
    # Read live driver versions rather than ORT's cached device metadata. If this
    # cannot be read, qualification still runs but its verdict is never reused.
    try:
        driver = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
            'Get-CimInstance Win32_VideoController | Sort-Object PNPDeviceID | Select-Object PNPDeviceID,DriverVersion | ConvertTo-Json -Compress'],
            check=True, capture_output=True, text=True, timeout=10).stdout.strip()
        if not driver: driver = None
    except (OSError, subprocess.SubprocessError): driver = None
    return {'gpu': gpu.id, 'adapter': gpu.adapter, 'runtime': ort.__version__, 'driver': driver, 'models': hashes}


def verify_gpu(gpu):
    result = {'name': gpu.name, 'adapter': gpu.adapter, 'status': 'gpu_failed', 'passed': False,
              'fallback_cpu_bit_exact': True, 'models': {}}
    try:
        resolved, note = processors.resolve(gpu.id)
        if note or resolved.adapter != gpu.adapter: raise RuntimeError(note or 'GPU adapter changed during verification')
        result.update(fingerprint(gpu))
        threads = runtime.Options().inference_threads
        with tempfile.TemporaryDirectory(prefix='spine-gpu-parity-') as directory:
            for kind in models.MODEL_NAMES:
                if kind == 's1_batch':
                    continue
                runtime.checkpoint()
                runtime.report('gpu_verification', f'Checking GPU against CPU: {models.MODEL_NAMES[kind]}')
                for low in ([False, True] if kind in ('vertebra', 's1') else [False]):
                    cases = []
                    policy = (1 if low else threads, low, gpu.adapter)
                    cpu = new_session(kind, (policy[0], low, None))
                    fallback = new_session(kind, policy)
                    candidate = new_session(kind, policy, True, str(Path(directory) / f'{kind}-{low}'))
                    try:
                        for name, feeds in probes(kind):
                            runtime.checkpoint()
                            reference = cpu.run(None, feeds)
                            cpu_retry = fallback.run(None, feeds)
                            exact = len(reference) == len(cpu_retry) and all(np.array_equal(a, b) for a, b in zip(reference, cpu_retry))
                            result['fallback_cpu_bit_exact'] &= exact
                            actual = candidate.run(None, feeds)
                            repeated = candidate.run(None, feeds)
                            check = outputs(reference, actual)
                            repeat_check = outputs(reference, repeated)
                            repeat_delta = outputs(actual, repeated)
                            checks_decoded = decoded(kind, reference, actual, feeds) and decoded(kind, reference, repeated, feeds)
                            if kind == 'femoral':
                                checks_decoded &= femoral_flip_check(cpu, candidate, feeds, reference, actual)
                            cases.append({'input': name, 'low_memory': low, **check,
                                          'repeat': repeat_check, 'gpu_repeat_max_abs': max((v.get('max_abs', 0) for v in repeat_delta.get('outputs', [])), default=0),
                                          'decoded_passed': bool(checks_decoded), 'fallback_cpu_bit_exact': exact})
                    finally:
                        profile = json.loads(Path(candidate.end_profiling()).read_text())
                        placement = {}
                        for event in profile:
                            provider = event.get('args', {}).get('provider')
                            if provider: placement[provider] = placement.get(provider, 0) + 1
                        del cpu, fallback, candidate
                        gc.collect()
                    if not placement.get(processors.DIRECTML): raise RuntimeError(f'{kind}: no DirectML nodes executed')
                    result['models'][kind + ('_low_memory' if low else '')] = {
                        'cases': cases.copy(), 'node_placement': placement,
                        'passed': all(v['passed'] and v['repeat']['passed'] and v['decoded_passed'] and v['fallback_cpu_bit_exact'] for v in cases)}
        result['passed'] = result['fallback_cpu_bit_exact'] and all(v['passed'] for v in result['models'].values())
        result['status'] = 'passed' if result['passed'] else 'parity_failed'
    except runtime.Cancelled:
        raise
    except Exception as error:
        logging.getLogger(__name__).exception('GPU qualification failed for %s', gpu.id)
        result['error'] = f'{type(error).__name__}: {error}'
    return result


def ensure_verified(gpu):
    try:
        identity = fingerprint(gpu)
        key = json.dumps(identity, sort_keys=True)
        verdict = _VERDICTS.get(key) if identity['driver'] else None
        if verdict is None:
            models.release_models()
            verdict = verify_gpu(gpu)
            if identity['driver']: _VERDICTS[key] = verdict
        runtime.record_qualification({**identity, 'passed': verdict['passed']})
        if not verdict['passed']:
            raise runtime.GpuFailure('qualification', verdict.get('error', 'GPU outputs failed CPU parity checks'))
    except runtime.Cancelled:
        raise
    except runtime.GpuFailure:
        raise
    except Exception as error:
        raise runtime.GpuFailure('qualification', f'{type(error).__name__}: {error}') from error


def verify_all(gpu_id=None):
    gpus = processors.available()[1:]
    if gpu_id is not None:
        gpus = [g for g in gpus if g.id == gpu_id]
        if not gpus:
            return {'tolerance': TOLERANCE, 'passed': False, 'gpus': {gpu_id: {'status': 'gpu_failed', 'passed': False, 'error': 'GPU not found'}}}
    reports = {g.id: verify_gpu(g) for g in gpus}
    return {'tolerance': TOLERANCE, 'passed': all(v['passed'] for v in reports.values()), 'gpus': reports}


def verify_films(manifest, gpu_id):
    """Replay every real CPU feed, then compare whole-film production results.

    The manifest stays on the workstation. Only source hashes and numeric
    differences are emitted. A missing model/region is not silently skipped.
    """
    from dataclasses import replace
    from unittest.mock import patch
    try:
        from . import server
    except ImportError:
        import server
    path = Path(manifest)
    entries = json.loads(path.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError('Film manifest must be a non-empty list')
    gpu, note = processors.resolve(gpu_id)
    if note: raise RuntimeError(note)
    reports = []
    for index, entry in enumerate(entries):
        film_path = Path(entry['path'])
        if not film_path.is_absolute(): film_path = path.parent / film_path
        payload = film_path.read_bytes()
        settings = runtime.parse_options(entry.get('mode', 'standard'), entry.get('cpu_threads', 2),
                                         entry.get('localizer', True), entry.get('toolbar_removal', False))
        request = dict(payload=payload, modality='xray', body_part=entry['region'], view='lateral',
                       laterality=None, vertebra_model=entry.get('vertebra_model'), femoral_model=None,
                       s1_model=None, calibration=None, anterior_side=entry.get('anterior_side'), settings=settings)
        report = {'index': index, 'source_sha256': hashlib.sha256(payload).hexdigest(), 'passed': False, 'raw_feeds': []}
        # One replay session at a time; Auto and localizer searches may issue
        # hundreds of feeds. Do not retain tensors or all six GPU sessions.
        active = [None, None]
        original_run = models.InferenceModel.run
        def recorded_run(model, names, feeds):
            result = original_run(model, names, feeds)
            kind = model.path.stem
            if active[0] != kind:
                active[:] = [None, None]; gc.collect()
                active[:] = [kind, new_session(kind, (settings.inference_threads, settings.low_memory, gpu.adapter), True)]
            actual = active[1].run(names, feeds)
            repeated = active[1].run(names, feeds)
            check, repeat_check = outputs(result, actual), outputs(result, repeated)
            report['raw_feeds'].append({'kind': kind, **check, 'repeat': repeat_check,
                                       'decoded_passed': bool(decoded(kind, result, actual, feeds) and decoded(kind, result, repeated, feeds))})
            return result
        try:
            models.release_models()
            with patch.object(models.InferenceModel, 'run', recorded_run):
                reference = server.run_prediction(request)
            request['calibration'] = json.dumps(reference['calibration'])
            active[:] = [None, None]; models.release_models()
            # Qualification was performed by verify_all immediately before this
            # function. Skip requalification here, not request-level fallback.
            candidates = []
            for _ in range(2):
                with patch(__name__ + '.ensure_verified', lambda device: None):
                    candidates.append(server.run_prediction({**request, 'settings': replace(settings, processor=gpu_id)}))
                models.release_models()
            checks = []
            for candidate in candidates:
                processing = candidate['qc']['processing']
                checks.append({
                    'geometry': bool(tree_close(reference['geometry'], candidate['geometry'], TOLERANCE['landmark_px'])),
                    'measurements': bool(tree_close(reference['measurements'], candidate['measurements'], TOLERANCE['measurement'])),
                    'framing': bool(tree_close(reference['qc'].get('framing'), candidate['qc'].get('framing'), .25)),
                    'coverage': bool(tree_close(reference['qc'].get('coverage'), candidate['qc'].get('coverage'), 0)),
                    'detection': bool(tree_close(reference['qc'].get('film_detection'), candidate['qc'].get('film_detection'), .002)),
                    'gpu': processing['processor']['resolved'] == gpu_id and processing['processor']['note'] is None and
                           bool(processing['providers']) and all(processors.DIRECTML in p for p in processing['providers'].values()),
                })
            report['runs'] = checks
            report['passed'] = all(all(c.values()) for c in checks) and bool(report['raw_feeds']) and all(
                c['passed'] and c['repeat']['passed'] and c['decoded_passed'] for c in report['raw_feeds'])
        except runtime.Cancelled:
            raise
        except Exception as error:
            # Avoid leaking manifest paths or patient identifiers from exceptions.
            report['error_type'] = type(error).__name__
        finally:
            active[:] = [None, None]; models.release_models()
        reports.append(report)
    return reports
