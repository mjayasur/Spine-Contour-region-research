"""Bounded image-only search for standing-film regional and global landmarks.

The two HRNETs operate on overlapping regional crops of an upright lateral
radiograph. Detectors propose crops; independent crop agreement gates the
landmarks. Agreement is a repeatability check, not calibrated confidence or
proof of vertebral identity. Every result requires anatomical review.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from . import cervical, models
try:
    from .. import framing, runtime, learned_region
except ImportError:
    import framing
    import runtime
    import learned_region

MODEL_NAME = "dual_hrnet"
MAX_CERVICAL_CROPS = 12
MIN_SUPPORT = 2
CONSENSUS_RADIUS = .75  # in local body/endplate widths; not physical units


def cervical_windows(height, width):
    """Overlap upper-film windows at several scales; retain source origins."""
    windows = set()
    for fraction in (.18, .28, .40):
        h = max(64, round(height * fraction))
        w = min(width, h)
        ys = np.arange(0, max(1, round(height * .55) - h + 1), max(1, h // 2))
        ys = np.unique(np.append(ys, max(0, min(height-h, round(height*.55)-h))))
        for y in ys:
            for x in np.linspace(0, max(0, width-w), 3):
                left, top = int(round(x)), int(y)
                windows.add((left, top, min(width, left+w), min(height, top+h)))
    return sorted(w for w in windows if w[2]-w[0] >= 32 and w[3]-w[1] >= 32)


def _inside(points, window):
    p = np.asarray(points)
    left, top, right, bottom = window
    return (p.ndim == 2 and p.shape[1] == 2 and np.isfinite(p).all()
            and (p[:, 0] >= left).all() and (p[:, 0] <= right-1).all()
            and (p[:, 1] >= top).all() and (p[:, 1] <= bottom-1).all())


def _body_chain(points):
    """Broad geometric rejection of collapsed, reversed, or unordered bodies."""
    bodies = np.asarray(points, np.float64).reshape(-1, 4, 2)
    widths = (np.linalg.norm(bodies[:, 1]-bodies[:, 0], axis=1)
              + np.linalg.norm(bodies[:, 3]-bodies[:, 2], axis=1))/2
    heights = bodies[:, 2:, 1].mean(1)-bodies[:, :2, 1].mean(1)
    centres = bodies.mean(1)
    scale = float(np.median(widths))
    if (not np.isfinite(bodies).all() or scale < 3 or (widths < scale*.3).any()
            or (widths > scale*3).any() or (heights < widths*.1).any()
            or (heights > widths*2.5).any() or (np.diff(centres[:, 1]) < scale*.2).any()):
        return None
    # The model frame is canonical anterior-left; do not fix crossed endpoints.
    if ((bodies[:, 1, 0] <= bodies[:, 0, 0]).any()
            or (bodies[:, 3, 0] <= bodies[:, 2, 0]).any()):
        return None
    polygon = bodies[:, [0, 1, 3, 2], :]
    edges = np.roll(polygon, -1, axis=1)-polygon
    turns = edges[..., 0]*np.roll(edges[..., 1], -1, axis=1)-edges[..., 1]*np.roll(edges[..., 0], -1, axis=1)
    if not ((turns > 1e-6).all(1) | (turns < -1e-6).all(1)).all():
        return None
    return scale


def select_consensus(candidates):
    """Choose an observed medoid, never average incompatible level assignments."""
    if not candidates:
        return None, {"candidates": 0, "support": 0, "status": "not_found"}
    anchors = np.array([c["anchor"] for c in candidates], np.float64)
    scales = np.array([c["scale"] for c in candidates], np.float64)
    distance = np.linalg.norm(anchors[:, None]-anchors[None, :], axis=-1)
    agreement = distance <= CONSENSUS_RADIUS * np.minimum(scales[:, None], scales[None, :])
    if all("points" in candidate for candidate in candidates):
        # Regional measurements now use every retained endplate. Agreement on
        # C7 or S1 alone cannot corroborate inconsistent C2/L1 assignments.
        points = np.asarray([candidate["points"] for candidate in candidates], np.float64)
        regional_distance = np.linalg.norm(points[:, None]-points[None, :], axis=-1).max(axis=-1)
        agreement &= regional_distance <= CONSENSUS_RADIUS * np.minimum(scales[:, None], scales[None, :])
    support = agreement.sum(1)
    order = sorted(range(len(candidates)), key=lambda i: (
        -int(support[i]), float(np.mean(distance[i, agreement[i]])),
        -candidates[i]["score"]))
    best = order[0]
    info = {"candidates": len(candidates), "support": int(support[best]),
            "status": "accepted", "spread_px": float(distance[best, agreement[best]].max())}
    if support[best] < MIN_SUPPORT:
        return None, {**info, "status": "insufficient_agreement"}
    # Similar-sized distant clusters may be different vertebrae. Withhold them.
    rivals = [i for i in order[1:] if not agreement[best, i]
              and support[i] >= max(MIN_SUPPORT, .75*support[best])]
    if rivals:
        return None, {**info, "status": "ambiguous_clusters"}
    return candidates[best], info


def unique_cervical_crops(proposals):
    """Identical source rectangles provide only one spatial observation."""
    chosen = {}
    for proposal in sorted(proposals, key=lambda p: -p["score"]):
        ox, oy = proposal["origin"]
        t = proposal["transform"]
        source_crop = (ox+t.x0, oy+t.y0, ox+t.x1, oy+t.y1)
        if source_crop not in chosen:
            chosen[source_crop] = {**proposal, "source_crop": source_crop}
    return list(chosen.values())


def _model_cervical_candidates(raw, found):
    """Evaluate bounded detector-guided crops with the cervical landmark model."""
    height, width = raw.shape
    boxes = learned_region.boxes(found, "cervical", raw.shape, limit=3)
    windows = set()
    for box in boxes:
        left, top, right, bottom = box
        side = min(max(right-left, bottom-top)*1.05, width, height)
        cx, cy = (left+right)/2, (top+bottom)/2
        for dx, dy, scale in ((0, 0, 1), (-.04, 0, 1.04), (.04, 0, 1.04)):
            size = max(32, min(round(side*scale), width, height))
            x0 = round(np.clip(cx+dx*side-size/2, 0, width-size))
            y0 = round(np.clip(cy+dy*side-size/2, 0, height-size))
            window = (x0, y0, x0+size, y0+size)
            windows.add(window)
    candidates = []
    invalid = 0
    def predict(session):
        nonlocal invalid
        for index, window in enumerate(sorted(windows)):
            runtime.report("landmarks", "Checking model-selected cervical crops", index, len(windows))
            left, top, right, bottom = window
            crop = raw[top:bottom, left:right]
            crop = crop if crop.dtype == np.uint8 else models._robust_rescale(crop)
            resized = np.asarray(Image.fromarray(crop).resize((384, 384), Image.Resampling.BILINEAR))
            heat = session.run(None, {"image": cervical._rgb_tensor(resized)})[0]
            try:
                points = cervical.decode_heatmaps(heat) * ((right-left)/96, (bottom-top)/96) + (left, top)
            except ValueError:
                invalid += 1
                continue
            if not _inside(points, window):
                continue
            body_scale = _body_chain(points[3:])
            if body_scale is None or points[2, 1] >= points[3:7, 1].mean():
                continue
            body = points[19:23]
            c7_scale = float((np.linalg.norm(body[1]-body[0])+np.linalg.norm(body[3]-body[2]))/2)
            candidates.append({"anchor": body.mean(0), "scale": c7_scale,
                               "points": points, "window": window, "score": 1.})
    if windows:
        models._infer("cervical_hrnet", predict, "Locating C7 with HRNET")
    return candidates, {"windows": len(windows), "hrnet_crops": len(windows),
                        "invalid_outputs": invalid,
                        "model_proposals": [list(box) for box in boxes]}


def _cervical_candidates(raw, _found=None, *, force_search=False):
    if runtime.options().crop_method == "model" and not force_search:
        return _model_cervical_candidates(
            raw, learned_region.proposals(raw) if _found is None else _found)
    height, width = raw.shape
    windows = cervical_windows(height, width)
    proposals = []
    # Group calls by graph to keep low-memory mode from swapping on every crop.
    def detect(session):
        for index, window in enumerate(windows):
            runtime.report("search", "Searching upper-spine crops", index, len(windows))
            left, top, right, bottom = window
            crop = raw[top:bottom, left:right]
            crop = crop.copy() if crop.dtype == np.uint8 else models._robust_rescale(crop)
            logits, boxes = session.run(None, cervical.detector_input(crop))
            found = cervical.detection_from_output(logits, boxes, crop.shape, threshold=.2)
            if found is None:
                continue
            prepared = cervical.landmark_input(crop, found["bbox"])
            if prepared is not None:
                tensor, transform = prepared
                proposals.append({"tensor": tensor, "transform": transform,
                                  "origin": [left, top], "window": window,
                                  "score": found["score"]})
        runtime.report("search", "Upper-spine crop search complete", len(windows), len(windows))
    models._infer("cervical_detr", detect, "Locating the cervical spine")
    selected = unique_cervical_crops(proposals)[:MAX_CERVICAL_CROPS]
    candidates = []
    invalid_outputs = 0
    if selected:
        def predict(session):
            nonlocal invalid_outputs
            for index, proposal in enumerate(selected):
                runtime.report("landmarks", "Comparing cervical HRNET crops", index, len(selected))
                heat = session.run(None, {"image": proposal["tensor"]})[0]
                try:
                    decoded = cervical.decode_heatmaps(heat)
                except ValueError:
                    invalid_outputs += 1
                    continue
                points = proposal["transform"].restore(decoded) + proposal["origin"]
                if not _inside(points, proposal["window"]):
                    continue
                scale = _body_chain(points[3:])
                if scale is None or points[2, 1] >= points[3:7, 1].mean():
                    continue
                body = points[19:23]
                c7_scale = float((np.linalg.norm(body[1]-body[0])+np.linalg.norm(body[3]-body[2]))/2)
                candidates.append({"anchor": body.mean(0), "scale": c7_scale,
                                   "points": points, "window": proposal["source_crop"],
                                   "score": proposal["score"]})
        models._infer("cervical_hrnet", predict, "Locating C7 with HRNET")
    return candidates, {"windows": len(windows), "detections": len(proposals),
                        "hrnet_crops": len(selected), "invalid_outputs": invalid_outputs}


def lumbar_windows(window, shape):
    """Small translations and scale changes around the detector-selected crop."""
    left, top, right, bottom = window
    height, width = shape
    w, h = right-left, bottom-top
    cx, cy = (left+right)/2, (top+bottom)/2
    variants = [(0, 0, 1), (-.06, 0, 1), (.06, 0, 1), (0, -.06, 1),
                (0, .06, 1), (0, 0, .9), (0, 0, 1.1)]
    return sorted({framing.clip_window(cx+dx*w, cy+dy*h, w*scale, h*scale, height, width)
                   for dx, dy, scale in variants})


def _lumbar_candidates(raw, _found=None, *, force_search=False, located=Ellipsis):
    # Existing image-only sliding detector establishes a lumbosacral crop.
    learned_boxes = []
    if runtime.options().crop_method == "model" and not force_search:
        learned_boxes = learned_region.boxes(
            learned_region.proposals(raw) if _found is None else _found, "lumbar", raw.shape, limit=3)
    if located is not Ellipsis:
        pass
    elif learned_boxes:
        located = {"window": learned_boxes[0], "source": "crop_detector"}
    elif runtime.options().crop_method == "model" and not force_search:
        located = None
    else:
        located = framing.locate(raw, models._score_s1)
    if located is None:
        return [], {"windows": 0, "hrnet_crops": 0, "localizer": None,
                    "model_proposal": None}
    windows = sorted({window for box in (learned_boxes or [located["window"]])
                      for window in lumbar_windows(box, raw.shape)})
    prepared = [framing.prepare_crop(raw, window) for window in windows]
    detections = models._score_s1([canvas for canvas, _ in prepared])
    proposals = []
    for window, (canvas, transform), (score, detected) in zip(windows, prepared, detections):
        if detected is None or score < .5:
            continue
        detected_source = transform.restore_points(detected)
        if _inside(detected_source, window):
            proposals.append((window, canvas, transform, score, detected_source))
    candidates = []
    if proposals:
        def predict(session):
            for index, (window, canvas, transform, score, detected) in enumerate(proposals):
                runtime.report("landmarks", "Comparing lumbar HRNET crops", index, len(proposals))
                points = session.run(None, {"image": models._segmentation_input(canvas)})[0][0]
                if points.shape != (22, 2) or not np.isfinite(points).all():
                    continue
                points = transform.restore_points(points)
                if not _inside(points, window):
                    continue
                body_scale = _body_chain(points[:20])
                endplate = points[20:22]
                scale = float(np.linalg.norm(endplate[1]-endplate[0]))
                if (body_scale is None or scale < 3 or not .5 < scale/body_scale < 2.5
                        or endplate[1, 0] <= endplate[0, 0]
                        or endplate[:, 1].mean() <= points[16:20, 1].mean()):
                    continue
                # HRNET supplies S1; an independent detector corroborates its location.
                # Keypoint identities are anatomical [anterior, posterior].
                # Sorting by x would erase evidence of the wrong orientation.
                if detected[1, 0] <= detected[0, 0]:
                    continue
                discrepancy = float(np.linalg.norm(endplate-detected, axis=1).max()/scale)
                if discrepancy > .75:
                    continue
                candidates.append({"anchor": endplate[1], "scale": scale,
                                   "points": points, "endplate": endplate,
                                   "window": window, "score": float(score),
                                   "detector_disagreement_widths": discrepancy})
        models._infer("hrnet", predict, "Locating the S1 endplate with HRNET")
    return candidates, {"windows": len(windows), "hrnet_crops": len(proposals),
                        "localizer": located,
                        "model_proposals": [list(box) for box in learned_boxes]}


def _source_points(points, width, mirrored):
    result = np.asarray(points, np.float64).copy()
    if mirrored:
        result[..., 0] = width-1-result[..., 0]
    return result


def search_orientation(raw, anterior_side):
    """Repeatable regional evidence for one anterior-side hypothesis.

    This is shared with film classification so its expensive crop searches can
    be reused for the eventual standing-film prediction.
    """
    if anterior_side not in ("left", "right"):
        raise ValueError("An orientation hypothesis must be left or right")
    canonical = np.ascontiguousarray(raw[:, ::-1]) if anterior_side == "right" else raw
    if runtime.options().crop_method == "model":
        found = learned_region.proposals(canonical)
        neck_candidates, neck_search = _cervical_candidates(canonical, found)
        pelvis_candidates, pelvis_search = _lumbar_candidates(canonical, found)
    else:
        neck_candidates, neck_search = _cervical_candidates(canonical)
        pelvis_candidates, pelvis_search = _lumbar_candidates(canonical)
    neck, neck_qc = select_consensus(neck_candidates)
    pelvis, pelvis_qc = select_consensus(pelvis_candidates)
    if runtime.options().crop_method == "model":
        if neck is None:
            runtime.report("search", "Model cervical crop was inconclusive; searching upper spine")
            model_search = neck_search
            model_status = neck_qc["status"]
            neck_candidates, neck_search = _cervical_candidates(canonical, force_search=True)
            neck, neck_qc = select_consensus(neck_candidates)
            neck_search = {**neck_search, "method_used": "search_fallback",
                           "model_fallback_reason": model_status,
                           "model_proposals": model_search.get("model_proposals", [])}
        else:
            neck_search = {**neck_search, "method_used": "model"}
        if pelvis is None:
            runtime.report("search", "Model lumbar crop was inconclusive; searching lower spine")
            model_search = pelvis_search
            model_status = pelvis_qc["status"]
            pelvis_candidates, pelvis_search = _lumbar_candidates(canonical, force_search=True)
            pelvis, pelvis_qc = select_consensus(pelvis_candidates)
            pelvis_search = {**pelvis_search, "method_used": "search_fallback",
                             "model_fallback_reason": model_status,
                             "model_proposals": model_search.get("model_proposals", [])}
        else:
            pelvis_search = {**pelvis_search, "method_used": "model"}
    else:
        neck_search = {**neck_search, "method_used": "search"}
        pelvis_search = {**pelvis_search, "method_used": "search"}
    return {"anterior_side": anterior_side, "neck": neck, "pelvis": pelvis,
            "neck_qc": neck_qc, "pelvis_qc": pelvis_qc,
            "neck_search": neck_search, "pelvis_search": pelvis_search,
            "neck_candidates": neck_candidates}


def _mirror_cervical_candidate(candidate, width):
    """Reflect image-sided cervical points while preserving their index labels."""
    order = [1, 0, 2] + [index for start in range(3, 23, 4)
                         for index in (start+1, start, start+3, start+2)]
    left, top, right, bottom = candidate["window"]
    return {**candidate, "anchor": _source_points(candidate["anchor"], width, True),
            "points": _source_points(candidate["points"], width, True)[order],
            "window": (width-right, top, width-left, bottom)}


def reconcile_cervical_searches(raw, searches, *, _allow_model_fallback=True):
    """Share one cervical identity across mirror hypotheses in source space.

    Image-sided cervical landmarks cannot vote on anterior direction. A local
    detector may mistake thoracic bodies for the neck in just one mirror view;
    require the complete chosen chain to agree in both views as well as across
    distinct source crops. Explicit side overrides retain this anatomy check.
    """
    if not searches or not any("neck_candidates" in value for value in searches.values()):
        return searches
    width = raw.shape[1]
    candidates_by_side = {}
    for side in ("left", "right"):
        if side in searches and "neck_candidates" in searches[side]:
            candidates_by_side[side] = searches[side]["neck_candidates"]
        else:
            canonical = raw if side == "left" else np.ascontiguousarray(raw[:, ::-1])
            candidates_by_side[side], _ = _cervical_candidates(canonical)
    unique, conflicted = {}, set()
    for side, candidates in candidates_by_side.items():
        for candidate in candidates:
            source = (_mirror_cervical_candidate(candidate, width) if side == "right"
                      else dict(candidate))
            key = tuple(source["window"])
            if key in conflicted:
                continue
            if key not in unique:
                unique[key] = {**source, "_orientation_sides": {side}}
            else:
                previous = unique[key]
                radius = CONSENSUS_RADIUS*min(source["scale"], previous["scale"])
                if (np.linalg.norm(source["anchor"]-previous["anchor"]) > radius
                        or np.linalg.norm(source["points"]-previous["points"], axis=1).max() > radius):
                    # One rectangle supplies only one spatial observation. A
                    # conflicting reflection cannot lend its side to the
                    # higher-scoring prediction from that same rectangle.
                    del unique[key]
                    conflicted.add(key)
                    continue
                chosen = source if source["score"] > previous["score"] else previous
                unique[key] = {**chosen, "_orientation_sides": previous["_orientation_sides"] | {side}}
    candidates = list(unique.values())
    selected, qc = select_consensus(candidates)
    orientation_support = set()
    if selected is not None:
        for candidate in candidates:
            radius = CONSENSUS_RADIUS*min(candidate["scale"], selected["scale"])
            if (np.linalg.norm(candidate["anchor"]-selected["anchor"]) <= radius
                    and np.linalg.norm(candidate["points"]-selected["points"], axis=1).max() <= radius):
                orientation_support.update(candidate["_orientation_sides"])
        if orientation_support != {"left", "right"}:
            selected = None
            qc = {**qc, "status": "unconfirmed_across_mirrors"}
    qc = {**qc, "orientation_support": sorted(orientation_support),
          "conflicted_source_crops": len(conflicted),
          "method": "shared_source_cervical_consensus"}
    reconciled = {}
    for side, evidence in searches.items():
        neck = selected
        if selected is not None and side == "right":
            neck = _mirror_cervical_candidate(selected, width)
        regional_qc = dict(qc)
        pelvis = evidence.get("pelvis")
        if neck is not None and pelvis is not None:
            if pelvis["anchor"][1]-neck["anchor"][1] <= 2*max(neck["scale"], pelvis["scale"]):
                neck = None
                regional_qc["status"] = "incompatible_region_order"
        reconciled[side] = {**evidence, "neck": neck, "neck_qc": regional_qc}
    if (_allow_model_fallback and runtime.options().crop_method == "model"
            and any(value["neck"] is None and
                    value.get("neck_search", {}).get("method_used") == "model"
                    for value in reconciled.values())):
        runtime.report("search", "Model cervical crops did not agree; trying crop search")
        fallback_searches = dict(reconciled)
        for side in ("left", "right"):
            evidence = fallback_searches.get(side)
            if evidence is not None and evidence.get("neck_search", {}).get("method_used") == "search_fallback":
                continue
            canonical = raw if side == "left" else np.ascontiguousarray(raw[:, ::-1])
            candidates, search_info = _cervical_candidates(canonical, force_search=True)
            selected, search_qc = select_consensus(candidates)
            previous = {} if evidence is None else evidence.get("neck_search", {})
            fallback_searches[side] = {**(evidence or {}), "anterior_side": side,
                                       "neck": selected, "neck_qc": search_qc,
                                       "neck_candidates": candidates,
                                       "neck_search": {**search_info,
                                                       "method_used": "search_fallback",
                                                       "model_fallback_reason": (
                                                           evidence.get("neck_qc", {}).get("status")
                                                           if evidence is not None else
                                                           next(iter(reconciled.values()))["neck_qc"]["status"]),
                                                       "model_proposals": previous.get("model_proposals", [])}}
        reviewed = reconcile_cervical_searches(
            raw, fallback_searches, _allow_model_fallback=False)
        return {side: reviewed[side] for side in searches}
    return reconciled


def select_orientation(searches):
    """Select only a uniquely better supported orientation; never break ties.

    Crop agreement is a repeatability gate, not calibrated probability. A
    higher detector score or more overlapping crops alone cannot resolve two
    hypotheses with matching anatomical endpoint identities. Cervical labels
    are image-sided; only the anatomical S1 detector order provides evidence
    for anterior direction, corroborated by the lumbar HRNET chain.
    """
    hypotheses = {}
    for side, evidence in searches.items():
        regions = [name for name, key in (("cervical", "neck"), ("lumbar", "pelvis"))
                   if evidence.get(key) is not None]
        hypotheses[side] = {"accepted_regions": regions,
                            "cervical_support": evidence.get("neck_qc", {}).get("support", 0),
                            "lumbar_support": evidence.get("pelvis_qc", {}).get("support", 0)}
    selected = None
    if set(hypotheses) == {"left", "right"}:
        supported = [side for side in ("left", "right")
                     if "lumbar" in hypotheses[side]["accepted_regions"]]
        if len(supported) == 1:
            selected = supported[0]
    return selected, {"status": "accepted" if selected else "ambiguous",
                      "source": "automatic", "anterior_side": selected,
                      "method": "mirrored_regional_consensus", "hypotheses": hypotheses,
                      "review_required": True}


def anatomical_orientation(raw, searches):
    side, qc = _anatomical_orientation_crop(raw, searches)
    neck = searches.get("left", {}).get("neck")
    if side is not None or neck is None or searches.get("right", {}).get("neck") is None:
        return side, qc
    # The anatomical model was trained on cervical films, not tight detector
    # crops. Require two contextual windows to independently satisfy all of
    # the same mirror and landmark checks before recovering a failed crop.
    height, width = raw.shape
    left, top, right, bottom = neck["window"]
    checks = []
    for margin in (.1, .2):
        dx, dy = (right-left)*margin, (bottom-top)*margin
        window = [max(0, int(left-dx)), max(0, int(top-dy)),
                  min(width, int(right+dx)), min(height, int(bottom+dy))]
        expanded = {**searches, "left": {**searches["left"],
                    "neck": {**neck, "window": window}}}
        candidate, evidence = _anatomical_orientation_crop(raw, expanded)
        checks.append({"window": window, "anterior_side": candidate, **evidence})
    if (checks[0]["window"] != checks[1]["window"]
            and checks[0]["anterior_side"] is not None
            and checks[0]["anterior_side"] == checks[1]["anterior_side"]):
        return checks[0]["anterior_side"], {
            **checks[0], "context_retry": True, "context_checks": checks,
            "initial_crop_status": qc["status"]}
    return None, {**qc, "context_checks": checks}


def _anatomical_orientation_crop(raw, searches):
    """Corroborate an agreed cervical crop with anatomical, mirror-stable labels.

    This is only a fallback after regional orientation fails. The old crop
    consensus must first establish visible cervical anatomy; the anatomical
    model is never asked to find a neck in an arbitrary standing-film window.
    """
    from . import cervical_anatomical
    neck = searches.get("left", {}).get("neck")
    if neck is None or searches.get("right", {}).get("neck") is None:
        return None, {"status": "no_cervical_consensus"}
    left, top, right, bottom = neck["window"]
    crop = np.asarray(raw)[top:bottom, left:right]
    observations = []
    for mirrored in (False, True):
        image = np.ascontiguousarray(crop[:, ::-1]) if mirrored else crop
        try:
            geometry = cervical_anatomical.cervical_prediction(image)["landmarks"]
        except ValueError as error:
            return None, {"status": "unusable_anatomical_cervical", "reason": str(error)}
        pairs = [geometry["vertebrae"][f"C{level}"].get(end)
                 for level in range(3, 8) for end in ("superior", "inferior")]
        if any(pair is None for pair in pairs):
            return None, {"status": "incomplete_anatomical_cervical"}
        points = np.asarray(pairs, dtype=float).reshape(20, 2)
        if mirrored:
            points[:, 0] = crop.shape[1]-1-points[:, 0]
        observations.append(points + [left, top])
    original, reflected = observations
    radius = CONSENSUS_RADIUS*neck["scale"]
    if np.linalg.norm(original-reflected, axis=1).max() > radius:
        return None, {"status": "anatomical_mirror_disagreement"}
    # Compare locations without assuming that the old image-sided endpoints
    # already identify anterior/posterior. Never reorder the anatomical output.
    legacy = np.asarray(neck["points"])[3:].reshape(10, 2, 2)
    pairs = original.reshape(10, 2, 2)
    direct = np.linalg.norm(pairs-legacy, axis=2).max(axis=1)
    swapped = np.linalg.norm(pairs-legacy[:, ::-1], axis=2).max(axis=1)
    if np.minimum(direct, swapped).max() > radius:
        return None, {"status": "anatomical_crop_disagreement"}
    deltas = np.concatenate([(p.reshape(10, 2, 2)[:, 0, 0]-
                              p.reshape(10, 2, 2)[:, 1, 0]) for p in observations])
    side = "right" if np.median(deltas) > 0 else "left"
    if np.mean(deltas > 0 if side == "right" else deltas < 0) < .9:
        return None, {"status": "anatomical_direction_disagreement"}
    return side, {"status": "accepted", "source": "automatic", "anterior_side": side,
                  "method": "anatomical_cervical_mirror_consensus", "review_required": True,
                  "model": cervical_anatomical.MODEL_NAME,
                  "mirror_max_distance_px": float(np.linalg.norm(original-reflected, axis=1).max())}


def _femoral_region(canonical, pelvis):
    """Read the heads without treating the lumbar crop boundary as anatomy.

    Consensus chooses a crop for vertebral landmarks. That crop can bisect a
    head even when the source radiograph contains it in full. Expand only the
    sides reached by the predicted mask, with at most two additional passes.
    Remaining crop truncation must request review just like a source-frame cut.
    """
    try:
        from ..utils import _femoral_geometry, EDGE_TOUCH_MAX_CONFIDENCE
    except ImportError:
        from utils import _femoral_geometry, EDGE_TOUCH_MAX_CONFIDENCE
    if pelvis is None:
        return np.zeros(canonical.shape, np.uint8), [], {
            "qc_pass": False, "reason": "No consistent lumbar crop was established."}
    initial = tuple(pelvis["window"])
    window = initial
    height, width = canonical.shape
    for attempt in range(3):
        left, top, right, bottom = window
        _, transform = framing.prepare_crop(canonical, window)
        probability = models._femoral_probabilities(canonical[top:bottom, left:right])
        mask = models._restore_femoral_mask(probability, transform, canonical.shape)
        touches = [bool(mask[top:bottom, left].any()), bool(mask[top, left:right].any()),
                   bool(mask[top:bottom, right-1].any()), bool(mask[bottom-1, left:right].any())]
        dx, dy = max(1, round((right-left) * .25)), max(1, round((bottom-top) * .25))
        expanded = (max(0, left-dx) if touches[0] else left,
                    max(0, top-dy) if touches[1] else top,
                    min(width, right+dx) if touches[2] else right,
                    min(height, bottom+dy) if touches[3] else bottom)
        if expanded == window or attempt == 2:
            break
        runtime.report("femoral", "Extending the crop to include the femoral heads")
        window = expanded
    # These windows are canonical here; the caller reflects them with the mask.
    crop_qc = {"initial_crop_window": list(initial), "crop_window": list(window),
               "crop_expansions": attempt, "touches_crop_edge": any(touches)}
    try:
        _, circles, qc = _femoral_geometry(mask)
        if any(touches):
            qc["confidence"] = min(qc["confidence"], EDGE_TOUCH_MAX_CONFIDENCE)
        return mask, [np.asarray(circle).tolist() for circle in circles] if qc.get("qc_pass") else [], {**qc, **crop_qc}
    except ValueError as error:
        return mask, [], {"qc_pass": False, "confidence": None, "reason": str(error), **crop_qc}


def full_spine_prediction(pixel_array, anterior_side=None, model=MODEL_NAME, *, detection_evidence=None):
    """Infer source-coordinate anatomy with automatic or explicit orientation.

    ``detection_evidence`` is internal, same-image crop-search output from film
    classification. It avoids running the two expensive orientation searches
    again and is never accepted as user-supplied anatomical geometry.
    """
    if anterior_side == "":
        anterior_side = None
    if anterior_side not in (None, "auto", "left", "right"):
        raise ValueError("The anterior image side must be auto, left or right")
    if model != MODEL_NAME:
        raise ValueError("The available full-spine model is dual_hrnet")
    raw = np.asarray(pixel_array)
    if (raw.ndim != 2 or not raw.size or not np.issubdtype(raw.dtype, np.number)
            or not np.isfinite(raw).all()):
        raise ValueError("pixel_array must contain finite two-dimensional grayscale pixels")
    runtime.report("preparing", "Preparing the full-spine radiograph")
    height, width = raw.shape
    automatic = anterior_side in (None, "auto")
    searches = dict(detection_evidence or {})
    for side in (("left", "right") if automatic else (anterior_side,)):
        if side not in searches:
            runtime.report("orientation", f"Checking anterior-{side} orientation")
            searches[side] = search_orientation(raw, side)
    searches = reconcile_cervical_searches(raw, searches)
    if automatic:
        anterior_side, orientation = select_orientation(searches)
        if anterior_side is None:
            anterior_side, anatomical_qc = anatomical_orientation(raw, searches)
            if anterior_side is not None:
                orientation = {**anatomical_qc, "regional_hypotheses": orientation["hypotheses"]}
        if anterior_side is None:
            raise ValueError("Automatic standing-film orientation is uncertain. Select anterior left or right and retry.")
    else:
        orientation = {"status": "user_selected", "source": "user", "anterior_side": anterior_side,
                       "review_required": True}
    evidence = searches[anterior_side]
    neck, pelvis = evidence["neck"], evidence["pelvis"]
    neck_qc, pelvis_qc = evidence["neck_qc"], evidence["pelvis_qc"]
    neck_search, pelvis_search = evidence["neck_search"], evidence["pelvis_search"]
    mirrored = anterior_side == "right"
    canonical = np.ascontiguousarray(raw[:, ::-1]) if mirrored else raw
    warnings = ["Review C7 identity, the S1 posterior corner, image orientation and calibration before accepting global SVA."]
    if automatic:
        if orientation.get("method") == "anatomical_cervical_mirror_consensus":
            warnings.append("Anterior orientation was selected from anatomical cervical landmarks checked in both image orientations; confirm it before accepting measurements.")
        else:
            warnings.append("Anterior orientation was selected automatically from regional crop agreement; confirm it before accepting measurements.")
    if neck is None:
        warnings.append("C7 centroid was not established consistently across crops; global SVA is unavailable.")
    if pelvis is None:
        warnings.append("S1 posterior corner was not established consistently across crops; global SVA is unavailable.")
    geometry = {"region": "full_spine", "anterior_side": anterior_side,
                "c2_centroid": None,
                "c7_centroid": None if neck is None else _source_points(neck["anchor"], width, mirrored).tolist(),
                "s1_superior": None if pelvis is None else _source_points(pelvis["endplate"], width, mirrored).tolist(),
                "vertebrae": {}}
    if neck is not None:
        # The model has already run in canonical anterior-left coordinates.
        # Preserve endpoint identity while mirroring back, rather than sorting
        # source x coordinates or reinterpreting them a second time.
        cervical_geometry = cervical.landmark_contract(neck["points"], "left")
        geometry["c2_centroid"] = _source_points(cervical_geometry["c2_centroid"], width, mirrored).tolist()
        for level, body in cervical_geometry["vertebrae"].items():
            geometry["vertebrae"][level] = {
                key: None if value is None else _source_points(value, width, mirrored).tolist()
                for key, value in body.items()}
    if pelvis is not None:
        p = _source_points(pelvis["points"][:20], width, mirrored).reshape(5, 4, 2)
        for level, body in zip(range(1, 6), p):
            geometry["vertebrae"][f"L{level}"] = {
                "superior": body[:2].tolist(), "inferior": body[2:].tolist(),
                "quadrilateral": body[[0, 1, 3, 2]].tolist()}
    femoral_mask, circles, femoral_qc = _femoral_region(canonical, pelvis)
    if any(len(circle) != 3 or not np.isfinite(circle).all()
           or not (0 <= circle[0] < width and 0 <= circle[1] < height)
           or circle[2] <= 0 for circle in circles):
        # An extrapolated circle is not a measured head center. Preserve the
        # observed mask and spine landmarks without passing invalid optional
        # geometry to measurement validation or clipping it onto the border.
        circles = []
        femoral_qc = {**femoral_qc, "qc_pass": False,
                      "reason": "Fitted femoral head center falls outside the image or is invalid."}
    geometry["femoral_circles"] = [[*_source_points(circle[:2], width, mirrored).tolist(), circle[2]]
                                   for circle in circles]
    if not femoral_qc["qc_pass"]:
        warnings.append("Femoral heads were not established reliably; PI, PT and L1PA are unavailable.")
    elif femoral_qc.get("touches_crop_edge"):
        warnings.append("The femoral mask reaches its crop boundary; review the head circles and pelvic measurements.")
    if mirrored:
        for key in ("initial_crop_window", "crop_window"):
            if key in femoral_qc:
                left, top, right, bottom = femoral_qc[key]
                femoral_qc[key] = [width-right, top, width-left, bottom]
        femoral_mask = np.ascontiguousarray(femoral_mask[:, ::-1])
    def source_window(candidate):
        if candidate is None:
            return None
        left, top, right, bottom = candidate["window"]
        return [width-right, top, width-left, bottom] if mirrored else [left, top, right, bottom]
    if mirrored and pelvis_search.get("localizer") is not None:
        localizer = dict(pelvis_search["localizer"])
        a, b, c, d = localizer["window"]
        localizer["window"] = [width-c, b, width-a, d]
        pelvis_search = {**pelvis_search, "localizer": localizer}
    return {"image": raw.copy() if raw.dtype == np.uint8 else models._robust_rescale(raw),
            "mask": np.zeros(raw.shape, np.uint8), "femoral_mask": femoral_mask,
            "landmarks": geometry,
            "models": {"vertebrae": MODEL_NAME, "cervical": "cervical_hrnet", "lumbar": "hrnet",
                       **({"orientation": orientation["model"]} if orientation.get("model") else {}),
                       "detector": "crop_detector+s1" if runtime.options().crop_method == "model"
                                   else "cervical_detr+s1", "femoral": "unet"},
            "framing": {"coordinate_space": "original_image", "canonical_mirror": mirrored,
                        "cervical_window": source_window(neck), "lumbar_window": source_window(pelvis),
                        "cervical": {**neck_search, **neck_qc}, "lumbar": {**pelvis_search, **pelvis_qc},
                        "orientation": orientation, "femoral": femoral_qc},
            "warnings": warnings,
            "provenance": {"reference_landmarks_used": False, "segmentation_available": False,
                           "c7_centroid_method": "mean_of_four_body_corners",
                           "s1_source": "lumbar_hrnet", "vertical_reference": "image_vertical",
                           "anterior_side_source": "automatic" if automatic else "user"}}
