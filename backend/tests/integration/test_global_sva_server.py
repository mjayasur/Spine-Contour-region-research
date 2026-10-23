from backend import fixed_pipeline
"""Full-spine API checks using generated pixels and mocked inference only."""

import base64
import copy
import hashlib
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import calibration as calibration_module
from backend import runtime, server
from backend.tests.unit.test_global_sva_measurements import geometry


def upload():
    buffer = io.BytesIO()
    Image.fromarray(np.full((120, 100), 127, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def calibration(payload, spacing=None):
    return {"version": 1, "source_sha256": hashlib.sha256(payload).hexdigest(),
            "width": 100, "height": 120, "coordinate_space": "original_image",
            "status": "dicom" if spacing else "not_found", "spacing": spacing,
            "candidates": [], "selected_index": None}


def stub_prediction(monkeypatch, observed, landmarks=None):
    def predict(pixels, anterior_side, model):
        observed.append((pixels.copy(), anterior_side, model))
        runtime.report("full_spine_landmarks", "Finding C7 and S1 landmarks")
        return {"image": pixels, "mask": np.zeros_like(pixels), "femoral_mask": np.zeros_like(pixels),
                "landmarks": copy.deepcopy(geometry() if landmarks is None else landmarks),
                "models": {"full_spine": "dual_hrnet", "vertebrae": "dual_hrnet"},
                "framing": {"coordinate_space": "original_image", "window": [0, 0, 100, 120]},
                "provenance": {"landmark_coordinates": "source_image_pixels", "segmentation_available": False},
                "warnings": ["Verify C7 and S1 landmarks."]}
    def unexpected(*args, **kwargs):
        pytest.fail("Regional inference must not be dispatched for full_spine")
    monkeypatch.setattr(server, "full_spine_prediction", predict)
    monkeypatch.setattr(server, "cervical_prediction", unexpected)
    monkeypatch.setattr(fixed_pipeline, "predict", unexpected)


def result_from(response, route):
    assert response.status_code == 200
    if route.endswith("stream"):
        events = [json.loads(line) for line in response.text.splitlines()]
        assert events[-1]["type"] == "result", events[-1]
        return events[-1]["result"], events
    return response.json(), []


@pytest.mark.parametrize("route", ["/predict", "/predict-stream"])
@pytest.mark.parametrize("mode", ["standard", "low-memory"])
def test_full_spine_dispatch_uses_original_image_calibration_and_its_own_measurements(monkeypatch, route, mode):
    payload, observed = upload(), []
    stub_prediction(monkeypatch, observed)
    spacing = {"row_mm": 2, "column_mm": .25, "source": "dicom_pixel_spacing"}
    def calibrate(source, include_preview, cached):
        assert source == payload and include_preview is False
        assert cached == {"synthetic_cache": True}
        return calibration(source, spacing)
    monkeypatch.setattr(server, "calibration_from_payload", calibrate)
    response = TestClient(server.app).post(route, data={
        "modality": "xray", "body_part": "full_spine", "view": "lateral",
        "vertebra_model": "dual_hrnet", "anterior_side": "left", "processing_mode": mode,
        "crop_localizer": "false", "crop_method": "model", "toolbar_removal": "true",
        "calibration": json.dumps({"synthetic_cache": True}),
    }, files={"file": ("synthetic.png", payload, "image/png")})
    result, events = result_from(response, route)
    assert len(observed) == 1 and observed[0][0].shape == (120, 100)
    assert observed[0][1:] == ("left", "dual_hrnet")
    assert result["measurements"]["region"] == "full_spine"
    assert result["measurements"]["GLOBAL_SVA_PX"] == 10
    assert result["measurements"]["GLOBAL_SVA_MM"] == 2.5
    assert result["geometry"]["c7_centroid"] == [30, 10]
    assert result["geometry"]["s1_superior"] == [[20, 80], [40, 90]]
    assert result["geometry"]["pixel_spacing"] == [2, .25]
    assert result["geometry"]["source_sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["geometry"]["image_width"] == 100 and result["geometry"]["image_height"] == 120
    assert result["qc"]["models"]["full_spine"] == "dual_hrnet"
    assert result["qc"]["provenance"]["segmentation_available"] is False
    assert result["qc"]["warnings"] == ["Verify C7 and S1 landmarks."]
    assert result["qc"]["processing"]["crop_localizer"] is True
    assert result["qc"]["processing"]["crop_method"] == "search"
    assert result["qc"]["processing"]["toolbar_removal"] is False
    assert result["qc"]["processing"]["requested_crop_localizer"] is False
    assert result["qc"]["processing"]["requested_toolbar_removal"] is True
    assert result["labels"] == {}
    for name in ("image_png", "mask_png", "femoral_mask_png"):
        image = np.asarray(Image.open(io.BytesIO(base64.b64decode(result[name]))))
        assert image.shape == (120, 100)
        assert np.all(image == (127 if name == "image_png" else 0))
    if events:
        assert {"full_spine_landmarks", "calibration", "measuring", "complete"} <= {e.get("stage") for e in events}


@pytest.mark.parametrize("route", ["/predict", "/predict-stream"])
@pytest.mark.parametrize("invalid", [
    {"anterior_side": "unknown"}, {"modality": "CT"}, {"view": "AP"},
    {"laterality": "AP"}, {"vertebra_model": "hrnet"}, {"vertebra_model": "cervical_hrnet"},
    {"femoral_model": "unet"}, {"s1_model": "keypointrcnn"},
])
def test_invalid_full_spine_setup_is_rejected_before_decode_or_inference(monkeypatch, route, invalid):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid requests must not reach image decoding or inference")
    monkeypatch.setattr(server, "_decode_grayscale", unexpected)
    monkeypatch.setattr(server, "full_spine_prediction", unexpected)
    response = TestClient(server.app).post(route, data={
        "modality": "xray", "body_part": "full_spine", "view": "lateral", "anterior_side": "left", **invalid,
    }, files={"file": ("synthetic.png", b"unused", "image/png")})
    assert response.status_code == 422


@pytest.mark.parametrize("route", ["/predict", "/predict-stream"])
def test_missing_landmarks_return_partial_results_instead_of_fabricated_values(monkeypatch, route):
    observed = []
    stub_prediction(monkeypatch, observed, {"c7_centroid": None, "s1_superior": None, "vertebrae": {}})
    monkeypatch.setattr(server, "calibration_from_payload", lambda payload, **_: calibration(payload))
    response = TestClient(server.app).post(route, data={
        "modality": "xray", "body_part": "full_spine", "view": "lateral", "anterior_side": "right",
    }, files={"file": ("synthetic.png", upload(), "image/png")})
    result, _ = result_from(response, route)
    assert observed[0][1:] == ("right", "dual_hrnet")
    assert result["measurements"]["GLOBAL_SVA_PX"] is None
    assert result["measurements"]["GLOBAL_SVA_MM"] is None
    assert {"C7 centroid", "S1 superior"} <= set(result["qc"]["coverage"]["missing"])
    assert result["measurements"]["C2C7_COBB"] is None
    assert result["measurements"]["PI"] is None
    assert result["qc"]["global_sva"]["review_required"] is True


def test_optional_calibration_failure_keeps_global_sva_in_pixels(monkeypatch):
    stub_prediction(monkeypatch, [])
    def fail(*args, **kwargs):
        raise RuntimeError("Synthetic calibration failure")
    monkeypatch.setattr(server, "calibration_from_payload", fail)
    response = TestClient(server.app).post("/predict", data={
        "modality": "xray", "body_part": "full_spine", "view": "lateral", "anterior_side": "left",
    }, files={"file": ("synthetic.png", upload(), "image/png")})
    result, _ = result_from(response, "/predict")
    assert result["measurements"]["GLOBAL_SVA_PX"] == 10
    assert result["measurements"]["GLOBAL_SVA_MM"] is None
    assert result["calibration"]["status"] == "unavailable"


@pytest.mark.parametrize("cache_kind, expected_mm", [("matching", 2.5), ("different_source", None), ("cleared", None)])
def test_original_upload_binding_accepts_only_matching_calibration(monkeypatch, cache_kind, expected_mm):
    payload = upload()
    stub_prediction(monkeypatch, [])
    monkeypatch.setattr(calibration_module, "configure_ocr", lambda: None)
    monkeypatch.setattr(calibration_module, "extract", lambda *_: {"measurements": []})
    cached = {**calibration(payload), "status": "corrected", "selected_index": 0,
              "spacing": {"row_mm": .25, "column_mm": .25, "source": "manual_reference"},
              "candidates": [{"value_mm": 10, "length_px": 40, "endpoints": [[5, 5], [5, 45]], "status": "accepted"}]}
    if cache_kind == "different_source":
        cached["source_sha256"] = "b" * 64
    if cache_kind == "cleared":
        cached.update(status="cleared", spacing=None, selected_index=None, candidates=[])
    response = TestClient(server.app).post("/predict", data={
        "modality": "xray", "body_part": "full_spine", "view": "lateral", "anterior_side": "left",
        "calibration": json.dumps(cached),
    }, files={"file": ("synthetic.png", payload, "image/png")})
    result, _ = result_from(response, "/predict")
    assert result["measurements"]["GLOBAL_SVA_MM"] == expected_mm
    assert result["geometry"]["source_sha256"] == hashlib.sha256(payload).hexdigest()
    if cache_kind == "cleared":
        assert result["calibration"]["status"] == "cleared"


def test_measure_global_sva_edits_anchors_and_respects_current_scale_clear():
    client = TestClient(server.app)
    response = client.post("/measure", json={**geometry(), "pixel_spacing": [1, .25]})
    assert response.status_code == 200
    assert response.json()["measurements"]["GLOBAL_SVA_MM"] == 2.5
    edited = response.json()["geometry"]
    edited["c7_centroid"][0] = 45
    edited.update(pixel_spacing=None, spacing_source=None)
    response = client.post("/measure", json=edited)
    assert response.status_code == 200
    assert response.json()["measurements"]["GLOBAL_SVA_PX"] == -5
    assert response.json()["measurements"]["GLOBAL_SVA_MM"] is None
    edited["s1_superior"] = None
    assert client.post("/measure", json=edited).json()["measurements"]["GLOBAL_SVA_PX"] is None
    edited["c7_centroid"] = [100, 10]
    assert client.post("/measure", json=edited).status_code == 422


def test_model_choices_expose_full_spine_without_changing_regional_models():
    client = TestClient(server.app)
    assert client.get("/models?body_part=full_spine").json() == {"vertebrae": ["dual_hrnet"], "femoral": [], "s1": []}
    assert client.get("/models?body_part=cervical").json() == {"vertebrae": ["cervical_slope_hrnet"], "femoral": [], "s1": []}
    assert client.get("/models").json() == {"vertebrae": ["hrnet"], "femoral": ["unet"], "s1": ["keypointrcnn"]}
