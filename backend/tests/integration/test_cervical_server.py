from backend import fixed_pipeline
import base64
import hashlib
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import runtime, server
from backend.tests.unit.test_cervical_measurements import geometry


def upload():
    buffer = io.BytesIO()
    Image.fromarray(np.full((100, 80), 127, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def calibration(payload, spacing):
    return {"version": 1, "source_sha256": hashlib.sha256(payload).hexdigest(),
            "width": 80, "height": 100, "coordinate_space": "original_image",
            "status": "dicom" if spacing else "not_found", "spacing": spacing,
            "candidates": [], "selected_index": None}


def stub_prediction(monkeypatch, observed, landmarks=None):
    def predict(pixels, anterior_side, model):
        observed.append((pixels.copy(), anterior_side, model))
        runtime.report("cervical_landmarks", "Finding C2–C7 landmarks")
        return {"image": pixels, "mask": np.zeros_like(pixels), "femoral_mask": np.zeros_like(pixels),
                "landmarks": geometry() if landmarks is None else landmarks,
                "models": {"cervical": "cervical_slope_hrnet", "detector": "cervical_detr"},
                "framing": {"window": [10, 5, 70, 95], "coordinate_space": "original_image"},
                "provenance": {"training_dataset": "CSXA"},
                "warnings": ["Review predicted cervical landmarks."]}
    monkeypatch.setattr(server, "cervical_prediction", predict)
    monkeypatch.setattr(fixed_pipeline, "predict", lambda *_: pytest.fail("Lumbar inference called"))


@pytest.mark.parametrize("route", ["/predict", "/predict-stream"])
@pytest.mark.parametrize("mode", ["standard", "low-memory"])
def test_cervical_prediction_preserves_source_coords_calibration_and_provenance(monkeypatch, route, mode):
    payload = upload()
    observed = []
    stub_prediction(monkeypatch, observed)
    spacing = {"row_mm": .5, "column_mm": .25, "source": "dicom_pixel_spacing"}
    def calibrate(source, include_preview, cached):
        assert source == payload and include_preview is False
        assert cached == {"test": "original-upload-cache"}
        return calibration(source, spacing)
    monkeypatch.setattr(server, "calibration_from_payload", calibrate)
    response = TestClient(server.app).post(route, data={
        "modality": "xray", "body_part": "cervical", "view": "lateral",
        "vertebra_model": "cervical_slope_hrnet", "anterior_side": "left", "processing_mode": mode,
        "crop_localizer": "false", "toolbar_removal": "true",
        "calibration": json.dumps({"test": "original-upload-cache"}),
    }, files={"file": ("cervical.png", payload, "image/png")})
    assert response.status_code == 200
    if route.endswith("stream"):
        events = [json.loads(line) for line in response.text.splitlines()]
        assert events[-1]["type"] == "result", events[-1]
        assert {"cervical_landmarks", "calibration", "measuring", "complete"} <= {
            event.get("stage") for event in events}
        result = events[-1]["result"]
    else:
        result = response.json()
    assert len(observed) == 1
    assert observed[0][0].shape == (100, 80)
    assert observed[0][1:] == ("left", "cervical_slope_hrnet")
    assert result["geometry"]["c2_centroid"] == [30, 10]
    assert result["geometry"]["vertebrae"]["C7"]["superior"][1] == [40, 50]
    assert result["geometry"]["pixel_spacing"] == [.5, .25]
    assert result["geometry"]["source_sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["measurements"]["C2C7_SVA_MM"] == 2.5
    assert result["measurements"]["C2C7_COBB"] == pytest.approx(63.4349488)
    assert result["calibration"]["width"] == 80
    assert result["calibration"]["height"] == 100
    assert result["qc"]["models"]["cervical"] == "cervical_slope_hrnet"
    assert result["qc"]["provenance"]["training_dataset"] == "CSXA"
    assert result["qc"]["warnings"] == ["Review predicted cervical landmarks."]
    assert result["qc"]["processing"]["crop_localizer"] is False
    assert result["qc"]["processing"]["toolbar_removal"] is False
    assert result["qc"]["processing"]["requested_crop_localizer"] is False
    assert result["qc"]["processing"]["requested_toolbar_removal"] is True
    assert result["labels"] == {}  # A heatmap network does not predict anatomical masks.
    for key in ("image_png", "mask_png", "femoral_mask_png"):
        image = np.array(Image.open(io.BytesIO(base64.b64decode(result[key]))))
        assert image.shape == (100, 80)


@pytest.mark.parametrize("route", ["/predict", "/predict-stream"])
@pytest.mark.parametrize("invalid", [
    {"anterior_side": "unknown"}, {"view": "AP"},
    {"vertebra_model": "hrnet"}, {"femoral_model": "unet"}, {"s1_model": "keypointrcnn"},
])
def test_invalid_cervical_requests_fail_before_inference_and_streaming(monkeypatch, route, invalid):
    monkeypatch.setattr(server, "cervical_prediction", lambda *_: pytest.fail("Inference should not start"))
    response = TestClient(server.app).post(route, data={
        "modality": "xray", "body_part": "cervical", "view": "lateral", "anterior_side": "left", **invalid,
    }, files={"file": ("cervical.png", upload(), "image/png")})
    assert response.status_code == 422


def test_calibration_failure_keeps_cervical_pixel_measurements(monkeypatch):
    observed = []
    stub_prediction(monkeypatch, observed)
    def fail_calibration(*args, **kwargs):
        raise RuntimeError("OCR unavailable")
    monkeypatch.setattr(server, "calibration_from_payload", fail_calibration)
    response = TestClient(server.app).post("/predict", data={
        "modality": "xray", "body_part": "cervical", "view": "lateral", "anterior_side": "left",
    }, files={"file": ("cervical.png", upload(), "image/png")})
    assert response.status_code == 200
    result = response.json()
    assert result["measurements"]["C2C7_COBB"] == 45
    assert result["measurements"]["C2C7_SVA_PX"] == 10
    assert result["measurements"]["C2C7_SVA_MM"] is None
    assert result["calibration"]["status"] == "unavailable"


def test_measure_cervical_roundtrip_edits_and_cleared_calibration():
    client = TestClient(server.app)
    source = {**geometry(), "pixel_spacing": [.5, .25], "spacing_source": "manual_reference"}
    response = client.post("/measure", json=source)
    assert response.status_code == 200
    corrected = response.json()["geometry"]
    assert response.json()["measurements"]["C2C7_SVA_MM"] == 2.5
    corrected["c2_centroid"][0] = 35
    corrected.update(pixel_spacing=None, spacing_source=None)
    response = client.post("/measure", json=corrected)
    assert response.status_code == 200
    assert response.json()["measurements"]["C2C7_SVA_PX"] == 5
    assert response.json()["measurements"]["C2C7_SVA_MM"] is None
    assert response.json()["geometry"]["vertebrae"]["C2"]["superior"] is None
    assert response.json()["geometry"]["s1_superior"] is None


def test_empty_cervical_geometry_can_be_saved_but_malformed_geometry_is_rejected():
    client = TestClient(server.app)
    response = client.post("/measure", json={"region": "cervical", "anterior_side": "left", "vertebrae": {}})
    assert response.status_code == 200
    assert response.json()["measurements"]["C2C7_COBB"] is None
    assert client.post("/measure", json={**geometry(), "c2_centroid": ["unknown", 5]}).status_code == 422
    assert client.post("/measure", json={**geometry(), "region": "thoracic"}).status_code == 422


def test_cervical_models_are_opt_in_and_lumbar_models_unchanged():
    client = TestClient(server.app)
    assert client.get("/models?body_part=cervical").json() == {
        "vertebrae": ["cervical_slope_hrnet"], "femoral": [], "s1": [],
    }
    assert client.get("/models").json() == {
        "vertebrae": ["hrnet"], "femoral": ["unet"], "s1": ["keypointrcnn"],
    }
    assert client.get("/models?body_part=thoracic").status_code == 422


@pytest.mark.parametrize("side", [None, "", "auto"])
def test_cervical_does_not_require_manual_orientation(monkeypatch, side):
    observed=[]
    stub_prediction(monkeypatch, observed)
    monkeypatch.setattr(server, "calibration_from_payload", lambda payload, **_: calibration(payload, None))
    data={"modality":"xray", "body_part":"cervical", "view":"lateral"}
    if side is not None:data["anterior_side"]=side
    response=TestClient(server.app).post("/predict", data=data, files={"file":("cervical.png",upload(),"image/png")})
    assert response.status_code==200
    assert observed[0][2]=="cervical_slope_hrnet"
