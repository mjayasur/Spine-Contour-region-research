import base64
import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend import server
from backend.models import VertebraLabel


def test_predict_endpoint_returns_fixed_landmark_geometry(monkeypatch):
    from backend import fixed_pipeline
    from backend.tests.unit.test_global_sva_measurements import geometry
    def prediction(pixel_array, anterior_side, *, region, detection_evidence):
        assert region == "lumbar"
        g = geometry()
        g.update(region=region, pipeline_revision=fixed_pipeline.REVISION)
        return {"image": pixel_array, "mask": np.zeros_like(pixel_array),
                "femoral_mask": np.zeros_like(pixel_array), "landmarks": g,
                "models": {"vertebrae": fixed_pipeline.REVISION}, "framing": {}}
    monkeypatch.setattr(fixed_pipeline, "predict", prediction)
    monkeypatch.setattr(server, "calibration_from_payload", lambda *a, **k: {"source_sha256": "test", "spacing": None})
    upload = io.BytesIO()
    Image.fromarray(np.full((300, 300), 127, dtype=np.uint8)).save(upload, format="PNG")
    response = TestClient(server.app).post("/predict", data={"modality": "xray", "body_part": "lumbar", "view": "lateral", "vertebra_model": "hrnet"},
        files={"file": ("contract.png", upload.getvalue(), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["geometry"]["pipeline_revision"] == fixed_pipeline.REVISION
    assert body["geometry"]["region"] == "lumbar"
    assert body["labels"] == {}
    assert body["qc"]["models"]["vertebrae"] == fixed_pipeline.REVISION


def test_predict_endpoint_rejects_a_model_that_is_not_offered():
    upload = io.BytesIO()
    Image.fromarray(np.full((24, 16), 127, dtype=np.uint8)).save(upload, format="PNG")
    response = TestClient(server.app).post(
        "/predict",
        data={"modality": "xray", "body_part": "lumbar", "view": "lateral", "vertebra_model": "resnet"},
        files={"file": ("radiograph.png", upload.getvalue(), "image/png")},
    )
    assert response.status_code == 422
    assert "fixed lumbar pipeline" in response.json()["detail"]


def test_predict_endpoint_rejects_an_ap_view_before_inference():
    upload = io.BytesIO()
    Image.fromarray(np.full((24, 16), 127, dtype=np.uint8)).save(upload, format="PNG")
    response = TestClient(server.app).post(
        "/predict", data={"modality": "xray", "body_part": "lumbar", "view": "AP"},
        files={"file": ("ap.png", upload.getvalue(), "image/png")},
    )
    assert response.status_code == 422
    assert "view='lateral'" in response.json()["detail"]


def test_models_endpoint_lists_a_choice_only_for_the_vertebrae():
    response = TestClient(server.app).get("/models")
    assert response.status_code == 200
    assert response.json() == {"vertebrae": ["hrnet"], "femoral": ["unet"], "s1": ["keypointrcnn"]}


def test_predict_endpoint_rejects_an_empty_upload():
    response = TestClient(server.app).post(
        "/predict",
        data={"modality": "xray", "body_part": "lumbar", "view": "lateral"},
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 400


def test_measure_endpoint_recalculates_corrected_landmarks():
    vertebrae = {
        level: {
            "quadrilateral": [[20, top], [80, top], [80, top + 10], [20, top + 10]],
            "superior": [[20, top], [80, top]],
            "inferior": [[20, top + 10], [80, top + 10]],
        }
        for level, top in zip(("L1", "L2", "L3", "L4", "L5"), (10, 30, 50, 70, 90))
    }
    response = TestClient(server.app).post(
        "/measure",
        json={
            "vertebrae": vertebrae,
            "s1_superior": [[20, 120], [80, 110]],
            "femoral_circles": [[35, 145, 12], [65, 145, 12]],
        },
    )

    assert response.status_code == 200
    assert set(response.json()["measurements"]["LL"]) == {
        "L1-S1", "L2-S1", "L3-S1", "L4-S1", "L5-S1"
    }
    assert response.json()["geometry"]["hip_midpoint"] == [50.0, 145.0]


def test_health_endpoint_reports_ready():
    response = TestClient(server.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
