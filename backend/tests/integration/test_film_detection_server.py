"""Auto dispatch and user-override contracts, using synthetic images only."""
import copy
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import server
from backend.tests.integration.test_global_sva_server import upload, calibration
from backend.tests.unit.test_global_sva_measurements import geometry


def prediction(pixels, landmarks):
    return {"image": pixels, "mask": np.zeros_like(pixels), "femoral_mask": np.zeros_like(pixels),
            "landmarks": copy.deepcopy(landmarks), "models": {}, "framing": {}}


def result(response, route):
    if route.endswith("stream"):
        event = json.loads(response.text.splitlines()[-1])
        assert event["type"] == "result", event
        return event["result"]
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize("route", ["/predict", "/predict-stream"])
@pytest.mark.parametrize("mode", ["standard", "low-memory"])
def test_auto_full_spine_reuses_detection_and_retains_auto_orientation_provenance(monkeypatch, route, mode):
    evidence = {"sentinel": object()}
    monkeypatch.setattr(server, "detect_film", lambda pixels, anterior_side: {
        "body_part": "full_spine", "anterior_side": "right", "status": "detected",
        "qc": {"method": "landmark_consensus"}, "_orientation_evidence": evidence,
    })
    def infer(pixels, anterior_side, model, detection_evidence):
        assert anterior_side is None and model == "dual_hrnet"
        assert detection_evidence is evidence
        value = prediction(pixels, {**geometry(), "anterior_side": "right"})
        value["provenance"] = {"anterior_side_source": "automatic"}
        return value
    monkeypatch.setattr(server, "full_spine_prediction", infer)
    monkeypatch.setattr(server, "calibration_from_payload", lambda payload, **_: calibration(payload))
    response = TestClient(server.app).post(route, data={
        "modality": "xray", "body_part": "auto", "view": "lateral", "processing_mode": mode,
    }, files={"file": ("synthetic.png", upload(), "image/png")})
    body = result(response, route)
    assert body["geometry"]["region"] == "full_spine"
    assert body["geometry"]["anterior_side"] == "right"
    assert body["qc"]["provenance"]["anterior_side_source"] == "automatic"
    assert body["qc"]["film_detection"]["body_part"] == "full_spine"
    assert "_orientation_evidence" not in body["qc"]["film_detection"]


@pytest.mark.parametrize("region", ["cervical", "lumbar"])
def test_auto_dispatches_regional_models_and_persists_resolved_region(monkeypatch, region):
    monkeypatch.setattr(server, "detect_film", lambda pixels, anterior_side: {
        "body_part": region, "anterior_side": None, "status": "detected", "qc": {},
    })
    monkeypatch.setattr(server, "calibration_from_payload", lambda payload, **_: calibration(payload))
    if region == "cervical":
        def cervical(pixels, anterior_side, model):
            assert anterior_side == "left" and model == "cervical_slope_hrnet"
            return prediction(pixels, {"region": "cervical", "anterior_side": "left", "vertebrae": {}})
        monkeypatch.setattr(server, "cervical_prediction", cervical)
    else:
        from backend.tests.integration.fixed_contract import stub
        stub(monkeypatch, (), [[20, 80], [40, 90]])
    response = TestClient(server.app).post("/predict", data={
        "modality": "xray", "body_part": "auto", "view": "lateral", "anterior_side": "left",
    }, files={"file": ("synthetic.png", upload(), "image/png")})
    body = result(response, "/predict")
    assert body["geometry"]["region"] == region
    assert body["measurements"]["region"] == region
    edited = TestClient(server.app).post("/measure", json=body["geometry"])
    assert edited.status_code == 200
    assert edited.json()["geometry"]["region"] == region
    assert edited.json()["measurements"]["region"] == region


@pytest.mark.parametrize("detection, message", [
    ({"body_part": None, "warnings": ["Choose a spine region manually."]}, "Choose a spine region"),
])
@pytest.mark.parametrize("route", ["/predict", "/predict-stream"])
def test_ambiguous_region_detection_requests_user_selection(monkeypatch, detection, message, route):
    monkeypatch.setattr(server, "detect_film", lambda *a, **kw: detection)
    monkeypatch.setattr(server, "cervical_prediction", lambda *a, **kw: pytest.fail("No guessed side"))
    response = TestClient(server.app).post(route, data={
        "modality": "xray", "body_part": "auto", "view": "lateral",
    }, files={"file": ("synthetic.png", upload(), "image/png")})
    if route.endswith("stream"):
        event = json.loads(response.text.splitlines()[-1])
        assert event["type"] == "error"
        assert message in event["message"]
    else:
        assert response.status_code == 422 and message in response.json()["detail"]


@pytest.mark.parametrize("invalid", [{"view": "AP"}, {"modality": "CT"},
                                      {"anterior_side": "unknown"}, {"vertebra_model": "hrnet"},
                                      {"femoral_model": "unet"}, {"laterality": "AP"}])
def test_invalid_auto_requests_fail_before_decoding(monkeypatch, invalid):
    monkeypatch.setattr(server, "_decode_grayscale", lambda *a: pytest.fail("Invalid request decoded"))
    response = TestClient(server.app).post("/predict", data={
        "modality": "xray", "body_part": "auto", "view": "lateral", **invalid,
    }, files={"file": ("synthetic.png", b"unused", "image/png")})
    assert response.status_code == 422


@pytest.mark.parametrize("side", [None, "auto", "left", "right"])
@pytest.mark.parametrize("view_selection", ["landmarks", "classifier"])
def test_explicit_full_spine_skips_region_detection_and_accepts_auto_side(monkeypatch, side, view_selection):
    observed = []
    monkeypatch.setattr(server, "detect_film", lambda *a, **kw: pytest.fail("Explicit region must bypass detection"))
    def infer(pixels, anterior_side, model):
        observed.append(anterior_side)
        return prediction(pixels, {**geometry(), "anterior_side": side if side in ("left", "right") else "right"})
    monkeypatch.setattr(server, "full_spine_prediction", infer)
    monkeypatch.setattr(server, "calibration_from_payload", lambda payload, **_: calibration(payload))
    data = {"modality": "xray", "body_part": "full_spine", "view": "lateral", "view_selection": view_selection}
    if side is not None:
        data["anterior_side"] = side
    body = result(TestClient(server.app).post("/predict", data=data,
        files={"file": ("synthetic.png", upload(), "image/png")}), "/predict")
    assert observed == [side]
    assert body["geometry"]["anterior_side"] == (side if side in ("left", "right") else "right")


def test_auto_models_are_deferred_until_region_is_known():
    assert TestClient(server.app).get("/models?body_part=auto").json() == {
        "vertebrae": [], "femoral": [], "s1": [],
    }
