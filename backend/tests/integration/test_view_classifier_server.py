from backend import fixed_pipeline
"""HTTP routing contracts only; model accuracy is measured on real films."""
import pytest
from fastapi.testclient import TestClient

from backend import runtime, server, view_classifier
from backend.tests.integration.test_global_sva_server import upload


def test_preview_uses_classifier_and_never_runs_segmentation(monkeypatch):
    observed = []

    def classify(pixels, anterior_side=None):
        observed.append((pixels.shape, runtime.options().view_selection))
        return {"body_part": None, "status": "needs_selection", "warnings": ["Preview only"]}

    monkeypatch.setattr(view_classifier, "detect_film", classify)
    monkeypatch.setattr(fixed_pipeline, "predict", lambda *_: pytest.fail("preview must not segment"))
    monkeypatch.setattr(server, "cervical_prediction", lambda *_: pytest.fail("preview must not segment"))
    response = TestClient(server.app).post('/classify-view', files={
        'file': ('contract.png', upload(), 'image/png')})
    assert response.status_code == 200
    assert response.json()['status'] == 'needs_selection'
    assert observed[0][1] == 'classifier'
    assert runtime.options().view_selection == 'landmarks'


def test_preview_rejects_empty_and_undecodable_files():
    client = TestClient(server.app)
    for payload in (b'', b'not an image'):
        response = client.post('/classify-view', files={'file': ('bad.png', payload, 'image/png')})
        assert response.status_code == 422


def test_invalid_view_selection_is_rejected_before_inference(monkeypatch):
    monkeypatch.setattr(server, 'detect_film', lambda *_: pytest.fail('invalid request must not infer'))
    response = TestClient(server.app).post('/predict', data={
        'modality': 'xray', 'body_part': 'auto', 'view': 'lateral', 'view_selection': 'unrecognized'},
        files={'file': ('contract.png', upload(), 'image/png')})
    assert response.status_code == 422
