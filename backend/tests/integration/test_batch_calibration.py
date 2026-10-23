"""The shared /predict path calibrates every batch image without model dependencies."""
import io
import json
import shutil

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from backend import calibration, server, fixed_pipeline


def image_bytes():
    image = np.zeros((260, 420, 3), np.uint8)
    cv2.line(image, (320, 40), (320, 120), (255, 255, 255), 1)
    for y in (40, 120):
        cv2.line(image, (314, y), (326, y), (255, 255, 255), 1)
    cv2.putText(image, '25.4 mm', (255, 165), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
    stream = io.BytesIO(); Image.fromarray(image).save(stream, format='PNG')
    return stream.getvalue()


@pytest.fixture
def client(monkeypatch):
    def cropped_prediction(pixels, *_, **kwargs):
        image = pixels[:100, :100]
        return {'image': image, 'mask': np.zeros_like(image), 'femoral_mask': np.zeros_like(image),
                'landmarks': {'vertebrae': {}, 'S1': {'superior': []}},
                'models': {}, 'framing': {'window': [0, 0, 100, 100]}}
    monkeypatch.setattr(fixed_pipeline, 'predict', cropped_prediction)
    monkeypatch.setattr(server, 'global_sva_measurements_from_geometry',
                        lambda g: {'measurements': {'SS': 30}, 'geometry': g})
    return TestClient(server.app)


def predict(client, payload, cached=None):
    return client.post('/predict', files={'file': ('film.png', payload, 'image/png')},
                       data={'modality': 'xray', 'body_part': 'lumbar', 'view': 'lateral',
                             **({'calibration': json.dumps(cached)} if cached else {})})


@pytest.mark.skipif(not shutil.which('tesseract'), reason='Tesseract required')
def test_batch_path_detects_off_crop_reference_from_original_bytes(client):
    response = predict(client, image_bytes())
    assert response.status_code == 200
    result = response.json()['calibration']
    assert result['status'] == 'detected'
    assert result['width'] == 420
    assert result['candidates'][result['selected_index']]['value_mm'] == 25.4
    assert result['candidates'][result['selected_index']]['endpoints'][0][0] > 300
    assert 'image_png' not in result


def test_missing_ocr_or_calibration_error_does_not_fail_segmentation(client, monkeypatch):
    def fail(*_, **__):
        raise RuntimeError('OCR not available')
    monkeypatch.setattr(server, 'calibration_from_payload', fail)
    result = predict(client, image_bytes())
    assert result.status_code == 200
    assert result.json()['measurements']['SS'] == 30
    assert result.json()['calibration']['status'] == 'unavailable'
    assert result.json()['calibration']['spacing'] is None


def test_batch_reuses_reviewed_scale_only_for_matching_image(client, monkeypatch):
    payload = image_bytes()
    cached = calibration.calibration_from_payload(payload, preview_only=True)
    cached.update(status='corrected', selected_index=0,
                  candidates=[{'value_mm': 40, 'length_px': 80, 'endpoints': [[320, 40], [320, 120]],
                               'raw_text': '40 mm (corrected)', 'status': 'accepted'}])
    monkeypatch.setattr(calibration, 'extract', lambda *_: pytest.fail('Saved reference should avoid OCR'))
    result = predict(client, payload, cached).json()['calibration']
    assert result['status'] == 'corrected'
    assert result['spacing']['row_mm'] == .5


def test_no_reference_leaves_lengths_unavailable(client, monkeypatch):
    monkeypatch.setattr(calibration, 'extract', lambda *_: {'measurements': []})
    result = predict(client, image_bytes()).json()
    assert result['measurements']['SS'] == 30
    assert result['calibration']['status'] == 'not_found'
    assert result['calibration']['spacing'] is None


@pytest.mark.parametrize('route', ['/predict', '/predict-stream'])
@pytest.mark.parametrize('initial', ['not_found', 'unavailable'])
def test_manual_fallback_survives_calibration_reopen_and_segmentation(client, monkeypatch, route, initial):
    payload = image_bytes()
    def failed_detection(*_):
        if initial == 'unavailable':
            raise RuntimeError('OCR unavailable')
        return {'measurements': []}
    monkeypatch.setattr(calibration, 'extract', failed_detection)
    preview = client.post('/calibrate', files={'file': ('image.png', payload, 'image/png')}).json()
    assert preview['status'] == initial
    preview.pop('image_png')
    # The compact record produced by drawing/applying a ruler after detection failed.
    preview.update(status='corrected', selected_index=0, review_revision=3,
                   spacing={'row_mm': .5, 'column_mm': .5, 'source': 'manual_reference'},
                   candidates=[{'value_mm': 40, 'length_px': 80, 'endpoints': [[320, 40], [320, 120]],
                                'raw_text': '40 mm (corrected)', 'status': 'accepted'}])
    saved = json.loads(json.dumps(preview))
    monkeypatch.setattr(calibration, 'extract', lambda *_: pytest.fail('Manual reference must avoid OCR'))
    for calibration_route in ['/calibrate', '/calibrate-stream']:
        response = client.post(calibration_route, files={'file': ('renamed.png', payload, 'image/png')},
                               data={'calibration': json.dumps(saved), 'preview_only': 'true'})
        assert response.status_code == 200
        reopened = (json.loads(response.text.splitlines()[-1])['result']
                    if calibration_route.endswith('stream') else response.json())
        assert reopened['spacing'] == saved['spacing']
        assert reopened['review_revision'] == 3
        assert reopened['candidates'] == saved['candidates']
    response = client.post(route, files={'file': ('renamed.png', payload, 'image/png')},
                           data={'modality': 'xray', 'body_part': 'lumbar', 'view': 'lateral', 'calibration': json.dumps(saved)})
    assert response.status_code == 200
    result = (json.loads(response.text.splitlines()[-1])['result']
              if route.endswith('stream') else response.json())
    assert result['measurements']['SS'] == 30
    assert result['calibration']['spacing'] == saved['spacing']
    assert result['calibration']['review_revision'] == 3
    assert result['calibration']['candidates'] == saved['candidates']
    assert 'image_png' not in result['calibration']


@pytest.mark.parametrize('route', ['/calibrate', '/calibrate-stream', '/predict', '/predict-stream'])
def test_manual_reference_from_another_image_is_never_reused(client, monkeypatch, route):
    payload = image_bytes()
    saved = calibration.calibration_from_payload(payload, preview_only=True)
    saved.update(source_sha256='a' * 64, status='corrected', selected_index=0,
                 candidates=[{'value_mm': 40, 'length_px': 80, 'endpoints': [[320, 40], [320, 120]],
                              'status': 'accepted'}])
    monkeypatch.setattr(calibration, 'extract', lambda *_: {'measurements': []})
    response = client.post(route, files={'file': ('same-name.png', payload, 'image/png')},
                           data={'modality': 'xray', 'body_part': 'lumbar', 'view': 'lateral', 'calibration': json.dumps(saved)})
    assert response.status_code == 200
    result = (json.loads(response.text.splitlines()[-1])['result']
              if route.endswith('stream') else response.json())
    scale = result['calibration'] if route.startswith('/predict') else result
    assert scale['status'] == 'not_found'
    assert scale['spacing'] is None
