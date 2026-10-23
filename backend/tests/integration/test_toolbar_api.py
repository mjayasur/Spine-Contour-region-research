import base64
import io
import json

import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from backend import calibration, framing, server
from backend.models import models
from backend.tests.unit.test_partial_prediction import frame
from backend.tests.unit.test_toolbar import screenshot


@pytest.mark.parametrize('mode', ['standard', 'low-memory'])
@pytest.mark.parametrize('localizer', [True, False])
@pytest.mark.parametrize('route', ['/predict', '/predict-stream'])
def test_fixed_pipeline_keeps_source_pixels_and_calibration_with_toolbar_preference(monkeypatch, mode, localizer, route):
    image = screenshot()
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format='PNG')
    payload = buffer.getvalue()
    cached = calibration.calibration_from_payload(payload, preview_only=True)
    cached.pop('image_png')
    cached.update(status='corrected', selected_index=0,
                  candidates=[{'value_mm': 40, 'length_px': 80, 'endpoints': [[320, 40], [320, 120]],
                               'raw_text': '40 mm', 'status': 'accepted'}])
    monkeypatch.setattr(calibration, 'extract', lambda *_: pytest.fail('Original source cache should still match'))
    from backend.tests.integration.fixed_contract import stub
    stub(monkeypatch)
    client = TestClient(server.app)
    # ON followed by OFF exercises request isolation in the same worker.
    for enabled in [True, False]:
        response = client.post(route, files={'file': ('film.png', payload, 'image/png')}, data={
            'modality': 'xray', 'body_part': 'lumbar', 'view': 'lateral', 'processing_mode': mode,
            'crop_localizer': str(localizer).lower(), 'toolbar_removal': str(enabled).lower(),
            'calibration': json.dumps(cached)})
        assert response.status_code == 200
        if route.endswith('stream'):
            events = [json.loads(line) for line in response.text.splitlines()]
            assert events[-1]['type'] == 'result', events[-1]
            result = events[-1]['result']
            assert any(e.get('stage') == 'toolbar' for e in events) is False
        else:
            result = response.json()
        for field in ('image_png', 'mask_png', 'femoral_mask_png'):
            with Image.open(io.BytesIO(base64.b64decode(result[field]))) as png:
                assert png.size == (951, 801)
        scale = result['calibration']
        assert scale['height'] == 801 and scale['width'] == 951
        assert scale['source_sha256'] == cached['source_sha256']
        assert scale['status'] == 'corrected' and scale['spacing']['row_mm'] == .5
        assert scale['candidates'][0]['endpoints'] == [[320, 40], [320, 120]]
        assert result['qc']['processing']['toolbar_removal'] is False
        assert result['qc']['processing']['requested_toolbar_removal'] == enabled
        assert result['measurements']['PI'] is None and result['geometry']['femoral_circles'] == []
        assert list(result['geometry']['vertebrae']) == ['L1']


@pytest.mark.parametrize('route', ['/predict', '/predict-stream'])
def test_bad_toolbar_setting_fails_before_processing(route):
    response = TestClient(server.app).post(route, files={'file': ('film.png', b'file', 'image/png')},
        data={'modality': 'xray', 'body_part': 'lumbar', 'toolbar_removal': 'sometimes'})
    assert response.status_code == 422
