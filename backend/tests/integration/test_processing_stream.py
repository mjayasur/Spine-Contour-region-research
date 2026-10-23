import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import framing, runtime, server
from backend.models import models
from backend.tests.unit.test_partial_prediction import frame


def upload():
    image = io.BytesIO()
    Image.fromarray(np.zeros((768, 768), np.uint8)).save(image, format='PNG')
    return {'file': ('partial.png', image.getvalue(), 'image/png')}


@pytest.mark.parametrize('model', ['hrnet'])
@pytest.mark.parametrize('localizer', [True, False])
def test_stream_matches_legacy_prediction_for_partial_anatomy_in_both_modes(monkeypatch, model, localizer):
    from backend.tests.integration.fixed_contract import stub
    stub(monkeypatch)
    monkeypatch.setattr(server, 'calibration_from_payload', lambda *a, **k: {'status': 'unavailable', 'source_sha256': 'test'})
    data = {'modality': 'xray', 'body_part': 'lumbar', 'view': 'lateral', 'vertebra_model': model,
            'crop_localizer': str(localizer).lower()}
    client = TestClient(server.app)
    standard = client.post('/predict', data=data, files=upload()).json()
    for mode in ['standard', 'low-memory']:
        response = client.post('/predict-stream', data={**data, 'processing_mode': mode, 'cpu_threads': 1}, files=upload())
        assert response.status_code == 200
        events = [json.loads(line) for line in response.text.splitlines()]
        assert events[-1]['type'] == 'result'
        assert [e['type'] for e in events].count('result') == 1
        result = events[-1]['result']
        for key in ['measurements', 'geometry', 'image_png', 'mask_png', 'femoral_mask_png', 'calibration']:
            assert result[key] == standard[key]
        assert result['qc']['processing']['mode'] == mode
        assert result['qc']['processing']['crop_localizer'] is True
        assert result['qc']['processing']['runtime'] == 'onnxruntime'
        assert result['qc']['processing']['processor'] == {'requested': 'cpu', 'resolved': 'cpu', 'name': 'CPU', 'note': None}
        assert result['qc']['framing']['searched'] is True
        assert result['qc']['coverage']['partial']
        assert result['geometry']['femoral_circles'] == []
        assert result['measurements']['PI'] is None
        stages = [e['stage'] for e in events if e['type'] == 'progress']
        assert stages.index('decoding') < stages.index('landmarks') < stages.index('calibration') < stages.index('measuring') < stages.index('complete')


@pytest.mark.parametrize('mode,threads', [('unknown', 2), ('low-memory', 0), ('low-memory', 10)])
def test_bad_settings_fail_before_a_stream_is_started(mode, threads):
    response = TestClient(server.app).post('/predict-stream', data={'modality': 'xray', 'body_part': 'lumbar',
        'processing_mode': mode, 'cpu_threads': threads}, files=upload())
    assert response.status_code == 422


def test_invalid_localizer_is_rejected_before_prediction_starts():
    response = TestClient(server.app).post('/predict-stream', data={'modality': 'xray',
        'body_part': 'lumbar', 'crop_localizer': 'sometimes'}, files=upload())
    assert response.status_code == 422


def test_low_memory_models_are_released_on_failure(monkeypatch):
    releases = []
    monkeypatch.setattr(server, 'release_models', lambda: releases.append(True))
    monkeypatch.setattr(server, '_analyze', lambda **kwargs: (_ for _ in ()).throw(ValueError('failed')))
    with pytest.raises(ValueError): server.run_prediction({'settings': runtime.parse_options('low-memory', 1)})
    assert len(releases) == 2


@pytest.mark.parametrize('mode', ['standard', 'low-memory'])
def test_standalone_calibration_stream_keeps_policy_and_the_legacy_result(monkeypatch, mode):
    def calibrate(*args, cached=None):
        assert cached is None
        runtime.report('ocr', 'Reading scale labels', 1, 5)
        return {'status': 'unavailable', 'timeout': runtime.options().ocr_timeout}
    monkeypatch.setattr(server, 'calibration_from_payload', calibrate)
    client = TestClient(server.app)
    data = {'processing_mode': mode}
    original = client.post('/calibrate', data=data, files=upload()).json()
    response = client.post('/calibrate-stream', data=data, files=upload())
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[-1]['result'] == original
    assert original['timeout'] == (60 if mode == 'low-memory' else 8)
    assert any(e.get('stage') == 'ocr' and e['completed'] == 1 for e in events)
