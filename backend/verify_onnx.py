"""Offline packaged-model verification; no images or extra server endpoints."""
import hashlib
import json
import sys

import numpy as np
import onnxruntime as ort

from . import learned_region, processors, runtime
from .models import models


def verify(gpu=None, parity_films=None):
    assert 'torch' not in sys.modules, 'Desktop inference unexpectedly imports PyTorch'
    results = {}
    with runtime.session(runtime.parse_options('low-memory', 1, False)):
        for kind in models.MODEL_NAMES:
            path = models.ONNX_DIRECTORY / f'{kind}.onnx'
            if kind == 's1_batch' and not path.is_file():
                continue
            metadata = json.loads(path.with_suffix('.json').read_text())
            assert metadata['kind'] == kind and metadata['precision'] == 'float32'
            assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata['onnx_sha256']
            if kind == 'cervical_detr':
                inputs = {'pixel_values': np.zeros((1, 3, 800, 800), np.float32),
                          'pixel_mask': np.ones((1, 800, 800), np.int64)}
            elif kind in ('cervical_anatomical_hrnet', 'cervical_slope_hrnet'):
                inputs = {'image': np.zeros((1, 3, 512, 384), np.float32)}
            elif kind == 'lumbar_heatmaps':
                inputs = {'image': np.zeros((1, 1, 768, 768), np.float32)}
            elif kind == 's1_batch':
                inputs = {'image': np.zeros((4, 3, 768, 768), np.float32)}
            elif kind == 'cervical_hrnet':
                inputs = {'image': np.zeros((1, 3, 384, 384), np.float32)}
            else:
                size = models.FEMORAL_IMAGE_SIZE if kind == 'femoral' else models.MODEL_IMAGE_SIZE
                assert metadata['size'] == size
                shape = (1, 3 if kind == 's1' else 1, size, size)
                inputs = {'image': np.zeros(shape, np.float32)}
            output = models._infer(kind, lambda session: session.run(None, inputs), None)
            assert all(np.isfinite(value).all() for value in output)
            if kind in ('s1', 's1_batch'):
                assert output[0].ndim == 1 and output[1].shape == (len(output[0]), 2, 3)
            elif kind == 'cervical_detr':
                assert output[0].shape == (1, 100, 2) and output[1].shape == (1, 100, 4)
            else:
                expected = {'vertebra': (1, 6, 768, 768),
                            'femoral': (1, 1, models.FEMORAL_IMAGE_SIZE, models.FEMORAL_IMAGE_SIZE),
                            'hrnet': (1, 22, 2), 'cervical_hrnet': (1, 23, 96, 96),
                            'cervical_anatomical_hrnet': (1, 23, 128, 96),
                            'cervical_slope_hrnet': (1, 23, 128, 96), 'lumbar_heatmaps': (1, 22, 192, 192)}[kind]
                assert output[0].shape == expected
            results[kind] = [list(value.shape) for value in output]
        models.release_models()
        session, contract = learned_region._load(1)
        assert session.get_inputs()[0].shape == [1, 3, 960, 960]
        output = session.run([contract.output_name], {
            contract.input_name: np.zeros((1, 3, 960, 960), np.float32)})[0]
        assert output.shape[0:2] == (1, 6) and np.isfinite(output).all()
        results['crop_detector'] = [list(output.shape)]
        learned_region._load.cache_clear()
    # Exercise the production Apple provider configuration as well as CPU-only
    # low memory. Fallback remains valid and its actual providers are reported.
    if sys.platform == 'darwin':
        with runtime.session(runtime.parse_options('standard', 2, False)):
            models._infer('s1', lambda session: session.run(None, {'image': np.zeros((1, 3, 768, 768), np.float32)}), None)
            results['apple_s1_providers'] = runtime.providers()['s1']
            models.release_models()
    from .gpu_parity import verify_all, verify_films
    parity = verify_all(gpu)
    if parity_films:
        for identity, result in parity['gpus'].items():
            if result['passed']:
                try:
                    result['films'] = verify_films(parity_films, identity)
                    result['passed'] = all(film['passed'] for film in result['films'])
                    if not result['passed']: result['status'] = 'parity_failed'
                except Exception as error:
                    result.update(passed=False, status='gpu_failed', error_type=type(error).__name__)
        parity['passed'] = all(result['passed'] for result in parity['gpus'].values())
    print(json.dumps({'runtime': 'onnxruntime', 'available_providers': ort.get_available_providers(),
                      'gpus': [device.public() for device in processors.available()[1:]],
                      'verified': results, 'gpu_parity': parity}), flush=True)
    if not parity['passed']:
        raise SystemExit(4 if any(v['status'] == 'gpu_failed' for v in parity['gpus'].values()) else 3)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu')
    parser.add_argument('--parity-films')
    args = parser.parse_args()
    verify(args.gpu, args.parity_films)
