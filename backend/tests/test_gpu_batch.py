import numpy as np
import pytest
from types import SimpleNamespace
from backend.gpu_batch import run


class Session:
    def __init__(self, capacity=3, error=None):
        self.capacity, self.error, self.attempts = capacity, error, []

    def get_inputs(self):
        return [SimpleNamespace(shape=['batch', 1])]

    def get_providers(self):
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']

    def run(self, names, inputs):
        x = inputs['image']
        self.attempts.append(len(x))
        if self.error:
            raise self.error
        if len(x) > self.capacity:
            raise RuntimeError('CUDA out of memory')
        return [x*2, x+1]


def test_oom_retries_preserve_all_rows_in_order():
    session = Session()
    x = np.arange(11, dtype=np.float32)[:, None]
    a, b = run(session, None, {'image': x})
    np.testing.assert_array_equal(a, x*2)
    np.testing.assert_array_equal(b, x+1)
    assert session.attempts[:3] == [11, 5, 2]


def test_single_row_oom_is_not_an_infinite_retry():
    session = Session(capacity=0)
    with pytest.raises(RuntimeError, match='out of memory'):
        run(session, None, {'image': np.zeros((4, 1))})
    assert session.attempts == [4, 2, 1]


def test_non_memory_errors_are_not_retried():
    session = Session(error=ValueError('invalid model output'))
    with pytest.raises(ValueError, match='invalid model output'):
        run(session, None, {'image': np.zeros((4, 1))})
    assert session.attempts == [4]


def test_cpu_uses_bounded_batches():
    session = Session()
    session.get_providers = lambda: ['CPUExecutionProvider']
    run(session, None, {'image': np.zeros((4, 1))})
    assert session.attempts == [1]*4


def test_stage_oom_evicts_inactive_models_and_retries_once(monkeypatch):
    from backend import runtime, processors
    from backend.models import models
    monkeypatch.setattr(runtime, 'processor', lambda: SimpleNamespace(id='cuda:test', adapter=0, provider=processors.CUDA))
    monkeypatch.setattr(models, '_cache_policy', ((4, False, 0), 'cuda:test'))
    monkeypatch.setattr(runtime, 'options', lambda: SimpleNamespace(inference_threads=4, low_memory=False))
    monkeypatch.setattr(models, '_load_model', lambda *a: Session())
    evictions, attempts = [], []
    monkeypatch.setattr(models, 'release_models', lambda: evictions.append(True))
    def operation(session):
        attempts.append(True)
        if len(attempts) == 1:
            raise runtime.GpuFailure('test', 'CUDA out of memory')
        return 42
    assert models._infer('test', operation, None) == 42
    assert len(evictions) == 1 and len(attempts) == 2


def test_stage_oom_after_eviction_reaches_request_fallback(monkeypatch):
    from backend import runtime, processors
    from backend.models import models
    monkeypatch.setattr(runtime, 'processor', lambda: SimpleNamespace(id='cuda:test', adapter=0, provider=processors.CUDA))
    monkeypatch.setattr(models, '_cache_policy', ((4, False, 0), 'cuda:test'))
    monkeypatch.setattr(runtime, 'options', lambda: SimpleNamespace(inference_threads=4, low_memory=False))
    monkeypatch.setattr(models, '_load_model', lambda *a: Session())
    monkeypatch.setattr(models, 'release_models', lambda: None)
    attempts = []
    def operation(session):
        attempts.append(True)
        raise runtime.GpuFailure('test', 'CUDA out of memory')
    with pytest.raises(runtime.GpuFailure):
        models._infer('test', operation, None)
    assert len(attempts) == 2
