"""Per-request resource policy, real progress and cooperative cancellation.

Inference is serialized to bound model/session memory across concurrent requests.
Context variables keep progress and calibration options request-local.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import os
import threading

try:
    from . import processors
except ImportError:  # Support running modules directly from backend/.
    import processors

class GpuFailure(RuntimeError):
    """Abort the GPU attempt; only the request boundary may retry on CPU."""
    def __init__(self, kind, reason):
        self.kind, self.reason = kind, reason
        super().__init__(f"{kind}: {reason}")


class Cancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class Options:
    mode: str = "standard"
    cpu_threads: int = 2
    crop_localizer: bool = True
    toolbar_removal: bool = False
    processor: str = "cpu"
    crop_method: str = "search"
    view_selection: str = "landmarks"

    @property
    def low_memory(self):
        return self.mode == "low-memory"

    @property
    def inference_threads(self):
        return self.cpu_threads if self.low_memory else min(4, os.cpu_count() or 1)

    @property
    def search_batch(self):
        return 4 if not self.low_memory and processor().provider == processors.CUDA else 1

    @property
    def ocr_timeout(self):
        return 60 if self.low_memory else 8


def parse_options(mode="standard", cpu_threads=2, crop_localizer=True, toolbar_removal=False,
                  processor="cpu", crop_method="search", view_selection="landmarks"):
    if mode not in ("standard", "low-memory"):
        raise ValueError("Processing mode must be standard or low-memory")
    if isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int) or not 1 <= cpu_threads <= 4:
        raise ValueError("CPU threads must be an integer from 1 to 4")
    if not isinstance(crop_localizer, bool):
        raise ValueError("Crop localizer must be on or off")
    if not isinstance(toolbar_removal, bool):
        raise ValueError("Toolbar removal must be on or off")
    if not processors.valid(processor):
        raise ValueError("Processor must be cpu or a GPU id")
    if crop_method not in ("search", "model"):
        raise ValueError("Crop method must be search or model")
    if view_selection not in ("landmarks", "classifier"):
        raise ValueError("View selection must be landmarks or classifier")
    # Accept older clients, but route the retired trained-crop method through search.
    return Options(mode, min(cpu_threads, os.cpu_count() or 1), crop_localizer, toolbar_removal,
                   processor, "search", view_selection)


_options = ContextVar("processing_options", default=Options())
_reporter = ContextVar("processing_reporter", default=None)
_cancel = ContextVar("processing_cancel", default=None)
_last_progress = ContextVar("processing_last_progress", default=None)
_providers = ContextVar("processing_providers", default=None)
_processor = ContextVar("processing_processor", default=(processors.CPU, None))
_qualification = ContextVar("gpu_qualification", default=None)
_fallbacks = ContextVar("gpu_fallbacks", default=None)
_batches = ContextVar("gpu_batches", default=None)
_lock = threading.Lock()


def options():
    return _options.get()


def checkpoint():
    cancelled = _cancel.get()
    if cancelled is not None and cancelled.is_set():
        raise Cancelled("Processing cancelled.")


def report(stage, message, completed=None, total=None):
    checkpoint()
    event = {"type": "progress", "stage": stage, "message": message,
             "completed": completed, "total": total}
    _last_progress.set(event)
    reporter = _reporter.get()
    if reporter:
        reporter(event)


def current_progress():
    return _last_progress.get()


def record_providers(kind, providers):
    current = _providers.get()
    if current is not None:
        current[kind] = list(dict.fromkeys(current.get(kind, []) + list(providers)))


def providers():
    return dict(_providers.get() or {})


def record_batch(kind, size, retries):
    records = _batches.get()
    if records is not None:
        record = records.setdefault(kind, {'max_batch': 0, 'oom_retries': 0, 'rows': 0})
        record['max_batch'] = max(record['max_batch'], size)
        record['oom_retries'] += retries
        record['rows'] += size


def batches():
    return {key: dict(value) for key, value in (_batches.get() or {}).items()}


def processor():
    """The processor this request's model sessions are created for."""
    return _processor.get()[0]


def processor_record():
    # Final request target, qualification evidence and whole-film fallback reason.
    chosen, note = _processor.get()
    record = {"requested": options().processor, "resolved": chosen.id, "name": chosen.name, "note": note}
    if _qualification.get() is not None:
        record['qualification'] = _qualification.get()
    if _fallbacks.get():
        record['fallbacks'] = dict(_fallbacks.get())
    return record


def record_qualification(record):
    _qualification.set(record)


def fallback_to_cpu(error):
    checkpoint()
    note = f"GPU processing failed ({error}); restarting the entire film on the CPU"
    _processor.set((processors.CPU, note))
    _fallbacks.set({error.kind: error.reason})
    _providers.set({})  # Only the successful CPU attempt belongs to the result.
    report("processor", note)


@contextmanager
def session(settings=None, reporter=None, cancelled=None):
    settings = settings or Options()
    tokens = (_options.set(settings), _reporter.set(reporter), _cancel.set(cancelled), _last_progress.set(None),
              _providers.set({}), _processor.set((processors.CPU, None)),
              _qualification.set(None), _fallbacks.set(None), _batches.set({}))
    acquired = False
    try:
        report("waiting", "Waiting for the processing worker")
        while not _lock.acquire(timeout=.1):
            checkpoint()
        acquired = True
        checkpoint()
        resolved = processors.resolve(settings.processor)
        _processor.set(resolved)
        if resolved[1]:
            report("processor", resolved[1])
        yield
    finally:
        if acquired:
            _lock.release()
        _options.reset(tokens[0])
        _reporter.reset(tokens[1])
        _cancel.reset(tokens[2])
        _last_progress.reset(tokens[3])
        _providers.reset(tokens[4])
        _processor.reset(tokens[5])
        _qualification.reset(tokens[6])
        _fallbacks.reset(tokens[7])
        _batches.reset(tokens[8])
