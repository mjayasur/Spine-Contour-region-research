"""Adaptive, ordered ONNX batches with bounded CUDA out-of-memory recovery.

Only allocation failures shrink batches. Other execution errors propagate to
the request boundary. Inputs are retried without skipping or duplicating rows.
"""
import subprocess
import numpy as np
from . import runtime

MIB = 1024**2


def devices():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.free',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True,
                                timeout=3, check=True)
        rows = []
        for line in result.stdout.splitlines():
            index, identity, name, total, free = [s.strip() for s in line.split(',', 4)]
            rows.append({'index': int(index), 'uuid': identity, 'name': name,
                         'total': int(total)*MIB, 'free': int(free)*MIB})
        return rows
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def allocation_failure(error):
    text = str(error).lower()
    return any(marker in text for marker in ('out of memory', 'cudaerrormemoryallocation',
                                             'failed to allocate memory', 'cudnn_status_alloc_failed',
                                             'bfcarena::allocaterawinternal'))


def run(session, output_names, inputs, *, batch_limit=None, report=None):
    """Dynamic-batch graphs only; fixed graphs keep their supported batch size."""
    spec = session.get_inputs()
    if any(not x.shape or isinstance(x.shape[0], int) for x in spec):
        return session.run(output_names, inputs)
    sizes = {len(value) for value in inputs.values()}
    if len(sizes) != 1 or not sizes or next(iter(sizes)) == 0:
        raise ValueError('Batched model inputs must have the same nonzero row count')
    count = sizes.pop()
    cuda = 'CUDAExecutionProvider' in session.get_providers()
    # Start aggressively; allocation failures establish a safe ceiling per graph.
    limit = batch_limit if batch_limit is not None else count if cuda else 1
    limit = max(1, min(count, limit))
    chunks, offset, retries = [], 0, 0
    while offset < count:
        runtime.checkpoint()
        size = min(limit, count-offset)
        batch = {name: np.ascontiguousarray(value[offset:offset+size]) for name, value in inputs.items()}
        try:
            outputs = session.run(output_names, batch)
        except Exception as error:
            if not cuda or not allocation_failure(error) or size == 1:
                raise
            limit = max(1, size//2)
            retries += 1
            runtime.report('gpu_memory', f'GPU memory limit reached; retrying {limit} crops at a time')
            continue
        if any(len(value) != size for value in outputs):
            raise ValueError('Dynamic model output batch differs from its input')
        chunks.append(outputs)
        offset += size
        if report:
            report(size, retries, limit)
        retries = 0
    return [np.concatenate([chunk[i] for chunk in chunks], axis=0) for i in range(len(chunks[0]))]
