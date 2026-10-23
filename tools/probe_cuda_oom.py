"""Force a real CUDA arena OOM on real crop tensors, then check decoded points."""
import argparse
import gc
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from backend import framing
    from backend.models import models
    from backend.fixed_pipeline import crop_windows
    from backend.gpu_batch import run
    from backend.decoders import lumbar_points
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--memory-mib', type=int, default=1024)
    a = p.parse_args()
    row = json.loads(a.manifest.read_text())[0]
    raw = np.asarray(Image.open(row['image']).convert('L'))
    loc = row['localization']
    if loc['anterior_side'] == 'right':
        raw = raw[:, ::-1].copy()
    windows = crop_windows(loc['lumbar_localizer']['window'], raw.shape, 'original_plus_up15')
    x = np.concatenate([models._segmentation_input(framing.prepare_crop(raw, w)[0]) for w in windows])
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = 2
    def session(budget=None):
        config = {'device_id': a.gpu, 'cudnn_conv_algo_search': 'HEURISTIC'}
        if budget is not None:
            config.update(gpu_mem_limit=budget, arena_extend_strategy='kSameAsRequested')
        result = ort.InferenceSession(str(models.ONNX_DIRECTORY/'lumbar_heatmaps.onnx'), sess_options=settings,
                                     providers=[('CUDAExecutionProvider', config), 'CPUExecutionProvider'])
        if result.get_providers()[0] != 'CUDAExecutionProvider':
            raise RuntimeError('CUDA provider did not load')
        return result
    reference = session()
    expected = np.concatenate([reference.run(None, {'image': v[None]})[0] for v in x])
    del reference
    gc.collect()
    limited = session(a.memory_mib*1024**2)
    events = []
    actual = run(limited, None, {'image': x},
                 report=lambda size, retries, limit: events.append(dict(size=size, retries=retries, limit=limit)))[0]
    difference = float(np.abs(lumbar_points(actual)-lumbar_points(expected)).max())
    passed = sum(v['size'] for v in events) == len(x) and any(v['retries'] for v in events) and difference <= .25
    a.output.write_text(json.dumps(dict(passed=passed, crops=len(x), arena_limit_mib=a.memory_mib,
        decoded_tolerance_model_pixels=.25, max_decoded_difference_model_pixels=difference,
        max_heatmap_difference=float(np.abs(actual-expected).max()), batches=events), indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
