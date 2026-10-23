"""Export dynamic-batch HRNet graphs from the fixed production checkpoints.

Existing S1, femoral and automatic-localization graphs retain their exports.
Checkpoint paths are explicit inputs; no training or evaluation dataset is bundled.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def export(kind, checkpoint, image, output):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from PIL import Image
    from backend.models.cervical_anatomical_training import HRNetKeypointModel
    from backend.models.cervical_anatomical import landmark_input
    from backend.models.hrnet import build_hrnet_model
    from backend.models.models import _letterbox, _robust_rescale, _segmentation_input
    torch.set_num_threads(2)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    raw = np.asarray(Image.open(image).convert('L'))
    if kind == 'lumbar_heatmaps':
        model = build_hrnet_model(saved)
        example = _segmentation_input(_letterbox(_robust_rescale(raw))[0])
        expected_epoch = 45
    else:
        model = HRNetKeypointModel('hrnet_w32.ms_in1k', 23, False)
        model.load_state_dict(saved['model'], strict=True)
        example = landmark_input(raw)[0]
        expected_epoch = 12 if kind == 'cervical_slope_hrnet' else 87
    if saved['epoch'] != expected_epoch:
        raise ValueError(f'{kind}: expected epoch {expected_epoch}, got {saved["epoch"]}')
    model.eval()
    output.mkdir(parents=True, exist_ok=True)
    path = output/f'{kind}.onnx'
    with torch.inference_mode():
        torch.onnx.export(model, torch.from_numpy(example), str(path), opset_version=17,
                          input_names=['image'], output_names=['heatmaps'], dynamo=False,
                          dynamic_axes={'image': {0: 'batch'}, 'heatmaps': {0: 'batch'}})
    onnx.checker.check_model(str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    checks = []
    for batch in (1, 2):
        x = np.repeat(example, batch, axis=0)
        with torch.inference_mode():
            expected = model(torch.from_numpy(x)).numpy()
        actual = session.run(None, {'image': x})[0]
        np.testing.assert_allclose(actual, expected, rtol=.002, atol=.002)
        checks.append({'batch': batch, 'max_heatmap_error': float(np.abs(actual-expected).max())})
    metadata = {'kind': kind, 'checkpoint_epoch': saved['epoch'], 'precision': 'float32',
                'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'onnx_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'dynamic_batch': True, 'checks': checks}
    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', required=True, choices=['lumbar_heatmaps', 'cervical_slope_hrnet', 'cervical_anatomical_hrnet'])
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--output', type=Path, default=ROOT/'backend/onnx')
    a = p.parse_args()
    export(a.kind, a.checkpoint, a.image, a.output)
