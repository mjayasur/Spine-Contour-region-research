"""Export the unchanged S1 detector with four-image GPU throughput batches."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    import onnx
    from backend.models.training import build_s1_model
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    model = build_s1_model(768)
    model.load_state_dict(saved['model'])
    model.roi_heads.detections_per_img = 1
    model.eval()

    class Batch(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.detector = model

        def forward(self, image):
            predictions = self.detector([image[i] for i in range(4)])
            scores = torch.stack([torch.cat([p['scores'], p['scores'].new_zeros(1)])[0] for p in predictions])
            points = torch.stack([torch.cat([p['keypoints'], p['keypoints'].new_zeros((1, 2, 3))])[0] for p in predictions])
            return scores, points

    a.output.mkdir(parents=True, exist_ok=True)
    out = a.output/'s1_batch.onnx'
    with torch.inference_mode():
        torch.onnx.export(Batch().eval(), torch.zeros((4, 3, 768, 768)), str(out), dynamo=False,
                          opset_version=17, input_names=['image'], output_names=['scores', 'keypoints'])
    onnx.checker.check_model(str(out))
    out.with_suffix('.json').write_text(json.dumps({'kind': 's1_batch', 'batch': 4, 'precision': 'float32',
        'checkpoint_sha256': hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        'onnx_sha256': hashlib.sha256(out.read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
