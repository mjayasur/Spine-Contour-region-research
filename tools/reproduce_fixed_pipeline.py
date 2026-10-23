"""Run the application pipeline on a private manifest; compare separately.

Manifest rows contain image paths, image-derived localization and calibrated
pixel spacing. No reader or expected AI values enter the inference process.
"""
import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from backend import runtime, processors
    from backend.fixed_pipeline import predict
    from backend.measurement_routes import measure
    from backend.models.models import model_versions
    import onnxruntime as ort
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--fresh', action='store_true')
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    if a.shards < 1 or not 0 <= a.shard < a.shards or (a.limit is not None and a.limit < 1):
        p.error('Require positive shard count/limit and 0 <= shard < shards')
    processor = next(x for x in processors.available() if x.provider == processors.CUDA and x.adapter == a.gpu)
    rows = json.loads(a.manifest.read_text())[a.shard::a.shards][:a.limit]
    a.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    failures = 0
    for row in rows:
        start = time.monotonic()
        record = {'case': row['case'], 'localization': 'fresh' if a.fresh else 'frozen_image_derived'}
        try:
            path = Path(row['image'])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != row['sha256']:
                raise ValueError('Image hash differs from evaluation manifest')
            raw = np.asarray(Image.open(path).convert('L'))
            with runtime.session(runtime.parse_options(processor=processor.id)):
                prediction = predict(raw, localization=None if a.fresh else row['localization'])
                geometry = prediction['landmarks']
                geometry['pixel_spacing'] = [row['mm_per_pixel']]*2
                analysis = measure(geometry)
                record.update(analysis, framing=prediction['framing'], provenance=prediction['provenance'],
                              providers=runtime.providers(), batches=runtime.batches(), source_sha256=digest,
                              model_versions=model_versions(prediction['models']),
                              runtime_version=ort.__version__, processor=runtime.processor_record())
        except Exception as error:
            failures += 1
            record.update(error=str(error), traceback=traceback.format_exc())
        record['seconds'] = time.monotonic()-start
        (a.output/f'{row["case"]}.json').write_text(json.dumps(record, indent=2, allow_nan=False))
        print(row['case'], round(record['seconds'], 2), record.get('error', 'ok'), flush=True)
    (a.output/f'timing-{a.shard}.json').write_text(json.dumps({'seconds': time.monotonic()-started,
        'images': len(rows), 'failures': failures, 'localization': 'fresh' if a.fresh else 'frozen_image_derived',
        'calibration': 'manifest', 'includes_http_encoding_and_ocr': False}))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
