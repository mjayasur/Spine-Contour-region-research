"""Exercise the desktop HTTP route with real files and fresh localization/OCR."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from fastapi.testclient import TestClient
    from backend import processors
    from backend.server import app
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    a = p.parse_args()
    device = next(x for x in processors.available() if x.provider == processors.CUDA and x.adapter == a.gpu)
    a.output.mkdir(parents=True, exist_ok=True)
    client = TestClient(app)
    for row in json.loads(a.manifest.read_text()):
        started = time.monotonic()
        path = Path(row['image'])
        response = client.post('/predict', data={'modality': 'xray', 'body_part': 'full_spine',
            'view': 'lateral', 'processor': device.id}, files={'file': (path.name, path.read_bytes(), 'image/jpeg')})
        record = response.json()
        for key in ('image_png', 'mask_png', 'femoral_mask_png'):
            record.pop(key, None)
        record.update(http_status=response.status_code, seconds=time.monotonic()-started)
        (a.output/f'{row["case"]}.json').write_text(json.dumps(record, indent=2, allow_nan=False))
        print(row['case'], response.status_code, round(record['seconds'], 2), flush=True)
        if response.status_code != 200:
            raise RuntimeError(record)


if __name__ == '__main__':
    main()
