"""Compare completed inference outputs with workbook-extracted reference values."""
import argparse
import json
import math
from pathlib import Path
import numpy as np

KEYS = {'C2-C7 Cobb': 'C2C7_COBB', 'C2-C7 SVA': 'C2C7_SVA_MM', 'Global SVA': 'GLOBAL_SVA_MM',
        'L1-S1 Lordosis': 'LL', 'Sacral Slope': 'SS', 'Pelvic Incidence': 'PI', 'Pelvic Tilt': 'PT'}


def compare(reference, predictions, tolerance=.1):
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Comparison tolerance must be finite and nonnegative')
    summaries = {}
    for sheet, expected in reference.items():
        errors, mismatches, unavailable = [], [], []
        for case, target in expected.items():
            record = predictions.get(case)
            if record is None or record.get('error'):
                unavailable.append(case)
                continue
            value = record['measurements'].get(KEYS[sheet])
            if KEYS[sheet] == 'LL':
                value = (value or {}).get('L1-S1')
            if (value is None) != (target is None):
                mismatches.append({'case': case, 'expected': target, 'actual': value, 'availability_changed': True})
            elif value is not None:
                error = abs(value-target)
                errors.append(error)
                if error > tolerance:
                    mismatches.append({'case': case, 'expected': target, 'actual': value, 'absolute_error': error})
        summaries[sheet] = {'compared': len(errors), 'max_absolute_error': max(errors, default=None),
                            'mean_absolute_error': float(np.mean(errors)) if errors else None,
                            'mismatches': mismatches, 'unprocessed_or_failed': unavailable}
    return {'tolerance_degrees_or_mm': tolerance, 'passed': all(not v['mismatches'] and not v['unprocessed_or_failed'] for v in summaries.values()),
            'measurements': summaries}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tolerance', type=float, default=.1)
    a = p.parse_args()
    values = {f.stem: json.loads(f.read_text()) for f in a.predictions.glob('*.json') if not f.name.startswith('timing-')}
    result = compare(json.loads(a.reference.read_text()), values, a.tolerance)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
