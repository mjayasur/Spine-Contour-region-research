"""Read only image IDs and AI measurements from the seven-sheet workbook.

Requires openpyxl in the validation environment, not in the desktop runtime.
Reader columns and workbook text are never passed to inference.
"""
import argparse
import json
import math
from pathlib import Path

SHEETS = ('C2-C7 Cobb', 'C2-C7 SVA', 'Global SVA', 'L1-S1 Lordosis',
          'Sacral Slope', 'Pelvic Incidence', 'Pelvic Tilt')


def extract(path):
    import openpyxl
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        result = {}
        for name in SHEETS:
            values = {}
            for row in workbook[name].iter_rows(min_row=2, values_only=True):
                if row[0] is None:
                    continue
                case = str(row[0]).zfill(6)
                if case in values:
                    raise ValueError(f'Duplicate image ID in {name}')
                value = row[5]
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                    raise ValueError(f'Non-numeric AI measurement in {name}')
                values[case] = value
            if len(values) != 99:
                raise ValueError(f'{name}: expected 99 image rows')
            result[name] = values
        if any(set(value) != set(result[SHEETS[0]]) for value in result.values()):
            raise ValueError('Measurement sheets have different image IDs')
        return result
    finally:
        workbook.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workbook', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(extract(a.workbook), indent=2, allow_nan=False))
