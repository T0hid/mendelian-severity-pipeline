"""Compare completed runs, allowing only documented terminology changes."""
import argparse
import csv
import json
import math
from pathlib import Path

RENAMES = {'S1_Mechanistic': 'S1_Gene_Level', 'S2_Mech_Pheno': 'S2_Combined',
           'feat_gnn_': 'feat_annot_', 'has_gnn_': 'has_annot_'}


def normalise(value):
    if isinstance(value, str):
        for old, new in RENAMES.items():
            value = value.replace(old, new)
    return value


def differences(old, new, path=''):
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(old.keys() | new.keys()):
            if key not in old or key not in new:
                yield f'{path}/{key}: missing key'
            else:
                yield from differences(old[key], new[key], f'{path}/{key}')
    elif isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            yield f'{path}: row count {len(old)} != {len(new)}'
        for i, (a, b) in enumerate(zip(old, new)):
            yield from differences(a, b, f'{path}/{i}')
    elif normalise(old) != normalise(new):
        try:
            a, b = float(old), float(new)
            if a == b or (math.isnan(a) and math.isnan(b)):
                return
        except (ValueError, TypeError):
            pass
        yield f'{path}: {old!r} != {new!r}'


def read(path):
    if path.suffix == '.json':
        return json.loads(path.read_text())
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def compare(previous, current):
    files = [('results/publication_table.csv', 'main/results/publication_table.csv'),
             ('results/model_selection_validation.csv', 'main/results/model_selection_validation.csv'),
             ('prenatal/statistics.json', 'prenatal/statistics.json')]
    report = {}
    for reference_path, run_path in files:
        a, b = previous / reference_path, current / run_path
        if not a.is_file() or not b.is_file():
            report[run_path] = [f'Missing file: {a if not a.is_file() else b}']
        else:
            report[run_path] = list(differences(read(a), read(b), run_path))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, default=Path('data/reference'))
    parser.add_argument('--current', type=Path, default=Path('results'))
    args = parser.parse_args()
    report = compare(args.previous, args.current)
    print(json.dumps({'status': 'different' if any(report.values()) else 'identical',
                      'differences': report}, indent=2))
    raise SystemExit(1 if any(report.values()) else 0)
