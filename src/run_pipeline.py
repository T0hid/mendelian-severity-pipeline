"""Check exact reference inputs, then launch the complete analysis."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
from runtime_paths import reference_directory

PACKAGE_DIR = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_inputs(data_dir: Path) -> dict:
    items = json.loads((reference_directory(PACKAGE_DIR) / 'input_manifest.json').read_text())
    checks = []
    for item in items:
        path = data_dir / ('gene_annotations.csv' if item['relative_path'] == 'final_gnn_dataset.csv' else item['relative_path'])
        row = dict(item, path=str(path), exists=path.is_file())
        row['actual_sha256'] = sha256_file(path) if path.is_file() else None
        expected = item['expected_sha256']
        row['status'] = ('missing' if not row['exists'] else
                         'hash_mismatch' if expected and row['actual_sha256'] != expected else
                         'present_hash_not_recorded' if expected is None else 'matched')
        if not row['exists'] and not item.get('required_for_fitted_models', True):
            row['status'] = 'optional_excluded_not_provided'
        checks.append(row)
    modules = {name: importlib.util.find_spec(name) is not None for name in
               ['numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn', 'sklearn', 'lightgbm', 'shap', 'imblearn']}
    # SHAP and imbalanced-learn are necessary to reproduce the complete reference outputs.
    return {'status': 'passed' if all(x['status'] in {'matched', 'present_hash_not_recorded', 'optional_excluded_not_provided'} for x in checks)
            and all(modules.values()) else 'blocked', 'inputs': checks, 'dependencies': modules}


def require_fresh_output(output_dir: Path) -> None:
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f'Choose a fresh or empty output directory: {output_dir}')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Directory containing the original input tables and pinned ontologies.')
    parser.add_argument('--output-dir', type=Path,
                        help='Fresh output directory; existing results are never overwritten.')
    parser.add_argument('--check-inputs', action='store_true', help='Report missing inputs/dependencies without importing the pipeline.')
    parser.add_argument('--permutations', type=int,
                        help='Explicitly change the reference count of 100; a different count produces a new inference run.')
    args = parser.parse_args(argv)
    data_dir = args.data_dir.resolve()
    if args.permutations is not None and args.permutations < 1:
        parser.error('--permutations must be at least 1')
    report = check_inputs(data_dir)
    if args.check_inputs:
        print(json.dumps(report, indent=2))
        return 0 if report['status'] == 'passed' else 2
    if args.output_dir is None:
        parser.error('--output-dir is required for an analysis run')
    output_dir = args.output_dir.resolve()
    require_fresh_output(output_dir)
    if report['status'] != 'passed':
        print(json.dumps(report, indent=2))
        return 2
    os.chdir(data_dir)
    pipeline = importlib.import_module('severity_pipeline')
    pipeline.Config.OUTPUT_DIR = str(output_dir)
    if args.permutations is not None:
        pipeline.Config.N_PERMUTATIONS = args.permutations
    pipeline.main()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
