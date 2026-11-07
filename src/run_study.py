"""One entry point for benchmark, scorer training, prenatal analysis and review."""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from run_pipeline import require_fresh_output, sha256_file

from runtime_paths import reference_directory

ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = reference_directory(ROOT)
SCORER = 'phenotype_scorer.py'
PRENATAL = 'prenatal_analysis.py'


def check_scorer_inputs(data_dir, no_visualizations=False):
    reference = json.loads((REFERENCE_DIR / 'input_manifest.json').read_text(encoding='utf-8'))
    required = {item['relative_path']: item['expected_sha256'] for item in reference
                if item['label'] in {'primary_data', 'HPO', 'rebuilt_gene_disease_mapping'}}
    rows = []
    for name, expected in required.items():
        path = data_dir / name
        actual = sha256_file(path) if path.is_file() else None
        rows.append({'file': name, 'status': 'missing' if actual is None else
                     'mismatch' if expected and actual != expected else 'matched' if expected else 'present_hash_unrecorded'})
    modules = ['numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib'] + ([] if no_visualizations else ['shap'])
    dependencies = {name: importlib.util.find_spec(name) is not None for name in modules}
    return {'status': 'passed' if all(row['status'] in {'matched', 'present_hash_unrecorded'} for row in rows)
            and all(dependencies.values()) else 'blocked', 'inputs': rows, 'dependencies': dependencies}


def run_command(command, output, cwd=None):
    require_fresh_output(output)
    result = subprocess.run(command, cwd=cwd)
    if result.returncode:
        return result.returncode
    record = {'command': command, 'python': sys.version,
              'source_files': {p.name: sha256_file(p) for p in [ROOT / SCORER, ROOT / PRENATAL, ROOT / 'severity_pipeline.py']}}
    (output / 'workflow_execution.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('verify', help='Check saved results; no model training or new inferential analyses.')
    check.add_argument('--benchmark-results', type=Path)
    check.add_argument('--scorer-results', type=Path)
    check.add_argument('--prenatal-results', type=Path)
    check.add_argument('--model', type=Path)
    check.add_argument('--cohort', type=Path)
    check.add_argument('--report', type=Path)
    check.add_argument('--check-original-files', action='store_true',
                       help='Require complete original-folder file fingerprints, including models and family files.')
    benchmark = sub.add_parser('benchmark', help='Run the gene-disease S1/S1b/S2/S3 benchmark.')
    benchmark.add_argument('--data-dir', type=Path, required=True)
    benchmark.add_argument('--output-dir', type=Path)
    benchmark.add_argument('--permutations', type=int)
    benchmark.add_argument('--check-inputs', action='store_true')
    train = sub.add_parser('train-scorer', help='Train/evaluate and refit the separate disease-level scorer.')
    train.add_argument('--data-dir', type=Path, required=True)
    train.add_argument('--output-dir', type=Path)
    train.add_argument('--check-inputs', action='store_true')
    train.add_argument('--no-visualizations', action='store_true')
    train.add_argument('--max-configs', type=int, default=24)
    train.add_argument('--cv-folds', type=int, default=5)
    train.add_argument('--cv-repeats', type=int, default=3)
    prenatal = sub.add_parser('prenatal', help='Apply a frozen scorer to a local cohort and analyse outcomes.')
    prenatal.add_argument('--cohort', type=Path, required=True)
    prenatal.add_argument('--model', type=Path, default=ROOT / 'artifacts/phenotype_scorer/model.pkl')
    prenatal.add_argument('--scorer-heldout', type=Path, default=REFERENCE_DIR / 'scorer/held_out_test_predictions.csv')
    prenatal.add_argument('--output-dir', type=Path, required=True)
    prenatal.add_argument('--score-channel', choices=['phenotype', 'combined', 'headline'], default='phenotype')
    prenatal.add_argument('--n-boot', type=int, default=2000)
    prenatal.add_argument('--n-perm', type=int, default=2000)
    prenatal.add_argument('--check-inputs', action='store_true')
    demo = sub.add_parser('score-demo', help='Score invented demonstration profiles with the saved scorer.')
    demo.add_argument('--model', type=Path, default=ROOT / 'artifacts/phenotype_scorer/model.pkl')
    args = parser.parse_args(argv)
    if args.command == 'verify':
        from verify_results import verify as verify_benchmark
        from verify_clinical import verify as verify_clinical
        if args.cohort and (not args.prenatal_results or not args.model):
            parser.error('Local cohort replay requires --prenatal-results and --model')
        report = {
            'benchmark': verify_benchmark(args.benchmark_results or REFERENCE_DIR,
                                          check_file_fingerprints=args.check_original_files),
            'clinical': verify_clinical(args.scorer_results, args.prenatal_results, args.model, args.cohort,
                                       args.check_original_files),
            'status': 'passed', 'training_performed': False, 'individual_prenatal_records_exported': 0,
        }
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'status': report['status'], 'benchmark_metrics': report['benchmark']['metric_checks'],
                          'scorer_metrics': report['clinical']['scorer']['metric_comparisons'],
                          'prenatal_aggregate_checks': report['clinical']['prenatal_aggregate']['aggregate_numeric_checks'],
                          'model_training_performed': False}, indent=2))
        return 0
    if args.command == 'benchmark':
        command = [sys.executable, str(ROOT / 'run_severity_pipeline.py'), '--data-dir', str(args.data_dir.resolve())]
        if args.output_dir:
            command += ['--output-dir', str(args.output_dir.resolve())]
        if args.permutations is not None:
            command += ['--permutations', str(args.permutations)]
        if args.check_inputs:
            command += ['--check-inputs']
        return subprocess.run(command).returncode
    if args.command == 'train-scorer':
        data_dir = args.data_dir.resolve()
        readiness = check_scorer_inputs(data_dir, args.no_visualizations)
        if args.check_inputs or readiness['status'] != 'passed':
            print(json.dumps(readiness, indent=2))
            return 0 if readiness['status'] == 'passed' else 2
        if not args.output_dir:
            parser.error('--output-dir is required for training')
        if min(args.max_configs, args.cv_folds, args.cv_repeats) < 1:
            parser.error('Training counts must be positive')
        output = args.output_dir.resolve()
        command = [sys.executable, str(ROOT / SCORER), '--severity-csv', str(data_dir / 'final_output_with_severity.csv'),
                   '--hpo-obo', str(data_dir / 'hp.obo'), '--output', str(output),
                   '--max-configs', str(args.max_configs), '--cv-folds', str(args.cv_folds), '--cv-repeats', str(args.cv_repeats),
                   '--severity-term-policy', 'phenotype']
        if args.no_visualizations:
            command += ['--no-visualizations']
        return run_command(command, output, data_dir)
    if args.command == 'prenatal':
        if args.n_boot < 1 or args.n_perm < 1:
            parser.error('Bootstrap/permutation counts must be positive')
        dependencies = {name: importlib.util.find_spec(name) is not None
                        for name in ['numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'statsmodels']}
        readiness = {'cohort_exists': args.cohort.is_file(), 'model_exists': args.model.is_file(),
                     'dependencies': dependencies}
        ready = readiness['cohort_exists'] and readiness['model_exists'] and all(dependencies.values())
        if args.check_inputs or not ready:
            print(json.dumps(dict(readiness, status='passed' if ready else 'blocked'), indent=2))
            return 0 if ready else 2
        output = args.output_dir.resolve()
        command = [sys.executable, str(ROOT / PRENATAL), '--cohort', str(args.cohort.resolve()),
                   '--model', str(args.model.resolve()), '--scorer-code', str(ROOT / SCORER),
                   '--scorer-heldout', str(args.scorer_heldout.resolve()),
                   '--score-channel', args.score_channel, '--n-boot', str(args.n_boot), '--n-perm', str(args.n_perm),
                   '--output', str(output)]
        return run_command(command, output)
    if args.command == 'score-demo':
        from verify_clinical import load_module, load_scorer, synthetic_predictions
        module = load_module(ROOT / PRENATAL, 'prenatal_severity_analysis_v3_multichannel')
        model = load_scorer(args.model.resolve(), module)
        print(json.dumps(synthetic_predictions(model), indent=2))
        return 0
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
