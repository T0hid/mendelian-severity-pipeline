"""Verify saved predictions and result tables without fitting any model."""
from __future__ import annotations

import argparse
import ast
import collections
import copy
import hashlib
import importlib.metadata
import json
import os
import pickle
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn import metrics as sklearn_metrics

from verify_source import verify_source

from runtime_paths import reference_directory

PACKAGE_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = reference_directory(PACKAGE_DIR)


class CheckFailure(ValueError):
    """A stored result does not satisfy a reproducibility check."""


def require(condition, message):
    if not condition:
        raise CheckFailure(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_functions(names, config, source_path=None):
    """Load exact pure helpers from the source AST, without importing training dependencies.

    Only requested function definitions and their function dependencies are executed.
    Module-level imports, output creation and the analysis entry point are excluded.
    """
    source_path = Path(source_path or PACKAGE_DIR / 'severity_pipeline.py')
    tree = ast.parse(source_path.read_text(encoding='utf-8-sig'))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    needed, pending = set(), list(names)
    while pending:
        name = pending.pop()
        if name in needed:
            continue
        require(name in functions, f'Source helper not found: {name}')
        needed.add(name)
        dependencies = {n.id for n in ast.walk(functions[name])
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} & functions.keys()
        pending.extend(dependencies - needed)
    namespace = dict(np=np, pd=pd, os=os, re=re, json=json, spearmanr=spearmanr,
                     Config=SimpleNamespace(**config), RANDOM_SEED=42,
                     defaultdict=collections.defaultdict, Counter=collections.Counter,
                     logger=SimpleNamespace(log=lambda *args, **kwargs: None))
    namespace.update({name: getattr(sklearn_metrics, name) for name in dir(sklearn_metrics)
                      if not name.startswith('_')})
    selected = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    selected.extend(copy.deepcopy(n) for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name in needed)
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(source_path), 'exec'), namespace)
    return namespace


def load_predictions(results_dir):
    compact = results_dir / 'locked_test_predictions.npz'
    if compact.is_file():
        with np.load(compact, allow_pickle=False) as archive:
            keys = json.loads(str(archive['prediction_keys']))
            return {key: {field: archive[f'p{i}_{field}'].copy()
                          for field in ['y_true', 'y_pred', 'y_prob']}
                    for i, key in enumerate(keys)}, 'numeric NPZ snapshot'
    # This is the caller's own results-collector pickle, not a trained-model bundle.
    with (results_dir / 'results/all_results.pkl').open('rb') as handle:
        collector = pickle.load(handle)
    return collector['predictions'], 'supplied results/all_results.pkl'


def validate_prediction_arrays(item, n_classes):
    y, predicted, probability = (np.asarray(item[k]) for k in ['y_true', 'y_pred', 'y_prob'])
    require(y.ndim == predicted.ndim == 1 and len(y) == len(predicted), 'Prediction dimensions differ')
    require(probability.shape == (len(y), n_classes), 'Probability dimensions differ')
    require(np.isfinite(probability).all(), 'Nonfinite prediction probabilities')
    require((probability >= -1e-12).all() and (probability <= 1 + 1e-12).all(), 'Probabilities outside [0, 1]')
    require(np.allclose(probability.sum(axis=1), 1, atol=1e-10, rtol=0), 'Probability rows do not sum to one')
    require(np.array_equal(predicted, probability.argmax(axis=1)), 'Predicted class differs from probability argmax')
    require(np.isin(y, np.arange(n_classes)).all(), 'Unknown target class')


def compare_metrics(predictions, table, helpers):
    comparisons, scenario_counts = [], {}
    require(not table.duplicated(['Scenario', 'Stratum', 'Model']).any(), 'Duplicate metric rows')
    for row in table.to_dict('records'):
        key = f"{row['Scenario']}_{row['Model']}"
        require(key in predictions, f'Missing stored predictions: {key}')
        item = predictions[key]
        validate_prediction_arrays(item, helpers['Config'].NUM_CLASSES)
        computed = helpers['compute_all_metrics'](**{
            'y_true': item['y_true'], 'y_pred': item['y_pred'], 'y_prob': item['y_prob']})
        count = 0
        for metric, value in computed.items():
            require(metric in row, f'Missing metric column: {metric}')
            stored = row[metric]
            if isinstance(value, str):
                matched = str(stored) == value
                error = 0.0 if matched else None
            else:
                matched = bool(np.isclose(value, stored, rtol=1e-12, atol=1e-12, equal_nan=True))
                error = float(abs(value - stored)) if np.isfinite(value) and np.isfinite(stored) else 0.0
            require(matched, f'Metric mismatch: {key}/{metric}: computed={value}, saved={stored}')
            comparisons.append({'Scenario': row['Scenario'], 'Model': row['Model'], 'metric': metric,
                                'absolute_error': error, 'matched': matched})
            count += 1
        scenario_counts[row['Scenario']] = {'n_pairs': len(item['y_true']), 'metric_checks': count}
    require(len(predictions) == len(table), 'Stored predictions and headline result rows differ')
    return comparisons, scenario_counts


def check_split_frame(frame, mode):
    require(not frame['row_index'].duplicated().any(), f'Duplicate split rows: {mode}')
    require(set(frame['partition']) == {'train', 'validation', 'test'}, f'Incomplete partition names: {mode}')
    required_ids = ['gene_symbol'] if mode == 'primary' else (
        ['mondo_id'] if mode == 'disease_disjoint' else ['gene_symbol', 'mondo_id'])
    for name in required_ids:
        sets = {partition: set(block[name].dropna()) for partition, block in frame.groupby('partition')}
        for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
            require(not sets[a] & sets[b], f'{name} overlaps {a}/{b}: {mode}')


def check_model_selection(selection, manifest, helpers):
    chosen = {}
    for scenario, rows in selection.groupby('Scenario', sort=False):
        require(len(rows) == len(manifest['config']['CANDIDATE_MODEL_ORDER']), f'Incomplete candidate rows: {scenario}')
        candidates = {row['Model']: {'val_metrics': {
            'qwk': row['validation_qwk'], 'mcc': row['validation_mcc'],
            'balanced_accuracy': row['validation_balanced_accuracy']}}
            for row in rows.to_dict('records')}
        selected, _ = helpers['_select_candidate_lexicographically'](
            candidates, manifest['config']['CANDIDATE_MODEL_ORDER'])
        flags = rows['selected_on_validation'].astype(str).str.lower().eq('true')
        require(flags.sum() == 1, f'Expected exactly one validation-selected model: {scenario}')
        require(rows.loc[flags, 'Model'].iloc[0] == selected, f'Validation-selection mismatch: {scenario}')
        require(selected == manifest['selected_models'][scenario], f'Manifest model mismatch: {scenario}')
        chosen[scenario] = selected
    require(chosen == manifest['selected_models'], 'Scenario sets differ')
    return chosen


def check_config(manifest):
    tree = ast.parse((PACKAGE_DIR / 'severity_pipeline.py').read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Config')
    namespace = {'os': os}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<Config>', 'exec'), namespace)
    current = vars(namespace['Config'])
    path_keys = {'BASE_DIR', 'OUTPUT_DIR', 'CONSOLIDATED_DIR', 'PRIMARY_DATA', 'GNN_DATA',
                 'CORE_FEATURES', 'DOMAIN_FEATURES', 'EXTERNAL_FEATURES', 'NEW_GENE_FEATURES',
                 'MOUSE_LETHALITY_FEATURES', 'MOUSE_HUMAN_PHENOTYPE_FEATURES', 'HPO_OBO_PATH',
                 'MONDO_OBO_PATH', 'REBUILT_GENE_DISEASE_MAPPING'}
    def restore_names(value):
        if isinstance(value, dict):
            return {restore_names(k): restore_names(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [restore_names(v) for v in value]
        if isinstance(value, str):
            value = {'publication_figures': 'publication_figures_v2', 'run_report': 'manuscript_report'}.get(value, value)
            for new, old in [('S1_Gene_Level','S1_Mechanistic'), ('S2_Combined','S2_Mech_Pheno'),
                             ('annotation_text','gnn_text'), ('feat_annot_','feat_gnn_'),
                             ('has_annot_','has_gnn_')]:
                value = value.replace(new, old)
        return value
    checked = []
    for key, value in manifest['config'].items():
        if key in path_keys:
            continue
        if key in {'USE_ORGAN_CRITICALITY_FEATURES', 'DROP_HPO_PCA', 'HPO_PCA_N_COMPONENTS'} and key not in current:
            require(value == {'USE_ORGAN_CRITICALITY_FEATURES': False, 'DROP_HPO_PCA': True, 'HPO_PCA_N_COMPONENTS': 30}[key], 'Removed option differs from the reference configuration')
            continue
        require(key in current, f'Missing configuration: {key}')
        require(json.dumps(restore_names(current[key]), sort_keys=True) == json.dumps(value, sort_keys=True),
                f'Scientific configuration changed: {key}')
        checked.append(key)
    return checked


def check_calibration_bins(results_dir, predictions, manifest, helpers):
    saved = pd.read_csv(results_dir / 'results/calibration_bins.csv')
    columns = ['bin', 'bin_lower', 'bin_upper', 'n_bin', 'mean_predicted_probability', 'observed_frequency']
    checked = 0
    for scenario, model in manifest['selected_models'].items():
        item = predictions[f'{scenario}_{model}']
        for label, class_index in manifest['config']['TARGET_MAP'].items():
            expected = saved[(saved['Scenario'] == scenario) & (saved['Model'] == model)
                             & (saved['calibration_status'] == 'calibrated') & (saved['severity_class'] == label)]
            actual = helpers['_calibration_bin_table'](
                (item['y_true'] == class_index).astype(int), item['y_prob'][:, class_index],
                n_bins=manifest['config']['CALIBRATION_N_BINS'], strategy=manifest['config']['CALIBRATION_BINNING'])
            expected = expected.sort_values('bin')[columns].to_numpy(dtype=float)
            actual = actual.sort_values('bin')[columns].to_numpy(dtype=float)
            require(actual.shape == expected.shape and np.allclose(actual, expected, atol=1e-12, rtol=1e-12),
                    f'Calibration bins differ: {scenario}/{label}')
            checked += actual.size
    return checked


def check_saved_shap(results_dir, manifest, helpers):
    checks = 0
    maximum_error = 0.0
    for scenario in manifest['selected_models']:
        path = results_dir / f'per_disease_explanations/severity_shap_{scenario}.npz'
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as archive:
            matrix = archive['shap'].astype(float)
            names = archive['feature_names'].astype(str)
        feature_table = pd.read_csv(results_dir / f'results/severity_shap_features_{scenario}.csv').set_index('feature')
        actual = np.column_stack([np.abs(matrix).mean(axis=0), matrix.mean(axis=0)])
        expected = feature_table.loc[names, ['mean_abs_shap', 'mean_signed_shap']].to_numpy()
        require(np.allclose(actual, expected, rtol=1e-5, atol=1e-7), f'Saved feature SHAP summaries differ: {scenario}')
        maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected))))
        checks += actual.size
        table = pd.read_csv(results_dir / f'results/severity_shap_groups_{scenario}.csv').set_index('group')
        groups = np.asarray([helpers['severity_shap_group'](name) for name in names])
        for group in np.unique(groups):
            signed = matrix[:, groups == group].sum(axis=1)
            actual_group = np.asarray([np.abs(signed).mean(), signed.mean()])
            expected_group = table.loc[group, ['mean_abs_group_shap', 'mean_signed_group_shap']].to_numpy(dtype=float)
            require(np.allclose(actual_group, expected_group, rtol=1e-5, atol=1e-7), f'Saved group SHAP summaries differ: {scenario}/{group}')
            maximum_error = max(maximum_error, float(np.max(np.abs(actual_group - expected_group))))
            checks += 2
    return {'numeric_summary_checks': checks, 'maximum_absolute_error': maximum_error,
            'tolerance': 'rtol=1e-5, atol=1e-7 because saved SHAP matrices are float32',
            'scope': 'Summary replay from saved SHAP arrays; no SHAP estimator was run.'}


def verify(results_dir, report_path=None, check_file_fingerprints=False):
    results_dir = Path(results_dir).resolve()
    manifest = json.loads((results_dir / 'pipeline_manifest.json').read_text())
    source = verify_source(PACKAGE_DIR)
    require(source['status'] == 'passed', 'Source preservation failed')
    config_checks = check_config(manifest)
    helpers = source_functions(['compute_all_metrics', '_select_candidate_lexicographically',
                                '_calibration_bin_table', 'severity_shap_group'], manifest['config'])
    predictions, prediction_source = load_predictions(results_dir)
    metric_rows, scenarios = compare_metrics(predictions, pd.read_csv(results_dir / 'results/all_results.csv'), helpers)
    selected = check_model_selection(pd.read_csv(results_dir / 'results/model_selection_validation.csv'), manifest, helpers)
    split_results = {}
    for mode in ['primary', 'disease_disjoint', 'gene_and_disease_disjoint']:
        frame = pd.read_csv(results_dir / f'results/split_identifiers_{mode}.csv')
        check_split_frame(frame, mode)
        split_results[mode] = {part: len(block) for part, block in frame.groupby('partition')}
    split = pd.read_csv(results_dir / 'results/split_identifiers_primary.csv')
    require(len(split) == manifest['dataset']['n_pairs_total'], 'Dataset/split pair count mismatch')
    require(split['gene_symbol'].nunique() == manifest['dataset']['n_genes_total'], 'Dataset/split gene count mismatch')
    n_test = (split['partition'] == 'test').sum()
    first_y = next(iter(predictions.values()))['y_true']
    for key, item in predictions.items():
        require(len(item['y_true']) == n_test, f'Test size mismatch: {key}')
        require(np.array_equal(item['y_true'], first_y), f'Test target order mismatch: {key}')
    summary = pd.read_csv(results_dir / 'results/split_summary.csv').set_index('partition')
    test_counts = np.bincount(first_y.astype(int), minlength=manifest['config']['NUM_CLASSES'])
    require(np.array_equal(test_counts, summary.loc['test', [f'n_class_{i}' for i in range(4)]].to_numpy(dtype=int)),
            'Test class counts differ from split summary')
    publication = pd.read_csv(results_dir / 'results/publication_table.csv').set_index('Scenario')
    aliases = {'MCC':'mcc', 'QWK':'qwk', 'BalAcc':'balanced_accuracy', 'AUC_ROC':'auc_roc',
               'F1_macro':'f1_macro', 'RPS':'ranked_probability_score', 'Multiclass_Brier':'multiclass_brier'}
    table_comparisons = 0
    for row in pd.read_csv(results_dir / 'results/all_results.csv').to_dict('records'):
        for display, key in aliases.items():
            require(np.isclose(row[key], publication.loc[row['Scenario'], display], rtol=1e-12, atol=1e-12),
                    f'Publication table differs: {row["Scenario"]}/{key}')
            table_comparisons += 1
    bootstrap = pd.read_csv(results_dir / 'results/bootstrap_cis.csv').set_index('Scenario')
    ci_comparisons = 0
    for scenario in selected:
        for published, stored in [('MCC', 'mcc'), ('QWK', 'qwk'), ('RPS', 'ranked_probability_score')]:
            for bound in ['low', 'high']:
                require(np.isclose(publication.loc[scenario, f'{published}_CI_{bound}'],
                                   bootstrap.loc[scenario, f'{stored}_ci_{bound}'], rtol=1e-12, atol=1e-12),
                        f'Stored confidence intervals differ: {scenario}/{published}/{bound}')
                ci_comparisons += 1
    calibration_checks = check_calibration_bins(results_dir, predictions, manifest, helpers)
    shap_checks = check_saved_shap(results_dir, manifest, helpers)
    features = manifest['selected_features']
    hpo = {scenario: {x for x in columns if x.startswith('hpo_')}
           for scenario, columns in features.items() if scenario in {'S2_Mech_Pheno','S3_Phenotype'}}
    require(hpo['S2_Mech_Pheno'] == hpo['S3_Phenotype'], 'Shared HPO feature sets differ')
    completed = json.loads((results_dir / 'RUN_COMPLETE.json').read_text())
    require(completed['status'] == 'complete', 'Reference run is incomplete')
    validation = json.loads((results_dir / 'required_output_validation.json').read_text())
    require(validation['status'] == 'passed', 'Original required-output validation failed')
    file_comparisons = 0
    if check_file_fingerprints:
        snapshot = json.loads((REFERENCE_DIR / 'results_fingerprints.json').read_text())
        for name, expected in snapshot.items():
            p = results_dir / name
            require(p.is_file(), f'Missing reference output: {name}')
            require(sha256_file(p) == expected['sha256'], f'Reference output changed: {name}')
            file_comparisons += 1
    permutation = pd.read_csv(results_dir / 'results/permutation_test.csv')
    count = manifest['config']['N_PERMUTATIONS']
    require(permutation['Scenario'].is_unique and set(permutation['Scenario']) == set(selected),
            'Expected one permutation result per selected scenario')
    require(permutation['n_permutations_requested'].eq(count).all()
            and permutation['n_permutations_successful'].eq(count).all()
            and permutation['n_permutations_failed'].eq(0).all(),
            'Saved permutation counts differ from the completed run configuration')
    require(permutation['Observed_Model'].eq(permutation['Scenario'].map(selected)).all(),
            'Observed permutation models differ from the validation-selected models')
    empirical = permutation['empirical_p_value'].to_numpy(dtype=float)
    adjusted = permutation['bh_adjusted_p_value'].to_numpy(dtype=float)
    require(np.allclose(empirical * (count + 1), np.rint(empirical * (count + 1)), rtol=0, atol=1e-10)
            and np.all(empirical >= 1 / (count + 1) - 1e-12) and np.all(empirical <= 1),
            'Saved empirical P values do not match the permutation count')
    bh = source_functions(['_benjamini_hochberg'], manifest['config'])['_benjamini_hochberg']
    require(np.allclose(adjusted, bh(empirical), rtol=1e-12, atol=1e-12),
            'Saved adjusted P values differ from Benjamini–Hochberg correction')
    report = {
        'status': 'passed', 'scope': 'Source preservation and saved-result replay; no model was trained or refitted.',
        'prediction_source': prediction_source, 'source_preservation': source,
        'scientific_configurations_matched': len(config_checks),
        'metric_checks': len(metric_rows), 'maximum_metric_absolute_error': max(x['absolute_error'] or 0 for x in metric_rows),
        'scenarios': scenarios, 'selected_models': selected, 'split_checks': split_results,
        'publication_table_comparisons': table_comparisons,
        'stored_confidence_interval_comparisons': ci_comparisons,
        'calibration_bin_numeric_checks': calibration_checks,
        'saved_shap_summary_replay': shap_checks,
        'shared_direct_hpo_features': len(hpo['S2_Mech_Pheno']),
        'reference_files_unchanged': file_comparisons,
        'permutation_adjusted_p_values': dict(zip(permutation['Scenario'], adjusted)),
        'permutations_recorded': sorted(permutation['n_permutations_successful'].astype(int).unique().tolist()),
        'environment': {name: importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn']},
        'limitations': ['Training and SHAP estimation were not rerun.',
                        'Exact full reproduction requires the original inputs and recorded dependencies.'],
    }
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        pd.DataFrame(metric_rows).to_csv(report_path.with_suffix('.metrics.csv'), index=False)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=REFERENCE_DIR)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--check-file-fingerprints', action='store_true',
                        help='Require every original output byte to match the supplied reference run. Use for the original folder only.')
    args = parser.parse_args(argv)
    try:
        report = verify(args.results_dir, args.report, args.check_file_fingerprints)
    except (CheckFailure, OSError, KeyError, ValueError) as exc:
        print(f'FAILED: {exc}')
        return 1
    print(json.dumps({key: report[key] for key in ['status','metric_checks','maximum_metric_absolute_error',
        'scientific_configurations_matched','publication_table_comparisons','reference_files_unchanged','scenarios']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
