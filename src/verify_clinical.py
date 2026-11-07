"""Replay the disease-level scorer and prenatal summaries without fitting models.

Default review uses disease-level predictions and aggregate prenatal tables. Optional
local checks use governed family data in memory and export only verification counts.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics

from runtime_paths import reference_directory

ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = reference_directory(ROOT)
SCORER_FILE = 'phenotype_scorer.py'
PRENATAL_FILE = 'prenatal_analysis.py'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def semantic_hash(source):
    tree = ast.parse(source)
    if (tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)):
        tree.body.pop(0)
    # Empty generic-parameter fields differ between Python 3.11 and 3.12.
    for item in ast.walk(tree):
        if hasattr(item, 'type_params') and not item.type_params:
            del item.type_params
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'Could not load study module')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_clinical_source():
    expected = json.loads((REFERENCE_DIR / 'clinical_source_fingerprints.json').read_text(encoding='utf-8'))
    cleanup_path = REFERENCE_DIR.parent / 'cleanup_source_changes.json'
    cleanup = json.loads(cleanup_path.read_text()) if cleanup_path.is_file() else {}
    names = {'phenotype_severity_scorer_v3_3_monotone_ontology.py': SCORER_FILE,
             'prenatal_severity_analysis_v3_multichannel.py': PRENATAL_FILE}
    report = {}
    for historical_name, item in expected.items():
        name = names.get(historical_name, historical_name)
        source = (ROOT / name).read_text(encoding='utf-8-sig')
        require(semantic_hash(source) == cleanup.get('clinical_modules', {}).get(name, item['semantic_module_sha256']), f'Computational source changed: {name}')
        report[name] = {'status': 'passed', 'functions_retained': cleanup.get('clinical_counts', {}).get(name, item)['functions'],
                        'classes_retained': cleanup.get('clinical_counts', {}).get(name, item)['classes']}
    return report


def numeric_match(actual, saved, label, tolerance=1e-10):
    require(np.allclose(actual, saved, rtol=tolerance, atol=tolerance, equal_nan=True), f'Numerical mismatch: {label}')
    a, b = np.asarray(actual, dtype=float), np.asarray(saved, dtype=float)
    differences = np.abs(a - b)
    return float(np.nanmax(differences)) if differences.size and np.isfinite(differences).any() else 0.0


def verify_scorer_results(folder, scorer):
    metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    frame = pd.read_csv(folder / 'held_out_test_predictions.csv')
    y = frame['observed'].to_numpy(dtype=int)
    probability = frame[['p_mild', 'p_moderate', 'p_severe', 'p_profound']].to_numpy()
    require(len(frame) == metadata['split_audit']['test_diseases'], 'Scorer held-out size mismatch')
    require(np.allclose(probability.sum(axis=1), 1), 'Scorer probability sums differ from one')
    require(not frame['mondo_id'].duplicated().any(), 'Repeated held-out disease')

    class SavedCombinedPrediction:
        def expected_severity(self, X, calibrated=True):
            return frame['combined_expected'].to_numpy()
        def class_proba(self, X, calibrated=True):
            return probability
        def cumulative_proba(self, X, calibrated=True):
            return 1.0 - np.cumsum(probability, axis=1)[:, :-1]

    # This adapter exposes saved arrays to the unchanged evaluate_model function.
    # It never loads or fits an estimator and does not reconstruct held-out models.
    combined = scorer.evaluate_model(SavedCombinedPrediction(), None, y, True)
    count, maximum_error = 0, 0.0
    for key, actual in combined.items():
        maximum_error = max(maximum_error, numeric_match(actual, metadata['held_out_combined'][key], f'combined/{key}'))
        count += 1
    expected = frame['phenotype_expected'].to_numpy()
    predicted = np.clip(np.rint(expected), 0, 3).astype(int)
    mild = expected[y == 0]
    available = {
        'qwk': metrics.cohen_kappa_score(y, predicted, weights='quadratic'),
        'mcc': metrics.matthews_corrcoef(y, predicted),
        'balanced_accuracy': metrics.balanced_accuracy_score(y, predicted),
        'mae_expected': metrics.mean_absolute_error(y, expected),
        'mild_mae_expected': np.mean(mild), 'mild_median_expected': np.median(mild),
        'mild_p95_expected': np.percentile(mild, 95),
        'mild_class_recall': np.mean(predicted[y == 0] == 0), 'n': len(y),
    }
    for key, actual in available.items():
        maximum_error = max(maximum_error, numeric_match(actual, metadata['held_out_phenotype'][key], f'phenotype/{key}'))
        count += 1
    splits = pd.read_csv(folder / 'disease_split_membership.csv')
    require(not splits['mondo_id'].duplicated().any(), 'Disease occurs in multiple scorer partitions')
    require(len(splits) == metadata['n_diseases'], 'Scorer disease total differs')
    require(set(splits.loc[splits['partition'] == 'test', 'mondo_id']) == set(frame['mondo_id']),
            'Held-out predictions do not match scorer split')
    selection = pd.read_csv(folder / 'model_selection_results.csv')
    valid = selection[selection['complete'].astype(str).str.lower().eq('true')].sort_values(
        ['selection_score', 'mean_qwk', 'mean_mild_mae_expected', 'mean_ordinal_brier'],
        ascending=[False, False, True, True])
    require(len(valid) == metadata['model_selection_complete_configs'], 'Scorer valid configuration count differs')
    winner = valid.iloc[0]
    for name, value in metadata['selected_config'].items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric_match(winner[name], value, f'selected_config/{name}')
        else:
            require(str(winner[name]).lower() == str(value).lower(), f'Selected scorer setting differs: {name}')
    original_argv = sys.argv
    try:
        sys.argv = [SCORER_FILE]
        default_args = scorer.parse_args()
        candidate_labels = [item.label for item in scorer._build_candidate_configs(default_args)]
    finally:
        sys.argv = original_argv
    require(candidate_labels == selection['config_label'].tolist(), 'Default scorer candidate configurations differ from the saved selection run')
    return {'status': 'passed', 'diseases': len(splits), 'held_out_diseases': len(y),
            'metric_comparisons': count, 'maximum_absolute_error': maximum_error,
            'default_candidate_configurations_matched': len(candidate_labels),
            'held_out_probability_scope': 'Saved class probabilities describe the combined channel. Phenotype-channel probability metrics cannot be replayed from this CSV.',
            'training': 'not rerun'}


def verify_prenatal_aggregate(folder):
    stored = json.loads((folder / 'statistics.json').read_text(encoding='utf-8'))
    require(stored['score_channel'] == 'phenotype', 'Saved prenatal channel is not phenotype')
    counts = stored['audit']['outcome_counts_analyzed_4group']
    means = pd.read_csv(folder / 'mean_by_outcome_4group.csv')
    require(dict(zip(means['outcome'], means['n'])) == counts, 'Prenatal aggregate counts differ')
    require(int(means['n'].sum()) == stored['audit']['n_analyzed_4group'], 'Prenatal analysis total differs')
    comparisons = 0
    for groups in ['3group', '4group']:
        rows = pd.read_csv(folder / f'roc_contrasts_{groups}.csv')
        expected = {row['name']: row for row in stored[f'roc_contrasts_{groups}']}
        for row in rows.to_dict('records'):
            for field in ['auc', 'n_pos', 'n_neg']:
                numeric_match(row[field], expected[row['contrast']][field], f'prenatal/{groups}/{field}')
                comparisons += 1
        odds = stored[f'proportional_odds_{groups}']
        numeric_match(np.exp(odds['beta']), odds['or_per_sd'], f'prenatal/{groups}/odds')
        numeric_match(np.exp([odds['beta'] - 1.96 * odds['se'], odds['beta'] + 1.96 * odds['se']]),
                      odds['ci95'], f'prenatal/{groups}/Wald_interval')
        comparisons += 3
    return {'status': 'passed', 'family_profiles': stored['audit']['n_family_profiles'],
            'analysed_families': stored['audit']['n_analyzed_4group'], 'score_channel': stored['score_channel'],
            'aggregate_numeric_checks': comparisons,
            'individual_data_in_public_snapshot': False,
            'current_source_outputs_missing_from_saved_run': [
                name for name in ['term_count_baseline_rank_4group.csv', 'term_count_baseline_odds_4group.csv']
                if not (folder / name).exists()],
            'scope': 'Aggregate consistency and stored odds/interval arithmetic; family-bootstrap intervals and regression models were not refitted.'}


def load_scorer(model_path, prenatal, scorer_code=None):
    module = prenatal._load_scorer_module(str(scorer_code or ROOT / SCORER_FILE))
    with Path(model_path).open('rb') as handle:
        model = prenatal._ScorerCompatUnpickler(handle, module).load()
    prenatal._upgrade_loaded_scorer_state(model, module)
    prenatal._validate_model_compatibility(model)
    return model


def synthetic_predictions(model):
    examples = json.loads((ROOT / 'examples/synthetic_profiles.json').read_text(encoding='utf-8'))
    output = []
    for example in examples:
        result = model.score_profile(example['present_terms'], absent_hpo_terms=example['absent_terms'])
        phenotype = result.get('phenotype_only_expected_severity_0_3')
        combined = result.get('combined_expected_severity_0_3')
        output.append({'example_id': example['example_id'], 'status': result['status'],
                       'phenotype_score_0_1': phenotype / 3.0 if phenotype is not None else None,
                       'combined_score_0_1': combined / 3.0 if combined is not None else None,
                       'phenotype_probabilities': result.get('phenotype_only_class_probabilities'),
                       'combined_probabilities': result.get('class_probabilities')})
    return output


def verify_model(model_path, prenatal):
    expected_hash = json.loads((REFERENCE_DIR / 'clinical_results_fingerprints.json').read_text(encoding='utf-8'))['scorer']['model.pkl']['sha256']
    require(sha256_file(model_path) == expected_hash, 'Deployment model differs from the supplied reference model')
    model = load_scorer(model_path, prenatal)
    metadata = json.loads((REFERENCE_DIR / 'scorer/metadata.json').read_text(encoding='utf-8'))
    require(len(model.builder.feature_names) == metadata['feature_counts']['combined_total'], 'Model feature count differs')
    require(dataclasses.asdict(model.config) == metadata['selected_config'], 'Model configuration differs from metadata')
    actual = synthetic_predictions(model)
    expected = json.loads((REFERENCE_DIR / 'synthetic_profile_predictions.json').read_text(encoding='utf-8'))
    def compare(a, b):
        if isinstance(b, dict):
            require(isinstance(a, dict) and set(a) == set(b), 'Synthetic payload keys differ')
            for key in b:
                compare(a[key], b[key])
        elif isinstance(b, list):
            require(isinstance(a, list) and len(a) == len(b), 'Synthetic payload length differs')
            for left, right in zip(a, b):
                compare(left, right)
        elif isinstance(b, (int, float)) and not isinstance(b, bool):
            numeric_match(a, b, 'synthetic-profile-model-replay')
        else:
            require(a == b, 'Synthetic payload label/status differs')
    compare(actual, expected)
    return {'status': 'passed', 'synthetic_profiles_checked': len(actual), 'model_sha256': expected_hash}


def check_original_files(scorer_dir=None, prenatal_dir=None):
    expected = json.loads((REFERENCE_DIR / 'clinical_results_fingerprints.json').read_text(encoding='utf-8'))
    count = 0
    for label, folder in [('scorer', scorer_dir), ('prenatal', prenatal_dir)]:
        if folder is None:
            continue
        for name, item in expected[label].items():
            require(sha256_file(Path(folder) / name) == item['sha256'], f'Original {label} file changed: {name}')
            count += 1
    return count


def verify_local_prenatal(results_dir, prenatal, cohort=None, model_path=None):
    """Use governed data locally; return aggregate check counts, never family records."""
    stored = json.loads((results_dir / 'statistics.json').read_text(encoding='utf-8'))
    families = pd.read_csv(results_dir / 'severity_by_family.csv')
    require(len(families) == stored['audit']['n_family_profiles'], 'Family total differs')
    maximum_error, comparisons = 0.0, 0
    for groups, outcome, labels in [('3group', 'outcome_3level', prenatal.OUTCOME_LABELS),
                                    ('4group', 'outcome_4level', prenatal.OUTCOME4_LABELS)]:
        included = families[families[f'included_{groups}_analysis'].astype(str).str.lower().eq('true')].copy()
        require(len(included) == stored['audit'][f'n_analyzed_{groups}'], 'Analysis inclusion count differs')
        actual = prenatal.mean_ci_by_outcome(included, outcome, labels)
        expected = pd.read_csv(results_dir / f'mean_by_outcome_{groups}.csv')
        columns = ['level', 'n', 'mean', 'ci_low', 'ci_high']
        maximum_error = max(maximum_error, numeric_match(actual[columns].to_numpy(), expected[columns].to_numpy(), f'family-means/{groups}'))
        comparisons += actual[columns].size
        for metric, value in [('spearman_rho', stats.spearmanr(included['severity_score'], included[outcome]).statistic),
                              ('kendall_tau_b', stats.kendalltau(included['severity_score'], included[outcome]).statistic)]:
            maximum_error = max(maximum_error, numeric_match(value, stored[f'correlation_{groups}'][metric], f'family-rank/{groups}/{metric}'))
            comparisons += 1
        fn = prenatal.clinical_contrasts_3group if groups == '3group' else prenatal.clinical_contrasts_4group
        for actual, expected in zip(fn(included), stored[f'roc_contrasts_{groups}']):
            maximum_error = max(maximum_error, numeric_match(actual['auc'], expected['auc'], f'family-ROC/{groups}'))
            comparisons += 1
    replay = {'performed': False}
    if cohort is not None or model_path is not None:
        require(cohort is not None and model_path is not None, 'Both cohort and model are required to replay family scoring')
        profiles, audit = prenatal.load_cohort(str(cohort), [], [])
        require(audit['n_family_profiles'] == len(families), 'Reconstructed family count differs')
        scored, _ = prenatal.score_families(profiles, str(model_path), str(ROOT / SCORER_FILE), score_channel='phenotype')
        require(not scored['family_id'].duplicated().any(), 'Repeated replay family')
        scored = scored.set_index('family_id').loc[families['family_id']]
        columns = ['severity_score', 'severity_score_phenotype', 'severity_score_combined',
                   'expected_severity_0_3', 'phenotype_expected_severity_0_3', 'combined_expected_severity_0_3',
                   'class_probability_mild', 'class_probability_moderate', 'class_probability_severe', 'class_probability_profound',
                   'phenotype_probability_mild', 'phenotype_probability_moderate', 'phenotype_probability_severe', 'phenotype_probability_profound']
        maximum_error = max(maximum_error, numeric_match(scored[columns].to_numpy(dtype=float), families[columns].to_numpy(dtype=float), 'frozen-model-family-replay'))
        require(scored['score_status'].tolist() == families['score_status'].tolist(), 'Family scoring status differs')
        comparisons += len(families) * len(columns)
        replay = {'performed': True, 'family_profiles_scored': len(families), 'model_retrained': False}
    return {'status': 'passed', 'numeric_checks': comparisons, 'maximum_absolute_error': maximum_error,
            'frozen_scorer_replay': replay, 'individual_records_exported': 0}


def verify(scorer_dir=None, prenatal_dir=None, model_path=None, cohort=None, check_file_fingerprints=False):
    sources = verify_clinical_source()
    scorer = load_module(ROOT / SCORER_FILE, 'phenotype_severity_scorer_v3_3_monotone_ontology')
    prenatal = load_module(ROOT / PRENATAL_FILE, 'prenatal_severity_analysis_v3_multichannel')
    report = {'source': sources,
              'scorer': verify_scorer_results(Path(scorer_dir or REFERENCE_DIR / 'scorer'), scorer),
              'prenatal_aggregate': verify_prenatal_aggregate(Path(prenatal_dir or REFERENCE_DIR / 'prenatal')),
              'original_clinical_files_unchanged': check_original_files(scorer_dir, prenatal_dir) if check_file_fingerprints else 0}
    if model_path is not None:
        report['deployment_model'] = verify_model(model_path, prenatal)
    if prenatal_dir is not None:
        report['local_prenatal'] = verify_local_prenatal(Path(prenatal_dir), prenatal, cohort, model_path)
    report['status'] = 'passed'
    report['scope'] = 'Saved-result replay only. No model, bootstrap or permutation analysis was retrained or rerun.'
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scorer-results', type=Path)
    parser.add_argument('--prenatal-results', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--cohort', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--check-original-files', action='store_true')
    args = parser.parse_args(argv)
    if args.cohort and not args.prenatal_results:
        parser.error('--cohort replay requires --prenatal-results')
    result = verify(args.scorer_results, args.prenatal_results, args.model, args.cohort, args.check_original_files)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
