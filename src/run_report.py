"""Write a report from completed analysis tables."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


TABLES = (
    ('Dataset partitions', 'split_summary.csv'),
    ('Validation model selection', 'model_selection_validation.csv'),
    ('Shared direct-HPO budget', 'shared_hpo_budget_training_cv_selection.csv'),
    ('Locked-test performance and uncertainty', 'publication_table.csv'),
    ('Naive baselines', 'naive_baselines.csv'),
    ('Ordinal errors', 'error_decomposition.csv'),
    ('Calibration', 'calibration_metrics.csv'),
    ('Paired scenario contrasts', 'paired_scenario_contrasts.csv'),
    ('Gene-level companion evaluation', 'gene_level_evaluation.csv'),
    ('Additional generalisation', 'generalisation_evaluations.csv'),
    ('Conditional permutation analysis', 'permutation_test.csv'),
    ('Split sensitivity', 'split_sensitivity_fixed_configuration_summary.csv'),
    ('Minimum-MI sensitivity', 'mi_threshold_sensitivity.csv'),
    ('HPO-budget sensitivity', 'budget_sensitivity.csv'),
    ('Imbalance sensitivity', 'imbalance_sensitivity.csv'),
    ('Severity-HPO sensitivity', 'severity_hpo_term_sensitivity.csv'),
    ('SHAP diagnostics', 'severity_shap_checks.csv'),
)


def _markdown_table(path: Path) -> str:
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return 'No rows were recorded.'
    def cell(value):
        return value.replace('|', '\\|').replace('\n', ' ')
    header, *body = rows
    result = ['| ' + ' | '.join(map(cell, header)) + ' |',
              '| ' + ' | '.join(['---'] * len(header)) + ' |']
    result.extend('| ' + ' | '.join(map(cell, row)) + ' |' for row in body)
    return '\n'.join(result)


def generate_manuscript_report(all_results, container, pub_outputs, *, output_dir, pipeline_version,
                               config=None, report_path=None):
    """Write an evidence-linked report from completed tables and the active Config."""
    if config is None:
        from pipeline import Config
    else:
        Config = config

    output_dir = Path(output_dir)
    sections = [
        '# Genetic disease severity analysis report',
        f'Reference analysis version: {pipeline_version}. Generated: {datetime.now().isoformat(timespec="seconds")}.',
        'This report transcribes completed result tables. Candidate identity was selected on validation; '
        'stored test performance did not select the candidate.',
        '## Label provenance',
        Config.TARGET_LABEL_PROVENANCE,
        '## Recorded analysis settings',
        f'- Ordered classes: {", ".join(Config.TARGET_MAP)}.\n'
        f'- Model-selection rule: {Config.MODEL_SELECTION_RULE}.\n'
        f'- HPO-budget grid: {Config.HPO_BUDGET_CANDIDATES}; '
        f'{Config.HPO_BUDGET_ALL_ELIGIBLE} denotes all eligible terms.\n'
        f'- Missing-value indicators enabled: {Config.ADD_MISSING_INDICATORS}.\n'
        f'- Source-availability indicators enabled: {Config.ADD_SOURCE_AVAILABILITY_INDICATORS}.\n'
        f'- Imbalance strategy: {Config.IMBALANCE_STRATEGY}.\n'
        f'- Configured permutations: {Config.N_PERMUTATIONS}.\n'
        f'- Gene-cluster bootstrap replicates: {Config.N_BOOTSTRAP}.\n'
        f'- Excluded text sources: {Config.TEXT_EXCLUDED_SOURCE_COLUMNS}.',
        'Scenario membership and provenance are recorded in results/feature_provenance_table.csv; '
        'selected features are recorded in pipeline_manifest.json. These records define the actual feature sets.',
    ]
    if Config.N_PERMUTATIONS < 99:
        sections.extend(['## Permutation limitation',
            f'This run used {Config.N_PERMUTATIONS} permutation(s). '
            'Do not describe it as a 99-permutation analysis. Reconcile the Methods and this run before submission.'])
    for title, filename in TABLES:
        path = output_dir / 'results' / filename
        sections.extend([f'## {title}', f'Source: results/{filename}.',
                         _markdown_table(path) if path.is_file() else 'This table was not produced by this run.'])
    target = Path(report_path) if report_path is not None else output_dir / 'MANUSCRIPT_REPORT.md'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('\n\n'.join(sections) + '\n', encoding='utf-8')
    return str(target)


if __name__ == '__main__':
    import argparse
    import json
    from types import SimpleNamespace

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.results_dir / 'pipeline_manifest.json').read_text())
    result = generate_manuscript_report(None, None, None, output_dir=args.results_dir,
        pipeline_version=manifest['pipeline_version'], config=SimpleNamespace(**manifest['config']),
        report_path=args.output)
    print(result)
