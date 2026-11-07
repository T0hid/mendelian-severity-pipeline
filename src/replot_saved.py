"""Render publication plots from saved tables and arrays."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd


def _saved_model_results(reference, publication=None):
    from verify_results import load_predictions

    predictions, _ = load_predictions(reference)
    validation = pd.read_csv(reference / 'results/model_selection_validation.csv')
    metrics = pd.read_csv(reference / 'results/all_results.csv')
    candidate_metrics = None
    if publication is not None:
        candidate_metrics = pd.read_csv(Path(publication) / 'benchmark/figures/publication/figure_data/figure_2_candidate_validation_metrics.csv')
    results, selected = {}, {}
    for row in validation.to_dict('records'):
        scenario, model = row['Scenario'], row['Model']
        entry = {'val_metrics': {name: row['validation_' + name]
                                for name in ['qwk', 'mcc', 'balanced_accuracy']},
                 'selected_on_validation': bool(row['selected_on_validation'])}
        if candidate_metrics is not None:
            match = candidate_metrics[(candidate_metrics['Scenario'] == scenario) & (candidate_metrics['Model'] == model)]
            if len(match) != 1:
                raise ValueError('Expected one saved validation-metric row per candidate.')
            entry['val_metrics'] = {name: match.iloc[0][name] for name in [
                'qwk', 'mcc', 'balanced_accuracy', 'f1_macro', 'f1', 'auc_roc',
                'ranked_probability_score', 'multiclass_brier']}
        results.setdefault(scenario, {'all': {}})['all'][model] = entry
        if entry['selected_on_validation']:
            selected[scenario] = model
            key = next((key for key in predictions if key in {
                f'{scenario}_{model}', f'{scenario}__{model}', f'{scenario}::{model}'}), None)
            if key is None:
                raise ValueError(f'Saved predictions missing for {scenario}/{model}.')
            prediction = predictions[key]
            entry['y_true'] = prediction['y_true']
            entry['predictions'] = {'y_pred': prediction['y_pred'], 'y_prob': prediction['y_prob']}
            selected_metrics = metrics[(metrics['Scenario'] == scenario) & (metrics['Model'] == model)]
            if len(selected_metrics) != 1:
                raise ValueError('Expected one test metric row per selected model.')
            entry['metrics'] = selected_metrics.iloc[0].to_dict()
    names = {'S1_Mechanistic': 'S1_Gene_Level', 'S2_Mech_Pheno': 'S2_Combined'}
    results = {names.get(key, key): value for key, value in results.items()}
    selected = {names.get(key, key): value for key, value in selected.items()}
    return results, {'selected_models': selected}


def _render_cohort(publication, destination, native):
    """Replay descriptive counts already exported by the original figure code."""
    import matplotlib.pyplot as plt

    folder = publication / 'benchmark/figures/publication/figure_data'
    read = lambda name: pd.read_csv(folder / name)
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 5.4))
    frame = read('figure_1a_partition_composition.csv')
    frame.pivot(index='partition', columns='severity', values='proportion').reindex(
        index=['Train', 'Validation', 'Test'], columns=native.CLASS_NAMES).plot.barh(
            stacked=True, ax=axes[0, 0], legend=False, width=0.65)
    axes[0, 0].set(xlabel='Proportion', ylabel='', title='a  Partition composition')
    frame = read('figure_1b_gene_pair_ccdf.csv')
    axes[0, 1].loglog(frame['minimum_pairs'], frame['n_genes'], marker='o', markersize=3)
    axes[0, 1].set(xlabel='Minimum pairs per gene', ylabel='Number of genes', title='b  Pairs per gene')
    frame = read('figure_1c_inheritance_prevalence.csv').pivot(index='inheritance', columns='severity', values='prevalence').reindex(columns=native.CLASS_NAMES)
    axes[0, 2].imshow(frame.to_numpy(), cmap='Blues', aspect='auto', vmin=0, vmax=1)
    axes[0, 2].set_xticks(range(4), native.CLASS_NAMES, rotation=45, ha='right')
    axes[0, 2].set_yticks(range(len(frame)), frame.index)
    axes[0, 2].set_title('c  Inheritance prevalence')
    frame = read('figure_1d_disease_pair_ccdf.csv')
    axes[1, 0].loglog(frame['minimum_pairs'], frame['n_diseases'], marker='o', markersize=3)
    axes[1, 0].set(xlabel='Minimum pairs per disease', ylabel='Number of diseases', title='d  Pairs per disease')
    frame = read('figure_1e_scenario_feature_families.csv')
    frame.pivot_table(index='family_label', columns='Scenario', values='n_features', aggfunc='sum').fillna(0).plot.barh(ax=axes[1, 1], legend=False)
    axes[1, 1].set(xscale='symlog', xlabel='Candidate features', ylabel='', title='e  Feature families')
    axes[1, 2].axis('off')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    severity_legend = axes[1, 2].legend(handles, labels, title='Severity', loc='upper left',
                                      fontsize=6, title_fontsize=7, frameon=False)
    axes[1, 2].add_artist(severity_legend)
    handles, labels = axes[1, 1].get_legend_handles_labels()
    axes[1, 2].legend(handles, [native.SCENARIO_SHORT.get(label, label) for label in labels],
                      title='Feature set', loc='lower left', fontsize=6, title_fontsize=7, frameon=False)
    fig.tight_layout(pad=1.4)
    native._save(fig, 'replay_extended_data_fig_1', str(destination))


def render_saved_figures(reference, publication, destination):
    """Use original plotting functions with their saved numeric inputs.

    The PCA provider reads the original projected coordinates. It does not
    refit PCA, preprocessing, SHAP or any predictive model.
    """
    import matplotlib
    matplotlib.use('Agg')
    import publication_figures as native

    reference, publication, destination = map(Path, (reference, publication, destination))
    destination.mkdir(parents=True, exist_ok=True)
    panel_data = destination / 'panel_data'
    panel_data.mkdir(exist_ok=True)
    import shutil
    from tempfile import TemporaryDirectory
    temporary = TemporaryDirectory(prefix='severity-figure-inputs-')
    source = Path(temporary.name) / 'benchmark'
    shutil.copytree(publication / 'benchmark', source)
    names = {'S1_Mechanistic': 'S1_Gene_Level', 'S2_Mech_Pheno': 'S2_Combined',
             'feat_gnn_': 'feat_annot_', 'has_gnn_': 'has_annot_'}
    for path in list(source.rglob('*.csv')):
        text = path.read_text(encoding='utf-8-sig')
        filename = path.name
        for old, new in names.items():
            text = text.replace(old, new)
            filename = filename.replace(old, new)
        target = path.with_name(filename)
        target.write_text(text, encoding='utf-8')
        if target != path:
            path.unlink()
    results, outputs = _saved_model_results(reference, publication)
    native._setup_rcparams()
    native.figure_2_model_comparison(results, outputs, str(source), str(destination), str(panel_data))
    native.figure_3_prediction_calibration(results, outputs, str(source), str(destination), str(panel_data))
    native.figure_4_robustness_composite(results, outputs, SimpleNamespace(), {},
                                      str(source), str(destination), str(panel_data))

    pca = pd.read_csv(source / 'figures/publication/figure_data/figure_5e_s3_true_predicted_pca.csv')
    variance = pca[['PC1_explained_variance_ratio', 'PC2_explained_variance_ratio']].iloc[0].to_numpy(dtype=float)
    def saved_pca(*args, **kwargs):
        return pca.copy(), variance.copy(), outputs['selected_models']['S3_Phenotype'], None
    with patch.object(native, '_build_s3_locked_test_pca', saved_pca):
        native.figure_5_interpretation_biology(results, outputs, SimpleNamespace(), SimpleNamespace(), {},
                                              str(source), str(destination), str(panel_data),
                                              native.FeatureLabeler(str(source)), random_seed=42)
    _render_cohort(Path(temporary.name), destination, native)
    temporary.cleanup()
    files = [p.relative_to(destination).as_posix() for p in sorted(destination.rglob('*')) if p.is_file()]
    expected = ['figure_2_model_comparison', 'figure_3_prediction_and_calibration',
                'extended_data_fig_2_robustness_and_audits', 'figure_4_interpretation_and_biology',
                'replay_extended_data_fig_1']
    for name in expected:
        if not (destination / (name + '.pdf')).is_file():
            raise ValueError('Expected replay figure was not rendered: ' + name)
    return {'files': files, 'figures_rendered': len(expected), 'model_training_performed': False,
            'pca_refitted': False, 'shap_recomputed': False,
            'cohort_overview_layout': 'Cohort overview from saved descriptive tables.',
            'clinical_figure': 'Original exports and public source tables included; family-level replot uses the governed workflow.'}
