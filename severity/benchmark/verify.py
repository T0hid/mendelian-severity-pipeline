"""Check the published benchmark results from the saved models, without retraining.

1. The feature table is rebuilt from the training tables and the primary split is drawn again.
   The feature inventory (Supplementary Table 1) and the curation tables written while building
   it must be byte-identical to the saved ones, and every partition must match
   ``results/split_identifiers_primary.csv`` pair for pair.
2. Each validation-selected model bundle (preprocessing, selected features, model and
   calibrator) is applied to its test pairs. The probabilities must match the saved test
   predictions.
3. Test metrics are recomputed from these predictions and compared with Table 1
   (``results/publication_table.csv``).
4. Each bundle's model is also applied without its calibrator. The metrics before calibration
   are compared with the ``Uncalibrated_*`` columns of ``publication_table.csv``
   (Supplementary Table 5), and the confusion matrices before and after calibration with the
   saved Extended Data Fig. 3 data, when the run holds them.

5. The tables computed from stored predictions (the moderate/severe cut-off,
   ``moderate_severe_cutoff*.csv``; the comparison by number of recorded HPO terms,
   ``hpo_term_count_s2_vs_s3.csv``; metrics before and after calibration,
   ``raw_vs_calibrated.csv``; and the stricter splits, ``generalisation_evaluations.csv`` and
   ``generalisation_paired_contrasts.csv``) are recomputed from the saved test predictions and
   compared with the saved tables.

The report is written to ``verification.json`` and ``verification_metrics.csv``. The tables
behind Extended Data Fig. 3 are written to ``extended_data_fig_3_*.csv`` and the recomputed
tables of step 5 to ``recomputed/``.
"""

from __future__ import annotations

import importlib
import json
import os
import pickle
import pkgutil
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .annotation_terms import _add_training_text_features
from .calibration import _predict_uncalibrated_candidate
from .calibration_changes import (
    CHANGES_FILE,
    CONFUSION_FILE,
    TRANSITIONS_FILE,
    calibration_change_tables,
    confusion_counts_match,
)
from .config import RANDOM_SEED, Config
from .evaluation import (
    HPO_TERM_COUNT_FILE,
    compute_bootstrap_ci,
    hpo_term_count_comparison,
    moderate_severe_cutoff,
    recorded_hpo_term_counts,
    raw_vs_calibrated_table,
    summarise_generalisation_predictions,
)
from .features import assemble_full_dataframe
from .metrics import compute_all_metrics
from .models import _predict_ordinal_probabilities
from .partition import _split_development_train_validation, make_train_test_split

TABLE_1_COLUMNS = {
    "MCC": "mcc",
    "QWK": "qwk",
    "BalAcc": "balanced_accuracy",
    "AUC_ROC": "auc_roc",
    "F1_macro": "f1_macro",
    "RPS": "ranked_probability_score",
    "Multiclass_Brier": "multiclass_brier",
}
# Metrics before calibration, as saved in publication_table.csv (Supplementary Table 5).
UNCALIBRATED_COLUMNS = {
    "Uncalibrated_MCC": "mcc",
    "Uncalibrated_QWK": "qwk",
    "Uncalibrated_BalAcc": "balanced_accuracy",
    "Uncalibrated_RPS": "ranked_probability_score",
    "Uncalibrated_Multiclass_Brier": "multiclass_brier",
}
PROBABILITY_TOLERANCE = 1e-9
# Metrics are compared to 1e-5 (the paper reports three decimals). Probabilities can differ by
# about 1e-16 between operating systems, which can reorder near-tied pairs and move AUC by ~1e-6.
METRIC_TOLERANCE = 1e-5


class _BundleUnpickler(pickle.Unpickler):
    """Bundles saved by the original single-file script refer to classes in ``__main__``."""

    _modules = None

    def find_class(self, module: str, name: str):
        if module == "__main__":
            if _BundleUnpickler._modules is None:
                package = importlib.import_module(__package__)
                _BundleUnpickler._modules = [
                    importlib.import_module(f"{__package__}.{info.name}")
                    for info in pkgutil.iter_modules(package.__path__)
                    if info.name not in ("pipeline", "verify")
                ]
            for candidate in _BundleUnpickler._modules:
                value = vars(candidate).get(name)
                if isinstance(value, type) and value.__module__.startswith("severity"):
                    return value
            raise pickle.UnpicklingError(f"Unknown class in model bundle: {name}")
        return super().find_class(module, name)


def load_bundle(path: str) -> Dict[str, Any]:
    with open(path, "rb") as handle:
        return _BundleUnpickler(handle).load()


def bundle_probabilities(bundle: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Calibrated class probabilities, as used for the test evaluation."""
    if bundle["model_name"] == "OrdinalDecomp":
        return _predict_ordinal_probabilities(bundle.get("calibrated_models") or bundle["model"], X)
    model = bundle.get("calibrated_model") or bundle["model"]
    return model.predict_proba(X)


def _test_matrix(container, bundle: Dict[str, Any], test_idx: np.ndarray) -> np.ndarray:
    frame = container.full_df
    if bundle.get("text_vocabulary"):
        frame, _, _ = _add_training_text_features(
            container, np.arange(len(container.full_df), dtype=int), bundle["text_vocabulary"], {}
        )
    X_full = bundle["preprocessor"].transform(frame, test_idx)
    mask = np.asarray(bundle["sel_mask"], dtype=bool)
    if X_full.shape[1] != mask.size:
        raise ValueError(
            f"{bundle['scenario']}: {X_full.shape[1]} columns but a selection mask of {mask.size}."
        )
    return X_full[:, mask]


def _check_split(container, run_dir: str) -> Dict[str, Any]:
    saved = pd.read_csv(os.path.join(run_dir, "results", "split_identifiers_primary.csv"))
    train_idx, val_idx = _split_development_train_validation(
        container, container.train_idx, container.groups, RANDOM_SEED
    )
    metadata = container.metadata.reset_index(drop=True)
    checks = {}
    for partition, idx in (("train", train_idx), ("validation", val_idx), ("test", container.test_idx)):
        expected = saved.loc[saved["partition"] == partition]
        rebuilt = metadata.iloc[np.asarray(idx)][["gene_symbol", "mondo_id"]].reset_index(drop=True)
        same = (
            len(expected) == len(rebuilt)
            and np.array_equal(expected["row_index"].to_numpy(), np.asarray(idx))
            and expected[["gene_symbol", "mondo_id"]]
            .astype(str)
            .reset_index(drop=True)
            .equals(rebuilt.astype(str))
        )
        checks[partition] = {"n_pairs": int(len(rebuilt)), "matches_saved_split": bool(same)}
    return checks


BUILD_TABLES = (
    "feature_provenance_table.csv",
    "gene_disease_mapping_filter_audit.csv",
    "hpo_column_term_names.csv",
    "label_curation_summary.csv",
    "severity_hpo_terms_removed_at_build.csv",
    "severity_label_conflicts.csv",
)


TABLE_1_BEFORE_AFTER = {
    "mcc": "MCC",
    "qwk": "QWK",
    "balanced_accuracy": "BalAcc",
    "auc_roc": "AUC_ROC",
    "f1_macro": "F1_macro",
    "ranked_probability_score": "RPS",
    "adjacent_accuracy": "Adj_acc",
    "multiclass_brier": "Multiclass_Brier",
}


SAVED_TABLE_TOLERANCE = 1e-9


def _tables_match(recomputed: pd.DataFrame, saved: pd.DataFrame) -> bool:
    """Same rows in the same order; numbers equal to within SAVED_TABLE_TOLERANCE."""
    saved = saved[[c for c in recomputed.columns if c in saved.columns]].reset_index(drop=True)
    recomputed = recomputed[saved.columns].reset_index(drop=True)
    if recomputed.shape != saved.shape:
        return False
    for column in saved.columns:
        if pd.api.types.is_numeric_dtype(saved[column]):
            a = recomputed[column].to_numpy(float)
            b = saved[column].to_numpy(float)
            if not np.allclose(a, b, rtol=0.0, atol=SAVED_TABLE_TOLERANCE, equal_nan=True):
                return False
        elif not (recomputed[column].astype(str) == saved[column].astype(str)).all():
            return False
    return True


def check_tables_from_stored_predictions(data_dir: str, run_dir: str, output_dir: str) -> Dict[str, bool]:
    """Recompute the tables that the run derives from its stored test predictions and compare
    them with the saved ones. Nothing is refitted."""
    results_dir = os.path.join(run_dir, "results")
    out = os.path.join(output_dir, "recomputed")
    os.makedirs(out, exist_ok=True)
    checks: Dict[str, bool] = {}

    raw = raw_vs_calibrated_table(results_dir)
    raw.to_csv(os.path.join(out, "raw_vs_calibrated.csv"), index=False)
    checks["raw_vs_calibrated.csv"] = _tables_match(raw, pd.read_csv(os.path.join(results_dir, "raw_vs_calibrated.csv")))

    with open(os.path.join(results_dir, "all_results.pkl"), "rb") as handle:
        saved = pickle.load(handle)
    table_1 = pd.read_csv(os.path.join(results_dir, "publication_table.csv"))
    split = pd.read_csv(os.path.join(results_dir, "split_identifiers_primary.csv"))
    test = split[split["partition"] == "test"].sort_values("row_index", kind="stable")
    predictions: Dict[str, np.ndarray] = {}
    models: Dict[str, str] = {}
    y_true = None
    for record in table_1.loc[table_1["Selected_on_validation"].astype(bool)].itertuples(index=False):
        payload = saved["predictions"][f"{record.Scenario}_{record.Model}"]
        predictions[record.Scenario] = np.asarray(payload["y_pred"], dtype=int)
        models[record.Scenario] = record.Model
        if y_true is None:
            y_true = np.asarray(payload["y_true"], dtype=int)
    cutoff, cutoff_contrasts = moderate_severe_cutoff(
        y_true, predictions, models, test["gene_symbol"].to_numpy()
    )
    for name, table in (
        ("moderate_severe_cutoff.csv", cutoff),
        ("moderate_severe_cutoff_paired_contrasts.csv", cutoff_contrasts),
    ):
        table.to_csv(os.path.join(out, name), index=False)
        checks[name] = _tables_match(table, pd.read_csv(os.path.join(results_dir, name)))

    counts = recorded_hpo_term_counts(os.path.join(data_dir, Config.PRIMARY_DATA))
    n_terms = np.asarray(
        [int(counts.get((gene, mondo), 0)) for gene, mondo in zip(test["gene_symbol"], test["mondo_id"])]
    )
    term_count = hpo_term_count_comparison(
        y_true,
        predictions["S2_Mech_Pheno"],
        predictions["S3_Phenotype"],
        test["gene_symbol"].to_numpy(),
        n_terms,
    )
    term_count.to_csv(os.path.join(out, HPO_TERM_COUNT_FILE), index=False)
    checks[HPO_TERM_COUNT_FILE] = _tables_match(
        term_count, pd.read_csv(os.path.join(results_dir, HPO_TERM_COUNT_FILE))
    )

    evaluations = pd.read_csv(os.path.join(results_dir, "generalisation_evaluations.csv"))
    contrasts = pd.read_csv(os.path.join(results_dir, "generalisation_paired_contrasts.csv"))
    summaries, contrast_tables = [], []
    for mode in Config.ADDITIONAL_GENERALISATION_MODES:
        path = os.path.join(results_dir, f"generalisation_test_predictions_{mode}.csv")
        if not os.path.exists(path):
            continue
        summary, contrast = summarise_generalisation_predictions(pd.read_csv(path))
        summaries.append(summary)
        contrast_tables.append(contrast)
    summary = pd.concat(summaries, ignore_index=True)
    contrast = pd.concat(contrast_tables, ignore_index=True)
    summary.to_csv(os.path.join(out, "generalisation_evaluations.csv"), index=False)
    contrast.to_csv(os.path.join(out, "generalisation_paired_contrasts.csv"), index=False)
    checks["generalisation_evaluations.csv"] = _tables_match(summary, evaluations)
    checks["generalisation_paired_contrasts.csv"] = _tables_match(contrast, contrasts)
    return checks


def _same_file(first: str, second: str) -> bool:
    with open(first, "rb") as a, open(second, "rb") as b:
        return a.read() == b.read()


def run_verification(data_dir: str, run_dir: str, output_dir: str) -> Dict[str, Any]:
    """Run the three checks and write the report. Returns the report."""
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    Config.configure(data_dir, output_dir)
    container = assemble_full_dataframe()
    make_train_test_split(container)
    split = _check_split(container, run_dir)
    tables = {
        name: _same_file(os.path.join(output_dir, "results", name), os.path.join(run_dir, "results", name))
        for name in BUILD_TABLES
    }

    with open(os.path.join(run_dir, "results", "all_results.pkl"), "rb") as handle:
        saved = pickle.load(handle)
    table_1 = pd.read_csv(os.path.join(run_dir, "results", "publication_table.csv"))
    selected = table_1.loc[table_1["Selected_on_validation"].astype(bool)]
    test_idx = np.asarray(container.test_idx)
    y_test = container.y[test_idx]

    rows: List[Dict[str, Any]] = []
    models: Dict[str, Any] = {}
    calibration_records = []
    before_after: List[Dict[str, Any]] = []
    test_genes = np.asarray(container.groups)[test_idx]
    for record in selected.itertuples(index=False):
        key = f"{record.Scenario}_{record.Model}"
        bundle = load_bundle(os.path.join(run_dir, "models", f"{key}.pkl"))
        X_test = _test_matrix(container, bundle, test_idx)
        probabilities = bundle_probabilities(bundle, X_test)
        predictions = probabilities.argmax(axis=1)
        raw_probabilities = _predict_uncalibrated_candidate(bundle, X_test)
        raw_predictions = raw_probabilities.argmax(axis=1)
        calibration_records.append((record.Scenario, record.Model, y_test, raw_predictions, predictions))
        raw_metrics = compute_all_metrics(y_test, raw_predictions, raw_probabilities)
        # Table 1a: the same model before and after calibration, with gene-cluster intervals
        for status, preds, probs in (
            ("before calibration", raw_predictions, raw_probabilities),
            ("after calibration", probabilities.argmax(axis=1), probabilities),
        ):
            values = compute_all_metrics(y_test, preds, probs)
            ci = compute_bootstrap_ci(
                y_test, preds, probs, n_bootstrap=Config.N_BOOTSTRAP, metrics_list=["mcc", "qwk"], groups=test_genes
            )
            before_after.append(
                {
                    "Scenario": record.Scenario,
                    "Model": record.Model,
                    "status": status,
                    **{name: values[key] for key, name in TABLE_1_BEFORE_AFTER.items()},
                    "MCC_CI_low": ci["mcc"]["ci_low"],
                    "MCC_CI_high": ci["mcc"]["ci_high"],
                    "QWK_CI_low": ci["qwk"]["ci_low"],
                    "QWK_CI_high": ci["qwk"]["ci_high"],
                }
            )
        reference = saved["predictions"][key]
        max_difference = float(np.max(np.abs(probabilities - reference["y_prob"])))
        metrics = compute_all_metrics(y_test, predictions, probabilities)
        models[key] = {
            "n_test_pairs": int(len(y_test)),
            "labels_match_saved": bool(np.array_equal(y_test, reference["y_true"])),
            "predicted_classes_match_saved": bool(np.array_equal(predictions, reference["y_pred"])),
            "max_probability_difference": max_difference,
            "probabilities_match_saved": max_difference <= PROBABILITY_TOLERANCE,
        }
        for column, name in TABLE_1_COLUMNS.items():
            published = float(getattr(record, column))
            rows.append(
                {
                    "Scenario": record.Scenario,
                    "Model": record.Model,
                    "metric": column,
                    "table_1": published,
                    "recomputed": float(metrics[name]),
                    "matches": bool(abs(published - float(metrics[name])) <= METRIC_TOLERANCE),
                }
            )
        for column, name in UNCALIBRATED_COLUMNS.items():
            published = float(getattr(record, column))
            rows.append(
                {
                    "Scenario": record.Scenario,
                    "Model": record.Model,
                    "metric": column,
                    "table_1": published,
                    "recomputed": float(raw_metrics[name]),
                    "matches": bool(abs(published - float(raw_metrics[name])) <= METRIC_TOLERANCE),
                }
            )

    confusion, changes, transitions = calibration_change_tables(calibration_records)
    confusion.to_csv(os.path.join(output_dir, CONFUSION_FILE), index=False)
    changes.to_csv(os.path.join(output_dir, CHANGES_FILE), index=False)
    transitions.to_csv(os.path.join(output_dir, TRANSITIONS_FILE), index=False)
    saved_confusion = os.path.join(run_dir, "figures", "publication", "figure_data", CONFUSION_FILE)
    calibration_figure_data = (
        confusion_counts_match(confusion, pd.read_csv(saved_confusion))
        if os.path.exists(saved_confusion)
        else None
    )

    stored_prediction_tables = check_tables_from_stored_predictions(data_dir, run_dir, output_dir)

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(output_dir, "verification_metrics.csv"), index=False)
    pd.DataFrame(before_after).to_csv(os.path.join(output_dir, "table_1_before_after_calibration.csv"), index=False)
    passed = (
        all(tables.values())
        and all(item["matches_saved_split"] for item in split.values())
        and all(item["labels_match_saved"] and item["probabilities_match_saved"] for item in models.values())
        and bool(table["matches"].all())
        and calibration_figure_data is not False
        and all(stored_prediction_tables.values())
    )
    report = {
        "passed": passed,
        "feature_tables_identical": tables,
        "split": split,
        "models": models,
        "metrics_checked": int(len(table)),
        "metrics_matching": int(table["matches"].sum()),
        "extended_data_fig_3_matches_saved": calibration_figure_data,
        "tables_from_stored_predictions_match_saved": stored_prediction_tables,
        "class_changes_with_calibration": changes[
            ["Scenario", "Model", "n_changed", "n_became_correct", "n_became_incorrect"]
        ].to_dict(orient="records"),
    }
    with open(os.path.join(output_dir, "verification.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    return report
