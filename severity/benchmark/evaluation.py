"""Test-set statistics (Statistics and reproducibility).

Confidence intervals come from 1,000 bootstrap samples drawn by gene and stratified by severity.
Paired differences between feature sets use the same samples. Expected calibration error uses
five equal-count bins per class (equation 10). Also writes Table 1, Supplementary Tables 5-8 and
the error decomposition behind Fig. 3i.
"""

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import RANDOM_SEED, Config
from .metrics import (
    _brier_one_class,
    _calibration_bin_table,
    _compute_ece_one_class,
    _paired_contrast_metric,
    _ranked_probability_score,
    compute_all_metrics,
)
from .runtime import DataContainer, _fold_artifacts, logger, results_collector
from .workflow import _pick_best_model_per_scenario


def analyze_error_types(all_results: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.subsection("Error decomposition (global + per-class)")
    severity_names = ["Mild", "Moderate", "Severe", "Profound"]
    global_rows = []
    per_class_rows = []
    for scenario, strata in all_results.items():
        for stratum, models in strata.items():
            for mn, res in models.items():
                if "predictions" not in res or "y_true" not in res:
                    continue
                yt = res["y_true"]
                yp = res["predictions"]["y_pred"]
                diff = np.abs(yt - yp)
                correct = int((diff == 0).sum())
                adjacent = int((diff == 1).sum())
                non_adjacent = int((diff >= 2).sum())
                total = len(yt)
                global_rows.append(
                    {
                        "Scenario": scenario,
                        "Model": mn,
                        "correct": correct,
                        "adjacent_error": adjacent,
                        "non_adjacent_error": non_adjacent,
                        "frac_correct": correct / total if total else 0,
                        "frac_adjacent": adjacent / total if total else 0,
                        "frac_non_adjacent": non_adjacent / total if total else 0,
                        "n_total": total,
                    }
                )
                for c in range(Config.NUM_CLASSES):
                    mask = yt == c
                    n_c = int(mask.sum())
                    if n_c == 0:
                        per_class_rows.append(
                            {
                                "Scenario": scenario,
                                "Model": mn,
                                "true_class": severity_names[c],
                                "n_total": 0,
                                "frac_correct": np.nan,
                                "frac_adjacent": np.nan,
                                "frac_non_adjacent": np.nan,
                            }
                        )
                        continue
                    diff_c = np.abs(yt[mask] - yp[mask])
                    correct_c = int((diff_c == 0).sum())
                    adjacent_c = int((diff_c == 1).sum())
                    non_adj_c = int((diff_c >= 2).sum())
                    per_class_rows.append(
                        {
                            "Scenario": scenario,
                            "Model": mn,
                            "true_class": severity_names[c],
                            "n_total": n_c,
                            "frac_correct": correct_c / n_c,
                            "frac_adjacent": adjacent_c / n_c,
                            "frac_non_adjacent": non_adj_c / n_c,
                        }
                    )
    global_df = pd.DataFrame(global_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    global_df.to_csv(os.path.join(Config.OUTPUT_DIR, "results", "error_decomposition.csv"), index=False)
    per_class_df.to_csv(
        os.path.join(Config.OUTPUT_DIR, "results", "error_decomposition_per_class.csv"), index=False
    )
    return (global_df, per_class_df)


def _cluster_bootstrap_indices(
    y_true: np.ndarray, groups: np.ndarray, rng: np.random.RandomState
) -> np.ndarray:
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    representatives: Dict[int, List[Any]] = defaultdict(list)
    for group in unique_groups:
        labels = y_true[groups == group]
        counts = np.bincount(labels.astype(int), minlength=Config.NUM_CLASSES)
        representatives[int(np.argmax(counts))].append(group)
    sampled_rows: List[int] = []
    for _, group_list in representatives.items():
        if not group_list:
            continue
        sampled_groups = rng.choice(group_list, size=len(group_list), replace=True)
        for group in sampled_groups:
            sampled_rows.extend(np.where(groups == group)[0].tolist())
    return np.asarray(sampled_rows, dtype=int)


def compute_bootstrap_ci(
    y_true, y_pred, y_prob, n_bootstrap=1000, metrics_list=None, groups=None
) -> Dict[str, Dict[str, float]]:
    if metrics_list is None:
        metrics_list = [
            "mcc",
            "qwk",
            "balanced_accuracy",
            "f1",
            "auc_roc",
            "ranked_probability_score",
            "multiclass_brier",
            "spearman_rho",
        ]
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    groups = np.asarray(groups if groups is not None else np.arange(len(y_true)))
    rng = np.random.RandomState(RANDOM_SEED)
    values = {metric: [] for metric in metrics_list}
    attempted = 0
    for _ in range(n_bootstrap):
        idx = _cluster_bootstrap_indices(y_true, groups, rng)
        if len(idx) == 0:
            continue
        attempted += 1
        try:
            metrics = compute_all_metrics(y_true[idx], y_pred[idx], y_prob[idx])
            for metric in metrics_list:
                value = metrics.get(metric, np.nan)
                if np.isfinite(value):
                    values[metric].append(float(value))
        except Exception as exc:
            logger.log(f"  Bootstrap replicate failed: {type(exc).__name__}: {exc}", level="WARNING")
    output: Dict[str, Dict[str, float]] = {}
    for metric, metric_values in values.items():
        array = np.asarray(metric_values, dtype=float)
        n_valid = int(len(array))
        output[metric] = {
            "mean": float(np.mean(array)) if n_valid else np.nan,
            "ci_low": float(np.percentile(array, 2.5)) if n_valid >= 10 else np.nan,
            "ci_high": float(np.percentile(array, 97.5)) if n_valid >= 10 else np.nan,
            "n": n_valid,
            "n_valid": n_valid,
            "n_attempted": int(attempted),
            "n_requested": int(n_bootstrap),
            "n_undefined": int(max(attempted - n_valid, 0)),
        }
    return output


def run_bootstrap_all(all_results: Dict) -> pd.DataFrame:
    logger.subsection("Gene-cluster bootstrap confidence intervals")
    rows = []
    for scenario, model_name in _pick_best_model_per_scenario(all_results).items():
        result = all_results[scenario]["all"][model_name]
        if "predictions" not in result:
            continue
        ci = compute_bootstrap_ci(
            result["y_true"],
            result["predictions"]["y_pred"],
            result["predictions"]["y_prob"],
            n_bootstrap=Config.N_BOOTSTRAP,
            groups=result.get("test_gene_groups", result.get("test_groups")),
        )
        row = {
            "Scenario": scenario,
            "Stratum": "all",
            "Model": model_name,
            "bootstrap_unit": "gene_cluster",
        }
        for metric, stats_dict in ci.items():
            row[f"{metric}_mean"] = stats_dict["mean"]
            row[f"{metric}_ci_low"] = stats_dict["ci_low"]
            row[f"{metric}_ci_high"] = stats_dict["ci_high"]
            row[f"{metric}_n_valid"] = stats_dict["n_valid"]
            row[f"{metric}_n_attempted"] = stats_dict["n_attempted"]
            row[f"{metric}_n_undefined"] = stats_dict["n_undefined"]
        rows.append(row)
    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(os.path.join(Config.OUTPUT_DIR, "results", "bootstrap_cis.csv"), index=False)
    return dataframe


PAIRED_SCENARIO_CONTRASTS: Tuple[Tuple[str, str], ...] = (
    ("S3_Phenotype", "S2_Mech_Pheno"),
    ("S2_Mech_Pheno", "S1b_Disease_Context"),
    ("S1b_Disease_Context", "S1_Mechanistic"),
)


PAIRED_CONTRAST_METRICS: Tuple[str, ...] = ("mcc", "qwk")


def run_paired_scenario_contrasts(all_results: Dict, container: DataContainer) -> pd.DataFrame:
    """Paired gene-cluster bootstrap contrasts between adjacent scenarios.

    Nothing is refit and no additional test evaluation occurs. The stored
    locked-test predictions in ``results_collector.predictions`` (keyed
    ``"<scenario>_<model>"``) are resampled with the same severity-stratified
    gene-cluster scheme used for the Table 2 confidence intervals
    (``_cluster_bootstrap_indices``, Config.N_BOOTSTRAP replicates, RANDOM_SEED).
    Each replicate draws one set of rows and scores *both* scenarios on it before
    the difference is taken; the pairing is what makes the interval informative.

    The reported fraction of replicates at or below zero is a bootstrap tail
    fraction, not a P value: it is not calibrated against a null hypothesis and
    must not be described as one.
    """
    logger.subsection("Paired between-scenario gene-cluster bootstrap contrasts")
    output_path = os.path.join(Config.OUTPUT_DIR, "results", "paired_scenario_contrasts.csv")
    selected = _pick_best_model_per_scenario(all_results)
    predictions: Dict[str, Dict[str, np.ndarray]] = {}
    models: Dict[str, str] = {}
    groups_by_scenario: Dict[str, np.ndarray] = {}
    for scenario, model_name in selected.items():
        payload = results_collector.predictions.get(f"{scenario}_{model_name}")
        if not payload:
            logger.log(
                f"  No stored locked-test predictions for {scenario}_{model_name}; that scenario is excluded from the paired contrasts.",
                level="WARNING",
            )
            continue
        predictions[scenario] = {
            "y_true": np.asarray(payload["y_true"], dtype=int),
            "y_pred": np.asarray(payload["y_pred"], dtype=int),
            "y_prob": np.asarray(payload["y_prob"], dtype=float),
        }
        models[scenario] = model_name
        result = all_results[scenario]["all"][model_name]
        groups = result.get("test_gene_groups", result.get("test_groups"))
        if groups is None:
            artifact = _fold_artifacts.get(scenario, {})
            test_idx = np.asarray(artifact.get("test_idx", container.test_idx), dtype=int)
            groups = np.asarray(container.groups)[test_idx]
        groups_by_scenario[scenario] = np.asarray(groups)
    rows: List[Dict[str, Any]] = []
    if len(predictions) < 2:
        logger.log(
            "  Fewer than two scenarios have stored locked-test predictions; paired contrasts cannot be computed.",
            level="WARNING",
        )
        pd.DataFrame(rows).to_csv(output_path, index=False)
        return pd.DataFrame(rows)
    reference = next(iter(predictions))
    y_true = predictions[reference]["y_true"]
    groups = groups_by_scenario[reference]
    for scenario, payload in predictions.items():
        if not np.array_equal(payload["y_true"], y_true):
            raise RuntimeError(
                f"Paired contrasts require an identical locked-test row order across scenarios, but {scenario} has different stored true labels than {reference}."
            )
        if not np.array_equal(
            np.asarray(groups_by_scenario[scenario], dtype=str), np.asarray(groups, dtype=str)
        ):
            raise RuntimeError(
                f"Paired contrasts require identical test gene clusters across scenarios, but {scenario} differs from {reference}."
            )
    if len(groups) != len(y_true):
        raise RuntimeError(
            f"Test gene-cluster vector has length {len(groups)} but the stored predictions have length {len(y_true)}."
        )
    rng = np.random.RandomState(RANDOM_SEED)
    replicate_indices: List[np.ndarray] = []
    for _ in range(Config.N_BOOTSTRAP):
        idx = _cluster_bootstrap_indices(y_true, groups, rng)
        if len(idx) == 0:
            continue
        replicate_indices.append(idx)
    logger.log(
        f"  Drew {len(replicate_indices)} of {Config.N_BOOTSTRAP} non-empty severity-stratified gene-cluster replicates (seed={RANDOM_SEED}); {len(np.unique(groups))} gene clusters over {len(y_true)} locked-test pairs."
    )
    replicate_values: Dict[Tuple[str, str], np.ndarray] = {}
    observed: Dict[Tuple[str, str], float] = {}
    for scenario, payload in predictions.items():
        for metric in PAIRED_CONTRAST_METRICS:
            observed[scenario, metric] = _paired_contrast_metric(y_true, payload["y_pred"], metric)
            values = np.full(len(replicate_indices), np.nan, dtype=float)
            for position, idx in enumerate(replicate_indices):
                try:
                    values[position] = _paired_contrast_metric(y_true[idx], payload["y_pred"][idx], metric)
                except Exception as exc:
                    logger.log(
                        f"  Paired replicate undefined for {scenario}/{metric}: {type(exc).__name__}: {exc}",
                        level="WARNING",
                    )
            replicate_values[scenario, metric] = values
    for scenario_a, scenario_b in PAIRED_SCENARIO_CONTRASTS:
        if scenario_a not in predictions or scenario_b not in predictions:
            logger.log(
                f"  Skipping contrast {scenario_a} minus {scenario_b}: one or both scenarios have no stored locked-test predictions.",
                level="WARNING",
            )
            continue
        for metric in PAIRED_CONTRAST_METRICS:
            differences = replicate_values[scenario_a, metric] - replicate_values[scenario_b, metric]
            valid = differences[np.isfinite(differences)]
            n_valid = int(len(valid))
            rows.append(
                {
                    "contrast": f"{scenario_a}_minus_{scenario_b}",
                    "scenario_a": scenario_a,
                    "scenario_b": scenario_b,
                    "model_a": models[scenario_a],
                    "model_b": models[scenario_b],
                    "metric": metric,
                    "observed_a": observed[scenario_a, metric],
                    "observed_b": observed[scenario_b, metric],
                    "observed_difference": observed[scenario_a, metric] - observed[scenario_b, metric],
                    "difference_percentile_2.5": float(np.percentile(valid, 2.5))
                    if n_valid >= 10
                    else np.nan,
                    "difference_percentile_97.5": float(np.percentile(valid, 97.5))
                    if n_valid >= 10
                    else np.nan,
                    "bootstrap_tail_fraction_at_or_below_zero": float(np.mean(valid <= 0.0))
                    if n_valid
                    else np.nan,
                    "n_valid_replicates": n_valid,
                    "n_replicates_drawn": int(len(replicate_indices)),
                    "n_replicates_requested": int(Config.N_BOOTSTRAP),
                    "n_test_pairs": int(len(y_true)),
                    "n_test_gene_clusters": int(len(np.unique(groups))),
                    "bootstrap_unit": "gene_cluster",
                    "resampling": "severity_stratified_gene_cluster_with_replacement_paired",
                    "paired": True,
                    "random_seed": int(RANDOM_SEED),
                    "refit": False,
                    "tail_fraction_interpretation": "bootstrap tail fraction, not a P value",
                }
            )
    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(output_path, index=False)
    for row in rows:
        logger.log(
            f"  {row['contrast']} {row['metric']}: delta={row['observed_difference']:+.4f} [{row['difference_percentile_2.5']:+.4f}, {row['difference_percentile_97.5']:+.4f}] tail_fraction<=0={row['bootstrap_tail_fraction_at_or_below_zero']:.4f} (n_valid={row['n_valid_replicates']})"
        )
    return dataframe


def _bootstrap_calibration_ci(
    y_true_binary: np.ndarray,
    y_prob_class: np.ndarray,
    n_bins: int,
    strategy: str,
    n_bootstrap: int,
    seed: int,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    y = np.asarray(y_true_binary, dtype=int)
    probability = np.asarray(y_prob_class, dtype=float)
    groups_array = np.asarray(groups if groups is not None else np.arange(len(y)))
    rng = np.random.RandomState(seed)
    ece_values: List[float] = []
    brier_values: List[float] = []
    for _ in range(n_bootstrap):
        idx = _cluster_bootstrap_indices(y, groups_array, rng)
        if len(idx) == 0:
            continue
        ece_values.append(_compute_ece_one_class(y[idx], probability[idx], n_bins, strategy))
        brier_values.append(_brier_one_class(y[idx], probability[idx]))

    def summary(values: List[float]) -> Tuple[float, float]:
        arr = np.asarray([v for v in values if np.isfinite(v)])
        if len(arr) < 10:
            return (np.nan, np.nan)
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    ece_low, ece_high = summary(ece_values)
    brier_low, brier_high = summary(brier_values)
    ece_valid = int(np.isfinite(np.asarray(ece_values, dtype=float)).sum())
    brier_valid = int(np.isfinite(np.asarray(brier_values, dtype=float)).sum())
    return {
        "ece_ci_low": ece_low,
        "ece_ci_high": ece_high,
        "brier_ci_low": brier_low,
        "brier_ci_high": brier_high,
        "ece_n_valid": ece_valid,
        "brier_n_valid": brier_valid,
        "bootstrap_n_requested": int(n_bootstrap),
        "ece_n_undefined": int(max(n_bootstrap - ece_valid, 0)),
        "brier_n_undefined": int(max(n_bootstrap - brier_valid, 0)),
    }


def compute_ece_table(all_results: Dict) -> pd.DataFrame:
    logger.subsection("Calibrated and uncalibrated probability metrics with gene-cluster CIs")
    severity_names = ["Mild", "Moderate", "Severe", "Profound"]
    rows: List[Dict[str, Any]] = []
    bin_rows: List[Dict[str, Any]] = []
    for scenario_index, (scenario, model_name) in enumerate(
        _pick_best_model_per_scenario(all_results).items()
    ):
        result = all_results[scenario]["all"][model_name]
        y_true = np.asarray(result["y_true"])
        groups = result.get("test_gene_groups", result.get("test_groups"))
        applied_status = (
            "calibrated" if result.get("calibration_status") == "success" else "uncalibrated_fallback"
        )
        for status, prediction_key in (
            (applied_status, "predictions"),
            ("uncalibrated", "predictions_uncalibrated"),
        ):
            probabilities = np.asarray(result[prediction_key]["y_prob"])
            one_hot = np.eye(Config.NUM_CLASSES)[y_true.astype(int)]
            multiclass_brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
            rps = _ranked_probability_score(y_true, probabilities)
            for class_index, class_name in enumerate(severity_names):
                binary = (y_true == class_index).astype(int)
                ece = _compute_ece_one_class(
                    binary,
                    probabilities[:, class_index],
                    Config.CALIBRATION_N_BINS,
                    Config.CALIBRATION_BINNING,
                )
                brier = _brier_one_class(binary, probabilities[:, class_index])
                ci = _bootstrap_calibration_ci(
                    binary,
                    probabilities[:, class_index],
                    Config.CALIBRATION_N_BINS,
                    Config.CALIBRATION_BINNING,
                    Config.CALIBRATION_N_BOOTSTRAP,
                    RANDOM_SEED + scenario_index * 100 + class_index,
                    groups=np.asarray(groups),
                )
                bins = _calibration_bin_table(
                    binary,
                    probabilities[:, class_index],
                    Config.CALIBRATION_N_BINS,
                    Config.CALIBRATION_BINNING,
                )
                for _, bin_row in bins.iterrows():
                    bin_rows.append(
                        {
                            "Scenario": scenario,
                            "Model": model_name,
                            "calibration_status": status,
                            "severity_class": class_name,
                            **bin_row.to_dict(),
                        }
                    )
                rows.append(
                    {
                        "Scenario": scenario,
                        "Model": model_name,
                        "calibration_status": status,
                        "severity_class": class_name,
                        "calibration_method": result.get("calibration_method", "none"),
                        "calibration_method_attempted": result.get("calibration_method_attempted", "none"),
                        "calibration_success": bool(result.get("calibration_success", False)),
                        "pipeline_calibration_status": result.get("calibration_status", "not_attempted"),
                        "bootstrap_unit": "gene_cluster",
                        "ece": ece,
                        **ci,
                        "brier": brier,
                        "multiclass_brier": multiclass_brier,
                        "ranked_probability_score": rps,
                        "n_test": len(y_true),
                        "n_test_in_class": int(binary.sum()),
                    }
                )
    dataframe = pd.DataFrame(rows)
    bins_dataframe = pd.DataFrame(bin_rows)
    results_dir = os.path.join(Config.OUTPUT_DIR, "results")
    dataframe.to_csv(os.path.join(results_dir, "calibration_metrics.csv"), index=False)
    dataframe.to_csv(os.path.join(results_dir, "calibration_ece.csv"), index=False)
    bins_dataframe.to_csv(os.path.join(results_dir, "calibration_bins.csv"), index=False)
    return dataframe


def create_publication_table(all_results: Dict, bootstrap_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for scenario, model_name in _pick_best_model_per_scenario(all_results).items():
        result = all_results[scenario]["all"][model_name]
        calibrated = result.get("metrics", {})
        raw = result.get("test_metrics_uncalibrated", {})
        row: Dict[str, Any] = {
            "Scenario": scenario,
            "Stratum": "all",
            "Model": model_name,
            "Selected_on_validation": True,
            "Calibration_status": result.get("calibration_status", "not_attempted"),
            "Calibration_success": bool(result.get("calibration_success", False)),
            "Calibration_method": result.get("calibration_method", "none"),
            "Calibration_method_attempted": result.get("calibration_method_attempted", "none"),
            "Calibration_failure_details": result.get("calibration_failure_details", ""),
            "MCC": calibrated.get("mcc", np.nan),
            "QWK": calibrated.get("qwk", np.nan),
            "BalAcc": calibrated.get("balanced_accuracy", np.nan),
            "AUC_ROC": calibrated.get("auc_roc", np.nan),
            "F1_macro": calibrated.get("f1_macro", np.nan),
            "RPS": calibrated.get("ranked_probability_score", np.nan),
            "Multiclass_Brier": calibrated.get("multiclass_brier", np.nan),
            "Uncalibrated_MCC": raw.get("mcc", np.nan),
            "Uncalibrated_QWK": raw.get("qwk", np.nan),
            "Uncalibrated_BalAcc": raw.get("balanced_accuracy", np.nan),
            "Uncalibrated_RPS": raw.get("ranked_probability_score", np.nan),
            "Uncalibrated_Multiclass_Brier": raw.get("multiclass_brier", np.nan),
        }
        if not bootstrap_df.empty:
            match = bootstrap_df[
                (bootstrap_df["Scenario"] == scenario) & (bootstrap_df["Model"] == model_name)
            ]
            if not match.empty:
                boot = match.iloc[0]
                for metric, label in (
                    ("mcc", "MCC"),
                    ("qwk", "QWK"),
                    ("ranked_probability_score", "RPS"),
                ):
                    row[f"{label}_CI_low"] = boot.get(f"{metric}_ci_low", np.nan)
                    row[f"{label}_CI_high"] = boot.get(f"{metric}_ci_high", np.nan)
        rows.append(row)
    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(os.path.join(Config.OUTPUT_DIR, "results", "publication_table.csv"), index=False)
    return dataframe


def _benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 1.0
    n = len(p)
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = n - reverse_rank + 1
        running = min(running, p[index] * n / rank)
        adjusted[index] = min(running, 1.0)
    return adjusted
