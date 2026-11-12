"""Model development: nested cross-validation, model selection and the single test evaluation.

For each feature set and each of the four model families, hyperparameters are tuned by random
search over three gene-grouped training folds (preprocessing and feature selection refitted in
each fold), ranked by QWK, then MCC, then balanced accuracy. Each family is refitted on all
training pairs and the best family on validation QWK is selected. The selected model is
calibrated, frozen and evaluated once on the test set. The disease-disjoint and
gene-and-disease-disjoint evaluations rerun this whole workflow on their own splits.
"""

import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .annotation_terms import (
    SCENARIO_TEXT_SOURCES,
    _add_training_text_features,
    _learn_text_vocabulary,
)
from .calibration import (
    _predict_calibrated_candidate,
    _predict_uncalibrated_candidate,
    freeze_selected_model_and_calibrator,
)
from .config import RANDOM_SEED, Config
from .metrics import (
    _finite_metric_or_neg_inf,
    _metric_key,
    compute_all_metrics,
    validation_selection_score,
)
from .models import (
    _available_candidate_names,
    _fit_model_matrix,
    _get_multinomial_logistic_estimator,
    _importance_table,
    _parameter_candidates,
    _predict_model_matrix,
)
from .partition import (
    _gene_disease_component_groups,
    _nested_splits,
    _optimised_group_holdout,
    _split_development_train_validation,
    _write_split_identifiers,
    write_split_summary,
)
from .preprocessing import FoldPreprocessor
from .runtime import (
    HPO_COLUMN_TERM_NAMES,
    DataContainer,
    _fold_artifacts,
    _generalisation_results,
    _high_mi_audit_records,
    _test_evaluation_registry,
    logger,
    results_collector,
)
from .selection import select_features_mi, select_features_mi_grouped, uses_grouped_mi
from .severity_terms import filter_leaky_columns_for_phenotype_scenarios


def _register_final_test_evaluation(scenario_name: str, model_name: str) -> None:
    _test_evaluation_registry[scenario_name] += 1
    if _test_evaluation_registry[scenario_name] > 1:
        raise RuntimeError(
            f"Main test set requested more than once for {scenario_name}. The second request came from {model_name}."
        )
    logger.log(
        f"  FINAL TEST GATE OPENED ONCE: {scenario_name}/{model_name}; model and calibrator are frozen."
    )


def evaluate_frozen_selected_model_on_test(
    result: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray, scenario_name: str
) -> Dict[str, Any]:
    """Evaluate an already-frozen model/calibrator on the main test set once."""
    model_name = result["model_name"]
    if not result.get("model_frozen", False) or not result.get("calibrator_frozen", False):
        raise RuntimeError(
            f"Refusing test evaluation for {scenario_name}/{model_name}: model and calibrator were not frozen first."
        )
    if _test_evaluation_registry[scenario_name] >= 1:
        raise RuntimeError(
            f"Main test set requested more than once for {scenario_name}. The repeated request came from {model_name}."
        )
    _register_final_test_evaluation(scenario_name, model_name)
    raw_test_probability = _predict_uncalibrated_candidate(result, X_test)
    calibrated_test_probability = _predict_calibrated_candidate(result, X_test)
    raw_test_prediction = np.argmax(raw_test_probability, axis=1)
    calibrated_test_prediction = np.argmax(calibrated_test_probability, axis=1)
    result["test_metrics_uncalibrated"] = compute_all_metrics(
        y_test, raw_test_prediction, raw_test_probability
    )
    result["metrics"] = compute_all_metrics(y_test, calibrated_test_prediction, calibrated_test_probability)
    result["predictions_uncalibrated"] = {
        "y_pred": raw_test_prediction,
        "y_prob": raw_test_probability,
    }
    result["predictions"] = {
        "y_pred": calibrated_test_prediction,
        "y_prob": calibrated_test_probability,
    }
    result["y_true"] = np.asarray(y_test)
    result["test_evaluated"] = True
    result["test_evaluation_count"] = _test_evaluation_registry[scenario_name]
    result["evaluation_partition"] = "untouched_test"
    logger.log(
        f"    FINAL {model_name}: test MCC={result['metrics']['mcc']:.4f}, QWK={result['metrics']['qwk']:.4f}; uncalibrated MCC={result['test_metrics_uncalibrated']['mcc']:.4f}"
    )
    return result


def _pick_best_model_per_scenario(all_results: Dict) -> Dict[str, str]:
    """The model family selected on validation for each feature set (never chosen by test metrics)."""
    best: Dict[str, str] = {}
    for scenario, strata in all_results.items():
        models = strata.get("all", {})
        selected = [name for name, result in models.items() if result.get("selected_on_validation", False)]
        if selected:
            best[scenario] = selected[0]
            continue
        eligible = [
            (validation_selection_score(result["val_metrics"]), name)
            for name, result in models.items()
            if result.get("val_metrics")
        ]
        if eligible:
            eligible.sort(key=lambda item: (-item[0], item[1]))
            best[scenario] = eligible[0][1]
    return best


def _fit_scenario_representation(
    container: DataContainer,
    scenario: str,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    budget: Optional[Dict[str, int]],
    tag: str,
    drop_severity_hpo_terms: Optional[bool] = None,
) -> Dict[str, Any]:
    allowed_text_sources = tuple(SCENARIO_TEXT_SOURCES.get(scenario, ()))
    unknown_sources = set(allowed_text_sources) - {"primary", "gnn"}
    if unknown_sources:
        raise ValueError(f"{scenario}: unknown text sources configured: {sorted(unknown_sources)}")
    dynamic_text_cols: List[str] = []
    dynamic_lookup: Dict[str, str] = {}
    vocabulary: Dict[str, Any] = {}
    if allowed_text_sources:
        full_vocabulary = _learn_text_vocabulary(container, train_idx)
        vocabulary = {
            source: full_vocabulary[source] for source in allowed_text_sources if source in full_vocabulary
        }
        all_indices = np.arange(len(container.full_df), dtype=int)
        temp_df, dynamic_text_cols, dynamic_lookup = _add_training_text_features(
            container=container,
            indices=all_indices,
            vocabulary=vocabulary,
            family_lookup=dynamic_lookup,
        )
        if len(temp_df) != len(container.full_df):
            raise RuntimeError("Text-feature frame lost global row alignment.")
    else:
        temp_df = container.full_df
    candidate_cols = list(container.scenario_cols.get(scenario, [])) + dynamic_text_cols
    if not HPO_COLUMN_TERM_NAMES and getattr(container, "hpo_column_term_names", None):
        HPO_COLUMN_TERM_NAMES.update(container.hpo_column_term_names)
    restored_severity_cols: List[str] = []
    if drop_severity_hpo_terms is False:
        restored_severity_cols = [
            column
            for column in container.severity_hpo_candidates.get(scenario, [])
            if column not in candidate_cols
        ]
        candidate_cols = candidate_cols + restored_severity_cols
    candidate_cols = list(dict.fromkeys((column for column in candidate_cols if column in temp_df.columns)))
    restored_severity_cols = [column for column in restored_severity_cols if column in candidate_cols]
    candidate_cols, dynamic_static_drops = filter_leaky_columns_for_phenotype_scenarios(
        candidate_cols, scenario, drop_severity_hpo_terms=drop_severity_hpo_terms
    )
    combined_static_drops = list(container.scenario_static_drops.get(scenario, []))
    if drop_severity_hpo_terms is False:
        combined_static_drops = [
            item for item in combined_static_drops if item[1] != "severity_hpo_blocklist"
        ]
    combined_static_drops.extend(dynamic_static_drops)
    combined_static_drops = list(dict.fromkeys(combined_static_drops))
    text_prefix_to_source = {
        "feat_primary_": "primary",
        "has_primary_": "primary",
        "feat_gnn_": "gnn",
        "has_gnn_": "gnn",
    }
    disallowed_text_cols = [
        column
        for column in candidate_cols
        for prefix, source in text_prefix_to_source.items()
        if column.startswith(prefix) and source not in allowed_text_sources
    ]
    if disallowed_text_cols:
        raise RuntimeError(
            f"{scenario}: disallowed text features entered the scenario: {disallowed_text_cols[:10]}"
        )
    logger.log(
        f"  {scenario}: allowed text sources={(list(allowed_text_sources) if allowed_text_sources else 'none')}; dynamic text columns={len(dynamic_text_cols)}"
    )
    family_lookup = dict(container.feature_family_lookup)
    family_lookup.update(dynamic_lookup)
    preprocessor = FoldPreprocessor(
        scenario,
        container.hpo_binary_cols,
        container.mondo_cols,
        static_drops=combined_static_drops,
    )
    X_train_full, feature_names = preprocessor.fit_transform(temp_df, train_idx, candidate_cols)
    X_eval_full = preprocessor.transform(temp_df, eval_idx)
    y_train = container.y[train_idx]
    if uses_grouped_mi(scenario):
        actual_budget = dict(budget or Config.FEATURE_BUDGET)
        X_train, selected_names, _, selected_mask, runtime_drops, family_counts = select_features_mi_grouped(
            X_train_full,
            y_train,
            feature_names,
            family_lookup=family_lookup,
            budget=actual_budget,
            scenario_name=tag,
        )
    else:
        mechanistic = (
            ("mechanistic" in scenario.lower() or "S1" in scenario)
            and "S1b" not in scenario
            and ("pheno" not in scenario.lower())
        )
        X_train, selected_names, _, selected_mask, runtime_drops = select_features_mi(
            X_train_full,
            y_train,
            feature_names,
            mi_threshold=Config.MI_THRESHOLD_MECHANISTIC if mechanistic else Config.MI_THRESHOLD_DEFAULT,
            min_features=Config.MI_MIN_FEATURES_MECHANISTIC
            if mechanistic
            else Config.MI_MIN_FEATURES_DEFAULT,
            scenario_name=tag,
            family_lookup=family_lookup,
        )
        family_counts = {}
        actual_budget = dict(budget or Config.FEATURE_BUDGET)
    return {
        "X_train": X_train,
        "X_eval": X_eval_full[:, selected_mask],
        "X_train_full": X_train_full,
        "X_eval_full": X_eval_full,
        "feature_names_full": feature_names,
        "selected_mask": selected_mask,
        "selected_names": selected_names,
        "preprocessor": preprocessor,
        "temp_df": temp_df,
        "text_vocabulary": vocabulary,
        "runtime_mi_drops": runtime_drops,
        "family_counts": family_counts,
        "family_lookup": family_lookup,
        "budget": actual_budget,
        "candidate_cols": candidate_cols,
        "restored_severity_cols": restored_severity_cols,
    }


def _build_fold_cache(
    container: DataContainer,
    scenario: str,
    train_idx: np.ndarray,
    grouping: np.ndarray,
    budget: Dict[str, int],
    seed: int,
    tag: str,
) -> List[Dict[str, Any]]:
    cache: List[Dict[str, Any]] = []
    for fold, (fold_train, fold_val) in enumerate(_nested_splits(train_idx, container.y, grouping, seed)):
        rep = _fit_scenario_representation(
            container,
            scenario,
            fold_train,
            fold_val,
            budget,
            f"{tag}_fold{fold}",
        )
        rep.update(
            {
                "y_train": container.y[fold_train],
                "y_eval": container.y[fold_val],
                "groups_train": grouping[fold_train],
            }
        )
        cache.append(rep)
    return cache


def _tune_model_nested(
    model_name: str, fold_cache: List[Dict[str, Any]], scenario: str, seed: int
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    candidates = _parameter_candidates(model_name, seed)
    if not candidates:
        raise RuntimeError(f"No hyperparameter candidates generated for {scenario}/{model_name}.")
    expected_folds = int(len(fold_cache))
    if expected_folds == 0:
        raise RuntimeError(f"No nested CV folds available for {scenario}/{model_name}.")
    rows: List[Dict[str, Any]] = []
    best_params: Optional[Dict[str, Any]] = None
    best_key = (-np.inf, -np.inf, -np.inf)
    for candidate_index, params in enumerate(candidates):
        fold_metrics: List[Dict[str, float]] = []
        failures: List[str] = []
        for fold, item in enumerate(fold_cache):
            try:
                model = _fit_model_matrix(
                    model_name,
                    params,
                    item["X_train"],
                    item["y_train"],
                    scenario,
                    seed + candidate_index * 100 + fold,
                )
                probability = _predict_model_matrix(model_name, model, item["X_eval"])
                prediction = np.argmax(probability, axis=1)
                fold_metrics.append(compute_all_metrics(item["y_eval"], prediction, probability))
            except Exception as exc:
                message = f"fold={fold}:{type(exc).__name__}:{exc}"
                failures.append(message)
                logger.log(
                    f"    Nested CV failed: {scenario}/{model_name}/candidate {candidate_index}/{message}",
                    level="WARNING",
                )
        base_row: Dict[str, Any] = {
            "Scenario": scenario,
            "Model": model_name,
            "candidate_index": candidate_index,
            "n_successful_folds": int(len(fold_metrics)),
            "n_required_folds": expected_folds,
            "failure_messages": " | ".join(failures),
            "params": json.dumps(params, sort_keys=True),
        }
        if len(fold_metrics) != expected_folds:
            rows.append(
                {
                    **base_row,
                    "status": "invalid_fold_failure",
                    "mean_qwk": np.nan,
                    "mean_mcc": np.nan,
                    "mean_balanced_accuracy": np.nan,
                }
            )
            continue
        means = {
            metric: float(np.mean([m.get(metric, np.nan) for m in fold_metrics]))
            for metric in ("qwk", "mcc", "balanced_accuracy")
        }
        if not all((np.isfinite(means[name]) for name in means)):
            rows.append(
                {
                    **base_row,
                    "status": "invalid_nonfinite_metric",
                    "mean_qwk": means["qwk"],
                    "mean_mcc": means["mcc"],
                    "mean_balanced_accuracy": means["balanced_accuracy"],
                }
            )
            continue
        key = _metric_key(means)
        rows.append(
            {
                **base_row,
                "status": "valid",
                "mean_qwk": means["qwk"],
                "mean_mcc": means["mcc"],
                "mean_balanced_accuracy": means["balanced_accuracy"],
            }
        )
        if key > best_key:
            best_key = key
            best_params = dict(params)
    table = pd.DataFrame(rows)
    if best_params is None:
        n_fold_failures = int((table.get("status") == "invalid_fold_failure").sum()) if not table.empty else 0
        n_metric_failures = (
            int((table.get("status") == "invalid_nonfinite_metric").sum()) if not table.empty else 0
        )
        raise RuntimeError(
            f"No valid hyperparameter candidate completed all {expected_folds} nested CV folds with finite selection metrics for {scenario}/{model_name}; fold_failures={n_fold_failures}, metric_failures={n_metric_failures}."
        )
    return (best_params, table)


def _choose_shared_hpo_budget_training_cv(
    container: DataContainer,
    train_idx: np.ndarray,
    grouping: np.ndarray,
    seed: int,
    analysis_label: str,
) -> Tuple[Dict[str, int], pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    best_budget: Optional[Dict[str, int]] = None
    best_key = (-np.inf, -np.inf, -np.inf)
    all_eligible = Config.HPO_BUDGET_ALL_ELIGIBLE
    n_hpo_candidates = len(container.scenario_cols.get("S3_Phenotype", []))
    if all_eligible and all_eligible < n_hpo_candidates:
        raise ValueError(
            f"Config.HPO_BUDGET_ALL_ELIGIBLE={all_eligible} is smaller than the {n_hpo_candidates} HPO candidate columns; raise it so the last grid step is uncapped."
        )
    logger.log(
        f"  Shared HPO-limit grid: {sorted(set(map(int, Config.HPO_BUDGET_CANDIDATES)))} ({all_eligible} = all eligible terms; {n_hpo_candidates} HPO candidate columns)"
    )
    for cap in sorted(set(map(int, Config.HPO_BUDGET_CANDIDATES))):
        budget = dict(Config.FEATURE_BUDGET)
        budget["hpo_binary"] = cap
        try:
            cache = _build_fold_cache(
                container,
                "S3_Phenotype",
                train_idx,
                grouping,
                budget,
                seed,
                f"{analysis_label}_S3_budget{cap}",
            )
            expected_folds = len(cache)
            if expected_folds == 0:
                raise RuntimeError("no nested folds were created")
            fold_metrics: List[Dict[str, float]] = []
            for fold, item in enumerate(cache):
                model = _fit_model_matrix(
                    "LightGBM",
                    dict(Config.S3_BUDGET_SELECTION_LGB_PARAMS),
                    item["X_train"],
                    item["y_train"],
                    "S3_Phenotype",
                    seed + fold,
                )
                probability = model.predict_proba(item["X_eval"])
                prediction = np.argmax(probability, axis=1)
                fold_metrics.append(compute_all_metrics(item["y_eval"], prediction, probability))
            means = {
                metric: float(np.mean([m.get(metric, np.nan) for m in fold_metrics]))
                for metric in ("qwk", "mcc", "balanced_accuracy")
            }
            if not all((np.isfinite(means[name]) for name in means)):
                raise RuntimeError(f"non-finite selection metrics: {means}")
            key = _metric_key(means)
            rows.append(
                {
                    "hpo_binary_budget": cap,
                    "status": "valid",
                    "n_successful_folds": int(len(fold_metrics)),
                    "n_required_folds": int(expected_folds),
                    "failure_message": "",
                    "mean_training_cv_qwk": means["qwk"],
                    "mean_training_cv_mcc": means["mcc"],
                    "mean_training_cv_balanced_accuracy": means["balanced_accuracy"],
                    "selection_partition": "training_only_nested_cv",
                    "selection_scenario": "S3_Phenotype",
                    "applied_to_scenarios": "S2_Mech_Pheno|S3_Phenotype",
                }
            )
            if key > best_key:
                best_key = key
                best_budget = budget
        except Exception as exc:
            logger.log(
                f"  Shared HPO training-CV budget {cap} failed: {type(exc).__name__}: {exc}",
                level="WARNING",
            )
            rows.append(
                {
                    "hpo_binary_budget": cap,
                    "status": "invalid",
                    "n_successful_folds": 0,
                    "n_required_folds": Config.NESTED_CV_FOLDS,
                    "failure_message": f"{type(exc).__name__}: {exc}",
                    "mean_training_cv_qwk": np.nan,
                    "mean_training_cv_mcc": np.nan,
                    "mean_training_cv_balanced_accuracy": np.nan,
                    "selection_partition": "training_only_nested_cv",
                    "selection_scenario": "S3_Phenotype",
                    "applied_to_scenarios": "S2_Mech_Pheno|S3_Phenotype",
                }
            )
    table = pd.DataFrame(rows)
    if best_budget is None:
        if analysis_label == "primary":
            table.to_csv(
                os.path.join(Config.OUTPUT_DIR, "results", "shared_hpo_budget_training_cv_selection.csv"),
                index=False,
            )
        raise RuntimeError(
            f"No HPO budget completed the S3 training-only nested CV selection for analysis {analysis_label}."
        )
    table["selected_for_s2_and_s3"] = (table["status"] == "valid") & (
        table["hpo_binary_budget"].astype(int) == int(best_budget["hpo_binary"])
    )
    if analysis_label == "primary":
        table.to_csv(
            os.path.join(Config.OUTPUT_DIR, "results", "shared_hpo_budget_training_cv_selection.csv"),
            index=False,
        )
    logger.log(f"  Shared S2/S3 HPO budget selected inside S3 training CV: {best_budget['hpo_binary']}")
    return (best_budget, table)


def _fit_final_candidate(
    model_name: str,
    params: Dict[str, Any],
    representation: Dict[str, Any],
    y_train: np.ndarray,
    y_val: np.ndarray,
    scenario: str,
    seed: int,
) -> Dict[str, Any]:
    model = _fit_model_matrix(model_name, params, representation["X_train"], y_train, scenario, seed)
    if model_name == "MultinomialLogisticRegression":
        fitted_logistic = _get_multinomial_logistic_estimator(model)
        realised_iterations = int(np.max(np.asarray(fitted_logistic.n_iter_)))
        logger.log(
            f"    Multinomial logistic converged in {realised_iterations} iterations (limit={fitted_logistic.max_iter}, tol={fitted_logistic.tol})."
        )
    probability = _predict_model_matrix(model_name, model, representation["X_eval"])
    prediction = np.argmax(probability, axis=1)
    metrics = compute_all_metrics(y_val, prediction, probability)
    return {
        "model_name": model_name,
        "model": model,
        "calibrated_model": None,
        "calibrated_models": None,
        "calibration_status": "not_attempted",
        "calibration_success": False,
        "calibration_method": "none",
        "calibration_method_attempted": "none",
        "calibration_failure_details": "",
        "val_metrics": metrics,
        "validation_selection_score": validation_selection_score(metrics),
        "best_params": params,
        "val_predictions": {"y_pred": prediction, "y_prob": probability},
        "importance": _importance_table(model_name, model, representation["selected_names"]),
        "test_evaluated": False,
        "eligible_for_headline_selection": True,
    }


def _select_candidate_lexicographically(
    candidates: Dict[str, Dict[str, Any]], model_order: List[str]
) -> Tuple[str, Dict[str, Any]]:
    order_lookup = {name: index for index, name in enumerate(model_order)}
    eligible = [
        (name, result)
        for name, result in candidates.items()
        if _metric_key(result.get("val_metrics", {}))[0] > -np.inf
    ]
    if not eligible:
        raise RuntimeError("No candidate has a finite validation QWK for model selection.")
    return sorted(
        eligible,
        key=lambda item: (
            -_finite_metric_or_neg_inf(item[1].get("val_metrics", {}), "qwk"),
            -_finite_metric_or_neg_inf(item[1].get("val_metrics", {}), "mcc"),
            -_finite_metric_or_neg_inf(item[1].get("val_metrics", {}), "balanced_accuracy"),
            order_lookup.get(item[0], 999),
        ),
    )[0]


def _audit_shared_hpo_selection(
    artifacts: Dict[str, Dict[str, Any]], analysis_label: str, write_csv: bool
) -> None:
    """Require S2 and S3 to use an identical direct-HPO representation."""
    scenario_a, scenario_b = Config.SHARED_HPO_BUDGET_SCENARIOS
    if scenario_a not in artifacts or scenario_b not in artifacts:
        raise RuntimeError(
            f"{analysis_label}: both {scenario_a} and {scenario_b} are required for the shared HPO fairness audit."
        )

    def direct_hpo_names(scenario: str) -> set:
        return {
            str(name)
            for name in artifacts[scenario].get("selected_names", [])
            if str(name).startswith("hpo_")
        }

    selected_a = direct_hpo_names(scenario_a)
    selected_b = direct_hpo_names(scenario_b)
    nested_a = artifacts[scenario_a].get("nested_hpo_sets", [])
    nested_b = artifacts[scenario_b].get("nested_hpo_sets", [])
    budget_a = int(artifacts[scenario_a]["selection_budget"]["hpo_binary"])
    budget_b = int(artifacts[scenario_b]["selection_budget"]["hpo_binary"])
    rows = [
        {
            "analysis": analysis_label,
            "feature": feature,
            "selected_in_s2": feature in selected_a,
            "selected_in_s3": feature in selected_b,
            "shared_selection": feature in selected_a and feature in selected_b,
            "hpo_budget_s2": budget_a,
            "hpo_budget_s3": budget_b,
        }
        for feature in sorted(selected_a | selected_b)
    ]
    if write_csv:
        pd.DataFrame(rows).to_csv(
            os.path.join(Config.OUTPUT_DIR, "results", "shared_hpo_feature_audit.csv"), index=False
        )
    nested_match = len(nested_a) == len(nested_b) and all(
        (set(left) == set(right) for left, right in zip(nested_a, nested_b))
    )
    if budget_a != budget_b or selected_a != selected_b or (not nested_match):
        only_a = sorted(selected_a - selected_b)[:10]
        only_b = sorted(selected_b - selected_a)[:10]
        raise RuntimeError(
            f"{analysis_label}: unfair HPO representation detected; budgets S2={budget_a}, S3={budget_b}, nested_match={nested_match}, only_S2={only_a}, only_S3={only_b}."
        )
    logger.log(
        f"  Shared HPO fairness audit passed: S2 and S3 both use {len(selected_a)} direct HPO features at budget {budget_a}; all {len(nested_a)} nested folds also match."
    )


def _run_workflow_for_split(
    container: DataContainer,
    development_idx: np.ndarray,
    test_idx: np.ndarray,
    grouping: np.ndarray,
    analysis_label: str,
    seed: int,
    primary: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    train_idx, val_idx = _split_development_train_validation(container, development_idx, grouping, seed)
    if set(grouping[train_idx]) & set(grouping[val_idx]):
        raise RuntimeError(f"{analysis_label}: train/validation grouping overlap detected.")
    if set(grouping[development_idx]) & set(grouping[test_idx]):
        raise RuntimeError(f"{analysis_label}: development/test grouping overlap detected.")
    if primary:
        write_split_summary(container, train_idx, val_idx, test_idx)
        _write_split_identifiers(container, train_idx, val_idx, test_idx, "primary")
    all_results: Dict[str, Any] = {}
    local_artifacts: Dict[str, Any] = {}
    selection_rows: List[Dict[str, Any]] = []
    candidate_names = _available_candidate_names()
    shared_hpo_budget, shared_hpo_budget_table = _choose_shared_hpo_budget_training_cv(
        container, train_idx, grouping, seed, analysis_label
    )
    for scenario in container.scenario_cols:
        logger.subsection(f"{analysis_label} scenario: {scenario}")
        all_results[scenario] = {"all": {}}
        budget = (
            dict(shared_hpo_budget)
            if scenario in Config.SHARED_HPO_BUDGET_SCENARIOS
            else dict(Config.FEATURE_BUDGET)
        )
        budget_table = pd.DataFrame()
        if scenario in Config.SHARED_HPO_BUDGET_SCENARIOS:
            budget_table = shared_hpo_budget_table.copy()
        fold_cache = _build_fold_cache(
            container,
            scenario,
            train_idx,
            grouping,
            budget,
            seed,
            f"{analysis_label}_{scenario}_nested",
        )
        final_representation = _fit_scenario_representation(
            container,
            scenario,
            train_idx,
            val_idx,
            budget,
            f"{analysis_label}_{scenario}_final",
        )
        y_train = container.y[train_idx]
        y_val = container.y[val_idx]
        candidate_results: Dict[str, Dict[str, Any]] = {}
        tuning_tables: List[pd.DataFrame] = []
        for model_index, model_name in enumerate(candidate_names):
            try:
                params, tuning_table = _tune_model_nested(
                    model_name, fold_cache, scenario, seed + 1000 * model_index
                )
                tuning_tables.append(tuning_table)
                result = _fit_final_candidate(
                    model_name,
                    params,
                    final_representation,
                    y_train,
                    y_val,
                    scenario,
                    seed + model_index,
                )
                candidate_results[model_name] = result
                all_results[scenario]["all"][model_name] = result
                if primary:
                    results_collector.add_feature_importance(f"{scenario}_{model_name}", result["importance"])
            except Exception as exc:
                logger.log(f"  {scenario}/{model_name} failed: {exc}", level="WARNING")
        if not candidate_results:
            raise RuntimeError(f"{analysis_label}/{scenario}: all candidate models failed before selection.")
        selected_name, selected_result = _select_candidate_lexicographically(
            candidate_results, candidate_names
        )
        for model_name, result in candidate_results.items():
            selected = model_name == selected_name
            result["selected_on_validation"] = selected
            result["selection_partition"] = "outer_validation"
            result["selection_rule"] = Config.MODEL_SELECTION_RULE
            selection_rows.append(
                {
                    "Analysis": analysis_label,
                    "Scenario": scenario,
                    "Model": model_name,
                    "selected_on_validation": selected,
                    "validation_qwk": result["val_metrics"].get("qwk", np.nan),
                    "validation_mcc": result["val_metrics"].get("mcc", np.nan),
                    "validation_balanced_accuracy": result["val_metrics"].get("balanced_accuracy", np.nan),
                    "selection_rule": Config.MODEL_SELECTION_RULE,
                    "test_evaluated": False,
                }
            )
        freeze_selected_model_and_calibrator(
            selected_result, final_representation["X_eval"], y_val, f"{analysis_label}:{scenario}"
        )
        if primary:
            final_representation["preprocessor"].save_audit(
                scenario, final_representation["runtime_mi_drops"]
            )
        local_artifacts[scenario] = {
            **final_representation,
            "sel_mask": final_representation["selected_mask"],
            "sel_names": final_representation["selected_names"],
            "X_val_full": final_representation["X_eval_full"],
            "X_val": final_representation["X_eval"],
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "y_train": y_train,
            "y_val": y_val,
            "groups_train": grouping[train_idx],
            "selected_model": selected_name,
            "selection_budget": budget,
            "budget_selection_table": budget_table,
            "nested_hpo_sets": [
                [str(name) for name in fold_item.get("selected_names", []) if str(name).startswith("hpo_")]
                for fold_item in fold_cache
            ],
            "tuning_table": pd.concat(tuning_tables, ignore_index=True) if tuning_tables else pd.DataFrame(),
        }
        if primary and (not local_artifacts[scenario]["tuning_table"].empty):
            local_artifacts[scenario]["tuning_table"].to_csv(
                os.path.join(Config.OUTPUT_DIR, "results", f"nested_cv_tuning_{scenario}.csv"),
                index=False,
            )
    _audit_shared_hpo_selection(local_artifacts, analysis_label, write_csv=primary)
    y_test = container.y[test_idx]
    for scenario, artifacts in local_artifacts.items():
        selected_name = artifacts["selected_model"]
        selected_result = all_results[scenario]["all"][selected_name]
        X_test_full = artifacts["preprocessor"].transform(artifacts["temp_df"], test_idx)
        X_test = X_test_full[:, artifacts["selected_mask"]]
        evaluate_frozen_selected_model_on_test(
            selected_result, X_test, y_test, f"{analysis_label}:{scenario}"
        )
        selected_result["test_groups"] = np.asarray(grouping[test_idx])
        selected_result["test_gene_groups"] = np.asarray(container.groups[test_idx])
        artifacts["X_test_full"] = X_test_full
        artifacts["X_test"] = X_test
        artifacts["y_test"] = y_test
        for row in reversed(selection_rows):
            if row["Scenario"] == scenario and row["Model"] == selected_name:
                row["test_evaluated"] = True
                break
        if primary:
            results_collector.add_result(scenario, "all", selected_name, selected_result["metrics"])
            results_collector.add_predictions(
                f"{scenario}_{selected_name}",
                y_test,
                selected_result["predictions"]["y_pred"],
                selected_result["predictions"]["y_prob"],
            )
    if primary:
        write_split_summary(container, train_idx, val_idx, test_idx, include_test_labels=True)
    return (
        all_results,
        {
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "selection_rows": selection_rows,
            "artifacts": local_artifacts,
        },
    )


def run_additional_generalisation_evaluations(container: DataContainer) -> pd.DataFrame:
    logger.section("ADDITIONAL GENERALISATION EVALUATIONS")
    metadata = container.metadata.reset_index(drop=True)
    grouping_modes = {
        "disease_disjoint": metadata["mondo_id"].astype(str).to_numpy(),
        "gene_and_disease_disjoint": _gene_disease_component_groups(container),
    }
    summary_rows: List[Dict[str, Any]] = []
    all_idx = np.arange(len(container.y), dtype=int)
    for mode in Config.ADDITIONAL_GENERALISATION_MODES:
        grouping = grouping_modes[mode]
        try:
            development_idx, test_idx = _optimised_group_holdout(
                all_idx, container.y, grouping, Config.TEST_SIZE, RANDOM_SEED + 500
            )
            results, details = _run_workflow_for_split(
                container,
                development_idx,
                test_idx,
                grouping,
                mode,
                RANDOM_SEED + 500,
                primary=False,
            )
            _generalisation_results[mode] = {"results": results, "details": details}
            _write_split_identifiers(
                container, details["train_idx"], details["val_idx"], details["test_idx"], mode
            )
            for scenario, model_name in _pick_best_model_per_scenario(results).items():
                result = results[scenario]["all"][model_name]
                metrics = result.get("metrics", {})
                summary_rows.append(
                    {
                        "generalisation_mode": mode,
                        "Scenario": scenario,
                        "Model": model_name,
                        "mcc": metrics.get("mcc", np.nan),
                        "qwk": metrics.get("qwk", np.nan),
                        "balanced_accuracy": metrics.get("balanced_accuracy", np.nan),
                        "ranked_probability_score": metrics.get("ranked_probability_score", np.nan),
                        "n_test": len(details["test_idx"]),
                    }
                )
        except Exception as exc:
            logger.log(f"  {mode} evaluation skipped/failed: {exc}", level="WARNING")
    dataframe = pd.DataFrame(summary_rows)
    dataframe.to_csv(
        os.path.join(Config.OUTPUT_DIR, "results", "generalisation_evaluations.csv"), index=False
    )
    return dataframe


def run_experiments(container: DataContainer):
    logger.section("RUNNING NESTED MANUSCRIPT-COMPLETE EXPERIMENTS")
    _fold_artifacts.clear()
    _test_evaluation_registry.clear()
    _high_mi_audit_records.clear()
    _generalisation_results.clear()
    all_results, details = _run_workflow_for_split(
        container,
        container.train_idx,
        container.test_idx,
        container.groups,
        "primary",
        RANDOM_SEED,
        primary=True,
    )
    _fold_artifacts.update(details["artifacts"])
    selection_df = pd.DataFrame(details["selection_rows"])
    selection_df.to_csv(
        os.path.join(Config.OUTPUT_DIR, "results", "model_selection_validation.csv"), index=False
    )
    results_collector.save(os.path.join(Config.OUTPUT_DIR, "results", "all_results.csv"))
    with open(os.path.join(Config.OUTPUT_DIR, "results", "all_results_full.pkl"), "wb") as handle:
        pickle.dump(all_results, handle)
    run_additional_generalisation_evaluations(container)
    return all_results


def train_naive_baselines(container: DataContainer) -> Dict[str, Dict]:
    logger.subsection("Training naive baselines from the primary training partition")
    if _fold_artifacts:
        first_artifact = next(iter(_fold_artifacts.values()))
        train_idx = np.asarray(first_artifact["train_idx"])
        test_idx = np.asarray(first_artifact["test_idx"])
    else:
        train_idx = np.asarray(container.train_idx)
        test_idx = np.asarray(container.test_idx)
    y_train = container.y[train_idx]
    y_test = container.y[test_idx]
    baselines: Dict[str, Dict[str, Any]] = {}
    majority = int(np.bincount(y_train, minlength=Config.NUM_CLASSES).argmax())
    majority_pred = np.full(len(y_test), majority, dtype=int)
    majority_prob = np.zeros((len(y_test), Config.NUM_CLASSES), dtype=float)
    majority_prob[:, majority] = 1.0
    baselines["MajorityClass"] = {
        "metrics": compute_all_metrics(y_test, majority_pred, majority_prob),
        "y_pred": majority_pred,
        "y_prob": majority_prob,
        "y_true": y_test,
    }
    class_probabilities = np.bincount(y_train, minlength=Config.NUM_CLASSES).astype(float)
    class_probabilities /= max(class_probabilities.sum(), 1.0)
    rng = np.random.RandomState(RANDOM_SEED)
    random_pred = rng.choice(Config.NUM_CLASSES, size=len(y_test), p=class_probabilities)
    random_prob = np.tile(class_probabilities, (len(y_test), 1))
    baselines["StratifiedRandom"] = {
        "metrics": compute_all_metrics(y_test, random_pred, random_prob),
        "y_pred": random_pred,
        "y_prob": random_prob,
        "y_true": y_test,
    }
    ordinal_mean = int(round(np.clip(float(y_train.mean()), 0, Config.NUM_CLASSES - 1)))
    ordinal_pred = np.full(len(y_test), ordinal_mean, dtype=int)
    ordinal_prob = np.zeros((len(y_test), Config.NUM_CLASSES), dtype=float)
    ordinal_prob[:, ordinal_mean] = 1.0
    baselines["OrdinalMean"] = {
        "metrics": compute_all_metrics(y_test, ordinal_pred, ordinal_prob),
        "y_pred": ordinal_pred,
        "y_prob": ordinal_prob,
        "y_true": y_test,
    }
    rows = []
    for name, result in baselines.items():
        row = {"Baseline": name, "training_partition": "primary_train_only"}
        row.update({k: v for k, v in result["metrics"].items() if not isinstance(v, str)})
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(Config.OUTPUT_DIR, "results", "naive_baselines.csv"), index=False)
    return baselines
