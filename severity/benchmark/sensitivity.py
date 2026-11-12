"""Sensitivity analyses and robustness checks.

All of these use the validation pairs, except the five further gene-disjoint splits (seeds
43-47), which refit the fixed configurations and report test metrics on their own test sets:

* further gene-disjoint splits (Extended Data Fig. 2a; Supplementary Table 9)
* minimum mutual-information thresholds (Extended Data Fig. 2b; Supplementary Table 10)
* gene-block permutation tests, equation (11) (Extended Data Fig. 2c; Supplementary Table 12)
* HPO term limits (Extended Data Fig. 2d; Supplementary Table 2)
* class weighting, oversampling or no adjustment (Extended Data Fig. 2e; Supplementary Table 11)
* severity-stating HPO terms returned to the candidate pool (Supplementary Table 13)
* diseases shared between the primary training and test sets (Supplementary Table 15)
"""

import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .calibration import _fit_selected_calibration, _predict_calibrated_candidate
from .config import RANDOM_SEED, Config
from .evaluation import _benjamini_hochberg
from .metrics import compute_all_metrics
from .models import (
    _available_candidate_names,
    _fit_model_matrix,
    _predict_model_matrix,
    _train_candidate_dispatch,
)
from .partition import (
    _optimised_group_holdout,
    _primary_partition_indices,
    _split_development_train_validation,
)
from .runtime import DataContainer, _fold_artifacts, logger
from .selection import (
    _feature_provenance_for_mi,
    select_features_mi,
    select_features_mi_grouped,
    uses_grouped_mi,
)
from .severity_terms import is_severity_hpo_column
from .workflow import (
    _fit_final_candidate,
    _fit_scenario_representation,
    _pick_best_model_per_scenario,
    _select_candidate_lexicographically,
)


def run_mi_threshold_sensitivity(
    container: DataContainer, all_results: Dict, thresholds: Optional[List[float]] = None
) -> pd.DataFrame:
    """Validation-only sensitivity to the minimum positive-MI selection threshold."""
    if thresholds is None:
        thresholds = Config.MI_SENSITIVITY_THRESHOLDS
    logger.section(f"Minimum-MI selection sensitivity on validation ({len(thresholds)} values)")
    best_models = _pick_best_model_per_scenario(all_results)
    rows: List[Dict[str, Any]] = []
    for scenario in [value for value in ["S2_Mech_Pheno", "S3_Phenotype"] if value in best_models]:
        model_name = best_models[scenario]
        artifacts = _fold_artifacts.get(scenario)
        if artifacts is None:
            continue
        fixed_params = all_results[scenario]["all"][model_name].get("best_params")
        for threshold in thresholds:
            try:
                X_train, selected_names, _, mask, _, _ = select_features_mi_grouped(
                    artifacts["X_train_full"],
                    artifacts["y_train"],
                    artifacts["feature_names_full"],
                    family_lookup=artifacts.get("family_lookup", container.feature_family_lookup),
                    budget=artifacts.get("selection_budget", Config.FEATURE_BUDGET),
                    scenario_name=f"{scenario}_mithr_{float(threshold):.4f}",
                    mi_threshold=float(threshold),
                )
                X_validation = artifacts["X_val_full"][:, mask]
                result = _train_candidate_dispatch(
                    model_name,
                    X_train,
                    artifacts["y_train"],
                    artifacts["groups_train"],
                    X_validation,
                    artifacts["y_val"],
                    selected_names,
                    scenario,
                    fixed_params=fixed_params,
                )
                if result is None:
                    continue
                metrics = result["val_metrics"]
                rows.append(
                    {
                        "Scenario": scenario,
                        "Model": model_name,
                        "evaluation_partition": "validation_only",
                        "minimum_mi_threshold": float(threshold),
                        "n_features_selected": int(mask.sum()),
                        "mcc": metrics.get("mcc", np.nan),
                        "qwk": metrics.get("qwk", np.nan),
                        "balanced_accuracy": metrics.get("balanced_accuracy", np.nan),
                    }
                )
                logger.log(
                    f"  {scenario} minimum MI={float(threshold):.4f}: validation MCC={metrics.get('mcc', 0):.3f}, QWK={metrics.get('qwk', 0):.3f}"
                )
            except Exception as exc:
                logger.log(f"  {scenario} minimum MI={threshold} failed: {exc}", level="WARNING")
    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(os.path.join(Config.OUTPUT_DIR, "results", "mi_threshold_sensitivity.csv"), index=False)
    return dataframe


def run_budget_sensitivity(
    container: DataContainer, all_results: Dict, hpo_budgets: Optional[List[int]] = None
) -> pd.DataFrame:
    """Validation-only sensitivity to a shared direct-HPO feature budget."""
    if hpo_budgets is None:
        hpo_budgets = list(Config.HPO_BUDGET_CANDIDATES)
    hpo_budgets = sorted({int(value) for value in hpo_budgets if int(value) > 0})
    logger.section(f"Shared HPO-budget sensitivity on validation ({len(hpo_budgets)} values)")
    best_models = _pick_best_model_per_scenario(all_results)
    rows: List[Dict[str, Any]] = []
    for scenario in [s for s in ["S2_Mech_Pheno", "S3_Phenotype"] if s in best_models]:
        model_name = best_models[scenario]
        artifacts = _fold_artifacts.get(scenario)
        if artifacts is None:
            continue
        X_train_full = artifacts["X_train_full"]
        X_val_full = artifacts["X_val_full"]
        feature_names = artifacts["feature_names_full"]
        y_train = artifacts["y_train"]
        y_val = artifacts["y_val"]
        groups_train = artifacts["groups_train"]
        fixed_params = all_results[scenario]["all"][model_name].get("best_params")
        for hpo_budget in hpo_budgets:
            candidate_budget = dict(Config.FEATURE_BUDGET)
            candidate_budget["hpo_binary"] = hpo_budget
            try:
                X_train, selected_names, _, mask, _, family_counts = select_features_mi_grouped(
                    X_train_full,
                    y_train,
                    feature_names,
                    family_lookup=artifacts.get("family_lookup", container.feature_family_lookup),
                    budget=candidate_budget,
                    scenario_name=f"{scenario}_hpobudget_{hpo_budget}",
                )
                X_val = X_val_full[:, mask]
                result = _train_candidate_dispatch(
                    model_name,
                    X_train,
                    y_train,
                    groups_train,
                    X_val,
                    y_val,
                    selected_names,
                    scenario,
                    fixed_params=fixed_params,
                )
                if result is None:
                    continue
                metrics = result["val_metrics"]
                rows.append(
                    {
                        "Scenario": scenario,
                        "Model": model_name,
                        "evaluation_partition": "validation_only",
                        "hpo_binary_budget": hpo_budget,
                        "n_selected": int(mask.sum()),
                        "mcc": metrics.get("mcc", np.nan),
                        "qwk": metrics.get("qwk", np.nan),
                        "balanced_accuracy": metrics.get("balanced_accuracy", np.nan),
                        "family_counts": json.dumps(family_counts),
                    }
                )
                logger.log(
                    f"  {scenario}/{model_name} HPO budget {hpo_budget}: validation MCC={metrics.get('mcc', 0):.3f}, QWK={metrics.get('qwk', 0):.3f}"
                )
            except Exception as exc:
                logger.log(
                    f"  {scenario}/{model_name} HPO budget {hpo_budget} failed: {exc}",
                    level="WARNING",
                )
    dataframe = pd.DataFrame(rows)
    output_path = os.path.join(Config.OUTPUT_DIR, "results", "budget_sensitivity.csv")
    dataframe.to_csv(output_path, index=False)
    logger.log(f"  Wrote {output_path}")
    return dataframe


DISEASE_OVERLAP_MIN_GENES = 10


def run_disease_overlap_audit(container: DataContainer) -> pd.DataFrame:
    """Quantify MONDO-level overlap between the locked test set and training.

    The primary split is gene-disjoint but not disease-disjoint, so the same
    MONDO identifier can appear on both sides. This audit makes the extent of
    that overlap explicit. It reads only the stored split indices and metadata;
    nothing is refit and no model is evaluated.
    """
    logger.subsection("Disease-overlap audit across primary partitions")
    output_path = os.path.join(Config.OUTPUT_DIR, "results", "disease_overlap_audit.csv")
    partitions = _primary_partition_indices(container)
    metadata = container.metadata.reset_index(drop=True)
    mondo = metadata["mondo_id"].astype(str).to_numpy()
    genes = metadata["gene_symbol"].astype(str).to_numpy()
    names = (
        metadata["disease_name"].astype(str).to_numpy()
        if "disease_name" in metadata.columns
        else np.array([""] * len(metadata))
    )
    if "target_severity" in metadata.columns:
        labels = metadata["target_severity"].astype(str).to_numpy()
    else:
        inverse = {value: key for key, value in Config.TARGET_MAP.items()}
        labels = np.array([inverse.get(int(value), "") for value in container.y])
    partition_of = np.array(["unassigned"] * len(metadata), dtype=object)
    for partition, idx in partitions.items():
        partition_of[idx] = partition
    train_mondo = set(mondo[partitions["train"]])
    validation_mondo = set(mondo[partitions["validation"]])
    test_mondo = set(mondo[partitions["test"]])
    test_idx = partitions["test"]
    n_test_pairs = int(len(test_idx))
    test_pairs_in_train = int(np.sum([m in train_mondo for m in mondo[test_idx]]))
    percent_test_pairs_in_train = 100.0 * test_pairs_in_train / n_test_pairs if n_test_pairs else np.nan
    hpo_columns = [
        column for column in getattr(container, "hpo_binary_cols", []) if column in container.full_df.columns
    ]
    gene_counts = (
        pd.DataFrame({"mondo_id": mondo, "gene_symbol": genes}).groupby("mondo_id")["gene_symbol"].nunique()
    )
    reportable_diseases = sorted(gene_counts.index[gene_counts >= DISEASE_OVERLAP_MIN_GENES].astype(str))
    hpo_present_by_disease: Dict[str, int] = {}
    if hpo_columns and reportable_diseases:
        reportable_mask = np.isin(mondo, reportable_diseases)
        hpo_frame = (
            container.full_df.loc[reportable_mask, hpo_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )
        present = hpo_frame.to_numpy() > 0
        subset_mondo = mondo[reportable_mask]
        for disease in reportable_diseases:
            hpo_present_by_disease[disease] = int(present[subset_mondo == disease].any(axis=0).sum())
    elif not hpo_columns:
        logger.log(
            "  No direct HPO binary columns available; per-disease HPO counts are reported as 0.",
            level="WARNING",
        )
    rows: List[Dict[str, Any]] = []

    def summary_row(metric: str, value: Any) -> Dict[str, Any]:
        return {"record_type": "summary", "metric": metric, "value": value}

    rows.append(summary_row("n_pairs_total", int(len(metadata))))
    rows.append(summary_row("n_train_pairs", int(len(partitions["train"]))))
    rows.append(summary_row("n_validation_pairs", int(len(partitions["validation"]))))
    rows.append(summary_row("n_test_pairs", n_test_pairs))
    rows.append(summary_row("n_test_pairs_with_mondo_id_also_in_train", test_pairs_in_train))
    rows.append(
        summary_row(
            "percent_test_pairs_with_mondo_id_also_in_train",
            round(float(percent_test_pairs_in_train), 4),
        )
    )
    rows.append(summary_row("n_distinct_mondo_total", int(len(set(mondo)))))
    rows.append(summary_row("n_distinct_mondo_train", int(len(train_mondo))))
    rows.append(summary_row("n_distinct_mondo_validation", int(len(validation_mondo))))
    rows.append(summary_row("n_distinct_mondo_test", int(len(test_mondo))))
    rows.append(summary_row("n_distinct_mondo_in_test_and_train", int(len(test_mondo & train_mondo))))
    rows.append(
        summary_row("n_distinct_mondo_in_test_and_validation", int(len(test_mondo & validation_mondo)))
    )
    rows.append(
        summary_row(
            "percent_distinct_test_mondo_also_in_train",
            round(100.0 * len(test_mondo & train_mondo) / len(test_mondo), 4) if test_mondo else np.nan,
        )
    )
    rows.append(summary_row("min_genes_for_per_disease_table", int(DISEASE_OVERLAP_MIN_GENES)))
    rows.append(summary_row("split_policy", "gene-disjoint; MONDO identifiers are not disjoint"))
    disease_rows: List[Dict[str, Any]] = []
    for disease in reportable_diseases:
        mask = mondo == disease
        n_genes = int(len(set(genes[mask])))
        disease_labels = pd.Series(labels[mask])
        label_counts = disease_labels.value_counts()
        disease_names = pd.Series([n for n in names[mask] if n and n != "nan"])
        disease_rows.append(
            {
                "record_type": "disease",
                "mondo_id": disease,
                "disease_name": disease_names.mode().iloc[0] if not disease_names.empty else "",
                "n_genes": n_genes,
                "n_pairs": int(mask.sum()),
                "n_direct_hpo_terms": int(hpo_present_by_disease.get(disease, 0)),
                "severity_label": str(label_counts.index[0]) if len(label_counts) else "",
                "severity_label_is_unique": bool(len(label_counts) == 1),
                "severity_labels_observed": "|".join(
                    (f"{label}:{count}" for label, count in label_counts.items())
                ),
                "n_train_pairs": int(np.sum(partition_of[mask] == "train")),
                "n_validation_pairs": int(np.sum(partition_of[mask] == "validation")),
                "n_test_pairs": int(np.sum(partition_of[mask] == "test")),
                "spans_train_and_test": bool(
                    np.any(partition_of[mask] == "train") and np.any(partition_of[mask] == "test")
                ),
            }
        )
    disease_rows.sort(key=lambda item: (-item["n_genes"], item["mondo_id"]))
    rows.extend(disease_rows)
    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(output_path, index=False)
    logger.log(
        f"  {test_pairs_in_train} of {n_test_pairs} locked-test pairs ({percent_test_pairs_in_train:.1f}%) carry a MONDO identifier that also appears in training; distinct MONDO ids train={len(train_mondo)}, validation={len(validation_mondo)}, test={len(test_mondo)}."
    )
    logger.log(
        f"  Per-disease table covers {len(disease_rows)} diseases with at least {DISEASE_OVERLAP_MIN_GENES} genes."
    )
    return dataframe


def run_imbalance_sensitivity(container: DataContainer, all_results: Dict) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for scenario, model_name in _pick_best_model_per_scenario(all_results).items():
        artifacts = _fold_artifacts[scenario]
        params = all_results[scenario]["all"][model_name].get("best_params", {})
        for strategy in ("sample_weight", "resampling", "none"):
            try:
                model = _fit_model_matrix(
                    model_name,
                    params,
                    artifacts["X_train"],
                    artifacts["y_train"],
                    scenario,
                    RANDOM_SEED + 900,
                    imbalance_strategy=strategy,
                )
                probability = _predict_model_matrix(model_name, model, artifacts["X_eval"])
                prediction = np.argmax(probability, axis=1)
                metrics = compute_all_metrics(artifacts["y_val"], prediction, probability)
                rows.append(
                    {
                        "Scenario": scenario,
                        "Model": model_name,
                        "imbalance_strategy": strategy,
                        "evaluation_partition": "validation_only",
                        "qwk": metrics.get("qwk"),
                        "mcc": metrics.get("mcc"),
                        "balanced_accuracy": metrics.get("balanced_accuracy"),
                        "ranked_probability_score": metrics.get("ranked_probability_score"),
                    }
                )
            except Exception as exc:
                logger.log(f"  Imbalance sensitivity failed {scenario}/{strategy}: {exc}", level="WARNING")
    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(os.path.join(Config.OUTPUT_DIR, "results", "imbalance_sensitivity.csv"), index=False)
    return dataframe


def _gene_block_permutation_capacity(groups: np.ndarray) -> float:
    groups = np.asarray(groups)
    group_sizes = Counter(groups.tolist())
    genes_by_size: Dict[int, List[Any]] = defaultdict(list)
    for gene, size in group_sizes.items():
        genes_by_size[int(size)].append(gene)
    movable_rows = sum((size * len(genes) for size, genes in genes_by_size.items() if len(genes) >= 2))
    return float(movable_rows / max(len(groups), 1))


def permute_gene_label_blocks(
    y: np.ndarray, groups: np.ndarray, rng: np.random.RandomState
) -> Tuple[np.ndarray, float]:
    """Exchange complete gene label profiles among genes of equal cluster size."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    if len(y) != len(groups):
        raise ValueError("y and groups must have identical lengths.")
    indices_by_gene: Dict[Any, np.ndarray] = {
        gene: np.flatnonzero(groups == gene) for gene in np.unique(groups)
    }
    genes_by_size: Dict[int, List[Any]] = defaultdict(list)
    for gene, indices in indices_by_gene.items():
        genes_by_size[len(indices)].append(gene)
    permuted = y.copy()
    movable_rows = 0
    for cluster_size, genes in sorted(genes_by_size.items()):
        if len(genes) < 2:
            continue
        donor_order = rng.permutation(len(genes))
        for recipient_position, donor_position in enumerate(donor_order):
            recipient_gene = genes[recipient_position]
            donor_gene = genes[int(donor_position)]
            recipient_indices = indices_by_gene[recipient_gene]
            donor_indices = indices_by_gene[donor_gene]
            if len(recipient_indices) != cluster_size or len(donor_indices) != cluster_size:
                raise RuntimeError("Gene cluster-size stratum changed unexpectedly.")
            permuted[recipient_indices] = y[donor_indices]
            movable_rows += cluster_size
    if not np.array_equal(
        np.bincount(y.astype(int), minlength=Config.NUM_CLASSES),
        np.bincount(permuted.astype(int), minlength=Config.NUM_CLASSES),
    ):
        raise RuntimeError("Gene-block permutation did not preserve global class counts.")
    return (permuted, float(movable_rows / max(len(y), 1)))


def run_permutation_test(
    container: DataContainer, all_results: Dict, n_permutations: Optional[int] = None
) -> pd.DataFrame:
    """Conditional validation permutation test using complete gene label blocks.

    Feature selection and candidate-family selection are repeated for every
    permutation. Hyperparameters remain fixed at their nested-CV-selected values,
    so this remains a conditional rather than a fully re-tuned pipeline test.
    """
    n_permutations = int(n_permutations or Config.N_PERMUTATIONS)
    rows: List[Dict[str, Any]] = []
    candidate_names = _available_candidate_names()
    selected_by_scenario = _pick_best_model_per_scenario(all_results)
    for scenario, observed_name in selected_by_scenario.items():
        artifacts = _fold_artifacts[scenario]
        observed = float(all_results[scenario]["all"][observed_name]["val_metrics"].get("mcc", np.nan))
        groups_train = np.asarray(artifacts["groups_train"])
        movable_fraction = _gene_block_permutation_capacity(groups_train)
        if movable_fraction < float(Config.PERMUTATION_MIN_MOVABLE_FRACTION):
            logger.log(
                f"  Skipping permutation test for {scenario}: only {movable_fraction:.1%} of training rows can move under exact equal-size gene-block permutation.",
                level="WARNING",
            )
            continue
        parameters_by_model = {
            name: all_results[scenario]["all"][name].get("best_params", {})
            for name in candidate_names
            if name in all_results[scenario]["all"]
        }
        rng = np.random.RandomState(RANDOM_SEED + 700)
        null_statistics: List[float] = []
        failures = 0
        observed_movable_fractions: List[float] = []
        for permutation_index in range(n_permutations):
            permuted_y, realised_movable = permute_gene_label_blocks(artifacts["y_train"], groups_train, rng)
            observed_movable_fractions.append(realised_movable)
            try:
                if uses_grouped_mi(scenario):
                    X_train, _, _, mask, _, _ = select_features_mi_grouped(
                        artifacts["X_train_full"],
                        permuted_y,
                        artifacts["feature_names_full"],
                        family_lookup=artifacts.get("family_lookup", container.feature_family_lookup),
                        budget=artifacts["selection_budget"],
                        scenario_name=f"{scenario}_perm{permutation_index}",
                    )
                else:
                    mechanistic = "S1_Mechanistic" in scenario
                    X_train, _, _, mask, _ = select_features_mi(
                        artifacts["X_train_full"],
                        permuted_y,
                        artifacts["feature_names_full"],
                        mi_threshold=Config.MI_THRESHOLD_MECHANISTIC
                        if mechanistic
                        else Config.MI_THRESHOLD_DEFAULT,
                        min_features=Config.MI_MIN_FEATURES_MECHANISTIC
                        if mechanistic
                        else Config.MI_MIN_FEATURES_DEFAULT,
                        scenario_name=f"{scenario}_perm{permutation_index}",
                        family_lookup=artifacts.get("family_lookup", container.feature_family_lookup),
                    )
                X_validation = artifacts["X_eval_full"][:, mask]
                permutation_candidates: Dict[str, Dict[str, Any]] = {}
                for model_index, model_name in enumerate(parameters_by_model):
                    model = _fit_model_matrix(
                        model_name,
                        parameters_by_model[model_name],
                        X_train,
                        permuted_y,
                        scenario,
                        RANDOM_SEED + permutation_index * 10 + model_index,
                    )
                    probability = _predict_model_matrix(model_name, model, X_validation)
                    prediction = np.argmax(probability, axis=1)
                    metrics = compute_all_metrics(artifacts["y_val"], prediction, probability)
                    permutation_candidates[model_name] = {"val_metrics": metrics}
                _, selected = _select_candidate_lexicographically(
                    permutation_candidates, list(parameters_by_model)
                )
                statistic = float(selected["val_metrics"].get("mcc", np.nan))
                if np.isfinite(statistic):
                    null_statistics.append(statistic)
                else:
                    failures += 1
            except Exception as exc:
                failures += 1
                if failures <= 3:
                    logger.log(
                        f"  {scenario} permutation {permutation_index} failed: {exc}",
                        level="WARNING",
                    )
        values = np.asarray(null_statistics, dtype=float)
        if not len(values):
            logger.log(f"  No successful gene-block permutations for {scenario}.", level="WARNING")
            continue
        empirical_p = (int(np.sum(values >= observed)) + 1) / (len(values) + 1)
        rows.append(
            {
                "Scenario": scenario,
                "Observed_Model": observed_name,
                "permutation_mode": Config.PERMUTATION_MODE,
                "permutation_unit": "complete_gene_label_profile_within_equal_cluster_size",
                "evaluation_partition": "validation_only",
                "observed_mcc": observed,
                "n_permutations_requested": n_permutations,
                "n_permutations_successful": len(values),
                "n_permutations_failed": failures,
                "movable_row_fraction": float(np.mean(observed_movable_fractions)),
                "permuted_mcc_mean": float(values.mean()),
                "permuted_mcc_std": float(values.std()),
                "empirical_p_value": empirical_p,
                "hyperparameters_retuned_per_permutation": False,
            }
        )
    dataframe = pd.DataFrame(rows)
    if not dataframe.empty:
        dataframe["bh_adjusted_p_value"] = _benjamini_hochberg(
            dataframe["empirical_p_value"].to_numpy(dtype=float)
        )
    dataframe.to_csv(os.path.join(Config.OUTPUT_DIR, "results", "permutation_test.csv"), index=False)
    return dataframe


def run_split_sensitivity_fixed_configuration(
    container: DataContainer, all_results: Dict, n_seeds: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extended Data Fig. 2a: the selected model of each feature set, with its tuned settings,
    refitted on new gene-disjoint splits. Model choice and hyperparameters are not re-tuned."""
    n_seeds = int(n_seeds or Config.N_SEEDS_SENSITIVITY)
    rows: List[Dict[str, Any]] = []
    selected = _pick_best_model_per_scenario(all_results)
    all_idx = np.arange(len(container.y), dtype=int)
    first_seed = RANDOM_SEED + int(Config.SPLIT_SENSITIVITY_SEED_OFFSET)
    primary_test_idx = np.sort(np.asarray(container.test_idx, dtype=int))
    for seed in range(first_seed, first_seed + n_seeds):
        development_idx, test_idx = _optimised_group_holdout(
            all_idx, container.y, container.groups, Config.TEST_SIZE, seed
        )
        if np.array_equal(np.sort(np.asarray(test_idx, dtype=int)), primary_test_idx):
            raise RuntimeError(
                f"Split-sensitivity seed {seed} reproduced the primary test holdout; refusing repeated primary-test evaluation."
            )
        train_idx, val_idx = _split_development_train_validation(
            container, development_idx, container.groups, seed
        )
        for scenario, model_name in selected.items():
            try:
                budget = _fold_artifacts[scenario].get("selection_budget", Config.FEATURE_BUDGET)
                rep = _fit_scenario_representation(
                    container,
                    scenario,
                    train_idx,
                    val_idx,
                    budget,
                    f"{scenario}_splitseed{seed}",
                )
                params = all_results[scenario]["all"][model_name].get("best_params", {})
                result = _fit_final_candidate(
                    model_name,
                    params,
                    rep,
                    container.y[train_idx],
                    container.y[val_idx],
                    scenario,
                    seed,
                )
                _fit_selected_calibration(
                    result, rep["X_eval"], container.y[val_idx], f"splitseed{seed}:{scenario}"
                )
                X_test_full = rep["preprocessor"].transform(rep["temp_df"], test_idx)
                X_test = X_test_full[:, rep["selected_mask"]]
                probability = _predict_calibrated_candidate(result, X_test)
                prediction = np.argmax(probability, axis=1)
                metrics = compute_all_metrics(container.y[test_idx], prediction, probability)
                rows.append(
                    {
                        "Scenario": scenario,
                        "Model": model_name,
                        "seed": seed,
                        "analysis_type": "split_sensitivity_fixed_configuration",
                        "mcc": metrics.get("mcc"),
                        "qwk": metrics.get("qwk"),
                        "balanced_accuracy": metrics.get("balanced_accuracy"),
                        "ranked_probability_score": metrics.get("ranked_probability_score"),
                    }
                )
            except Exception as exc:
                logger.log(f"  Split sensitivity failed {scenario}/seed{seed}: {exc}", level="WARNING")
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(
        os.path.join(Config.OUTPUT_DIR, "results", "split_sensitivity_fixed_configuration.csv"),
        index=False,
    )
    summary_rows: List[Dict[str, Any]] = []
    if not per_seed.empty:
        for (scenario, model_name), group in per_seed.groupby(["Scenario", "Model"]):
            row = {"Scenario": scenario, "Model": model_name, "n_seeds": len(group)}
            for metric in ("mcc", "qwk", "balanced_accuracy", "ranked_probability_score"):
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"{metric}_std"] = float(group[metric].std())
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        os.path.join(Config.OUTPUT_DIR, "results", "split_sensitivity_fixed_configuration_summary.csv"),
        index=False,
    )
    return (per_seed, summary)


def run_severity_hpo_term_sensitivity(container: DataContainer, all_results: Dict) -> pd.DataFrame:
    """Validation-only ablation of the explicit severity-related HPO terms.

    Two arms are refitted on the same training indices with the same selected
    model family and the same nested-CV hyperparameters, differing only in
    whether the severity-naming HPO terms are admitted to the feature space:

        retained arm  drop_severity_hpo_terms=False  (terms available)
        removed arm   drop_severity_hpo_terms=True   (terms blocked)

    The reported delta is removed minus retained. The primary run already sets
    Config.DROP_SEVERITY_HPO_TERMS = True, so its stored validation metrics are
    identical to the removed arm by construction and are reported separately as
    primary_run_validation_* rather than used as the ablation baseline.

    No primary test labels or primary test predictions are accessed here.
    """
    summary_path = os.path.join(Config.OUTPUT_DIR, "results", "severity_hpo_term_sensitivity.csv")
    removed_path = os.path.join(Config.OUTPUT_DIR, "results", "severity_hpo_terms_removed.csv")
    logger.section("SEVERITY-RELATED HPO TERM ABLATION (VALIDATION ONLY)")
    rows: List[Dict[str, Any]] = []
    removed_rows: List[Dict[str, Any]] = []
    selected = _pick_best_model_per_scenario(all_results)
    for scenario in Config.SHARED_HPO_BUDGET_SCENARIOS:
        if scenario not in selected or scenario not in _fold_artifacts:
            continue
        model_name = selected[scenario]
        artifact = _fold_artifacts[scenario]
        train_idx = np.asarray(artifact["train_idx"], dtype=int)
        val_idx = np.asarray(artifact["val_idx"], dtype=int)
        budget = artifact.get("selection_budget", Config.FEATURE_BUDGET)
        params = all_results[scenario]["all"][model_name].get("best_params", {})
        representation_retained = _fit_scenario_representation(
            container,
            scenario,
            train_idx,
            val_idx,
            budget,
            f"severity_hpo_retained_{scenario}",
            drop_severity_hpo_terms=False,
        )
        result_retained = _fit_final_candidate(
            model_name,
            params,
            representation_retained,
            container.y[train_idx],
            container.y[val_idx],
            scenario,
            RANDOM_SEED + 1400,
        )
        representation = _fit_scenario_representation(
            container,
            scenario,
            train_idx,
            val_idx,
            budget,
            f"severity_hpo_removed_{scenario}",
            drop_severity_hpo_terms=True,
        )
        result = _fit_final_candidate(
            model_name,
            params,
            representation,
            container.y[train_idx],
            container.y[val_idx],
            scenario,
            RANDOM_SEED + 1400,
        )
        restored = list(representation_retained.get("restored_severity_cols", []))
        n_candidates_retained = len(representation_retained.get("candidate_cols", []))
        n_candidates_removed = len(representation.get("candidate_cols", []))
        if not restored or n_candidates_retained <= n_candidates_removed:
            raise RuntimeError(
                f"{scenario}: severity-term ablation did not return any term to the candidate pool (restored={len(restored)}, candidates retained={n_candidates_retained}, removed={n_candidates_removed})."
            )
        logger.log(
            f"  {scenario}: {len(restored)} severity-term columns returned to the pool ({n_candidates_retained} vs {n_candidates_removed} candidates)"
        )
        primary = all_results[scenario]["all"][model_name].get("val_metrics", {})
        baseline = result_retained.get("val_metrics", {})
        metrics = result.get("val_metrics", {})
        retained_names = list(representation_retained.get("selected_names", []))
        severity_selected_when_retained = [
            column
            for column in retained_names
            if (column.startswith("hpo_") or column.startswith("feat_hpo_sys_"))
            and is_severity_hpo_column(column)
        ]
        removed = [
            item
            for item in representation["preprocessor"].dropped_features
            if item.get("drop_reason") == "severity_hpo_blocklist"
        ]
        for item in removed:
            provenance = _feature_provenance_for_mi(
                str(item.get("feature", "")), representation.get("family_lookup", {})
            )
            removed_rows.append(
                {
                    "Scenario": scenario,
                    "feature": item.get("feature", ""),
                    "drop_reason": item.get("drop_reason", ""),
                    **provenance,
                }
            )
        rows.append(
            {
                "Scenario": scenario,
                "Model": model_name,
                "evaluation_partition": "validation_only",
                "model_family_and_hyperparameters_fixed": True,
                "severity_hpo_terms_removed": True,
                "baseline_arm": "severity_terms_retained_refit",
                "n_severity_hpo_terms_removed": len(removed),
                "n_severity_terms_returned_to_pool": len(restored),
                "n_candidates_terms_retained": n_candidates_retained,
                "n_candidates_terms_removed": n_candidates_removed,
                "n_severity_terms_selected_when_retained": len(severity_selected_when_retained),
                "severity_terms_selected_when_retained": ";".join(sorted(severity_selected_when_retained)),
                "n_features_selected": len(representation.get("selected_names", [])),
                "n_features_selected_retained": len(retained_names),
                "mcc": metrics.get("mcc", np.nan),
                "qwk": metrics.get("qwk", np.nan),
                "balanced_accuracy": metrics.get("balanced_accuracy", np.nan),
                "baseline_validation_mcc": baseline.get("mcc", np.nan),
                "baseline_validation_qwk": baseline.get("qwk", np.nan),
                "baseline_validation_balanced_accuracy": baseline.get("balanced_accuracy", np.nan),
                "delta_mcc": metrics.get("mcc", np.nan) - baseline.get("mcc", np.nan),
                "delta_qwk": metrics.get("qwk", np.nan) - baseline.get("qwk", np.nan),
                "primary_run_validation_mcc": primary.get("mcc", np.nan),
                "primary_run_validation_qwk": primary.get("qwk", np.nan),
                "delta_mcc_vs_primary_run": metrics.get("mcc", np.nan) - primary.get("mcc", np.nan),
                "delta_qwk_vs_primary_run": metrics.get("qwk", np.nan) - primary.get("qwk", np.nan),
            }
        )
        logger.log(
            f"  {scenario}: retained arm selected {len(severity_selected_when_retained)} severity-related HPO terms (validation MCC={baseline.get('mcc', np.nan):.4f}, QWK={baseline.get('qwk', np.nan):.4f}); removed arm dropped {len(removed)} such terms (validation MCC={metrics.get('mcc', np.nan):.4f}, QWK={metrics.get('qwk', np.nan):.4f}); delta MCC={metrics.get('mcc', np.nan) - baseline.get('mcc', np.nan):+.4f}, delta QWK={metrics.get('qwk', np.nan) - baseline.get('qwk', np.nan):+.4f}."
        )
    dataframe = pd.DataFrame(rows)
    removed_dataframe = pd.DataFrame(
        removed_rows,
        columns=[
            "Scenario",
            "feature",
            "drop_reason",
            "feature_family",
            "source_dataset",
            "same_source_as_target_label",
            "provenance_status",
        ],
    )
    dataframe.to_csv(summary_path, index=False)
    removed_dataframe.to_csv(removed_path, index=False)
    return dataframe
