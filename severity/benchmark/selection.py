"""Feature selection and leakage controls.

Selection is fitted on the training pairs of each fold. Each feature is scored by its mutual
information (MI) with the severity class (scikit-learn; binary and low-cardinality integer
features treated as discrete, others with five nearest neighbours).

S1 keeps features with MI above 0.005 (at least 30); S1b uses 0.01 (at least 15). S2 and S3 keep
every feature with positive MI up to a limit per feature family. Features with unusually high MI
(top 1% of positive scores and at least 0.20) are flagged for review but never removed
(Supplementary Table 14).
"""

import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from .config import RANDOM_SEED, Config
from .features import _classify_feature_family
from .runtime import _high_mi_audit_records, logger


def _should_write_mi_csv(scenario_name: str) -> bool:
    internal_tokens = ("_perm", "_seed", "_mithr_", "_budgetx_", "_hpobudget_")
    is_internal = any((tok in str(scenario_name) for tok in internal_tokens))
    return not is_internal


def _write_mi_scores(mi_df: pd.DataFrame, scenario_name: str) -> None:
    if not _should_write_mi_csv(scenario_name):
        return
    safe_name = re.sub("[^A-Za-z0-9_.-]+", "_", str(scenario_name))
    out_dir = os.path.join(Config.OUTPUT_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    mi_df.to_csv(os.path.join(out_dir, f"mutual_information_scores_{safe_name}.csv"), index=False)


def _feature_provenance_for_mi(
    feature: str, family_lookup: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    lookup = family_lookup or {}
    family = _classify_feature_family(feature, lookup)
    if family == "unclassified":
        if feature.startswith("hpo_"):
            family = "hpo_binary"
        elif feature.startswith("feat_hpo_sys_"):
            family = "organ_system"
        elif feature.startswith("feat_mondo_"):
            family = "mondo"
        elif feature.startswith("feat_gnn_"):
            family = "gnn_text"
        elif feature.startswith("feat_primary_"):
            family = "primary_text"
        elif feature.startswith("feat_is_") or feature.startswith("feat_pli_x_"):
            family = "inheritance"
    source_by_family = {
        "mechanistic_core": "core_mechanistic_features",
        "external": "external_mechanistic_features",
        "variant": "variant_features",
        "inheritance": "primary_or_mechanistic_annotation",
        "engineered": "derived_from_allowed_inputs",
        "primary_text": "primary_label_source",
        "gnn_text": "gnn_dataset",
        "organ_system": "primary_label_source",
        "mondo": "MONDO_ontology",
        "hpo_binary": "primary_label_source_and_HPO",
        "hpo_derived": "HPO_training_fold_derivation",
        "mouse_lethality": "mouse_lethality_features",
        "mouse_phenotype": "mouse_human_phenotype_features",
        "missingness_indicator": "source_availability",
    }
    same_source = family in {"primary_text", "organ_system", "hpo_binary"}
    return {
        "feature_family": family,
        "source_dataset": source_by_family.get(family, "manual_review_required"),
        "same_source_as_target_label": same_source,
        "provenance_status": "same_source_review_required"
        if same_source
        else "independent_or_derived_review",
    }


def write_high_mi_provenance_report() -> pd.DataFrame:
    columns = [
        "analysis_context",
        "feature",
        "mi_score",
        "high_mi_review_cutoff",
        "feature_type_for_mi",
        "feature_family",
        "source_dataset",
        "same_source_as_target_label",
        "provenance_status",
        "mi_used_to_declare_leakage",
    ]
    dataframe = pd.DataFrame(_high_mi_audit_records, columns=columns)
    if not dataframe.empty:
        dataframe = dataframe.drop_duplicates().sort_values(
            ["analysis_context", "mi_score", "feature"], ascending=[True, False, True]
        )
    path = os.path.join(Config.OUTPUT_DIR, "results", "high_mi_provenance_flags.csv")
    dataframe.to_csv(path, index=False)
    logger.log(
        f"  Wrote {path} with {len(dataframe)} high-MI context/feature records ({(dataframe['feature'].nunique() if not dataframe.empty else 0)} unique features)."
    )
    return dataframe


def _infer_discrete_feature_mask(X: np.ndarray, feature_names: List[str]) -> np.ndarray:
    """Binary indicators and integer features with at most ten values are treated as discrete."""
    mask = np.zeros(len(feature_names), dtype=bool)
    explicit_prefixes = ("hpo_", "feat_hpo_sys_", "feat_is_", "has_", "feat_mondo_")
    for index, feature_name in enumerate(feature_names):
        name = str(feature_name)
        if name.endswith("__missing") or name.startswith(explicit_prefixes):
            mask[index] = name not in {"feat_mondo_depth"}
            continue
        column = np.asarray(X[:, index], dtype=float)
        finite = column[np.isfinite(column)]
        if not len(finite):
            continue
        unique = np.unique(finite)
        integer_like = np.allclose(unique, np.round(unique), atol=1e-10)
        if integer_like and len(unique) <= Config.MI_MAX_DISCRETE_UNIQUE:
            mask[index] = True
    return mask


def _compute_mutual_information_with_audit(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    scenario_name: str,
    family_lookup: Optional[Dict[str, str]] = None,
) -> Tuple[np.ndarray, pd.DataFrame, List[Dict[str, Any]]]:
    discrete_mask = _infer_discrete_feature_mask(X, feature_names)
    mi_matrix = np.asarray(X, dtype=float).copy()
    for index in np.flatnonzero(discrete_mask):
        _, inverse = np.unique(mi_matrix[:, index], return_inverse=True)
        mi_matrix[:, index] = inverse.astype(float)
    mi_scores = mutual_info_classif(
        mi_matrix,
        np.asarray(y, dtype=int),
        discrete_features=discrete_mask,
        random_state=RANDOM_SEED,
        n_neighbors=5,
    )
    mi_scores = np.nan_to_num(mi_scores, nan=0.0, posinf=0.0, neginf=0.0)
    positive = mi_scores[mi_scores > Config.MI_POSITIVE_EPS]
    if len(positive):
        quantile_cutoff = float(np.quantile(positive, Config.MI_HIGH_AUDIT_QUANTILE))
    else:
        quantile_cutoff = np.inf
    audit_cutoff = max(Config.MI_HIGH_AUDIT_MIN_SCORE, quantile_cutoff)
    high_mi_flag = np.isfinite(audit_cutoff) & (mi_scores >= audit_cutoff)
    mi_dataframe = pd.DataFrame(
        {
            "feature": feature_names,
            "mi_score": mi_scores,
            "feature_type_for_mi": np.where(discrete_mask, "discrete", "continuous"),
            "high_mi_review_flag": high_mi_flag,
            "high_mi_review_cutoff": audit_cutoff,
            "mi_used_to_declare_leakage": False,
        }
    ).sort_values("mi_score", ascending=False)
    audit_entries: List[Dict[str, Any]] = []
    for row in mi_dataframe.loc[mi_dataframe["high_mi_review_flag"]].itertuples(index=False):
        provenance = _feature_provenance_for_mi(str(row.feature), family_lookup)
        record = {
            "analysis_context": str(scenario_name),
            "feature": str(row.feature),
            "mi_score": float(row.mi_score),
            "high_mi_review_cutoff": float(row.high_mi_review_cutoff),
            "feature_type_for_mi": str(row.feature_type_for_mi),
            **provenance,
            "mi_used_to_declare_leakage": False,
        }
        _high_mi_audit_records.append(record)
        audit_entries.append(
            {
                "feature": record["feature"],
                "audit_action": "review_only",
                "drop_reason": "high_mi_provenance_review_not_dropped",
                "drop_statistic": record["mi_score"],
                "analysis_context": record["analysis_context"],
                "feature_family": record["feature_family"],
                "source_dataset": record["source_dataset"],
                "same_source_as_target_label": record["same_source_as_target_label"],
                "provenance_status": record["provenance_status"],
            }
        )
    if audit_entries:
        logger.log(
            f"  MI audit for {scenario_name}: {len(audit_entries)} high-value features flagged for provenance review; none were removed because of MI magnitude.",
            level="WARNING",
        )
        if "_perm" not in str(scenario_name):
            for entry in audit_entries:
                logger.log(
                    f"    HIGH-MI FEATURE name={entry['feature']} MI={entry['drop_statistic']:.6f} family={entry['feature_family']} source={entry['source_dataset']} same_source_as_label={entry['same_source_as_target_label']}",
                    level="WARNING",
                )
    return (mi_scores, mi_dataframe, audit_entries)


def select_features_mi(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    mi_threshold: float = 0.01,
    min_features: int = 15,
    scenario_name: str = "default",
    family_lookup: Optional[Dict[str, str]] = None,
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray, List[Dict]]:
    """S1 and S1b: features with MI above ``mi_threshold``, at least ``min_features`` of them."""
    logger.log(
        f"  Computing MI for {scenario_name} ({len(feature_names)} features, minimum selection threshold={mi_threshold}, min_features={min_features})..."
    )
    mi_scores, mi_dataframe, audit_entries = _compute_mutual_information_with_audit(
        X, y, feature_names, scenario_name, family_lookup
    )
    positive_epsilon = Config.MI_POSITIVE_EPS
    positive_indices = np.flatnonzero(mi_scores > positive_epsilon)
    if not len(positive_indices):
        raise ValueError(f"{scenario_name}: no feature had positive training-fold MI.")
    final_mask = np.zeros(len(feature_names), dtype=bool)
    final_mask[mi_scores > max(float(mi_threshold), positive_epsilon)] = True
    if int(final_mask.sum()) < int(min_features):
        # Fill up to the minimum with the best remaining positive-MI features (never zero-MI ones).
        ranked_positive = positive_indices[np.argsort(mi_scores[positive_indices])[::-1]]
        final_mask[ranked_positive[: min(int(min_features), len(ranked_positive))]] = True
    selected_names = [feature for feature, selected in zip(feature_names, final_mask) if selected]
    if not selected_names:
        raise ValueError(f"{scenario_name}: MI selection produced zero features.")
    selected_scores = mi_scores[final_mask]
    logger.log(
        f"    Selected {len(selected_names)} / {len(feature_names)} positive-MI features (min={selected_scores.min():.4g}, max={selected_scores.max():.4g})"
    )
    _write_mi_scores(mi_dataframe, scenario_name)
    return (X[:, final_mask], selected_names, mi_scores, final_mask, audit_entries)


def select_features_mi_grouped(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    family_lookup: Dict[str, str],
    budget: Dict[str, int],
    scenario_name: str = "default",
    mi_threshold: float = 0.0,
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray, List[Dict], Dict[str, int]]:
    """S2 and S3: positive-MI features, at most ``budget[family]`` per feature family."""
    logger.log(
        f"  Grouped MI selection for {scenario_name} ({len(feature_names)} features, {len(budget)} families, minimum MI={mi_threshold})..."
    )
    mi_scores, mi_dataframe, audit_entries = _compute_mutual_information_with_audit(
        X, y, feature_names, scenario_name, family_lookup
    )
    families = [_classify_feature_family(name, family_lookup) for name in feature_names]
    family_indices: Dict[str, List[int]] = defaultdict(list)
    for index, family in enumerate(families):
        family_indices[family].append(index)
    unclassified = family_indices.get("unclassified", [])
    if unclassified:
        logger.log(
            f"  WARNING: {len(unclassified)} unclassified features; sample: {[feature_names[index] for index in unclassified[:10]]}",
            level="WARNING",
        )
    positive_epsilon = Config.MI_POSITIVE_EPS
    minimum_score = max(float(mi_threshold), positive_epsilon)
    selected_indices: List[int] = []
    for family, indices in sorted(family_indices.items()):
        cap = 50 if family == "unclassified" else int(budget.get(family, 50))
        candidates = [index for index in indices if mi_scores[index] > minimum_score]
        candidates.sort(key=lambda index: mi_scores[index], reverse=True)
        selected_indices.extend(candidates[:cap])
    selected_indices = list(dict.fromkeys(selected_indices))
    minimum_features = max(15, len(budget))
    positive_indices = np.flatnonzero(mi_scores > positive_epsilon)
    if len(selected_indices) < minimum_features and len(positive_indices):
        ranked_positive = positive_indices[np.argsort(mi_scores[positive_indices])[::-1]]
        for index in ranked_positive:
            if index not in selected_indices:
                selected_indices.append(int(index))
            if len(selected_indices) >= minimum_features:
                break
        if len(selected_indices) < minimum_features:
            logger.log(
                f"    Only {len(selected_indices)} features had positive MI; requested minimum was {minimum_features}. Zero-MI features were not forced in.",
                level="WARNING",
            )
    if not selected_indices:
        raise ValueError(f"{scenario_name}: grouped MI selection found no positive-MI feature.")
    final_mask = np.zeros(len(feature_names), dtype=bool)
    final_mask[selected_indices] = True
    family_counts: Dict[str, int] = {}
    for index in np.flatnonzero(final_mask):
        family = families[index]
        family_counts[family] = family_counts.get(family, 0) + 1
    logger.log(
        f"    Grouped selection: {int(final_mask.sum())}/{len(feature_names)} features across {len(family_counts)} families"
    )
    for family in sorted(family_counts):
        available = len(family_indices.get(family, []))
        selected = family_counts[family]
        cap = 50 if family == "unclassified" else int(budget.get(family, 50))
        logger.log(f"      {family:25s}  {selected:4d}/{available:4d}  (budget={cap})")
    selected_names = [feature for feature, selected in zip(feature_names, final_mask) if selected]
    selected_scores = mi_scores[final_mask]
    _write_mi_scores(mi_dataframe, scenario_name)
    logger.log(
        f"    Final: {len(selected_names)} selected features (min MI={selected_scores.min():.4g}, max MI={selected_scores.max():.4g})"
    )
    return (X[:, final_mask], selected_names, mi_scores, final_mask, audit_entries, family_counts)


def uses_grouped_mi(scenario: str) -> bool:
    return any((s in scenario for s in ["S2", "S3"]))
