"""Model interpretation with SHAP (Fig. 4a-d).

SHAP values explain the expected severity class sum_k k * p_k(x) of the uncalibrated selected
model on all test pairs, so a positive value pushes a pair towards a more severe class.
ExtraTrees uses the tree-path-dependent TreeExplainer, the ordinal learner's boundary
classifiers the interventional TreeExplainer on the probability scale (100 training pairs as
background), and logistic regression the permutation explainer. Features are also summed by
biological source.
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import shap

from .config import RANDOM_SEED, Config
from .labels import readable_feature_label
from .models import ordinal_cumulative_to_class_probs
from .runtime import DataContainer, _fold_artifacts, logger
from .workflow import _pick_best_model_per_scenario


def _normalise_multiclass_shap_values(values: Any, n_features: int) -> List[np.ndarray]:
    """Return SHAP values as one (samples x features) array per output class."""
    if isinstance(values, list):
        arrays = [np.asarray(item) for item in values]
        return [item for item in arrays if item.ndim == 2]
    array = np.asarray(values)
    if array.ndim == 2:
        return [array]
    if array.ndim == 3:
        if array.shape[1] == n_features:
            return [array[..., class_index] for class_index in range(array.shape[2])]
        if array.shape[2] == n_features:
            return [array[:, class_index, :] for class_index in range(array.shape[1])]
    raise ValueError(f"Unsupported SHAP output shape: {array.shape}")


def severity_shap_group(name: str) -> str:
    """Biologically defined group for a feature column (missing flags follow their parent)."""
    base = str(name)
    if base.endswith("__missing"):
        base = base[: -len("__missing")]
    low = base.lower()
    if low.startswith("hpo_"):
        return "HPO findings"
    if low.startswith("feat_hpo_sys_"):
        return "Organ-system indicators"
    if low.startswith("feat_mondo_"):
        return "MONDO disease categories"
    if low.startswith("mouse_") or low in {
        "has_lethal_phenotype",
        "feat_lethal_x_pli",
        "feat_lethal_x_essential",
        "feat_has_lethal_x_pli",
    }:
        return "Mouse knockout"
    if low in {
        "feat_pli_x_dominant",
        "feat_pli_x_recessive",
        "feat_dominant_low_constraint",
        "feat_recessive_high_constraint",
    }:
        return "Constraint by inheritance"
    if low.startswith("feat_is_") or "inheritance" in low:
        return "Inheritance"
    if low.startswith(("clingen_", "hi_sensitive", "ts_sensitive")) or low in {
        "phaplo",
        "ptriplo",
        "is_haploinsufficient",
        "is_triplosensitive",
        "dosage_sensitive_any",
        "ordered_dosage_sensitive",
    }:
        return "Dosage sensitivity"
    if low in {"gnomad_gene_length", "gnomad_cds_length", "af_protein_length"}:
        return "Gene and protein size"
    if low.startswith(("gnomad_", "shet_", "gdi_")) or low in {
        "domino_score",
        "feat_pli_squared",
        "feat_missz_x_pli",
        "feat_missz_minus_synz",
        "feat_triple_constraint",
        "feat_pli_over_loeuf",
    }:
        return "Constraint"
    if low.startswith("depmap_") or low in {
        "feat_essential_constrained",
        "feat_essential_over_loeuf",
        "feat_central_essential",
        "feat_hub_essential",
    }:
        return "Cellular essentiality"
    if low.startswith(("network_", "ppi_")) or low in {
        "feat_log_ppi_degree",
        "feat_hub_constrained",
        "feat_betweenness_x_pli",
    }:
        return "Interaction network"
    if low.startswith("brainspan_"):
        return "Brain expression (BrainSpan)"
    if low.startswith(("gtex_", "hpa_")):
        return "Tissue expression (GTEx, HPA)"
    if low.startswith("af_") or low == "feat_conserved_disordered":
        return "Protein structure"
    if "ortholog" in low or low in {
        "conservation_score",
        "is_ancient",
        "gene_age",
        "feat_conserved_constrained",
    }:
        return "Conservation"
    if "paralog" in low or low in {"redundancy_score", "is_singleton"}:
        return "Paralogues and redundancy"
    if low.startswith(("var_", "feat_density_", "feat_prop_")) or low == "feat_ratio_lof_missense":
        return "Reported variants"
    if low.startswith(("feat_gnn_", "feat_primary_", "has_gnn_", "has_primary_")):
        return "Functional annotations"
    return "Other"


def _merge_missing_indicator_columns(
    shap_matrix: np.ndarray, value_matrix: np.ndarray, feature_names: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, List[str]]]:
    """Add each X__missing column's SHAP to X. Feature values (for colour) stay X's."""
    names = [str(name) for name in feature_names]
    out_names: List[str] = []
    out_index: Dict[str, int] = {}
    merged_from: Dict[str, List[str]] = {}
    shap_columns: List[np.ndarray] = []
    value_columns: List[np.ndarray] = []
    for j, name in enumerate(names):
        if name.endswith("__missing"):
            continue
        out_index[name] = len(out_names)
        out_names.append(name)
        shap_columns.append(np.asarray(shap_matrix[:, j], dtype=float).copy())
        value_columns.append(np.asarray(value_matrix[:, j], dtype=float))
        merged_from[name] = [name]
    for j, name in enumerate(names):
        if not name.endswith("__missing"):
            continue
        parent = name[: -len("__missing")]
        if parent in out_index:
            k = out_index[parent]
            shap_columns[k] = shap_columns[k] + np.asarray(shap_matrix[:, j], dtype=float)
            merged_from[parent].append(name)
        else:
            out_index[name] = len(out_names)
            out_names.append(name)
            shap_columns.append(np.asarray(shap_matrix[:, j], dtype=float).copy())
            value_columns.append(np.asarray(value_matrix[:, j], dtype=float))
            merged_from[name] = [name]
    return (np.column_stack(shap_columns), np.column_stack(value_columns), out_names, merged_from)


def _ordinal_probability_shap(boundary_model, X_test, reference):
    """Keep each LightGBM split exact in TreeExplainer's probability kernel."""
    trees = boundary_model.booster_.dump_model()["tree_info"]
    nodes_by_feature = defaultdict(list)
    for tree in trees:
        pending = [tree["tree_structure"]]
        while pending:
            node = pending.pop()
            if "split_index" not in node:
                continue
            if node["decision_type"] != "<=" or node["missing_type"] == "Zero":
                raise ValueError("Expected numeric LightGBM splits without zero-as-missing.")
            nodes_by_feature[node["split_feature"]].append(node)
            pending.extend((node["left_child"], node["right_child"]))
    if not np.isfinite(X_test).all() or not np.isfinite(reference).all():
        raise ValueError("Ordinal SHAP requires finite preprocessed features.")
    ranked_test = np.zeros_like(X_test, dtype=float)
    ranked_reference = np.zeros_like(reference, dtype=float)
    for feature, nodes in nodes_by_feature.items():
        cutoffs = np.unique([node["threshold"] for node in nodes])
        ranked_test[:, feature] = np.searchsorted(cutoffs, X_test[:, feature], side="left")
        ranked_reference[:, feature] = np.searchsorted(cutoffs, reference[:, feature], side="left")
        for node in nodes:
            node["threshold"] = float(np.searchsorted(cutoffs, node["threshold"])) + 0.5
    explainer = shap.TreeExplainer(
        {"trees": trees, "objective": "binary_crossentropy", "tree_output": "log_odds"},
        data=ranked_reference,
        model_output="probability",
        feature_perturbation="interventional",
    )
    probability = boundary_model.predict_proba(X_test)[:, 1]
    reference_mean = float(boundary_model.predict_proba(reference)[:, 1].mean())
    values = np.asarray(explainer.shap_values(ranked_test), dtype=float)
    base = float(np.asarray(explainer.expected_value).item())
    error = float(np.max(np.abs(values.sum(axis=1) + base - probability)))
    if not np.isfinite(error) or max(error, abs(base - reference_mean)) > 1e-06:
        raise ValueError(f"Ordinal probability SHAP failed the additivity check ({error:.3g}).")
    return (values, base, probability)


def _severity_shap_matrix(
    scenario: str, model_name: str, model: Any, X_test: np.ndarray, X_train: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, Dict[str, Any]]:
    """Return (SHAP n x p, base value per pair, explained quantity, method, extras)."""
    weights = np.arange(Config.NUM_CLASSES, dtype=float)
    rng = np.random.RandomState(RANDOM_SEED)

    def background(n_rows: int) -> np.ndarray:
        source = X_train if X_train is not None and len(X_train) else X_test
        size = min(int(n_rows), len(source))
        index = np.sort(rng.choice(len(source), size=size, replace=False))
        return np.asarray(source, dtype=float)[index]

    if model_name == "OrdinalDecomp":
        reference = background(Config.SEVERITY_SHAP_BACKGROUND_N)
        total = np.zeros(X_test.shape, dtype=float)
        base = 0.0
        cumulative = []
        for boundary_model in model:
            if scenario == "S2_Mech_Pheno":
                values, boundary_base, probability = _ordinal_probability_shap(
                    boundary_model, X_test, reference
                )
                total += values
                base += boundary_base
                cumulative.append(probability)
                continue
            explainer = shap.TreeExplainer(
                boundary_model,
                data=reference,
                model_output="probability",
                feature_perturbation="interventional",
            )
            values = explainer.shap_values(X_test)
            if isinstance(values, list):
                values = values[-1]
            values = np.asarray(values, dtype=float)
            if values.ndim == 3:
                values = values[..., -1]
            total += values
            base += float(np.asarray(explainer.expected_value, dtype=float).ravel()[-1])
            cumulative.append(boundary_model.predict_proba(X_test)[:, 1])
        cumulative_matrix = np.column_stack(cumulative)
        crossing = float(np.mean(np.any(np.diff(cumulative_matrix, axis=1) > 1e-12, axis=1)))
        class_probabilities = ordinal_cumulative_to_class_probs(cumulative_matrix)
        gap = np.abs(cumulative_matrix.sum(axis=1) - class_probabilities @ weights)
        method = "interventional TreeExplainer (probability) summed over the three ordinal boundaries"
        if scenario == "S2_Mech_Pheno":
            method += "; split cutoffs represented by exact ranks"
        return (
            total,
            np.full(len(X_test), base),
            cumulative_matrix.sum(axis=1),
            method,
            {
                "boundary_crossing_fraction": crossing,
                "mean_abs_gap_to_expected_class": float(np.mean(gap)),
                "max_abs_gap_to_expected_class": float(np.max(gap)),
                "background_n": int(len(reference)),
            },
        )
    if model_name == "ExtraTrees":
        explainer = shap.TreeExplainer(model)
        per_class = _normalise_multiclass_shap_values(explainer.shap_values(X_test), X_test.shape[1])
        expected = np.asarray(explainer.expected_value, dtype=float).ravel()
        values = sum((weights[k] * np.asarray(per_class[k], dtype=float) for k in range(Config.NUM_CLASSES)))
        base = float(np.dot(weights, expected[: Config.NUM_CLASSES]))
        explained = np.asarray(model.predict_proba(X_test), dtype=float) @ weights
        return (
            values,
            np.full(len(X_test), base),
            explained,
            "tree-path-dependent TreeExplainer (probability), expected severity class",
            {},
        )
    reference = background(Config.SEVERITY_SHAP_PERMUTATION_BACKGROUND_N)

    def predict_expected(Z: np.ndarray) -> np.ndarray:
        return np.asarray(model.predict_proba(Z), dtype=float) @ weights

    explainer = shap.PermutationExplainer(
        predict_expected,
        shap.maskers.Independent(reference, max_samples=len(reference)),
        seed=RANDOM_SEED,
    )
    explanation = explainer(X_test, max_evals=2 * X_test.shape[1] + 1, silent=True)
    return (
        np.asarray(explanation.values, dtype=float),
        np.asarray(explanation.base_values, dtype=float).ravel(),
        predict_expected(X_test),
        "permutation SHAP, expected severity class",
        {"background_n": int(len(reference))},
    )


def _severity_feature_summary(
    scenario: str,
    model_name: str,
    shap_matrix: np.ndarray,
    value_matrix: np.ndarray,
    names: Sequence[str],
    merged_from: Dict[str, List[str]],
) -> pd.DataFrame:
    """Per-feature mean |SHAP| and direction on the expected-severity scale."""
    rows: List[Dict[str, Any]] = []
    for j, name in enumerate(names):
        s = np.asarray(shap_matrix[:, j], dtype=float)
        v = np.asarray(value_matrix[:, j], dtype=float)
        finite = np.isfinite(v) & np.isfinite(s)
        rho = np.nan
        if finite.sum() > 2 and np.std(v[finite]) > 0 and (np.std(s[finite]) > 0):
            rho = float(pd.Series(v[finite]).corr(pd.Series(s[finite]), method="spearman"))
        high = np.zeros(len(v), dtype=bool)
        if finite.any():
            observed = v[finite]
            median = float(np.median(observed))
            high = finite & (v > median)
            if not high.any() and float(np.max(observed)) > float(np.min(observed)):
                high = finite & (v >= float(np.max(observed)))
        low = finite & ~high
        rows.append(
            {
                "Scenario": scenario,
                "Model": model_name,
                "feature": name,
                "label": readable_feature_label(name),
                "group": severity_shap_group(name),
                "merged_columns": "|".join(merged_from.get(name, [name])),
                "mean_abs_shap": float(np.mean(np.abs(s))),
                "mean_signed_shap": float(np.mean(s)),
                "spearman_value_vs_shap": rho,
                "n_high_value": int(high.sum()),
                "share_high_value_raises_severity": float(np.mean(s[high] > 0)) if high.any() else np.nan,
                "mean_shap_high_value": float(np.mean(s[high])) if high.any() else np.nan,
                "mean_shap_low_value": float(np.mean(s[low])) if low.any() else np.nan,
                "varies_in_test": bool(finite.any() and np.ptp(v[finite]) > 0),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["varies_in_test", "mean_abs_shap"], ascending=[False, False])
    frame = frame.reset_index(drop=True)
    frame.insert(3, "rank", np.arange(1, len(frame) + 1))
    return frame


def _severity_group_summary(
    scenario: str,
    model_name: str,
    shap_matrix: np.ndarray,
    names: Sequence[str],
    feature_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Group SHAP = sum of a pair's signed SHAP over the group's features."""
    groups = np.asarray([severity_shap_group(name) for name in names])
    rows: List[Dict[str, Any]] = []
    for group in pd.unique(groups):
        columns = np.where(groups == group)[0]
        group_shap = np.asarray(shap_matrix[:, columns], dtype=float).sum(axis=1)
        members = feature_frame[feature_frame["group"] == group]
        top = members.iloc[0] if len(members) else None
        rows.append(
            {
                "Scenario": scenario,
                "Model": model_name,
                "group": group,
                "n_features": int(len(columns)),
                "mean_abs_group_shap": float(np.mean(np.abs(group_shap))),
                "mean_signed_group_shap": float(np.mean(group_shap)),
                "share_pairs_raised": float(np.mean(group_shap > 0)),
                "top_feature": None if top is None else top["feature"],
                "top_feature_label": None if top is None else top["label"],
                "top_feature_mean_abs_shap": None if top is None else float(top["mean_abs_shap"]),
            }
        )
    frame = pd.DataFrame(rows)
    total = float(frame["mean_abs_group_shap"].sum()) or 1.0
    frame["share_of_total"] = frame["mean_abs_group_shap"] / total
    frame = frame.sort_values("mean_abs_group_shap", ascending=False).reset_index(drop=True)
    frame.insert(3, "rank", np.arange(1, len(frame) + 1))
    return frame


def save_severity_shap(container: DataContainer, all_results: Dict) -> None:
    """Full-matrix SHAP on the expected severity class for each selected model."""
    if False:
        logger.log("  Severity-score SHAP skipped", level="WARNING")
        return
    logger.section("SEVERITY-SCORE SHAP (EXPECTED SEVERITY CLASS, ALL TEST PAIRS)")
    explanation_dir = os.path.join(Config.OUTPUT_DIR, "per_disease_explanations")
    results_dir = os.path.join(Config.OUTPUT_DIR, "results")
    os.makedirs(explanation_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    metadata = container.metadata.reset_index(drop=True)
    feature_frames: List[pd.DataFrame] = []
    group_frames: List[pd.DataFrame] = []
    checks: List[Dict[str, Any]] = []
    for scenario, model_name in _pick_best_model_per_scenario(all_results).items():
        artifact = _fold_artifacts.get(scenario)
        result = all_results[scenario]["all"].get(model_name, {})
        if artifact is None or "X_test" not in artifact or result.get("model") is None:
            continue
        X_test = np.asarray(artifact["X_test"], dtype=float)
        X_train = artifact.get("X_train")
        X_train = None if X_train is None else np.asarray(X_train, dtype=float)
        names = [str(name) for name in artifact["selected_names"]]
        test_idx = np.asarray(artifact["test_idx"])
        started = datetime.now()
        try:
            values, base, explained, method, extras = _severity_shap_matrix(
                scenario, model_name, result["model"], X_test, X_train
            )
        except Exception as exc:
            logger.log(f"  Severity SHAP failed for {scenario}/{model_name}: {exc}", level="WARNING")
            continue
        additivity_error = float(np.max(np.abs(values.sum(axis=1) + base - explained)))
        merged_shap, merged_values, merged_names, merged_from = _merge_missing_indicator_columns(
            values, X_test, names
        )
        feature_frame = _severity_feature_summary(
            scenario, model_name, merged_shap, merged_values, merged_names, merged_from
        )
        group_frame = _severity_group_summary(scenario, model_name, merged_shap, merged_names, feature_frame)
        meta = metadata.iloc[test_idx].reset_index(drop=True)
        predictions = np.asarray(
            result.get("predictions", {}).get("y_pred", np.full(len(X_test), -1)), dtype=int
        )
        y_true = np.asarray(result.get("y_true", np.full(len(X_test), -1)), dtype=int)
        np.savez_compressed(
            os.path.join(explanation_dir, f"severity_shap_{scenario}.npz"),
            shap=merged_shap.astype(np.float32),
            values=merged_values.astype(np.float32),
            feature_names=np.asarray(merged_names, dtype=str),
            base_value=np.asarray(base, dtype=float),
            expected_severity=np.asarray(explained, dtype=float),
            gene_symbol=np.asarray(meta.get("gene_symbol", pd.Series([""] * len(meta))), dtype=str),
            mondo_id=np.asarray(meta.get("mondo_id", pd.Series([""] * len(meta))), dtype=str),
            true_class=y_true,
            predicted_class=predictions,
            model_name=np.asarray(model_name),
            method=np.asarray(method),
            merged_from_json=np.asarray(json.dumps(merged_from)),
        )
        feature_frame.to_csv(os.path.join(results_dir, f"severity_shap_features_{scenario}.csv"), index=False)
        group_frame.to_csv(os.path.join(results_dir, f"severity_shap_groups_{scenario}.csv"), index=False)
        feature_frames.append(feature_frame)
        group_frames.append(group_frame)
        seconds = (datetime.now() - started).total_seconds()
        checks.append(
            {
                "Scenario": scenario,
                "Model": model_name,
                "method": method,
                "n_pairs": int(len(X_test)),
                "n_columns": int(len(names)),
                "n_features_after_merging_missing_flags": int(len(merged_names)),
                "max_additivity_error": additivity_error,
                "seconds": seconds,
                **extras,
            }
        )
        logger.log(
            f"  {scenario}/{model_name}: {method}; {len(X_test)} pairs x {len(names)} columns; max |sum(SHAP) + base - prediction| = {additivity_error:.2e}; {seconds:.0f} s"
        )
    if feature_frames:
        pd.concat(feature_frames, ignore_index=True).to_csv(
            os.path.join(results_dir, "severity_shap_features.csv"), index=False
        )
        pd.concat(group_frames, ignore_index=True).to_csv(
            os.path.join(results_dir, "severity_shap_groups.csv"), index=False
        )
        pd.DataFrame(checks).to_csv(os.path.join(results_dir, "severity_shap_checks.csv"), index=False)
