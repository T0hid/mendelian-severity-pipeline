"""Main and Extended Data figures of the benchmark, drawn from a completed run.

Manuscript figure   function                              file in <out>/main
Fig. 2              figure_2_model_comparison             figure_2_model_comparison.*
Fig. 3              figure_3_prediction_calibration       figure_3_prediction_and_calibration.*
Fig. 4              figure_4_interpretation               figure_4_interpretation_and_biology.*
Extended Data Fig. 1  extended_data_fig_1_cohort          extended_data_fig_1_cohort_composition.*
Extended Data Fig. 2  extended_data_fig_2_robustness      extended_data_fig_2_robustness_and_audits.*
Extended Data Fig. 3  extended_data_fig_3_calibration     extended_data_fig_3_calibration_confusion.*
Table 1             table_1_headline_metrics              table_1_headline_metrics.csv

The numbers behind every panel are written to <out>/figure_data. Those file names keep the
internal numbering used while the figures were drafted (figure_1* = Extended Data Fig. 1,
figure_4* = Extended Data Fig. 2, figure_5* = Fig. 4); figure_name_map.csv lists the mapping.

Candidate models are compared on the validation partition only. Test-set panels show the
validation-selected model of each feature set.
"""

from __future__ import annotations

import os
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import LogLocator, NullFormatter, PercentFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from severity.benchmark.calibration_changes import (
    CHANGES_FILE,
    CONFUSION_FILE,
    STAGES,
    TRANSITIONS_FILE,
    calibration_change_tables,
)

CLASS_NAMES = ["Mild", "Moderate", "Severe", "Profound"]
CLASS_INDICES = {n: i for i, n in enumerate(CLASS_NAMES)}
SCENARIO_LABELS = {
    "S1_Mechanistic": "S1: Gene-level",
    "S1b_Disease_Context": "S1b: Gene-level + disease context",
    "S2_Mech_Pheno": "S2: Combined",
    "S3_Phenotype": "S3: Phenotype-only",
    "S2_blind_diagnostic": "S2-blind: S2 without tier features",
}
SCENARIO_SHORT = {
    "S1_Mechanistic": "S1",
    "S1b_Disease_Context": "S1b",
    "S2_Mech_Pheno": "S2",
    "S3_Phenotype": "S3",
    "S2_blind_diagnostic": "S2-blind",
}
SCENARIO_ORDER = ["S1_Mechanistic", "S1b_Disease_Context", "S2_Mech_Pheno", "S3_Phenotype"]
PAL_CLASS = {"Mild": "#0072B2", "Moderate": "#56B4E9", "Severe": "#E69F00", "Profound": "#CC79A7"}
PAL_SCENARIO = {
    "S1_Mechanistic": "#5F7F9E",
    "S1b_Disease_Context": "#4C9A87",
    "S2_Mech_Pheno": "#D8892B",
    "S3_Phenotype": "#9473A9",
    "S2_blind_diagnostic": "#8A8A8A",
}
MODEL_ORDER = ["MultinomialLogisticRegression", "ExtraTrees", "LightGBM", "OrdinalDecomp"]
MODEL_LABELS = {
    "MultinomialLogisticRegression": "Multinomial logistic regression",
    "ExtraTrees": "ExtraTrees",
    "LightGBM": "LightGBM",
    "OrdinalDecomp": "Ordinal learner",
}
PAL_MODEL = {
    "MultinomialLogisticRegression": "#E69F00",
    "ExtraTrees": "#009E73",
    "LightGBM": "#0072B2",
    "OrdinalDecomp": "#CC79A7",
}
MODEL_MARKERS = {
    "MultinomialLogisticRegression": "s",
    "ExtraTrees": "^",
    "LightGBM": "o",
    "OrdinalDecomp": "D",
}
MODEL_SHORT_LABELS = {
    "MultinomialLogisticRegression": "Multinomial logistic",
    "ExtraTrees": "ExtraTrees",
    "LightGBM": "LightGBM",
    "OrdinalDecomp": "Ordinal learner",
}
_BROAD_MONDO_TERMS = {
    "disease",
    "human disease",
    "hereditary disease",
    "genetic disease",
    "monogenic disease",
    "autosomal recessive disease",
    "autosomal dominant disease",
    "disease by anatomical system",
    "disease or disorder",
}
_GNN_LABEL_PREFIXES = (
    ("feat_gnn_biological_processes_", "GO process: ", "GO biological processes annotated"),
    ("feat_gnn_molecular_activities_", "GO function: ", "GO molecular functions annotated"),
    ("feat_gnn_expression_sites_", "Expressed in: ", "Expression sites annotated"),
    ("feat_gnn_interacting_genes_", "Interacts with: ", "Interaction partners annotated"),
    ("feat_gnn_pathways_", "Pathway: ", "Pathways annotated"),
)
HPO_BUDGET_ALL_ELIGIBLE = 10000


def _is_specific_mondo_label(label: str) -> bool:
    text = str(label).strip().lower()
    if not text:
        return False
    if text in _BROAD_MONDO_TERMS:
        return False
    broad_patterns = [
        "human disease",
        "hereditary disease",
        "genetic disease",
        "monogenic disease",
        "disease by ",
    ]
    return not any((pattern in text for pattern in broad_patterns))


COLOR_CORRECT = "#4C78A8"
COLOR_ADJ_ERROR = "#F2CF70"
COLOR_NONADJ_ERROR = "#E07B54"
NEUTRAL_DARK = "#333333"
NEUTRAL_MID = "#777777"
S3_PCA_DUPLICATE_CLUSTER_MIN = 25
S3_PCA_TARGET_POINTS = 700
S3_PCA_ERROR_COLOURS = ("#DCDCDC", "#009E73", "#B2182B")
S3_PCA_ERROR_LABELS = ("Correct", "Adjacent error", "Non-adjacent error")
S3_PCA_FS = {
    "panel_letter": 11.0,
    "panel_title": 10.0,
    "subplot_title": 9.5,
    "axis_label": 9.0,
    "tick": 8.5,
    "annotation": 7.4,
    "legend": 8.2,
}
NEUTRAL_LIGHT = "#D9D9D9"
_BUILTIN_NON_HPO_LABELS: Dict[str, str] = {
    "pli": "pLI (LoF intolerance)",
    "loeuf": "LOEUF",
    "mis_z": "Missense Z-score",
    "syn_z": "Synonymous Z-score",
    "mis_zscore": "Missense Z-score",
    "var_mean_cadd": "Mean CADD score",
    "var_total_variants": "Total variants",
    "var_n_pathogenic_clinvar": "Pathogenic variants (ClinVar)",
    "var_n_lof": "LoF variants",
    "var_n_missense": "Missense variants",
    "ppi_degree": "PPI network degree",
    "string_degree": "STRING network degree",
    "betweenness": "Network betweenness",
    "network_degree": "Network degree",
    "gtex_tau": "GTEx tissue specificity (tau)",
    "expression_breadth": "Tissue expression breadth",
    "n_tissues": "Tissue expression breadth",
    "tissue_breadth": "Tissue expression breadth",
    "paralog": "Paralog count",
    "n_paralogs": "Paralog count",
    "paralog_count": "Paralog count",
    "phaplo": "pHaplo (haploinsufficiency)",
    "ptriplo": "pTriplo (triplosensitivity)",
    "depmap": "DepMap essentiality",
    "essentiality": "Essentiality score",
    "essential": "Essential gene flag",
    "af_structural_rigidity": "AlphaFold structural rigidity",
    "af_frac_disordered": "AlphaFold disorder fraction",
    "structural_rigidity": "Structural rigidity",
    "frac_disordered": "Disorder fraction",
    "domain_count": "Domain count",
    "ptm_count": "PTM count",
    "protein_length": "Protein length",
    "conservation": "Sequence conservation",
    "phastcons": "phastCons",
    "phylop": "phyloP",
    "gerp": "GERP",
    "critical_brain": "Critical organ: brain",
    "critical_heart": "Critical organ: heart",
    "critical_immune": "Critical organ: immune",
    "critical_lung": "Critical organ: lung",
    "critical_kidney": "Critical organ: kidney",
    "n_critical_organs": "Number of critical organs affected",
    "n_noncritical_organs": "Number of non-critical organs affected",
    "noncritical_sensory": "Non-critical organ: sensory",
    "noncritical_total": "Non-critical organs (total)",
    "tissue_criticality_score": "Tissue criticality score",
    "feat_is_dominant": "Inheritance: dominant",
    "feat_is_recessive": "Inheritance: recessive",
    "feat_is_xlinked": "Inheritance: X-linked",
    "feat_has_mixed_inheritance": "Mixed inheritance",
    "feat_inheritance_not_recorded": "Inheritance not recorded",
    "feat_pli_x_dominant": "pLI x dominant",
    "feat_pli_x_recessive": "pLI x recessive",
    "feat_gene_n_diseases": "Gene: number of associated diseases",
    "feat_pli_squared": "pLI squared",
    "feat_loeuf_squared": "LOEUF squared",
    "feat_log_ppi_degree": "log(1 + PPI degree)",
    "feat_mondo_depth": "MONDO ontology depth",
    "feat_mondo_n_classes": "MONDO disease classes (count)",
    "feat_hpo_max_ic": "HPO information content (max)",
    "feat_hpo_mean_ic": "HPO information content (mean)",
    "feat_hpo_median_ic": "HPO information content (median)",
    "feat_hpo_ic_range": "HPO IC range",
    "feat_hpo_ic_std": "HPO IC std",
    "feat_hpo_rare_term_ratio": "HPO rare-term ratio",
}
_SYS_PREFIX = "feat_hpo_sys_"


class FeatureLabeler:
    """Readable feature names: the run's feature_label_dictionary_extra.csv first, then built-in
    labels and prefix rules for names that are not in the dictionary."""

    def __init__(self, output_dir: Optional[str] = None):
        self.extra_dict: Dict[str, str] = {}
        path = os.path.join(output_dir, "feature_label_dictionary_extra.csv") if output_dir else None
        if path and os.path.exists(path):
            df = pd.read_csv(path)
            self.extra_dict = dict(zip(df["feature_name"].astype(str), df["label"].astype(str)))

    def __call__(self, name: str, max_len: Optional[int] = None) -> str:
        out = self._prettify(name)
        if max_len and len(out) > max_len:
            out = out[: max_len - 1].rstrip() + "…"
        return out

    def _prettify(self, name: str) -> str:
        if not isinstance(name, str):
            name = str(name)
        if name in self.extra_dict:
            return self.extra_dict[name]
        m_hp = re.match("^(?:feat_)?hpo_HP[_:](\\d{7})$", name, re.IGNORECASE)
        if m_hp:
            return f"HP:{m_hp.group(1)}"
        low = name.lower()
        if low in _BUILTIN_NON_HPO_LABELS:
            return _BUILTIN_NON_HPO_LABELS[low]
        if low.startswith(_SYS_PREFIX):
            tail = name[len(_SYS_PREFIX) :].replace("_", " ").strip()
            tail = tail[0].upper() + tail[1:] if tail else tail
            return f"{tail} (parent HPO term)"
        if low.startswith("feat_tier_"):
            tail = name[len("feat_tier_") :]
            return self._humanize_tier(tail)
        if low.startswith("feat_mondo_"):
            tail = name[len("feat_mondo_") :].replace("_", " ").strip()
            return f"MONDO: {tail}"
        for prefix, human, count_label in _GNN_LABEL_PREFIXES:
            if low.startswith(prefix):
                tail = name[len(prefix) :]
                if tail.lower() == "count":
                    return f"{count_label}, count"
                tail = re.sub("_[0-9a-f]{6}$", "...", tail)
                return human + tail.replace("_", " ").strip()
        if low.startswith("feat_"):
            tail = name[len("feat_") :].replace("_", " ").strip()
            return tail[0].upper() + tail[1:] if tail else tail
        return name.replace("_", " ").strip().capitalize()

    @staticmethod
    def _humanize_tier(tail: str) -> str:
        m = re.match("^(n|frac)_cat_(.+)$", tail)
        if m:
            kind = "Count" if m.group(1) == "n" else "Fraction"
            cat = m.group(2).replace("_", " ").strip()
            return f"Tier {kind.lower()}: {cat}"
        m = re.match("^(n|frac)_qol_(.+)$", tail)
        if m:
            kind = "count" if m.group(1) == "n" else "fraction"
            cat = m.group(2).replace("_", " ").strip()
            return f"QoL {kind}: {cat}"
        m = re.match("^(n|frac)_tier(\\d+|nfc)$", tail)
        if m:
            kind = "count" if m.group(1) == "n" else "fraction"
            tier = m.group(2)
            return f"Tier-{tier} {kind}"
        return tail.replace("_", " ").strip()


def _save_csv(df: pd.DataFrame, filename: str, fig_data_dir: str):
    os.makedirs(fig_data_dir, exist_ok=True)
    df.to_csv(os.path.join(fig_data_dir, filename), index=False)


def _present_scenarios(scenarios_iter) -> List[str]:
    """Feature sets in manuscript order, keeping only those present."""
    have = set(scenarios_iter)
    return [s for s in SCENARIO_ORDER if s in have]


def _stable_models(models_iter) -> List[str]:
    """Candidate models in manuscript order."""
    have = set(models_iter)
    return [model for model in MODEL_ORDER if model in have]


def _model_result_keys(scenario: str, preferred_model: Optional[str] = None) -> List[str]:
    """Result keys ``<scenario>_<model>``, the preferred model first."""
    ordered: List[str] = [str(preferred_model)] if preferred_model else []
    ordered.extend((model for model in MODEL_ORDER if model not in ordered))
    return [f"{scenario}_{model}" for model in ordered]


MIN_PERMUTATIONS_FOR_INFERENCE = 20


def _permutation_inference_available(
    frame: Optional[pd.DataFrame], minimum: int = MIN_PERMUTATIONS_FOR_INFERENCE
) -> bool:
    """True when enough permutations succeeded to report intervals or P values."""
    if frame is None or frame.empty:
        return False
    count_col = next(
        (
            name
            for name in ("n_permutations_successful", "n_permutations_requested", "n_permutations")
            if name in frame.columns
        ),
        None,
    )
    if count_col is None:
        return False
    counts = pd.to_numeric(frame[count_col], errors="coerce").dropna()
    return bool(len(counts) and (counts >= int(minimum)).all())


def _get_test_metrics(result: Dict) -> Dict[str, float]:
    """Test-set metrics stored for one model."""
    return result.get("metrics") or {}


def _selected_publication_models(all_results: Dict, pub_outputs: Optional[Dict] = None) -> Dict[str, str]:
    """The validation-selected model of each feature set."""
    return _validation_selected_models(all_results, pub_outputs)


def _plot_saved_calibration(
    ax, scenario: str, model_name: str, bins_df: pd.DataFrame, ece_df: pd.DataFrame
) -> pd.DataFrame:
    scenario_bins = bins_df[
        (bins_df["Scenario"] == scenario) & (bins_df["Model"].astype(str) == str(model_name))
    ].copy()
    scenario_ece = ece_df[
        (ece_df["Scenario"] == scenario) & (ece_df["Model"].astype(str) == str(model_name))
    ].copy()
    if "calibration_status" in scenario_bins.columns:
        calibrated = scenario_bins[
            scenario_bins["calibration_status"].astype(str).str.lower() == "calibrated"
        ]
        if not calibrated.empty:
            scenario_bins = calibrated.copy()
    if "calibration_status" in scenario_ece.columns:
        calibrated = scenario_ece[scenario_ece["calibration_status"].astype(str).str.lower() == "calibrated"]
        if not calibrated.empty:
            scenario_ece = calibrated.copy()
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6, lw=0.7)
    export_rows = []
    for class_name in CLASS_NAMES:
        class_bins = scenario_bins[scenario_bins["severity_class"] == class_name].sort_values("bin")
        class_ece = scenario_ece[scenario_ece["severity_class"] == class_name]
        if class_bins.empty or class_ece.empty:
            continue
        ece = float(class_ece["ece"].iloc[0])
        x_values = class_bins["mean_predicted_probability"].to_numpy(dtype=float)
        y_values = class_bins["observed_frequency"].to_numpy(dtype=float)
        ax.plot(
            x_values,
            y_values,
            marker="o",
            markersize=3.5,
            lw=1,
            color=PAL_CLASS[class_name],
            label=f"{class_name} (ECE={ece:.2f})",
        )
        exported = class_bins.copy()
        exported["ece"] = ece
        export_rows.append(exported)
    if not export_rows:
        return pd.DataFrame()
    return pd.concat(export_rows, ignore_index=True)


_MODEL_SUFFIXES = (
    "_MultinomialLogisticRegression",
    "_ExtraTrees",
    "_LightGBM",
    "_OrdinalDecomp",
    "_MLR",
    "_LogisticRegression",
    "_ET",
    "_LGB",
    "_OrdDecomp",
)
SEVERITY_SHAP_SOURCE = "SHAP on expected severity class"


def _read_scenario_shap_observations(output_dir: str, scenario: str):
    """Per-pair SHAP values on the expected severity class saved by the pipeline.

    Returns ``(obs, source_label, n_rows)`` where ``obs`` maps feature name to
    ``(shap_values, feature_values)``.
    """
    path = os.path.join(output_dir, "per_disease_explanations", f"severity_shap_{scenario}.npz")
    if not os.path.exists(path):
        return ({}, None, 0)
    with np.load(path, allow_pickle=False) as data:
        names = [str(name) for name in data["feature_names"]]
        shap_matrix = np.asarray(data["shap"], dtype=float)
        value_matrix = np.asarray(data["values"], dtype=float)
    obs = {name: (shap_matrix[:, j], value_matrix[:, j]) for j, name in enumerate(names)}
    return (obs, SEVERITY_SHAP_SOURCE, int(shap_matrix.shape[0]))


def figure_4_family_summed_shap(
    output_dir: str,
    all_results: Dict,
    pub_outputs: Dict,
    container,
    fig_data_dir: str,
    labeler: "FeatureLabeler",
    logger=print,
) -> pd.DataFrame:
    """Fig. 4 companion table: SHAP summed within feature families.

    Correlated columns of one family split their attribution between them, so per-feature
    SHAP understates multi-column families; this table sums them."""
    groups_path = os.path.join(output_dir, "results", "severity_shap_groups.csv")
    if os.path.exists(groups_path):
        try:
            frame = pd.read_csv(groups_path)
        except Exception:
            frame = pd.DataFrame()
        if not frame.empty:
            frame = frame.rename(
                columns={
                    "group": "family_label",
                    "mean_abs_group_shap": "summed_mean_abs_shap",
                    "top_feature": "largest_single_feature",
                    "top_feature_label": "largest_single_label",
                    "top_feature_mean_abs_shap": "largest_single_mean_abs_shap",
                }
            )
            frame["source"] = SEVERITY_SHAP_SOURCE
            _save_csv(frame, "figure_5f_family_summed_shap.csv", fig_data_dir)
            logger(f"[Fig. 4] wrote severity-SHAP group summary for {frame['Scenario'].nunique()} scenarios")
            return frame
    lookup = getattr(container, "feature_family_lookup", {}) or {}
    selected = _validation_selected_models(all_results, pub_outputs)
    rows: List[Dict[str, Any]] = []
    for scenario in _present_scenarios(all_results.keys()):
        obs, source_label, n_rows = _read_scenario_shap_observations(output_dir, scenario)
        if not obs:
            continue
        denom = float(n_rows) if n_rows else 1.0
        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        best: Dict[str, Tuple[str, float]] = {}
        for feature, (shap_values, _) in obs.items():
            values = np.asarray(shap_values, dtype=float)
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            magnitude = float(np.abs(values).sum() / denom)
            family = _coarse_feature_family(str(feature), lookup)
            totals[family] = totals.get(family, 0.0) + magnitude
            counts[family] = counts.get(family, 0) + 1
            if family not in best or magnitude > best[family][1]:
                best[family] = (str(feature), magnitude)
        grand = sum(totals.values()) or 1.0
        for family, total in sorted(totals.items(), key=lambda kv: -kv[1]):
            top_feature, top_value = best[family]
            rows.append(
                {
                    "Scenario": scenario,
                    "Model": selected.get(scenario),
                    "family": family,
                    "family_label": _family_display(family),
                    "n_features": counts[family],
                    "summed_mean_abs_shap": total,
                    "share_of_total": total / grand,
                    "largest_single_feature": top_feature,
                    "largest_single_label": labeler(top_feature),
                    "largest_single_mean_abs_shap": top_value,
                    "dilution_ratio": total / top_value if top_value else np.nan,
                    "source": source_label,
                    "n_explained": n_rows,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        logger("[Fig. 4] family-summed SHAP unavailable; skipping")
        return frame
    _save_csv(frame, "figure_5f_family_summed_shap.csv", fig_data_dir)
    logger(f"[Fig. 4] wrote family-summed SHAP for {frame['Scenario'].nunique()} scenarios")
    return frame


def table_1_headline_metrics(
    all_results: Dict, pub_outputs: Dict, out_dir: str, fig_data_dir: str, logger=print
):
    bootstrap = pub_outputs.get("bootstrap", pd.DataFrame())
    perm = pub_outputs.get("permutation_test", pd.DataFrame())
    selected_models = pub_outputs.get("selected_models", {}) or {}
    rows = []
    for scenario in _present_scenarios(all_results.keys()):
        all_models = all_results.get(scenario, {}).get("all", {})
        selected_model = selected_models.get(scenario)
        if selected_model is not None:
            model_names = [selected_model]
        else:
            model_names = _stable_models(all_models.keys())
        for model_name in model_names:
            result = all_models.get(model_name)
            if not result:
                continue
            metrics = _get_test_metrics(result)
            if not metrics:
                logger(f"[table_1] skipping {scenario}/{model_name}: test metrics unavailable")
                continue
            if not bootstrap.empty and {"Scenario", "Model"}.issubset(bootstrap.columns):
                boot = bootstrap[
                    (bootstrap["Scenario"] == scenario) & (bootstrap["Model"].astype(str) == str(model_name))
                ]
            else:
                boot = pd.DataFrame()
            if not boot.empty:
                bootstrap_row = boot.iloc[0]
                mcc_ci = (bootstrap_row.get("mcc_ci_low", np.nan), bootstrap_row.get("mcc_ci_high", np.nan))
                qwk_ci = (bootstrap_row.get("qwk_ci_low", np.nan), bootstrap_row.get("qwk_ci_high", np.nan))
            else:
                mcc_ci = (np.nan, np.nan)
                qwk_ci = (np.nan, np.nan)
            perm_model_col = (
                "Model"
                if "Model" in perm.columns
                else "Observed_Model"
                if "Observed_Model" in perm.columns
                else None
            )
            if (
                _permutation_inference_available(perm)
                and "Scenario" in perm.columns
                and (perm_model_col is not None)
            ):
                perm_match = perm[
                    (perm["Scenario"] == scenario) & (perm[perm_model_col].astype(str) == str(model_name))
                ]
            else:
                perm_match = pd.DataFrame()
            if not perm_match.empty and "empirical_p_value" in perm_match.columns:
                permutation_p = float(perm_match["empirical_p_value"].iloc[0])
            else:
                permutation_p = np.nan
            if "y_true" in result and result["y_true"] is not None:
                n_test = int(len(result["y_true"]))
            else:
                n_test = np.nan
            rows.append(
                {
                    "Scenario_id": scenario,
                    "Scenario": SCENARIO_LABELS.get(scenario, scenario),
                    "Model": model_name,
                    "MCC": float(metrics.get("mcc", np.nan)),
                    "MCC_ci_low": float(mcc_ci[0]) if not pd.isna(mcc_ci[0]) else np.nan,
                    "MCC_ci_high": float(mcc_ci[1]) if not pd.isna(mcc_ci[1]) else np.nan,
                    "QWK": float(metrics.get("qwk", np.nan)),
                    "QWK_ci_low": float(qwk_ci[0]) if not pd.isna(qwk_ci[0]) else np.nan,
                    "QWK_ci_high": float(qwk_ci[1]) if not pd.isna(qwk_ci[1]) else np.nan,
                    "Balanced_acc": float(metrics.get("balanced_accuracy", np.nan)),
                    "F1_macro": float(metrics.get("f1_macro", metrics.get("f1", np.nan))),
                    "Adjacent_acc": float(metrics.get("adjacent_accuracy", np.nan)),
                    "Spearman_rho": float(metrics.get("spearman_rho", np.nan)),
                    "AUC_ge_Moderate": float(metrics.get("auc_threshold_ge1", np.nan)),
                    "AUC_ge_Severe": float(metrics.get("auc_threshold_ge2", np.nan)),
                    "AUC_ge_Profound": float(metrics.get("auc_threshold_ge3", np.nan)),
                    "n_test": n_test,
                    "Permutation_p": permutation_p,
                }
            )
    dataframe = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "table_1_headline_metrics.csv")
    dataframe.to_csv(csv_path, index=False)
    _save_csv(dataframe, "table_1_headline_metrics.csv", fig_data_dir)

    def _fmt(value, dp=3):
        if value is None or pd.isna(value):
            return "n/a"
        return f"{float(value):.{dp}f}"

    markdown_rows = []
    markdown_rows.append(
        "| Scenario | Model | MCC [95% CI] | QWK [95% CI] | Bal. acc. | F1 macro | Adj. acc. | n test | Perm. p |"
    )
    markdown_rows.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        if not pd.isna(row["MCC_ci_low"]):
            mcc_string = f"{_fmt(row['MCC'])} [{_fmt(row['MCC_ci_low'])}, {_fmt(row['MCC_ci_high'])}]"
        else:
            mcc_string = _fmt(row["MCC"])
        if not pd.isna(row["QWK_ci_low"]):
            qwk_string = f"{_fmt(row['QWK'])} [{_fmt(row['QWK_ci_low'])}, {_fmt(row['QWK_ci_high'])}]"
        else:
            qwk_string = _fmt(row["QWK"])
        if not pd.isna(row["n_test"]):
            n_test_string = f"{int(row['n_test']):,}"
        else:
            n_test_string = "n/a"
        markdown_rows.append(
            f"| {row['Scenario']} | {row['Model']} | {mcc_string} | {qwk_string} | {_fmt(row['Balanced_acc'])} | {_fmt(row['F1_macro'])} | {_fmt(row['Adjacent_acc'])} | {n_test_string} | {_fmt(row['Permutation_p'], dp=4)} |"
        )
    markdown_path = os.path.join(out_dir, "table_1_headline_metrics.md")
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write("# Table 1 - Validation-selected model metrics per scenario\n\n")
        handle.write("\n".join(markdown_rows) + "\n")
    logger(f"[table_1] saved: {csv_path}")

    def _fmt(v, dp=3):
        if v is None or pd.isna(v):
            return "n/a"
        return f"{float(v):.{dp}f}"

    md = []
    md.append(
        "| Scenario | Model | MCC [95% CI] | QWK [95% CI] | Bal. acc. | F1 macro | Adj. acc. | n test | Perm. p |"
    )
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        mcc_str = (
            f"{_fmt(r['MCC'])} [{_fmt(r['MCC_ci_low'])}, {_fmt(r['MCC_ci_high'])}]"
            if not pd.isna(r["MCC_ci_low"])
            else _fmt(r["MCC"])
        )
        qwk_str = (
            f"{_fmt(r['QWK'])} [{_fmt(r['QWK_ci_low'])}, {_fmt(r['QWK_ci_high'])}]"
            if not pd.isna(r["QWK_ci_low"])
            else _fmt(r["QWK"])
        )
        n_str = f"{int(r['n_test']):,}" if not pd.isna(r["n_test"]) else "n/a"
        md.append(
            f"| {r['Scenario']} | {r['Model']} | {mcc_str} | {qwk_str} | {_fmt(r['Balanced_acc'])} | {_fmt(r['F1_macro'])} | {_fmt(r['Adjacent_acc'])} | {n_str} | {_fmt(r['Permutation_p'], dp=4)} |"
        )
    with open(os.path.join(out_dir, "table_1_headline_metrics.md"), "w", encoding="utf-8") as f:
        f.write("# Table 1 - All models metrics per scenario\n\n")
        f.write("\n".join(md) + "\n")
    logger("[table_1] saved")


_GEN_MODE_LABELS = {
    "gene_disjoint": "Gene-disjoint (primary)",
    "disease_disjoint": "Disease-disjoint",
    "gene_and_disease_disjoint": "Gene + disease-disjoint",
}
_GEN_MODE_COLORS = {
    "gene_disjoint": "#0072B2",
    "disease_disjoint": "#E69F00",
    "gene_and_disease_disjoint": "#009E73",
}
_GEN_MODE_ORDER = ["gene_disjoint", "disease_disjoint", "gene_and_disease_disjoint"]
_IMB_LABELS = {
    "sample_weight": "Ordinal weighting (main)",
    "resampling": "Synthetic resampling",
    "none": "No imbalance handling",
}
_IMB_COLORS = {"sample_weight": "#0072B2", "resampling": "#E69F00", "none": "#999999"}
_IMB_ORDER = ["sample_weight", "resampling", "none"]


def _read_result_dataframe(
    pub_outputs: Dict, key: str, output_dir: str, filenames: List[str]
) -> pd.DataFrame:
    """A table from ``pub_outputs``, or else the first readable file in ``<output_dir>/results``."""
    value = (pub_outputs or {}).get(key)
    if isinstance(value, pd.DataFrame) and (not value.empty):
        return value.copy()
    for filename in filenames:
        path = filename if os.path.isabs(filename) else os.path.join(output_dir, "results", filename)
        if not os.path.exists(path):
            continue
        try:
            frame = pd.read_csv(path)
            if not frame.empty:
                return frame
        except Exception:
            continue
    return pd.DataFrame()


def _validation_selected_models(all_results: Dict, pub_outputs: Optional[Dict] = None) -> Dict[str, str]:
    """The validation-selected model of each feature set, from ``pub_outputs['selected_models']``."""
    supplied = (pub_outputs or {})["selected_models"]
    return {
        scenario: str(supplied[scenario])
        for scenario in _present_scenarios(all_results.keys())
        if scenario in supplied
    }


def _candidate_validation_frame(all_results: Dict, pub_outputs: Optional[Dict] = None) -> pd.DataFrame:
    """One row per feature set and candidate model with validation metrics."""
    selected = _validation_selected_models(all_results, pub_outputs)
    rows: List[Dict[str, Any]] = []
    for scenario in _present_scenarios(all_results.keys()):
        for model_name in _stable_models(all_results.get(scenario, {}).get("all", {}).keys()):
            result = all_results[scenario]["all"].get(model_name, {})
            metrics = result.get("val_metrics", {}) or {}
            if not metrics:
                continue
            row = {
                "Scenario": scenario,
                "Model": model_name,
                "selected_on_validation": model_name == selected.get(scenario),
                "evaluation_partition": "outer_validation",
            }
            for metric in (
                "qwk",
                "mcc",
                "balanced_accuracy",
                "f1_macro",
                "f1",
                "auc_roc",
                "ranked_probability_score",
                "multiclass_brier",
            ):
                row[metric] = metrics.get(metric, np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def _selected_test_frame(all_results: Dict, pub_outputs: Dict, output_dir: str) -> pd.DataFrame:
    """Test metrics and bootstrap CIs of the validation-selected models."""
    selected = _validation_selected_models(all_results, pub_outputs)
    bootstrap = _read_result_dataframe(pub_outputs, "bootstrap", output_dir, ["bootstrap_cis.csv"])
    rows: List[Dict[str, Any]] = []
    for scenario in _present_scenarios(selected.keys()):
        model_name = selected[scenario]
        result = all_results.get(scenario, {}).get("all", {}).get(model_name, {})
        metrics = _get_test_metrics(result)
        if not metrics:
            continue
        row: Dict[str, Any] = {
            "Scenario": scenario,
            "Model": model_name,
            "evaluation_partition": "locked_test",
            "n_test": len(result.get("y_true", [])),
        }
        row.update(metrics)
        if not bootstrap.empty and {"Scenario", "Model"}.issubset(bootstrap.columns):
            match = bootstrap[
                (bootstrap["Scenario"].astype(str) == str(scenario))
                & (bootstrap["Model"].astype(str) == str(model_name))
            ]
            if not match.empty:
                for column, value in match.iloc[0].items():
                    if column not in {"Scenario", "Model", "Stratum"}:
                        row[column] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _resolve_partition_indices(
    container, fold_artifacts: Optional[Dict]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Training, validation and test row indices of the primary split."""
    for scenario in SCENARIO_ORDER:
        artifact = (fold_artifacts or {}).get(scenario, {})
        if all((key in artifact for key in ("train_idx", "val_idx", "test_idx"))):
            return (
                np.asarray(artifact["train_idx"], dtype=int),
                np.asarray(artifact["val_idx"], dtype=int),
                np.asarray(artifact["test_idx"], dtype=int),
            )
    raise ValueError("The primary split indices are not available.")


def _coarse_feature_family(name: str, lookup: Optional[Dict[str, str]] = None) -> str:
    lookup = lookup or {}
    family = str(lookup.get(name, "")).strip()
    if family:
        return family
    low = str(name).lower()
    if low.startswith("hpo_"):
        return "hpo_binary"
    if low.startswith("feat_hpo_sys_"):
        return "organ_system"
    if low.startswith("feat_hpo_"):
        return "hpo_derived"
    if low.startswith("feat_mondo_"):
        return "mondo"
    if "inherit" in low or low.startswith("feat_is_"):
        return "inheritance"
    if low.startswith(("feat_gnn_", "has_gnn_")):
        return "gnn_text"
    if low.startswith(("feat_primary_", "has_primary_")):
        return "primary_text"
    if "mouse" in low:
        return "mouse_phenotype"
    if low.startswith("var_") or "variant" in low:
        return "variant"
    if low.startswith("has_") or low.endswith("_missing"):
        return "missingness_indicator"
    return "mechanistic_core"


def _family_display(family: str) -> str:
    labels = {
        "hpo_binary": "HPO terms",
        "hpo_derived": "HPO summaries",
        "organ_system": "Organ systems",
        "mondo": "Disease context",
        "inheritance": "Inheritance",
        "gnn_text": "Functional annotations",
        "primary_text": "Primary text",
        "engineered": "Engineered",
        "mechanistic_core": "Gene-level core",
        "mouse_lethality": "Mouse lethality",
        "mouse_phenotype": "Mouse phenotype",
        "variant": "Variants",
        "external": "External",
        "gene_constraint_v2": "Constraint and expression",
        "missingness_indicator": "Availability",
    }
    return labels.get(family, family.replace("_", " ").capitalize())


_FAMILY_COLORS = {
    "hpo_binary": "#9473A9",
    "hpo_derived": "#B7A4C7",
    "organ_system": "#D4C5DF",
    "mondo": "#4C9A87",
    "inheritance": "#72B89F",
    "gnn_text": "#5F7F9E",
    "primary_text": "#7A9CC6",
    "engineered": "#D8892B",
    "mechanistic_core": "#0072B2",
    "mouse_lethality": "#E69F00",
    "mouse_phenotype": "#F2CF70",
    "variant": "#CC79A7",
    "external": "#999999",
    "gene_constraint_v2": "#56B4E9",
    "missingness_indicator": "#C8C8C8",
}


def _scenario_feature_family_frame(
    container, selected_only: bool = False, fold_artifacts: Optional[Dict] = None
) -> pd.DataFrame:
    lookup = getattr(container, "feature_family_lookup", {}) or {}
    rows: List[Dict[str, Any]] = []
    for scenario in _present_scenarios(getattr(container, "scenario_cols", {}).keys()):
        if selected_only:
            names = list((fold_artifacts or {}).get(scenario, {}).get("selected_names", []))
        else:
            names = list(getattr(container, "scenario_cols", {}).get(scenario, []))
        counts: Dict[str, int] = {}
        for name in names:
            family = _coarse_feature_family(str(name), lookup)
            counts[family] = counts.get(family, 0) + 1
        for family, count in counts.items():
            rows.append(
                {
                    "Scenario": scenario,
                    "family": family,
                    "family_label": _family_display(family),
                    "n_features": int(count),
                    "representation": "selected" if selected_only else "candidate",
                }
            )
    return pd.DataFrame(rows)


def _calibration_tables(pub_outputs: Dict, output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ece = _read_result_dataframe(
        pub_outputs, "ece", output_dir, ["calibration_metrics.csv", "calibration_ece.csv"]
    )
    bins = _read_result_dataframe(pub_outputs, "calibration_bins", output_dir, ["calibration_bins.csv"])
    return (ece, bins)


def _leakage_audit_frame(output_dir: str) -> pd.DataFrame:
    """Extended Data Fig. 2f: candidate features removed per feature set, by reason."""
    rows: List[Dict[str, Any]] = []
    for scenario in SCENARIO_ORDER:
        path = os.path.join(output_dir, "results", f"feature_audit_{scenario}.csv")
        if not os.path.exists(path):
            continue
        frame = pd.read_csv(path)
        for reason, count in frame["drop_reason"].astype(str).value_counts().items():
            rows.append({"Scenario": scenario, "drop_reason": reason, "n": int(count)})
    return pd.DataFrame(rows)


def _importance_for_scenario(results_collector, scenario: str, model_name: Optional[str]) -> pd.DataFrame:
    fi = getattr(results_collector, "feature_importances", {}) or {}
    candidates = []
    if model_name:
        candidates.extend([f"{scenario}_{model_name}"])
    candidates.extend(_model_result_keys(scenario, model_name))
    for key in candidates:
        frame = fi.get(key)
        if isinstance(frame, pd.DataFrame) and (not frame.empty):
            return frame.copy()
    return pd.DataFrame()


def generate_publication_figures(
    all_results: Dict,
    container,
    pub_outputs: Dict,
    results_collector,
    output_dir: str,
    fold_artifacts: Optional[Dict] = None,
    figure_output_dir: Optional[str] = None,
    random_seed: int = 42,
):
    """Draw all figures into ``figure_output_dir`` (default ``<output_dir>/figures/publication``).

    ``output_dir`` is the run whose saved results are read.
    """
    fold_artifacts = fold_artifacts or {}
    source_output_dir = os.path.abspath(str(output_dir))
    if figure_output_dir is None:
        publication_root = os.path.join(source_output_dir, "figures", "publication")
    else:
        publication_root = os.path.abspath(str(figure_output_dir))
    main_dir = os.path.join(publication_root, "main")
    fig_data_dir = os.path.join(publication_root, "figure_data")
    os.makedirs(main_dir, exist_ok=True)
    os.makedirs(fig_data_dir, exist_ok=True)
    output_dir = source_output_dir
    _setup_rcparams()
    labeler = FeatureLabeler(output_dir=output_dir)
    print(f"[figures] source pipeline run: {source_output_dir}")
    print(f"[figures] publication output: {publication_root}")
    print("Extended Data Fig. 1: cohort")
    extended_data_fig_1_cohort(all_results, container, fold_artifacts, main_dir, fig_data_dir, labeler)
    print("Fig. 2: model comparison")
    figure_2_model_comparison(all_results, pub_outputs, output_dir, main_dir, fig_data_dir)
    print("Fig. 3: predictions and calibration")
    figure_3_prediction_calibration(all_results, pub_outputs, output_dir, main_dir, fig_data_dir)
    print("Extended Data Fig. 3: predicted classes before and after calibration")
    extended_data_fig_3_calibration(all_results, pub_outputs, output_dir, main_dir, fig_data_dir)
    print("Extended Data Fig. 2: robustness and audits")
    extended_data_fig_2_robustness(
        all_results, pub_outputs, container, fold_artifacts, output_dir, main_dir, fig_data_dir
    )
    print("Fig. 4: interpretation")
    figure_4_interpretation(
        all_results,
        pub_outputs,
        container,
        results_collector,
        fold_artifacts,
        output_dir,
        main_dir,
        fig_data_dir,
        labeler,
        random_seed=random_seed,
    )
    figure_4_family_summed_shap(output_dir, all_results, pub_outputs, container, fig_data_dir, labeler)
    print("Table 1")
    table_1_headline_metrics(all_results, pub_outputs, main_dir, fig_data_dir)
    _write_main_figure_manifest(main_dir, fig_data_dir)
    print(f"[figures] done: {main_dir}")


def _calibration_change_inputs(all_results: Dict, pub_outputs: Dict, output_dir: str):
    """Tables for Extended Data Fig. 3.

    Built from this run's test predictions when they include the predictions before
    calibration (a pipeline run); otherwise read from the run's saved figure data
    (``run.py figures`` on ``published_run``). Returns None when neither is available.
    """
    selected = _validation_selected_models(all_results, pub_outputs)
    records = []
    for scenario in SCENARIO_ORDER:
        model = selected.get(scenario)
        result = all_results.get(scenario, {}).get("all", {}).get(model, {}) if model else {}
        if not all(key in result for key in ("y_true", "predictions", "predictions_uncalibrated")):
            records = None
            break
        records.append(
            (
                scenario,
                model,
                result["y_true"],
                result["predictions_uncalibrated"]["y_pred"],
                result["predictions"]["y_pred"],
            )
        )
    if records:
        return calibration_change_tables(records)
    saved = os.path.join(output_dir, "figures", "publication", "figure_data")
    paths = [os.path.join(saved, name) for name in (CONFUSION_FILE, CHANGES_FILE, TRANSITIONS_FILE)]
    if not all(os.path.exists(path) for path in paths):
        return None
    return tuple(pd.read_csv(path) for path in paths)


def extended_data_fig_3_calibration(
    all_results: Dict, pub_outputs: Dict, output_dir: str, out_dir: str, fig_data_dir: str, logger=print
):
    """Confusion matrices of the selected models before (a-d) and after (e-h) calibration."""
    tables = _calibration_change_inputs(all_results, pub_outputs, output_dir)
    if tables is None:
        logger("[extended_data_fig_3] no predictions before calibration in this run; skipping")
        return
    confusion, changes, transitions = tables
    confusion.to_csv(os.path.join(fig_data_dir, CONFUSION_FILE), index=False)
    changes.to_csv(os.path.join(fig_data_dir, CHANGES_FILE), index=False)
    transitions.to_csv(os.path.join(fig_data_dir, TRANSITIONS_FILE), index=False)

    short = ["Mi", "Mo", "Se", "Pr"]
    fig, axes = plt.subplots(2, 4, figsize=(7.09, 4.05))
    image = None
    for col, scenario in enumerate(SCENARIO_ORDER):
        rows = confusion[confusion["Scenario"] == scenario]
        summary = changes[changes["Scenario"] == scenario]
        if rows.empty or summary.empty:
            for row_i in range(2):
                axes[row_i, col].set_axis_off()
            continue
        model = rows["Model"].iloc[0]
        for row_i, stage in enumerate(STAGES):
            ax = axes[row_i, col]
            stage_rows = rows[rows["stage"] == stage]
            cm = (
                stage_rows.pivot(index="true_class", columns="predicted_class", values="count")
                .reindex(index=CLASS_NAMES, columns=CLASS_NAMES)
                .fillna(0)
                .to_numpy(dtype=int)
            )
            norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
            image = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
            for r in range(4):
                for c in range(4):
                    ax.text(
                        c,
                        r,
                        f"{cm[r, c]}\n{norm[r, c]:.0%}",
                        ha="center",
                        va="center",
                        fontsize=5.45,
                        color=_adaptive_text_color(norm[r, c], 0, 1),
                        fontweight="bold" if r == c else "normal",
                    )
            ax.set_xticks(range(4))
            ax.set_xticklabels(short)
            ax.set_yticks(range(4))
            ax.set_yticklabels(short if col == 0 else [])
            ax.set_xlabel("Predicted")
            if col == 0:
                label = "Before calibration" if stage == STAGES[0] else "After calibration"
                ax.set_ylabel(f"{label}\nObserved")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.45)
            if row_i == 0:
                ax.set_title(SCENARIO_SHORT[scenario], loc="left", pad=6, fontsize=8.0, color="#252525")
                ax.text(
                    0.0,
                    1.01,
                    MODEL_SHORT_LABELS.get(model, model),
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=5.3,
                    color="#666666",
                )
            else:
                n_changed = int(summary["n_changed"].iloc[0])
                n_test = int(summary["n_test"].iloc[0])
                ax.set_title(
                    f"{n_changed:,} of {n_test:,} changed class",
                    loc="left",
                    pad=4,
                    fontsize=5.3,
                    color="#444444",
                )
            _panel_label(ax, chr(ord("a") + row_i * 4 + col), x=-0.18)
    if image is not None:
        cax = fig.add_axes([0.93, 0.30, 0.012, 0.40])
        bar = fig.colorbar(image, cax=cax)
        bar.set_label("Within-observed-class proportion")
        bar.outline.set_linewidth(0.4)
        bar.ax.tick_params(labelsize=6.2)
    _save(fig, "extended_data_fig_3_calibration_confusion", out_dir)


NATURE_MIN_FONT_PT = 6.2
MANUSCRIPT_FIGURE_NAMES = {
    "figure_1_study_design": "extended_data_fig_1_cohort_composition",
    "figure_2_model_comparison": "figure_2_model_comparison",
    "figure_3_prediction_and_calibration": "figure_3_prediction_and_calibration",
    "figure_4_robustness_and_audits": "extended_data_fig_2_robustness_and_audits",
    "figure_5_interpretation_and_biology": "figure_4_interpretation_and_biology",
}
_SAVED_FIGURE_NAME_MAP: List[Dict[str, str]] = []


def _save(fig, name: str, out_dir: str):
    """Write PNG, PDF, SVG and TIFF under the manuscript name in MANUSCRIPT_FIGURE_NAMES and
    record the internal and manuscript names in figure_name_map.csv."""
    os.makedirs(out_dir, exist_ok=True)
    manuscript_name = MANUSCRIPT_FIGURE_NAMES.get(name, name)
    _SAVED_FIGURE_NAME_MAP.append(
        {
            "internal_name": name,
            "manuscript_name": manuscript_name,
            "panel_data_prefix": name.split("_")[0] + "_" + name.split("_")[1],
        }
    )
    try:
        pd.DataFrame(_SAVED_FIGURE_NAME_MAP).to_csv(os.path.join(out_dir, "figure_name_map.csv"), index=False)
    except Exception:
        pass
    name = manuscript_name
    base = os.path.join(out_dir, name)
    fig.savefig(base + ".png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(base + ".pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(base + ".svg", bbox_inches="tight", facecolor="white")
    try:
        fig.savefig(
            base + ".tiff",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
            pil_kwargs={"compression": "tiff_lzw"},
        )
    except Exception:
        pass
    plt.close(fig)


def _empty_panel(ax, msg: str):
    """Hide a panel whose data are unavailable."""
    ax.set_axis_off()


def _wrap_label(value: Any, width: int = 24) -> str:
    text = str(value).strip()
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _clean_mondo_columns(columns) -> List[str]:
    excluded = {
        "feat_mondo_depth",
        "feat_mondo_n_classes",
        "feat_has_mondo_record",
        "feat_mondo_record",
        "feat_mondo_count",
        "feat_mondo_available",
    }
    return [c for c in columns if str(c).startswith("feat_mondo_") and c not in excluded]


def _plot_interpretation_nature(
    ax,
    output_dir: str,
    scenario: str,
    model_name: Optional[str],
    results_collector,
    labeler: FeatureLabeler,
    rng_seed: int,
    top_n: int = 6,
):
    obs, source_label, n_rows = _read_scenario_shap_observations(output_dir, scenario)
    rows = []
    scatter = None
    if obs:
        denom = float(n_rows) if n_rows else 1.0
        ranked = sorted(
            ((k, val) for k, val in obs.items() if np.nanstd(np.asarray(val[1], dtype=float)) > 0),
            key=lambda kv: float(np.abs(kv[1][0]).sum()) / denom,
            reverse=True,
        )[:top_n]
        rng = np.random.RandomState(rng_seed)
        for rank, (feature, (shap_values, feature_values)) in enumerate(ranked):
            sv = np.asarray(shap_values, dtype=float)
            fv = np.asarray(feature_values, dtype=float)
            finite = np.isfinite(sv)
            sv = sv[finite]
            fv = fv[finite] if len(fv) == len(finite) else np.full(len(sv), np.nan)
            if not len(sv):
                continue
            good = fv[np.isfinite(fv)]
            if good.size:
                lo, hi = np.percentile(good, [5, 95])
                if hi <= lo:
                    hi = lo + 1e-09
                cvals = np.clip((fv - lo) / (hi - lo), 0, 1)
                cvals = np.where(np.isfinite(cvals), cvals, 0.5)
            else:
                cvals = np.full(len(sv), 0.5)
            yj = np.full(len(sv), rank) + rng.normal(0, 0.06, len(sv))
            scatter = ax.scatter(
                sv,
                yj,
                c=cvals,
                cmap="coolwarm",
                vmin=0,
                vmax=1,
                s=3.1,
                alpha=0.29,
                linewidths=0,
                rasterized=True,
                zorder=2,
            )
            ax.plot([np.median(sv)] * 2, [rank - 0.12, rank + 0.12], color="black", lw=0.7, zorder=3)
            rows.append(
                {
                    "Scenario": scenario,
                    "Model": model_name,
                    "raw_feature": feature,
                    "label": labeler(feature),
                    "mean_abs_shap": float(np.abs(sv).sum() / denom),
                    "source": source_label,
                    "n_explained": n_rows,
                }
            )
        ax.set_yticks(np.arange(len(ranked)))
        ax.margins(y=0.05)
        ax.set_yticklabels([_wrap_label(labeler(f), 24) for f, _ in ranked], fontsize=5.2)
        ax.invert_yaxis()
        ax.axvline(0, color="#777777", lw=0.6, ls=(0, (2, 2)))
        ax.set_xlabel(
            "SHAP value (severity classes)" if source_label == SEVERITY_SHAP_SOURCE else "Signed SHAP value"
        )
        _finish_axis(ax, "x")
        return (pd.DataFrame(rows), scatter)
    importance = _importance_for_scenario(results_collector, scenario, model_name)
    if not importance.empty and "feature" in importance.columns:
        value_col = (
            "mean_abs_shap"
            if "mean_abs_shap" in importance.columns
            else "importance"
            if "importance" in importance.columns
            else None
        )
        if value_col:
            top = importance.nlargest(top_n, value_col).copy()
            yy = np.arange(len(top))
            values = _numeric(top, value_col).fillna(0).to_numpy()
            ax.hlines(yy, 0, values, color="#D7D7D7", lw=1.2)
            ax.scatter(
                values,
                yy,
                s=20,
                color=PAL_SCENARIO.get(scenario, "#4C78A8"),
                edgecolor="black",
                linewidth=0.35,
            )
            ax.set_yticks(yy)
            ax.set_yticklabels([_wrap_label(labeler(f), 24) for f in top["feature"]], fontsize=5.2)
            ax.invert_yaxis()
            ax.set_xlabel("Feature importance")
            _finish_axis(ax, "x")
            top["Scenario"] = scenario
            top["Model"] = model_name
            top["source"] = "native_importance"
            top["label"] = [labeler(f) for f in top["feature"]]
            return (top.rename(columns={"feature": "raw_feature"}), None)
    _empty_panel(ax, "Interpretation unavailable")
    return (pd.DataFrame(), None)


def _setup_rcparams():
    """Matplotlib style shared by all figures."""
    mpl.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.4,
            "axes.titlesize": 8.0,
            "axes.titleweight": "normal",
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "legend.title_fontsize": 6.8,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#333333",
            "text.color": "#333333",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "lines.linewidth": 1.05,
            "lines.markersize": 4.3,
            "patch.linewidth": 0.4,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.7,
            "ytick.major.size": 2.7,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.unicode_minus": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _finish_axis(ax, grid_axis: Optional[str] = None):
    """Light grid on the given axis, no top or right spines."""
    ax.set_axisbelow(True)
    if grid_axis in {"x", "y", "both"}:
        ax.grid(axis=grid_axis, color="#E6E8EB", linewidth=0.48, alpha=0.95, zorder=0, linestyle=(0, (2, 2)))
    ax.tick_params(pad=2.4)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#3A3A3A")
        ax.spines[side].set_linewidth(0.6)


def _panel_label(ax, label: str, x: float = -0.14, y: float = 1.055, fs: float = 9.0):
    kwargs = dict(fontsize=fs, fontweight="bold", va="bottom", ha="left", color="#111111", clip_on=False)
    if hasattr(ax, "get_zlim"):
        ax.text2D(x, y, label, transform=ax.transAxes, **kwargs)
    else:
        ax.text(x, y, label, transform=ax.transAxes, **kwargs)


def _soft_title(ax, title: str, pad: float = 8.0):
    ax.set_title(title, loc="left", pad=pad, fontsize=8.0, color="#252525")


def _adaptive_text_color(value: float, vmin: float, vmax: float) -> str:
    if not np.isfinite(value) or vmax <= vmin:
        return "#222222"
    return "white" if (value - vmin) / (vmax - vmin) > 0.56 else "#222222"


def _small_colorbar(ax, mappable, label: str, size: str = "4.5%", pad: float = 0.06):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    cb = ax.figure.colorbar(mappable, cax=cax)
    cb.set_label(label)
    cb.outline.set_linewidth(0.45)
    cb.ax.tick_params(labelsize=6.2, width=0.45, length=2.2)
    return cb


def _scenario_short_list(values) -> List[str]:
    return [SCENARIO_SHORT.get(v, str(v)) for v in values]


def extended_data_fig_1_cohort(
    all_results: Dict,
    container,
    fold_artifacts: Dict,
    out_dir: str,
    fig_data_dir: str,
    labeler: FeatureLabeler,
    logger=print,
):
    """Extended Data Fig. 1: split composition, pairs per gene, inheritance by severity,
    pairs per disease and feature families per feature set (panels a-e)."""
    y = np.asarray(container.y)
    full_df = container.full_df
    train_idx, val_idx, test_idx = _resolve_partition_indices(container, fold_artifacts)
    fig = plt.figure(figsize=(7.25, 6.05))
    outer = fig.add_gridspec(
        2, 1, height_ratios=[0.96, 1.08], left=0.105, right=0.965, top=0.955, bottom=0.095, hspace=0.62
    )
    top = outer[0].subgridspec(1, 3, wspace=0.52)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.68)
    ax = fig.add_subplot(top[0, 0])
    active = [
        ("Train", np.asarray(train_idx, dtype=int)),
        ("Validation", np.asarray(val_idx, dtype=int)),
        ("Test", np.asarray(test_idx, dtype=int)),
    ]
    active = [(name, idx) for name, idx in active if len(idx)]
    rows_a = []
    ypos = np.arange(len(active))
    left = np.zeros(len(active), dtype=float)
    class_fracs = []
    for _, idx in active:
        counts = np.bincount(y[idx].astype(int), minlength=4)
        class_fracs.append(counts / max(counts.sum(), 1))
    class_fracs = np.asarray(class_fracs)
    for class_index, class_name in enumerate(CLASS_NAMES):
        values = class_fracs[:, class_index]
        ax.barh(
            ypos,
            values,
            left=left,
            height=0.52,
            color=PAL_CLASS[class_name],
            edgecolor="white",
            linewidth=0.55,
            label=class_name,
        )
        for row_index, value in enumerate(values):
            if value >= 0.075:
                ax.text(
                    left[row_index] + value / 2,
                    row_index,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=6.0,
                    color="white" if class_index != 1 else "#1E1E1E",
                    fontweight="bold",
                )
        left += values
    for row_index, (partition_name, idx) in enumerate(active):
        ax.text(
            1.015,
            row_index,
            f"n={len(idx):,}",
            va="center",
            ha="left",
            fontsize=5.8,
            transform=ax.get_yaxis_transform(),
        )
        counts = np.bincount(y[idx].astype(int), minlength=4)
        rows_a.extend(
            (
                {
                    "partition": partition_name,
                    "severity": CLASS_NAMES[class_index],
                    "n": int(counts[class_index]),
                    "proportion": float(counts[class_index] / max(counts.sum(), 1)),
                }
                for class_index in range(4)
            )
        )
    ax.set_yticks(ypos)
    ax.set_yticklabels([name for name, _ in active])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Within-partition proportion")
    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        handlelength=0.9,
        columnspacing=0.9,
    )
    _finish_axis(ax, "x")
    _soft_title(ax, "Severity composition")
    _panel_label(ax, "a", x=-0.18)
    _save_csv(pd.DataFrame(rows_a), "figure_1a_partition_composition.csv", fig_data_dir)
    ax = fig.add_subplot(top[0, 1])
    pairs_per_gene = pd.Series(container.groups).value_counts().to_numpy(dtype=int)
    x_values = np.arange(1, int(np.max(pairs_per_gene)) + 1)
    survival = np.asarray([(pairs_per_gene >= x).sum() for x in x_values], dtype=float)
    ax.step(x_values, survival, where="post", color=PAL_SCENARIO["S1_Mechanistic"], lw=1.25)
    ax.scatter(
        x_values,
        survival,
        s=16,
        color=PAL_SCENARIO["S1_Mechanistic"],
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Gene–disease pairs per gene")
    ax.set_ylabel("Genes with ≥ x pairs")
    median = int(np.median(pairs_per_gene))
    p95 = int(np.percentile(pairs_per_gene, 95))
    ax.text(
        0.98,
        0.96,
        f"Median = {median}\n95th percentile = {p95}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.1,
    )
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    _finish_axis(ax, "both")
    _soft_title(ax, "Gene-level clustering")
    _panel_label(ax, "b", x=-0.18)
    _save_csv(
        pd.DataFrame({"minimum_pairs": x_values, "n_genes": survival.astype(int)}),
        "figure_1b_gene_pair_ccdf.csv",
        fig_data_dir,
    )
    ax = fig.add_subplot(top[0, 2])
    inheritance_columns = ["feat_is_dominant", "feat_is_recessive", "feat_is_xlinked"]
    inheritance_labels = ["Dominant", "Recessive", "X-linked"]
    available = [
        (column, label)
        for column, label in zip(inheritance_columns, inheritance_labels)
        if column in full_df.columns
    ]
    if available:
        matrix = []
        rows_c = []
        for column, inheritance_label in available:
            values = pd.to_numeric(full_df[column], errors="coerce").fillna(0).to_numpy() > 0
            row_values = []
            for class_index, class_name in enumerate(CLASS_NAMES):
                mask = y == class_index
                prevalence = float(values[mask].mean()) if mask.any() else np.nan
                row_values.append(prevalence)
                rows_c.append(
                    {
                        "inheritance": inheritance_label,
                        "severity": class_name,
                        "prevalence": prevalence,
                        "numerator": int((values & mask).sum()),
                        "denominator": int(mask.sum()),
                    }
                )
            matrix.append(row_values)
        matrix = np.asarray(matrix, dtype=float)
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color=_adaptive_text_color(value, 0, 1),
                )
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(CLASS_NAMES, rotation=28, ha="right")
        ax.set_yticks(np.arange(len(available)))
        ax.set_yticklabels([item[1] for item in available])
        colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.08, pad=0.22, aspect=30)
        colorbar.set_label("Prevalence")
        colorbar.ax.xaxis.set_major_formatter(PercentFormatter(1.0, 0))
        colorbar.outline.set_linewidth(0.4)
        for spine in ax.spines.values():
            spine.set_visible(False)
        _save_csv(pd.DataFrame(rows_c), "figure_1c_inheritance_prevalence.csv", fig_data_dir)
    else:
        _empty_panel(ax, "Inheritance indicators unavailable")
    _soft_title(ax, "Inheritance prevalence by severity")
    _panel_label(ax, "c", x=-0.18)
    ax = fig.add_subplot(bottom[0, 0])
    _disease_column = next(
        (column for column in ("mondo_id", "MONDO_ID", "disease_id") if column in full_df.columns), None
    )
    if _disease_column is not None:
        pairs_per_disease = full_df[_disease_column].astype(str).value_counts().to_numpy(dtype=int)
        dx_values = np.arange(1, int(np.max(pairs_per_disease)) + 1)
        dsurvival = np.asarray([(pairs_per_disease >= x).sum() for x in dx_values], dtype=float)
        ax.step(dx_values, dsurvival, where="post", color=PAL_SCENARIO["S1b_Disease_Context"], lw=1.25)
        observed = np.unique(pairs_per_disease)
        ax.scatter(
            observed,
            [(pairs_per_disease >= x).sum() for x in observed],
            s=16,
            color=PAL_SCENARIO["S1b_Disease_Context"],
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Gene–disease pairs per disease")
        ax.set_ylabel("Diseases with ≥ x pairs")
        n_diseases = int(pairs_per_disease.size)
        n_pairs_total = int(pairs_per_disease.sum())
        n_singleton = int((pairs_per_disease == 1).sum())
        largest = np.sort(pairs_per_disease)[::-1]
        top3_share = float(largest[:3].sum()) / max(n_pairs_total, 1)
        ax.text(
            0.98,
            0.96,
            f"{n_diseases:,} diseases\n{n_singleton / max(n_diseases, 1):.0%} carry one pair\nlargest = {int(largest[0]):,} pairs\ntop 3 = {top3_share:.0%} of pairs",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.1,
        )
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.xaxis.set_minor_formatter(NullFormatter())
        _finish_axis(ax, "both")
        _save_csv(
            pd.DataFrame({"minimum_pairs": dx_values, "n_diseases": dsurvival.astype(int)}),
            "figure_1d_disease_pair_ccdf.csv",
            fig_data_dir,
        )
        _disease_sizes = full_df[_disease_column].astype(str).value_counts()
        _name_column = next(
            (column for column in ("disease_name", "Disease", "disease") if column in full_df.columns), None
        )
        _detail = pd.DataFrame(
            {"mondo_id": _disease_sizes.index, "n_pairs": _disease_sizes.to_numpy(dtype=int)}
        )
        if _name_column is not None:
            _lookup = (
                full_df[[_disease_column, _name_column]]
                .astype(str)
                .drop_duplicates(subset=[_disease_column])
                .set_index(_disease_column)[_name_column]
            )
            _detail["disease_name"] = _detail["mondo_id"].map(_lookup).fillna("")
        _detail["percent_of_pairs"] = _detail["n_pairs"] / max(n_pairs_total, 1) * 100.0
        _save_csv(_detail.head(30), "figure_1d_largest_diseases.csv", fig_data_dir)
        _soft_title(ax, "Disease-level clustering")
        _panel_label(ax, "d", x=-0.13)
    else:
        _empty_panel(ax, "Disease identifiers unavailable")
        _soft_title(ax, "Disease-level clustering")
        _panel_label(ax, "d", x=-0.13)
    ax = fig.add_subplot(bottom[0, 1])
    family_frame = _scenario_feature_family_frame(container, selected_only=False)
    if not family_frame.empty:
        family_frame = family_frame.copy()
        family_frame.loc[family_frame["family"] == "mouse_lethality", "family"] = "mouse_phenotype"
        family_frame["family_label"] = family_frame["family"].map(
            lambda fam: "Mouse model phenotype" if fam == "mouse_phenotype" else _family_display(fam)
        )
        family_frame = family_frame.groupby(
            ["Scenario", "family", "family_label", "representation"], as_index=False
        )["n_features"].sum()
        pivot = family_frame.pivot_table(
            index="family", columns="Scenario", values="n_features", aggfunc="sum", fill_value=0
        )
        scenarios = [scenario for scenario in SCENARIO_ORDER if scenario in pivot.columns]
        pivot = pivot.reindex(columns=scenarios)
        order = list(pivot.sum(axis=1).sort_values(ascending=False).head(9).index)
        remainder = pivot.drop(index=order, errors="ignore").sum(axis=0)
        pivot = pivot.reindex(order)
        if np.any(remainder > 0):
            pivot.loc["other"] = remainder
        raw_values = pivot.to_numpy(dtype=float)
        log_values = np.log10(raw_values + 1)
        v_max = max(4.0, float(np.nanmax(log_values)))
        image = ax.imshow(log_values, cmap="Blues", aspect="auto", vmin=0, vmax=v_max)
        for row_index in range(raw_values.shape[0]):
            for column_index in range(raw_values.shape[1]):
                raw_value = raw_values[row_index, column_index]
                log_value = log_values[row_index, column_index]
                ax.text(
                    column_index,
                    row_index,
                    f"{int(raw_value):,}",
                    ha="center",
                    va="center",
                    fontsize=5.8,
                    color=_adaptive_text_color(log_value, 0, v_max),
                )
        ax.set_xticks(np.arange(len(scenarios)))
        ax.set_xticklabels(_scenario_short_list(scenarios))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(
            [
                ("Mouse model phenotype" if family == "mouse_phenotype" else _family_display(family))
                if family != "other"
                else "Other"
                for family in pivot.index
            ],
            fontsize=6.0,
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
        _small_colorbar(ax, image, "log10(candidate features + 1)", size="3.8%", pad=0.05)
        _save_csv(family_frame, "figure_1e_scenario_feature_families.csv", fig_data_dir)
    else:
        _empty_panel(ax, "Scenario features unavailable")
    _soft_title(ax, "Candidate representation by feature family")
    _panel_label(ax, "e", x=-0.13)
    _save(fig, "figure_1_study_design", out_dir)
    logger("[figure_1] saved as five-panel layout (a–e)")


def _plot_candidate_model_lines(ax, frame: pd.DataFrame, metric: str, title: str):
    scenarios = _present_scenarios(frame["Scenario"].unique())
    models = _stable_models(frame["Model"].unique())
    x = np.arange(len(scenarios))
    for model in models:
        sub = frame[frame["Model"] == model].set_index("Scenario").reindex(scenarios)
        values = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
        ax.plot(
            x,
            values,
            marker=MODEL_MARKERS.get(model, "o"),
            markersize=4.1,
            color=PAL_MODEL.get(model, "#777777"),
            alpha=0.86,
            lw=0.9,
            label=MODEL_LABELS.get(model, model),
            zorder=2,
        )
        selected_mask = sub.get("selected_on_validation", pd.Series(False, index=sub.index))
        selected_mask = selected_mask.fillna(False).astype(bool).to_numpy()
        finite = selected_mask & np.isfinite(values)
        if finite.any():
            ax.scatter(
                x[finite],
                values[finite],
                s=52,
                marker=MODEL_MARKERS.get(model, "o"),
                facecolor=PAL_MODEL.get(model, "#777777"),
                edgecolor="black",
                linewidth=0.95,
                zorder=4,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(_scenario_short_list(scenarios))
    ax.set_ylim((-0.02, 1.02) if metric == "mcc" else (0, 1.02))
    if metric == "balanced_accuracy":
        ax.axhline(0.25, color="#888888", lw=0.65, ls=(0, (2, 2)), zorder=0)
    ax.set_ylabel(title)
    _finish_axis(ax, "both")
    _soft_title(ax, title, pad=7)


def _selection_margin_frame(validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in _present_scenarios(validation["Scenario"].unique()):
        sub = validation[validation["Scenario"] == scenario].copy()
        sub["qwk"] = pd.to_numeric(sub["qwk"], errors="coerce")
        sub = sub.dropna(subset=["qwk"]).sort_values("qwk", ascending=False)
        if len(sub) < 2:
            continue
        selected_flag = (
            sub["selected_on_validation"]
            if "selected_on_validation" in sub.columns
            else pd.Series(False, index=sub.index)
        )
        selected_rows = sub[selected_flag.fillna(False).astype(bool)]
        winner = selected_rows.iloc[0] if len(selected_rows) else sub.iloc[0]
        runner = sub[sub["Model"] != winner["Model"]].iloc[0]
        rows.append(
            {
                "Scenario": scenario,
                "winner": winner["Model"],
                "runner_up": runner["Model"],
                "margin": float(winner["qwk"] - runner["qwk"]),
            }
        )
    return pd.DataFrame(rows)


def figure_2_model_comparison(
    all_results: Dict, pub_outputs: Dict, output_dir: str, out_dir: str, fig_data_dir: str, logger=print
):
    validation = _candidate_validation_frame(all_results, pub_outputs)
    test = _selected_test_frame(all_results, pub_outputs, output_dir)
    selected = _validation_selected_models(all_results, pub_outputs)
    if validation.empty:
        logger("[figure_2] no validation candidate metrics; skipping")
        return
    fig = plt.figure(figsize=(7.25, 6.55))
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[0.92, 1.05, 1.05],
        width_ratios=[1.0, 1.0, 1.0],
        left=0.075,
        right=0.975,
        top=0.865,
        bottom=0.105,
        wspace=0.46,
        hspace=0.62,
    )
    for i, (metric, title) in enumerate(
        [("qwk", "QWK"), ("mcc", "MCC"), ("balanced_accuracy", "Balanced accuracy")]
    ):
        ax = fig.add_subplot(gs[0, i])
        _plot_candidate_model_lines(ax, validation, metric, title)
        if i > 0:
            ax.set_ylabel("")
        _panel_label(ax, chr(ord("a") + i), x=-0.16)
    models = _stable_models(validation["Model"].unique())
    handles = [
        mpl.lines.Line2D(
            [],
            [],
            color=PAL_MODEL[m],
            marker=MODEL_MARKERS.get(m, "o"),
            lw=1,
            markersize=4.5,
            label=MODEL_LABELS.get(m, m),
        )
        for m in models
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.985),
        frameon=False,
        ncol=len(handles),
        title="Candidate models; black outline = selected",
        handlelength=1.4,
        columnspacing=1.2,
    )
    ax = fig.add_subplot(gs[1:3, 0:2])
    scenarios = _present_scenarios(test.get("Scenario", pd.Series(dtype=str)).unique())
    metrics = [("mcc", "MCC", "o"), ("qwk", "QWK", "s"), ("balanced_accuracy", "Balanced accuracy", "^")]
    nrows = len(scenarios) * len(metrics)
    positions = np.arange(nrows)
    export_rows = []
    for si, scenario in enumerate(scenarios):
        start = si * len(metrics)
        if si % 2 == 0:
            ax.axhspan(start - 0.45, start + len(metrics) - 0.55, color="#F5F6F7", zorder=0)
        row = test[test["Scenario"] == scenario]
        if row.empty:
            continue
        row = row.iloc[0]
        for mi, (metric, label, marker) in enumerate(metrics):
            ypos = start + mi
            mean = float(row.get(f"{metric}_mean", row.get(metric, np.nan)))
            lo = float(row.get(f"{metric}_ci_low", np.nan))
            hi = float(row.get(f"{metric}_ci_high", np.nan))
            if not np.isfinite(mean):
                continue
            xerr = None
            if np.isfinite(lo) and np.isfinite(hi):
                xerr = np.array([[max(mean - lo, 0)], [max(hi - mean, 0)]])
            color = PAL_SCENARIO.get(scenario, "#777777")
            ax.errorbar(
                mean,
                ypos,
                xerr=xerr,
                fmt=marker,
                markersize=4.9,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.5,
                ecolor=color,
                elinewidth=1.05,
                capsize=2.2,
                zorder=3,
            )
            export_rows.append(
                {
                    "Scenario": scenario,
                    "Model": row.get("Model"),
                    "metric": metric,
                    "estimate": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    labels = []
    for scenario in scenarios:
        labels.extend([f"{SCENARIO_SHORT[scenario]}  {name}" for _, name, _ in metrics])
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=6.1)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Locked-test score (gene-cluster bootstrap 95% CI)")
    _finish_axis(ax, "x")
    common_model = (
        next(iter(set(selected.values()))) if selected and len(set(selected.values())) == 1 else None
    )
    _soft_title(ax, "Locked-test performance", pad=9)
    if common_model:
        ax.text(
            0.995,
            1.035,
            f"Selected family: {MODEL_SHORT_LABELS.get(common_model, common_model)}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.6,
            color="#555555",
        )
    _panel_label(ax, "d", x=-0.2)
    _save_csv(pd.DataFrame(export_rows), "figure_2d_locked_test_forest.csv", fig_data_dir)
    ax = fig.add_subplot(gs[1, 2])
    margins = _selection_margin_frame(validation)
    if not margins.empty:
        margins = (
            margins.set_index("Scenario")
            .reindex(_present_scenarios(margins["Scenario"].unique()))
            .reset_index()
        )
        yy = np.arange(len(margins))
        ax.axvline(0, color="#333333", lw=0.7)
        xmax = max(0.03, float(margins["margin"].max()) * 1.33)
        for yi, row in margins.iterrows():
            color = PAL_SCENARIO.get(row["Scenario"], "#777777")
            ax.hlines(yi, 0, row["margin"], color=color, lw=1.35)
            ax.scatter(row["margin"], yi, s=34, color=color, edgecolor="black", linewidth=0.4)
            ax.text(row["margin"] + xmax * 0.025, yi, f"{row['margin']:.3f}", va="center", fontsize=5.8)
            ax.text(
                xmax * 1.02,
                yi,
                f"vs {MODEL_LABELS.get(row['runner_up'], row['runner_up'])}",
                va="center",
                fontsize=5.55,
                clip_on=False,
            )
        ax.set_yticks(yy)
        ax.set_yticklabels(_scenario_short_list(margins["Scenario"]))
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.set_xlabel("Δ validation QWK versus runner-up")
        _finish_axis(ax, "x")
        _save_csv(margins, "figure_2e_model_selection_margin.csv", fig_data_dir)
    else:
        _empty_panel(ax, "Selection margins unavailable")
    _soft_title(ax, "Model-selection margin")
    _panel_label(ax, "e", x=-0.18)
    ax = fig.add_subplot(gs[2, 2])
    contrasts = [
        ("S1b − S1", "S1_Mechanistic", "S1b_Disease_Context"),
        ("S2 − S1b", "S1b_Disease_Context", "S2_Mech_Pheno"),
        ("S2 − S3", "S3_Phenotype", "S2_Mech_Pheno"),
    ]
    paired_path = os.path.join(output_dir, "results", "paired_scenario_contrasts.csv")
    paired = pd.read_csv(paired_path) if os.path.exists(paired_path) else pd.DataFrame()

    def _paired_interval(base, enhanced, metric):
        if paired.empty:
            return (np.nan, np.nan)
        direct = paired[
            (paired["scenario_a"] == enhanced) & (paired["scenario_b"] == base) & (paired["metric"] == metric)
        ]
        if len(direct):
            row = direct.iloc[0]
            return (float(row["difference_percentile_2.5"]), float(row["difference_percentile_97.5"]))
        flipped = paired[
            (paired["scenario_a"] == base) & (paired["scenario_b"] == enhanced) & (paired["metric"] == metric)
        ]
        if len(flipped):
            row = flipped.iloc[0]
            return (-float(row["difference_percentile_97.5"]), -float(row["difference_percentile_2.5"]))
        return (np.nan, np.nan)

    rows_f = []
    ylabels = []
    yy = []
    for ci, (label, base, enhanced) in enumerate(contrasts):
        if base not in set(test.get("Scenario", [])) or enhanced not in set(test.get("Scenario", [])):
            continue
        rb = test[test["Scenario"] == base].iloc[0]
        re_ = test[test["Scenario"] == enhanced].iloc[0]
        ylabels.append(label)
        yy.append(ci)
        for metric, marker, yoff in (("qwk", "o", +0.14), ("mcc", "s", -0.14)):
            diff = float(re_.get(metric, np.nan) - rb.get(metric, np.nan))
            lo, hi = _paired_interval(base, enhanced, metric)
            if np.isfinite(lo) and np.isfinite(hi):
                ax.errorbar(
                    diff,
                    ci + yoff,
                    xerr=[[diff - lo], [hi - diff]],
                    fmt="none",
                    ecolor="#555555",
                    elinewidth=0.8,
                    capsize=1.8,
                    zorder=2,
                )
            ax.scatter(
                diff,
                ci + yoff,
                s=26,
                marker=marker,
                facecolor="white",
                edgecolor="#222222",
                linewidth=0.8,
                zorder=3,
            )
            anchor = hi if np.isfinite(hi) else diff
            ax.text(anchor + 0.012, ci + yoff, f"{diff:+.3f}", va="center", ha="left", fontsize=5.5)
            rows_f.append(
                {"contrast": label, "metric": metric.upper(), "difference": diff, "ci_low": lo, "ci_high": hi}
            )
    if yy:
        ax.axvline(0, color="#777777", lw=0.7, ls=(0, (2, 2)))
        ax.set_yticks(yy)
        ax.set_yticklabels(ylabels)
        ax.invert_yaxis()
        lows = [r["ci_low"] for r in rows_f if np.isfinite(r["ci_low"])] + [r["difference"] for r in rows_f]
        highs = [r["ci_high"] for r in rows_f if np.isfinite(r["ci_high"])] + [
            r["difference"] for r in rows_f
        ]
        ax.set_xlim(min(-0.05, min(lows) - 0.03), max(0.18, max(highs) + 0.11))
        ax.set_xlabel("Test-set difference (95% CI)")
        ax.text(
            0.98,
            0.03,
            "○ QWK   □ MCC",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.4,
            color="#444444",
        )
        _finish_axis(ax, "x")
        _save_csv(pd.DataFrame(rows_f), "figure_2f_information_source_contrasts.csv", fig_data_dir)
    else:
        _empty_panel(ax, "Representation contrasts unavailable")
    _soft_title(ax, "Paired differences between feature sets")
    _panel_label(ax, "f", x=-0.18)
    _save_csv(validation, "figure_2_candidate_validation_metrics.csv", fig_data_dir)
    _save_csv(test, "figure_2_selected_test_metrics.csv", fig_data_dir)
    _save(fig, "figure_2_model_comparison", out_dir)
    logger("[figure_2] saved aesthetic v3")


def figure_3_prediction_calibration(
    all_results: Dict, pub_outputs: Dict, output_dir: str, out_dir: str, fig_data_dir: str, logger=print
):
    selected = _validation_selected_models(all_results, pub_outputs)
    scenarios = [s for s in SCENARIO_ORDER if s in selected]
    if not scenarios:
        logger("[figure_3] no selected models; skipping")
        return
    ece_df, bins_df = _calibration_tables(pub_outputs, output_dir)
    ordinal_errors = _per_class_ordinal_error_frame(all_results, selected, output_dir)
    fig = plt.figure(figsize=(7.25, 7.45))
    outer = fig.add_gridspec(
        3, 1, height_ratios=[1.0, 1.05, 1.1], left=0.065, right=0.94, top=0.935, bottom=0.075, hspace=0.62
    )
    top_gs = outer[0].subgridspec(1, 4, wspace=0.4)
    mid_gs = outer[1].subgridspec(1, 4, wspace=0.48)
    bot_gs = outer[2].subgridspec(1, 3, width_ratios=[2.3, 1.0, 0.95], wspace=0.62)
    confusion_axes = []
    confusion_export = []
    calibration_export = []
    last_im = None
    for i in range(4):
        ax = fig.add_subplot(top_gs[0, i])
        confusion_axes.append(ax)
        if i >= len(scenarios):
            ax.set_axis_off()
            continue
        scenario = scenarios[i]
        model = selected[scenario]
        result = all_results[scenario]["all"].get(model, {})
        if "y_true" in result and "predictions" in result:
            yt = np.asarray(result["y_true"])
            yp = np.asarray(result["predictions"]["y_pred"])
            cm = confusion_matrix(yt, yp, labels=list(range(4)))
            norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
            last_im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
            for r in range(4):
                for c in range(4):
                    ax.text(
                        c,
                        r,
                        f"{cm[r, c]}\n{norm[r, c]:.0%}",
                        ha="center",
                        va="center",
                        fontsize=5.45,
                        color=_adaptive_text_color(norm[r, c], 0, 1),
                        fontweight="bold" if r == c else "normal",
                    )
                    confusion_export.append(
                        {
                            "Scenario": scenario,
                            "Model": model,
                            "true_class": CLASS_NAMES[r],
                            "predicted_class": CLASS_NAMES[c],
                            "count": int(cm[r, c]),
                            "row_fraction": float(norm[r, c]),
                        }
                    )
            ax.set_xticks(range(4))
            ax.set_xticklabels(["Mi", "Mo", "Se", "Pr"])
            ax.set_yticks(range(4))
            ax.set_yticklabels(["Mi", "Mo", "Se", "Pr"] if i == 0 else [])
            ax.set_xlabel("Predicted")
            if i == 0:
                ax.set_ylabel("Observed")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.45)
        else:
            _empty_panel(ax, "Predictions unavailable")
        ax.set_title(SCENARIO_SHORT[scenario], loc="left", pad=6, fontsize=8.0, color="#252525")
        ax.text(
            0.0,
            1.01,
            MODEL_SHORT_LABELS.get(model, model),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=5.3,
            color="#666666",
        )
        _panel_label(ax, chr(ord("a") + i), x=-0.18)
    if last_im is not None:
        cax = fig.add_axes([0.952, 0.695, 0.012, 0.2])
        cb = fig.colorbar(last_im, cax=cax)
        cb.set_label("Within-observed-class proportion")
        cb.outline.set_linewidth(0.4)
        cb.ax.tick_params(labelsize=6.2)
    for i in range(4):
        ax = fig.add_subplot(mid_gs[0, i])
        if i >= len(scenarios):
            ax.set_axis_off()
            continue
        scenario = scenarios[i]
        model = selected[scenario]
        ax.plot([0, 1], [0, 1], color="#9B9B9B", lw=0.8, ls=(0, (3, 2)), zorder=0)
        if not ece_df.empty and (not bins_df.empty):
            rows = _plot_saved_calibration(ax, scenario, model, bins_df, ece_df)
            if not rows.empty:
                rows = rows.copy()
                rows["Scenario"] = scenario
                rows["Model"] = model
                calibration_export.append(rows)
            for line in ax.lines[1:]:
                line.set_linewidth(1.05)
                line.set_markersize(3.7)
        else:
            _empty_panel(ax, "Calibration unavailable")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Predicted probability")
        if i == 0:
            ax.set_ylabel("Observed frequency")
        else:
            ax.set_ylabel("")
        _finish_axis(ax, "both")
        _soft_title(ax, f"Reliability · {SCENARIO_SHORT[scenario]}", pad=6)
        _panel_label(ax, chr(ord("e") + i), x=-0.18)
    fig.legend(
        handles=[
            mpl.lines.Line2D([], [], color=PAL_CLASS[c], marker="o", lw=1, markersize=3.8, label=c)
            for c in CLASS_NAMES
        ],
        loc="upper center",
        bbox_to_anchor=(0.51, 0.665),
        ncol=4,
        frameon=False,
        title="Severity class",
    )
    ax = fig.add_subplot(bot_gs[0, 0])
    if not ordinal_errors.empty:
        _plot_per_class_ordinal_composition_matrix_panel(
            ax,
            ordinal_errors,
            fig_data_dir,
            export_filename="figure_3i_per_class_ordinal_error_composition.csv",
            legend_bbox=(0.5, 1.08),
        )
    else:
        _empty_panel(ax, "Per-class ordinal errors unavailable")
    _soft_title(ax, "Per-class ordinal error composition", pad=20)
    _panel_label(ax, "i", x=-0.09, y=1.19)
    ax = fig.add_subplot(bot_gs[0, 1])
    cal = ece_df.copy()
    if not cal.empty:
        if "calibration_status" in cal.columns:
            pref = cal[cal["calibration_status"].astype(str).str.lower() == "calibrated"]
            if not pref.empty:
                cal = pref
        cal = cal[cal["Scenario"].isin(scenarios) & cal["severity_class"].isin(CLASS_NAMES)]
        pivot = cal.pivot_table(index="severity_class", columns="Scenario", values="ece", aggfunc="first")
        pivot = pivot.reindex(index=CLASS_NAMES, columns=scenarios) * 100
        im = ax.imshow(
            pivot.to_numpy(),
            cmap="Oranges",
            vmin=0,
            vmax=max(5, float(np.nanmax(pivot.to_numpy()))),
            aspect="auto",
        )
        for r in range(4):
            for c in range(len(scenarios)):
                v = pivot.iloc[r, c]
                if np.isfinite(v):
                    ax.text(
                        c,
                        r,
                        f"{v:.1f}",
                        ha="center",
                        va="center",
                        fontsize=5.7,
                        color=_adaptive_text_color(v, 0, max(5, float(np.nanmax(pivot.to_numpy())))),
                    )
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(_scenario_short_list(scenarios))
        ax.set_yticks(range(4))
        ax.set_yticklabels(CLASS_NAMES, fontsize=6.0)
        [sp.set_visible(False) for sp in ax.spines.values()]
        _small_colorbar(ax, im, "ECE (%)", size="4%", pad=0.05)
        _save_csv(cal, "figure_3j_ece.csv", fig_data_dir)
    else:
        _empty_panel(ax, "ECE unavailable")
    _soft_title(ax, "Calibration error (ECE, %)")
    _panel_label(ax, "j", x=-0.18)
    ax = fig.add_subplot(bot_gs[0, 2])
    probability_rows = []
    for scenario in scenarios:
        model = selected[scenario]
        result = all_results[scenario]["all"].get(model, {})
        calmet = result.get("metrics", {}) or {}
        raw = result.get("test_metrics_uncalibrated", {}) or {}
        probability_rows.append(
            {
                "Scenario": scenario,
                "ΔRPS": float(
                    raw.get("ranked_probability_score", np.nan)
                    - calmet.get("ranked_probability_score", np.nan)
                ),
                "ΔBrier": float(raw.get("multiclass_brier", np.nan) - calmet.get("multiclass_brier", np.nan)),
            }
        )
    prob = pd.DataFrame(probability_rows).set_index("Scenario")
    if prob.notna().any().any():
        arr = prob.to_numpy(dtype=float)
        lim = max(0.01, float(np.nanmax(np.abs(arr))))
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
        im = ax.imshow(arr, cmap="RdBu", norm=norm, aspect="auto")
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                v = arr[r, c]
                if np.isfinite(v):
                    ax.text(
                        c,
                        r,
                        f"{v:+.3f}",
                        ha="center",
                        va="center",
                        fontsize=5.55,
                        color="white" if abs(v) > 0.58 * lim else "#222222",
                    )
        ax.set_xticks(range(2))
        ax.set_xticklabels(["ΔRPS", "ΔBrier"])
        ax.set_yticks(range(len(scenarios)))
        ax.set_yticklabels(_scenario_short_list(scenarios))
        [sp.set_visible(False) for sp in ax.spines.values()]
        _small_colorbar(ax, im, "Raw − calibrated", size="5%", pad=0.05)
        _save_csv(prob.reset_index(), "figure_3k_calibration_improvement.csv", fig_data_dir)
    else:
        _empty_panel(ax, "Probability metrics unavailable")
    _soft_title(ax, "Calibration improvement")
    _panel_label(ax, "k", x=-0.2)
    if confusion_export:
        _save_csv(pd.DataFrame(confusion_export), "figure_3_confusion_matrices.csv", fig_data_dir)
    if calibration_export:
        _save_csv(
            pd.concat(calibration_export, ignore_index=True), "figure_3_calibration_curves.csv", fig_data_dir
        )
    _save(fig, "figure_3_prediction_and_calibration", out_dir)
    logger("[figure_3] saved aesthetic v3")


def _humanize_drop_reason(reason: str) -> str:
    """Short labels for feature-audit drop reasons."""
    key = str(reason).strip().lower()
    exact = {
        "high_mi_provenance_review_not_dropped": "High-MI review\n(retained)",
        "high_mi_provenance_review_noted": "High-MI review\n(retained)",
        "high_missing_training_fold": "Missing in\ntraining fold",
        "hpo_low_frequency_training_fold": "Rare HPO term",
        "low_variance_training_fold": "Low variance",
        "mondo_low_frequency_training_fold": "Rare MONDO class",
        "severity_hpo_blocklist": "Severity-term\nrules",
        "other": "Other",
    }
    if key in exact:
        return exact[key]
    cleaned = re.sub("_training_fold$", "", key)
    cleaned = cleaned.replace("high_", "").replace("_", " ").strip()
    cleaned = cleaned[:1].upper() + cleaned[1:] if cleaned else "Other"
    return _wrap_label(cleaned, 18)


def _per_class_ordinal_error_frame(
    all_results: Dict, selected: Dict[str, str], output_dir: str
) -> pd.DataFrame:
    """Fig. 3i: share of correct, adjacent and non-adjacent predictions per observed class
    for the selected models (results/error_decomposition_per_class.csv)."""
    frame = pd.read_csv(os.path.join(output_dir, "results", "error_decomposition_per_class.csv"))
    keep = frame.apply(
        lambda row: selected.get(str(row.get("Scenario", ""))) == str(row.get("Model", "")), axis=1
    )
    frame = frame[keep].copy()
    frame = frame[frame["Scenario"].isin(SCENARIO_ORDER) & frame["true_class"].isin(CLASS_NAMES)].copy()
    frame["_scenario_order"] = frame["Scenario"].map(
        {name: index for index, name in enumerate(SCENARIO_ORDER)}
    )
    frame["_class_order"] = frame["true_class"].map(CLASS_INDICES)
    return frame.sort_values(["_scenario_order", "_class_order"]).reset_index(drop=True)


def _plot_per_class_ordinal_composition_matrix_panel(
    ax,
    frame: pd.DataFrame,
    fig_data_dir: str,
    export_filename: str = "figure_3i_per_class_ordinal_error_composition.csv",
    legend_bbox: Tuple[float, float] = (0.5, 1.12),
) -> None:
    """Feature set by observed class grid of stacked bars (correct, adjacent, non-adjacent)."""
    scenarios = _present_scenarios(frame["Scenario"].unique())
    if not scenarios:
        _empty_panel(ax, "Per-class ordinal errors unavailable")
        return
    cell_width = 0.84
    cell_height = 0.62
    segment_specs = [
        ("frac_correct", "Correct", COLOR_CORRECT),
        ("frac_adjacent", "Adjacent", COLOR_ADJ_ERROR),
        ("frac_non_adjacent", "Non-adjacent", COLOR_NONADJ_ERROR),
    ]
    export_rows: List[Dict[str, Any]] = []
    for row_index, scenario in enumerate(scenarios):
        scenario_rows = frame[frame["Scenario"] == scenario]
        for class_index, class_name in enumerate(CLASS_NAMES):
            subset = scenario_rows[scenario_rows["true_class"] == class_name]
            if subset.empty:
                continue
            row = subset.iloc[0]
            values = np.asarray([row.get(column, np.nan) for column, _, _ in segment_specs], dtype=float)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            values = np.clip(values, 0.0, 1.0)
            total = float(values.sum())
            if total > 0:
                values /= total
            x_left = class_index - cell_width / 2
            y_bottom = row_index - cell_height / 2
            cumulative = 0.0
            for value, (_, _, color) in zip(values, segment_specs):
                width = value * cell_width
                if width > 0:
                    ax.add_patch(
                        mpl.patches.Rectangle(
                            (x_left + cumulative * cell_width, y_bottom),
                            width,
                            cell_height,
                            facecolor=color,
                            edgecolor="white",
                            linewidth=0.35,
                        )
                    )
                cumulative += value
            correct_fraction = float(values[0])
            if correct_fraction >= 0.17:
                ax.text(
                    x_left + correct_fraction * cell_width / 2,
                    row_index,
                    f"{correct_fraction:.0%}",
                    ha="center",
                    va="center",
                    fontsize=5.1,
                    color="white",
                    fontweight="bold",
                )
            export_rows.append(
                {
                    "Scenario": scenario,
                    "Model": row.get("Model", ""),
                    "true_class": class_name,
                    "n_total": row.get("n_total", np.nan),
                    "frac_correct": float(values[0]),
                    "frac_adjacent": float(values[1]),
                    "frac_non_adjacent": float(values[2]),
                }
            )
    ax.set_xlim(-0.58, len(CLASS_NAMES) - 0.42)
    ax.set_ylim(len(scenarios) - 0.48, -0.52)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(["Mild", "Moderate", "Severe", "Profound"], rotation=16, ha="right")
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(_scenario_short_list(scenarios))
    ax.set_xlabel("Observed severity class")
    ax.set_ylabel("Scenario")
    ax.tick_params(length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles = [
        mpl.patches.Patch(facecolor=color, edgecolor="none", label=label) for _, label, color in segment_specs
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=legend_bbox,
        handlelength=0.9,
        handletextpad=0.35,
        columnspacing=0.8,
        fontsize=5.5,
    )
    _save_csv(pd.DataFrame(export_rows), export_filename, fig_data_dir)


def extended_data_fig_2_robustness(
    all_results: Dict,
    pub_outputs: Dict,
    container,
    fold_artifacts: Dict,
    output_dir: str,
    out_dir: str,
    fig_data_dir: str,
    logger=print,
):
    """Plot split sensitivity, validation sweeps and the feature audit."""
    seed_df = _read_result_dataframe(
        pub_outputs,
        "seed_sensitivity",
        output_dir,
        ["split_sensitivity_fixed_configuration.csv", "seed_sensitivity.csv"],
    )
    mi = _read_result_dataframe(
        pub_outputs, "mi_threshold_sensitivity", output_dir, ["mi_threshold_sensitivity.csv"]
    )
    perm = _read_result_dataframe(pub_outputs, "permutation_test", output_dir, ["permutation_test.csv"])
    budget = _read_result_dataframe(pub_outputs, "budget_sensitivity", output_dir, ["budget_sensitivity.csv"])
    budget_selection = _read_result_dataframe(
        pub_outputs,
        "shared_hpo_budget_selection",
        output_dir,
        ["shared_hpo_budget_training_cv_selection.csv"],
    )
    imbalance = _read_result_dataframe(
        pub_outputs, "imbalance_sensitivity", output_dir, ["imbalance_sensitivity.csv"]
    )
    leakage = _leakage_audit_frame(output_dir)
    fig = plt.figure(figsize=(7.1, 5.3))
    outer = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 1.0], left=0.07, right=0.982, top=0.958, bottom=0.07, hspace=0.62
    )
    top = outer[0].subgridspec(1, 3, wspace=0.5)
    mid = outer[1].subgridspec(1, 3, width_ratios=[0.96, 0.96, 1.34], wspace=0.56)
    ax = fig.add_subplot(top[0, 0])
    if not seed_df.empty and {"Scenario", "mcc"}.issubset(seed_df.columns):
        scenarios = _present_scenarios(seed_df["Scenario"].unique())
        rng = np.random.RandomState(37)
        for xi, scenario in enumerate(scenarios):
            values = _numeric(seed_df[seed_df["Scenario"] == scenario], "mcc").dropna().to_numpy()
            jitter = rng.normal(0, 0.045, len(values))
            ax.scatter(
                np.full(len(values), xi) + jitter,
                values,
                s=13,
                color=PAL_SCENARIO[scenario],
                alpha=0.65,
                edgecolor="none",
            )
            if len(values):
                mean = float(np.mean(values))
                sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                lo, hi = (mean - sd, mean + sd)
                ax.errorbar(
                    xi,
                    mean,
                    yerr=np.array([[mean - lo], [hi - mean]]),
                    fmt="o",
                    markersize=4.6,
                    color=PAL_SCENARIO[scenario],
                    markeredgecolor="black",
                    markeredgewidth=0.45,
                    ecolor="#555555",
                    elinewidth=0.9,
                    capsize=2.4,
                    zorder=4,
                )
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(_scenario_short_list(scenarios))
        ax.set_ylabel("Test-set MCC")
        _finish_axis(ax, "y")
        _save_csv(seed_df, "figure_4a_split_sensitivity.csv", fig_data_dir)
    else:
        _empty_panel(ax, "Split sensitivity unavailable")
    _soft_title(ax, "Split sensitivity")
    _panel_label(ax, "a", x=-0.19)
    ax = fig.add_subplot(top[0, 1])
    if not mi.empty and "Scenario" in mi.columns and ("mcc" in mi.columns):
        threshold_col = "minimum_mi_threshold" if "minimum_mi_threshold" in mi.columns else "mi_threshold"
        for scenario in _present_scenarios(mi["Scenario"].unique()):
            sub = mi[mi["Scenario"] == scenario].sort_values(threshold_col)
            x_values = _numeric(sub, threshold_col)
            y_values = _numeric(sub, "mcc")
            ax.plot(x_values, y_values, marker="o", markersize=4.0, color=PAL_SCENARIO[scenario])
            if len(sub):
                dx = max(float(x_values.max()) * 0.018, 0.0002)
                ax.text(
                    float(x_values.iloc[-1]) + dx,
                    float(y_values.iloc[-1]),
                    SCENARIO_SHORT[scenario],
                    color=PAL_SCENARIO[scenario],
                    va="center",
                    fontsize=5.8,
                )
        ax.set_xlabel("Minimum MI threshold")
        ax.set_ylabel("Validation MCC")
        _finish_axis(ax, "both")
        _save_csv(mi, "figure_4b_mi_threshold.csv", fig_data_dir)
    else:
        _empty_panel(ax, "MI threshold unavailable")
    _soft_title(ax, "Feature-selection threshold")
    _panel_label(ax, "b", x=-0.19)
    ax = fig.add_subplot(top[0, 2])
    if not perm.empty and "Scenario" in perm.columns:
        rows = perm[perm["Scenario"].isin(SCENARIO_ORDER)].copy()
        rows["_o"] = rows["Scenario"].map({scenario: i for i, scenario in enumerate(SCENARIO_ORDER)})
        rows = rows.sort_values("_o")
        yy = np.arange(len(rows))
        count_col = next(
            (
                name
                for name in ("n_permutations_successful", "n_permutations_requested", "n_permutations")
                if name in rows.columns
            ),
            None,
        )
        for yi, (_, row) in enumerate(rows.iterrows()):
            null_mean = float(row.get("permuted_mcc_mean", np.nan))
            observed = float(row.get("observed_mcc", np.nan))
            if np.isfinite(null_mean) and np.isfinite(observed):
                ax.plot([null_mean, observed], [yi, yi], color="#B5B5B5", lw=1.1, zorder=1)
            ax.scatter(null_mean, yi, s=28, facecolor="white", edgecolor="#666666", linewidth=0.8, zorder=3)
            ax.scatter(
                observed,
                yi,
                s=34,
                color=PAL_SCENARIO[row["Scenario"]],
                edgecolor="black",
                linewidth=0.45,
                zorder=4,
            )
            if count_col is not None and pd.notna(row.get(count_col)):
                ax.text(
                    1.01,
                    yi,
                    f"n={int(float(row[count_col]))}",
                    transform=ax.get_yaxis_transform(),
                    va="center",
                    fontsize=5.4,
                    color="#555555",
                )
        ax.set_yticks(yy)
        ax.set_yticklabels(_scenario_short_list(rows["Scenario"]))
        ax.invert_yaxis()
        ax.set_xlabel("Validation MCC")
        null_label = "Gene-block null"
        ax.legend(
            handles=[
                mpl.lines.Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="none",
                    markerfacecolor="white",
                    markeredgecolor="#666666",
                    label=null_label,
                ),
                mpl.lines.Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="none",
                    markerfacecolor="#777777",
                    markeredgecolor="black",
                    label="Observed",
                ),
            ],
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=2,
            fontsize=5.6,
            handletextpad=0.35,
            columnspacing=0.8,
        )
        _finish_axis(ax, "x")
        _save_csv(rows, "figure_4c_permutation.csv", fig_data_dir)
    else:
        _empty_panel(ax, "Permutation analysis unavailable")
    _soft_title(ax, "Observed versus gene-block null")
    _panel_label(ax, "c", x=-0.19)
    ax = fig.add_subplot(mid[0, 0])
    if not budget.empty and {"Scenario", "hpo_binary_budget", "qwk"}.issubset(budget.columns):
        plotted = []
        _finite_caps = sorted(
            {
                float(v)
                for v in _numeric(budget, "hpo_binary_budget").dropna()
                if float(v) < HPO_BUDGET_ALL_ELIGIBLE
            }
        )
        _all_pos = _finite_caps[-1] * 1.3 if _finite_caps else float(HPO_BUDGET_ALL_ELIGIBLE)
        _has_all = bool((_numeric(budget, "hpo_binary_budget") >= HPO_BUDGET_ALL_ELIGIBLE).any())

        def _cap_position(values: pd.Series) -> pd.Series:
            return values.where(values < HPO_BUDGET_ALL_ELIGIBLE, _all_pos)

        for scenario in _present_scenarios(budget["Scenario"].unique()):
            sub = budget[budget["Scenario"] == scenario].sort_values("hpo_binary_budget")
            x_values = _cap_position(_numeric(sub, "hpo_binary_budget"))
            y_values = _numeric(sub, "qwk")
            ax.plot(x_values, y_values, marker="o", markersize=4.0, color=PAL_SCENARIO[scenario])
            if len(sub):
                plotted.append((scenario, float(x_values.iloc[-1]), float(y_values.iloc[-1])))
        if not budget_selection.empty and "selected_for_s2_and_s3" in budget_selection.columns:
            selected_rows = budget_selection[
                budget_selection["selected_for_s2_and_s3"].astype(str).str.lower().isin(["true", "1", "yes"])
            ]
            if not selected_rows.empty:
                cap = float(_numeric(selected_rows, "hpo_binary_budget").dropna().iloc[0])
                cap_label = "all" if cap >= HPO_BUDGET_ALL_ELIGIBLE else f"{cap:g}"
                cap = _all_pos if cap >= HPO_BUDGET_ALL_ELIGIBLE else cap
                ax.axvline(cap, color="#444444", lw=0.75, ls=(0, (2, 2)))
                ax.text(
                    cap,
                    0.04,
                    f"Selected = {cap_label} ",
                    transform=ax.get_xaxis_transform(),
                    ha="right",
                    va="bottom",
                    fontsize=5.7,
                )
        x_span = max(ax.get_xlim()[1] - ax.get_xlim()[0], 1.0)
        for scenario, x_last, y_last in plotted:
            ax.text(
                x_last + 0.015 * x_span,
                y_last,
                SCENARIO_SHORT[scenario],
                color=PAL_SCENARIO[scenario],
                va="center",
                fontsize=5.8,
                clip_on=False,
            )
        if _has_all and _finite_caps:
            _ticks = [t for t in ax.get_xticks() if 0 <= t <= _finite_caps[-1] * 1.05]
            ax.set_xticks(_ticks + [_all_pos])
            ax.set_xticklabels([f"{t:g}" for t in _ticks] + ["All"])
        ax.set_xlabel("Direct-HPO feature cap")
        ax.set_ylabel("Validation QWK")
        _finish_axis(ax, "both")
        _save_csv(budget, "figure_4d_hpo_budget.csv", fig_data_dir)
    else:
        _empty_panel(ax, "HPO capacity unavailable")
    _soft_title(ax, "Shared HPO capacity")
    _panel_label(ax, "d", x=-0.19)
    ax = fig.add_subplot(mid[0, 1])
    if not imbalance.empty and {"Scenario", "imbalance_strategy", "mcc"}.issubset(imbalance.columns):
        scenarios = _present_scenarios(imbalance["Scenario"].unique())
        strategies = [
            s
            for s in ["sample_weight", "resampling", "none"]
            if s in set(imbalance["imbalance_strategy"].astype(str))
        ]
        colors = {"sample_weight": "#0072B2", "resampling": "#E69F00", "none": "#909090"}
        markers = {"sample_weight": "o", "resampling": "s", "none": "D"}
        labels = {"sample_weight": "Weighting", "resampling": "Resampling", "none": "None"}
        yy = np.arange(len(scenarios))
        for yi, scenario in enumerate(scenarios):
            values = []
            for strategy in strategies:
                sub = imbalance[
                    (imbalance["Scenario"] == scenario)
                    & (imbalance["imbalance_strategy"].astype(str) == strategy)
                ]
                values.append(float(_numeric(sub, "mcc").iloc[0]) if len(sub) else np.nan)
            finite = [value for value in values if np.isfinite(value)]
            if finite:
                ax.hlines(yi, min(finite), max(finite), color="#D1D1D1", lw=1.0)
            for strategy, value in zip(strategies, values):
                if np.isfinite(value):
                    ax.scatter(
                        value,
                        yi,
                        s=30,
                        marker=markers[strategy],
                        color=colors[strategy],
                        edgecolor="black",
                        linewidth=0.4,
                        label=labels[strategy] if yi == 0 else None,
                        zorder=3,
                    )
        ax.set_yticks(yy)
        ax.set_yticklabels(_scenario_short_list(scenarios))
        ax.invert_yaxis()
        ax.set_xlabel("Validation MCC")
        ax.legend(
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.02, 0.03),
            ncol=1,
            fontsize=5.25,
            handletextpad=0.3,
            labelspacing=0.25,
            borderaxespad=0,
        )
        _finish_axis(ax, "x")
        _save_csv(imbalance, "figure_4e_imbalance.csv", fig_data_dir)
    else:
        _empty_panel(ax, "Imbalance analysis unavailable")
    _soft_title(ax, "Imbalance handling")
    _panel_label(ax, "e", x=-0.19)
    ax = fig.add_subplot(mid[0, 2])
    if not leakage.empty:
        totals = leakage.groupby("drop_reason")["n"].sum().sort_values(ascending=False)
        reasons = list(totals.head(5).index)
        display = leakage.copy()
        display["display_reason"] = display["drop_reason"].where(
            display["drop_reason"].isin(reasons), "Other"
        )
        pivot = display.pivot_table(
            index="display_reason", columns="Scenario", values="n", aggfunc="sum", fill_value=0
        )
        scenarios = [scenario for scenario in SCENARIO_ORDER if scenario in pivot.columns]
        pivot = pivot.reindex(columns=scenarios)
        raw = pivot.to_numpy(float)
        log_values = np.log10(raw + 1)
        image = ax.imshow(log_values, cmap="Blues", aspect="auto")
        vmax = max(1.0, float(np.nanmax(log_values)))
        for row_index in range(raw.shape[0]):
            for col_index in range(raw.shape[1]):
                ax.text(
                    col_index,
                    row_index,
                    f"{int(raw[row_index, col_index]):,}",
                    ha="center",
                    va="center",
                    fontsize=5.35,
                    color=_adaptive_text_color(log_values[row_index, col_index], 0, vmax),
                )
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(_scenario_short_list(scenarios))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([_humanize_drop_reason(reason) for reason in pivot.index], fontsize=5.45)
        ax.tick_params(axis="y", pad=3)
        for spine in ax.spines.values():
            spine.set_visible(False)
        _small_colorbar(ax, image, "log10(count + 1)", size="3.8%", pad=0.035)
        _save_csv(leakage, "figure_4f_leakage_audit.csv", fig_data_dir)
    else:
        _empty_panel(ax, "Leakage audit unavailable")
    _soft_title(ax, "Feature exclusions and review")
    _panel_label(ax, "f", x=-0.16)
    _save(fig, "figure_4_robustness_and_audits", out_dir)
    logger("[figure_4] saved test split sensitivity, validation sweeps and feature audit")


FIGURE_5_SHARED_FEATURE_COUNT = 50
FIGURE_5_S3_PHENOTYPE_COUNT = 50
_PARENT_HPO_TERM_NAMES = {
    "Abnormality of blood and blood-forming tissues",
    "Abnormality of the cardiovascular system",
    "Abnormality of the digestive system",
    "Abnormality of the ear",
    "Abnormality of the endocrine system",
    "Abnormality of the eye",
    "Abnormality of the genitourinary system",
    "Abnormality of the head or neck",
    "Abnormality of the immune system",
    "Abnormality of the integument",
    "Abnormality of metabolism/homeostasis",
    "Abnormality of the musculoskeletal system",
    "Abnormality of the nervous system",
    "Abnormality of prenatal development or birth",
    "Abnormality of the respiratory system",
    "Constitutional symptom",
    "Growth abnormality",
    "Neoplasm",
}
_PARENT_HPO_TERM_NAMES_FOLDED = {name.casefold() for name in _PARENT_HPO_TERM_NAMES}


def _to_dense_float_matrix(matrix: Any) -> np.ndarray:
    """Dense float array from a dense or sparse matrix."""
    if matrix is None:
        return np.empty((0, 0), dtype=float)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2:
        return np.empty((0, 0), dtype=float)
    return array


def _build_s3_locked_test_pca(
    all_results: Dict, pub_outputs: Dict, container, fold_artifacts: Dict, random_seed: int = 42
) -> Tuple[pd.DataFrame, np.ndarray, Optional[str], str]:
    """Fig. 4e: PCA of the selected S3 feature matrix, fitted on training rows and applied to
    the test rows.

    During a pipeline run the matrices come from ``fold_artifacts``. When figures are redrawn
    from a saved run, the projection saved by that run is reused after checking that it matches
    the saved test labels and predictions.
    """
    scenario = "S3_Phenotype"
    selected = _validation_selected_models(all_results, pub_outputs)
    model_name = selected.get(scenario)
    if model_name is None:
        return (
            pd.DataFrame(),
            np.asarray([np.nan, np.nan]),
            None,
            "No validation-selected S3 model was found.",
        )
    result = all_results.get(scenario, {}).get("all", {}).get(model_name, {})
    saved_projection = pub_outputs.get("s3_locked_test_pca")
    if isinstance(saved_projection, pd.DataFrame) and (not saved_projection.empty):
        saved_true = saved_projection["true_severity_numeric"].to_numpy(dtype=int)
        saved_pred = saved_projection["predicted_severity_numeric"].to_numpy(dtype=int)
        variances = saved_projection[
            ["PC1_explained_variance_ratio", "PC2_explained_variance_ratio"]
        ].to_numpy(dtype=float)
        expected_true = np.asarray(result.get("y_true", []), dtype=int).reshape(-1)
        expected_pred = np.asarray(
            (result.get("predictions", {}) or {}).get("y_pred", []), dtype=int
        ).reshape(-1)
        if not (
            saved_projection["Model"].eq(model_name).all()
            and np.array_equal(saved_true, expected_true)
            and np.array_equal(saved_pred, expected_pred)
        ):
            raise ValueError("The saved S3 projection does not match the saved S3 test predictions.")
        return (saved_projection.copy(), variances[0], str(model_name), "")
    artifact = dict((fold_artifacts or {}).get(scenario, {}) or {})
    full_df = getattr(container, "full_df", None)
    test_idx = np.asarray(artifact.get("test_idx", []), dtype=int)
    X_train = _to_dense_float_matrix(artifact.get("X_train"))
    X_test = _to_dense_float_matrix(artifact.get("X_test"))
    if X_train.size == 0 or X_test.size == 0:
        return (
            pd.DataFrame(),
            np.asarray([np.nan, np.nan]),
            str(model_name),
            "The selected S3 training and test matrices are not available.",
        )
    train_source = "in-memory selected X_train"
    test_source = "in-memory selected X_test"
    y_true = artifact.get("y_test")
    if y_true is None:
        y_true = result.get("y_true")
    y_pred = (result.get("predictions", {}) or {}).get("y_pred")
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=int).reshape(-1)
    n_test = min(len(X_test), len(y_true), len(y_pred))
    X_test = X_test[:n_test]
    y_true = y_true[:n_test]
    y_pred = y_pred[:n_test]
    train_finite = np.where(np.isfinite(X_train), X_train, np.nan)
    medians = np.nanmedian(train_finite, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    X_train = np.where(np.isfinite(X_train), X_train, medians)
    X_test = np.where(np.isfinite(X_test), X_test, medians)
    pca = PCA(n_components=2, random_state=random_seed)
    pca.fit(X_train)
    coordinates = pca.transform(X_test)
    frame = pd.DataFrame(
        {
            "Scenario": scenario,
            "Model": str(model_name),
            "test_row": np.arange(n_test, dtype=int),
            "PC1": coordinates[:, 0],
            "PC2": coordinates[:, 1],
            "true_severity_numeric": y_true,
            "predicted_severity_numeric": y_pred,
            "prediction_correct": y_true == y_pred,
        }
    )
    frame["true_severity"] = [
        CLASS_NAMES[value] if 0 <= value < len(CLASS_NAMES) else str(value) for value in y_true
    ]
    frame["predicted_severity"] = [
        CLASS_NAMES[value] if 0 <= value < len(CLASS_NAMES) else str(value) for value in y_pred
    ]
    if full_df is not None and len(test_idx) >= n_test:
        test_rows = full_df.iloc[test_idx[:n_test]].reset_index(drop=True)
        for column in ("gene_symbol", "mondo_id", "disease_name"):
            if column in test_rows.columns:
                frame[column] = test_rows[column].astype(str).to_numpy()
    variance = np.asarray(pca.explained_variance_ratio_[:2], dtype=float)
    frame["PC1_explained_variance_ratio"] = variance[0]
    frame["PC2_explained_variance_ratio"] = variance[1]
    frame["pca_fit_partition"] = "training"
    frame["pca_projection_partition"] = "locked_test"
    frame["n_pca_features"] = int(X_train.shape[1])
    frame["pca_feature_source"] = train_source
    frame["matrix_source_train"] = train_source
    frame["matrix_source_test"] = test_source
    return (frame, variance, str(model_name), "")


def _plot_s3_two_part_pca_panel(
    ax_left,
    ax_right,
    frame: pd.DataFrame,
    variance: np.ndarray,
    model_name: Optional[str],
    unavailable_reason: str = "",
):
    """Fig. 4e: test pairs on the first two principal components, coloured by observed (left)
    and predicted (right) severity.

    A class-stratified sample of about S3_PCA_TARGET_POINTS pairs is drawn. Axis limits leave
    out clusters of at least S3_PCA_DUPLICATE_CLUSTER_MIN identical projections, and the number
    of pairs outside the limits is reported."""
    if frame is None or frame.empty:
        for ax in (ax_left, ax_right):
            ax.set_axis_off()
        message = "S3 phenotype PCA unavailable"
        if unavailable_reason:
            message += "\n" + unavailable_reason
        ax_left.text(
            1.02,
            0.5,
            message,
            transform=ax_left.transAxes,
            ha="center",
            va="center",
            fontsize=6.8,
            color="#666666",
            style="italic",
            wrap=True,
        )
        ax_left.text(
            0.0,
            1.16,
            "S3 phenotype PCA",
            transform=ax_left.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.0,
            color="#252525",
            clip_on=False,
        )
        _panel_label(ax_left, "e", x=-0.18, y=1.15, fs=8.6)
        return
    pc1 = float(variance[0]) if len(variance) > 0 and np.isfinite(variance[0]) else 0.0
    pc2 = float(variance[1]) if len(variance) > 1 and np.isfinite(variance[1]) else 0.0
    model_label = MODEL_LABELS.get(str(model_name), str(model_name or "selected model"))
    x = frame["PC1"].to_numpy(dtype=float)
    y = frame["PC2"].to_numpy(dtype=float)

    def _robust_span(values: np.ndarray) -> float:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            return 1.0
        lo, hi = np.nanpercentile(finite, [2, 98])
        span = float(hi - lo)
        if not np.isfinite(span) or span <= 0:
            span = float(np.nanmax(finite) - np.nanmin(finite))
        return span if np.isfinite(span) and span > 0 else 1.0

    rng = np.random.RandomState(1701)
    display_x = x + rng.normal(0.0, max(0.008, 0.0038 * _robust_span(x)), len(x))
    display_y = y + rng.normal(0.0, max(0.008, 0.0048 * _robust_span(y)), len(y))
    plot_frame = frame.copy()
    plot_frame["_display_PC1"] = display_x
    plot_frame["_display_PC2"] = display_y
    true_values = plot_frame["true_severity_numeric"].astype(int).to_numpy()
    pred_values = plot_frame["predicted_severity_numeric"].astype(int).to_numpy()
    valid = np.isin(true_values, np.arange(len(CLASS_NAMES))) & np.isin(
        pred_values, np.arange(len(CLASS_NAMES))
    )
    misclassified = true_values != pred_values
    shown = np.zeros(len(plot_frame), dtype=bool)
    target_fraction = min(1.0, float(S3_PCA_TARGET_POINTS) / max(int(valid.sum()), 1))
    for class_index in range(len(CLASS_NAMES)):
        indices = np.where(valid & (true_values == class_index))[0]
        if not len(indices):
            continue
        n_take = int(round(target_fraction * len(indices)))
        n_take = min(max(n_take, 1), len(indices))
        shown[rng.choice(indices, size=n_take, replace=False)] = True
    plot_frame = plot_frame.loc[shown].copy().reset_index(drop=True)
    true_values = plot_frame["true_severity_numeric"].astype(int).to_numpy()
    pred_values = plot_frame["predicted_severity_numeric"].astype(int).to_numpy()
    valid = np.isin(true_values, np.arange(len(CLASS_NAMES))) & np.isin(
        pred_values, np.arange(len(CLASS_NAMES))
    )
    misclassified = true_values != pred_values
    finite_xy = np.isfinite(plot_frame["_display_PC1"].to_numpy()) & np.isfinite(
        plot_frame["_display_PC2"].to_numpy()
    )
    if not finite_xy.any():
        for ax in (ax_left, ax_right):
            ax.set_axis_off()
        _empty_panel(ax_left, "Finite PCA coordinates unavailable")
        return

    def _quantile_limits(
        values: np.ndarray, lower: float, upper: float, pad_fraction: float
    ) -> tuple[float, float]:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        lo, hi = np.nanpercentile(finite, [lower, upper])
        span = max(float(hi - lo), 1e-09)
        pad = pad_fraction * span
        return (float(lo - pad), float(hi + pad))

    _full_xy = frame[["PC1", "PC2"]].to_numpy(dtype=float)
    _full_finite = np.isfinite(_full_xy).all(axis=1)
    _core_mask = _full_finite.copy()
    _n_duplicate_clusters = 0
    if _full_finite.any():
        _keys, _counts = np.unique(np.round(_full_xy[_full_finite], 9), axis=0, return_counts=True)
        _dup_keys = _keys[_counts >= int(S3_PCA_DUPLICATE_CLUSTER_MIN)]
        _n_duplicate_clusters = int(len(_dup_keys))
        for _key in _dup_keys:
            _core_mask &= ~(np.isclose(_full_xy[:, 0], _key[0]) & np.isclose(_full_xy[:, 1], _key[1]))
    if _core_mask.sum() < 20:
        _core_mask = _full_finite
        _n_duplicate_clusters = 0
    xlim = _quantile_limits(_full_xy[_core_mask, 0], 0.0, 100.0, 0.07)
    ylim = _quantile_limits(_full_xy[_core_mask, 1], 0.0, 100.0, 0.07)
    _off_axis_mask = (
        (_full_xy[:, 0] < xlim[0])
        | (_full_xy[:, 0] > xlim[1])
        | (_full_xy[:, 1] < ylim[0])
        | (_full_xy[:, 1] > ylim[1])
    )
    _n_off_axis = int(_off_axis_mask.sum())
    _n_duplicate_clusters_dropped_from_limits = int(_n_duplicate_clusters)
    if _n_duplicate_clusters_dropped_from_limits > 0 and _off_axis_mask.any():
        _off_keys = np.unique(np.round(_full_xy[_off_axis_mask & _full_finite], 9), axis=0)
        _dup_key_set = {(round(float(_k[0]), 9), round(float(_k[1]), 9)) for _k in _dup_keys}
        _n_duplicate_clusters = int(
            sum((1 for _k in _off_keys if (round(float(_k[0]), 9), round(float(_k[1]), 9)) in _dup_key_set))
        )
    else:
        _n_duplicate_clusters = 0

    def _scatter_subset(ax, xvals, yvals, facecolors, *, size, edgecolor, linewidth, alpha, zorder):
        if len(xvals) == 0:
            return
        ax.scatter(
            xvals,
            yvals,
            s=size,
            marker="o",
            facecolor=facecolors,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
            rasterized=True,
            zorder=zorder,
        )

    def _draw(ax, colour_col: str, title: str, mark_errors: bool = True, colour_mode: str = "class"):
        if colour_mode == "error":
            gap = np.abs(
                plot_frame["predicted_severity_numeric"].astype(int).to_numpy()
                - plot_frame["true_severity_numeric"].astype(int).to_numpy()
            )
            point_colours = np.asarray(
                [S3_PCA_ERROR_COLOURS[int(min(value, 2))] for value in gap], dtype=object
            )
        else:
            class_values = plot_frame[colour_col].astype(int).to_numpy()
            point_colours = np.asarray(
                [
                    PAL_CLASS[CLASS_NAMES[value]] if 0 <= value < len(CLASS_NAMES) else "#999999"
                    for value in class_values
                ],
                dtype=object,
            )
        draw_seed = 2601 if colour_col == "true_severity_numeric" else 2602
        order = np.random.RandomState(draw_seed).permutation(len(plot_frame))
        correct_order = order[valid[order] & ~misclassified[order]]
        error_order = order[valid[order] & misclassified[order]]
        _scatter_subset(
            ax,
            plot_frame.iloc[correct_order]["_display_PC1"],
            plot_frame.iloc[correct_order]["_display_PC2"],
            point_colours[correct_order],
            size=15.0,
            edgecolor="none",
            linewidth=0.0,
            alpha=0.45,
            zorder=2,
        )
        if mark_errors:
            _scatter_subset(
                ax,
                plot_frame.iloc[error_order]["_display_PC1"],
                plot_frame.iloc[error_order]["_display_PC2"],
                point_colours[error_order],
                size=17.0,
                edgecolor="black" if colour_mode == "class" else "none",
                linewidth=0.4 if colour_mode == "class" else 0.0,
                alpha=0.8,
                zorder=3,
            )
        else:
            _scatter_subset(
                ax,
                plot_frame.iloc[error_order]["_display_PC1"],
                plot_frame.iloc[error_order]["_display_PC2"],
                point_colours[error_order],
                size=15.0,
                edgecolor="none",
                linewidth=0.0,
                alpha=0.45,
                zorder=2,
            )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(title, fontsize=S3_PCA_FS["subplot_title"], pad=6.0)
        _finish_axis(ax, grid_axis="both")
        ax.tick_params(axis="both", labelsize=S3_PCA_FS["tick"], pad=2.5)

    _draw(ax_left, "true_severity_numeric", "Observed severity", mark_errors=False)
    _draw(ax_right, "predicted_severity_numeric", "Predicted severity", mark_errors=True, colour_mode="class")
    logging_note = f"panel e: n={len(frame)} locked-test pairs, display sample={len(plot_frame)}, off-axis={_n_off_axis} in {_n_duplicate_clusters} identical-projection clusters, xlim={xlim}, ylim={ylim}"
    try:
        print("[Fig. 4] " + logging_note)
    except Exception:
        pass
    ax_left.set_ylabel(
        f"PC2 ({pc2:.1%} of training variance)", fontsize=S3_PCA_FS["axis_label"], labelpad=3.0
    )
    ax_left.set_xlabel(
        f"PC1 ({pc1:.1%} of training variance)", fontsize=S3_PCA_FS["axis_label"], labelpad=3.0
    )
    ax_right.set_xlabel(
        f"PC1 ({pc1:.1%} of training variance)", fontsize=S3_PCA_FS["axis_label"], labelpad=3.0
    )
    ax_right.set_ylabel("")
    ax_right.tick_params(labelleft=False)
    ax_left.text(
        0.0,
        1.16,
        f"S3 phenotype principal-component projection · {model_label}",
        transform=ax_left.transAxes,
        ha="left",
        va="bottom",
        fontsize=S3_PCA_FS["panel_title"],
        color="#252525",
        clip_on=False,
    )
    _panel_label(ax_left, "e", x=-0.18, y=1.15, fs=S3_PCA_FS["panel_letter"])
    legend_handles = [
        mpl.lines.Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=6.4,
            markerfacecolor=PAL_CLASS[class_name],
            markeredgecolor="none",
            label=class_name,
        )
        for class_name in CLASS_NAMES
    ]
    legend_handles.append(
        mpl.lines.Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=6.6,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.7,
            label="Misclassified (right panel only)",
        )
    )
    ax_left.legend(
        handles=legend_handles,
        title="Severity class",
        loc="upper center",
        bbox_to_anchor=(1.04, -0.18),
        ncol=min(len(legend_handles), 6),
        frameon=False,
        fontsize=S3_PCA_FS["legend"],
        title_fontsize=S3_PCA_FS["legend"],
        handletextpad=0.45,
        columnspacing=1.4,
        borderaxespad=0.0,
    )
    gap_all = np.abs(
        plot_frame["predicted_severity_numeric"].astype(int).to_numpy()
        - plot_frame["true_severity_numeric"].astype(int).to_numpy()
    )
    return {
        "n_locked_test_pairs": int(len(frame)),
        "n_display_sample": int(len(plot_frame)),
        "n_misclassified_total": int(
            (
                frame["predicted_severity_numeric"].astype(int) != frame["true_severity_numeric"].astype(int)
            ).sum()
        ),
        "n_misclassified_displayed": int((gap_all > 0).sum()),
        "n_adjacent_errors_displayed": int((gap_all == 1).sum()),
        "n_non_adjacent_errors_displayed": int((gap_all >= 2).sum()),
        "sampling_mode": "proportional",
        "displayed_error_fraction": round(float((gap_all > 0).mean()), 4),
        "true_error_fraction": round(
            float(
                (
                    frame["predicted_severity_numeric"].astype(int)
                    != frame["true_severity_numeric"].astype(int)
                ).mean()
            ),
            4,
        ),
        "n_identical_projection_clusters_excluded": int(_n_duplicate_clusters),
        "n_identical_projection_clusters_dropped_from_limits": int(_n_duplicate_clusters_dropped_from_limits),
        "duplicate_cluster_min_size": int(S3_PCA_DUPLICATE_CLUSTER_MIN),
        "n_pairs_off_axis": int(_n_off_axis),
        "xlim_low": float(xlim[0]),
        "xlim_high": float(xlim[1]),
        "ylim_low": float(ylim[0]),
        "ylim_high": float(ylim[1]),
        "pc1_explained_variance_ratio": float(pc1),
        "pc2_explained_variance_ratio": float(pc2),
        "right_panel_mode": "predicted_class",
    }


def figure_4_interpretation(
    all_results: Dict,
    pub_outputs: Dict,
    container,
    results_collector,
    fold_artifacts: Dict,
    output_dir: str,
    out_dir: str,
    fig_data_dir: str,
    labeler: FeatureLabeler,
    random_seed: int = 42,
    logger=print,
):
    """Fig. 4: SHAP summaries of the selected models (a-d) and the S3 PCA projection (e)."""
    selected = _validation_selected_models(all_results, pub_outputs)
    scenarios = [scenario for scenario in SCENARIO_ORDER if scenario in selected]
    fig = plt.figure(figsize=(7.25, 9.65))
    outer = fig.add_gridspec(
        2, 1, height_ratios=[6.7, 3.5], left=0.155, right=0.925, top=0.965, bottom=0.09, hspace=0.34
    )
    top_grid = outer[0].subgridspec(2, 2, wspace=0.84, hspace=0.6)
    interpretation_rows = []
    last_scatter = None
    for panel_index in range(4):
        ax = fig.add_subplot(top_grid[panel_index // 2, panel_index % 2])
        if panel_index >= len(scenarios):
            ax.set_axis_off()
            continue
        scenario = scenarios[panel_index]
        data, scatter = _plot_interpretation_nature(
            ax,
            output_dir,
            scenario,
            selected.get(scenario),
            results_collector,
            labeler,
            random_seed + panel_index,
            top_n=10,
        )
        if not data.empty:
            interpretation_rows.append(data)
        if scatter is not None:
            last_scatter = scatter
        ax.tick_params(axis="y", labelsize=5.55, pad=2.0)
        ax.tick_params(axis="x", labelsize=6.2, pad=2.0)
        ax.set_xlabel(ax.get_xlabel(), fontsize=6.7, labelpad=2.6)
        _soft_title(
            ax,
            f"{SCENARIO_SHORT[scenario]} · {MODEL_LABELS.get(selected[scenario], selected[scenario])}",
            pad=7.0,
        )
        _panel_label(ax, chr(ord("a") + panel_index), x=-0.31, y=1.045, fs=8.6)
    if last_scatter is not None:
        colorbar_axis = fig.add_axes([0.948, 0.515, 0.012, 0.245])
        colorbar = fig.colorbar(last_scatter, cax=colorbar_axis)
        colorbar.set_ticks([0, 1])
        colorbar.set_ticklabels(["Low", "High"])
        colorbar.set_label("Within-feature value", fontsize=7.0)
        colorbar.ax.tick_params(labelsize=6.6)
        colorbar.outline.set_linewidth(0.4)
    if interpretation_rows:
        _save_csv(
            pd.concat(interpretation_rows, ignore_index=True),
            "figure_5a_d_scenario_interpretation.csv",
            fig_data_dir,
        )
    pca_frame, variance, pca_model, pca_reason = _build_s3_locked_test_pca(
        all_results, pub_outputs, container, fold_artifacts, random_seed=random_seed
    )
    pca_grid = outer[1].subgridspec(1, 2, wspace=0.18)
    ax_e_left = fig.add_subplot(pca_grid[0, 0])
    ax_e_right = fig.add_subplot(pca_grid[0, 1], sharex=ax_e_left, sharey=ax_e_left)
    _panel_e_stats = _plot_s3_two_part_pca_panel(
        ax_e_left, ax_e_right, pca_frame, variance, pca_model, unavailable_reason=pca_reason
    )
    if _panel_e_stats:
        _save_csv(
            pd.DataFrame(sorted(_panel_e_stats.items()), columns=["quantity", "value"]),
            "figure_5e_panel_stats.csv",
            fig_data_dir,
        )
        logger(
            f"[Fig. 4] panel e: display sample {_panel_e_stats['n_display_sample']} of {_panel_e_stats['n_locked_test_pairs']}; {_panel_e_stats['n_pairs_off_axis']} pairs off-axis in {_panel_e_stats['n_identical_projection_clusters_excluded']} identical-projection clusters"
        )
    if pca_reason:
        logger(f"[Fig. 4] S3 PCA unavailable: {pca_reason}")
    elif isinstance(pub_outputs.get("s3_locked_test_pca"), pd.DataFrame) and pca_frame.equals(
        pub_outputs["s3_locked_test_pca"]
    ):
        logger(f"[Fig. 4] S3 PCA plotted from saved validated projection: n={len(pca_frame)}")
    else:
        logger(
            f"[Fig. 4] S3 PCA plotted: n={len(pca_frame)}, features reconstructed from {pca_frame['matrix_source_test'].iloc[0]}"
        )
    if not pca_frame.empty:
        _save_csv(pca_frame, "figure_5e_s3_true_predicted_pca.csv", fig_data_dir)
    _save(fig, "figure_5_interpretation_and_biology", out_dir)
    logger("[Fig. 4] saved enlarged panels a-d plus one two-part S3 phenotype PCA panel e")


def _write_main_figure_manifest(out_dir: str, fig_data_dir: str) -> None:
    """main_figure_manifest.csv: what each figure shows and which partition it uses."""
    rows = [
        (
            "Extended Data Fig. 1",
            "a-e",
            "Severity composition, pairs per gene, inheritance prevalence, pairs per disease, feature families",
            "all pairs",
            "figure_1*",
        ),
        ("Fig. 2", "a-c", "Candidate model comparison", "validation", "figure_2*"),
        (
            "Fig. 2",
            "d-f",
            "Test performance of the selected models, selection margin, feature-set contrasts",
            "test; selected models",
            "figure_2*",
        ),
        ("Fig. 3", "a-h", "Confusion matrices and reliability curves", "test; selected models", "figure_3*"),
        (
            "Fig. 3",
            "i-k",
            "Ordinal error composition, ECE, calibration improvement",
            "test; selected models",
            "figure_3*",
        ),
        (
            "Extended Data Fig. 3",
            "a-h",
            "Confusion matrices before and after calibration, pairs that changed class",
            "test; selected models",
            "extended_data_fig_3*",
        ),
        (
            "Extended Data Fig. 2",
            "a-f",
            "Split sensitivity, MI threshold, permutation test, HPO budget, imbalance, feature audit",
            "validation",
            "figure_4*",
        ),
        ("Fig. 4", "a-e", "SHAP summaries and S3 PCA projection", "test; selected models", "figure_5*"),
        ("Table 1", "-", "Test metrics of the selected models", "test; selected models", "table_1*"),
    ]
    frame = pd.DataFrame(
        rows, columns=["figure", "panels", "content", "evaluation_partition", "figure_data_files"]
    )
    frame.to_csv(os.path.join(out_dir, "main_figure_manifest.csv"), index=False)
    _save_csv(frame, "main_figure_manifest.csv", fig_data_dir)
