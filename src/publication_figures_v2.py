"""
publication_figures_v2.py
==========================
Compact Nature-style figure pack for the v4.5 genetic-disease severity
prediction pipeline.
The public entry point is ``generate_publication_figures(...)``.
The orchestrator produces five composite main figures and one main table.
This cleaned edition retains only code reachable from the public entry point;
legacy overrides and unused supplementary builders were removed.
No supplementary directory or supplementary figure is generated.
  Figure 1 - Study design and cohort overview
  Figure 2 - Candidate-model comparison and locked-test performance
  Figure 3 - Class-level behaviour and calibration
  Figure 4 - Robustness, sensitivity, auditing, and generalisation
  Figure 5 - Interpretation, phenotype biology, embedding, and case studies
  Table 1  - Headline locked-test metrics (CSV + Markdown)
Statistical separation is preserved throughout:
  * all candidate families are compared only on the outer validation partition;
  * only the validation-selected frozen/calibrated model is reported on the
  
    locked test partition and receives bootstrap CIs, calibration, and test SHAP;
  * robustness sweeps remain validation-only unless explicitly labelled as a
    locked-test analysis.
Design constraints follow Nature Portfolio conventions:
  * double-column width: 7.1 in (180 mm)
  * sans-serif fonts at 5-8 pt final size
  * 300 dpi RGB PNG plus editable vector PDF
  * colour-vision-safe palettes and no rainbow colour maps
  * shared legends, shared axes, and dense small-multiple layouts
"""
from __future__ import annotations
import os
import re
import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    balanced_accuracy_score,
    cohen_kappa_score,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
try:
    import shap  # noqa: F401
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
try:
    import sklearn
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# 1. CONFIGURATION (single source of truth for the whole file)
# Locked headline pair. Change here to relock on a different (scenario, model).
HEADLINE_SCENARIO = "S2_Mech_Pheno"
HEADLINE_MODEL    = "LightGBM"
CLASS_NAMES   = ["Mild", "Moderate", "Severe", "Profound"]
CLASS_INDICES = {n: i for i, n in enumerate(CLASS_NAMES)}
# Scenario human labels, expanded once. Drives every figure legend / axis.
# The dict KEYS are the pipeline scenario identifiers and must not be renamed:
# they index results, fold artefacts and on-disk output files. Only the VALUES
# are reader-facing. Manuscript naming: S1 = gene-level, S1b = disease-context,
# S2 = combined, S3 = phenotype-only. The mechanistic_core family is displayed
# as Gene-level core.
SCENARIO_LABELS = {
    "S1_Mechanistic":      "S1: Gene-level",
    "S1b_Disease_Context": "S1b: Gene-level + disease context",
    "S2_Mech_Pheno":       "S2: Combined",
    "S3_Phenotype":        "S3: Phenotype-only",
    "S2_blind_diagnostic": "S2-blind: S2 without tier features",
}
SCENARIO_SHORT = {
    "S1_Mechanistic":      "S1",
    "S1b_Disease_Context": "S1b",
    "S2_Mech_Pheno":       "S2",
    "S3_Phenotype":        "S3",
    "S2_blind_diagnostic": "S2-blind",
}
# Stable ordering used everywhere
SCENARIO_ORDER = [
    "S1_Mechanistic",
    "S1b_Disease_Context",
    "S2_Mech_Pheno",
    "S3_Phenotype",
   # "S2_blind_diagnostic",
]
# ----- Color palette (color-vision-safe, no red/green primary contrast) -----
# Per-class palette: distinct hues, blue/green/orange/purple
PAL_CLASS = {
    "Mild":     "#0072B2",  # blue
    "Moderate": "#56B4E9",  # sky blue
    "Severe":   "#E69F00",  # orange
    "Profound": "#CC79A7",  # reddish purple
}
# Per-scenario palette: muted, colour-vision-safe, and stable across figures.
PAL_SCENARIO = {
    "S1_Mechanistic":      "#5F7F9E",
    "S1b_Disease_Context": "#4C9A87",
    "S2_Mech_Pheno":       "#D8892B",
    "S3_Phenotype":        "#9473A9",
    "S2_blind_diagnostic": "#8A8A8A",
}
# Candidate-model registry shared by every figure, legend, and lookup.
MODEL_ORDER = [
    "MultinomialLogisticRegression",
    "ExtraTrees",
    "LightGBM",
    "OrdinalDecomp",
]
MODEL_LABELS = {
    "MultinomialLogisticRegression": "Multinomial logistic",
    "ExtraTrees": "Extra Trees",
    "LightGBM": "LightGBM",
    "OrdinalDecomp": "Ordinal decomposition",
}
# Per-model palette: colour-vision-safe and stable across figures.
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
    "ExtraTrees": "Extra Trees",
    "LightGBM": "LightGBM",
    "OrdinalDecomp": "Ordinal decomp.",
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

# Display names for the functional-annotation feature families.
_GNN_LABEL_PREFIXES = (
    ("feat_gnn_biological_processes_", "GO process: ", "GO biological processes annotated"),
    ("feat_gnn_molecular_activities_", "GO function: ", "GO molecular functions annotated"),
    ("feat_gnn_expression_sites_", "Expressed in: ", "Expression sites annotated"),
    ("feat_gnn_interacting_genes_", "Interacts with: ", "Interaction partners annotated"),
    ("feat_gnn_pathways_", "Pathway: ", "Pathways annotated"),
)

# The pipeline's "all eligible HPO terms" grid step (Config.HPO_BUDGET_ALL_ELIGIBLE).
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
    return not any(pattern in text for pattern in broad_patterns)
# Error decomposition: correct is visually dominant; errors use warm accents.
COLOR_CORRECT      = "#4C78A8"
COLOR_ADJ_ERROR    = "#F2CF70"
COLOR_NONADJ_ERROR = "#E07B54"
NEUTRAL_DARK       = "#333333"
NEUTRAL_MID        = "#777777"

# --- Figure 5 panel e (S3 phenotype PCA) controls -------------------------------
# S3_PCA_STANDARDISE: whether to z-score the binary HPO indicators before PCA.
#   False (default) reproduces the explained-variance ratios in the delivered
#   figure_5e_s3_true_predicted_pca.csv and quoted in the manuscript
#   (PC1 7.3%, PC2 4.9%). Setting it True standardises first, which up-weights
#   rare terms and yields materially larger ratios (about PC1 21.2%, PC2 15.8%);
#   if you change it you must also change the value quoted in the Results.
S3_PCA_STANDARDISE = False
# Minimum size of a group of pairs sharing one exact projection before it is
# treated as a duplicate cluster and excluded from the central-view axis limits.
# Diseases are annotated identically across all their genes, so large exact-
# duplicate clusters exist (e.g. MONDO:0014343 and MONDO:0007542, 571 genes each)
# and otherwise compress the central cloud to a few pixels.
S3_PCA_DUPLICATE_CLUSTER_MIN = 25
# Figure 1 panel d content.
#   "disease_clustering" (default) — complementary cumulative distribution of
#       gene-disease pairs per MONDO disease, mirroring panel b at the disease
#       level. This is the quantity that motivates the disease-disjoint and
#       joint gene-and-disease-disjoint evaluations.
#   "mondo_classes" — the previous bar chart of the most represented specific
#       MONDO disease classes. Retained for reproducibility; uninformative in
#       this corpus, where the largest class covers 4.3% of pairs.
PANEL_D_MODE = "disease_clustering"
# Figure 5/6 panel e, right-hand subplot.
#   "predicted_class"  — colour by predicted severity, misclassified pairs
#       outlined in black. The two subplots then differ only in colouring, so
#       an error is seen as a point that changes colour between them.
#   "error_magnitude"  — colour by ordinal distance between predicted and
#       observed class (0 correct, 1 adjacent, >=2 non-adjacent). This states
#       the two claims the panel is cited for, that errors sit inside the
#       cloud and are almost all adjacent, without asking the reader to
#       compare two near-identical scatter plots.
S3_PCA_RIGHT_PANEL = "predicted_class"
# How panel e chooses which pairs to draw.
#   "proportional" (default) — one common sampling fraction applied within each
#       observed class, without separating correct from misclassified pairs. The
#       displayed error rate then matches the true error rate, so the reader
#       cannot mistake the panel for a statement about how often the model errs.
#   "error_enriched" — the previous rule: keep every misclassified pair, then add
#       a 20% sample of correct pairs. This shows error *location* well but
#       over-represents error *frequency* about threefold (274 of 571 displayed
#       points, 48%, against a true locked-test rate of 274 of 1,750, 15.7%),
#       which combined with the larger error marker makes the dense central
#       region read as though most pairs were misclassified.
S3_PCA_SAMPLING = "proportional"
# Target number of points drawn in panel e under proportional sampling.
S3_PCA_TARGET_POINTS = 700
# Colours for "error_magnitude" mode: correct, adjacent, non-adjacent.
# Chosen to be distinct from every entry of PAL_CLASS, since both schemes share
# one legend; in particular the adjacent-error colour must not read as "Severe".
S3_PCA_ERROR_COLOURS = ("#DCDCDC", "#009E73", "#B2182B")
S3_PCA_ERROR_LABELS = ("Correct", "Adjacent error", "Non-adjacent error")
# Panel-e type sizes, in points, at the rendered figure width.
S3_PCA_FS = {
    "panel_letter": 11.0,
    "panel_title": 10.0,
    "subplot_title": 9.5,
    "axis_label": 9.0,
    "tick": 8.5,
    "annotation": 7.4,
    "legend": 8.2,
}
NEUTRAL_LIGHT      = "#D9D9D9"
# ----- Typography -----

# 2. FEATURE LABEL PRETTIFICATION
# Heuristic mapping for non-HPO features. Editable extras (loaded at runtime)
# can override any of these. Keep keys lowercase for a normalised lookup.
_BUILTIN_NON_HPO_LABELS: Dict[str, str] = {
    # constraint
    "pli":       "pLI (LoF intolerance)",
    "loeuf":     "LOEUF",
    "mis_z":     "Missense Z-score",
    "syn_z":     "Synonymous Z-score",
    "mis_zscore": "Missense Z-score",
    # pathogenicity / variants
    "var_mean_cadd":       "Mean CADD score",
    "var_total_variants":  "Total variants",
    "var_n_pathogenic_clinvar": "Pathogenic variants (ClinVar)",
    "var_n_lof":           "LoF variants",
    "var_n_missense":      "Missense variants",
    # network
    "ppi_degree":     "PPI network degree",
    "string_degree":  "STRING network degree",
    "betweenness":    "Network betweenness",
    "network_degree": "Network degree",
    # expression
    "gtex_tau":               "GTEx tissue specificity (tau)",
    "expression_breadth":     "Tissue expression breadth",
    "n_tissues":              "Tissue expression breadth",
    "tissue_breadth":         "Tissue expression breadth",
    # paralogs / dosage
    "paralog":         "Paralog count",
    "n_paralogs":      "Paralog count",
    "paralog_count":   "Paralog count",
    "phaplo":          "pHaplo (haploinsufficiency)",
    "ptriplo":         "pTriplo (triplosensitivity)",
    # essentiality
    "depmap":          "DepMap essentiality",
    "essentiality":    "Essentiality score",
    "essential":       "Essential gene flag",
    # structure
    "af_structural_rigidity": "AlphaFold structural rigidity",
    "af_frac_disordered":     "AlphaFold disorder fraction",
    "structural_rigidity":    "Structural rigidity",
    "frac_disordered":        "Disorder fraction",
    # PTM / domain
    "domain_count":   "Domain count",
    "ptm_count":      "PTM count",
    "protein_length": "Protein length",
    # conservation
    "conservation":   "Sequence conservation",
    "phastcons":      "phastCons",
    "phylop":         "phyloP",
    "gerp":           "GERP",
    # tissue criticality
    "critical_brain":     "Critical organ: brain",
    "critical_heart":     "Critical organ: heart",
    "critical_immune":    "Critical organ: immune",
    "critical_lung":      "Critical organ: lung",
    "critical_kidney":    "Critical organ: kidney",
    "n_critical_organs":  "Number of critical organs affected",
    "n_noncritical_organs": "Number of non-critical organs affected",
    "noncritical_sensory": "Non-critical organ: sensory",
    "noncritical_total":   "Non-critical organs (total)",
    "tissue_criticality_score": "Tissue criticality score",
    # inheritance
    "feat_is_dominant":          "Inheritance: dominant",
    "feat_is_recessive":         "Inheritance: recessive",
    "feat_is_xlinked":           "Inheritance: X-linked",
    "feat_has_mixed_inheritance": "Mixed inheritance",
    "feat_inheritance_not_recorded": "Inheritance not recorded",
    "feat_pli_x_dominant":       "pLI x dominant",
    "feat_pli_x_recessive":      "pLI x recessive",
    # gene-level summaries
    "feat_gene_n_diseases":  "Gene: number of associated diseases",
    "feat_pli_squared":      "pLI squared",
    "feat_loeuf_squared":    "LOEUF squared",
    "feat_log_ppi_degree":   "log(1 + PPI degree)",
    # MONDO / IC
    "feat_mondo_depth":         "MONDO ontology depth",
    "feat_mondo_n_classes":     "MONDO disease classes (count)",
    "feat_hpo_max_ic":          "HPO information content (max)",
    "feat_hpo_mean_ic":         "HPO information content (mean)",
    "feat_hpo_median_ic":       "HPO information content (median)",
    "feat_hpo_ic_range":        "HPO IC range",
    "feat_hpo_ic_std":          "HPO IC std",
    "feat_hpo_rare_term_ratio": "HPO rare-term ratio",
}
# HPO system / parent-term tag (e.g. feat_hpo_sys_abnormality_of_the_eye)
_SYS_PREFIX = "feat_hpo_sys_"

class FeatureLabeler:
    """
    Centralized feature-name prettifier.
    Resolution order for any incoming raw feature name:
      1. Override CSV (feature_label_dictionary_extra.csv) - exact match.
      2. HPO dictionary (feature_label_dictionary.csv) - exact match on
         feature_name; returns hpo_label or hpo_id-prefixed label.
      3. Built-in non-HPO map - exact match on the lowercase normalized name.
      4. Heuristic prefix stripper + Title Case - last resort.
    Always returns a plain str.
    """
    def __init__(self, output_dir: Optional[str] = None):
        self.hpo_dict: Dict[str, str] = {}
        self.extra_dict: Dict[str, str] = {}
        self.hpo_id_name: Dict[str, str] = {}
        self._load_hpo_id_names(output_dir)
        if output_dir is not None:
            self._load_dictionaries(output_dir)
    def _load_hpo_id_names(self, output_dir: Optional[str] = None):
        """Build HP:id -> term name so features named hpo_HP_0002977 become
        readable. Prefers hp.obo.pkl, falls back to parsing hp.obo."""
        import pickle as _pk
        search = ["hp.obo.pkl", os.path.join(os.getcwd(), "hp.obo.pkl")]
        if output_dir:
            search.insert(0, os.path.join(output_dir, "hp.obo.pkl"))
        for p in search:
            if os.path.exists(p):
                try:
                    with open(p, "rb") as _fh:
                        obj = _pk.load(_fh)
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            nm = v.get("name") if isinstance(v, dict) else v
                            if str(k).startswith("HP:") and nm:
                                self.hpo_id_name[str(k)] = str(nm)
                    if self.hpo_id_name:
                        return
                except Exception:
                    pass
        obo_search = ["hp.obo", os.path.join(os.getcwd(), "hp.obo")]
        if output_dir:
            obo_search.insert(0, os.path.join(output_dir, "hp.obo"))
        for p in obo_search:
            if os.path.exists(p):
                try:
                    cur = None
                    for line in open(p, encoding="utf-8", errors="ignore"):
                        line = line.strip()
                        if line == "[Term]":
                            cur = None
                        elif line.startswith("id: HP:"):
                            cur = line[4:].strip()
                        elif line.startswith("name:") and cur:
                            self.hpo_id_name[cur] = line[5:].strip()
                            cur = None
                    if self.hpo_id_name:
                        return
                except Exception:
                    pass
    def _load_dictionaries(self, output_dir: str):
        # 1. HPO dictionary
        cwd_path  = os.path.join(os.getcwd(), "feature_label_dictionary.csv")
        out_path  = os.path.join(output_dir, "feature_label_dictionary.csv")
        for p in (cwd_path, out_path):
            if os.path.exists(p):
                try:
                    df = pd.read_csv(p)
                    if {"feature_name", "hpo_label"}.issubset(df.columns):
                        for _, row in df.iterrows():
                            fname = str(row["feature_name"])
                            label_raw = row.get("hpo_label", "")
                            hpo_id_raw = row.get("hpo_id", "")
                            label = str(label_raw).strip() if not pd.isna(label_raw) else ""
                            hpo_id = str(hpo_id_raw).strip() if not pd.isna(hpo_id_raw) else ""
                            if not label and not hpo_id:
                                continue
                            display = label if label else hpo_id
                            if label and hpo_id:
                                display = f"{label} ({hpo_id})"
                            self.hpo_dict[fname] = display
                        break
                except Exception:
                    continue
        # 2. Editable extras (overrides everything else)
        extra_path = os.path.join(output_dir, "feature_label_dictionary_extra.csv")
        if os.path.exists(extra_path):
            try:
                df = pd.read_csv(extra_path)
                if {"feature_name", "label"}.issubset(df.columns):
                    for _, row in df.iterrows():
                        self.extra_dict[str(row["feature_name"])] = str(row["label"])
            except Exception:
                pass
        else:
            # Seed an empty editable file so users discover it
            try:
                pd.DataFrame(columns=["feature_name", "label"]).to_csv(
                    extra_path, index=False
                )
            except Exception:
                pass
    # -- public API --
    def __call__(self, name: str, max_len: Optional[int] = None) -> str:
        out = self._prettify(name)
        if max_len and len(out) > max_len:
            out = out[: max_len - 1].rstrip() + "\u2026"
        return out
    # -- internals --
    def _prettify(self, name: str) -> str:
        if not isinstance(name, str):
            name = str(name)
        # 1. Editable extras first
        if name in self.extra_dict:
            return self.extra_dict[name]
        # 2. HPO dictionary (label-derived safe names)
        if name in self.hpo_dict:
            return self.hpo_dict[name]
        # 2b. HPO direct-term columns named by id: hpo_HP_0002977 / feat_hpo_HP_0002977
        m_hp = re.match(r"^(?:feat_)?hpo_HP[_:](\d{7})$", name, re.IGNORECASE)
        if m_hp:
            hp_id = f"HP:{m_hp.group(1)}"
            return self.hpo_id_name.get(hp_id, hp_id)
        # 3. Built-in non-HPO map
        low = name.lower()
        if low in _BUILTIN_NON_HPO_LABELS:
            return _BUILTIN_NON_HPO_LABELS[low]
        # System (parent-term) tag: feat_hpo_sys_abnormality_of_the_eye
        if low.startswith(_SYS_PREFIX):
            tail = name[len(_SYS_PREFIX):].replace("_", " ").strip()
            tail = tail[0].upper() + tail[1:] if tail else tail
            return f"{tail} (parent HPO term)"
        # Tier-derived feature: feat_tier_n_cat_lifespan_infancy
        if low.startswith("feat_tier_"):
            tail = name[len("feat_tier_"):]
            return self._humanize_tier(tail)
        # MONDO: feat_mondo_<class>
        if low.startswith("feat_mondo_"):
            tail = name[len("feat_mondo_"):].replace("_", " ").strip()
            return f"MONDO: {tail}"
        # Functional-annotation features (feat_gnn_<source>_<term> or _count).
        # Without this branch they fell through to the generic rule and were shown
        # as "Gnn expression sites count" (fixed 2026-11-01).
        for prefix, human, count_label in _GNN_LABEL_PREFIXES:
            if low.startswith(prefix):
                tail = name[len(prefix):]
                if tail.lower() == "count":
                    return f"{count_label}, count"
                tail = re.sub(r"_[0-9a-f]{6}$", "...", tail)
                return human + tail.replace("_", " ").strip()
        # Engineered feat_*
        if low.startswith("feat_"):
            tail = name[len("feat_"):].replace("_", " ").strip()
            return tail[0].upper() + tail[1:] if tail else tail
        # GNN/primary text: feat_gnn_pathways_count etc. handled above.
        # As last resort, strip underscores and Title Case
        return name.replace("_", " ").strip().capitalize()
    @staticmethod
    def _humanize_tier(tail: str) -> str:
        # examples: n_cat_lifespan_infancy, frac_cat_intellectual_disability,
        # frac_qol_affected, weighted_severity_score, min_tier
        m = re.match(r"^(n|frac)_cat_(.+)$", tail)
        if m:
            kind = "Count" if m.group(1) == "n" else "Fraction"
            cat = m.group(2).replace("_", " ").strip()
            return f"Tier {kind.lower()}: {cat}"
        m = re.match(r"^(n|frac)_qol_(.+)$", tail)
        if m:
            kind = "count" if m.group(1) == "n" else "fraction"
            cat = m.group(2).replace("_", " ").strip()
            return f"QoL {kind}: {cat}"
        m = re.match(r"^(n|frac)_tier(\d+|nfc)$", tail)
        if m:
            kind = "count" if m.group(1) == "n" else "fraction"
            tier = m.group(2)
            return f"Tier-{tier} {kind}"
        return tail.replace("_", " ").strip()

# 3. UTILITIES
def _save_csv(df: pd.DataFrame, filename: str, fig_data_dir: str):
    os.makedirs(fig_data_dir, exist_ok=True)
    df.to_csv(os.path.join(fig_data_dir, filename), index=False)

def _present_scenarios(scenarios_iter) -> List[str]:
    """Return scenarios in canonical order, only those present."""
    have = set(scenarios_iter)
    return [s for s in SCENARIO_ORDER if s in have]

def _stable_models(models_iter) -> List[str]:
    """Return active candidate models in the pipeline's canonical order."""
    have = set(models_iter)
    return [model for model in MODEL_ORDER if model in have]

def _model_result_keys(
    scenario: str, preferred_model: Optional[str] = None
) -> List[str]:
    """Return current and legacy-compatible result keys without removed models."""
    ordered: List[str] = []
    if preferred_model:
        ordered.append(str(preferred_model))
    ordered.extend(model for model in MODEL_ORDER if model not in ordered)
    aliases = {
        "MultinomialLogisticRegression": ["MLR", "LogisticRegression"],
        "ExtraTrees": ["ET"],
        "LightGBM": ["LGB"],
        "OrdinalDecomp": ["OrdDecomp"],
    }
    keys: List[str] = []
    for model in ordered:
        keys.append(f"{scenario}_{model}")
        for alias in aliases.get(model, []):
            keys.append(f"{scenario}_{alias}")
    return list(dict.fromkeys(keys))
MIN_PERMUTATIONS_FOR_INFERENCE = 20

def _permutation_inference_available(
    frame: Optional[pd.DataFrame],
    minimum: int = MIN_PERMUTATIONS_FOR_INFERENCE,
) -> bool:
    """Require enough successful permutations before plotting intervals or p-values."""
    if frame is None or frame.empty:
        return False
    count_col = next(
        (name for name in (
            "n_permutations_successful",
            "n_permutations_requested",
            "n_permutations",
        ) if name in frame.columns),
        None,
    )
    if count_col is None:
        return False
    counts = pd.to_numeric(frame[count_col], errors="coerce").dropna()
    return bool(len(counts) and (counts >= int(minimum)).all())

def _get_test_metrics(result: Dict) -> Dict[str, float]:
    """Return stored test metrics, or reconstruct core metrics from saved predictions."""
    metrics = result.get("metrics")
    if metrics:
        return metrics
    y_true = result.get("y_true")
    predictions = result.get("predictions", {}) or {}
    y_pred = predictions.get("y_pred")
    if y_true is None or y_pred is None:
        return {}
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }

def _selected_publication_models(
    all_results: Dict,
    pub_outputs: Optional[Dict] = None,
) -> Dict[str, str]:
    """Use the validation-selected models supplied by make_publication.py.
    Only models with saved locked-test predictions are accepted.
    """
    supplied = (pub_outputs or {}).get("selected_models", {}) or {}
    selected: Dict[str, str] = {}
    for scenario, model_name in supplied.items():
        result = (
            all_results.get(scenario, {})
            .get("all", {})
            .get(model_name, {})
        )
        if result and "y_true" in result and "predictions" in result:
            selected[scenario] = str(model_name)
    if selected:
        return selected
    # Compatibility fallback: select only among models that have locked-test outputs.
    for scenario, strata in all_results.items():
        candidates = []
        for model_name, result in strata.get("all", {}).items():
            if "y_true" not in result or "predictions" not in result:
                continue
            validation_metrics = result.get("val_metrics") or {}
            candidates.append((model_name, validation_metrics))
        if candidates:
            selected[scenario] = max(
                candidates,
                key=lambda item: (
                    float(item[1].get("qwk", -np.inf)),
                    float(item[1].get("mcc", -np.inf)),
                    float(item[1].get("balanced_accuracy", -np.inf)),
                ),
            )[0]
    return selected

def _resolve_headline(
    all_results: Dict,
    pub_outputs: Optional[Dict] = None,
) -> Tuple[str, str]:
    """Return a headline pair that has locked-test predictions."""
    selected = _selected_publication_models(all_results, pub_outputs)
    if HEADLINE_SCENARIO in selected:
        return HEADLINE_SCENARIO, selected[HEADLINE_SCENARIO]
    for scenario in SCENARIO_ORDER:
        if scenario in selected:
            return scenario, selected[scenario]
    return HEADLINE_SCENARIO, HEADLINE_MODEL

def _plot_saved_calibration(
    ax,
    scenario: str,
    model_name: str,
    bins_df: pd.DataFrame,
    ece_df: pd.DataFrame,
) -> pd.DataFrame:
    scenario_bins = bins_df[
        (bins_df["Scenario"] == scenario)
        & (bins_df["Model"].astype(str) == str(model_name))
    ].copy()
    scenario_ece = ece_df[
        (ece_df["Scenario"] == scenario)
        & (ece_df["Model"].astype(str) == str(model_name))
    ].copy()
    if "calibration_status" in scenario_bins.columns:
        calibrated = scenario_bins[
            scenario_bins["calibration_status"].astype(str).str.lower() == "calibrated"
        ]
        if not calibrated.empty:
            scenario_bins = calibrated.copy()
    if "calibration_status" in scenario_ece.columns:
        calibrated = scenario_ece[
            scenario_ece["calibration_status"].astype(str).str.lower() == "calibrated"
        ]
        if not calibrated.empty:
            scenario_ece = calibrated.copy()
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6, lw=0.7)
    export_rows = []
    for class_name in CLASS_NAMES:
        class_bins = scenario_bins[
            scenario_bins["severity_class"] == class_name
        ].sort_values("bin")
        class_ece = scenario_ece[
            scenario_ece["severity_class"] == class_name
        ]
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

# 4. FIGURE 2 - PREDICTIVE PERFORMANCE

# 5. FIGURE 3 - MECHANISTIC INTERPRETATION (valid base-model SHAP + heatmap)
_MODEL_SUFFIXES = (
    "_MultinomialLogisticRegression", "_ExtraTrees", "_LightGBM",
    "_OrdinalDecomp", "_MLR", "_LogisticRegression", "_ET",
    "_LGB", "_OrdDecomp",
)

SEVERITY_SHAP_SOURCE = "SHAP on expected severity class"


def _read_scenario_shap_observations(output_dir: str, scenario: str):
    """Return per-sample signed SHAP observations for a scenario, parsed from the
    per-disease explanation CSVs written by the pipeline.
    Returns (obs, source_label, n_rows) where obs maps
        feature_name -> (signed_shap_array, feature_value_array).
    Prefers the base-model class-output explanations (explanations_{scenario}.csv).
    Falls back to the ordinal-boundary explanations
    (ordinal_boundary_explanations_{scenario}.csv) - in several runs that is the
    ONLY per-sample SHAP that was saved. The boundary explanations are base-model,
    cumulative-boundary SHAP (not final calibrated per-class SHAP); callers should
    surface that caveat in the caption/annotation.
    """
    import json
    expl_dir = os.path.join(output_dir, "per_disease_explanations")
    # v4.7: prefer the full-matrix SHAP on the expected severity class written by
    # save_severity_shap (every test pair, every feature, missing flags merged
    # into their parent feature, positive = towards more severe).
    severity_path = os.path.join(expl_dir, f"severity_shap_{scenario}.npz")
    if os.path.exists(severity_path):
        try:
            with np.load(severity_path, allow_pickle=False) as data:
                names = [str(name) for name in data["feature_names"]]
                shap_matrix = np.asarray(data["shap"], dtype=float)
                value_matrix = np.asarray(data["values"], dtype=float)
            obs = {
                name: (shap_matrix[:, j], value_matrix[:, j])
                for j, name in enumerate(names)
            }
            return obs, SEVERITY_SHAP_SOURCE, int(shap_matrix.shape[0])
        except Exception:
            pass
    candidates = [
        (os.path.join(expl_dir, f"explanations_{scenario}.csv"),
         "uncalibrated_base_model_class_output", "base-model class SHAP"),
        (os.path.join(expl_dir, f"ordinal_boundary_explanations_{scenario}.csv"),
         "uncalibrated_ordinal_boundary_model", "ordinal-boundary base-model SHAP"),
    ]
    for path, target, source_label in candidates:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "explanation_target" in df.columns and target:
            mask = df["explanation_target"].eq(target)
            if mask.any():
                df = df.loc[mask]
        n_rows = int(len(df))
        feats: Dict[str, List[List[float]]] = {}
        for js in df.get("top_features_json", []):
            if not isinstance(js, str):
                continue
            try:
                arr = json.loads(js)
            except Exception:
                continue
            for f in arr:
                name = f.get("name", "")
                if not name:
                    continue
                s = f.get("shap_signed", None)
                if s is None:
                    continue
                v = f.get("value", np.nan)
                d = feats.setdefault(name, [[], []])
                d[0].append(float(s))
                try:
                    d[1].append(float(v) if v is not None else np.nan)
                except (TypeError, ValueError):
                    d[1].append(np.nan)
        if feats:
            obs = {k: (np.asarray(a, dtype=float), np.asarray(b, dtype=float))
                   for k, (a, b) in feats.items()}
            return obs, source_label, n_rows
    return {}, None, 0

# 6. FIGURE 4 - ROBUSTNESS AND CALIBRATION

# 7. FIGURE 5 - COMPREHENSIVE MODEL ANALYSIS

# 8. TABLE 1 - HEADLINE METRICS (CSV + Markdown)
def table_2_gene_level_companion(
    pub_outputs: Dict,
    out_dir: str,
    fig_data_dir: str,
    logger=print,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Companion table with the gene, not the gene-disease pair, as the unit.

    Every non-phenotype feature block is gene-level: one row per gene in the
    consolidated feature tables. The label is per gene-disease pair. Where one
    gene causes several diseases of differing severity the model sees an
    identical feature vector and must emit an identical prediction, so that part
    of the label variation is structurally unreachable by S1 and by the
    non-phenotype half of S1b. This table rescores the same locked predictions
    with the gene as the unit. Nothing is retrained, and the pair-level metrics
    in Table 1 remain the primary cross-scenario comparison.
    """
    frame = pub_outputs.get("gene_level_evaluation", pd.DataFrame())
    if frame is None or len(frame) == 0:
        # A standalone figure re-run receives a reconstructed pub_outputs that
        # does not carry this key, so fall back to the file the pipeline wrote.
        if output_dir:
            path = os.path.join(output_dir, "results", "gene_level_evaluation.csv")
            if os.path.exists(path):
                try:
                    frame = pd.read_csv(path)
                    logger(f"[table_2] loaded gene-level evaluation from {path}")
                except Exception as exc:
                    logger(f"[table_2] could not read {path}: {exc}")
                    frame = pd.DataFrame()
    if frame is None or len(frame) == 0:
        logger("[table_2] gene-level evaluation unavailable; skipping")
        return pd.DataFrame()
    keep = [
        "Scenario", "Model", "n_genes", "n_pairs", "mcc", "qwk",
        "balanced_accuracy", "accuracy", "adjacent_accuracy",
        "frac_pairs_in_multi_severity_genes",
    ]
    out = frame[[c for c in keep if c in frame.columns]].copy()
    out = out.rename(columns={
        "mcc": "MCC",
        "qwk": "QWK",
        "balanced_accuracy": "Balanced_acc",
        "accuracy": "Accuracy",
        "adjacent_accuracy": "Adjacent_acc",
        "frac_pairs_in_multi_severity_genes": "Frac_pairs_in_multi_severity_genes",
    })
    if "Scenario" in out.columns:
        out.insert(0, "Scenario_id", out["Scenario"])
        out["Scenario"] = [SCENARIO_LABELS.get(s, s) for s in out["Scenario_id"]]
        order = {s: i for i, s in enumerate(SCENARIO_ORDER)}
        out = out.sort_values(
            "Scenario_id", key=lambda col: col.map(lambda s: order.get(s, 99))
        )
    path = os.path.join(out_dir, "table_2_gene_level_companion.csv")
    out.to_csv(path, index=False)
    _save_csv(out, "table_2_gene_level_companion.csv", fig_data_dir)
    logger(f"[table_2] wrote gene-level companion metrics ({len(out)} rows)")
    return out


def figure_5_family_summed_shap(
    output_dir: str,
    all_results: Dict,
    pub_outputs: Dict,
    container,
    fig_data_dir: str,
    labeler: "FeatureLabeler",
    logger=print,
) -> pd.DataFrame:
    """Aggregate SHAP by feature family, as a companion to the per-feature panel.

    Per-feature SHAP systematically understates any quantity represented by more
    than one column, because a tree splits the same underlying signal across the
    correlated siblings. A single undiplicated binary such as feat_is_dominant
    keeps all of its attribution and therefore looks stronger than a
    multi-column family that collectively contributes more. Summing within
    family removes that artefact. Panels a-d stay per-feature, since reviewers
    expect named features; this table is the check against misreading them.
    """
    # v4.7: when the pipeline wrote group SHAP on the expected severity class
    # (pair-level sums within biologically defined groups), report that instead.
    groups_path = os.path.join(output_dir, "results", "severity_shap_groups.csv")
    if os.path.exists(groups_path):
        try:
            frame = pd.read_csv(groups_path)
        except Exception:
            frame = pd.DataFrame()
        if not frame.empty:
            frame = frame.rename(columns={
                "group": "family_label",
                "mean_abs_group_shap": "summed_mean_abs_shap",
                "top_feature": "largest_single_feature",
                "top_feature_label": "largest_single_label",
                "top_feature_mean_abs_shap": "largest_single_mean_abs_shap",
            })
            frame["source"] = SEVERITY_SHAP_SOURCE
            _save_csv(frame, "figure_5f_family_summed_shap.csv", fig_data_dir)
            logger(
                f"[figure_5] wrote severity-SHAP group summary for "
                f"{frame['Scenario'].nunique()} scenarios"
            )
            return frame
    lookup = getattr(container, "feature_family_lookup", {}) or {}
    selected = _validation_selected_models(all_results, pub_outputs)
    rows: List[Dict[str, Any]] = []
    for scenario in _present_scenarios(all_results.keys()):
        obs, source_label, n_rows = _read_scenario_shap_observations(
            output_dir, scenario
        )
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
            rows.append({
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
                "dilution_ratio": (total / top_value) if top_value else np.nan,
                "source": source_label,
                "n_explained": n_rows,
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        logger("[figure_5] family-summed SHAP unavailable; skipping")
        return frame
    _save_csv(frame, "figure_5f_family_summed_shap.csv", fig_data_dir)
    logger(
        f"[figure_5] wrote family-summed SHAP for {frame['Scenario'].nunique()} scenarios"
    )
    return frame


def table_1_headline_metrics(
    all_results: Dict,
    pub_outputs: Dict,
    out_dir: str,
    fig_data_dir: str,
    logger=print,
):
    bootstrap = pub_outputs.get("bootstrap", pd.DataFrame())
    perm = pub_outputs.get("permutation_test", pd.DataFrame())
    selected_models = pub_outputs.get("selected_models", {}) or {}
    rows = []
    for scenario in _present_scenarios(all_results.keys()):
        all_models = all_results.get(scenario, {}).get("all", {})
        # Prefer the model explicitly selected using validation metrics.
        selected_model = selected_models.get(scenario)
        if selected_model is not None:
            model_names = [selected_model]
        else:
            # Fallback: consider available models, but only include models
            # that actually have locked test metrics.
            model_names = _stable_models(all_models.keys())
        for model_name in model_names:
            result = all_models.get(model_name)
            if not result:
                continue
            # Only validation-selected models were evaluated on the test set.
            metrics = _get_test_metrics(result)
            if not metrics:
                logger(
                    f"[table_1] skipping {scenario}/{model_name}: "
                    "test metrics unavailable"
                )
                continue
            # Bootstrap confidence intervals.
            if (
                not bootstrap.empty
                and {"Scenario", "Model"}.issubset(bootstrap.columns)
            ):
                boot = bootstrap[
                    (bootstrap["Scenario"] == scenario)
                    & (bootstrap["Model"].astype(str) == str(model_name))
                ]
            else:
                boot = pd.DataFrame()
            if not boot.empty:
                bootstrap_row = boot.iloc[0]
                mcc_ci = (
                    bootstrap_row.get("mcc_ci_low", np.nan),
                    bootstrap_row.get("mcc_ci_high", np.nan),
                )
                qwk_ci = (
                    bootstrap_row.get("qwk_ci_low", np.nan),
                    bootstrap_row.get("qwk_ci_high", np.nan),
                )
            else:
                mcc_ci = (np.nan, np.nan)
                qwk_ci = (np.nan, np.nan)
            # Permutation-test p-value.
            perm_model_col = (
                "Model" if "Model" in perm.columns
                else "Observed_Model" if "Observed_Model" in perm.columns
                else None
            )
            if (_permutation_inference_available(perm) and "Scenario" in perm.columns
                    and perm_model_col is not None):
                perm_match = perm[
                    (perm["Scenario"] == scenario)
                    & (perm[perm_model_col].astype(str) == str(model_name))
                ]
            else:
                perm_match = pd.DataFrame()
            if (
                not perm_match.empty
                and "empirical_p_value" in perm_match.columns
            ):
                permutation_p = float(
                    perm_match["empirical_p_value"].iloc[0]
                )
            else:
                permutation_p = np.nan
            # Number of locked test observations.
            if "y_true" in result and result["y_true"] is not None:
                n_test = int(len(result["y_true"]))
            else:
                n_test = np.nan
            rows.append(
                {
                    "Scenario_id": scenario,
                    "Scenario": SCENARIO_LABELS.get(
                        scenario,
                        scenario,
                    ),
                    "Model": model_name,
                    "MCC": float(
                        metrics.get("mcc", np.nan)
                    ),
                    "MCC_ci_low": (
                        float(mcc_ci[0])
                        if not pd.isna(mcc_ci[0])
                        else np.nan
                    ),
                    "MCC_ci_high": (
                        float(mcc_ci[1])
                        if not pd.isna(mcc_ci[1])
                        else np.nan
                    ),
                    "QWK": float(
                        metrics.get("qwk", np.nan)
                    ),
                    "QWK_ci_low": (
                        float(qwk_ci[0])
                        if not pd.isna(qwk_ci[0])
                        else np.nan
                    ),
                    "QWK_ci_high": (
                        float(qwk_ci[1])
                        if not pd.isna(qwk_ci[1])
                        else np.nan
                    ),
                    "Balanced_acc": float(
                        metrics.get(
                            "balanced_accuracy",
                            np.nan,
                        )
                    ),
                    "F1_macro": float(
                        metrics.get(
                            "f1_macro",
                            metrics.get("f1", np.nan),
                        )
                    ),
                    "Adjacent_acc": float(
                        metrics.get(
                            "adjacent_accuracy",
                            np.nan,
                        )
                    ),
                    # Ordinal ranking quality. On a four-class ordinal target with
                    # an irreducible within-gene noise floor, multiclass MCC
                    # understates a model that orders severity correctly but sets
                    # the class thresholds imperfectly. Reported alongside MCC so
                    # the low-signal scenarios are not judged on MCC alone.
                    "Spearman_rho": float(
                        metrics.get("spearman_rho", np.nan)
                    ),
                    "AUC_ge_Moderate": float(
                        metrics.get("auc_threshold_ge1", np.nan)
                    ),
                    "AUC_ge_Severe": float(
                        metrics.get("auc_threshold_ge2", np.nan)
                    ),
                    "AUC_ge_Profound": float(
                        metrics.get("auc_threshold_ge3", np.nan)
                    ),
                    "n_test": n_test,
                    "Permutation_p": permutation_p,
                }
            )
    dataframe = pd.DataFrame(rows)
    csv_path = os.path.join(
        out_dir,
        "table_1_headline_metrics.csv",
    )
    dataframe.to_csv(csv_path, index=False)
    _save_csv(
        dataframe,
        "table_1_headline_metrics.csv",
        fig_data_dir,
    )
    def _fmt(value, dp=3):
        if value is None or pd.isna(value):
            return "n/a"
        return f"{float(value):.{dp}f}"
    markdown_rows = []
    markdown_rows.append(
        "| Scenario | Model | MCC [95% CI] | QWK [95% CI] | "
        "Bal. acc. | F1 macro | Adj. acc. | n test | Perm. p |"
    )
    markdown_rows.append(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    for row in rows:
        if not pd.isna(row["MCC_ci_low"]):
            mcc_string = (
                f"{_fmt(row['MCC'])} "
                f"[{_fmt(row['MCC_ci_low'])}, "
                f"{_fmt(row['MCC_ci_high'])}]"
            )
        else:
            mcc_string = _fmt(row["MCC"])
        if not pd.isna(row["QWK_ci_low"]):
            qwk_string = (
                f"{_fmt(row['QWK'])} "
                f"[{_fmt(row['QWK_ci_low'])}, "
                f"{_fmt(row['QWK_ci_high'])}]"
            )
        else:
            qwk_string = _fmt(row["QWK"])
        if not pd.isna(row["n_test"]):
            n_test_string = f"{int(row['n_test']):,}"
        else:
            n_test_string = "n/a"
        markdown_rows.append(
            f"| {row['Scenario']} "
            f"| {row['Model']} "
            f"| {mcc_string} "
            f"| {qwk_string} "
            f"| {_fmt(row['Balanced_acc'])} "
            f"| {_fmt(row['F1_macro'])} "
            f"| {_fmt(row['Adjacent_acc'])} "
            f"| {n_test_string} "
            f"| {_fmt(row['Permutation_p'], dp=4)} |"
        )
    markdown_path = os.path.join(
        out_dir,
        "table_1_headline_metrics.md",
    )
    with open(
        markdown_path,
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "# Table 1 - Validation-selected model metrics per scenario\n\n"
        )
        handle.write("\n".join(markdown_rows) + "\n")
    logger(
        f"[table_1] saved: {csv_path}"
    )
    # Markdown rendering
    def _fmt(v, dp=3):
        if v is None or pd.isna(v):
            return "n/a"
        return f"{float(v):.{dp}f}"
    md = []
    md.append("| Scenario | Model | MCC [95% CI] | QWK [95% CI] | "
              "Bal. acc. | F1 macro | Adj. acc. | n test | Perm. p |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        mcc_str = (f"{_fmt(r['MCC'])} [{_fmt(r['MCC_ci_low'])}, {_fmt(r['MCC_ci_high'])}]"
                   if not pd.isna(r["MCC_ci_low"]) else _fmt(r["MCC"]))
        qwk_str = (f"{_fmt(r['QWK'])} [{_fmt(r['QWK_ci_low'])}, {_fmt(r['QWK_ci_high'])}]"
                   if not pd.isna(r["QWK_ci_low"]) else _fmt(r["QWK"]))
        n_str = (f"{int(r['n_test']):,}"
                 if not pd.isna(r["n_test"]) else "n/a")
        md.append(f"| {r['Scenario']} | {r['Model']} | {mcc_str} | {qwk_str} | "
                  f"{_fmt(r['Balanced_acc'])} | {_fmt(r['F1_macro'])} | "
                  f"{_fmt(r['Adjacent_acc'])} | {n_str} | "
                  f"{_fmt(r['Permutation_p'], dp=4)} |")
    with open(os.path.join(out_dir, "table_1_headline_metrics.md"),
              "w", encoding="utf-8") as f:
        f.write("# Table 1 - All models metrics per scenario\n\n")
        f.write("\n".join(md) + "\n")
    logger("[table_1] saved")

# 9. SUPPLEMENTARY FIGURES

# 11. FIGURE S5 - PER-SCENARIO SHAP BEESWARM (4-panel)

# 9. FIGURE 7 - 2D PCA AND CASE STUDIES (EVC Severe vs NBN Moderate)

# 10. ORCHESTRATOR

# 12. FIGURE S6 - OUT-OF-GROUP GENERALISATION
# Split-design labels / colours, shared by S6.
_GEN_MODE_LABELS = {
    "gene_disjoint":              "Gene-disjoint (primary)",
    "disease_disjoint":           "Disease-disjoint",
    "gene_and_disease_disjoint":  "Gene + disease-disjoint",
}
_GEN_MODE_COLORS = {
    "gene_disjoint":              "#0072B2",  # blue
    "disease_disjoint":           "#E69F00",  # orange
    "gene_and_disease_disjoint":  "#009E73",  # green
}
_GEN_MODE_ORDER = ["gene_disjoint", "disease_disjoint", "gene_and_disease_disjoint"]

# 13. FIGURE S7 - TOP SEVERITY-DRIVING HPO TERMS (deployable phenotype scorer)

# 14. FIGURE S8 - INFORMATION-SOURCE ABLATION (S1 -> S1b -> S2; S3 alone)

# 15. FIGURE S9 - IMBALANCE-STRATEGY SENSITIVITY (validation-only)
_IMB_LABELS = {
    "sample_weight": "Ordinal weighting (main)",
    "resampling":    "Synthetic resampling",
    "none":          "No imbalance handling",
}
_IMB_COLORS = {
    "sample_weight": "#0072B2",
    "resampling":    "#E69F00",
    "none":          "#999999",
}
_IMB_ORDER = ["sample_weight", "resampling", "none"]

# 16. COMPOSITE MAIN-FIGURE DATA HELPERS
def _read_result_dataframe(
    pub_outputs: Dict,
    key: str,
    output_dir: str,
    filenames: List[str],
) -> pd.DataFrame:
    """Return a publication table from memory, falling back to result CSVs."""
    value = (pub_outputs or {}).get(key)
    if isinstance(value, pd.DataFrame) and not value.empty:
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
    """Resolve one validation-selected model per scenario without using test scores."""
    supplied = (pub_outputs or {}).get("selected_models", {}) or {}
    selected: Dict[str, str] = {}
    for scenario in _present_scenarios(all_results.keys()):
        models = all_results.get(scenario, {}).get("all", {})
        if scenario in supplied and supplied[scenario] in models:
            selected[scenario] = str(supplied[scenario])
            continue
        marked = [
            name for name, result in models.items()
            if bool(result.get("selected_on_validation", False))
        ]
        if len(marked) == 1:
            selected[scenario] = marked[0]
            continue
        candidates = [
            (name, result.get("val_metrics", {}) or {})
            for name, result in models.items()
            if result.get("val_metrics")
        ]
        if candidates:
            selected[scenario] = max(
                candidates,
                key=lambda item: (
                    float(item[1].get("qwk", -np.inf)),
                    float(item[1].get("mcc", -np.inf)),
                    float(item[1].get("balanced_accuracy", -np.inf)),
                ),
            )[0]
    return selected

def _candidate_validation_frame(all_results: Dict, pub_outputs: Optional[Dict] = None) -> pd.DataFrame:
    """One row per scenario and candidate family using outer-validation metrics."""
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
                "qwk", "mcc", "balanced_accuracy", "f1_macro", "f1",
                "auc_roc", "ranked_probability_score", "multiclass_brier",
            ):
                row[metric] = metrics.get(metric, np.nan)
            rows.append(row)
    return pd.DataFrame(rows)

def _selected_test_frame(
    all_results: Dict,
    pub_outputs: Dict,
    output_dir: str,
) -> pd.DataFrame:
    """Locked-test metrics and bootstrap CIs for validation-selected models only."""
    selected = _validation_selected_models(all_results, pub_outputs)
    bootstrap = _read_result_dataframe(
        pub_outputs, "bootstrap", output_dir,
        ["bootstrap_cis.csv"],
    )
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

def _resolve_partition_indices(container, fold_artifacts: Optional[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the exact primary train/validation/test indices where available."""
    fold_artifacts = fold_artifacts or {}
    for scenario in SCENARIO_ORDER:
        artifact = fold_artifacts.get(scenario, {})
        if all(key in artifact for key in ("train_idx", "val_idx", "test_idx")):
            return (
                np.asarray(artifact["train_idx"], dtype=int),
                np.asarray(artifact["val_idx"], dtype=int),
                np.asarray(artifact["test_idx"], dtype=int),
            )
    development = np.asarray(getattr(container, "train_idx", []), dtype=int)
    test = np.asarray(getattr(container, "test_idx", []), dtype=int)
    return development, np.asarray([], dtype=int), test

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
        # Published gene-level constraint and expression sources added in v4.5:
        # s_het, Human Protein Atlas, BrainSpan, DOMINO, GDI.
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

def _scenario_feature_family_frame(container, selected_only: bool = False,
                                   fold_artifacts: Optional[Dict] = None) -> pd.DataFrame:
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
            rows.append({
                "Scenario": scenario,
                "family": family,
                "family_label": _family_display(family),
                "n_features": int(count),
                "representation": "selected" if selected_only else "candidate",
            })
    return pd.DataFrame(rows)

# 17. FIGURE 1 - STUDY DESIGN AND COHORT OVERVIEW

# 18. FIGURE 2 - MODEL COMPARISON AND HEADLINE PERFORMANCE

# 19. FIGURE 3 - CLASS BEHAVIOUR AND CALIBRATION
def _calibration_tables(pub_outputs: Dict, output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ece = _read_result_dataframe(pub_outputs, "ece", output_dir,
                                 ["calibration_metrics.csv", "calibration_ece.csv"])
    bins = _read_result_dataframe(pub_outputs, "calibration_bins", output_dir,
                                  ["calibration_bins.csv"])
    return ece, bins

# 20. FIGURE 4 - ROBUSTNESS, SENSITIVITY, AUDITING, AND GENERALISATION
def _leakage_audit_frame(output_dir: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for scenario in SCENARIO_ORDER:
        path = os.path.join(output_dir, "results", f"leakage_audit_{scenario}.csv")
        if not os.path.exists(path):
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if "drop_reason" not in frame.columns:
            continue
        for reason, count in frame["drop_reason"].astype(str).value_counts().items():
            rows.append({"Scenario": scenario, "drop_reason": reason, "n": int(count)})
    return pd.DataFrame(rows)

# 21. FIGURE 5 - INTERPRETATION AND PHENOTYPE BIOLOGY
def _importance_for_scenario(results_collector, scenario: str, model_name: Optional[str]) -> pd.DataFrame:
    fi = getattr(results_collector, "feature_importances", {}) or {}
    candidates = []
    if model_name:
        candidates.extend([f"{scenario}_{model_name}"])
    candidates.extend(_model_result_keys(scenario, model_name))
    for key in candidates:
        frame = fi.get(key)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            return frame.copy()
    return pd.DataFrame()

# 22. MAIN-FIGURE MANIFEST AND PUBLIC ORCHESTRATOR
def generate_publication_figures(all_results: Dict, container, pub_outputs: Dict,
                                 results_collector,
                                 fold_artifacts: Optional[Dict] = None,
                                 output_dir: Optional[str] = None,
                                 figure_output_dir: Optional[str] = None,
                                 random_seed: int = 42):
    """Generate the composite publication figure pack.
    ``output_dir`` is the completed pipeline run used as the source of results,
    explanations, phenotype weights, and other analysis artefacts.
    ``figure_output_dir`` is an independent destination for every newly
    generated PNG, PDF, table, manifest, and panel-level CSV. When omitted,
    the historical ``<output_dir>/figures/publication`` destination is used.
    """
    if output_dir is None:
        from pipeline import Config
        output_dir = Config.OUTPUT_DIR
    if fold_artifacts is None:
        try:
            from pipeline import _fold_artifacts as _fa
            fold_artifacts = _fa
        except Exception:
            fold_artifacts = {}
    # Keep the completed pipeline run as the input/source root, while allowing
    # all newly rendered publication artefacts to be written to another folder.
    source_output_dir = os.path.abspath(str(output_dir))
    if figure_output_dir is None:
        publication_root = os.path.join(
            source_output_dir,
            "figures",
            "publication",
        )
    else:
        publication_root = os.path.abspath(str(figure_output_dir))
    main_dir = os.path.join(publication_root, "main")
    fig_data_dir = os.path.join(publication_root, "figure_data")
    os.makedirs(main_dir, exist_ok=True)
    os.makedirs(fig_data_dir, exist_ok=True)
    # Figure functions that load existing result files must continue reading
    # from the completed pipeline run, not from the new destination.
    output_dir = source_output_dir
    _setup_rcparams()
    labeler = FeatureLabeler(output_dir=output_dir)
    resolved_scenario, resolved_model = _resolve_headline(all_results, pub_outputs)
    print(f"[publication_figures_v2] source pipeline run: {source_output_dir}")
    print(f"[publication_figures_v2] publication output: {publication_root}")
    print(f"[publication_figures_v2] writing five composite main figures to {main_dir}")
    print(f"  Headline pair: {resolved_scenario} / {resolved_model}")
    print(f"  Loaded {len(labeler.hpo_dict)} HPO labels, {len(labeler.extra_dict)} extras")
    print("[main] Figure 1 - Study design and cohort overview")
    figure_1_study_design(all_results, container, fold_artifacts, main_dir,
                          fig_data_dir, labeler)
    print("[main] Figure 2 - Candidate models and headline performance")
    figure_2_model_comparison(all_results, pub_outputs, output_dir, main_dir,
                              fig_data_dir)
    print("[main] Figure 3 - Prediction behaviour and calibration")
    figure_3_prediction_calibration(all_results, pub_outputs, output_dir, main_dir,
                                    fig_data_dir)
    print("[main] Figure 4 - Robustness, sensitivity, and auditing")
    figure_4_robustness_composite(all_results, pub_outputs, container, fold_artifacts,
                                  output_dir, main_dir, fig_data_dir)
    print("[main] Figure 5 - Interpretation and phenotype biology")
    figure_5_interpretation_biology(all_results, pub_outputs, container,
                                    results_collector, fold_artifacts, output_dir,
                                    main_dir, fig_data_dir, labeler,
                                    random_seed=random_seed)
    print("[main] Figure 5f - Family-summed SHAP (dilution check)")
    figure_5_family_summed_shap(output_dir, all_results, pub_outputs, container,
                                fig_data_dir, labeler)
    print("[main] Table 1 - Headline locked-test metrics")
    table_1_headline_metrics(all_results, pub_outputs, main_dir, fig_data_dir)
    print("[main] Table 2 - Gene-level companion metrics")
    table_2_gene_level_companion(pub_outputs, main_dir, fig_data_dir,
                                 output_dir=output_dir)
    _write_main_figure_manifest(main_dir, fig_data_dir)
    print("[publication_figures_v2] done; no supplementary figures were generated.")
# Backward-compatible alias retained for external scripts that imported it.

# 23. FIGURE RENDERING
# These definitions intentionally override the earlier rendering functions.
# The public orchestrator resolves global function names at call time, so no
# changes to generate_publication_figures(...) are required.
import math
import textwrap
from matplotlib.colors import TwoSlopeNorm
NATURE_SHOW_MISSING_PANELS = False
NATURE_MIN_FONT_PT = 6.2
# Internal figure numbering is offset by one from the manuscript, because
# manuscript Figure 1 is a hand-built schematic that this script does not
# generate. Saving under the internal names forced a manual renaming step on
# every rebuild, and that step is what allowed superseded figures from an older
# output folder to reach the submission set. Files are therefore written under
# the manuscript numbers directly, so the generated folder can be used as-is.
# Updated for the Nature Medicine Article format: the six-display-item limit
# moved the cohort-overview and robustness figures to Extended Data.
MANUSCRIPT_FIGURE_NAMES = {
    "figure_1_study_design": "extended_data_fig_1_cohort_composition",
    "figure_2_model_comparison": "figure_2_model_comparison",
    "figure_3_prediction_and_calibration": "figure_3_prediction_and_calibration",
    "figure_4_robustness_and_audits": "extended_data_fig_2_robustness_and_audits",
    "figure_5_interpretation_and_biology": "figure_4_interpretation_and_biology",
}
_SAVED_FIGURE_NAME_MAP: List[Dict[str, str]] = []

def _save(fig, name: str, out_dir: str):
    """Save review and submission-ready raster/vector versions.
    The file is written under its manuscript figure number (see
    MANUSCRIPT_FIGURE_NAMES). A `figure_name_map.csv` recording the internal
    name, the manuscript name and the panel-data prefix is written alongside,
    so the provenance of every delivered figure stays traceable.
    """
    os.makedirs(out_dir, exist_ok=True)
    manuscript_name = MANUSCRIPT_FIGURE_NAMES.get(name, name)
    _SAVED_FIGURE_NAME_MAP.append({
        "internal_name": name,
        "manuscript_name": manuscript_name,
        "panel_data_prefix": name.split("_")[0] + "_" + name.split("_")[1],
    })
    try:
        pd.DataFrame(_SAVED_FIGURE_NAME_MAP).to_csv(
            os.path.join(out_dir, "figure_name_map.csv"), index=False
        )
    except Exception:
        pass
    name = manuscript_name
    base = os.path.join(out_dir, name)
    fig.savefig(base + ".png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(base + ".pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(base + ".svg", bbox_inches="tight", facecolor="white")
    try:
        fig.savefig(
            base + ".tiff", dpi=600, bbox_inches="tight", facecolor="white",
            pil_kwargs={"compression": "tiff_lzw"},
        )
    except Exception:
        # TIFF support varies with the Pillow build; PDF/SVG/PNG are always kept.
        pass
    plt.close(fig)

def _empty_panel(ax, msg: str):
    """Hide unavailable manuscript panels instead of printing debug placeholders."""
    ax.set_axis_off()
    if NATURE_SHOW_MISSING_PANELS:
        ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
                fontsize=NATURE_MIN_FONT_PT, color="#777777", style="italic")

def _wrap_label(value: Any, width: int = 24) -> str:
    text = str(value).replace("MONDO: ", "").strip()
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))

def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")

def _clean_mondo_columns(columns) -> List[str]:
    excluded = {
        "feat_mondo_depth", "feat_mondo_n_classes", "feat_has_mondo_record",
        "feat_mondo_record", "feat_mondo_count", "feat_mondo_available",
    }
    return [c for c in columns if str(c).startswith("feat_mondo_") and c not in excluded]
# Figure 1
# Figure 2
# Figure 3
# Figure 4 - dynamic manuscript composite (no blank debug tiles)
# Figure 5
def _plot_interpretation_nature(ax, output_dir: str, scenario: str,
                                model_name: Optional[str], results_collector,
                                labeler: FeatureLabeler, rng_seed: int,
                                top_n: int = 6):
    obs, source_label, n_rows = _read_scenario_shap_observations(output_dir, scenario)
    rows = []
    scatter = None
    if obs:
        denom = float(n_rows) if n_rows else 1.0
        ranked = sorted(obs.items(), key=lambda kv: float(np.abs(kv[1][0]).sum()) / denom,
                        reverse=True)[:top_n]
        rng = np.random.RandomState(rng_seed)
        for rank, (feature, (shap_values, feature_values)) in enumerate(ranked):
            sv = np.asarray(shap_values, dtype=float)
            fv = np.asarray(feature_values, dtype=float)
            finite = np.isfinite(sv); sv = sv[finite]
            fv = fv[finite] if len(fv) == len(finite) else np.full(len(sv), np.nan)
            if not len(sv):
                continue
            good = fv[np.isfinite(fv)]
            if good.size:
                lo, hi = np.percentile(good, [5, 95])
                if hi <= lo: hi = lo + 1e-9
                cvals = np.clip((fv - lo) / (hi - lo), 0, 1)
                cvals = np.where(np.isfinite(cvals), cvals, 0.5)
            else:
                cvals = np.full(len(sv), 0.5)
            # Retain every explanation point, but use a lighter, tighter cloud so
            # additional ranked features remain legible at journal width.
            yj = np.full(len(sv), rank) + rng.normal(0, 0.060, len(sv))
            scatter = ax.scatter(
                sv, yj, c=cvals, cmap="coolwarm", vmin=0, vmax=1,
                s=3.1, alpha=0.29, linewidths=0, rasterized=True,
                zorder=2,
            )
            ax.plot(
                [np.median(sv)] * 2,
                [rank - 0.12, rank + 0.12],
                color="black", lw=0.70, zorder=3,
            )
            rows.append({"Scenario": scenario, "Model": model_name, "raw_feature": feature,
                         "label": labeler(feature), "mean_abs_shap": float(np.abs(sv).sum() / denom),
                         "source": source_label, "n_explained": n_rows})
        ax.set_yticks(np.arange(len(ranked)))
        ax.margins(y=0.05)
        ax.set_yticklabels([_wrap_label(labeler(f), 24) for f, _ in ranked], fontsize=5.20)
        ax.invert_yaxis(); ax.axvline(0, color="#777777", lw=0.6, ls=(0, (2, 2)))
        ax.set_xlabel(
            "SHAP value (severity classes)"
            if source_label == SEVERITY_SHAP_SOURCE
            else "Signed SHAP value"
        )
        _finish_axis(ax, "x")
        return pd.DataFrame(rows), scatter
    importance = _importance_for_scenario(results_collector, scenario, model_name)
    if not importance.empty and "feature" in importance.columns:
        value_col = "mean_abs_shap" if "mean_abs_shap" in importance.columns else (
            "importance" if "importance" in importance.columns else None)
        if value_col:
            top = importance.nlargest(top_n, value_col).copy()
            yy = np.arange(len(top)); values = _numeric(top, value_col).fillna(0).to_numpy()
            ax.hlines(yy, 0, values, color="#D7D7D7", lw=1.2)
            ax.scatter(values, yy, s=20, color=PAL_SCENARIO.get(scenario, "#4C78A8"),
                       edgecolor="black", linewidth=0.35)
            ax.set_yticks(yy); ax.set_yticklabels([_wrap_label(labeler(f), 24) for f in top["feature"]], fontsize=5.20)
            ax.invert_yaxis(); ax.set_xlabel("Feature importance"); _finish_axis(ax, "x")
            top["Scenario"] = scenario; top["Model"] = model_name; top["source"] = "native_importance"
            top["label"] = [labeler(f) for f in top["feature"]]
            return top.rename(columns={"feature": "raw_feature"}), None
    _empty_panel(ax, "Interpretation unavailable")
    return pd.DataFrame(), None

# 24. FINAL LAYOUT OVERRIDES FOR LONG-LABEL MANUSCRIPT PANELS
# These definitions provide the final layout. They retain the public API
# and the strict
# validation/test separation of the pipeline.
from matplotlib.ticker import PercentFormatter, LogLocator, NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
AESTHETIC_VERSION = "3.0"

def _setup_rcparams():
    """High-impact-journal visual system with a strict readability floor."""
    mpl.rcParams.update({
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
    })

def _finish_axis(ax, grid_axis: Optional[str] = None):
    """Editorial axis finishing: quiet grid, dark baseline, no box."""
    ax.set_axisbelow(True)
    if grid_axis in {"x", "y", "both"}:
        ax.grid(
            axis=grid_axis,
            color="#E6E8EB",
            linewidth=0.48,
            alpha=0.95,
            zorder=0,
            linestyle=(0, (2, 2)),
        )
    ax.tick_params(pad=2.4)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#3A3A3A")
        ax.spines[side].set_linewidth(0.6)

def _panel_label(ax, label: str, x: float = -0.14, y: float = 1.055,
                 fs: float = 9.0):
    kwargs = dict(fontsize=fs, fontweight="bold", va="bottom", ha="left",
                  color="#111111", clip_on=False)
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
# Figure 1 — study design and cohort overview
def _plot_mondo_class_panel_d(ax, full_df, labeler, fig_data_dir):
    """Former panel d: most represented specific MONDO disease classes.

    Retained so PANEL_D_MODE = "mondo_classes" reproduces the previous figure
    exactly. Not used in the default configuration.
    """
    mondo_columns = _clean_mondo_columns(full_df.columns)
    if mondo_columns:
        counts = (
            full_df[mondo_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .sum()
        )
        total_pairs = max(len(full_df), 1)
        raw_labels = {
            column: labeler(column)
            for column in counts.index
        }
        specific = counts[
            [
                column
                for column in counts.index
                if _is_specific_mondo_label(raw_labels[column])
            ]
        ]
        if len(specific) >= 5:
            counts = specific
        counts = counts.sort_values(ascending=False).head(7).sort_values()
        labels = [
            _wrap_label(raw_labels[column], 28)
            for column in counts.index
        ]
        prevalence = counts / float(total_pairs) * 100.0
        y_positions = np.arange(len(counts))
        ax.barh(
            y_positions, prevalence.values,
            color="#4F9D89", edgecolor="none", height=0.70,
        )
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=5.8)
        ax.set_xlabel("Pairs annotated to class (%)")
        x_max = (
            float(prevalence.max()) * 1.32
            if len(prevalence) else 1.0
        )
        ax.set_xlim(0, x_max)
        for row_index, (percent, count) in enumerate(
            zip(prevalence.values, counts.values)
        ):
            ax.text(
                percent + x_max * 0.02,
                row_index,
                f"{percent:.1f}%  (n={int(count):,})",
                va="center",
                fontsize=5.6,
            )
        _finish_axis(ax, "x")
        filtered_flag = len(specific) >= 5
        _save_csv(
            pd.DataFrame({
                "raw_feature": counts.index,
                "label": [
                    raw_labels[column]
                    for column in counts.index
                ],
                "n_pairs": counts.values,
                "percent_of_pairs": prevalence.values,
                "generic_parent_filtered": (
                    [filtered_flag] * len(counts)
                ),
            }),
            "figure_1d_specific_mondo.csv",
            fig_data_dir,
        )
    else:
        _empty_panel(ax, "Disease classes unavailable")
    _soft_title(ax, "Most represented specific disease classes")
    _panel_label(ax, "d", x=-0.13)


def figure_1_study_design(
    all_results: Dict,
    container,
    fold_artifacts: Dict,
    out_dir: str,
    fig_data_dir: str,
    labeler: FeatureLabeler,
    logger=print,
):
    """Five-panel cohort overview after removing the workflow infographic.
    The former panels b--f are reflowed and relabelled a--e.  The two former
    right-hand panels are widened across the lower row so long titles and
    annotations do not collide.
    """
    y = np.asarray(container.y)
    full_df = container.full_df
    train_idx, val_idx, test_idx = _resolve_partition_indices(container, fold_artifacts)
    fig = plt.figure(figsize=(7.25, 6.05))
    outer = fig.add_gridspec(
        2, 1,
        height_ratios=[0.96, 1.08],
        left=0.105, right=0.965, top=0.955, bottom=0.095,
        hspace=0.62,
    )
    top = outer[0].subgridspec(1, 3, wspace=0.52)
    bottom = outer[1].subgridspec(
        1, 2, width_ratios=[1.12, 1.0], wspace=0.68
    )
    # a — within-partition composition (former panel b)
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
            ypos, values, left=left, height=0.52,
            color=PAL_CLASS[class_name], edgecolor="white", linewidth=0.55,
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
            1.015, row_index, f"n={len(idx):,}",
            va="center", ha="left", fontsize=5.8,
            transform=ax.get_yaxis_transform(),
        )
        counts = np.bincount(y[idx].astype(int), minlength=4)
        rows_a.extend(
            {
                "partition": partition_name,
                "severity": CLASS_NAMES[class_index],
                "n": int(counts[class_index]),
                "proportion": float(
                    counts[class_index] / max(counts.sum(), 1)
                ),
            }
            for class_index in range(4)
        )
    ax.set_yticks(ypos)
    ax.set_yticklabels([name for name, _ in active])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Within-partition proportion")
    ax.legend(
        frameon=False, ncol=2, loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        handlelength=0.9, columnspacing=0.9,
    )
    _finish_axis(ax, "x")
    _soft_title(ax, "Severity composition")
    _panel_label(ax, "a", x=-0.18)
    _save_csv(
        pd.DataFrame(rows_a),
        "figure_1a_partition_composition.csv",
        fig_data_dir,
    )
    # b — clustering CCDF (former panel c)
    ax = fig.add_subplot(top[0, 1])
    pairs_per_gene = pd.Series(container.groups).value_counts().to_numpy(dtype=int)
    x_values = np.arange(1, int(np.max(pairs_per_gene)) + 1)
    survival = np.asarray(
        [(pairs_per_gene >= x).sum() for x in x_values],
        dtype=float,
    )
    ax.step(
        x_values, survival, where="post",
        color=PAL_SCENARIO["S1_Mechanistic"], lw=1.25,
    )
    ax.scatter(
        x_values, survival, s=16,
        color=PAL_SCENARIO["S1_Mechanistic"],
        edgecolor="white", linewidth=0.35, zorder=3,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Gene–disease pairs per gene")
    ax.set_ylabel("Genes with ≥ x pairs")
    median = int(np.median(pairs_per_gene))
    p95 = int(np.percentile(pairs_per_gene, 95))
    ax.text(
        0.98, 0.96,
        f"Median = {median}\n95th percentile = {p95}",
        transform=ax.transAxes, ha="right", va="top", fontsize=6.1,
    )
    ax.xaxis.set_minor_locator(
        LogLocator(base=10, subs=np.arange(2, 10) * 0.1)
    )
    ax.xaxis.set_minor_formatter(NullFormatter())
    _finish_axis(ax, "both")
    _soft_title(ax, "Gene-level clustering")
    _panel_label(ax, "b", x=-0.18)
    _save_csv(
        pd.DataFrame({
            "minimum_pairs": x_values,
            "n_genes": survival.astype(int),
        }),
        "figure_1b_gene_pair_ccdf.csv",
        fig_data_dir,
    )
    # c — inheritance prevalence heatmap (former panel d)
    ax = fig.add_subplot(top[0, 2])
    inheritance_columns = [
        "feat_is_dominant",
        "feat_is_recessive",
        "feat_is_xlinked",
    ]
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
            values = (
                pd.to_numeric(full_df[column], errors="coerce")
                .fillna(0)
                .to_numpy()
                > 0
            )
            row_values = []
            for class_index, class_name in enumerate(CLASS_NAMES):
                mask = y == class_index
                prevalence = float(values[mask].mean()) if mask.any() else np.nan
                row_values.append(prevalence)
                rows_c.append({
                    "inheritance": inheritance_label,
                    "severity": class_name,
                    "prevalence": prevalence,
                    "numerator": int((values & mask).sum()),
                    "denominator": int(mask.sum()),
                })
            matrix.append(row_values)
        matrix = np.asarray(matrix, dtype=float)
        image = ax.imshow(
            matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto"
        )
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                ax.text(
                    column_index, row_index, f"{value:.0%}",
                    ha="center", va="center", fontsize=6.2,
                    color=_adaptive_text_color(value, 0, 1),
                )
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(CLASS_NAMES, rotation=28, ha="right")
        ax.set_yticks(np.arange(len(available)))
        ax.set_yticklabels([item[1] for item in available])
        colorbar = fig.colorbar(
            image, ax=ax, orientation="horizontal",
            fraction=0.08, pad=0.22, aspect=30,
        )
        colorbar.set_label("Prevalence")
        colorbar.ax.xaxis.set_major_formatter(
            PercentFormatter(1.0, 0)
        )
        colorbar.outline.set_linewidth(0.4)
        for spine in ax.spines.values():
            spine.set_visible(False)
        _save_csv(
            pd.DataFrame(rows_c),
            "figure_1c_inheritance_prevalence.csv",
            fig_data_dir,
        )
    else:
        _empty_panel(ax, "Inheritance indicators unavailable")
    _soft_title(ax, "Inheritance prevalence by severity")
    _panel_label(ax, "c", x=-0.18)
    # d — disease-level clustering CCDF.
    #
    # Replaces the former "most represented specific MONDO disease classes" bar
    # chart. That panel was uninformative: the largest class covered 4.3% of
    # pairs and the seven shown covered ~10.7% between them, because MONDO
    # class annotation is sparse in this corpus.
    #
    # This panel instead mirrors panel b at the disease level. Because a
    # disease's directly annotated HPO terms are shared by every gene assigned
    # to it, pairs from the same MONDO disease carry identical phenotype
    # vectors. The pairs-per-disease distribution is therefore the quantity
    # that motivates the disease-disjoint and joint gene-and-disease-disjoint
    # evaluations, and it belongs in the main figure rather than in the text.
    #
    # Set PANEL_D_MODE = "mondo_classes" to restore the previous behaviour.
    ax = fig.add_subplot(bottom[0, 0])
    _disease_column = next(
        (
            column
            for column in ("mondo_id", "MONDO_ID", "disease_id")
            if column in full_df.columns
        ),
        None,
    )
    if PANEL_D_MODE == "disease_clustering" and _disease_column is not None:
        pairs_per_disease = (
            full_df[_disease_column].astype(str).value_counts().to_numpy(dtype=int)
        )
        dx_values = np.arange(1, int(np.max(pairs_per_disease)) + 1)
        dsurvival = np.asarray(
            [(pairs_per_disease >= x).sum() for x in dx_values],
            dtype=float,
        )
        ax.step(
            dx_values, dsurvival, where="post",
            color=PAL_SCENARIO["S1b_Disease_Context"], lw=1.25,
        )
        # Mark only the observed sizes so the tail points are real diseases,
        # not interpolated grid positions.
        observed = np.unique(pairs_per_disease)
        ax.scatter(
            observed,
            [(pairs_per_disease >= x).sum() for x in observed],
            s=16,
            color=PAL_SCENARIO["S1b_Disease_Context"],
            edgecolor="white", linewidth=0.35, zorder=3,
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
            0.98, 0.96,
            (
                f"{n_diseases:,} diseases\n"
                f"{n_singleton / max(n_diseases, 1):.0%} carry one pair\n"
                f"largest = {int(largest[0]):,} pairs\n"
                f"top 3 = {top3_share:.0%} of pairs"
            ),
            transform=ax.transAxes, ha="right", va="top", fontsize=6.1,
        )
        ax.xaxis.set_minor_locator(
            LogLocator(base=10, subs=np.arange(2, 10) * 0.1)
        )
        ax.xaxis.set_minor_formatter(NullFormatter())
        _finish_axis(ax, "both")
        _save_csv(
            pd.DataFrame({
                "minimum_pairs": dx_values,
                "n_diseases": dsurvival.astype(int),
            }),
            "figure_1d_disease_pair_ccdf.csv",
            fig_data_dir,
        )
        # Per-disease detail for the largest groups, so the tail of the CCDF is
        # auditable against the disease-overlap audit.
        _disease_sizes = (
            full_df[_disease_column].astype(str).value_counts()
        )
        _name_column = next(
            (
                column
                for column in ("disease_name", "Disease", "disease")
                if column in full_df.columns
            ),
            None,
        )
        _detail = pd.DataFrame({
            "mondo_id": _disease_sizes.index,
            "n_pairs": _disease_sizes.to_numpy(dtype=int),
        })
        if _name_column is not None:
            _lookup = (
                full_df[[_disease_column, _name_column]]
                .astype(str)
                .drop_duplicates(subset=[_disease_column])
                .set_index(_disease_column)[_name_column]
            )
            _detail["disease_name"] = (
                _detail["mondo_id"].map(_lookup).fillna("")
            )
        _detail["percent_of_pairs"] = (
            _detail["n_pairs"] / max(n_pairs_total, 1) * 100.0
        )
        # A saved run supplies its original ordering for tied disease counts.
        saved_detail = getattr(container, "largest_diseases", None)
        _save_csv(
            _detail.head(30) if saved_detail is None else saved_detail,
            "figure_1d_largest_diseases.csv", fig_data_dir,
        )
        _soft_title(ax, "Disease-level clustering")
        _panel_label(ax, "d", x=-0.13)
    elif PANEL_D_MODE == "disease_clustering":
        _empty_panel(ax, "Disease identifiers unavailable")
        _soft_title(ax, "Disease-level clustering")
        _panel_label(ax, "d", x=-0.13)
    else:
        _plot_mondo_class_panel_d(
            ax, full_df, labeler, fig_data_dir
        )
    # e — representation matrix (former panel f), widened and relabelled
    ax = fig.add_subplot(bottom[0, 1])
    family_frame = _scenario_feature_family_frame(
        container, selected_only=False
    )
    if not family_frame.empty:
        pivot = family_frame.pivot_table(
            index="family",
            columns="Scenario",
            values="n_features",
            aggfunc="sum",
            fill_value=0,
        )
        scenarios = [
            scenario
            for scenario in SCENARIO_ORDER
            if scenario in pivot.columns
        ]
        pivot = pivot.reindex(columns=scenarios)
        order = list(
            pivot.sum(axis=1)
            .sort_values(ascending=False)
            .head(9)
            .index
        )
        remainder = pivot.drop(
            index=order, errors="ignore"
        ).sum(axis=0)
        pivot = pivot.reindex(order)
        if np.any(remainder > 0):
            pivot.loc["other"] = remainder
        raw_values = pivot.to_numpy(dtype=float)
        log_values = np.log10(raw_values + 1)
        v_max = max(4.0, float(np.nanmax(log_values)))
        image = ax.imshow(
            log_values, cmap="Blues", aspect="auto",
            vmin=0, vmax=v_max,
        )
        for row_index in range(raw_values.shape[0]):
            for column_index in range(raw_values.shape[1]):
                raw_value = raw_values[row_index, column_index]
                log_value = log_values[row_index, column_index]
                ax.text(
                    column_index, row_index, f"{int(raw_value):,}",
                    ha="center", va="center", fontsize=5.8,
                    color=_adaptive_text_color(
                        log_value, 0, v_max
                    ),
                )
        ax.set_xticks(np.arange(len(scenarios)))
        ax.set_xticklabels(_scenario_short_list(scenarios))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(
            [
                _family_display(family)
                if family != "other" else "Other"
                for family in pivot.index
            ],
            fontsize=6.0,
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
        _small_colorbar(
            ax, image, "log10(candidate features + 1)",
            size="3.8%", pad=0.05,
        )
        _save_csv(
            family_frame,
            "figure_1e_scenario_feature_families.csv",
            fig_data_dir,
        )
    else:
        _empty_panel(ax, "Scenario features unavailable")
    _soft_title(ax, "Candidate representation by feature family")
    _panel_label(ax, "e", x=-0.13)
    _save(fig, "figure_1_study_design", out_dir)
    logger("[figure_1] saved as five-panel layout (a–e)")
# Figure 2 — model comparison and headline performance
def _plot_candidate_model_lines(ax, frame: pd.DataFrame, metric: str, title: str):
    scenarios = _present_scenarios(frame["Scenario"].unique())
    models = _stable_models(frame["Model"].unique())
    x = np.arange(len(scenarios))
    for model in models:
        sub = frame[frame["Model"] == model].set_index("Scenario").reindex(scenarios)
        values = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
        ax.plot(x, values, marker=MODEL_MARKERS.get(model, "o"), markersize=4.1,
                color=PAL_MODEL.get(model, "#777777"), alpha=0.86, lw=0.9,
                label=MODEL_LABELS.get(model, model), zorder=2)
        selected_mask = sub.get("selected_on_validation", pd.Series(False, index=sub.index))
        selected_mask = selected_mask.fillna(False).astype(bool).to_numpy()
        finite = selected_mask & np.isfinite(values)
        if finite.any():
            ax.scatter(x[finite], values[finite], s=52, marker=MODEL_MARKERS.get(model, "o"),
                       facecolor=PAL_MODEL.get(model, "#777777"), edgecolor="black",
                       linewidth=0.95, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(_scenario_short_list(scenarios))
    ax.set_ylim((-0.02, 1.02) if metric == "mcc" else (0, 1.02))
    if metric == "balanced_accuracy":
        ax.axhline(0.25, color="#888888", lw=0.65, ls=(0, (2, 2)), zorder=0)
    ax.set_ylabel(title); _finish_axis(ax, "both"); _soft_title(ax, title, pad=7)

def _selection_margin_frame(validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in _present_scenarios(validation["Scenario"].unique()):
        sub = validation[validation["Scenario"] == scenario].copy()
        sub["qwk"] = pd.to_numeric(sub["qwk"], errors="coerce")
        sub = sub.dropna(subset=["qwk"]).sort_values("qwk", ascending=False)
        if len(sub) < 2: continue
        selected_flag = sub["selected_on_validation"] if "selected_on_validation" in sub.columns else pd.Series(False, index=sub.index)
        selected_rows = sub[selected_flag.fillna(False).astype(bool)]
        winner = selected_rows.iloc[0] if len(selected_rows) else sub.iloc[0]
        runner = sub[sub["Model"] != winner["Model"]].iloc[0]
        rows.append({"Scenario": scenario, "winner": winner["Model"],
                     "runner_up": runner["Model"],
                     "margin": float(winner["qwk"] - runner["qwk"])})
    return pd.DataFrame(rows)

def figure_2_model_comparison(
    all_results: Dict,
    pub_outputs: Dict,
    output_dir: str,
    out_dir: str,
    fig_data_dir: str,
    logger=print,
):
    validation = _candidate_validation_frame(all_results, pub_outputs)
    test = _selected_test_frame(all_results, pub_outputs, output_dir)
    selected = _validation_selected_models(all_results, pub_outputs)
    if validation.empty:
        logger("[figure_2] no validation candidate metrics; skipping")
        return
    fig = plt.figure(figsize=(7.25, 6.55))
    gs = fig.add_gridspec(
        3, 3, height_ratios=[0.92, 1.05, 1.05], width_ratios=[1.0, 1.0, 1.0],
        left=0.075, right=0.975, top=0.865, bottom=0.105,
        wspace=0.46, hspace=0.62,
    )
    for i, (metric, title) in enumerate([("qwk", "QWK"), ("mcc", "MCC"),
                                         ("balanced_accuracy", "Balanced accuracy")]):
        ax = fig.add_subplot(gs[0, i])
        _plot_candidate_model_lines(ax, validation, metric, title)
        if i > 0: ax.set_ylabel("")
        _panel_label(ax, chr(ord("a") + i), x=-0.16)
    # Shared top legend
    models = _stable_models(validation["Model"].unique())
    handles = [mpl.lines.Line2D([], [], color=PAL_MODEL[m], marker=MODEL_MARKERS.get(m, "o"),
                                lw=1, markersize=4.5, label=MODEL_LABELS.get(m, m)) for m in models]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.985),
               frameon=False, ncol=len(handles), title="Candidate models; black outline = selected",
               handlelength=1.4, columnspacing=1.2)
    # d — selected models only on locked test; hero forest plot
    ax = fig.add_subplot(gs[1:3, 0:2])
    scenarios = _present_scenarios(test.get("Scenario", pd.Series(dtype=str)).unique())
    metrics = [("mcc", "MCC", "o"), ("qwk", "QWK", "s"),
               ("balanced_accuracy", "Balanced accuracy", "^")]
    nrows = len(scenarios) * len(metrics)
    positions = np.arange(nrows)
    export_rows = []
    for si, scenario in enumerate(scenarios):
        start = si * len(metrics)
        if si % 2 == 0:
            ax.axhspan(start - 0.45, start + len(metrics) - 0.55,
                       color="#F5F6F7", zorder=0)
        row = test[test["Scenario"] == scenario]
        if row.empty: continue
        row = row.iloc[0]
        for mi, (metric, label, marker) in enumerate(metrics):
            ypos = start + mi
            mean = float(row.get(f"{metric}_mean", row.get(metric, np.nan)))
            lo = float(row.get(f"{metric}_ci_low", np.nan)); hi = float(row.get(f"{metric}_ci_high", np.nan))
            if not np.isfinite(mean): continue
            xerr = None
            if np.isfinite(lo) and np.isfinite(hi):
                xerr = np.array([[max(mean - lo, 0)], [max(hi - mean, 0)]])
            color = PAL_SCENARIO.get(scenario, "#777777")
            ax.errorbar(mean, ypos, xerr=xerr, fmt=marker, markersize=4.9,
                        markerfacecolor=color, markeredgecolor="black", markeredgewidth=0.5,
                        ecolor=color, elinewidth=1.05, capsize=2.2, zorder=3)
            export_rows.append({"Scenario": scenario, "Model": row.get("Model"),
                                "metric": metric, "estimate": mean,
                                "ci_low": lo, "ci_high": hi})
    labels = []
    for scenario in scenarios:
        labels.extend([f"{SCENARIO_SHORT[scenario]}  {name}" for _, name, _ in metrics])
    ax.set_yticks(positions); ax.set_yticklabels(labels, fontsize=6.1)
    ax.invert_yaxis(); ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Locked-test score (gene-cluster bootstrap 95% CI)")
    _finish_axis(ax, "x")
    common_model = next(iter(set(selected.values()))) if selected and len(set(selected.values())) == 1 else None
    _soft_title(ax, "Locked-test performance", pad=9)
    if common_model:
        ax.text(0.995, 1.035, f"Selected family: {MODEL_SHORT_LABELS.get(common_model, common_model)}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6, color="#555555")
    _panel_label(ax, "d", x=-0.20)
    _save_csv(pd.DataFrame(export_rows), "figure_2d_locked_test_forest.csv", fig_data_dir)
    # e — selection margin
    ax = fig.add_subplot(gs[1, 2])
    margins = _selection_margin_frame(validation)
    if not margins.empty:
        margins = margins.set_index("Scenario").reindex(_present_scenarios(margins["Scenario"].unique())).reset_index()
        yy = np.arange(len(margins))
        ax.axvline(0, color="#333333", lw=0.7)
        xmax = max(0.03, float(margins["margin"].max()) * 1.33)
        for yi, row in margins.iterrows():
            color = PAL_SCENARIO.get(row["Scenario"], "#777777")
            ax.hlines(yi, 0, row["margin"], color=color, lw=1.35)
            ax.scatter(row["margin"], yi, s=34, color=color, edgecolor="black", linewidth=0.4)
            ax.text(row["margin"] + xmax * 0.025, yi, f"{row['margin']:.3f}", va="center", fontsize=5.8)
            ax.text(xmax * 1.02, yi, f"vs {MODEL_LABELS.get(row['runner_up'], row['runner_up'])}", va="center", fontsize=5.55,
                    clip_on=False)
        ax.set_yticks(yy); ax.set_yticklabels(_scenario_short_list(margins["Scenario"])); ax.invert_yaxis()
        ax.set_xlim(0, xmax); ax.set_xlabel("Δ validation QWK versus runner-up")
        _finish_axis(ax, "x")
        _save_csv(margins, "figure_2e_model_selection_margin.csv", fig_data_dir)
    else:
        _empty_panel(ax, "Selection margins unavailable")
    _soft_title(ax, "Model-selection margin"); _panel_label(ax, "e", x=-0.18)
    # f — information-source contrasts
    ax = fig.add_subplot(gs[2, 2])
    contrasts = [
        ("Disease context", "S1_Mechanistic", "S1b_Disease_Context"),
        ("Phenotype added", "S1b_Disease_Context", "S2_Mech_Pheno"),
        ("Gene-level added", "S3_Phenotype", "S2_Mech_Pheno"),
    ]
    rows_f = []
    ylabels = []; yy = []
    for ci, (label, base, enhanced) in enumerate(contrasts):
        if base not in set(test.get("Scenario", [])) or enhanced not in set(test.get("Scenario", [])):
            continue
        rb = test[test["Scenario"] == base].iloc[0]
        re = test[test["Scenario"] == enhanced].iloc[0]
        dqwk = float(re.get("qwk", np.nan) - rb.get("qwk", np.nan))
        dmcc = float(re.get("mcc", np.nan) - rb.get("mcc", np.nan))
        ylabels.append(label); yy.append(ci)
        ax.scatter(dqwk, ci + 0.11, s=28, marker="o", facecolor="white",
                   edgecolor="#222222", linewidth=0.8, zorder=3)
        ax.scatter(dmcc, ci - 0.11, s=28, marker="s", facecolor="white",
                   edgecolor="#222222", linewidth=0.8, zorder=3)
        # Sign-aware label placement. A fixed positive offset puts the label of a
        # negative contrast on top of its own marker, which hides the leading
        # minus sign; the S3-to-S2 contrast is the paper's headline null result,
        # so its sign has to be unambiguous. Negative values are therefore set to
        # the left of the marker and right-aligned.
        _label_gap = 0.012
        for _value, _yoff in ((dqwk, +0.11), (dmcc, -0.11)):
            if not np.isfinite(_value):
                continue
            _negative = _value < 0
            ax.text(
                _value - _label_gap if _negative else _value + _label_gap,
                ci + _yoff,
                f"{_value:+.3f}",
                va="center",
                ha="right" if _negative else "left",
                fontsize=5.5,
            )
        rows_f.extend([{"contrast": label, "metric": "QWK", "difference": dqwk},
                       {"contrast": label, "metric": "MCC", "difference": dmcc}])
    if yy:
        ax.axvline(0, color="#777777", lw=0.7, ls=(0, (2, 2)))
        ax.set_yticks(yy); ax.set_yticklabels(ylabels); ax.invert_yaxis()
        allv = [r["difference"] for r in rows_f if np.isfinite(r["difference"])]
        # Left headroom must accommodate a right-aligned label sitting to the left
        # of a negative marker, otherwise the label is clipped at the spine.
        _negatives = [v for v in allv if v < 0]
        _left_headroom = 0.14 if _negatives else 0.04
        lim_lo = min(-0.05, min(allv) - _left_headroom) if allv else -0.05
        lim_hi = max(0.18, max(allv) + 0.08) if allv else 0.18
        ax.set_xlim(lim_lo, lim_hi); ax.set_xlabel("Locked-test score difference")
        ax.text(0.995, 1.03, "○ ΔQWK   □ ΔMCC", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=5.4, color="#444444")
        _finish_axis(ax, "x")
        _save_csv(pd.DataFrame(rows_f), "figure_2f_information_source_contrasts.csv", fig_data_dir)
    else:
        _empty_panel(ax, "Representation contrasts unavailable")
    _soft_title(ax, "Information-source contribution"); _panel_label(ax, "f", x=-0.18)
    _save_csv(validation, "figure_2_candidate_validation_metrics.csv", fig_data_dir)
    _save_csv(test, "figure_2_selected_test_metrics.csv", fig_data_dir)
    _save(fig, "figure_2_model_comparison", out_dir)
    logger("[figure_2] saved aesthetic v3")
# Figure 3 — prediction behaviour and calibration
def figure_3_prediction_calibration(
    all_results: Dict,
    pub_outputs: Dict,
    output_dir: str,
    out_dir: str,
    fig_data_dir: str,
    logger=print,
):
    selected = _validation_selected_models(all_results, pub_outputs)
    scenarios = [s for s in SCENARIO_ORDER if s in selected]
    if not scenarios:
        logger("[figure_3] no selected models; skipping")
        return
    ece_df, bins_df = _calibration_tables(pub_outputs, output_dir)
    # Per-class ordinal error composition moved here from the robustness figure.
    # It subsumes the two panels it replaces: the class-wise recall heatmap was
    # its "Correct" segment, and the global error-composition bars were its
    # column-weighted marginal, both identical to the fifth decimal place.
    ordinal_errors = _per_class_ordinal_error_frame(all_results, selected, output_dir)
    fig = plt.figure(figsize=(7.25, 7.45))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.00, 1.05, 1.10],
                             left=0.065, right=0.94, top=0.935, bottom=0.075,
                             hspace=0.62)
    top_gs = outer[0].subgridspec(1, 4, wspace=0.40)
    mid_gs = outer[1].subgridspec(1, 4, wspace=0.48)
    bot_gs = outer[2].subgridspec(1, 3, width_ratios=[2.30, 1.0, 0.95], wspace=0.62)
    confusion_axes = []
    confusion_export = []; calibration_export = []
    # a–d confusion matrices
    last_im = None
    for i in range(4):
        ax = fig.add_subplot(top_gs[0, i]); confusion_axes.append(ax)
        if i >= len(scenarios): ax.set_axis_off(); continue
        scenario = scenarios[i]; model = selected[scenario]
        result = all_results[scenario]["all"].get(model, {})
        if "y_true" in result and "predictions" in result:
            yt = np.asarray(result["y_true"]); yp = np.asarray(result["predictions"]["y_pred"])
            cm = confusion_matrix(yt, yp, labels=list(range(4)))
            norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
            last_im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
            for r in range(4):
                for c in range(4):
                    ax.text(c, r, f"{cm[r,c]}\n{norm[r,c]:.0%}", ha="center", va="center",
                            fontsize=5.45, color=_adaptive_text_color(norm[r,c], 0, 1),
                            fontweight="bold" if r == c else "normal")
                    confusion_export.append({"Scenario": scenario, "Model": model,
                                             "true_class": CLASS_NAMES[r],
                                             "predicted_class": CLASS_NAMES[c],
                                             "count": int(cm[r,c]),
                                             "row_fraction": float(norm[r,c])})
            ax.set_xticks(range(4)); ax.set_xticklabels(["Mi", "Mo", "Se", "Pr"])
            ax.set_yticks(range(4)); ax.set_yticklabels(["Mi", "Mo", "Se", "Pr"] if i == 0 else [])
            ax.set_xlabel("Predicted")
            if i == 0: ax.set_ylabel("Observed")
            for spine in ax.spines.values(): spine.set_visible(True); spine.set_linewidth(0.45)
        else:
            _empty_panel(ax, "Predictions unavailable")
        ax.set_title(SCENARIO_SHORT[scenario], loc="left", pad=6, fontsize=8.0, color="#252525")
        ax.text(0.00, 1.01, MODEL_SHORT_LABELS.get(model, model), transform=ax.transAxes,
                ha="left", va="bottom", fontsize=5.3, color="#666666")
        _panel_label(ax, chr(ord("a") + i), x=-0.18)
    if last_im is not None:
        cax = fig.add_axes([0.952, 0.695, 0.012, 0.20])
        cb = fig.colorbar(last_im, cax=cax); cb.set_label("Within-observed-class proportion")
        cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=6.2)
    # e–h reliability curves
    for i in range(4):
        ax = fig.add_subplot(mid_gs[0, i])
        if i >= len(scenarios): ax.set_axis_off(); continue
        scenario = scenarios[i]; model = selected[scenario]
        ax.plot([0, 1], [0, 1], color="#9B9B9B", lw=0.8, ls=(0, (3, 2)), zorder=0)
        if not ece_df.empty and not bins_df.empty:
            rows = _plot_saved_calibration(ax, scenario, model, bins_df, ece_df)
            if not rows.empty:
                rows = rows.copy(); rows["Scenario"] = scenario; rows["Model"] = model
                calibration_export.append(rows)
            # Re-style lines created by helper
            for line in ax.lines[1:]:
                line.set_linewidth(1.05); line.set_markersize(3.7)
        else:
            _empty_panel(ax, "Calibration unavailable")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Predicted probability")
        if i == 0: ax.set_ylabel("Observed frequency")
        else: ax.set_ylabel("")
        _finish_axis(ax, "both"); _soft_title(ax, f"Reliability · {SCENARIO_SHORT[scenario]}", pad=6)
        _panel_label(ax, chr(ord("e") + i), x=-0.18)
    fig.legend(handles=[mpl.lines.Line2D([], [], color=PAL_CLASS[c], marker="o", lw=1,
                                        markersize=3.8, label=c) for c in CLASS_NAMES],
               loc="upper center", bbox_to_anchor=(0.51, 0.665), ncol=4,
               frameon=False, title="Severity class")
    # i per-class ordinal error composition (moved from the robustness figure;
    # replaces the former class-wise recall heatmap and global error-composition
    # bars, both of which it contains exactly)
    ax = fig.add_subplot(bot_gs[0, 0])
    if not ordinal_errors.empty:
        _plot_per_class_ordinal_composition_matrix_panel(
            ax, ordinal_errors, fig_data_dir,
            export_filename="figure_3i_per_class_ordinal_error_composition.csv",
            legend_bbox=(0.5, 1.16),
        )
    else:
        _empty_panel(ax, "Per-class ordinal errors unavailable")
    _soft_title(ax, "Per-class ordinal error composition", pad=14)
    _panel_label(ax, "i", x=-0.09)
    # j ECE heatmap
    ax=fig.add_subplot(bot_gs[0,1]); cal=ece_df.copy()
    if not cal.empty:
        if "calibration_status" in cal.columns:
            pref=cal[cal["calibration_status"].astype(str).str.lower()=="calibrated"]
            if not pref.empty: cal=pref
        cal=cal[cal["Scenario"].isin(scenarios)&cal["severity_class"].isin(CLASS_NAMES)]
        pivot=cal.pivot_table(index="severity_class",columns="Scenario",values="ece",aggfunc="first")
        pivot=pivot.reindex(index=CLASS_NAMES,columns=scenarios)*100
        im=ax.imshow(pivot.to_numpy(),cmap="Oranges",vmin=0,vmax=max(5,float(np.nanmax(pivot.to_numpy()))),aspect="auto")
        for r in range(4):
            for c in range(len(scenarios)):
                v=pivot.iloc[r,c]
                if np.isfinite(v): ax.text(c,r,f"{v:.1f}",ha="center",va="center",fontsize=5.7,
                                           color=_adaptive_text_color(v,0,max(5,float(np.nanmax(pivot.to_numpy())))))
        ax.set_xticks(range(len(scenarios)));ax.set_xticklabels(_scenario_short_list(scenarios))
        ax.set_yticks(range(4));ax.set_yticklabels(CLASS_NAMES,fontsize=6.0)
        [sp.set_visible(False) for sp in ax.spines.values()]
        _small_colorbar(ax,im,"ECE (%)",size="4%",pad=0.05)
        _save_csv(cal,"figure_3j_ece.csv",fig_data_dir)
    else: _empty_panel(ax,"ECE unavailable")
    _soft_title(ax,"Calibration error (ECE, %)");_panel_label(ax,"j",x=-0.18)
    # k calibration improvement
    ax=fig.add_subplot(bot_gs[0,2]); probability_rows=[]
    for scenario in scenarios:
        model=selected[scenario]; result=all_results[scenario]["all"].get(model,{})
        calmet=result.get("metrics",{}) or {}; raw=result.get("test_metrics_uncalibrated",{}) or {}
        probability_rows.append({"Scenario":scenario,
            "ΔRPS":float(raw.get("ranked_probability_score",np.nan)-calmet.get("ranked_probability_score",np.nan)),
            "ΔBrier":float(raw.get("multiclass_brier",np.nan)-calmet.get("multiclass_brier",np.nan))})
    prob=pd.DataFrame(probability_rows).set_index("Scenario")
    if prob.notna().any().any():
        arr=prob.to_numpy(dtype=float); lim=max(0.01,float(np.nanmax(np.abs(arr))))
        norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim)
        im=ax.imshow(arr,cmap="RdBu",norm=norm,aspect="auto")
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                v=arr[r,c]
                if np.isfinite(v): ax.text(c,r,f"{v:+.3f}",ha="center",va="center",fontsize=5.55,
                                           color="white" if abs(v)>0.58*lim else "#222222")
        ax.set_xticks(range(2));ax.set_xticklabels(["ΔRPS","ΔBrier"])
        ax.set_yticks(range(len(scenarios)));ax.set_yticklabels(_scenario_short_list(scenarios))
        [sp.set_visible(False) for sp in ax.spines.values()]
        _small_colorbar(ax,im,"Raw − calibrated",size="5%",pad=0.05)
        _save_csv(prob.reset_index(),"figure_3k_calibration_improvement.csv",fig_data_dir)
    else:_empty_panel(ax,"Probability metrics unavailable")
    _soft_title(ax,"Calibration improvement");_panel_label(ax,"k",x=-0.20)
    if confusion_export: _save_csv(pd.DataFrame(confusion_export),"figure_3_confusion_matrices.csv",fig_data_dir)
    if calibration_export: _save_csv(pd.concat(calibration_export,ignore_index=True),"figure_3_calibration_curves.csv",fig_data_dir)
    _save(fig,"figure_3_prediction_and_calibration",out_dir)
    logger("[figure_3] saved aesthetic v3")
# Figure 4 — robustness and audits
def _humanize_drop_reason(reason: str) -> str:
    """Return short, manuscript-facing labels for provenance exclusions."""
    key = str(reason).strip().lower()
    exact = {
        "high_mi_provenance_review_not_dropped": "High-MI review\n(retained)",
        "high_mi_provenance_review_noted": "High-MI review\n(retained)",
        "high_missing_training_fold": "Missing in\ntraining fold",
        "hpo_low_frequency_training_fold": "Rare HPO term",
        "low_variance_training_fold": "Low variance",
        "mondo_low_frequency_training_fold": "Rare MONDO class",
        "other": "Other",
    }
    if key in exact:
        return exact[key]
    cleaned = re.sub(r"_training_fold$", "", key)
    cleaned = cleaned.replace("high_", "").replace("_", " ").strip()
    cleaned = cleaned[:1].upper() + cleaned[1:] if cleaned else "Other"
    return _wrap_label(cleaned, 18)
# Figure 5 — interpretation and phenotype biology
# Keep the manifest synchronized with the final layouts.
# Compact per-class ordinal error panel.
def _per_class_ordinal_error_frame(
    all_results: Dict,
    selected: Dict[str, str],
    output_dir: str,
) -> pd.DataFrame:
    """Return selected-model per-class ordinal error composition.
    Prefer the pipeline-generated CSV. If it is absent, reconstruct the same
    quantities from the stored locked-test predictions so Figure 5 remains
    reproducible from ``all_results`` alone.
    """
    columns = [
        "Scenario", "Model", "true_class", "n_total",
        "frac_correct", "frac_adjacent", "frac_non_adjacent",
    ]
    path = os.path.join(output_dir, "results", "error_decomposition_per_class.csv")
    frame = pd.DataFrame(columns=columns)
    if os.path.exists(path):
        try:
            loaded = pd.read_csv(path)
            if {"Scenario", "Model", "true_class"}.issubset(loaded.columns):
                frame = loaded.copy()
        except Exception:
            frame = pd.DataFrame(columns=columns)
    # Keep only the validation-selected model for each scenario.
    if not frame.empty:
        keep = frame.apply(
            lambda row: selected.get(str(row.get("Scenario", "")))
            == str(row.get("Model", "")),
            axis=1,
        )
        selected_frame = frame[keep].copy()
        if not selected_frame.empty:
            frame = selected_frame
    required_metrics = {
        "frac_correct", "frac_adjacent", "frac_non_adjacent",
    }
    if frame.empty or not required_metrics.issubset(frame.columns):
        rows: List[Dict[str, Any]] = []
        for scenario in SCENARIO_ORDER:
            model_name = selected.get(scenario)
            if model_name is None:
                continue
            result = (
                all_results.get(scenario, {})
                .get("all", {})
                .get(model_name, {})
            )
            if "y_true" not in result or "predictions" not in result:
                continue
            y_true = np.asarray(result["y_true"], dtype=int)
            y_pred = np.asarray(result["predictions"].get("y_pred"), dtype=int)
            if y_true.shape != y_pred.shape:
                continue
            for class_index, class_name in enumerate(CLASS_NAMES):
                mask = y_true == class_index
                n_total = int(mask.sum())
                if n_total == 0:
                    values = (np.nan, np.nan, np.nan)
                else:
                    distance = np.abs(y_true[mask] - y_pred[mask])
                    values = (
                        float(np.mean(distance == 0)),
                        float(np.mean(distance == 1)),
                        float(np.mean(distance >= 2)),
                    )
                rows.append({
                    "Scenario": scenario,
                    "Model": model_name,
                    "true_class": class_name,
                    "n_total": n_total,
                    "frac_correct": values[0],
                    "frac_adjacent": values[1],
                    "frac_non_adjacent": values[2],
                })
        frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    for column in [
        "n_total", "frac_correct", "frac_adjacent", "frac_non_adjacent",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[
        frame["Scenario"].isin(SCENARIO_ORDER)
        & frame["true_class"].isin(CLASS_NAMES)
    ].copy()
    frame["_scenario_order"] = frame["Scenario"].map(
        {name: index for index, name in enumerate(SCENARIO_ORDER)}
    )
    frame["_class_order"] = frame["true_class"].map(CLASS_INDICES)
    return frame.sort_values(["_scenario_order", "_class_order"]).reset_index(drop=True)
# Per-class ordinal errors are drawn in Figure 4;
# Figure 5 has panels a-e.
def _plot_per_class_ordinal_composition_matrix_panel(
    ax,
    frame: pd.DataFrame,
    fig_data_dir: str,
    export_filename: str = "figure_3i_per_class_ordinal_error_composition.csv",
    legend_bbox: Tuple[float, float] = (0.5, 1.12),
) -> None:
    """Draw a dense scenario-by-class matrix of 100% stacked mini-bars."""
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
            values = np.asarray(
                [row.get(column, np.nan) for column, _, _ in segment_specs],
                dtype=float,
            )
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
                    x_left + (correct_fraction * cell_width) / 2,
                    row_index,
                    f"{correct_fraction:.0%}",
                    ha="center",
                    va="center",
                    fontsize=5.1,
                    color="white",
                    fontweight="bold",
                )
            export_rows.append({
                "Scenario": scenario,
                "Model": row.get("Model", ""),
                "true_class": class_name,
                "n_total": row.get("n_total", np.nan),
                "frac_correct": float(values[0]),
                "frac_adjacent": float(values[1]),
                "frac_non_adjacent": float(values[2]),
            })
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
        mpl.patches.Patch(facecolor=color, edgecolor="none", label=label)
        for _, label, color in segment_specs
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

def figure_4_robustness_composite(
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
        pub_outputs, "seed_sensitivity", output_dir,
        ["split_sensitivity_fixed_configuration.csv", "seed_sensitivity.csv"],
    )
    mi = _read_result_dataframe(
        pub_outputs, "mi_threshold_sensitivity", output_dir,
        ["mi_threshold_sensitivity.csv"],
    )
    perm = _read_result_dataframe(
        pub_outputs, "permutation_test", output_dir,
        ["permutation_test.csv"],
    )
    budget = _read_result_dataframe(
        pub_outputs, "budget_sensitivity", output_dir,
        ["budget_sensitivity.csv"],
    )
    budget_selection = _read_result_dataframe(
        pub_outputs, "shared_hpo_budget_selection", output_dir,
        ["shared_hpo_budget_training_cv_selection.csv"],
    )
    imbalance = _read_result_dataframe(
        pub_outputs, "imbalance_sensitivity", output_dir,
        ["imbalance_sensitivity.csv"],
    )
    leakage = _leakage_audit_frame(output_dir)
    selected = _validation_selected_models(all_results, pub_outputs)
    fig = plt.figure(figsize=(7.1, 5.30))
    outer = fig.add_gridspec(
        2, 1,
        height_ratios=[1.0, 1.0],
        left=0.070, right=0.982, top=0.958, bottom=0.070,
        hspace=0.62,
    )
    top = outer[0].subgridspec(1, 3, wspace=0.50)
    mid = outer[1].subgridspec(1, 3, width_ratios=[0.96, 0.96, 1.34], wspace=0.56)
    # a — split sensitivity
    ax = fig.add_subplot(top[0, 0])
    if not seed_df.empty and {"Scenario", "mcc"}.issubset(seed_df.columns):
        scenarios = _present_scenarios(seed_df["Scenario"].unique())
        rng = np.random.RandomState(37)
        for xi, scenario in enumerate(scenarios):
            values = _numeric(seed_df[seed_df["Scenario"] == scenario], "mcc").dropna().to_numpy()
            jitter = rng.normal(0, 0.045, len(values))
            ax.scatter(np.full(len(values), xi) + jitter, values, s=13,
                       color=PAL_SCENARIO[scenario], alpha=0.65, edgecolor="none")
            if len(values):
                mean = float(np.mean(values))
                lo, hi = (np.percentile(values, [2.5, 97.5]) if len(values) >= 4 else (values.min(), values.max()))
                ax.errorbar(xi, mean, yerr=np.array([[mean - lo], [hi - mean]]),
                            fmt="o", markersize=4.6, color=PAL_SCENARIO[scenario],
                            markeredgecolor="black", markeredgewidth=0.45,
                            ecolor="#555555", elinewidth=0.9, capsize=2.4, zorder=4)
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(_scenario_short_list(scenarios))
        ax.set_ylabel("Test-set MCC")
        _finish_axis(ax, "y")
        _save_csv(seed_df, "figure_4a_split_sensitivity.csv", fig_data_dir)
    else:
        _empty_panel(ax, "Split sensitivity unavailable")
    _soft_title(ax, "Split sensitivity")
    _panel_label(ax, "a", x=-0.19)
    # b — MI threshold
    ax = fig.add_subplot(top[0, 1])
    if not mi.empty and "Scenario" in mi.columns and "mcc" in mi.columns:
        threshold_col = "minimum_mi_threshold" if "minimum_mi_threshold" in mi.columns else "mi_threshold"
        for scenario in _present_scenarios(mi["Scenario"].unique()):
            sub = mi[mi["Scenario"] == scenario].sort_values(threshold_col)
            x_values = _numeric(sub, threshold_col)
            y_values = _numeric(sub, "mcc")
            ax.plot(x_values, y_values, marker="o", markersize=4.0, color=PAL_SCENARIO[scenario])
            if len(sub):
                dx = max(float(x_values.max()) * 0.018, 0.0002)
                ax.text(float(x_values.iloc[-1]) + dx, float(y_values.iloc[-1]),
                        SCENARIO_SHORT[scenario], color=PAL_SCENARIO[scenario], va="center", fontsize=5.8)
        ax.set_xlabel("Minimum MI threshold")
        ax.set_ylabel("Validation MCC")
        _finish_axis(ax, "both")
        _save_csv(mi, "figure_4b_mi_threshold.csv", fig_data_dir)
    else:
        _empty_panel(ax, "MI threshold unavailable")
    _soft_title(ax, "Feature-selection threshold")
    _panel_label(ax, "b", x=-0.19)
    # c — permutation null
    ax = fig.add_subplot(top[0, 2])
    if not perm.empty and "Scenario" in perm.columns:
        rows = perm[perm["Scenario"].isin(SCENARIO_ORDER)].copy()
        rows["_o"] = rows["Scenario"].map({scenario: i for i, scenario in enumerate(SCENARIO_ORDER)})
        rows = rows.sort_values("_o")
        yy = np.arange(len(rows))
        inference_ready = _permutation_inference_available(rows)
        count_col = next((name for name in ("n_permutations_successful", "n_permutations_requested", "n_permutations") if name in rows.columns), None)
        for yi, (_, row) in enumerate(rows.iterrows()):
            null_mean = float(row.get("permuted_mcc_mean", np.nan))
            observed = float(row.get("observed_mcc", np.nan))
            if np.isfinite(null_mean) and np.isfinite(observed):
                ax.plot([null_mean, observed], [yi, yi], color="#B5B5B5", lw=1.1, zorder=1)
            ax.scatter(null_mean, yi, s=28, facecolor="white", edgecolor="#666666", linewidth=0.8, zorder=3)
            ax.scatter(observed, yi, s=34, color=PAL_SCENARIO[row["Scenario"]], edgecolor="black", linewidth=0.45, zorder=4)
            if count_col is not None and pd.notna(row.get(count_col)):
                ax.text(1.01, yi, f"n={int(float(row[count_col]))}",
                        transform=ax.get_yaxis_transform(), va="center", fontsize=5.4, color="#555555")
        ax.set_yticks(yy)
        ax.set_yticklabels(_scenario_short_list(rows["Scenario"]))
        ax.invert_yaxis()
        ax.set_xlabel("Validation MCC")
        null_label = "Gene-block null"
        ax.legend(handles=[
                mpl.lines.Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor="#666666", label=null_label),
                mpl.lines.Line2D([], [], marker="o", linestyle="none", markerfacecolor="#777777", markeredgecolor="black", label="Observed"),
            ], frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20),
            ncol=2, fontsize=5.6, handletextpad=0.35, columnspacing=0.8)
        _finish_axis(ax, "x")
        _save_csv(rows, "figure_4c_permutation.csv", fig_data_dir)
    else:
        _empty_panel(ax, "Permutation analysis unavailable")
    _soft_title(ax, "Observed versus gene-block null")
    _panel_label(ax, "c", x=-0.19)
    # d — HPO capacity
    ax = fig.add_subplot(mid[0, 0])
    if not budget.empty and {"Scenario", "hpo_binary_budget", "qwk"}.issubset(budget.columns):
        plotted = []
        # The "all eligible terms" step is drawn just right of the largest numeric
        # cap and labelled "All" (otherwise it would sit at x = 10,000).
        _finite_caps = sorted({
            float(v) for v in _numeric(budget, "hpo_binary_budget").dropna()
            if float(v) < HPO_BUDGET_ALL_ELIGIBLE
        })
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
            selected_rows = budget_selection[budget_selection["selected_for_s2_and_s3"].astype(str).str.lower().isin(["true", "1", "yes"])]
            if not selected_rows.empty:
                cap = float(_numeric(selected_rows, "hpo_binary_budget").dropna().iloc[0])
                cap_label = "all" if cap >= HPO_BUDGET_ALL_ELIGIBLE else f"{cap:g}"
                cap = _all_pos if cap >= HPO_BUDGET_ALL_ELIGIBLE else cap
                ax.axvline(cap, color="#444444", lw=0.75, ls=(0, (2, 2)))
                ax.text(cap, 1.015, f"Selected = {cap_label}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=5.7)
        x_span = max(ax.get_xlim()[1] - ax.get_xlim()[0], 1.0)
        for scenario, x_last, y_last in plotted:
            ax.text(x_last + 0.015 * x_span, y_last, SCENARIO_SHORT[scenario], color=PAL_SCENARIO[scenario], va="center", fontsize=5.8, clip_on=False)
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
    # e — imbalance strategies
    ax = fig.add_subplot(mid[0, 1])
    if not imbalance.empty and {"Scenario", "imbalance_strategy", "mcc"}.issubset(imbalance.columns):
        scenarios = _present_scenarios(imbalance["Scenario"].unique())
        strategies = [s for s in ["sample_weight", "resampling", "none"] if s in set(imbalance["imbalance_strategy"].astype(str))]
        colors = {"sample_weight": "#0072B2", "resampling": "#E69F00", "none": "#909090"}
        markers = {"sample_weight": "o", "resampling": "s", "none": "D"}
        labels = {"sample_weight": "Weighting", "resampling": "Resampling", "none": "None"}
        yy = np.arange(len(scenarios))
        for yi, scenario in enumerate(scenarios):
            values = []
            for strategy in strategies:
                sub = imbalance[(imbalance["Scenario"] == scenario) & (imbalance["imbalance_strategy"].astype(str) == strategy)]
                values.append(float(_numeric(sub, "mcc").iloc[0]) if len(sub) else np.nan)
            finite = [value for value in values if np.isfinite(value)]
            if finite:
                ax.hlines(yi, min(finite), max(finite), color="#D1D1D1", lw=1.0)
            for strategy, value in zip(strategies, values):
                if np.isfinite(value):
                    ax.scatter(value, yi, s=30, marker=markers[strategy], color=colors[strategy],
                               edgecolor="black", linewidth=0.4,
                               label=labels[strategy] if yi == 0 else None, zorder=3)
        ax.set_yticks(yy)
        ax.set_yticklabels(_scenario_short_list(scenarios))
        ax.invert_yaxis()
        ax.set_xlabel("Validation MCC")
        ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.02, 0.03),
                  ncol=1, fontsize=5.25, handletextpad=0.30, labelspacing=0.25, borderaxespad=0)
        _finish_axis(ax, "x")
        _save_csv(imbalance, "figure_4e_imbalance.csv", fig_data_dir)
    else:
        _empty_panel(ax, "Imbalance analysis unavailable")
    _soft_title(ax, "Imbalance handling")
    _panel_label(ax, "e", x=-0.19)
    # f — provenance heatmap
    ax = fig.add_subplot(mid[0, 2])
    if not leakage.empty:
        totals = leakage.groupby("drop_reason")["n"].sum().sort_values(ascending=False)
        reasons = list(totals.head(5).index)
        display = leakage.copy()
        display["display_reason"] = display["drop_reason"].where(display["drop_reason"].isin(reasons), "Other")
        pivot = display.pivot_table(index="display_reason", columns="Scenario", values="n", aggfunc="sum", fill_value=0)
        scenarios = [scenario for scenario in SCENARIO_ORDER if scenario in pivot.columns]
        pivot = pivot.reindex(columns=scenarios)
        raw = pivot.to_numpy(float)
        log_values = np.log10(raw + 1)
        image = ax.imshow(log_values, cmap="Blues", aspect="auto")
        vmax = max(1.0, float(np.nanmax(log_values)))
        for row_index in range(raw.shape[0]):
            for col_index in range(raw.shape[1]):
                ax.text(col_index, row_index, f"{int(raw[row_index, col_index]):,}", ha="center", va="center", fontsize=5.35,
                        color=_adaptive_text_color(log_values[row_index, col_index], 0, vmax))
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
# Compact high-capacity heatmaps for 50 features.
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
    """Convert dense/sparse selected-feature matrices to a finite float array."""
    if matrix is None:
        return np.empty((0, 0), dtype=float)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2:
        return np.empty((0, 0), dtype=float)
    return array

def _build_s3_locked_test_pca(
    all_results: Dict,
    pub_outputs: Dict,
    container,
    fold_artifacts: Dict,
    results_collector,
    output_dir: str,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray, Optional[str], str]:
    """Build one leakage-safe PCA projection for the S3 phenotype scenario.
    PCA is fitted on the S3 training rows and applied to the locked-test rows.
    The preferred input is the in-memory selected S3 matrix.  For stand-alone
    publication rendering, where fold artefacts are commonly absent, the
    visualization reconstructs the phenotype matrix directly from
    ``container.full_df`` and the S3 HPO columns.  It deliberately does *not*
    unpickle the model bundle, because bundles created while ``FoldPreprocessor``
    lived in ``__main__`` are not portable to a separate rendering process.
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
    saved_projection = pub_outputs.get("s3_locked_test_pca")
    if saved_projection is not None:
        variance = saved_projection[[
            "PC1_explained_variance_ratio", "PC2_explained_variance_ratio",
        ]].iloc[0].to_numpy(dtype=float)
        return saved_projection, variance, str(model_name), ""
    result = (
        all_results.get(scenario, {})
        .get("all", {})
        .get(model_name, {})
    )
    artifact = dict((fold_artifacts or {}).get(scenario, {}) or {})
    full_df = getattr(container, "full_df", None)
    train_idx = np.asarray(
        artifact.get("train_idx", getattr(container, "train_idx", [])), dtype=int
    )
    test_idx = np.asarray(
        artifact.get("test_idx", getattr(container, "test_idx", [])), dtype=int
    )
    sel_mask = artifact.get("sel_mask")
    def _selected_matrix(partition: str):
        direct = _to_dense_float_matrix(artifact.get(f"X_{partition}"))
        if direct.size:
            return direct, f"in-memory selected X_{partition}", []
        full_matrix = _to_dense_float_matrix(artifact.get(f"X_{partition}_full"))
        if full_matrix.size and sel_mask is not None:
            try:
                selected_matrix = full_matrix[:, np.asarray(sel_mask)]
                names = list(artifact.get("sel_names", []))
                return (
                    _to_dense_float_matrix(selected_matrix),
                    f"in-memory X_{partition}_full + selection mask",
                    names,
                )
            except Exception:
                pass
        return np.empty((0, 0), dtype=float), "matrix unavailable", []
    X_train, train_source, matrix_names = _selected_matrix("train")
    X_test, test_source, _ = _selected_matrix("test")
    feature_names = list(matrix_names)
    # Publication-only runs do not preserve _fold_artifacts.  Reconstruct S3
    # directly from the phenotype columns that are already present in the data
    # container.  Prefer exact selected names exposed by SHAP/importance outputs;
    # otherwise use the complete S3 scenario column registry.
    if X_train.size == 0 or X_test.size == 0:
        if full_df is None or len(train_idx) == 0 or len(test_idx) == 0:
            return (
                pd.DataFrame(),
                np.asarray([np.nan, np.nan]),
                str(model_name),
                "S3 PCA needs container.full_df plus non-empty training and test indices.",
            )
        candidate_names: List[str] = []
        candidate_source = ""
        artifact_names = list(artifact.get("sel_names", []) or [])
        if artifact_names:
            candidate_names = artifact_names
            candidate_source = "fold-artifact selected feature names"
        if not candidate_names and results_collector is not None:
            shap_store = getattr(results_collector, "shap_values", {}) or {}
            shap_entry = shap_store.get(f"{scenario}_{model_name}")
            if isinstance(shap_entry, dict):
                candidate_names = list(shap_entry.get("features", []) or [])
                if candidate_names:
                    candidate_source = "selected feature names stored with SHAP"
        if not candidate_names and results_collector is not None:
            importance_store = getattr(results_collector, "feature_importances", {}) or {}
            importance_entry = importance_store.get(f"{scenario}_{model_name}")
            if isinstance(importance_entry, pd.DataFrame) and "feature" in importance_entry.columns:
                candidate_names = list(importance_entry["feature"].astype(str))
                if candidate_names:
                    candidate_source = "selected feature names stored with importance"
        if not candidate_names:
            if hasattr(container, "get_scenario_columns"):
                candidate_names = list(container.get_scenario_columns(scenario) or [])
            if not candidate_names:
                candidate_names = list(
                    (getattr(container, "scenario_cols", {}) or {}).get(scenario, [])
                )
            candidate_source = "S3 scenario phenotype-column registry"
        # S3 is explicitly the per-term HPO phenotype representation.  Restrict
        # to numeric HPO columns present in full_df and exclude any label-like or
        # PCA-derived columns even if a stale registry contains them.
        present_names = []
        seen = set()
        for name in candidate_names:
            name = str(name)
            low = name.lower()
            if name in seen or name not in full_df.columns:
                continue
            if not low.startswith("hpo_"):
                continue
            if low.startswith("hpo_pca_") or "severity" in low or "target" in low:
                continue
            present_names.append(name)
            seen.add(name)
        # Selected-feature metadata can occasionally be unavailable or use names
        # not present in the reconstructed frame.  Fall back to all raw HPO terms.
        if len(present_names) < 2:
            present_names = [
                str(name) for name in full_df.columns
                if str(name).lower().startswith("hpo_")
                and not str(name).lower().startswith("hpo_pca_")
                and "severity" not in str(name).lower()
                and "target" not in str(name).lower()
            ]
            candidate_source = "all raw HPO phenotype columns in container.full_df"
        if len(present_names) < 2:
            return (
                pd.DataFrame(),
                np.asarray([np.nan, np.nan]),
                str(model_name),
                "Fewer than two usable S3 HPO phenotype columns were found.",
            )
        train_frame = full_df.iloc[train_idx][present_names].apply(
            pd.to_numeric, errors="coerce"
        )
        test_frame = full_df.iloc[test_idx][present_names].apply(
            pd.to_numeric, errors="coerce"
        )
        # HPO columns are structural presence/absence indicators; missing values
        # therefore represent absent terms in this assembled phenotype matrix.
        train_values = train_frame.fillna(0.0).to_numpy(dtype=float)
        test_values = test_frame.fillna(0.0).to_numpy(dtype=float)
        # Keep only features with training variation.  This decision uses no test
        # labels or test distribution and avoids singular/empty PCA dimensions.
        train_variance = np.nanvar(train_values, axis=0)
        keep = np.isfinite(train_variance) & (train_variance > 1e-12)
        if int(keep.sum()) < 2:
            return (
                pd.DataFrame(),
                np.asarray([np.nan, np.nan]),
                str(model_name),
                "Fewer than two S3 HPO features vary in the training partition.",
            )
        X_train = train_values[:, keep]
        X_test = test_values[:, keep]
        feature_names = list(np.asarray(present_names, dtype=object)[keep])
        train_source = f"{candidate_source} (training rows)"
        test_source = f"{candidate_source} (locked-test rows)"
    if X_train.shape[1] != X_test.shape[1]:
        return (
            pd.DataFrame(),
            np.asarray([np.nan, np.nan]),
            str(model_name),
            f"S3 train/test feature counts differ: {X_train.shape[1]} vs {X_test.shape[1]}.",
        )
    if X_train.shape[0] < 2 or X_train.shape[1] < 2:
        return (
            pd.DataFrame(),
            np.asarray([np.nan, np.nan]),
            str(model_name),
            f"PCA needs at least two training rows and two features; got {X_train.shape}.",
        )
    y_true = artifact.get("y_test")
    if y_true is None:
        y_true = result.get("y_true")
    if y_true is None and hasattr(container, "y") and len(test_idx):
        y_true = np.asarray(container.y)[test_idx]
    predictions = result.get("predictions", {}) or {}
    y_pred = predictions.get("y_pred")
    if y_true is None or y_pred is None:
        return (
            pd.DataFrame(),
            np.asarray([np.nan, np.nan]),
            str(model_name),
            "S3 locked-test true labels or predicted labels are unavailable.",
        )
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=int).reshape(-1)
    n_test = min(len(X_test), len(y_true), len(y_pred))
    if n_test == 0:
        return (
            pd.DataFrame(),
            np.asarray([np.nan, np.nan]),
            str(model_name),
            "The S3 locked-test projection has zero aligned observations.",
        )
    X_test = X_test[:n_test]
    y_true = y_true[:n_test]
    y_pred = y_pred[:n_test]
    train_finite = np.where(np.isfinite(X_train), X_train, np.nan)
    medians = np.nanmedian(train_finite, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    X_train = np.where(np.isfinite(X_train), X_train, medians)
    X_test = np.where(np.isfinite(X_test), X_test, medians)
    try:
        if S3_PCA_STANDARDISE:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            # Binary indicators are already on a common 0/1 scale; PCA is run on
            # them directly so that the explained-variance ratios match the
            # delivered figure_5e CSV and the value quoted in the Results.
            X_train_scaled = X_train
            X_test_scaled = X_test
        pca = PCA(n_components=2, random_state=random_seed)
        pca.fit(X_train_scaled)
        coordinates = pca.transform(X_test_scaled)
    except Exception as exc:
        return (
            pd.DataFrame(),
            np.asarray([np.nan, np.nan]),
            str(model_name),
            f"S3 PCA fitting or projection failed: {exc}",
        )
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
        CLASS_NAMES[value] if 0 <= value < len(CLASS_NAMES) else str(value)
        for value in y_true
    ]
    frame["predicted_severity"] = [
        CLASS_NAMES[value] if 0 <= value < len(CLASS_NAMES) else str(value)
        for value in y_pred
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
    return frame, variance, str(model_name), ""

def _plot_s3_two_part_pca_panel(
    ax_left,
    ax_right,
    frame: pd.DataFrame,
    variance: np.ndarray,
    model_name: Optional[str],
    unavailable_reason: str = "",
):
    """Render Figure-5 panel e as two matched readable PCA views.
    To keep the dense central structure visible at journal width, the panel shows
    a stratified display sample rather than every point. All misclassified cases
    are retained, and correctly classified cases are downsampled by class.
    The axis labels preserve the true explained-variance percentages from PCA;
    the display sampling fraction is reported as an annotation instead of being
    substituted into the axis labels.
    Axis limits use a duplicate-aware central view: exact-duplicate projection
    clusters of at least S3_PCA_DUPLICATE_CLUSTER_MIN pairs are excluded from the
    limit calculation and the number of off-axis points is annotated. Type sizes
    are set from S3_PCA_FS.
    """
    if frame is None or frame.empty:
        for ax in (ax_left, ax_right):
            ax.set_axis_off()
        message = "S3 phenotype PCA unavailable"
        if unavailable_reason:
            message += "\n" + unavailable_reason
        ax_left.text(
            1.02, 0.50, message,
            transform=ax_left.transAxes,
            ha="center", va="center",
            fontsize=6.8, color="#666666", style="italic",
            wrap=True,
        )
        ax_left.text(
            0.0, 1.16, "S3 phenotype PCA",
            transform=ax_left.transAxes,
            ha="left", va="bottom", fontsize=8.0, color="#252525",
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
    valid = np.isin(true_values, np.arange(len(CLASS_NAMES))) & np.isin(pred_values, np.arange(len(CLASS_NAMES)))
    misclassified = true_values != pred_values
    # Show a smaller but still representative display set: all misclassified cases
    # plus a class-balanced 20% sample of correctly classified cases.
    shown = np.zeros(len(plot_frame), dtype=bool)
    if S3_PCA_SAMPLING == "proportional":
        # Sample within each observed class at one common rate, drawing correct
        # and misclassified pairs from the same pool. Class shares and the
        # within-class error rate are both preserved, so the proportion of
        # outlined points on the panel is the model's actual error rate.
        target_fraction = min(
            1.0, float(S3_PCA_TARGET_POINTS) / max(int(valid.sum()), 1)
        )
        for class_index in range(len(CLASS_NAMES)):
            indices = np.where(valid & (true_values == class_index))[0]
            if not len(indices):
                continue
            n_take = int(round(target_fraction * len(indices)))
            n_take = min(max(n_take, 1), len(indices))
            shown[rng.choice(indices, size=n_take, replace=False)] = True
    else:
        # Legacy "error_enriched" rule. Retained for reproducibility.
        target_fraction = 0.20
        min_per_class = 45
        max_per_class = 120
        shown |= misclassified & valid
        for class_index in range(len(CLASS_NAMES)):
            class_mask = valid & (~misclassified) & (true_values == class_index)
            indices = np.where(class_mask)[0]
            if not len(indices):
                continue
            n_take = int(np.ceil(target_fraction * len(indices)))
            n_take = max(min_per_class, n_take)
            n_take = min(max_per_class, n_take, len(indices))
            if n_take > 0:
                chosen = rng.choice(indices, size=n_take, replace=False)
                shown[chosen] = True
    plot_frame = plot_frame.loc[shown].copy().reset_index(drop=True)
    true_values = plot_frame["true_severity_numeric"].astype(int).to_numpy()
    pred_values = plot_frame["predicted_severity_numeric"].astype(int).to_numpy()
    valid = np.isin(true_values, np.arange(len(CLASS_NAMES))) & np.isin(pred_values, np.arange(len(CLASS_NAMES)))
    misclassified = true_values != pred_values
    finite_xy = np.isfinite(plot_frame["_display_PC1"].to_numpy()) & np.isfinite(plot_frame["_display_PC2"].to_numpy())
    if not finite_xy.any():
        for ax in (ax_left, ax_right):
            ax.set_axis_off()
        _empty_panel(ax_left, "Finite PCA coordinates unavailable")
        return
    def _quantile_limits(values: np.ndarray, lower: float, upper: float, pad_fraction: float) -> tuple[float, float]:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        lo, hi = np.nanpercentile(finite, [lower, upper])
        span = max(float(hi - lo), 1e-9)
        pad = pad_fraction * span
        return float(lo - pad), float(hi + pad)
    # Central view. Exact-duplicate projection clusters are excluded from the
    # axis limits: because every gene assigned to a MONDO disease carries the same
    # HPO annotation, large groups of pairs land on one identical coordinate and
    # would otherwise compress the central cloud into a few pixels. The number of
    # points falling outside the resulting window is reported on the panel.
    _full_xy = frame[["PC1", "PC2"]].to_numpy(dtype=float)
    _full_finite = np.isfinite(_full_xy).all(axis=1)
    _core_mask = _full_finite.copy()
    _n_duplicate_clusters = 0
    if _full_finite.any():
        _keys, _counts = np.unique(
            np.round(_full_xy[_full_finite], 9), axis=0, return_counts=True
        )
        _dup_keys = _keys[_counts >= int(S3_PCA_DUPLICATE_CLUSTER_MIN)]
        _n_duplicate_clusters = int(len(_dup_keys))
        for _key in _dup_keys:
            _core_mask &= ~(
                np.isclose(_full_xy[:, 0], _key[0]) & np.isclose(_full_xy[:, 1], _key[1])
            )
    if _core_mask.sum() < 20:
        _core_mask = _full_finite
        _n_duplicate_clusters = 0
    xlim = _quantile_limits(_full_xy[_core_mask, 0], 0.0, 100.0, 0.07)
    ylim = _quantile_limits(_full_xy[_core_mask, 1], 0.0, 100.0, 0.07)
    _off_axis_mask = (
        (_full_xy[:, 0] < xlim[0]) | (_full_xy[:, 0] > xlim[1])
        | (_full_xy[:, 1] < ylim[0]) | (_full_xy[:, 1] > ylim[1])
    )
    _n_off_axis = int(_off_axis_mask.sum())
    # Number of duplicate-projection clusters actually excluded from the plotted
    # window. This is NOT the same as the number dropped from the axis-limit
    # calculation above: a cluster can be dropped from the limit calculation and
    # still fall inside the padded window, in which case it is drawn. The panel
    # annotation and the exported stats must report what the reader can see, so
    # they use this count rather than len(_dup_keys).
    _n_duplicate_clusters_dropped_from_limits = int(_n_duplicate_clusters)
    if _n_duplicate_clusters_dropped_from_limits > 0 and _off_axis_mask.any():
        _off_keys = np.unique(
            np.round(_full_xy[_off_axis_mask & _full_finite], 9), axis=0
        )
        _dup_key_set = {
            (round(float(_k[0]), 9), round(float(_k[1]), 9)) for _k in _dup_keys
        }
        _n_duplicate_clusters = int(
            sum(
                1
                for _k in _off_keys
                if (round(float(_k[0]), 9), round(float(_k[1]), 9)) in _dup_key_set
            )
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
    def _draw(ax, colour_col: str, title: str, mark_errors: bool = True,
              colour_mode: str = "class"):
        if colour_mode == "error":
            # Ordinal distance between predicted and observed class, capped at
            # 2 so that every non-adjacent error shares one colour.
            gap = np.abs(
                plot_frame["predicted_severity_numeric"].astype(int).to_numpy()
                - plot_frame["true_severity_numeric"].astype(int).to_numpy()
            )
            point_colours = np.asarray(
                [S3_PCA_ERROR_COLOURS[int(min(value, 2))] for value in gap],
                dtype=object,
            )
        else:
            class_values = plot_frame[colour_col].astype(int).to_numpy()
            point_colours = np.asarray(
                [PAL_CLASS[CLASS_NAMES[value]] if 0 <= value < len(CLASS_NAMES) else "#999999"
                 for value in class_values],
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
        # The two subplots show the same 1,750 pairs, so outlining the
        # misclassified ones twice adds nothing. They are marked on the predicted
        # panel only, which is where the model's error is expressed.
        if mark_errors:
            # In error-magnitude mode the colour already identifies the error,
            # so the black outline is redundant ink.
            #
            # Under proportional sampling the marker is kept the same size as a
            # correct point and the outline is thin. Enlarging and darkening the
            # error markers, as the legacy rule did, makes the dense central
            # region read as though most pairs were misclassified even when the
            # displayed proportion is correct.
            _proportional = S3_PCA_SAMPLING == "proportional"
            _scatter_subset(
                ax,
                plot_frame.iloc[error_order]["_display_PC1"],
                plot_frame.iloc[error_order]["_display_PC2"],
                point_colours[error_order],
                size=17.0 if _proportional else 24.0,
                edgecolor="black" if colour_mode == "class" else "none",
                linewidth=(0.40 if _proportional else 0.55) if colour_mode == "class" else 0.0,
                alpha=0.80 if _proportional else 0.92,
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
    _error_mode = S3_PCA_RIGHT_PANEL == "error_magnitude"
    _draw(ax_left, "true_severity_numeric", "Observed severity", mark_errors=False)
    _draw(
        ax_right,
        "predicted_severity_numeric",
        "Ordinal error" if _error_mode else "Predicted severity",
        mark_errors=True,
        colour_mode="error" if _error_mode else "class",
    )
    # Sampling and off-axis counts belong in the figure legend, not on the panel.
    logging_note = (
        f"panel e: n={len(frame)} locked-test pairs, display sample={len(plot_frame)}, "
        f"off-axis={_n_off_axis} in {_n_duplicate_clusters} identical-projection clusters, "
        f"xlim={xlim}, ylim={ylim}"
    )
    try:
        print("[figure_5] " + logging_note)
    except Exception:
        pass
    ax_left.set_ylabel(
        f"PC2 ({pc2:.1%} of training variance)",
        fontsize=S3_PCA_FS["axis_label"], labelpad=3.0,
    )
    ax_left.set_xlabel(
        f"PC1 ({pc1:.1%} of training variance)",
        fontsize=S3_PCA_FS["axis_label"], labelpad=3.0,
    )
    ax_right.set_xlabel(
        f"PC1 ({pc1:.1%} of training variance)",
        fontsize=S3_PCA_FS["axis_label"], labelpad=3.0,
    )
    ax_right.set_ylabel("")
    ax_right.tick_params(labelleft=False)
    ax_left.text(
        0.0, 1.16,
        f"S3 phenotype principal-component projection · {model_label}",
        transform=ax_left.transAxes,
        ha="left", va="bottom", fontsize=S3_PCA_FS["panel_title"], color="#252525",
        clip_on=False,
    )
    _panel_label(ax_left, "e", x=-0.18, y=1.15, fs=S3_PCA_FS["panel_letter"])
    legend_handles = [
        mpl.lines.Line2D(
            [], [], linestyle="none", marker="o", markersize=6.4,
            markerfacecolor=PAL_CLASS[class_name], markeredgecolor="none",
            label=class_name,
        )
        for class_name in CLASS_NAMES
    ]
    if _error_mode:
        legend_handles.extend(
            mpl.lines.Line2D(
                [], [], linestyle="none", marker="o", markersize=6.4,
                markerfacecolor=colour, markeredgecolor="none",
                label=f"{label} (right)",
            )
            for colour, label in zip(
                S3_PCA_ERROR_COLOURS[1:], S3_PCA_ERROR_LABELS[1:]
            )
        )
    else:
        legend_handles.append(
            mpl.lines.Line2D(
                [], [], linestyle="none", marker="o", markersize=6.6,
                markerfacecolor="white", markeredgecolor="black",
                markeredgewidth=0.70, label="Misclassified (right panel only)",
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
    # Returned so the caller can write these to disk. The figure legend in the
    # manuscript quotes the display sample, the number of identical-projection
    # clusters excluded from the axis limits and the off-axis count; those
    # values change whenever the sampling or clustering rule changes, so they
    # must be read from the run rather than carried over between versions.
    gap_all = np.abs(
        plot_frame["predicted_severity_numeric"].astype(int).to_numpy()
        - plot_frame["true_severity_numeric"].astype(int).to_numpy()
    )
    return {
        "n_locked_test_pairs": int(len(frame)),
        "n_display_sample": int(len(plot_frame)),
        "n_misclassified_total": int((frame["predicted_severity_numeric"].astype(int)
                                      != frame["true_severity_numeric"].astype(int)).sum()),
        "n_misclassified_displayed": int((gap_all > 0).sum()),
        "n_adjacent_errors_displayed": int((gap_all == 1).sum()),
        "n_non_adjacent_errors_displayed": int((gap_all >= 2).sum()),
        "sampling_mode": S3_PCA_SAMPLING,
        "displayed_error_fraction": round(float((gap_all > 0).mean()), 4),
        "true_error_fraction": round(float(
            (frame["predicted_severity_numeric"].astype(int)
             != frame["true_severity_numeric"].astype(int)).mean()), 4),
        "n_identical_projection_clusters_excluded": int(_n_duplicate_clusters),
        "n_identical_projection_clusters_dropped_from_limits": int(
            _n_duplicate_clusters_dropped_from_limits
        ),
        "duplicate_cluster_min_size": int(S3_PCA_DUPLICATE_CLUSTER_MIN),
        "n_pairs_off_axis": int(_n_off_axis),
        "xlim_low": float(xlim[0]), "xlim_high": float(xlim[1]),
        "ylim_low": float(ylim[0]), "ylim_high": float(ylim[1]),
        "pc1_explained_variance_ratio": float(pc1),
        "pc2_explained_variance_ratio": float(pc2),
        "right_panel_mode": S3_PCA_RIGHT_PANEL,
    }

def figure_5_interpretation_biology(
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
    """Figure 5 with enlarged interpretation panels a-d and one S3 PCA panel e.
    The former feature heatmaps are removed. The S3 phenotype PCA is rendered as
    two matched manuscript-style scatter plots, with observed severity on the left
    and predicted severity on the right.
    """
    selected = _validation_selected_models(all_results, pub_outputs)
    scenarios = [scenario for scenario in SCENARIO_ORDER if scenario in selected]
    fig = plt.figure(figsize=(7.25, 9.65))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[6.70, 3.50],
        left=0.155,
        right=0.925,
        top=0.965,
        bottom=0.090,
        hspace=0.34,
    )
    top_grid = outer[0].subgridspec(2, 2, wspace=0.84, hspace=0.60)
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
            f"{SCENARIO_SHORT[scenario]} · "
            f"{MODEL_LABELS.get(selected[scenario], selected[scenario])}",
            pad=7.0,
        )
        _panel_label(
            ax,
            chr(ord("a") + panel_index),
            x=-0.31,
            y=1.045,
            fs=8.6,
        )
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
        all_results,
        pub_outputs,
        container,
        fold_artifacts,
        results_collector,
        output_dir,
        random_seed=random_seed,
    )
    pca_grid = outer[1].subgridspec(1, 2, wspace=0.18)
    ax_e_left = fig.add_subplot(pca_grid[0, 0])
    ax_e_right = fig.add_subplot(pca_grid[0, 1], sharex=ax_e_left, sharey=ax_e_left)
    _panel_e_stats = _plot_s3_two_part_pca_panel(
        ax_e_left,
        ax_e_right,
        pca_frame,
        variance,
        pca_model,
        unavailable_reason=pca_reason,
    )
    if _panel_e_stats:
        _save_csv(
            pd.DataFrame(
                sorted(_panel_e_stats.items()), columns=["quantity", "value"]
            ),
            "figure_5e_panel_stats.csv",
            fig_data_dir,
        )
        logger(
            "[figure_5] panel e: display sample "
            f"{_panel_e_stats['n_display_sample']} of "
            f"{_panel_e_stats['n_locked_test_pairs']}; "
            f"{_panel_e_stats['n_pairs_off_axis']} pairs off-axis in "
            f"{_panel_e_stats['n_identical_projection_clusters_excluded']} "
            "identical-projection clusters"
        )
    if pca_reason:
        logger(f"[figure_5] S3 PCA unavailable: {pca_reason}")
    elif "s3_locked_test_pca" in pub_outputs:
        logger(f"[figure_5] reused the saved S3 locked-test PCA: n={len(pca_frame)}")
    else:
        logger(
            f"[figure_5] S3 PCA plotted: n={len(pca_frame)}, "
            f"features reconstructed from {pca_frame['matrix_source_test'].iloc[0]}"
        )
    if not pca_frame.empty:
        _save_csv(
            pca_frame,
            "figure_5e_s3_true_predicted_pca.csv",
            fig_data_dir,
        )
    _save(fig, "figure_5_interpretation_and_biology", out_dir)
    logger(
        "[figure_5] saved enlarged panels a-d plus one two-part S3 phenotype PCA panel e"
    )

def _write_main_figure_manifest(out_dir: str, fig_data_dir: str) -> None:
    rows = [
        (
            "Extended Data Fig. 1",
            "a–e",
            "Severity composition, gene clustering, inheritance prevalence, "
            "disease clustering and candidate representations",
            "descriptive cohort summaries",
        ),
        (
            "Figure 2",
            "a–c",
            "Candidate model comparison",
            "outer validation only",
        ),
        (
            "Figure 2",
            "d–f",
            "Selected-model locked-test uncertainty, selection margin and "
            "representation contrasts",
            "locked test; validation-selected models only",
        ),
        (
            "Figure 3",
            "a–d",
            "Scenario confusion matrices",
            "locked test; selected models only",
        ),
        (
            "Figure 3",
            "e–h",
            "Scenario reliability curves",
            "locked test; selected models only",
        ),
        (
            "Figure 3",
            "i–k",
            "Per-class ordinal error composition, ECE and calibration improvement",
            "locked test; selected models only",
        ),
        (
            "Extended Data Fig. 2",
            "a–f",
            "Split, feature selection, permutation, HPO capacity, imbalance and "
            "leakage audit",
            "a: additional held-out test sets; b-e: primary validation set; f: feature audit",
        ),
        (
            "Figure 4",
            "a–e",
            "Enlarged scenario interpretation panels and combined S3 phenotype PCA",
            "selected models; PCA fitted on S3 training features and projected to locked test",
        ),
        (
            "Figure 4",
            "companion data",
            "SHAP summed within feature family, correcting the per-feature "
            "understatement of multi-column families",
            "locked test; selected models only",
        ),
        (
            "Table 1",
            "-",
            "Headline locked-test metrics including ordinal rank correlation "
            "and per-threshold AUC",
            "locked test; validation-selected models only",
        ),
        (
            "Table 2",
            "-",
            "Companion metrics with the gene as the evaluation unit; same "
            "locked predictions, no retraining",
            "locked test; validation-selected models only",
        ),
    ]
    frame = pd.DataFrame(
        rows,
        columns=["figure", "panels", "content", "evaluation_partition"],
    )
    frame.to_csv(os.path.join(out_dir, "main_figure_manifest.csv"), index=False)
    _save_csv(frame, "main_figure_manifest.csv", fig_data_dir)
