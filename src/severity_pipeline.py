"""Train and evaluate gene-disease severity models with grouped data splits."""


import os
import re
import sys
import json
import pickle
import hashlib
import warnings
import subprocess
import platform
import importlib
import importlib.metadata as importlib_metadata
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any, Sequence
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    GroupKFold, StratifiedGroupKFold, RandomizedSearchCV, ParameterSampler
)
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline as SkPipeline
try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    FrozenEstimator = None
from sklearn.metrics import (
    roc_auc_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    confusion_matrix,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    make_scorer,
)
from sklearn.feature_selection import mutual_info_classif
try:
    import lightgbm as lgb

    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("ERROR: LightGBM required.")
    sys.exit(1)
try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: SHAP not installed.")
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.combine import SMOTETomek
    from imblearn.pipeline import Pipeline as ImbPipeline

    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("WARNING: imbalanced-learn not installed.")
warnings.filterwarnings("default")
warnings.filterwarnings("ignore", message="X does not have valid feature names.*", category=UserWarning, module="sklearn")
sys.setrecursionlimit(10000)
from gene_disease_mapping_filter import (
    apply_rebuilt_mapping as _apply_rebuilt_gene_disease_mapping,
    resolve_mapping_path as _resolve_gene_disease_mapping_path,
)


class Config:
    """Single authoritative configuration for the complete pipeline."""

    ORGAN_CRITICALITY_COLUMNS = [
        "critical_brain", "critical_heart", "critical_lung", "critical_kidney",
        "critical_liver", "critical_immune", "critical_metabolic",
        "n_critical_organs", "is_systemic", "tissue_criticality_score",
        "noncritical_skin", "noncritical_dental", "noncritical_sensory",
        "noncritical_total", "n_noncritical_organs",
    ]

    # Mixed-inheritance counts can reveal disease context, so S1 excludes them.
    DROP_INHERITANCE_COUNT_PROXIES_FROM_S1 = True
    # Pair-level inheritance: moi_title values in the primary file mapped to the
    # dominant / recessive / X-linked flags. Values not listed (Mitochondrial,
    # Y-linked inheritance) count as a known "other" mode with all three flags 0.
    # Unknown or blank modes start as NaN; build_inheritance_features handles their coding.
    USE_PAIR_LEVEL_INHERITANCE = True
    PAIR_INHERITANCE_COLUMN = "moi_title"
    MOI_DOMINANT = ("Autosomal dominant", "Semidominant")
    MOI_RECESSIVE = ("Autosomal recessive",)
    MOI_XLINKED = ("X-linked", "X-linked recessive", "X-linked dominant")
    MOI_UNKNOWN = ("Unknown", "")
    # Gene-level inheritance rollups in the CORE file. Removed at load so they cannot
    # merge in alongside the pair-level columns.
    CORE_INHERITANCE_ROLLUP_COLUMNS = [
        "is_dominant", "is_recessive", "is_xlinked", "has_mixed_inheritance",
        "n_inheritance_modes", "all_inheritance_modes", "primary_inheritance",
    ]

    # CORE and EXTERNAL share names but differ in content; exclusions are source-specific.
    REMOVED_INPUT_COLUMNS_CORE = {

        # copies of gnomad_regional_*. The standard gene-level gnomad_* values are kept.
        "pLI": "gnomad_min_across_transcripts",
        "oe_lof": "gnomad_min_across_transcripts",
        "oe_mis": "gnomad_min_across_transcripts",
        "gnomad_regional_pLI": "gnomad_min_across_transcripts",
        "gnomad_regional_oe_lof": "gnomad_min_across_transcripts",
        "gnomad_regional_oe_mis": "gnomad_min_across_transcripts",
        # CORE copy of the ClinGen HI score is 0 or missing for every gene; the
        # rebuilt ClinGen columns come from the external file.
        "clingen_hi_score": "core_copy_constant",
    }
    REMOVED_INPUT_COLUMNS_EXTERNAL = {
        # Not Collins scores / empty; the Collins scores are pHaplo and pTriplo (CORE).
        "collins_pHI": "unidentified_source",
        "collins_pTS": "empty_for_every_gene",
        # Exact copies of other columns.
        "compensation_failure": "copy_of_collins_pHI",
        "rigid_no_backup": "copy_of_af_structural_rigidity",
        "hi_no_backup": "derived_from_constant_no_backup_flag",
        # Paralogue extraction produced zeros or blanks for every gene. The CORE
        # paralogue columns (paralog_count, max_paralog_identity, has_close_paralog,
        # redundancy_score, is_singleton) are kept.
        "n_paralogs": "constant_for_every_gene",
        "max_paralog_identity": "constant_for_every_gene",
        "closest_paralog_identity": "constant_for_every_gene",
        "closest_paralog_symbol": "constant_for_every_gene",
        "has_close_paralog": "constant_for_every_gene",
        "paralog_expression_corr": "constant_for_every_gene",
        "has_functional_backup": "constant_for_every_gene",
        "compensation_score": "constant_for_every_gene",
        "no_backup_flag": "constant_for_every_gene",
    }
    REMOVED_INPUT_COLUMNS = {
        **{f"CORE:{k}": v for k, v in REMOVED_INPUT_COLUMNS_CORE.items()},
        **{f"EXTERNAL:{k}": v for k, v in REMOVED_INPUT_COLUMNS_EXTERNAL.items()},
    }
    # Annotation strings in the ANNOTATION file that mean "no annotation".
    TEXT_PLACEHOLDER_TOKENS = ("unknown", "na", "n/a", "none", "nan", "null", "-")
    INHERITANCE_COUNT_PROXY_COLUMNS = [
        "n_inheritance_modes",
        "has_mixed_inheritance",
        "feat_has_mixed_inheritance",
    ]

    # gene_age is an exact linear function of conservation_score (r = -1.0).
    DROP_DUPLICATE_COLUMNS = True
    DUPLICATE_COLUMNS = [
        "gene_age",
    ]

    BASE_DIR = "."
    OUTPUT_DIR = os.path.join(BASE_DIR, "results/main")
    CONSOLIDATED_DIR = os.path.join(BASE_DIR, "CONSOLIDATED_FEATURES_v2")

    # Inputs.
    PRIMARY_DATA = "final_output_with_severity.csv"
    # HPO-verified mapping excludes gene-panel-collapse artefacts.
    REBUILT_GENE_DISEASE_MAPPING = os.path.join(
        BASE_DIR, "gene_disease_mapping_HPO_rebuilt.csv"
    )
    APPLY_REBUILT_GENE_DISEASE_MAPPING = True
    INCLUDE_HPO_ADDED_PAIRS = False
    ANNOTATION_DATA = "gene_annotations.csv"
    CORE_FEATURES = os.path.join(CONSOLIDATED_DIR, "CORE_MECHANISTIC_FEATURES.csv")
    DOMAIN_FEATURES = os.path.join(
        BASE_DIR, "enhanced_variant_features", "DOMAIN_CARRIER_FEATURES.csv"
    )
    # external file: AlphaFold columns plus ClinGen dosage columns rebuilt from
    # the ClinGen gene curation list dated 30 Oct 2026
    # (CONSOLIDATED_FEATURES_v2/clingen_2026-10-31/). Uncurated genes are 0 in every
    # ClinGen column with clingen_dosage_curated = 0. clingen_hi_score and
    # clingen_ts_score are ordinal 0-3; ClinGen codes 30 (autosomal recessive) and 40
    # (dosage sensitivity unlikely) are separate flags. Built by build_clingen_v46.py.
    # REMOVED_INPUT_COLUMNS also applies to the fallback source.
    _EXTERNAL_V46 = os.path.join(
        CONSOLIDATED_DIR, "EXTERNAL_MECHANISTIC_FEATURES_v4_6.csv"
    )
    _EXTERNAL_FIXED = os.path.join(
        CONSOLIDATED_DIR, "EXTERNAL_MECHANISTIC_FEATURES_CLINGEN_FIXED.csv"
    )
    EXTERNAL_FEATURES = (
        _EXTERNAL_V46
        if os.path.exists(_EXTERNAL_V46)
        else _EXTERNAL_FIXED
        if os.path.exists(_EXTERNAL_FIXED)
        else os.path.join(CONSOLIDATED_DIR, "EXTERNAL_MECHANISTIC_FEATURES.csv")
    )
    # Published gene-level constraint / expression sources assembled by
    # build_new_gene_features.py: s_het (GeneBayes), Human Protein Atlas
    # specificity, BrainSpan prenatal-vs-postnatal expression, DOMINO, GDI.
    # Optional: the pipeline runs without this file.
    NEW_GENE_FEATURES = os.path.join(CONSOLIDATED_DIR, "NEW_GENE_LEVEL_FEATURES.csv")
    MOUSE_LETHALITY_FEATURES = os.path.join(CONSOLIDATED_DIR, "MOUSE_LETHALITY_FEATURES.csv")
    MOUSE_HUMAN_PHENOTYPE_FEATURES = os.path.join(
        CONSOLIDATED_DIR, "MOUSE_HUMAN_PHENOTYPE_FEATURES.csv"
    )
    MONDO_OBO_PATH = os.path.join(BASE_DIR, "mondo.obo")
    HPO_OBO_PATH = os.path.join(BASE_DIR, "hp.obo")
    HPO_OBO_URL = "https://purl.obolibrary.org/obo/hp.obo"
    ALLOW_ONTOLOGY_DOWNLOAD = False

    # Reporting dependencies are imported and checked before data assembly/training.
    REPORTING_MODULES = {
        "publication_figures": "generate_publication_figures",
        "run_report": "generate_manuscript_report",
    }

    # Target and split policy.
    TARGET_MAP = {"Mild": 0, "Moderate": 1, "Severe": 2, "Profound": 3}
    NUM_CLASSES = 4
    TEST_SIZE = 0.20
    VAL_SIZE = 0.15
    SPLIT_SEARCH_ITERATIONS = 1000
    SPLIT_MIN_CLASS_COUNT = 1
    STRICT_SINGLE_TEST_EVALUATION = True
    SPLIT_SENSITIVITY_SEED_OFFSET = 1  # excludes the primary seed 42
    RUN_SEED_SENSITIVITY = True
    N_SEEDS_SENSITIVITY = 5

    # Nested model development.
    NESTED_CV_FOLDS = 3
    CANDIDATE_MODEL_ORDER = (
        "MultinomialLogisticRegression",
        "ExtraTrees",
        "LightGBM",
        "OrdinalDecomp",
    )
    NESTED_SEARCH_ITERATIONS = {
        "MultinomialLogisticRegression": 10,
        "ExtraTrees": 20,
        "LightGBM": 20,
        "OrdinalDecomp": 15,
    }
    MODEL_SELECTION_RULE = "lexicographic_qwk_then_mcc_then_balanced_accuracy"


    HPO_ANCESTOR_EXPANSION_ENABLED = False
    HPO_ANCESTOR_MAX_HOPS = 0
    HPO_ANCESTOR_EXCLUDED_IDS = ("HP:0000001", "HP:0000118")
    HPO_MIN_TRAIN_COUNT = 2
    HPO_MAX_TRAIN_FRACTION = 0.90
    # Select the shared HPO budget in training folds; the final sentinel is uncapped.
    HPO_BUDGET_ALL_ELIGIBLE = 10000
    HPO_BUDGET_CANDIDATES = [
        200, 300, 400, 600, 800, 1200, 1600, 2400, 3200, 5200, HPO_BUDGET_ALL_ELIGIBLE,
    ]
    SHARED_HPO_BUDGET_SCENARIOS = ("S2_Combined", "S3_Phenotype")
    S3_BUDGET_SELECTION_LGB_PARAMS = {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 7,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.5,
    }
    DROP_SEVERITY_HPO_TERMS = True
    DROP_HPO_DERIVED_IC = True
    RUN_SEVERITY_HPO_SENSITIVITY = True

    # Feature selection and missingness.
    MI_TOP_K = None
    MI_THRESHOLD_DEFAULT = 0.01
    MI_THRESHOLD_MECHANISTIC = 0.005
    MI_LEAKAGE_THRESHOLD = None
    MI_HIGH_AUDIT_QUANTILE = 0.99
    MI_HIGH_AUDIT_MIN_SCORE = 0.20
    MI_POSITIVE_EPS = 1e-12
    MI_MAX_DISCRETE_UNIQUE = 10
    MI_MIN_FEATURES_DEFAULT = 15
    MI_MIN_FEATURES_MECHANISTIC = 30
    # Missingness flags measure database coverage; missing values use the training median.
    ADD_MISSING_INDICATORS = False
    # Source-presence flags are constant in this corpus.
    ADD_SOURCE_AVAILABILITY_INDICATORS = False
    MISSING_DROP_THRESHOLD = 0.5
    VAR_DROP_THRESHOLD = 1e-10
    MONDO_MIN_TRAIN_COUNT = 5
    # Disease classes are direct subclasses of MONDO:0700096 in the pinned ontology.
    MONDO_CLASS_ROOT = "MONDO:0700096"
    FEATURE_BUDGET = {
        "hpo_binary": 200,
        "hpo_derived": 10,
        "organ_system": 25,
        "mondo": 50,
        "inheritance": 8,
        "annotation_text": 80,
        "primary_text": 80,
        "engineered": 50,
        "mechanistic_core": 120,
        "mouse_lethality": 20,
        "mouse_phenotype": 20,
        "variant": 40,
        "external": 40,
        "missingness_indicator": 20,
    }
    TEXT_VOCAB_TOP_N = 20

    # Models, imbalance, calibration and explanations.
    SAVE_ALL_BASE_MODEL_BUNDLES = True
    MULTINOMIAL_LOGISTIC_ENABLED = True
    EXTRA_TREES_ENABLED = True
    ORDINAL_DECOMPOSITION_ENABLED = True
    RESAMPLING_ENABLED = False
    IMBALANCE_STRATEGY = "sample_weight"
    RUN_IMBALANCE_SENSITIVITY = True
    CALIBRATION_METHOD_LARGE = "isotonic"
    CALIBRATION_METHOD_SMALL = "sigmoid"
    CALIBRATION_MIN_POSITIVES = 100
    CALIBRATION_MIN_NEGATIVES = 100
    CALIBRATION_N_BINS = 5
    CALIBRATION_BINNING = "quantile"
    CALIBRATION_N_BOOTSTRAP = 1000
    ORDINAL_EXPLANATION_POLICY = "uncalibrated_boundary_models_only"
    ORDINAL_BOUNDARY_MULT = [2.5, 1.5, 1.0, 1.0]

    # These columns measure annotation coverage rather than biology.
    DROP_RECORD_KEEPING_FEATURES = True
    RECORD_KEEPING_COLUMNS = (
        "feat_mondo_depth",                      # level of the disease in the MONDO tree
        "feat_mondo_n_classes",                  # number of MONDO top-level classes
        "mouse_lethal_pheno_n_hpos_with_data",   # coverage of mouse data, not lethality
        "mouse_phenotype_count",                 # number of mouse phenotypes recorded
    )
    # Expression sites in gene_annotations.csv are capped at 10 per gene, so their
    # count and indicators mainly reflect annotation depth; HPA, GTEx and BrainSpan
    # already describe expression.
    # interaction partners are also excluded, so that no protein-protein
    # interaction data enter the models (same reason as the network features).
    TEXT_EXCLUDED_SOURCE_COLUMNS = ("Expression_Sites", "Interacting_Genes")
    # Per-category annotation counts and "annotation present" flags measure how well
    # a gene is studied. Term indicators (top 20 per category) are kept.
    TEXT_ADD_COUNT_AND_AVAILABILITY = False


    INHERITANCE_UNKNOWN_AS_FLAG = True
    # no separate "inheritance not recorded" feature. Unrecorded pairs stay 0 on
    # every inheritance flag, so no mode is invented, but the gap is not a feature.
    INHERITANCE_NOT_RECORDED_FLAG = False
    # Exclude study-intensity proxies, interaction-network features, s_het uncertainty
    # and the ClinGen curation flag; curated sources favour well-studied genes.
    DROP_NON_BIOLOGICAL_FEATURES = True
    NON_BIOLOGICAL_PREFIXES = ("var_", "ppi_", "network_")
    NON_BIOLOGICAL_COLUMNS = (
        "shet_ci_width_log10",
        "clingen_dosage_curated",
        "feat_density_pathogenic", "feat_density_lof", "feat_density_missense",
        "feat_prop_pathogenic", "feat_prop_lof", "feat_ratio_lof_missense",
        "feat_hub_constrained", "feat_hub_essential", "feat_betweenness_x_pli",
        "feat_triple_constraint", "feat_central_essential", "feat_log_ppi_degree",
    )

    # Exclude derived features and near-duplicates (|Spearman rho| >= 0.9, 4,733 genes).
    # Keep one expression-specificity measure per level: gtex_tau and
    # hpa_celltype_group_specificity_ord. Selection does not use SHAP ranks.
    DROP_REDUNDANT_FEATURES = True
    REDUNDANT_FEATURE_COLUMNS = (
        # Derived from a kept feature (cut-off, count or transform).
        "is_ancient",                   # = conservation_score >= 0.75
        "depmap_n_essential",           # rank-identical to depmap_essential_fraction (rho 1.0)
        "depmap_is_pan_essential",      # = depmap_essential_fraction >= 0.9
        "depmap_is_context_essential",  # = 0.1 < depmap_essential_fraction < 0.9
        "hpa_is_low_specificity",       # = hpa_tissue_specificity_ord == 1
        "feat_pli_squared",             # monotone transform of gnomad_pLI
        # Orthologue flags, summarised by conservation_score (evolutionary depth).
        "has_yeast_ortholog", "has_fly_ortholog", "has_worm_ortholog",
        "has_zebrafish_ortholog", "has_mouse_ortholog",
        # AlphaFold: near-duplicates of af_mean_plddt (rho 0.92 to 0.996);
        # af_mean_plddt and af_plddt_std (rho -0.68) are kept.
        "af_median_plddt", "af_frac_confident", "af_frac_very_high",
        "af_structural_rigidity", "af_frac_disordered", "af_disorder_ratio",
        # Gene-size proxies: rho 0.90 to 0.98 with coding length; af_protein_length and
        # gnomad_gene_length are kept (gnomad_obs_lof, rho 0.47, is kept).
        "gnomad_cds_length", "gnomad_exp_lof", "gnomad_exp_mis", "gnomad_obs_mis",
        # Synonymous constraint is a calibration check in gnomAD, not a severity measure.
        "gnomad_syn_z",
        # Inheritance x constraint products; tree models form these interactions themselves.
        "feat_pli_x_dominant", "feat_pli_x_recessive",
        "feat_dominant_low_constraint", "feat_recessive_high_constraint",
    )

    # SHAP explains sum_k k * p_k(x) from the uncalibrated selected model.
    # Positive values mean greater severity on the same scale across scenarios.
    SEVERITY_SHAP_ENABLED = True
    SEVERITY_SHAP_BACKGROUND_N = 100              # training pairs, interventional SHAP
    SEVERITY_SHAP_PERMUTATION_BACKGROUND_N = 25   # fallback for non-tree models

    # Sensitivity and inference.
    RUN_MI_SENSITIVITY = True
    MI_SENSITIVITY_THRESHOLDS = [0.0, 0.005, 0.01, 0.02]
    RUN_BUDGET_SENSITIVITY = True
    RUN_PERMUTATION_TEST = True
    N_PERMUTATIONS = 100
    PERMUTATION_MODE = "gene_block_candidate_selection_fixed_hyperparameters"
    PERMUTATION_MIN_MOVABLE_FRACTION = 0.80
    RUN_ADDITIONAL_GENERALISATION = True
    ADDITIONAL_GENERALISATION_MODES = [
        "disease_disjoint", "gene_and_disease_disjoint"
    ]

    # Uncertainty, compute and output policy.
    N_BOOTSTRAP = 1000
    FIGURE_DPI = 300
    MODEL_N_JOBS = 4
    USE_GPU = False
    SAVE_MI_CSV_FOR_INTERNAL_RUNS = False
    ALLOW_OUTPUT_REUSE = False

    # Labels come from an evidence-grounded AI agent using the Lazarin and ACMG
    # framework; validation uses expert-curated cohorts.
    TARGET_LABEL_PROVENANCE = (
        "Severity labels were assigned by an evidence-grounded artificial-intelligence "
        "agent operationalising the Lazarin and ACMG severity framework from published "
        "phenotype evidence, with concordance reported against expert-curated cohorts; "
        "they are not case-by-case expert adjudications. "
        "HPO-aware scenarios therefore assess reproducibility and internal "
        "generalisation of phenotype-informed curation, not validation against a "
        "phenotype-independent outcome."
    )


class Logger:

    def __init__(self, log_file: str):
        self.log_file = log_file
        self.logs = []
        self.start_time = datetime.now()

    def log(self, msg: str, level: str = "INFO", print_msg: bool = True):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full = f"[{ts}] [{level}] {msg}"
        self.logs.append(full)
        if print_msg:
            print(full)

    def section(self, title: str):
        self.log(title)

    def subsection(self, title: str):
        self.log(title)

    def save(self):
        elapsed = datetime.now() - self.start_time
        self.log(f"Total runtime: {elapsed}")
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"SEVERITY PREDICTION PIPELINE v{PIPELINE_VERSION} LOG (GENE-DISEASE PAIR)\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            for entry in self.logs:
                f.write(entry + "\n")


class ResultsCollector:

    def __init__(self):
        self.results = defaultdict(lambda: defaultdict(dict))
        self.predictions = defaultdict(dict)
        self.feature_importances = defaultdict(dict)
        self.shap_values = defaultdict(dict)

    def add_result(self, scenario, stratum, model, metrics):
        self.results[scenario][stratum][model] = metrics

    def add_predictions(self, key, y_true, y_pred, y_prob):
        self.predictions[key] = {
            "y_true": np.array(y_true),
            "y_pred": np.array(y_pred),
            "y_prob": np.array(y_prob),
        }

    def add_feature_importance(self, key, importance_df):
        self.feature_importances[key] = importance_df

    def add_shap_values(self, key, shap_vals, feature_names):
        self.shap_values[key] = {"values": shap_vals, "features": feature_names}

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for scenario, strata in self.results.items():
            for stratum, models in strata.items():
                for model, metrics in models.items():
                    rows.append(
                        {"Scenario": scenario, "Stratum": stratum, "Model": model, **metrics}
                    )
        return pd.DataFrame(rows)

    def save(self, filepath: str):
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
        with open(filepath.replace(".csv", ".pkl"), "wb") as f:
            pickle.dump(
                {
                    "results": dict(self.results),
                    "predictions": dict(self.predictions),
                    "feature_importances": dict(self.feature_importances),
                    "shap_values": dict(self.shap_values),
                },
                f,
            )


class DataContainer:

    def __init__(self):
        self.full_df: Optional[pd.DataFrame] = None
        self.y: Optional[np.ndarray] = None
        self.groups: Optional[np.ndarray] = None
        self.train_idx: Optional[np.ndarray] = None
        self.test_idx: Optional[np.ndarray] = None
        self.metadata: Optional[pd.DataFrame] = None
        self.scenario_cols: Dict[str, List[str]] = {}
        self.hpo_binary_cols: List[str] = []
        self.organ_sys_cols: List[str] = []
        self.mondo_cols: List[str] = []
        self.source_availability_cols: List[str] = []
        self.scenario_static_drops: Dict[str, List[Tuple[str, str]]] = {}
        self.feature_family_lookup: Dict[str, str] = {}
        # Scenario-scope exclusions, distinct from severity-circularity drops.
        self.inheritance_proxy_drops_S1: List[str] = []

    def get_scenario_columns(self, scenario: str) -> List[str]:
        return self.scenario_cols.get(scenario, [])


PIPELINE_VERSION = "1.0.0"


RANDOM_SEED = 42


np.random.seed(RANDOM_SEED)


qwk_scorer = make_scorer(cohen_kappa_score, weights="quadratic")


_high_mi_audit_records: List[Dict[str, Any]] = []


_REPORTING_MODULE_CACHE: Dict[str, Any] = {}


SEVERITY_HPO_IDS = {
    "HP:0001522",
    "HP:0003811",
    "HP:0011421",
    "HP:0011420",
    "HP:0003819",
    "HP:0034241",
    "HP:0002187",
    "HP:0011344",
    "HP:0011342",
    "HP:0011343",
    "HP:0001263",
    "HP:0006887",
    "HP:0010864",
    "HP:0002342",
    "HP:0001256",
    "HP:0001249",
    "HP:0003828",
}


SEVERITY_HPO_KEYWORDS = ["profound", "severe", "mild", "moderate"]


PLI_HIGH_CONSTRAINT = 0.9  # pLI above this => high loss-of-function intolerance


PLI_LOW_CONSTRAINT = 0.1   # pLI below this => low constraint


_UNWEIGHTED_TUNING_FALLBACKS: List[Dict[str, str]] = []


_test_evaluation_registry: Dict[str, int] = defaultdict(int)


_fold_artifacts: Dict[str, Dict[str, Any]] = {}


FEATURE_DISPLAY_LABELS: Dict[str, str] = {

    "shet_log10": "LoF selection, log10 s_het",
    "shet_post_mean": "LoF selection (s_het)",
    "shet_percentile": "s_het percentile rank",
    "shet_ci_width_log10": "s_het estimate uncertainty (95% CI width)",
    "shet_post_lower95": "s_het lower 95% bound",
    "shet_post_upper95": "s_het upper 95% bound",
    "shet_prior_mean": "s_het prior mean",
    "hpa_tissue_specificity_ord": "Tissue-restricted expression (HPA)",
    "hpa_celltype_specificity_ord": "Cell-type-restricted expression (HPA)",
    "hpa_celltype_group_specificity_ord": "Cell-lineage-restricted expression (HPA)",
    "hpa_is_low_specificity": "Broadly expressed across tissues (HPA)",
    "hpa_is_tissue_enriched": "Enriched in a single tissue (HPA)",
    "hpa_is_celltype_enriched": "Enriched in a single cell type (HPA)",
    "hpa_not_detected": "Not detected in any tissue (HPA)",
    "hpa_brain_regional_specific": "Brain-region-specific expression (HPA)",
    "hpa_tissue_specificity_score": "Tissue enrichment score (HPA)",
    "hpa_celltype_specificity_score": "Cell-type enrichment score (HPA)",
    "brainspan_prenatal_mean_rpkm": "Prenatal brain expression (BrainSpan)",
    "brainspan_postnatal_mean_rpkm": "Postnatal brain expression (BrainSpan)",
    "brainspan_prenatal_fraction": "Prenatal share of brain expression (BrainSpan)",
    "brainspan_prenatal_log2fc": "Prenatal vs postnatal brain expression, log2",
    "brainspan_peak_dpc": "Age at peak brain expression (BrainSpan)",
    "brainspan_peak_is_prenatal": "Brain expression peaks before birth",
    "brainspan_max_rpkm_log": "Peak brain expression level (BrainSpan)",
    "brainspan_expressed_in_brain": "Expressed in brain at any stage (BrainSpan)",
    "domino_score": "Probability of dominant inheritance (DOMINO)",
    "domino_lda": "DOMINO discriminant score",
    "domino_likely_dominant": "Predicted dominant gene (DOMINO)",
    "domino_likely_recessive": "Predicted recessive gene (DOMINO)",
    "gdi_phred": "Gene damage index, phred-scaled (GDI)",
    "gdi_raw": "Gene damage index (GDI)",
    "gdi_log": "Gene damage index, log10 (GDI)",
    "gdi_mendelian_damage_high": "High background damage in Mendelian genes (GDI)",

    "gnomad_pLI": "Loss-of-function intolerance (pLI)",
    "pLI": "Loss-of-function intolerance (pLI)",
    "gnomad_regional_pLI": "Regional loss-of-function intolerance (pLI)",
    "feat_pli_squared": "pLI, squared",
    "gnomad_oe_lof": "Observed/expected LoF variants",
    "oe_lof": "Observed/expected LoF variants",
    "gnomad_oe_lof_upper": "LoF constraint, upper bound (LOEUF)",
    "gnomad_oe_mis": "Observed/expected missense variants",
    "oe_mis": "Observed/expected missense variants",
    "gnomad_regional_oe_lof": "Regional observed/expected LoF",
    "gnomad_regional_oe_mis": "Regional observed/expected missense",
    "gnomad_mis_z": "Missense constraint (Z score)",
    "gnomad_lof_z": "LoF constraint (Z score)",
    "gnomad_syn_z": "Synonymous constraint (Z score)",
    "gnomad_obs_lof": "Observed LoF variants",
    "gnomad_exp_lof": "Expected LoF variants",
    "gnomad_obs_mis": "Observed missense variants",
    "gnomad_exp_mis": "Expected missense variants",
    "gnomad_cds_length": "Coding sequence length",
    "gnomad_gene_length": "Gene length",
    "feat_missz_minus_synz": "Missense minus synonymous constraint",
    "feat_missz_x_pli": "Missense constraint x pLI",
    "feat_triple_constraint": "Constrained by LoF, missense and dosage",

    "pHaplo": "Haploinsufficiency probability (pHaplo)",
    "pTriplo": "Triplosensitivity probability (pTriplo)",
    "collins_pHI": "Haploinsufficiency probability (Collins)",
    "collins_pTS": "Triplosensitivity probability (Collins)",
    "clingen_hi_score": "ClinGen haploinsufficiency rating",
    "clingen_hi_score_x": "ClinGen haploinsufficiency rating",
    "clingen_hi_score_y": "ClinGen haploinsufficiency rating",
    "clingen_ts_score": "ClinGen triplosensitivity rating",
    "hi_sensitive": "ClinGen haploinsufficiency flag set",
    "ts_sensitive": "ClinGen triplosensitivity flag set",
    "is_haploinsufficient": "Haploinsufficient gene",
    "is_triplosensitive": "Triplosensitive gene",
    # hi_sensitive OR ts_sensitive; both source flags are unreliable in the
    # current DOSAGE_FEATURES.csv. Label states what the column holds.
    "dosage_sensitive_any": "ClinGen HI or TS flag set",
    "ordered_dosage_sensitive": "Dosage sensitivity, graded",
    "hi_no_backup": "Haploinsufficient with no paralogue backup",

    "depmap_mean_score": "Cell-line essentiality (DepMap mean)",
    "depmap_mean_effect": "Mean gene-effect score (DepMap)",
    "depmap_min_effect": "Strongest gene-effect score (DepMap)",
    "depmap_std_effect": "Variability of gene effect across lines (DepMap)",
    "depmap_essential_fraction": "Fraction of cell lines where essential (DepMap)",
    "depmap_n_essential": "Cell lines where essential (DepMap)",
    "depmap_is_pan_essential": "Essential in >90% of cell lines (DepMap)",
    "depmap_is_context_essential": "Essential in 10-90% of cell lines (DepMap)",
    "feat_essential_constrained": "Essential and constrained",
    "feat_central_essential": "Network-central and essential",

    "gtex_tau": "Tissue specificity of expression (GTEx tau)",
    "gtex_max_exp": "Peak tissue expression (GTEx)",

    "feat_is_dominant": "Dominant inheritance",
    "feat_is_recessive": "Recessive inheritance",
    "feat_is_xlinked": "X-linked inheritance",
    "feat_has_mixed_inheritance": "Gene with more than one inheritance mode",
    "is_dominant": "Dominant inheritance",
    "is_recessive": "Recessive inheritance",
    "is_xlinked": "X-linked inheritance",
    "has_mixed_inheritance": "Gene with more than one inheritance mode",
    "n_inheritance_modes": "Number of inheritance modes for the gene",
    "feat_inheritance_not_recorded": "Inheritance not recorded",
    "feat_pli_x_dominant": "pLI in dominant genes",
    "feat_pli_x_recessive": "pLI in recessive genes",
    "feat_dominant_low_constraint": "Dominant gene under weak constraint",
    "feat_recessive_high_constraint": "Recessive gene under strong constraint",

    "paralog_count": "Number of paralogues",
    "n_paralogs": "Number of paralogues",
    "max_paralog_identity_x": "Closest paralogue sequence identity",
    "max_paralog_identity": "Closest paralogue sequence identity",
    "has_close_paralog": "Has a close paralogue",
    "clingen_dosage_curated": "Curated by ClinGen dosage sensitivity",
    "max_paralog_identity_y": "Closest paralogue sequence identity",
    "closest_paralog_identity": "Closest paralogue sequence identity",
    "has_close_paralog_x": "Has a close paralogue",
    "has_close_paralog_y": "Has a close paralogue",
    "is_singleton": "No paralogue in the genome",
    "redundancy_score": "Functional redundancy score",
    "compensation_score": "Paralogue compensation potential",
    "compensation_failure": "No paralogue compensation available",
    "has_functional_backup": "Has a functional paralogue backup",
    "no_backup_flag": "No functional backup",
    "paralog_expression_corr": "Expression correlation with closest paralogue",
    "feat_constrained_no_paralog": "Constrained with no paralogue",
    "rigid_no_backup": "Structurally rigid with no paralogue backup",

    "af_mean_plddt": "AlphaFold confidence, mean pLDDT",
    "af_median_plddt": "AlphaFold confidence, median pLDDT",
    "af_plddt_std": "AlphaFold confidence variability",
    "af_frac_disordered": "Fraction of residues predicted disordered",
    "af_frac_confident": "Fraction of residues confidently folded",
    "af_frac_very_high": "Fraction of residues at very high confidence",
    "af_disorder_ratio": "Disordered-to-ordered residue ratio",
    "af_structural_rigidity": "Predicted structural rigidity",
    "af_protein_length": "Protein length (AlphaFold)",
    "alphafold_plddt": "AlphaFold confidence (pLDDT)",
    "protein_length": "Protein length",
    "mass_kda": "Protein mass (kDa)",
    "domain_count": "Annotated protein domains",
    "n_domains": "Annotated protein domains",
    "transmembrane_segments": "Transmembrane segments",
    "is_membrane_protein": "Membrane protein",
    "has_signal_peptide": "Has a signal peptide",
    "is_nuclear": "Nuclear-localised protein",
    "is_mitochondrial": "Mitochondrial protein",
    "is_secreted": "Secreted protein",

    "ppi_degree": "Protein interaction partners",
    "ppi_betweenness": "Interaction network betweenness",
    "ppi_closeness": "Interaction network closeness",
    "ppi_pagerank": "Interaction network PageRank",
    "string_degree": "STRING interaction partners",
    "feat_log_ppi_degree": "Protein interaction partners, log",
    "feat_betweenness_x_pli": "Network betweenness x pLI",
    "network_degree": "Network degree",
    "network_degree_log": "Network degree, log",
    "network_degree_centrality": "Network degree centrality",
    "network_clustering_coef": "Network clustering coefficient",
    "network_pagerank": "Network PageRank",
    "network_pagerank_log": "Network PageRank, log",
    "network_hub_score": "Network hub score",
    "network_core_number": "Network k-core number",
    "network_avg_neighbor_degree": "Mean degree of network neighbours",
    "network_bottleneck_estimate": "Network bottleneck score",
    "network_is_hub": "Network hub gene",
    "network_is_core": "Network core gene",
    "is_hub": "Network hub gene",
    "feat_hub_constrained": "Network hub under constraint",
    "feat_hub_essential": "Network hub and essential",
    "complex_corum_complex_count": "Protein complexes joined (CORUM)",
    "complex_corum_avg_complex_size": "Mean protein complex size (CORUM)",

    "var_n_pathogenic_clinvar": "Pathogenic ClinVar variants",
    "var_n_lof": "Reported LoF variants",
    "var_n_missense": "Reported missense variants",
    "var_total_variants": "Reported variants, total",
    "var_mean_cadd": "Mean CADD score of reported variants",
    "var_max_cadd": "Maximum CADD score of reported variants",
    "feat_prop_pathogenic": "Proportion of variants that are pathogenic",
    "feat_prop_lof": "Proportion of variants that are LoF",
    "feat_density_pathogenic": "Pathogenic variants per coding base",
    "feat_density_lof": "LoF variants per coding base",
    "feat_density_missense": "Missense variants per coding base",
    "feat_ratio_lof_missense": "LoF-to-missense variant ratio",

    "conservation_score": "Cross-species conservation",
    "gene_age": "Evolutionary age of the gene",
    "is_ancient": "Ancient gene family",
    "feat_conserved_constrained": "Conserved and constrained",
    "has_mouse_ortholog": "Mouse orthologue present",
    "has_zebrafish_ortholog": "Zebrafish orthologue present",
    "has_fly_ortholog": "Fly orthologue present",
    "has_worm_ortholog": "Worm orthologue present",
    "has_yeast_ortholog": "Yeast orthologue present",

    "n_transcripts": "Annotated transcripts",
    "n_transcripts_log": "Annotated transcripts, log",
    "has_multiple_isoforms": "Has multiple isoforms",
    "gene_length": "Gene length",
    "gene_length_log": "Gene length, log",

    "mouse_lethal_pheno_max": "Strongest mouse-lethal phenotype match",
    "mouse_lethal_pheno_mean": "Mean mouse-lethal phenotype match",
    "mouse_lethal_pheno_n_hpos_with_data": "Phenotype terms with mouse-lethal data",
    "mouse_subviable_pheno_mean": "Mean mouse-subviable phenotype match",
    "mouse_subviable_pheno_max": "Strongest mouse-subviable phenotype match",
    "mouse_lethal_vs_pheno_group": "Mouse lethal vs phenotype-group contrast",
    "mouse_viability_score": "Mouse knockout viability",
    "mouse_phenotype_count": "Mouse knockout phenotypes recorded",
    "mouse_lethal": "Mouse knockout is lethal",
    "mouse_subviable": "Mouse knockout is subviable",
    "feat_lethal_x_pli": "Mouse-lethal knockout x pLI",
    "feat_lethal_x_essential": "Mouse-lethal knockout x essentiality",
    "feat_viability_x_pli": "Mouse viability x pLI",
    "feat_has_lethal_x_pli": "Has mouse-lethal record x pLI",
}


_FEATURE_SOURCE_PREFIXES = (
    ("feat_annot_biological_processes_", "GO process: ", "GO biological processes annotated"),
    ("feat_annot_molecular_activities_", "GO function: ", "GO molecular functions annotated"),
    ("feat_annot_expression_sites_", "Expressed in: ", "Expression sites annotated"),
    ("feat_annot_interacting_genes_", "Interacts with: ", "Interaction partners annotated"),
    ("feat_annot_pathways_", "Pathway: ", "Pathways annotated"),
)


_FEATURE_LABEL_CACHE: Dict[str, str] = {}


_FEATURE_LABEL_LOADED = {"done": False}


_HPO_ID_NAME_CACHE: Dict[str, str] = {}


_HPO_ID_NAME_LOADED = {"done": False}


_TEXT_STRUCT_COLS = [
    column
    for column in [
        "Pathways", "Biological_Processes", "Molecular_Activities",
        "Interacting_Genes", "Expression_Sites",
    ]
    # /Expression_Sites and Interacting_Genes are excluded
    # (Config.TEXT_EXCLUDED_SOURCE_COLUMNS).
    if column not in getattr(Config, "TEXT_EXCLUDED_SOURCE_COLUMNS", ())
]


SCENARIO_TEXT_SOURCES: Dict[str, Tuple[str, ...]] = {
    "S1_Gene_Level": ("annotation",),
    "S1b_Disease_Context": ("annotation",),
    "S2_Combined": ("annotation",),
    "S3_Phenotype": (),

}


_generalisation_results: Dict[str, Any] = {}


PAIRED_SCENARIO_CONTRASTS: Tuple[Tuple[str, str], ...] = (
    ("S3_Phenotype", "S2_Combined"),
    ("S2_Combined", "S1b_Disease_Context"),
    ("S1b_Disease_Context", "S1_Gene_Level"),
)


PAIRED_CONTRAST_METRICS: Tuple[str, ...] = ("mcc", "qwk")


DISEASE_OVERLAP_MIN_GENES = 10


_CLASS_SHAP_CACHE: Dict[str, Dict[str, Any]] = {}


logger = None
results_collector = None


def _initialise_runtime() -> None:
    global logger, results_collector
    for subdirectory in (
        "", "figures", "figures/diagnostics", "figures/publication",
        "figures/publication/figure_data", "models", "results", "per_disease_explanations",
    ):
        os.makedirs(os.path.join(Config.OUTPUT_DIR, subdirectory), exist_ok=True)
    logger = Logger(os.path.join(Config.OUTPUT_DIR, "pipeline_log.txt"))
    results_collector = ResultsCollector()


def find_col(patterns, cols):
    if not isinstance(patterns, list):
        patterns = [patterns]
    for p in patterns:
        if isinstance(p, (list, tuple)):
            for c in cols:
                if all((str(sub).lower() in str(c).lower() for sub in p)):
                    return c
        else:
            pat = re.compile(f"(^|[^a-zA-Z0-9]){re.escape(str(p).lower())}([^a-zA-Z0-9]|$)")
            for c in cols:
                if pat.search(str(c).lower()):
                    return c
    return None


def safe_hpo_colname(hpo_id_or_name: str) -> str:
    s = str(hpo_id_or_name).strip()
    if re.match("^HP:\\d{7}$", s):
        return f"hpo_{s.replace(':', '_')}"
    digest = hashlib.md5(s.encode("utf-8")).hexdigest()[:10]
    sanitized = re.sub("[^a-zA-Z0-9]", "_", s)[:30].strip("_")
    return f"hpo_{sanitized}_{digest}" if sanitized else f"hpo_{digest}"


def safe_feature_name(prefix: str, raw_value: str, max_len: int = 40) -> str:
    s = re.sub("\\W+", "_", str(raw_value)).strip("_")
    if len(s) > max_len:
        digest = hashlib.md5(str(raw_value).encode("utf-8")).hexdigest()[:6]
        s = f"{s[:max_len]}_{digest}"
    return f"{prefix}_{s}"


def get_git_commit_hash() -> str:
    search_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=search_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
    except FileNotFoundError:
        return "not_available:git_executable_not_found"
    except subprocess.TimeoutExpired:
        return "not_available:git_rev_parse_timed_out"
    except Exception as exc:  # pragma: no cover - defensive
        return f"not_available:git_call_failed:{type(exc).__name__}"

    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip().replace("\n", " ")
        if "not a git repository" in stderr.lower():
            return (
                "not_available:code_is_not_under_version_control "
                f"(no git repository at or above {search_dir})"
            )
        return f"not_available:git_rev_parse_failed:{stderr[:200]}"

    commit = completed.stdout.decode("utf-8", errors="replace").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        return f"not_available:unexpected_git_output:{commit[:80]}"

    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=search_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        if status.returncode == 0 and status.stdout.decode("utf-8", errors="replace").strip():
            return f"{commit}-dirty"
    except Exception:
        pass
    return commit


def is_severity_hpo_column(colname: str) -> bool:
    if not isinstance(colname, str):
        return False
    m = re.search("(HP[_:]\\d{7})", colname)
    if m:
        hpo_id = m.group(1).replace("_", ":")
        if hpo_id in SEVERITY_HPO_IDS:
            return True
    cn = colname.lower().replace("_", " ")
    for kw in SEVERITY_HPO_KEYWORDS:
        kw_norm = kw.lower().replace("_", " ")
        pat = re.compile(f"(^|[^a-z]){re.escape(kw_norm)}([^a-z]|$)")
        if pat.search(cn):
            return True
    return False


def filter_leaky_columns_for_phenotype_scenarios(
    cols: List[str],
    scenario_name: str,
    drop_severity_hpo_terms: Optional[bool] = None,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    if not any((s in scenario_name for s in ("S2", "S3"))):
        return (cols, [])
    drop_severity = (
        Config.DROP_SEVERITY_HPO_TERMS
        if drop_severity_hpo_terms is None
        else bool(drop_severity_hpo_terms)
    )
    kept: List[str] = []
    dropped: List[Tuple[str, str]] = []
    for column in cols:
        if (
            drop_severity
            and (column.startswith("hpo_") or column.startswith("feat_hpo_sys_"))
            and is_severity_hpo_column(column)
        ):
            dropped.append((column, "severity_hpo_blocklist"))
            continue
        if column.startswith("feat_hpo_pca_"):
            dropped.append((column, "hpo_pca_blocklist"))
            continue
        kept.append(column)
    return (kept, dropped)


def load_mechanistic_features() -> pd.DataFrame:
    logger.log("Loading mechanistic features...")
    if not os.path.exists(Config.CORE_FEATURES):
        logger.log(f"ERROR: {Config.CORE_FEATURES} not found", level="ERROR")
        sys.exit(1)
    df = pd.read_csv(Config.CORE_FEATURES)
    logger.log(f"  Loaded {len(df)} genes x {len(df.columns)} columns")
    df = _drop_removed_input_columns(df, "CORE")
    if getattr(Config, "USE_PAIR_LEVEL_INHERITANCE", False):
        rollups = [c for c in Config.CORE_INHERITANCE_ROLLUP_COLUMNS if c in df.columns]
        if rollups:
            df = df.drop(columns=rollups)
            logger.log(
                f"  Removed gene-level inheritance rollups (pair-level inheritance is used): {rollups}"
            )
    return df


def _drop_removed_input_columns(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    listed = getattr(Config, f"REMOVED_INPUT_COLUMNS_{source_name}", {})
    removed = {
        column: reason
        for column, reason in listed.items()
        if column in df.columns
    }
    if removed:
        df = df.drop(columns=list(removed))
        logger.log(f"  {source_name}: removed {len(removed)} input columns: {removed}")
    return df


def load_variant_features() -> pd.DataFrame:
    """Load gene-level variant features while preserving unknown values as NaN."""
    logger.subsection("Loading Variant Features")
    if not os.path.exists(Config.DOMAIN_FEATURES):
        logger.log("  Variant features not found", level="WARNING")
        return pd.DataFrame()
    df = pd.read_csv(Config.DOMAIN_FEATURES)
    if "Gene" in df.columns and "gene_symbol" not in df.columns:
        df = df.rename(columns={"Gene": "gene_symbol"})
    logger.log(f"  Loaded variant features for {len(df)} genes")
    return df


def load_mouse_lethality_features() -> pd.DataFrame:
    """Load mouse lethality features without equating missing records with zero."""
    logger.subsection("Loading Mouse Lethality Features")
    if not os.path.exists(Config.MOUSE_LETHALITY_FEATURES):
        logger.log("  Mouse lethality features file not found", level="WARNING")
        return pd.DataFrame()
    df = pd.read_csv(Config.MOUSE_LETHALITY_FEATURES)
    for col in [c for c in df.columns if c != "gene_symbol"]:
        df[col] = df[col].replace(
            {"True": 1.0, "False": 0.0, True: 1.0, False: 0.0}
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.log(
        f"  Loaded mouse lethality features for {len(df)} genes "
        f"({len([c for c in df.columns if c != 'gene_symbol'])} columns)"
    )
    return df


def load_mouse_human_phenotype_features() -> pd.DataFrame:
    """Load cross-species phenotype features while retaining genuine missingness."""
    logger.subsection("Loading Mouse-Human Phenotype Features")
    if not os.path.exists(Config.MOUSE_HUMAN_PHENOTYPE_FEATURES):
        logger.log("  Mouse-human phenotype features file not found", level="WARNING")
        return pd.DataFrame()
    df = pd.read_csv(Config.MOUSE_HUMAN_PHENOTYPE_FEATURES)
    for col in [c for c in df.columns if c != "gene_symbol"]:
        df[col] = df[col].replace({"True": 1.0, "False": 0.0, True: 1.0, False: 0.0})
        df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.log(f"  Loaded mouse-human phenotype features for {len(df)} genes")
    return df


def load_external_features() -> Tuple[pd.DataFrame, List[str]]:
    if not os.path.exists(Config.EXTERNAL_FEATURES):
        logger.log("  External features file not found", level="WARNING")
        return (pd.DataFrame(), [])
    logger.subsection("Loading External Mechanistic Features")
    ext_df = pd.read_csv(Config.EXTERNAL_FEATURES)
    logger.log(f"  External features file: {Config.EXTERNAL_FEATURES}")
    ext_df = _drop_removed_input_columns(ext_df, "EXTERNAL")
    drop_text = [c for c in ext_df.columns if ext_df[c].dtype == "object" and c != "gene_symbol"]
    if drop_text:
        logger.log(f"  Dropping text columns: {drop_text}")
        ext_df = ext_df.drop(columns=drop_text)
    ext_cols = [c for c in ext_df.columns if c != "gene_symbol"]
    logger.log(f"  External features: {len(ext_cols)} columns")
    return (ext_df, ext_cols)


def load_new_gene_features() -> Tuple[pd.DataFrame, List[str]]:
    """Load the published gene-level constraint/expression block."""
    if not os.path.exists(Config.NEW_GENE_FEATURES):
        logger.log(
            "  New gene-level features file not found, continuing without it",
            level="WARNING",
        )
        return (pd.DataFrame(), [])
    logger.subsection("Loading New Gene-Level Features (s_het / HPA / BrainSpan / DOMINO / GDI)")
    new_df = pd.read_csv(Config.NEW_GENE_FEATURES)
    # hgnc_id is only the join key used when the file was built; it is an
    # identifier, not a feature, and must never reach a model.
    identifier_columns = [c for c in ("hgnc_id",) if c in new_df.columns]
    if identifier_columns:
        logger.log(f"  Dropping identifier columns: {identifier_columns}")
        new_df = new_df.drop(columns=identifier_columns)
    drop_text = [
        c for c in new_df.columns
        if new_df[c].dtype == "object" and c != "gene_symbol"
    ]
    if drop_text:
        logger.log(f"  Dropping text columns: {drop_text}")
        new_df = new_df.drop(columns=drop_text)
    new_df = new_df.drop_duplicates(subset="gene_symbol")
    new_cols = [c for c in new_df.columns if c != "gene_symbol"]
    logger.log(f"  New gene-level features: {len(new_cols)} columns for {len(new_df)} genes")
    return (new_df, new_cols)


def canonicalize_hpo_token(
    token: Any, terms: Dict[str, Dict], alt_to_id: Dict[str, str], name_to_id: Dict[str, str]
) -> Optional[str]:
    value = str(token).strip()
    if not value:
        return None
    match = re.search("HP:\\d{7}", value)
    if match:
        hpo_id = match.group(0)
        return alt_to_id.get(hpo_id, hpo_id)
    return name_to_id.get(value.lower())


def expand_hpo_profile(
    raw_terms: Sequence[Any],
    terms: Dict[str, Dict],
    alt_to_id: Dict[str, str],
    name_to_id: Dict[str, str],
    max_hops: int,
    excluded_ids: set,
) -> Tuple[set, set, set]:
    direct_ids: set = set()
    unresolved: set = set()
    raw_by_id: Dict[str, str] = {}
    for raw in raw_terms or []:
        canonical = canonicalize_hpo_token(raw, terms, alt_to_id, name_to_id)
        if canonical:
            direct_ids.add(canonical)
            raw_by_id[canonical] = str(raw)
        else:
            unresolved.add(str(raw).strip())
    ancestors: set = set()
    if not Config.HPO_ANCESTOR_EXPANSION_ENABLED or max_hops <= 0:
        return (direct_ids, ancestors, unresolved)
    for direct_id in direct_ids:
        if is_severity_hpo_column(safe_hpo_colname(direct_id)):
            continue
        frontier = {direct_id}
        visited = {direct_id}
        for _ in range(max_hops):
            next_frontier: set = set()
            for term_id in frontier:
                for parent in terms.get(term_id, {}).get("parents", []):
                    parent = alt_to_id.get(parent, parent)
                    if parent in visited or parent in excluded_ids:
                        continue
                    visited.add(parent)
                    next_frontier.add(parent)
                    ancestors.add(parent)
            frontier = next_frontier
            if not frontier:
                break
    ancestors.difference_update(direct_ids)
    return (direct_ids, ancestors, unresolved)


def build_mondo_features(
    disease_df: pd.DataFrame,
    mondo_terms: Dict,
) -> Tuple[pd.DataFrame, List[str]]:
    logger.log("  Building MONDO disease class features...")

    if not mondo_terms:
        return disease_df, []

    depths = []
    class_lists = []
    known_flags = []

    for _, row in disease_df.iterrows():
        mid = row["mondo_id"]
        mid_clean = (
            str(mid).replace("_", ":").strip()
            if pd.notna(mid)
            else ""
        )

        known = mid_clean in mondo_terms
        known_flags.append(known)

        if known:
            depths.append(float(mondo_terms[mid_clean]["depth"]))
            class_lists.append(
                list(
                    mondo_terms[mid_clean].get(
                        "disease_classes",
                        [],
                    )
                )
            )
        else:
            depths.append(np.nan)
            class_lists.append([])

    all_classes = sorted(
        {
            class_name
            for classes in class_lists
            for class_name in classes
        }
    )

    mondo_data = {
        "has_mondo_record": np.asarray(
            known_flags,
            dtype=np.uint8,
        ),
        "feat_mondo_depth": np.asarray(
            depths,
            dtype=float,
        ),
    }

    for class_name in all_classes:
        column = f"feat_mondo_{class_name[:40]}"

        mondo_data[column] = np.asarray(
            [
                (
                    1.0 if class_name in classes else 0.0
                )
                if known
                else np.nan
                for classes, known in zip(
                    class_lists,
                    known_flags,
                )
            ],
            dtype=float,
        )

    mondo_data["feat_mondo_n_classes"] = np.asarray(
        [
            float(len(classes)) if known else np.nan
            for classes, known in zip(
                class_lists,
                known_flags,
            )
        ],
        dtype=float,
    )

    mondo_frame = pd.DataFrame(
        mondo_data,
        index=disease_df.index,
    )

    disease_df = pd.concat(
        [disease_df, mondo_frame],
        axis=1,
    ).copy()

    new_cols = list(mondo_frame.columns)

    logger.log(f"  Created {len(new_cols)} MONDO features")
    return disease_df, new_cols


def build_pipe_text_features(
    disease_df: pd.DataFrame,
    source_df: pd.DataFrame,
    source_label: str,
    n_top_items: int = 20,
) -> Tuple[pd.DataFrame, List[str]]:
    struct_cols = [
        "Pathways",
        "Biological_Processes",
        "Molecular_Activities",
        "Interacting_Genes",
        "Expression_Sites",
    ]
    new_cols: List[str] = []
    if "gene_symbol" not in source_df.columns:
        return disease_df, []

    for column in struct_cols:
        if column not in source_df.columns:
            continue

        def extract_items(series: pd.Series) -> List[str]:
            items: set = set()
            for value in series.dropna():
                items.update(_tokenise_pipe_value(value))
            return list(items)

        gene_struct = source_df.groupby("gene_symbol")[column].apply(extract_items).reset_index()
        count_column = f"feat_{source_label}_{column.lower()}_count"
        availability_column = safe_feature_name(
            f"has_{source_label}", f"{column}_record", max_len=40
        )
        gene_struct[count_column] = gene_struct[column].apply(len).astype(float)
        gene_struct[availability_column] = 1.0
        new_cols.extend([count_column, availability_column])

        all_items: List[str] = []
        for items in gene_struct[column]:
            all_items.extend(items)
        top_items = [item for item, _ in Counter(all_items).most_common(n_top_items)]
        binary_columns: List[str] = []
        for item in top_items:
            binary_column = safe_feature_name(
                f"feat_{source_label}_{column.lower()}", item, max_len=30
            )
            gene_struct[binary_column] = gene_struct[column].apply(
                lambda values, token=item: 1.0 if token in values else 0.0
            )
            binary_columns.append(binary_column)
            new_cols.append(binary_column)

        merge_columns = [
            "gene_symbol", count_column, availability_column, *binary_columns
        ]
        disease_df = disease_df.merge(
            gene_struct[merge_columns], on="gene_symbol", how="left", validate="m:1"
        )
        # Binary token values are zero only when a source record is present and the
        # token is absent. Source-unavailable rows remain NaN during source assembly.
        present = disease_df[availability_column].eq(1.0)
        for binary_column in binary_columns:
            disease_df.loc[present & disease_df[binary_column].isna(), binary_column] = 0.0

    logger.log(f"  Built {len(new_cols)} {source_label}-namespaced text features")
    return disease_df, new_cols


def _add_pair_level_inheritance(disease_df: pd.DataFrame, moi_column: str) -> pd.DataFrame:
    dominant = set(Config.MOI_DOMINANT)
    recessive = set(Config.MOI_RECESSIVE)
    xlinked = set(Config.MOI_XLINKED)
    unknown = {value.lower() for value in Config.MOI_UNKNOWN}
    rows = []
    for modes in disease_df[moi_column]:
        known = sorted({str(m).strip() for m in (modes or []) if str(m).strip().lower() not in unknown})
        if not known:
            rows.append((np.nan, np.nan, np.nan, np.nan, np.nan, ""))
            continue
        groups = set()
        for mode in known:
            if mode in dominant:
                groups.add("dominant")
            elif mode in recessive:
                groups.add("recessive")
            elif mode in xlinked:
                groups.add("xlinked")
            else:
                groups.add("other")
        rows.append((
            float("dominant" in groups),
            float("recessive" in groups),
            float("xlinked" in groups),
            float(len(groups) > 1),
            float(len(groups)),
            "|".join(known),
        ))
    flags = pd.DataFrame(
        rows,
        columns=["is_dominant", "is_recessive", "is_xlinked", "has_mixed_inheritance",
                 "n_inheritance_modes", "all_inheritance_modes"],
        index=disease_df.index,
    )
    disease_df = pd.concat([disease_df, flags], axis=1)
    n_unknown = int(flags["is_dominant"].isna().sum())
    logger.log(
        f"  Pair-level inheritance from '{moi_column}': dominant={int(flags['is_dominant'].sum())}, "
        f"recessive={int(flags['is_recessive'].sum())}, X-linked={int(flags['is_xlinked'].sum())}, "
        f"mixed={int(flags['has_mixed_inheritance'].sum())}, unknown (NaN)={n_unknown} "
        f"of {len(flags)} pairs"
    )
    return disease_df


def build_gene_disease_target(primary_data_path: str) -> pd.DataFrame:
    logger.subsection("Building gene-disease pair targets")
    df = pd.read_csv(primary_data_path, low_memory=False)
    col_map = {
        "Gene Name": "gene_symbol",
        "MONDO_ID": "mondo_id",
        "Severity": "target_severity",
        "HPO_Term": "hpo_term",
        "Phenotypes_HPO": "hpo_term",
        "Parent Term": "parent_term",
        "Disease": "disease_name",
    }
    df = df.rename(columns={key: value for key, value in col_map.items() if key in df.columns})
    required = ["gene_symbol", "mondo_id", "target_severity"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Primary data is missing required columns: {missing}")
    df = df.dropna(subset=required).copy()
    if getattr(Config, "APPLY_REBUILT_GENE_DISEASE_MAPPING", False):
        logger.log("  Applying HPO-verified gene-disease mapping")
        mapping_path = _resolve_gene_disease_mapping_path(
            getattr(Config, "REBUILT_GENE_DISEASE_MAPPING", None),
            os.path.dirname(os.path.abspath(primary_data_path)),
            Config.BASE_DIR,
            os.path.dirname(os.path.abspath(__file__)),
        )
        df = _apply_rebuilt_gene_disease_mapping(
            df,
            gene_col="gene_symbol",
            mondo_col="mondo_id",
            mapping_path=mapping_path,
            log=logger.log,
            audit_dir=os.path.join(Config.OUTPUT_DIR, "results"),
            audit_name="gene_disease_mapping_filter_audit.csv",
            include_added=getattr(Config, "INCLUDE_HPO_ADDED_PAIRS", False),
        )
    df["target_severity"] = df["target_severity"].astype(str).str.strip()
    valid_mask = df["target_severity"].isin(Config.TARGET_MAP)
    n_unsupported = int((~valid_mask).sum())
    if n_unsupported:
        logger.log(
            f"  Excluding {n_unsupported} rows with labels outside the four-class target",
            level="WARNING",
        )
    df = df.loc[valid_mask].copy()
    df["severity_numeric"] = df["target_severity"].map(Config.TARGET_MAP).astype(int)
    if df.empty:
        raise ValueError("No rows remain after applying the four-class severity target map.")
    conflict_audit = (
        df.groupby(["gene_symbol", "mondo_id"])
        .agg(
            n_source_rows=("severity_numeric", "size"),
            n_unique_numeric_labels=("severity_numeric", "nunique"),
            numeric_labels=(
                "severity_numeric",
                lambda values: "|".join(map(str, sorted(set(values)))),
            ),
            severity_labels=(
                "target_severity",
                lambda values: "|".join(sorted(set(map(str, values)))),
            ),
        )
        .reset_index()
    )
    conflict_audit["has_numeric_conflict"] = conflict_audit["n_unique_numeric_labels"] > 1
    conflict_path = os.path.join(Config.OUTPUT_DIR, "results", "severity_label_conflicts.csv")
    conflict_audit.to_csv(conflict_path, index=False)
    n_conflict = int(conflict_audit["has_numeric_conflict"].sum())
    logger.log(
        f'  Severity-label conflict check: {n_conflict} conflicting gene-disease pairs; wrote {conflict_path}'
    )
    if n_conflict:
        conflict_keys = set(
            map(
                tuple,
                conflict_audit.loc[
                    conflict_audit["has_numeric_conflict"], ["gene_symbol", "mondo_id"]
                ].values,
            )
        )
        before = len(df)
        df = df.loc[
            ~df[["gene_symbol", "mondo_id"]].apply(tuple, axis=1).isin(conflict_keys)
        ].copy()
        logger.log(
            f"  Excluding {before - len(df)} source rows from {n_conflict} gene-disease pairs with conflicting severity labels"
        )
    if df.empty:
        raise ValueError("No labelled rows remain after excluding conflicting severity labels.")

    def get_list(series):
        values = set()
        for value in series.dropna():
            values.update((item.strip() for item in str(value).split("|") if item.strip()))
        return list(values)

    def first_mode(series):
        mode = series.mode(dropna=True)
        return mode.iloc[0] if len(mode) else series.iloc[0]

    agg_dict = {"target_severity": first_mode, "severity_numeric": first_mode}
    if "disease_name" in df.columns:
        agg_dict["disease_name"] = "first"
    if "hpo_term" in df.columns:
        agg_dict["hpo_term"] = get_list
    if "parent_term" in df.columns:
        agg_dict["parent_term"] = get_list
    moi_column = getattr(Config, "PAIR_INHERITANCE_COLUMN", "moi_title")
    use_pair_inheritance = getattr(Config, "USE_PAIR_LEVEL_INHERITANCE", False)
    if use_pair_inheritance:
        if moi_column not in df.columns:
            raise ValueError(
                f"USE_PAIR_LEVEL_INHERITANCE is on but the primary data has no '{moi_column}' column."
            )
        agg_dict[moi_column] = get_list
    disease_df = df.groupby(["gene_symbol", "mondo_id"]).agg(agg_dict).reset_index()
    disease_df = disease_df.rename(columns={"severity_numeric": "Multiclass_Target"})
    disease_df["Multiclass_Target"] = disease_df["Multiclass_Target"].astype(int)
    if use_pair_inheritance:
        disease_df = _add_pair_level_inheritance(disease_df, moi_column)
        disease_df = disease_df.drop(columns=[moi_column])
    logger.log("  Building HPO binary indicators with ontology ancestors")
    hpo_terms, hpo_alt_to_id, hpo_name_to_id = parse_hpo_obo(Config.HPO_OBO_PATH)
    excluded_hpo_roots = set(Config.HPO_ANCESTOR_EXCLUDED_IDS)
    feature_rows = []
    expansion_audit_rows = []
    for _, row in disease_df.iterrows():
        feature_row = {"gene_symbol": row["gene_symbol"], "mondo_id": row["mondo_id"]}
        raw_hpo = row["hpo_term"] if "hpo_term" in row and isinstance(row["hpo_term"], list) else []
        direct_ids, ancestor_ids, unresolved_terms = expand_hpo_profile(
            raw_hpo,
            hpo_terms,
            hpo_alt_to_id,
            hpo_name_to_id,
            max_hops=Config.HPO_ANCESTOR_MAX_HOPS,
            excluded_ids=excluded_hpo_roots,
        )
        for hpo_id in sorted(direct_ids | ancestor_ids):
            feature_row[safe_hpo_colname(hpo_id)] = 1
        for raw_term in sorted(unresolved_terms):
            if raw_term:
                feature_row[safe_hpo_colname(raw_term)] = 1
        if "parent_term" in row and isinstance(row["parent_term"], list):
            for parent_term in row["parent_term"]:
                column = safe_feature_name("feat_hpo_sys", str(parent_term).lower(), max_len=50)
                feature_row[column] = 1
        feature_rows.append(feature_row)
        expansion_audit_rows.append(
            {
                "gene_symbol": row["gene_symbol"],
                "mondo_id": row["mondo_id"],
                "n_raw_hpo_terms": len(raw_hpo),
                "n_direct_canonical_hpo": len(direct_ids),
                "n_added_hpo_ancestors": len(ancestor_ids),
                "n_unresolved_hpo_terms_preserved": len(unresolved_terms),
            }
        )
    expansion_audit_path = os.path.join(
        Config.OUTPUT_DIR, "results", "hpo_ancestor_expansion_audit.csv"
    )
    pd.DataFrame(expansion_audit_rows).to_csv(expansion_audit_path, index=False)
    logger.log(
        f"  HPO ancestor expansion: max_hops={Config.HPO_ANCESTOR_MAX_HOPS}; wrote {expansion_audit_path}"
    )
    features_df = pd.DataFrame(feature_rows).fillna(0)
    disease_df = disease_df.merge(features_df, on=["gene_symbol", "mondo_id"], how="left", validate="1:1")
    disease_df, _ = build_pipe_text_features(disease_df, df, source_label="primary")
    label_summary = (
        disease_df["Multiclass_Target"]
        .value_counts()
        .sort_index()
        .rename_axis("severity_class")
        .reset_index(name="n_gene_disease_pairs")
    )
    label_summary.to_csv(
        os.path.join(Config.OUTPUT_DIR, "results", "label_curation_summary.csv"), index=False
    )
    logger.log(f"  Gene-disease pair dataset: {len(disease_df)} pairs")
    logger.log(
        f"  Class distribution: {np.bincount(disease_df['Multiclass_Target'].values, minlength=4).tolist()}"
    )
    return disease_df


def enrich_with_annotation_dataset(disease_df: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists(Config.ANNOTATION_DATA):
        logger.log(f"  ANNOTATION dataset not found", level="WARNING")
        return disease_df
    logger.subsection("Enriching with ANNOTATION biological text features")
    annotation_df = pd.read_csv(Config.ANNOTATION_DATA, low_memory=False)
    annotation_df = annotation_df.rename(columns={"Gene Name": "gene_symbol", "MONDO_ID": "mondo_id"})
    # ANNOTATION annotations are mechanistic gene annotations. Collapse repeated
    # gene-disease rows to a single union profile per gene before any feature
    # construction so disease identity cannot alter the S1 representation.
    annotation_df = _collapse_annotation_to_gene_level(annotation_df, write_audit=True)
    disease_df, _ = build_pipe_text_features(disease_df, annotation_df, source_label="annotation")
    return disease_df


def engineer_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    logger.subsection("Engineering ratio and interaction features")
    new_features = []
    eng_df = df.copy()
    pli_col = find_col(["gnomad_pli", "pli"], feature_cols)  # prefer exact gnomad_pLI
    loeuf_col = find_col(["loeuf"], feature_cols)
    missz_col = find_col(["mis_z", "missense_z", "mis_zscore"], feature_cols)
    synz_col = find_col(["syn_z", "synonymous_z"], feature_cols)
    if pli_col and loeuf_col:
        loeuf_safe = eng_df[loeuf_col].clip(lower=0.01)
        eng_df["feat_pli_over_loeuf"] = eng_df[pli_col] / loeuf_safe
        new_features.append("feat_pli_over_loeuf")
    if missz_col and pli_col:
        eng_df["feat_missz_x_pli"] = eng_df[missz_col] * eng_df[pli_col]
        new_features.append("feat_missz_x_pli")
    if missz_col and synz_col:
        # Use a difference: dividing by signed synonymous Z creates spikes near zero.
        eng_df["feat_missz_minus_synz"] = pd.to_numeric(
            eng_df[missz_col], errors="coerce"
        ) - pd.to_numeric(eng_df[synz_col], errors="coerce")
        new_features.append("feat_missz_minus_synz")
    dep_col = find_col(["depmap", "essentiality", "essential"], feature_cols)
    if dep_col and pli_col:
        eng_df["feat_essential_constrained"] = eng_df[dep_col] * eng_df[pli_col]
        new_features.append("feat_essential_constrained")
    if dep_col and loeuf_col:
        eng_df["feat_essential_over_loeuf"] = eng_df[dep_col] / eng_df[loeuf_col].clip(lower=0.01)
        new_features.append("feat_essential_over_loeuf")
    ppi_col = find_col(["ppi_degree", "string_degree", "network_degree", "degree"], feature_cols)
    if ppi_col and pli_col:
        eng_df["feat_hub_constrained"] = eng_df[ppi_col] * eng_df[pli_col]
        new_features.append("feat_hub_constrained")
    if ppi_col and dep_col:
        eng_df["feat_hub_essential"] = eng_df[ppi_col] * eng_df[dep_col]
        new_features.append("feat_hub_essential")
    betw_col = find_col(["betweenness", "between"], feature_cols)
    if betw_col and pli_col:
        eng_df["feat_betweenness_x_pli"] = eng_df[betw_col] * eng_df[pli_col]
        new_features.append("feat_betweenness_x_pli")
    cons_col = find_col(["conservation", "phastcons", "phylop", "gerp"], feature_cols)
    if cons_col and pli_col:
        eng_df["feat_conserved_constrained"] = eng_df[cons_col] * eng_df[pli_col]
        new_features.append("feat_conserved_constrained")
    para_col = find_col(["paralog_count", "n_paralogs", "paralog"], feature_cols)
    # Match 'paralog_count' exactly to avoid identity and close-paralogue columns.
    expr_col = find_col(
        ["expression_breadth", "n_tissues", "tissue_breadth", "gtex_n_tissues", "ubiquity"],
        feature_cols,
    )
    if expr_col and pli_col:
        eng_df["feat_broad_expression_constrained"] = eng_df[expr_col] * eng_df[pli_col]
        new_features.append("feat_broad_expression_constrained")
    rigidity_col = find_col(
        ["af_structural_rigidity", "structural_rigidity", ("af_", "rigid"), ("alphafold", "rigid")],
        feature_cols,
    )
    disorder_col = find_col(
        ["af_frac_disordered", "frac_disordered", ("af_", "disorder"), ("alphafold", "disorder")],
        feature_cols,
    )
    if rigidity_col and pli_col:
        eng_df["feat_rigid_constrained"] = eng_df[rigidity_col] * eng_df[pli_col]
        new_features.append("feat_rigid_constrained")
    if disorder_col and pli_col:
        eng_df["feat_disorder_x_pli"] = eng_df[disorder_col] * eng_df[pli_col]
        new_features.append("feat_disorder_x_pli")
    comp_col = find_col(["compensation_score", "compensation_failure"], feature_cols)
    if comp_col and pli_col:
        eng_df["feat_no_comp_constrained"] = (1 - pd.to_numeric(eng_df[comp_col], errors="coerce")) * eng_df[pli_col]
        new_features.append("feat_no_comp_constrained")
    hi_col = find_col(["hi_sensitive", "collins_pHI"], feature_cols)
    backup_col = find_col(["no_backup_flag"], feature_cols)
    if hi_col and backup_col:
        eng_df["feat_hi_no_backup"] = eng_df[hi_col] * eng_df[backup_col]
        new_features.append("feat_hi_no_backup")
    domain_col = find_col(["domain_count", ("domain", "count")], feature_cols)
    ptm_col = find_col(["ptm_count", ("ptm", "count")], feature_cols)
    len_col = find_col(["protein_length", "length", ("aa", "len")], feature_cols)
    if ptm_col and len_col:
        len_safe = eng_df[len_col].clip(lower=1.0)
        eng_df["feat_ptm_density"] = eng_df[ptm_col] / len_safe
        new_features.append("feat_ptm_density")
    if ptm_col and pli_col:
        eng_df["feat_ptm_constrained"] = eng_df[ptm_col] * eng_df[pli_col]
        new_features.append("feat_ptm_constrained")
    if domain_col and pli_col:
        eng_df["feat_domain_constrained"] = eng_df[domain_col] * eng_df[pli_col]
        new_features.append("feat_domain_constrained")
    var_total_col = find_col(["var_total_variants", ("total", "variant")], eng_df.columns)
    clinvar_col = find_col(["var_n_pathogenic_clinvar", ("pathogenic", "clinvar")], eng_df.columns)
    lof_col = find_col(["var_n_lof", ("n_", "lof")], eng_df.columns)
    missense_col = find_col(["var_n_missense", ("n_", "missense")], eng_df.columns)
    # Density features depend only on protein length; proportion features depend only
    # on the total-variant count. Each is gated on the input it actually uses, so a

    if len_col:
        len_safe = eng_df[len_col].clip(lower=1.0)
        if clinvar_col:
            eng_df["feat_density_pathogenic"] = eng_df[clinvar_col] / len_safe
            new_features.append("feat_density_pathogenic")
        if lof_col:
            eng_df["feat_density_lof"] = eng_df[lof_col] / len_safe
            new_features.append("feat_density_lof")
        if missense_col:
            eng_df["feat_density_missense"] = eng_df[missense_col] / len_safe
            new_features.append("feat_density_missense")
    if var_total_col:
        var_total_safe = eng_df[var_total_col].clip(lower=1.0)
        if clinvar_col:
            eng_df["feat_prop_pathogenic"] = eng_df[clinvar_col] / var_total_safe
            new_features.append("feat_prop_pathogenic")
        if lof_col:
            eng_df["feat_prop_lof"] = eng_df[lof_col] / var_total_safe
            new_features.append("feat_prop_lof")
    if lof_col and missense_col:
        missense_safe = eng_df[missense_col].clip(lower=1.0)
        eng_df["feat_ratio_lof_missense"] = eng_df[lof_col] / missense_safe
        new_features.append("feat_ratio_lof_missense")
    if domain_col and len_col:
        len_safe = eng_df[len_col].clip(lower=1.0)
        eng_df["feat_domain_density"] = eng_df[domain_col] / len_safe
        new_features.append("feat_domain_density")
    if pli_col and ppi_col and dep_col:
        eng_df["feat_triple_constraint"] = eng_df[pli_col] * eng_df[ppi_col] * eng_df[dep_col]
        new_features.append("feat_triple_constraint")
    if pli_col and "feat_is_recessive" in eng_df.columns:
        eng_df["feat_recessive_high_constraint"] = eng_df["feat_is_recessive"] * (
            eng_df[pli_col] > PLI_HIGH_CONSTRAINT
        ).astype(float)
        new_features.append("feat_recessive_high_constraint")
    if pli_col and "feat_is_dominant" in eng_df.columns:
        eng_df["feat_dominant_low_constraint"] = eng_df["feat_is_dominant"] * (
            eng_df[pli_col] < PLI_LOW_CONSTRAINT
        ).astype(float)
        new_features.append("feat_dominant_low_constraint")
    if cons_col and disorder_col:
        eng_df["feat_conserved_disordered"] = eng_df[cons_col] * eng_df[disorder_col]
        new_features.append("feat_conserved_disordered")
    if betw_col and dep_col:
        eng_df["feat_central_essential"] = eng_df[betw_col] * eng_df[dep_col]
        new_features.append("feat_central_essential")
    if pli_col and para_col:
        eng_df["feat_constrained_no_paralog"] = eng_df[pli_col] * (eng_df[para_col] == 0).astype(
            float
        )
        new_features.append("feat_constrained_no_paralog")
    if expr_col and dep_col:
        eng_df["feat_broad_expr_essential"] = eng_df[expr_col] * eng_df[dep_col]
        new_features.append("feat_broad_expr_essential")
    if ppi_col:
        eng_df["feat_log_ppi_degree"] = np.log1p(eng_df[ppi_col].clip(lower=0))
        new_features.append("feat_log_ppi_degree")
    if pli_col:
        eng_df["feat_pli_squared"] = eng_df[pli_col] ** 2
        new_features.append("feat_pli_squared")
    if loeuf_col:
        eng_df["feat_loeuf_squared"] = eng_df[loeuf_col] ** 2
        new_features.append("feat_loeuf_squared")
    if missz_col and loeuf_col:
        loeuf_safe = eng_df[loeuf_col].clip(lower=0.01)
        eng_df["feat_missz_over_loeuf"] = eng_df[missz_col] / loeuf_safe
        new_features.append("feat_missz_over_loeuf")
    lethal_col = find_col(["mouse_lethal"], feature_cols)  # exact: 'lethal' must not alias has_lethal_phenotype
    subviable_col = find_col(["mouse_subviable", "subviable"], feature_cols)
    viability_col = find_col(["mouse_viability_score", "viability"], feature_cols)
    has_lethal_col = find_col(["has_lethal_phenotype"], feature_cols)
    mouse_pheno_count_col = find_col(["mouse_phenotype_count", "phenotype_count"], feature_cols)
    if lethal_col and pli_col:
        eng_df["feat_lethal_x_pli"] = pd.to_numeric(eng_df[lethal_col], errors="coerce") * eng_df[pli_col]
        new_features.append("feat_lethal_x_pli")
    if lethal_col and dep_col:
        eng_df["feat_lethal_x_essential"] = pd.to_numeric(eng_df[lethal_col], errors="coerce") * eng_df[dep_col]
        new_features.append("feat_lethal_x_essential")
    if viability_col and pli_col:
        eng_df["feat_viability_x_pli"] = pd.to_numeric(eng_df[viability_col], errors="coerce") * eng_df[pli_col]
        new_features.append("feat_viability_x_pli")
    if has_lethal_col and pli_col:
        eng_df["feat_has_lethal_x_pli"] = pd.to_numeric(eng_df[has_lethal_col], errors="coerce") * eng_df[pli_col]
        new_features.append("feat_has_lethal_x_pli")
    logger.log(f"  Total engineered features: {len(new_features)}")
    return (eng_df, new_features)


def build_inheritance_features(
    full_df: pd.DataFrame, pli_col: Optional[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """Construct inheritance features without converting unknown annotations to zero."""
    inheritance_features: List[str] = []
    inheritance_df = pd.DataFrame(index=full_df.index)
    # a pair has a recorded mode exactly when its pair-level flags are not NaN
    # (_add_pair_level_inheritance gives NaN on all flags together for unknown modes).
    unknown_as_flag = bool(getattr(Config, "INHERITANCE_UNKNOWN_AS_FLAG", False))
    recorded = None
    if unknown_as_flag and "is_dominant" in full_df.columns:
        recorded = pd.to_numeric(full_df["is_dominant"], errors="coerce").notna()
    for feature_name, source_column in {
        "feat_is_dominant": "is_dominant",
        "feat_is_recessive": "is_recessive",
        "feat_is_xlinked": "is_xlinked",
        "feat_has_mixed_inheritance": "has_mixed_inheritance",
    }.items():
        if source_column in full_df.columns:
            values = pd.to_numeric(full_df[source_column], errors="coerce").astype(float)
            if recorded is not None:
                values = values.where(recorded, 0.0)
            inheritance_df[feature_name] = values
        else:
            inheritance_df[feature_name] = np.nan
        inheritance_features.append(feature_name)
    if recorded is not None and getattr(Config, "INHERITANCE_NOT_RECORDED_FLAG", True):
        inheritance_df["feat_inheritance_not_recorded"] = (~recorded).astype(float).to_numpy()
        inheritance_features.append("feat_inheritance_not_recorded")
        logger.log(
            f"  Inheritance: {int((~recorded).sum())} pairs without a recorded mode coded 0 "
            "on every flag and marked by feat_inheritance_not_recorded"
        )
    elif recorded is not None:
        logger.log(
            f"  Inheritance: {int((~recorded).sum())} pairs without a recorded mode coded 0 "
            "on every flag without a separate not-recorded feature"
        )

    if pli_col and pli_col in full_df.columns:
        pli = pd.to_numeric(full_df[pli_col], errors="coerce").to_numpy(dtype=float)
        inheritance_df["feat_pli_x_dominant"] = (
            pli * inheritance_df["feat_is_dominant"].to_numpy(dtype=float)
        )
        inheritance_df["feat_pli_x_recessive"] = (
            pli * inheritance_df["feat_is_recessive"].to_numpy(dtype=float)
        )
        inheritance_features.extend(["feat_pli_x_dominant", "feat_pli_x_recessive"])
    return inheritance_df, inheritance_features


def _validate_gene_level_source(source_df: pd.DataFrame, source_name: str) -> None:
    if "gene_symbol" not in source_df.columns:
        raise ValueError(f"{source_name} is missing the required gene_symbol column.")
    duplicated = source_df["gene_symbol"].notna() & source_df["gene_symbol"].duplicated(keep=False)
    if duplicated.any():
        audit = source_df.loc[duplicated].sort_values("gene_symbol")
        audit_path = os.path.join(
            Config.OUTPUT_DIR, "results", f"duplicate_gene_rows_{source_name}.csv"
        )
        audit.to_csv(audit_path, index=False)
        raise ValueError(
            f"{source_name} contains duplicate gene_symbol rows. "
            f"This would duplicate gene-disease observations; see {audit_path}."
        )


def _merge_gene_level_features(
    base_df: pd.DataFrame,
    source_df: pd.DataFrame,
    source_name: str,
    how: str = "left",
    add_availability: bool = True,
) -> Tuple[pd.DataFrame, Optional[str]]:
    _validate_gene_level_source(source_df, source_name)
    merged_base = base_df.copy()
    availability_column: Optional[str] = None
    if add_availability and getattr(Config, "ADD_SOURCE_AVAILABILITY_INDICATORS", True):
        availability_column = safe_feature_name("has", f"{source_name}_record", max_len=45)
        observed_genes = set(source_df["gene_symbol"].dropna().astype(str))
        merged_base[availability_column] = (
            merged_base["gene_symbol"].astype(str).isin(observed_genes).astype(np.uint8)
        )
    merged = merged_base.merge(
        source_df, on="gene_symbol", how=how, validate="m:1"
    )
    return merged, availability_column


def _missing_value_semantics_for_family(family: str) -> str:
    if family in {"hpo_binary", "organ_system"}:
        return "structural_zero_means_term_absent"
    if family == "missingness_indicator":
        return "explicit_source_or_column_availability_indicator"
    if family in {"primary_text", "annotation_text"}:
        return "fold_built_tokens_with_explicit_source_availability"
    return "unknown_preserved_as_nan_then_training_fold_median_imputation"


def parse_hpo_obo(obo_path: str):
    if not os.path.exists(obo_path) and not Config.ALLOW_ONTOLOGY_DOWNLOAD:
        raise FileNotFoundError(
            f"Pinned HPO ontology is required at {obo_path}. Automatic download is disabled "
            "for manuscript reproducibility."
        )
    logger.log(f"  Parsing HPO ontology from {obo_path}...")
    if not os.path.exists(obo_path):
        logger.log("  HPO OBO not found, attempting download...", level="WARNING")
        try:
            import urllib.request

            urllib.request.urlretrieve(Config.HPO_OBO_URL, obo_path)
        except Exception as e:
            logger.log(f"  Could not download HPO ontology: {e}", level="WARNING")
            return ({}, {}, {})
    records: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    def commit() -> None:
        nonlocal current
        if current and current.get("id"):
            records.append(current)
        current = None

    with open(obo_path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line == "[Term]":
                commit()
                current = {
                    "id": None,
                    "name": "",
                    "parents": [],
                    "alt_ids": [],
                    "obsolete": False,
                    "replaced_by": None,
                }
            elif line.startswith("["):
                commit()
            elif current is not None:
                if line.startswith("id: HP:"):
                    current["id"] = line[4:].strip()
                elif line.startswith("name: "):
                    current["name"] = line[6:].strip()
                elif line.startswith("is_a: HP:"):
                    current["parents"].append(line[6:].split("!")[0].strip())
                elif line.startswith("alt_id: HP:"):
                    current["alt_ids"].append(line[8:].strip())
                elif line.startswith("is_obsolete: true"):
                    current["obsolete"] = True
                elif line.startswith("replaced_by: HP:"):
                    current["replaced_by"] = line[13:].strip()
        commit()
    terms: Dict[str, Dict] = {}
    alt_to_id: Dict[str, str] = {}
    name_candidates: Dict[str, set] = defaultdict(set)
    for rec in records:
        tid = rec["id"]
        replacement = rec.get("replaced_by")
        if rec.get("obsolete"):
            if replacement:
                alt_to_id[tid] = replacement
                for alt in rec.get("alt_ids", []):
                    alt_to_id[alt] = replacement
            continue
        terms[tid] = {"name": rec.get("name", ""), "parents": list(rec.get("parents", []))}
        for alt in rec.get("alt_ids", []):
            alt_to_id[alt] = tid
        if rec.get("name"):
            name_candidates[str(rec["name"]).strip().lower()].add(tid)
    name_to_id = {name: next(iter(ids)) for name, ids in name_candidates.items() if len(ids) == 1}
    logger.log(f"  Parsed {len(terms)} current HPO terms, {len(alt_to_id)} alternate/replaced IDs")
    return (terms, alt_to_id, name_to_id)


def parse_mondo_obo(obo_path: str):
    if not os.path.exists(obo_path) and not Config.ALLOW_ONTOLOGY_DOWNLOAD:
        raise FileNotFoundError(
            f"Pinned MONDO ontology is required at {obo_path}. Automatic download is disabled "
            "for manuscript reproducibility."
        )
    logger.log(f"  Parsing MONDO ontology from {obo_path}...")
    if not os.path.exists(obo_path):
        logger.log(f"  MONDO OBO not found, attempting download...", level="WARNING")
        try:
            import urllib.request

            urllib.request.urlretrieve("http://purl.obolibrary.org/obo/mondo.obo", obo_path)
        except Exception as e:
            logger.log(f"  Could not download: {e}", level="WARNING")
            return {}
    terms: Dict[str, Dict] = {}
    current_id = None
    current_name = None
    current_parents: List[str] = []
    current_is_obsolete = False
    in_term = False
    with open(obo_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                if current_id and (not current_is_obsolete):
                    terms[current_id] = {"name": current_name or "", "parents": current_parents}
                current_id = None
                current_name = None
                current_parents = []
                current_is_obsolete = False
                in_term = True
            elif line == "" or line.startswith("["):
                if current_id and (not current_is_obsolete) and in_term:
                    terms[current_id] = {"name": current_name or "", "parents": current_parents}
                current_id = None
                current_name = None
                current_parents = []
                current_is_obsolete = False
                in_term = line.startswith("[Term]")
            elif in_term:
                if line.startswith("id: "):
                    current_id = line[4:].strip()
                elif line.startswith("name: "):
                    current_name = line[6:].strip()
                elif line.startswith("is_a: "):
                    # Strip the name and axiom annotation so only the parent ID remains.
                    parent_id = line[6:].split("!")[0].split("{")[0].strip()
                    current_parents.append(parent_id)
                elif line.startswith("is_obsolete: true"):
                    current_is_obsolete = True
        if current_id and (not current_is_obsolete):
            terms[current_id] = {"name": current_name or "", "parents": current_parents}
    logger.log(f"  Parsed {len(terms)} MONDO terms")
    depth_cache: Dict[str, int] = {}

    def get_depth(tid: str, visiting: Optional[set] = None) -> int:
        if tid in depth_cache:
            return depth_cache[tid]
        if visiting is None:
            visiting = set()
        if tid in visiting:
            depth_cache[tid] = 0
            return 0
        if tid not in terms or not terms[tid]["parents"]:
            depth_cache[tid] = 0
            return 0
        visiting = visiting | {tid}
        valid_parents = [p for p in terms[tid]["parents"] if p in terms]
        if not valid_parents:
            depth_cache[tid] = 0
            return 0
        d = 1 + max((get_depth(p, visiting) for p in valid_parents))
        depth_cache[tid] = d
        return d

    for tid in terms:
        terms[tid]["depth"] = get_depth(tid)
    # disease classes are the direct subclasses of MONDO_CLASS_ROOT in the pinned
    # release, named from their MONDO labels. Nothing is chosen by hand.
    class_root = getattr(Config, "MONDO_CLASS_ROOT", "MONDO:0700096")
    if class_root not in terms:
        raise ValueError(f"MONDO class root {class_root} not found (or obsolete) in {obo_path}.")
    disease_class_roots: Dict[str, str] = {}
    used_labels: set = set()
    for tid in sorted(terms):
        if not tid.startswith("MONDO:") or class_root not in terms[tid]["parents"]:
            continue
        label = re.sub(r"[^a-z0-9]+", "_", terms[tid]["name"].lower()).strip("_")[:40]
        label = label or tid.replace(":", "_")
        if label in used_labels:
            label = f"{label[:31]}_{tid.split(':')[1]}"
        used_labels.add(label)
        disease_class_roots[tid] = label
    if not disease_class_roots:
        raise ValueError(f"No direct subclasses of {class_root} found in {obo_path}.")
    logger.log(
        f"  MONDO disease classes: {len(disease_class_roots)} direct subclasses of "
        f"{class_root} ({terms[class_root]['name']})"
    )

    def get_ancestors(tid: str, visited: Optional[set] = None) -> set:
        if visited is None:
            visited = set()
        if tid in visited or tid not in terms:
            return set()
        visited.add(tid)
        ancestors = {tid}
        for p in terms[tid]["parents"]:
            ancestors.update(get_ancestors(p, visited))
        return ancestors

    for tid in terms:
        ancestors = get_ancestors(tid)
        classes = []
        for class_id, class_name in disease_class_roots.items():
            if class_id in ancestors and class_id != tid:
                classes.append(class_name)
        terms[tid]["disease_classes"] = classes
    return terms


def _collapse_annotation_to_gene_level(
    source: pd.DataFrame,
    write_audit: bool = False,
) -> pd.DataFrame:
    if "gene_symbol" not in source.columns:
        raise ValueError("ANNOTATION data must contain a gene_symbol column.")
    frame = source.copy()
    frame["gene_symbol"] = frame["gene_symbol"].map(_normalise_gene_key)
    text_columns = [column for column in _TEXT_STRUCT_COLS if column in frame.columns]
    if not text_columns:
        logger.log("  ANNOTATION data contains no configured functional text columns", level="WARNING")
        return frame[["gene_symbol"]].drop_duplicates().reset_index(drop=True)

    collapsed_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    for gene, group in frame.groupby("gene_symbol", sort=True):
        collapsed: Dict[str, Any] = {"gene_symbol": gene}
        for column in text_columns:
            token_sets = [
                frozenset(_tokenise_pipe_value(value))
                for value in group[column].dropna()
            ]
            union_tokens: set = set()
            for token_set in token_sets:
                union_tokens.update(token_set)
            collapsed[column] = "|".join(sorted(union_tokens))
            audit_rows.append({
                "gene_symbol": gene,
                "source_column": column,
                "n_source_rows": int(len(group)),
                "n_distinct_nonempty_profiles": int(len({tokens for tokens in token_sets if tokens})),
                "n_union_tokens": int(len(union_tokens)),
                "aggregation": "gene_level_union",
            })
        collapsed_rows.append(collapsed)

    collapsed_frame = pd.DataFrame(collapsed_rows)
    if collapsed_frame["gene_symbol"].duplicated().any():
        raise RuntimeError("ANNOTATION gene-level collapse produced duplicate genes.")
    if "mondo_id" in collapsed_frame.columns:
        raise RuntimeError("Disease identifiers must not survive ANNOTATION gene-level collapse.")

    if write_audit:
        audit_path = os.path.join(
            Config.OUTPUT_DIR, "results", "annotation_gene_level_invariance_audit.csv"
        )
        pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
        n_variable = sum(row["n_distinct_nonempty_profiles"] > 1 for row in audit_rows)
        logger.log(
            f"  Collapsed ANNOTATION annotations to {len(collapsed_frame)} gene-level profiles; "
            f"{n_variable} gene/column groups required token union; wrote {audit_path}"
        )
    return collapsed_frame


def _is_placeholder_token(item: str) -> bool:
    return item.strip().lower() in set(getattr(Config, "TEXT_PLACEHOLDER_TOKENS", ()))


def _tokenise_pipe_value(value: Any) -> set:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    # placeholders such as "Unknown" mean no annotation, not a real token.
    return {
        item.strip()
        for item in str(value).split("|")
        if item.strip() and not _is_placeholder_token(item)
    }


def _normalise_gene_key(gene: Any) -> str:
    return str(gene).strip().upper()


def _normalise_mondo_key(mondo: Any) -> str:
    value = str(mondo).strip().upper().replace("_", ":")
    match = re.search(r"MONDO:?([0-9]+)", value)
    if match:
        return f"MONDO:{int(match.group(1)):07d}"
    return value


def _text_pair_key(gene: Any, mondo: Any) -> str:
    return f"P|{_normalise_gene_key(gene)}|{_normalise_mondo_key(mondo)}"


def _text_gene_key(gene: Any) -> str:
    return f"G|{_normalise_gene_key(gene)}"


def _load_text_token_maps() -> Dict[str, Dict[str, Dict[str, set]]]:
    sources: Dict[str, Tuple[str, Dict[str, str]]] = {
        "primary": (Config.PRIMARY_DATA, {"Gene Name": "gene_symbol", "MONDO_ID": "mondo_id"}),
        "annotation": (Config.ANNOTATION_DATA, {"Gene Name": "gene_symbol", "MONDO_ID": "mondo_id"}),
    }
    output: Dict[str, Dict[str, Dict[str, set]]] = {}
    for label, (path, rename_map) in sources.items():
        output[label] = {}
        if not os.path.exists(path):
            continue
        source = pd.read_csv(path, low_memory=False).rename(columns=rename_map)
        if "gene_symbol" not in source.columns:
            continue
        if label == "annotation":
            source = _collapse_annotation_to_gene_level(source, write_audit=False)
        elif (
            getattr(Config, "APPLY_REBUILT_GENE_DISEASE_MAPPING", False)
            and "mondo_id" in source.columns
        ):
            # Without this the gene-level token fallback below would union tokens
            # from panel-collapse rows, letting a disease the gene does not cause
            # contribute annotation tokens to that gene's profile.
            source = _apply_rebuilt_gene_disease_mapping(
                source,
                gene_col="gene_symbol",
                mondo_col="mondo_id",
                mapping_path=_resolve_gene_disease_mapping_path(
                    getattr(Config, "REBUILT_GENE_DISEASE_MAPPING", None),
                    os.path.dirname(os.path.abspath(path)),
                    Config.BASE_DIR,
                    os.path.dirname(os.path.abspath(__file__)),
                ),
                log=None,
                audit_dir=None,
                include_added=getattr(Config, "INCLUDE_HPO_ADDED_PAIRS", False),
            )
        source["gene_symbol"] = source["gene_symbol"].astype(str)
        if "mondo_id" in source.columns:
            source["mondo_id"] = source["mondo_id"].astype(str)
        for column in _TEXT_STRUCT_COLS:
            if column not in source.columns:
                continue
            mapping: Dict[str, set] = {}
            # Primary-source text may be pair-specific. ANNOTATION data has already
            # been collapsed and contains no disease identifier.
            if label != "annotation" and "mondo_id" in source.columns:
                for (gene, mondo), values in source.groupby(["gene_symbol", "mondo_id"])[column]:
                    tokens: set = set()
                    for value in values.dropna():
                        tokens.update(_tokenise_pipe_value(value))
                    mapping[_text_pair_key(gene, mondo)] = tokens
            # Gene-level fallback is retained for sources that are genuinely
            # gene annotations or lack a usable disease identifier.
            for gene, values in source.groupby("gene_symbol")[column]:
                tokens: set = set()
                for value in values.dropna():
                    tokens.update(_tokenise_pipe_value(value))
                mapping[_text_gene_key(gene)] = tokens
            output[label][column] = mapping
    return output


def _strip_global_text_vocabulary_features(container: DataContainer) -> None:
    prefixes = ("feat_primary_", "feat_annot_", "has_primary_", "has_annot_")
    global_text_columns = [
        column for column in container.full_df.columns if column.startswith(prefixes)
    ]
    if global_text_columns:
        container.full_df = container.full_df.drop(
            columns=global_text_columns, errors="ignore"
        )
        for scenario, columns in container.scenario_cols.items():
            container.scenario_cols[scenario] = [
                column for column in columns if column not in global_text_columns
            ]
        for column in global_text_columns:
            container.feature_family_lookup.pop(column, None)
        logger.log(
            f"  Removed {len(global_text_columns)} globally constructed text features; "
            "vocabularies, counts, and availability indicators will be fitted inside "
            "each training fold."
        )


def assemble_full_dataframe() -> DataContainer:
    """Join source tables and record feature families and provenance before any train/evaluation fitting."""
    logger.section("ASSEMBLING FULL DATAFRAME (no fitting)")
    container = DataContainer()
    disease_df = build_gene_disease_target(Config.PRIMARY_DATA)
    disease_df = enrich_with_annotation_dataset(disease_df)
    mondo_terms = parse_mondo_obo(Config.MONDO_OBO_PATH)
    disease_df, mondo_cols = build_mondo_features(disease_df, mondo_terms)
    container.mondo_cols = mondo_cols

    availability_columns: List[str] = []
    non_mouse_availability: List[str] = []
    mouse_availability: List[str] = []


    mechanistic_df = load_mechanistic_features()
    n_pairs_before_mechanistic = int(len(disease_df))
    full_df, mechanistic_availability = _merge_gene_level_features(
        disease_df, mechanistic_df, "mechanistic", how="left", add_availability=True
    )
    if len(full_df) != n_pairs_before_mechanistic:
        raise RuntimeError(
            "Mechanistic merge changed the number of gene-disease pairs: "
            f"{n_pairs_before_mechanistic} -> {len(full_df)}."
        )
    if mechanistic_availability:
        availability_columns.append(mechanistic_availability)
        non_mouse_availability.append(mechanistic_availability)
        n_missing_mechanistic = int((full_df[mechanistic_availability] == 0).sum())
    else:
        n_missing_mechanistic = 0
    logger.log(
        "  After pair-preserving mechanistic left merge: "
        f"{len(full_df)} pairs; missing mechanistic records={n_missing_mechanistic}"
    )

    variant_df = load_variant_features()
    if len(variant_df) > 0:
        full_df, availability = _merge_gene_level_features(
            full_df, variant_df, "variant", how="left"
        )
        variant_cols = [column for column in variant_df.columns if column != "gene_symbol"]
        if availability:
            availability_columns.append(availability)
            non_mouse_availability.append(availability)
    else:
        variant_cols = []

    mouse_df = load_mouse_lethality_features()
    if len(mouse_df) > 0:
        full_df, availability = _merge_gene_level_features(
            full_df, mouse_df, "mouse_lethality", how="left"
        )
        mouse_lethality_cols = [
            column for column in mouse_df.columns if column != "gene_symbol"
        ]
        if availability:
            availability_columns.append(availability)
            mouse_availability.append(availability)
    else:
        mouse_lethality_cols = []

    mouse_human_df = load_mouse_human_phenotype_features()
    if len(mouse_human_df) > 0:
        full_df, availability = _merge_gene_level_features(
            full_df, mouse_human_df, "mouse_human_phenotype", how="left"
        )
        mouse_human_phenotype_cols = [
            column for column in mouse_human_df.columns if column != "gene_symbol"
        ]
        if availability:
            availability_columns.append(availability)
            mouse_availability.append(availability)
    else:
        mouse_human_phenotype_cols = []

    external_df, external_cols = load_external_features()
    if len(external_df) > 0:
        full_df, availability = _merge_gene_level_features(
            full_df, external_df, "external", how="left"
        )
        if availability:
            availability_columns.append(availability)
            non_mouse_availability.append(availability)

    # Gene-level source availability belongs to non_mouse_availability,
    # keeping S1 and S1b free of mouse-derived signal.
    new_gene_df, new_gene_cols = load_new_gene_features()
    if len(new_gene_df) > 0:
        full_df, availability = _merge_gene_level_features(
            full_df, new_gene_df, "new_gene", how="left"
        )
        if availability:
            availability_columns.append(availability)
            non_mouse_availability.append(availability)

    full_df = full_df.loc[:, ~full_df.columns.duplicated()].copy()
    container.source_availability_cols = list(dict.fromkeys(availability_columns))

    metadata_cols = [
        "gene_symbol", "mondo_id", "disease_name", "hgnc_id", "Multiclass_Target",
        "max_severity", "min_severity", "mean_severity", "median_severity",
        "std_severity", "severity_range", "n_diseases", "n_severity_levels",
        "severity_entropy", "original_severity", "target_severity",
        "primary_inheritance", "is_dominant", "is_recessive", "is_xlinked",
        "has_mixed_inheritance", "n_inheritance_modes", "all_inheritance_modes",
        "hpo_term", "parent_term",
    ]
    leakage_cols = [
        "total_hpo_terms", "hpo_n_terms", "n_hpo_terms", "has_early_onset",
        "age_of_onset", "onset_age", "early_onset", "late_onset", "adult_onset",
        "childhood_onset", "infantile_onset", "neonatal_onset",
        "congenital_onset", "typical_onset_age", "mean_onset_age",
        "median_onset_age", "disease_severity", "clinical_severity",
        "phenotype_severity", "n_symptoms", "symptom_count",
    ]
    leakage_keywords = ["severity", "target", "class_label", "prediction_score", "probability"]
    safe_keywords = [
        "depmap", "gnomad", "gtex", "ppi", "network", "paralog",
        "conservation", "constraint", "essential", "organ", "tissue",
        "alphafold", "af_", "clingen", "collins", "decipher", "compensation",
        "dosage", "backup", "rigid", "mouse", "lethal", "viability", "phenotype",
    ]
    artifact_cols = [column for column in full_df.columns if "unnamed" in column.lower()]
    merge_artifact_leakage: List[str] = []
    for leakage_column in leakage_cols + metadata_cols:
        for suffix in ["_x", "_y"]:
            candidate = leakage_column + suffix
            if candidate in full_df.columns and candidate not in leakage_cols:
                merge_artifact_leakage.append(candidate)

    exclude = set(metadata_cols + leakage_cols + artifact_cols + merge_artifact_leakage)
    potential_mechanistic = [
        column for column in full_df.columns
        if column not in variant_cols
        and column not in external_cols
        and column not in mondo_cols
        and column not in mouse_lethality_cols
        and column not in mouse_human_phenotype_cols
        and column not in availability_columns
        and not column.startswith("hpo_")
        and not column.startswith("feat_")
    ]

    cleaned_mechanistic: List[str] = []
    provenance_exclusions: Dict[str, str] = {}
    for column in potential_mechanistic:
        if column in exclude:
            provenance_exclusions[column] = "identifier_target_or_prespecified_prohibited_column"
            continue
        lower = column.lower()
        if any(keyword in lower for keyword in leakage_keywords) and not any(
            safe in lower for safe in safe_keywords
        ):
            provenance_exclusions[column] = "name_indicates_target_or_post_outcome_information"
            continue
        cleaned_mechanistic.append(column)

    for column in leakage_cols:
        if column in full_df.columns:
            provenance_exclusions[column] = "prespecified_clinical_or_target_proxy"
    for column in metadata_cols:
        if column in full_df.columns and column not in provenance_exclusions:
            provenance_exclusions[column] = "metadata_identifier_or_target"
    for column in artifact_cols + merge_artifact_leakage:
        provenance_exclusions[column] = "merge_or_file_artifact"

    existing_metadata = [column for column in metadata_cols if column in full_df.columns]
    container.metadata = full_df[existing_metadata].copy()
    container.y = full_df["Multiclass_Target"].to_numpy(dtype=int)
    container.groups = full_df["gene_symbol"].to_numpy()

    pli_column = None

    if "gnomad_pLI" in full_df.columns and "gnomad_pLI" not in exclude:
        pli_column = "gnomad_pLI"
    for column in ([] if pli_column else full_df.columns):
        if re.search(r"(^|[^a-z0-9])pli([^a-z0-9]|$)", column.lower()) and column not in exclude:
            if pd.to_numeric(full_df[column], errors="coerce").notna().sum() > 50:
                pli_column = column
                break
    logger.log(f"  pLI column used for inheritance interactions: {pli_column}")

    inheritance_df, inheritance_features = build_inheritance_features(full_df, pli_column)
    for column in inheritance_features:
        if column not in full_df.columns:
            full_df[column] = inheritance_df[column].to_numpy()

    full_df, engineered_features = engineer_features(
        full_df,
        cleaned_mechanistic
        + variant_cols
        + mouse_lethality_cols
        + mouse_human_phenotype_cols,
    )

    # Mouse-derived interactions belong to S2; S1 and S1b exclude mouse signal.
    _MOUSE_DERIVED_ENGINEERED = [
        column for column in (
            "feat_lethal_x_pli",
            "feat_lethal_x_essential",
            "feat_viability_x_pli",
            "feat_has_lethal_x_pli",
        )
        if column in engineered_features
    ]
    engineered_features = [
        column for column in engineered_features
        if column not in _MOUSE_DERIVED_ENGINEERED
    ]

    organ_system_features = [
        column for column in full_df.columns if column.startswith("feat_hpo_sys_")
    ]
    container.organ_sys_cols = organ_system_features
    primary_text_features = [
        column for column in full_df.columns if column.startswith("feat_primary_")
    ]
    annotation_text_features = [
        column for column in full_df.columns if column.startswith("feat_annot_")
    ]
    hpo_binary_cols = [
        column for column in full_df.columns
        if column.startswith("hpo_")
        and column != "hpo_term"
        and pd.api.types.is_numeric_dtype(full_df[column])
    ]
    container.hpo_binary_cols = hpo_binary_cols

    s1_cols = (
        cleaned_mechanistic
        + external_cols
        + new_gene_cols
        + variant_cols
        + inheritance_features
        + engineered_features
        # Text features are rebuilt inside each training fold by the active
        # representation code; globally assembled text columns are never candidates.
        + non_mouse_availability
    )
    mouse_cols = (
        mouse_lethality_cols + mouse_human_phenotype_cols + mouse_availability
        + _MOUSE_DERIVED_ENGINEERED
    )
    s1_cols = [
        column for column in s1_cols
        if column in full_df.columns
        and not column.startswith("hpo_")
        and not column.startswith("feat_hpo_sys_")
        and not column.startswith("feat_mondo_")
        and column != "has_mondo_record"
    ]
    s1b_cols = s1_cols + organ_system_features + mondo_cols
    s1b_cols = [
        column for column in s1b_cols
        if column in full_df.columns and not column.startswith("hpo_")
    ]
    # S2 and S3 use individual HPO terms; only S1b keeps organ-system rollups.
    s2_raw = [
        column for column in s1b_cols + hpo_binary_cols + mouse_cols
        if column in full_df.columns
        and not column.startswith("feat_hpo_sys_")
    ]
    s2_cols, s2_dropped = filter_leaky_columns_for_phenotype_scenarios(
        s2_raw, "S2_Combined"
    )
    container.scenario_static_drops["S2_Combined"] = s2_dropped
    s3_raw = [
        column for column in hpo_binary_cols
        if column in full_df.columns
    ]
    s3_cols, s3_dropped = filter_leaky_columns_for_phenotype_scenarios(
        s3_raw, "S3_Phenotype"
    )
    container.scenario_static_drops["S3_Phenotype"] = s3_dropped

    container.scenario_cols["S1_Gene_Level"] = list(dict.fromkeys(s1_cols))
    container.scenario_cols["S1b_Disease_Context"] = list(dict.fromkeys(s1b_cols))
    container.scenario_cols["S2_Combined"] = list(dict.fromkeys(s2_cols))
    container.scenario_cols["S3_Phenotype"] = list(dict.fromkeys(s3_cols))

    _organ_criticality_drop = set(Config.ORGAN_CRITICALITY_COLUMNS)
    _organ_criticality_removed = set()
    for _scenario, _columns in container.scenario_cols.items():
        _organ_criticality_removed.update(
            column for column in _columns if column in _organ_criticality_drop
        )
        container.scenario_cols[_scenario] = [
            column for column in _columns if column not in _organ_criticality_drop
        ]
    logger.log(
        f"  Organ-criticality block disabled: removed "
        f"{len(_organ_criticality_removed)}/{len(Config.ORGAN_CRITICALITY_COLUMNS)} "
        f"columns from all scenario candidate sets"
    )

    if getattr(Config, "DROP_DUPLICATE_COLUMNS", False):
        # Keep one member of each duplicate pair (Config.DUPLICATE_COLUMNS).
        _duplicate_drop = set(Config.DUPLICATE_COLUMNS)
        _duplicate_removed = set()
        for _scenario, _columns in container.scenario_cols.items():
            _duplicate_removed.update(
                column for column in _columns if column in _duplicate_drop
            )
            container.scenario_cols[_scenario] = [
                column for column in _columns if column not in _duplicate_drop
            ]
        container.duplicate_column_drops = sorted(_duplicate_removed)
        logger.log(
            f"  Duplicate columns removed: "
            f"{len(_duplicate_removed)}/{len(Config.DUPLICATE_COLUMNS)} "
            f"columns from all scenario candidate sets"
        )

    if getattr(Config, "DROP_MONDO_AVAILABILITY_FLAG", True) and "has_mondo_record" in full_df.columns:
        # Ontology coverage is not a biological feature.
        _mondo_flag_counts = (
            pd.to_numeric(full_df["has_mondo_record"], errors="coerce").value_counts(dropna=False).to_dict()
        )
        for _scenario, _columns in container.scenario_cols.items():
            container.scenario_cols[_scenario] = [
                column for column in _columns if column != "has_mondo_record"
            ]
        logger.log(
            f"  has_mondo_record removed from all scenario candidate sets (values: {_mondo_flag_counts})"
        )

    if getattr(Config, "DROP_RECORD_KEEPING_FEATURES", False):
        # These columns measure classification depth or annotation coverage.
        _record_drop = set(getattr(Config, "RECORD_KEEPING_COLUMNS", ()))
        _record_removed = set()
        for _scenario, _columns in container.scenario_cols.items():
            _record_removed.update(column for column in _columns if column in _record_drop)
            container.scenario_cols[_scenario] = [
                column for column in _columns if column not in _record_drop
            ]
        container.record_keeping_drops = sorted(_record_removed)
        logger.log(
            f"  Record-keeping features removed from all scenario candidate sets: "
            f"{sorted(_record_removed)}"
        )

    if getattr(Config, "DROP_NON_BIOLOGICAL_FEATURES", False):
        # Study intensity and source availability are not biological features.
        _nb_columns = set(getattr(Config, "NON_BIOLOGICAL_COLUMNS", ()))
        _nb_prefixes = tuple(getattr(Config, "NON_BIOLOGICAL_PREFIXES", ()))

        def _is_non_biological(column: str) -> bool:
            name = str(column)
            return (
                name in _nb_columns
                or (bool(_nb_prefixes) and name.startswith(_nb_prefixes))
                or (name.startswith("has_") and name.endswith("_record"))
            )

        _nb_removed = set()
        for _scenario, _columns in container.scenario_cols.items():
            _nb_removed.update(column for column in _columns if _is_non_biological(column))
            container.scenario_cols[_scenario] = [
                column for column in _columns if not _is_non_biological(column)
            ]
        container.non_biological_drops = sorted(_nb_removed)
        logger.log(
            f"  Non-biological features removed from all scenario candidate sets "
            f"({len(_nb_removed)}): {sorted(_nb_removed)}"
        )

    if getattr(Config, "DROP_REDUNDANT_FEATURES", False):
        # These columns duplicate or closely track retained features.
        _redundant = set(getattr(Config, "REDUNDANT_FEATURE_COLUMNS", ()))
        _redundant_removed = set()
        for _scenario, _columns in container.scenario_cols.items():
            _redundant_removed.update(column for column in _columns if column in _redundant)
            container.scenario_cols[_scenario] = [
                column for column in _columns if column not in _redundant
            ]
        container.redundant_feature_drops = sorted(_redundant_removed)
        _redundant_absent = sorted(_redundant - _redundant_removed)
        logger.log(
            f"  Redundant features removed from all scenario candidate sets "
            f"({len(_redundant_removed)}): {sorted(_redundant_removed)}"
        )
        if _redundant_absent:
            logger.log(f"  Listed as redundant but not among the candidates: {_redundant_absent}")

    if getattr(Config, "DROP_INHERITANCE_COUNT_PROXIES_FROM_S1", False):
        # S1 ONLY. These columns are gene-level rollups over the gene's disease set
        # and behave as disease-count proxies (Spearman rho vs n_diseases: +0.815
        # for n_inheritance_modes, +0.720 for has_mixed_inheritance). Disease
        # context is out of scope for a gene-intrinsic scenario, so they are
        # removed from S1 and retained for S1b/S2/S3.
        _inheritance_proxy_drop = set(Config.INHERITANCE_COUNT_PROXY_COLUMNS)
        _inheritance_removed = [
            column for column in container.scenario_cols["S1_Gene_Level"]
            if column in _inheritance_proxy_drop
        ]
        container.scenario_cols["S1_Gene_Level"] = [
            column for column in container.scenario_cols["S1_Gene_Level"]
            if column not in _inheritance_proxy_drop
        ]
        # Deliberately NOT recorded in scenario_static_drops: that channel is
        # consumed by the provenance table, which stamps every entry as
        # severity_semantic_circularity. These columns are a scenario-scope
        # exclusion (disease context in a gene-intrinsic scenario), not label
        # circularity, so they are reported separately in the manifest.
        container.inheritance_proxy_drops_S1 = list(_inheritance_removed)
        logger.log(
            f"  Inheritance count-proxy columns removed from S1 only: "
            f"{_inheritance_removed} "
            f"(mode-of-inheritance columns is_dominant/is_recessive/is_xlinked retained)"
        )

    family_lookup: Dict[str, str] = {}
    for column in cleaned_mechanistic:
        family_lookup[column] = "mechanistic_core"
    for column in external_cols:
        family_lookup[column] = "external"
    for column in new_gene_cols:
        family_lookup[column] = "gene_constraint_v2"
    for column in variant_cols:
        family_lookup[column] = "variant"
    for column in inheritance_features:
        family_lookup[column] = "inheritance"
    for column in engineered_features:
        family_lookup[column] = "engineered"
    for column in _MOUSE_DERIVED_ENGINEERED:
        family_lookup[column] = "engineered"
    for column in primary_text_features:
        family_lookup[column] = "primary_text"
    for column in annotation_text_features:
        family_lookup[column] = "annotation_text"
    for column in organ_system_features:
        family_lookup[column] = "organ_system"
    for column in mondo_cols:
        family_lookup[column] = (
            "missingness_indicator" if column == "has_mondo_record" else "mondo"
        )
    for column in hpo_binary_cols:
        family_lookup[column] = "hpo_binary"
    for column in mouse_lethality_cols:
        family_lookup[column] = "mouse_lethality"
    for column in mouse_human_phenotype_cols:
        family_lookup[column] = "mouse_phenotype"
    for column in availability_columns:
        family_lookup[column] = "missingness_indicator"

    container.feature_family_lookup = family_lookup
    container.full_df = full_df

    # DUMP_FEATURE_MATRIX exports candidate columns and their source families.
    if os.environ.get("DUMP_FEATURE_MATRIX"):
        try:
            _audit_dir = os.path.join(Config.OUTPUT_DIR, "FEATURE_MATRIX_DUMP")
            os.makedirs(_audit_dir, exist_ok=True)
            full_df.to_csv(os.path.join(_audit_dir, "full_feature_matrix.csv"), index=False)
            pd.DataFrame(
                [{"feature": k, "family": v} for k, v in family_lookup.items()]
            ).to_csv(os.path.join(_audit_dir, "feature_family_lookup.csv"), index=False)
            _scn_rows = []
            for _scn, _cols in container.scenario_cols.items():
                for _c in _cols:
                    _scn_rows.append({"scenario": _scn, "feature": _c})
            pd.DataFrame(_scn_rows).to_csv(
                os.path.join(_audit_dir, "scenario_columns.csv"), index=False
            )
            logger.log(
                f'  [check] Dumped full feature matrix ({full_df.shape[0]} rows x {full_df.shape[1]} cols) to {_audit_dir}'
            )
        except Exception as _audit_err:
            logger.log(
                f'  [check] feature matrix dump failed: {_audit_err}', level="WARNING"
            )


    for scenario, columns in container.scenario_cols.items():
        logger.log(f"  Scenario {scenario}: {len(columns)} candidate columns")

    risk_by_family = {
        "mechanistic_core": "low_mechanistic",
        "external": "low_to_moderate_external_mechanistic",
        "variant": "moderate_variant_annotation",
        "inheritance": "moderate_clinical_annotation",
        "engineered": "derived_from_allowed_inputs",
        "primary_text": "moderate_text_annotation_same_source_review",
        "annotation_text": "moderate_text_annotation",
        "organ_system": "disease_context_annotation_same_source_review",
        "mondo": "disease_context_annotation",
        "hpo_binary": "phenotype_annotation_same_source_review",
        "hpo_derived": "phenotype_annotation_derived",
        "mouse_lethality": "phenotype_linked_cross_species",
        "mouse_phenotype": "phenotype_linked_cross_species",
        "missingness_indicator": "annotation_availability_research_intensity_review",
    }
    source_by_family = {
        "mechanistic_core": "core_mechanistic_features",
        "external": "external_mechanistic_features",
        "variant": "variant_features",
        "inheritance": "primary_or_mechanistic_annotation",
        "engineered": "derived_from_allowed_inputs",
        "primary_text": "primary_label_source",
        "annotation_text": "annotation_dataset",
        "organ_system": "primary_label_source",
        "mondo": "MONDO_ontology",
        "hpo_binary": "primary_label_source_and_HPO",
        "mouse_lethality": "mouse_lethality_features",
        "mouse_phenotype": "mouse_human_phenotype_features",
        "missingness_indicator": "source_availability",
    }

    provenance_rows: List[Dict[str, Any]] = []
    active_features = sorted(set().union(*(set(value) for value in container.scenario_cols.values())))
    for feature in active_features:
        family = family_lookup.get(feature, _classify_feature_family(feature, family_lookup))
        used_in = [
            scenario for scenario, columns in container.scenario_cols.items()
            if feature in columns
        ]
        provenance_rows.append({
            "feature": feature,
            "family": family,
            "source_dataset": source_by_family.get(family, "manual_review_required"),
            "used_in_scenarios": "|".join(used_in),
            "provenance_risk": risk_by_family.get(family, "unclassified_review_manually"),
            "available_at_prediction_time": True,
            "derived_from_target": False,
            "same_source_as_target_label": family in {"primary_text", "organ_system", "hpo_binary"},
            "missing_value_semantics": _missing_value_semantics_for_family(family),
            "included_in_main_clean_run": True,
            "exclusion_reason": "",
            "mi_used_to_declare_leakage": False,
            "manual_review_status": "rule_based_inclusion_requires_domain_confirmation",
        })

    for feature, reason in sorted(provenance_exclusions.items()):
        provenance_rows.append({
            "feature": feature,
            "family": "excluded",
            "source_dataset": "assembled_input",
            "used_in_scenarios": "",
            "provenance_risk": "excluded_by_provenance_rule",
            "available_at_prediction_time": False,
            "derived_from_target": "target" in reason or feature in {"Multiclass_Target", "target_severity"},
            "same_source_as_target_label": feature in disease_df.columns,
            "missing_value_semantics": "not_applicable",
            "included_in_main_clean_run": False,
            "exclusion_reason": reason,
            "mi_used_to_declare_leakage": False,
            "manual_review_status": "excluded",
        })

    for scenario, drops in container.scenario_static_drops.items():
        for feature, reason in drops:
            provenance_rows.append({
                "feature": feature,
                "family": family_lookup.get(feature, "phenotype_annotation"),
                "source_dataset": "primary_label_source_and_HPO",
                "used_in_scenarios": scenario,
                "provenance_risk": "severity_semantic_circularity",
                "available_at_prediction_time": False,
                "derived_from_target": False,
                "same_source_as_target_label": True,
                "missing_value_semantics": "structural_zero_means_term_absent",
                "included_in_main_clean_run": False,
                "exclusion_reason": reason,
                "mi_used_to_declare_leakage": False,
                "manual_review_status": "excluded",
            })

    provenance_path = os.path.join(
        Config.OUTPUT_DIR, "results", "feature_provenance_table.csv"
    )
    pd.DataFrame(provenance_rows).drop_duplicates(
        subset=["feature", "used_in_scenarios", "included_in_main_clean_run"]
    ).to_csv(provenance_path, index=False)
    logger.log(f"  Wrote feature provenance table: {provenance_path}")
    _strip_global_text_vocabulary_features(container)
    container.text_token_maps = _load_text_token_maps()
    return container
    _strip_global_text_vocabulary_features(container)
    container.text_token_maps = _load_text_token_maps()
    return container


def _learn_text_vocabulary(container: DataContainer, train_idx: np.ndarray) -> Dict[str, Dict[str, List[str]]]:
    train_idx = np.asarray(train_idx, dtype=int)
    genes = container.full_df.iloc[train_idx]["gene_symbol"].astype(str).to_numpy()
    mondos = container.full_df.iloc[train_idx]["mondo_id"].astype(str).to_numpy()
    vocabulary: Dict[str, Dict[str, List[str]]] = {}
    for source_label, column_maps in getattr(container, "text_token_maps", {}).items():
        vocabulary[source_label] = {}
        for column, token_map in column_maps.items():
            counter: Counter = Counter()
            for gene, mondo in zip(genes, mondos):
                tokens = token_map.get(_text_pair_key(gene, mondo))
                if tokens is None:
                    tokens = token_map.get(_text_gene_key(gene), set())
                counter.update(tokens)
            vocabulary[source_label][column] = [
                token for token, _ in counter.most_common(Config.TEXT_VOCAB_TOP_N)
            ]
    return vocabulary


def _add_training_text_features(
    container: DataContainer,
    indices: np.ndarray,
    vocabulary: Dict[str, Dict[str, List[str]]],
    family_lookup: Dict[str, str],
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    frame = container.full_df.iloc[indices].copy()

    genes = frame["gene_symbol"].astype(str).values
    mondos = frame["mondo_id"].astype(str).values

    token_maps = _load_text_token_maps()

    generated = []
    pending_columns = {}

    for source_label, columns in vocabulary.items():
        for source_column, tokens in columns.items():
            token_map = token_maps.get(
                source_label,
                {},
            ).get(source_column, {})

            row_tokens = []
            available = np.zeros(
                len(genes),
                dtype=np.uint8,
            )

            for row_index, (gene, mondo) in enumerate(
                zip(genes, mondos)
            ):
                pair_key = _text_pair_key(gene, mondo)
                gene_key = _text_gene_key(gene)

                # Placeholder-only records have zero counts; absent records keep missing counts.
                if pair_key in token_map:
                    tokens_here = token_map[pair_key]
                elif gene_key in token_map:
                    tokens_here = token_map[gene_key]
                else:
                    tokens_here = None
                if tokens_here is None:
                    row_tokens.append(None)
                else:
                    row_tokens.append(tokens_here)
                    available[row_index] = 1 if len(tokens_here) else 0

            availability_name = safe_feature_name(
                f"has_{source_label}",
                f"{source_column}_record",
                max_len=40,
            )

            count_name = (
                f"feat_{source_label}_"
                f"{source_column.lower()}_count"
            )

            # annotation counts and "annotation present" flags describe how
            # well a gene is studied, so they are built only when
            # Config.TEXT_ADD_COUNT_AND_AVAILABILITY is True.
            if getattr(Config, "TEXT_ADD_COUNT_AND_AVAILABILITY", True):
                pending_columns[availability_name] = (
                    available.astype(float)
                )

                pending_columns[count_name] = np.asarray(
                    [
                        float(len(values))
                        if values is not None
                        else np.nan
                        for values in row_tokens
                    ],
                    dtype=float,
                )

                generated.extend(
                    [availability_name, count_name]
                )

                family_lookup[availability_name] = (
                    "missingness_indicator"
                )
                family_lookup[count_name] = (
                    f"{source_label}_text"
                )

            for token in tokens:
                name = safe_feature_name(
                    (
                        f"feat_{source_label}_"
                        f"{source_column.lower()}"
                    ),
                    token,
                    max_len=30,
                )

                if (
                    name in frame.columns
                    or name in pending_columns
                ):
                    continue

                pending_columns[name] = np.fromiter(
                    (
                        1.0 if values and token in values else 0.0
                        for values in row_tokens
                    ),
                    dtype=float,
                    count=len(row_tokens),
                )

                generated.append(name)
                family_lookup[name] = (
                    f"{source_label}_text"
                )

    if pending_columns:
        generated_frame = pd.DataFrame(
            pending_columns,
            index=frame.index,
        )

        frame = pd.concat(
            [frame, generated_frame],
            axis=1,
        ).copy()

    return frame, generated, family_lookup


class FoldPreprocessor:

    def __init__(
        self,
        scenario_name: str,
        hpo_binary_cols: List[str],
        mondo_cols: List[str],
        static_drops: Optional[List[Tuple[str, str]]] = None,
    ):
        self.scenario_name = scenario_name
        self.hpo_binary_cols = list(hpo_binary_cols)
        self.mondo_cols = list(mondo_cols)
        self.kept_cols: List[str] = []
        self.kept_mondo_cols: List[str] = []
        self.kept_hpo_cols: List[str] = []
        self.hpo_ic_scores: Optional[np.ndarray] = None
        self.hpo_ic_75th: Optional[float] = None
        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[RobustScaler] = None
        self.derived_feature_names: List[str] = []
        self.missing_indicator_feature_names: List[str] = []
        self.final_feature_names: List[str] = []
        self.dropped_features: List[Dict[str, Any]] = []
        if static_drops:
            for feature_name, reason in static_drops:
                self.dropped_features.append({
                    "feature": feature_name,
                    "audit_action": "drop",
                    "drop_reason": reason,
                    "drop_statistic": "",
                })

    def _add_hpo_derived_features(self, hpo_matrix: np.ndarray) -> np.ndarray:
        # Phenotype scenarios keep only the raw binary HPO presence indicators;
        # the IC / frequency aggregate features are intentionally disabled.
        if getattr(Config, "DROP_HPO_DERIVED_IC", True):
            return np.zeros((hpo_matrix.shape[0], 0))
        if hpo_matrix.size == 0 or self.hpo_ic_scores is None:
            return np.zeros((hpo_matrix.shape[0], 0))
        n_samples = hpo_matrix.shape[0]
        ic_weighted = hpo_matrix * self.hpo_ic_scores[np.newaxis, :]
        n_terms = hpo_matrix.sum(axis=1)
        n_terms_safe = np.clip(n_terms, 1, None)
        mean_ic = np.zeros(n_samples)
        max_ic = np.zeros(n_samples)
        median_ic = np.zeros(n_samples)
        ic_range = np.zeros(n_samples)
        ic_std = np.zeros(n_samples)
        rare_count = np.zeros(n_samples)
        for index in range(n_samples):
            present = hpo_matrix[index] > 0
            if not present.any():
                continue
            information_content = self.hpo_ic_scores[present]
            mean_ic[index] = float(np.mean(information_content))
            max_ic[index] = float(np.max(information_content))
            median_ic[index] = float(np.median(information_content))
            ic_range[index] = float(np.max(information_content) - np.min(information_content))
            ic_std[index] = float(np.std(information_content))
            rare_count[index] = float((information_content >= self.hpo_ic_75th).sum())
        rare_term_ratio = rare_count / n_terms_safe
        derived_blocks: List[np.ndarray] = [
            mean_ic, max_ic, median_ic, ic_range, ic_std, rare_term_ratio
        ]
        derived_names = [
            "feat_hpo_mean_ic", "feat_hpo_max_ic", "feat_hpo_median_ic",
            "feat_hpo_ic_range", "feat_hpo_ic_std", "feat_hpo_rare_term_ratio",
        ]
        if not self.derived_feature_names:
            self.derived_feature_names = derived_names
        return np.column_stack(derived_blocks) if derived_blocks else np.zeros((n_samples, 0))

    def fit_transform(
        self, full_df: pd.DataFrame, train_idx: np.ndarray, candidate_cols: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        train_df = full_df.iloc[train_idx]
        candidate_cols = [column for column in candidate_cols if column in full_df.columns]
        if not candidate_cols:
            raise ValueError(f"{self.scenario_name}: no candidate features are available.")
        train_subset = train_df[candidate_cols].apply(pd.to_numeric, errors="coerce")

        missing_fraction = train_subset.isna().mean()
        keep = missing_fraction[missing_fraction <= Config.MISSING_DROP_THRESHOLD].index.tolist()
        for column in candidate_cols:
            if column not in keep:
                self.dropped_features.append({
                    "feature": column,
                    "audit_action": "drop",
                    "drop_reason": "high_missing_training_fold",
                    "drop_statistic": float(missing_fraction.get(column, 1.0)),
                })
        if not keep:
            raise ValueError(f"{self.scenario_name}: all features failed the missingness filter.")

        training_medians = train_subset[keep].median(axis=0, skipna=True)
        variance_frame = train_subset[keep].fillna(training_medians)
        variances = variance_frame.var(axis=0)
        keep_after_variance = variances[
            variances >= Config.VAR_DROP_THRESHOLD
        ].index.tolist()
        for column in keep:
            if column not in keep_after_variance:
                self.dropped_features.append({
                    "feature": column,
                    "audit_action": "drop",
                    "drop_reason": "low_variance_training_fold",
                    "drop_statistic": float(variances.get(column, 0.0)),
                })
        keep = keep_after_variance
        if not keep:
            raise ValueError(f"{self.scenario_name}: all features failed the variance filter.")

        mondo_in_keep = [column for column in self.mondo_cols if column in keep]
        kept_mondo: List[str] = []
        if mondo_in_keep:
            mondo_training = train_df[mondo_in_keep].apply(pd.to_numeric, errors="coerce")
            positive_counts = (mondo_training > 0).sum(axis=0)
            kept_mondo = positive_counts[
                positive_counts >= Config.MONDO_MIN_TRAIN_COUNT
            ].index.tolist()
            for required_column in ["feat_mondo_depth", "feat_mondo_n_classes", "has_mondo_record"]:
                if required_column in keep and required_column not in kept_mondo:
                    kept_mondo.append(required_column)
            for column in mondo_in_keep:
                if column not in kept_mondo:
                    self.dropped_features.append({
                        "feature": column,
                        "audit_action": "drop",
                        "drop_reason": "mondo_low_frequency_training_fold",
                        "drop_statistic": int(positive_counts.get(column, 0)),
                    })
            keep = [column for column in keep if column not in mondo_in_keep] + kept_mondo
        self.kept_mondo_cols = kept_mondo

        hpo_in_keep = [column for column in self.hpo_binary_cols if column in keep]
        kept_hpo: List[str] = []
        if hpo_in_keep:
            hpo_training = train_df[hpo_in_keep].apply(pd.to_numeric, errors="coerce")
            positive_counts = (hpo_training.fillna(0) > 0).sum(axis=0)
            n_train = len(hpo_training)
            # Ontology-level granularity band, computed on TRAINING pairs only
            # (no leakage). Drop singletons (too rare to generalise) and
            # near-universal over-general terms (present in almost every pair,
            # no discriminative value). Applies to any scenario carrying HPO
            # columns, i.e. S2 and S3.
            min_count = int(getattr(Config, "HPO_MIN_TRAIN_COUNT", 2))
            max_fraction = float(getattr(Config, "HPO_MAX_TRAIN_FRACTION", 1.0))
            max_count = max_fraction * n_train
            in_band = (positive_counts >= min_count) & (positive_counts <= max_count)
            kept_hpo = positive_counts[in_band].index.tolist()
            for column in hpo_in_keep:
                if column not in kept_hpo:
                    count = int(positive_counts.get(column, 0))
                    reason = (
                        "hpo_high_frequency_training_fold"
                        if count > max_count
                        else "hpo_low_frequency_training_fold"
                    )
                    self.dropped_features.append({
                        "feature": column,
                        "audit_action": "drop",
                        "drop_reason": reason,
                        "drop_statistic": count,
                    })
            keep = [column for column in keep if column not in hpo_in_keep] + kept_hpo
        self.kept_hpo_cols = kept_hpo

        keep_set = set(keep)
        self.kept_cols = [column for column in candidate_cols if column in keep_set]
        training_base = train_df[self.kept_cols].apply(pd.to_numeric, errors="coerce")

        if (
            self.kept_hpo_cols
            and getattr(Config, "DROP_HPO_DERIVED_IC", True)
        ):
            logger.log(
                f"    {self.scenario_name}: HPO IC-derived features disabled; "
                f"retaining {len(self.kept_hpo_cols)} binary HPO columns."
            )

        if self.kept_hpo_cols:
            hpo_training_matrix = (
                training_base[self.kept_hpo_cols]
                .fillna(0)
                .to_numpy(dtype=float)
            )
            if getattr(Config, "DROP_HPO_DERIVED_IC", True):
                # Direct binary HPO terms need no frequency or information-content estimates.
                self.hpo_ic_scores = None
                self.hpo_ic_75th = None
                self.derived_feature_names = []
                hpo_derived = np.zeros((len(training_base), 0))
            else:
                n_train = hpo_training_matrix.shape[0]
                frequencies = hpo_training_matrix.sum(axis=0) / max(n_train, 1)
                frequencies = np.clip(
                    frequencies,
                    1.0 / max(n_train, 1),
                    1.0,
                )
                self.hpo_ic_scores = -np.log2(frequencies)
                self.hpo_ic_75th = float(
                    np.percentile(self.hpo_ic_scores, 75)
                )
                hpo_derived = self._add_hpo_derived_features(
                    hpo_training_matrix
                )
        else:
            self.hpo_ic_scores = None
            self.hpo_ic_75th = None
            self.derived_feature_names = []
            hpo_derived = np.zeros((len(training_base), 0))

        imputer_kwargs = {
            "strategy": "median",
            "add_indicator": bool(getattr(Config, "ADD_MISSING_INDICATORS", True)),
        }
        try:
            self.imputer = SimpleImputer(keep_empty_features=True, **imputer_kwargs)
        except TypeError:
            self.imputer = SimpleImputer(**imputer_kwargs)
        imputed = self.imputer.fit_transform(training_base.to_numpy(dtype=float))

        indicator_indices: List[int] = []
        if getattr(self.imputer, "indicator_", None) is not None:
            indicator_indices = [int(index) for index in self.imputer.indicator_.features_]
        self.missing_indicator_feature_names = [
            f"{self.kept_cols[index]}__missing" for index in indicator_indices
        ]
        imputed_feature_names = self.kept_cols + self.missing_indicator_feature_names
        if imputed.shape[1] != len(imputed_feature_names):
            raise RuntimeError(
                f"{self.scenario_name}: imputer produced {imputed.shape[1]} columns but "
                f"{len(imputed_feature_names)} names were constructed."
            )

        combined = np.hstack([imputed, hpo_derived])
        self.scaler = RobustScaler()
        scaled = self.scaler.fit_transform(combined)
        self.final_feature_names = imputed_feature_names + self.derived_feature_names
        if scaled.shape[1] != len(self.final_feature_names):
            raise RuntimeError(
                f"{self.scenario_name}: transformed matrix/name mismatch "
                f"({scaled.shape[1]} vs {len(self.final_feature_names)})."
            )
        return scaled, self.final_feature_names

    def transform(self, full_df: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
        if self.imputer is None or self.scaler is None:
            raise RuntimeError("FoldPreprocessor must be fitted before transform().")
        subset = full_df.iloc[idx]
        base = subset[self.kept_cols].apply(pd.to_numeric, errors="coerce")
        if (
            self.kept_hpo_cols
            and not getattr(Config, "DROP_HPO_DERIVED_IC", True)
        ):
            hpo_matrix = (
                base[self.kept_hpo_cols]
                .fillna(0)
                .to_numpy(dtype=float)
            )
            hpo_derived = self._add_hpo_derived_features(hpo_matrix)
        else:
            hpo_derived = np.zeros((len(base), 0))
        imputed = self.imputer.transform(base.to_numpy(dtype=float))
        combined = np.hstack([imputed, hpo_derived])
        transformed = self.scaler.transform(combined)
        if transformed.shape[1] != len(self.final_feature_names):
            raise RuntimeError(
                f"{self.scenario_name}: transform produced an unexpected number of columns."
            )
        return transformed

    def save_audit(self, scenario_name: str, runtime_mi_drops: Optional[List[Dict]] = None):
        rows = list(self.dropped_features)
        if runtime_mi_drops:
            rows.extend(runtime_mi_drops)
        if not rows:
            return
        for row in rows:
            row.setdefault("audit_action", "drop")
            row.setdefault("drop_reason", "unspecified")
            row.setdefault("drop_statistic", "")
        dataframe = pd.DataFrame(rows)
        safe_scenario = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(scenario_name))
        feature_audit_path = os.path.join(
            Config.OUTPUT_DIR, "results", f"feature_audit_{safe_scenario}.csv"
        )
        dataframe.to_csv(feature_audit_path, index=False)
        dataframe.to_csv(
            os.path.join(Config.OUTPUT_DIR, "results", f"leakage_audit_{safe_scenario}.csv"),
            index=False,
        )
        action_counts = dataframe["audit_action"].value_counts().to_dict()
        reason_counts = dataframe["drop_reason"].value_counts().to_dict()
        logger.log(
            f'    Feature check ({scenario_name}): {len(dataframe)} entries; actions={action_counts}; reasons={reason_counts}'
        )


def write_split_summary(
    container: DataContainer,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    out_path: Optional[str] = None,
    include_test_labels: bool = False,
) -> pd.DataFrame:
    """Write split counts while keeping test class labels sealed until evaluation."""
    if out_path is None:
        out_path = os.path.join(Config.OUTPUT_DIR, "results", "split_summary.csv")
    y = container.y
    groups = container.groups
    rows: List[Dict[str, Any]] = []
    for partition_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        expose_labels = partition_name != "test" or include_test_labels
        counts = (
            np.bincount(y[idx], minlength=Config.NUM_CLASSES).astype(float)
            if expose_labels
            else np.full(Config.NUM_CLASSES, np.nan)
        )
        row: Dict[str, Any] = {
            "partition": partition_name,
            "n_pairs": int(len(idx)),
            "n_genes": int(len(np.unique(groups[idx]))),
            "class_labels_exposed": bool(expose_labels),
        }
        for severity_class in range(Config.NUM_CLASSES):
            value = counts[severity_class]
            row[f"n_class_{severity_class}"] = int(value) if np.isfinite(value) else np.nan
        rows.append(row)
    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(out_path, index=False)
    logger.subsection("Realised split summary")
    for _, row in dataframe.iterrows():
        if row["class_labels_exposed"]:
            per_class = ", ".join(
                f"c{severity_class}={int(row[f'n_class_{severity_class}'])}"
                for severity_class in range(Config.NUM_CLASSES)
            )
        else:
            per_class = "class counts sealed"
        logger.log(
            f"  {row['partition']:6s} pairs={int(row['n_pairs']):5d} "
            f"genes={int(row['n_genes']):5d} [{per_class}]"
        )
    return dataframe


def _classify_feature_family(name: str, family_lookup: Dict[str, str]) -> str:
    hpo_derived_exact = {
        "feat_hpo_mean_ic",
        "feat_hpo_max_ic",
        "feat_hpo_median_ic",
        "feat_hpo_ic_range",
        "feat_hpo_ic_std",
        "feat_hpo_rare_term_ratio",
    }
    if name in hpo_derived_exact or name.startswith("feat_hpo_pca_"):
        return "hpo_derived"
    # only true availability flags (has_<source>_record, has_annot_<column>_record)
    # are missingness indicators. Biological flags such as has_mouse_ortholog,
    # has_close_paralog or has_lethal_phenotype keep their own family.
    if name.endswith("__missing") or (name.startswith("has_") and name.endswith("_record")):
        return "missingness_indicator"
    if name in family_lookup:
        return family_lookup[name]
    return "unclassified"


def _should_write_mi_csv(scenario_name: str) -> bool:
    internal_tokens = ("_perm", "_seed", "_mithr_", "_budgetx_", "_hpobudget_")
    is_internal = any((tok in str(scenario_name) for tok in internal_tokens))
    return not is_internal or getattr(Config, "SAVE_MI_CSV_FOR_INTERNAL_RUNS", False)


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
        elif feature.startswith("feat_annot_"):
            family = "annotation_text"
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
        "annotation_text": "annotation_dataset",
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
        "provenance_status": (
            "same_source_review_required" if same_source else "independent_or_derived_review"
        ),
    }


def write_high_mi_provenance_report() -> pd.DataFrame:
    columns = [
        "analysis_context", "feature", "mi_score", "high_mi_review_cutoff",
        "feature_type_for_mi", "feature_family", "source_dataset",
        "same_source_as_target_label", "provenance_status",
        "mi_used_to_declare_leakage",
    ]
    dataframe = pd.DataFrame(_high_mi_audit_records, columns=columns)
    if not dataframe.empty:
        dataframe = dataframe.drop_duplicates().sort_values(
            ["analysis_context", "mi_score", "feature"],
            ascending=[True, False, True],
        )
    path = os.path.join(
        Config.OUTPUT_DIR, "results", "high_mi_provenance_flags.csv"
    )
    dataframe.to_csv(path, index=False)
    logger.log(
        f"  Wrote {path} with {len(dataframe)} high-MI context/feature records "
        f"({dataframe['feature'].nunique() if not dataframe.empty else 0} unique features)."
    )
    return dataframe


def _infer_discrete_feature_mask(
    X: np.ndarray, feature_names: List[str]
) -> np.ndarray:
    mask = np.zeros(len(feature_names), dtype=bool)
    explicit_prefixes = (
        "hpo_", "feat_hpo_sys_", "feat_is_", "has_", "feat_mondo_",
    )
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
        if integer_like and len(unique) <= int(getattr(Config, "MI_MAX_DISCRETE_UNIQUE", 10)):
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
    positive = mi_scores[mi_scores > getattr(Config, "MI_POSITIVE_EPS", 1e-12)]
    if len(positive):
        quantile_cutoff = float(np.quantile(
            positive, float(getattr(Config, "MI_HIGH_AUDIT_QUANTILE", 0.99))
        ))
    else:
        quantile_cutoff = np.inf
    audit_cutoff = max(
        float(getattr(Config, "MI_HIGH_AUDIT_MIN_SCORE", 0.20)), quantile_cutoff
    )
    high_mi_flag = np.isfinite(audit_cutoff) & (mi_scores >= audit_cutoff)
    mi_dataframe = pd.DataFrame({
        "feature": feature_names,
        "mi_score": mi_scores,
        "feature_type_for_mi": np.where(discrete_mask, "discrete", "continuous"),
        "high_mi_review_flag": high_mi_flag,
        "high_mi_review_cutoff": audit_cutoff,
        "mi_used_to_declare_leakage": False,
    }).sort_values("mi_score", ascending=False)
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
        audit_entries.append({
            "feature": record["feature"],
            "audit_action": "review_only",
            "drop_reason": "high_mi_provenance_review_not_dropped",
            "drop_statistic": record["mi_score"],
            "analysis_context": record["analysis_context"],
            "feature_family": record["feature_family"],
            "source_dataset": record["source_dataset"],
            "same_source_as_target_label": record["same_source_as_target_label"],
            "provenance_status": record["provenance_status"],
        })
    if audit_entries:
        logger.log(
            f'  MI check for {scenario_name}: {len(audit_entries)} high-value features flagged for provenance review; none were removed because of MI magnitude.',
            level="WARNING",
        )
        # Detailed names are logged outside the very large permutation sweep; all
        # contexts, including permutations, are still written to the CSV report.
        if "_perm" not in str(scenario_name):
            for entry in audit_entries:
                logger.log(
                    "    HIGH-MI FEATURE "
                    f"name={entry['feature']} MI={entry['drop_statistic']:.6f} "
                    f"family={entry['feature_family']} source={entry['source_dataset']} "
                    f"same_source_as_label={entry['same_source_as_target_label']}",
                    level="WARNING",
                )
    return mi_scores, mi_dataframe, audit_entries


def select_features_mi(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    top_k: Optional[int] = None,
    mi_threshold: float = 0.01,
    mi_leakage_threshold: Optional[float] = None,
    min_features: int = 15,
    scenario_name: str = "default",
    family_lookup: Optional[Dict[str, str]] = None,
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray, List[Dict]]:
    """Training-only MI selection; MI magnitude is never treated as leakage."""
    if X.shape[1] != len(feature_names):
        if X.shape[1] > len(feature_names):
            feature_names = feature_names + [
                f"unknown_feat_{index}" for index in range(X.shape[1] - len(feature_names))
            ]
        else:
            feature_names = feature_names[: X.shape[1]]
    logger.log(
        f"  Computing MI for {scenario_name} ({len(feature_names)} features, "
        f"minimum selection threshold={mi_threshold}, min_features={min_features})..."
    )
    mi_scores, mi_dataframe, audit_entries = _compute_mutual_information_with_audit(
        X, y, feature_names, scenario_name, family_lookup
    )
    positive_epsilon = float(getattr(Config, "MI_POSITIVE_EPS", 1e-12))
    positive_indices = np.flatnonzero(mi_scores > positive_epsilon)
    if not len(positive_indices):
        raise ValueError(f"{scenario_name}: no feature had positive training-fold MI.")

    final_mask = np.zeros(len(feature_names), dtype=bool)
    if top_k is not None:
        ranked_positive = positive_indices[np.argsort(mi_scores[positive_indices])[::-1]]
        final_mask[ranked_positive[: min(int(top_k), len(ranked_positive))]] = True
    else:
        final_mask[mi_scores > max(float(mi_threshold), positive_epsilon)] = True
        if int(final_mask.sum()) < int(min_features):
            ranked_positive = positive_indices[np.argsort(mi_scores[positive_indices])[::-1]]
            target = min(int(min_features), len(ranked_positive))
            final_mask[ranked_positive[:target]] = True
            if len(ranked_positive) < int(min_features):
                logger.log(
                    f"    Only {len(ranked_positive)} features had positive MI; "
                    f"the requested minimum was {min_features}. Zero-MI features were not forced in.",
                    level="WARNING",
                )

    selected_names = [
        feature for feature, selected in zip(feature_names, final_mask) if selected
    ]
    if not selected_names:
        raise ValueError(f"{scenario_name}: MI selection produced zero features.")
    selected_scores = mi_scores[final_mask]
    logger.log(
        f"    Selected {len(selected_names)} / {len(feature_names)} positive-MI features "
        f"(min={selected_scores.min():.4g}, max={selected_scores.max():.4g})"
    )
    _write_mi_scores(mi_dataframe, scenario_name)
    return X[:, final_mask], selected_names, mi_scores, final_mask, audit_entries


def select_features_mi_grouped(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    family_lookup: Dict[str, str],
    budget: Dict[str, int],
    mi_leakage_threshold: Optional[float] = None,
    scenario_name: str = "default",
    mi_threshold: float = 0.0,
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray, List[Dict], Dict[str, int]]:
    """Family-budgeted MI selection using positive MI only."""
    if X.shape[1] != len(feature_names):
        if X.shape[1] > len(feature_names):
            feature_names = feature_names + [
                f"unknown_feat_{index}" for index in range(X.shape[1] - len(feature_names))
            ]
        else:
            feature_names = feature_names[: X.shape[1]]
    logger.log(
        f"  Grouped MI selection for {scenario_name} ({len(feature_names)} features, "
        f"{len(budget)} families, minimum MI={mi_threshold})..."
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
            f"  WARNING: {len(unclassified)} unclassified features; sample: "
            f"{[feature_names[index] for index in unclassified[:10]]}",
            level="WARNING",
        )

    positive_epsilon = float(getattr(Config, "MI_POSITIVE_EPS", 1e-12))
    minimum_score = max(float(mi_threshold), positive_epsilon)
    selected_indices: List[int] = []
    for family, indices in sorted(family_indices.items()):
        cap = 50 if family == "unclassified" else int(budget.get(family, 50))
        candidates = [
            index for index in indices if mi_scores[index] > minimum_score
        ]
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
                f"    Only {len(selected_indices)} features had positive MI; requested minimum "
                f"was {minimum_features}. Zero-MI features were not forced in.",
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
        f"    Grouped selection: {int(final_mask.sum())}/{len(feature_names)} features "
        f"across {len(family_counts)} families"
    )
    for family in sorted(family_counts):
        available = len(family_indices.get(family, []))
        selected = family_counts[family]
        cap = 50 if family == "unclassified" else int(budget.get(family, 50))
        logger.log(f"      {family:25s}  {selected:4d}/{available:4d}  (budget={cap})")

    selected_names = [
        feature for feature, selected in zip(feature_names, final_mask) if selected
    ]
    selected_scores = mi_scores[final_mask]
    _write_mi_scores(mi_dataframe, scenario_name)
    logger.log(
        f"    Final: {len(selected_names)} selected features "
        f"(min MI={selected_scores.min():.4g}, max MI={selected_scores.max():.4g})"
    )
    return (
        X[:, final_mask], selected_names, mi_scores, final_mask,
        audit_entries, family_counts,
    )


def _distribution(values: np.ndarray) -> np.ndarray:
    counts = np.bincount(values.astype(int), minlength=Config.NUM_CLASSES).astype(float)
    return counts / max(counts.sum(), 1.0)


def _split_objective(
    selected_idx: np.ndarray,
    all_idx: np.ndarray,
    y: np.ndarray,
    target_fraction: float,
) -> float:
    if len(selected_idx) == 0 or len(selected_idx) == len(all_idx):
        return float("inf")
    fraction_error = abs(len(selected_idx) / len(all_idx) - target_fraction)
    distribution_error = float(np.abs(_distribution(y[selected_idx]) - _distribution(y[all_idx])).sum())
    class_counts = np.bincount(y[selected_idx], minlength=Config.NUM_CLASSES)
    missing_penalty = 10.0 * float(np.sum(class_counts < Config.SPLIT_MIN_CLASS_COUNT))
    return 4.0 * fraction_error + distribution_error + missing_penalty


def _optimised_group_holdout(
    indices: np.ndarray,
    y: np.ndarray,
    group_values: np.ndarray,
    target_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=int)
    local_groups = np.asarray(group_values)[indices]
    unique_groups = np.unique(local_groups)
    if len(unique_groups) < 3:
        raise ValueError("At least three independent groups are required for a holdout split.")
    rng = np.random.RandomState(seed)
    group_to_indices = {g: indices[local_groups == g] for g in unique_groups}
    target_n = max(1, int(round(target_fraction * len(indices))))
    best_selected: Optional[np.ndarray] = None
    best_score = float("inf")
    for _ in range(Config.SPLIT_SEARCH_ITERATIONS):
        order = rng.permutation(unique_groups)
        chosen: List[int] = []
        current_n = 0
        for group in order:
            group_idx = group_to_indices[group]
            before = abs(current_n - target_n)
            after = abs(current_n + len(group_idx) - target_n)
            if current_n < target_n or after < before:
                chosen.extend(group_idx.tolist())
                current_n += len(group_idx)
            if current_n >= target_n and after > before:
                break
        selected = np.asarray(sorted(set(chosen)), dtype=int)
        score = _split_objective(selected, indices, y, target_fraction)
        if score < best_score:
            best_score = score
            best_selected = selected
    if best_selected is None:
        raise RuntimeError("Could not construct a valid group-disjoint holdout.")
    selected_set = set(best_selected.tolist())
    retained = np.asarray([i for i in indices if i not in selected_set], dtype=int)
    return retained, best_selected


def make_train_test_split(container: DataContainer, seed: int = RANDOM_SEED):
    """Optimise a gene-disjoint 80/20 development/test split."""
    logger.subsection(f"Creating optimised gene-disjoint development/test split (seed={seed})")
    all_idx = np.arange(len(container.y), dtype=int)
    development_idx, test_idx = _optimised_group_holdout(
        all_idx, container.y, container.groups, Config.TEST_SIZE, seed
    )
    container.train_idx = development_idx
    container.test_idx = test_idx
    train_genes = set(container.groups[development_idx])
    test_genes = set(container.groups[test_idx])
    if train_genes & test_genes:
        raise RuntimeError("Gene overlap detected between development and test partitions.")
    logger.log(
        f"  Development={len(development_idx)} pairs/{len(train_genes)} genes; "
        f"test={len(test_idx)} pairs/{len(test_genes)} genes."
    )


def _split_development_train_validation(
    container: DataContainer,
    development_idx: np.ndarray,
    grouping: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # VAL_SIZE is a fraction of the complete dataset, not of development.
    relative_fraction = Config.VAL_SIZE / max(1.0 - Config.TEST_SIZE, 1e-12)
    train_idx, val_idx = _optimised_group_holdout(
        development_idx, container.y, grouping, relative_fraction, seed + 17
    )
    return train_idx, val_idx


def _write_split_identifiers(
    container: DataContainer,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    prefix: str = "primary",
) -> None:
    metadata = container.metadata.reset_index(drop=True)
    rows: List[Dict[str, Any]] = []
    for partition, idx in (("train", train_idx), ("validation", val_idx), ("test", test_idx)):
        for row_idx in idx:
            rows.append({
                "analysis": prefix,
                "partition": partition,
                "row_index": int(row_idx),
                "gene_symbol": metadata.iloc[row_idx].get("gene_symbol", ""),
                "mondo_id": metadata.iloc[row_idx].get("mondo_id", ""),
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(Config.OUTPUT_DIR, "results", f"split_identifiers_{prefix}.csv"),
        index=False,
    )


def _fit_scenario_representation(
    container: DataContainer,
    scenario: str,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    budget: Optional[Dict[str, int]],
    mi_leakage_threshold: Optional[float],
    tag: str,
    drop_severity_hpo_terms: Optional[bool] = None,
) -> Dict[str, Any]:
    allowed_text_sources = tuple(
        SCENARIO_TEXT_SOURCES.get(scenario, ())
    )

    unknown_sources = set(allowed_text_sources) - {"primary", "annotation"}
    if unknown_sources:
        raise ValueError(
            f"{scenario}: unknown text sources configured: "
            f"{sorted(unknown_sources)}"
        )

    dynamic_text_cols: List[str] = []
    dynamic_lookup: Dict[str, str] = {}
    vocabulary: Dict[str, Any] = {}

    if allowed_text_sources:
        # Learn vocabulary only from the current training fold.
        full_vocabulary = _learn_text_vocabulary(
            container,
            train_idx,
        )

        vocabulary = {
            source: full_vocabulary[source]
            for source in allowed_text_sources
            if source in full_vocabulary
        }

        # Apply the training-learned vocabulary to all rows so that
        # the global train_idx and eval_idx positions remain valid.
        all_indices = np.arange(
            len(container.full_df),
            dtype=int,
        )

        (
            temp_df,
            dynamic_text_cols,
            dynamic_lookup,
        ) = _add_training_text_features(
            container=container,
            indices=all_indices,
            vocabulary=vocabulary,
            family_lookup=dynamic_lookup,
        )

        if len(temp_df) != len(container.full_df):
            raise RuntimeError(
                "Text-feature frame lost global row alignment."
            )
    else:
        # Pure non-text scenario: do not learn or construct text features.
        temp_df = container.full_df

    candidate_cols = (
        list(container.scenario_cols.get(scenario, []))
        + dynamic_text_cols
    )

    candidate_cols = list(
        dict.fromkeys(
            column
            for column in candidate_cols
            if column in temp_df.columns
        )
    )
    candidate_cols, dynamic_static_drops = filter_leaky_columns_for_phenotype_scenarios(
        candidate_cols,
        scenario,
        drop_severity_hpo_terms=drop_severity_hpo_terms,
    )
    combined_static_drops = list(container.scenario_static_drops.get(scenario, []))
    combined_static_drops.extend(dynamic_static_drops)
    combined_static_drops = list(dict.fromkeys(combined_static_drops))


    text_prefix_to_source = {
        "feat_primary_": "primary",
        "has_primary_": "primary",
        "feat_annot_": "annotation",
        "has_annot_": "annotation",
    }

    disallowed_text_cols = [
        column
        for column in candidate_cols
        for prefix, source in text_prefix_to_source.items()
        if column.startswith(prefix)
        and source not in allowed_text_sources
    ]

    if disallowed_text_cols:
        raise RuntimeError(
            f"{scenario}: disallowed text features entered the scenario: "
            f"{disallowed_text_cols[:10]}"
        )

    logger.log(
        f"  {scenario}: allowed text sources="
        f"{list(allowed_text_sources) if allowed_text_sources else 'none'}; "
        f"dynamic text columns={len(dynamic_text_cols)}"
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
        X_train, selected_names, _, selected_mask, runtime_drops, family_counts = (
            select_features_mi_grouped(
                X_train_full,
                y_train,
                feature_names,
                family_lookup=family_lookup,
                budget=actual_budget,
                mi_leakage_threshold=mi_leakage_threshold,
                scenario_name=tag,
            )
        )
    else:
        mechanistic = (
            ("mechanistic" in scenario.lower() or "S1" in scenario)
            and "S1b" not in scenario and "pheno" not in scenario.lower()
        )
        X_train, selected_names, _, selected_mask, runtime_drops = select_features_mi(
            X_train_full,
            y_train,
            feature_names,
            top_k=Config.MI_TOP_K,
            mi_threshold=(Config.MI_THRESHOLD_MECHANISTIC if mechanistic else Config.MI_THRESHOLD_DEFAULT),
            mi_leakage_threshold=mi_leakage_threshold,
            min_features=(Config.MI_MIN_FEATURES_MECHANISTIC if mechanistic else Config.MI_MIN_FEATURES_DEFAULT),
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
    }


def _nested_splits(
    train_idx: np.ndarray,
    y: np.ndarray,
    grouping: np.ndarray,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    local_y = y[train_idx]
    local_groups = grouping[train_idx]
    n_groups = len(np.unique(local_groups))
    n_splits = max(2, min(Config.NESTED_CV_FOLDS, n_groups))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for tr_local, va_local in splitter.split(np.zeros(len(train_idx)), local_y, local_groups):
        splits.append((train_idx[tr_local], train_idx[va_local]))
    return splits


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
            container, scenario, fold_train, fold_val, budget,
            Config.MI_LEAKAGE_THRESHOLD, f"{tag}_fold{fold}"
        )
        rep.update({
            "y_train": container.y[fold_train],
            "y_eval": container.y[fold_val],
            "groups_train": grouping[fold_train],
        })
        cache.append(rep)
    return cache


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
    # the "all eligible terms" step is only uncapped if it exceeds the number of
    # HPO candidate columns. Stop rather than silently capping.
    all_eligible = int(getattr(Config, "HPO_BUDGET_ALL_ELIGIBLE", 0))
    n_hpo_candidates = len(container.scenario_cols.get("S3_Phenotype", []))
    if all_eligible and all_eligible < n_hpo_candidates:
        raise ValueError(
            f"Config.HPO_BUDGET_ALL_ELIGIBLE={all_eligible} is smaller than the "
            f"{n_hpo_candidates} HPO candidate columns; raise it so the last grid step is uncapped."
        )
    logger.log(
        f"  Shared HPO-limit grid: {sorted(set(map(int, Config.HPO_BUDGET_CANDIDATES)))} "
        f"({all_eligible} = all eligible terms; {n_hpo_candidates} HPO candidate columns)"
    )
    for cap in sorted(set(map(int, Config.HPO_BUDGET_CANDIDATES))):
        budget = dict(Config.FEATURE_BUDGET)
        budget["hpo_binary"] = cap
        try:
            cache = _build_fold_cache(
                container, "S3_Phenotype", train_idx, grouping, budget, seed,
                f"{analysis_label}_S3_budget{cap}"
            )
            expected_folds = len(cache)
            if expected_folds == 0:
                raise RuntimeError("no nested folds were created")
            fold_metrics: List[Dict[str, float]] = []
            for fold, item in enumerate(cache):
                model = _fit_model_matrix(
                    "LightGBM", dict(Config.S3_BUDGET_SELECTION_LGB_PARAMS),
                    item["X_train"], item["y_train"], "S3_Phenotype", seed + fold
                )
                probability = model.predict_proba(item["X_eval"])
                prediction = np.argmax(probability, axis=1)
                fold_metrics.append(
                    compute_all_metrics(item["y_eval"], prediction, probability)
                )
            means = {
                metric: float(np.mean([m.get(metric, np.nan) for m in fold_metrics]))
                for metric in ("qwk", "mcc", "balanced_accuracy")
            }
            if not all(np.isfinite(means[name]) for name in means):
                raise RuntimeError(f"non-finite selection metrics: {means}")
            key = _metric_key(means)
            rows.append({
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
                "applied_to_scenarios": "S2_Combined|S3_Phenotype",
            })
            if key > best_key:
                best_key = key
                best_budget = budget
        except Exception as exc:
            logger.log(
                f"  Shared HPO training-CV budget {cap} failed: "
                f"{type(exc).__name__}: {exc}",
                level="WARNING",
            )
            rows.append({
                "hpo_binary_budget": cap,
                "status": "invalid",
                "n_successful_folds": 0,
                "n_required_folds": int(getattr(Config, "NESTED_CV_FOLDS", 0)),
                "failure_message": f"{type(exc).__name__}: {exc}",
                "mean_training_cv_qwk": np.nan,
                "mean_training_cv_mcc": np.nan,
                "mean_training_cv_balanced_accuracy": np.nan,
                "selection_partition": "training_only_nested_cv",
                "selection_scenario": "S3_Phenotype",
                "applied_to_scenarios": "S2_Combined|S3_Phenotype",
            })
    table = pd.DataFrame(rows)
    if best_budget is None:
        if analysis_label == "primary":
            table.to_csv(
                os.path.join(
                    Config.OUTPUT_DIR,
                    "results",
                    "shared_hpo_budget_training_cv_selection.csv",
                ),
                index=False,
            )
        raise RuntimeError(
            "No HPO budget completed the S3 training-only nested CV selection "
            f"for analysis {analysis_label}."
        )
    table["selected_for_s2_and_s3"] = (
        (table["status"] == "valid")
        & (table["hpo_binary_budget"].astype(int) == int(best_budget["hpo_binary"]))
    )
    if analysis_label == "primary":
        table.to_csv(
            os.path.join(Config.OUTPUT_DIR, "results", "shared_hpo_budget_training_cv_selection.csv"),
            index=False,
        )
    logger.log(
        f"  Shared S2/S3 HPO budget selected inside S3 training CV: "
        f"{best_budget['hpo_binary']}"
    )
    return best_budget, table


def _audit_shared_hpo_selection(
    artifacts: Dict[str, Dict[str, Any]],
    analysis_label: str,
    write_csv: bool,
) -> None:
    scenario_a, scenario_b = Config.SHARED_HPO_BUDGET_SCENARIOS
    if scenario_a not in artifacts or scenario_b not in artifacts:
        raise RuntimeError(
            f"{analysis_label}: both {scenario_a} and {scenario_b} are required "
            "for the shared HPO fairness audit."
        )

    def direct_hpo_names(scenario: str) -> set:
        return {
            str(name) for name in artifacts[scenario].get("selected_names", [])
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
            os.path.join(Config.OUTPUT_DIR, "results", "shared_hpo_feature_audit.csv"),
            index=False,
        )
    nested_match = (
        len(nested_a) == len(nested_b)
        and all(set(left) == set(right) for left, right in zip(nested_a, nested_b))
    )
    if budget_a != budget_b or selected_a != selected_b or not nested_match:
        only_a = sorted(selected_a - selected_b)[:10]
        only_b = sorted(selected_b - selected_a)[:10]
        raise RuntimeError(
            f"{analysis_label}: unfair HPO representation detected; budgets "
            f"S2={budget_a}, S3={budget_b}, nested_match={nested_match}, "
            f"only_S2={only_a}, only_S3={only_b}."
        )
    logger.log(
        f'  Shared HPO fairness check passed: S2 and S3 both use {len(selected_a)} direct HPO features at budget {budget_a}; all {len(nested_a)} nested folds also match.'
    )


def ordinal_cumulative_to_class_probs(cumulative_probs: np.ndarray) -> np.ndarray:
    cum = np.asarray(cumulative_probs, dtype=float)
    expected_cols = Config.NUM_CLASSES - 1
    if cum.ndim != 2 or cum.shape[1] != expected_cols:
        raise ValueError(
            f"Expected cumulative probabilities with shape (n, {expected_cols}); received {cum.shape}."
        )
    cum = np.clip(cum, 0.0, 1.0)
    cum = np.minimum.accumulate(cum, axis=1)
    probs = np.zeros((cum.shape[0], Config.NUM_CLASSES), dtype=float)
    probs[:, 0] = 1.0 - cum[:, 0]
    for k in range(1, expected_cols):
        probs[:, k] = cum[:, k - 1] - cum[:, k]
    probs[:, -1] = cum[:, -1]
    probs = np.clip(probs, 0.0, 1.0)
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return probs / row_sums


def compute_ordinal_sample_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=Config.NUM_CLASSES).astype(float)
    inv_freq = 1.0 / (counts + 1)
    inv_freq = inv_freq / inv_freq.sum() * Config.NUM_CLASSES
    boundary_mult = np.array(Config.ORDINAL_BOUNDARY_MULT)
    combined = inv_freq * boundary_mult
    combined = combined / combined.mean()
    return combined[y]


def make_smote_step(
    scenario_name: str,
    y_train: np.ndarray,
    enabled: Optional[bool] = None,
):
    resampling_enabled = Config.RESAMPLING_ENABLED if enabled is None else bool(enabled)
    if not resampling_enabled:
        return None
    if not HAS_IMBLEARN:
        return None
    counts = np.bincount(y_train, minlength=Config.NUM_CLASSES)
    largest = max(counts) if len(counts) else 0
    is_mechanistic = (
        ("mechanistic" in scenario_name.lower() or "S1" in scenario_name)
        and "S1b" not in scenario_name
        and ("pheno" not in scenario_name.lower())
    )
    if is_mechanistic:
        targets = {
            0: max(int(largest * 0.35), counts[0] * 3),
            1: max(int(largest * 0.5), counts[1] * 2),
        }
    else:
        targets = {
            0: max(int(largest * 0.6), counts[0] * 5),
            1: max(int(largest * 0.7), counts[1] * 2),
        }
    sampling_strategy = {cls: t for cls, t in targets.items() if counts[cls] < t}
    if not sampling_strategy:
        return None
    min_class_size = min(counts[counts > 0])
    k_neighbors = max(1, min(5, min_class_size - 1))
    try:
        if is_mechanistic:
            return (
                "resample",
                ADASYN(
                    sampling_strategy=sampling_strategy,
                    random_state=RANDOM_SEED,
                    n_neighbors=k_neighbors,
                ),
            )
        else:
            return (
                "resample",
                SMOTETomek(
                    smote=SMOTE(
                        sampling_strategy=sampling_strategy,
                        random_state=RANDOM_SEED,
                        k_neighbors=k_neighbors,
                    ),
                    random_state=RANDOM_SEED,
                ),
            )
    except Exception as e:
        logger.log(f"    Could not build resampler: {e}", level="WARNING")
        return None


def _hp_search_with_pipeline(
    estimator, param_grid, X_train, y_train, groups_train, scenario_name, n_iter=30, cv_folds=3
) -> Dict:
    smote_step = make_smote_step(scenario_name, y_train)
    n_groups = len(np.unique(groups_train))
    n_splits = max(2, min(cv_folds, n_groups))
    cv = GroupKFold(n_splits=n_splits)
    if smote_step is not None:
        pipe = ImbPipeline([smote_step, ("clf", estimator)])
        fit_params = {}
    else:
        pipe_cls = ImbPipeline if HAS_IMBLEARN else SkPipeline
        pipe = pipe_cls([("clf", estimator)])
        sw = compute_ordinal_sample_weights(y_train)
        fit_params = {"clf__sample_weight": sw}
    pg = {f"clf__{k}": v for k, v in param_grid.items()}
    search_n_jobs = 1 if Config.USE_GPU else min(16, os.cpu_count() or 1)
    gs = RandomizedSearchCV(
        pipe,
        pg,
        n_iter=n_iter,
        cv=cv,
        scoring=qwk_scorer,
        random_state=RANDOM_SEED,
        n_jobs=search_n_jobs,
        verbose=0,
        error_score="raise",
    )
    try:
        gs.fit(X_train, y_train, groups=groups_train, **fit_params)
    except (TypeError, ValueError) as exc:
        if not fit_params:
            raise
        # The hyperparameters chosen by this fallback are
        # tuned WITHOUT class weights, so the event is logged and recorded.
        estimator_name = type(estimator).__name__
        logger.log(
            f"  Class-weighted hyperparameter search failed for "
            f"{scenario_name}/{estimator_name} ({type(exc).__name__}: {exc}). "
            "Retrying WITHOUT sample weights; these tuned hyperparameters are unweighted.",
            level="WARNING",
        )
        _UNWEIGHTED_TUNING_FALLBACKS.append({
            "scenario": str(scenario_name),
            "estimator": estimator_name,
            "error": f"{type(exc).__name__}: {exc}",
        })
        gs.fit(X_train, y_train, groups=groups_train)
    best = gs.best_params_
    return {k.replace("clf__", ""): v for k, v in best.items()}


def _activate_lgb_subsampling(parameters: Dict[str, Any]) -> Dict[str, Any]:
    normalised = dict(parameters)
    try:
        fraction = float(normalised.get("subsample", normalised.get("bagging_fraction", 1.0)))
    except (TypeError, ValueError):
        fraction = 1.0
    if fraction < 1.0:
        normalised.setdefault("subsample_freq", 1)
    else:
        normalised.setdefault("subsample_freq", 0)
    return normalised


def _make_lgb_classifier(seed: int = RANDOM_SEED, **extra):
    base_kwargs = dict(
        objective="multiclass",
        num_class=Config.NUM_CLASSES,
        class_weight=None,
        random_state=seed,
        verbose=-1,
        n_jobs=getattr(Config, "MODEL_N_JOBS", 4),
        importance_type="gain",
        deterministic=True,
        force_col_wise=True,
        bagging_seed=seed,
        feature_fraction_seed=seed,
        data_random_seed=seed,
    )
    if Config.USE_GPU:
        base_kwargs["device"] = "gpu"
    base_kwargs.update(extra)
    base_kwargs = _activate_lgb_subsampling(base_kwargs)
    return lgb.LGBMClassifier(**base_kwargs)


def _make_multinomial_logistic_classifier(seed: int = RANDOM_SEED, **extra):
    logistic_overrides: Dict[str, Any] = {}
    for key, value in extra.items():
        normalised_key = key.split("logistic__", 1)[1] if key.startswith("logistic__") else key
        logistic_overrides[normalised_key] = value

    base_kwargs = dict(
        solver="lbfgs",
        C=1.0,
        fit_intercept=True,
        max_iter=10000,
        tol=1e-3,
        random_state=seed,
    )
    base_kwargs.update(logistic_overrides)
    logistic = LogisticRegression(**base_kwargs)
    return SkPipeline([
        ("standardize", StandardScaler()),
        ("logistic", logistic),
    ])


def _fit_multinomial_logistic_classifier(
    model: SkPipeline,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> SkPipeline:
    fit_kwargs: Dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["logistic__sample_weight"] = sample_weight
    model.fit(X, y, **fit_kwargs)
    return model


def _get_multinomial_logistic_estimator(model: Any) -> LogisticRegression:
    if isinstance(model, SkPipeline) and "logistic" in model.named_steps:
        estimator = model.named_steps["logistic"]
        if isinstance(estimator, LogisticRegression):
            return estimator
    if isinstance(model, LogisticRegression):
        return model
    raise TypeError(f"Unexpected multinomial logistic model type: {type(model).__name__}")


def _transform_for_multinomial_logistic(model: Any, X: np.ndarray) -> np.ndarray:
    if isinstance(model, SkPipeline) and "standardize" in model.named_steps:
        return np.asarray(model.named_steps["standardize"].transform(X), dtype=float)
    return np.asarray(X, dtype=float)


def _make_extra_trees_classifier(seed: int = RANDOM_SEED, **extra):
    base_kwargs = dict(
        n_estimators=500,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=False,
        random_state=seed,
        n_jobs=getattr(Config, "MODEL_N_JOBS", 4),
    )
    base_kwargs.update(extra)
    return ExtraTreesClassifier(**base_kwargs)


def _resample_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scenario_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    smote_step = make_smote_step(scenario_name, y_train)
    if smote_step is None:
        return X_train, y_train
    try:
        return smote_step[1].fit_resample(X_train, y_train)
    except Exception as exc:
        logger.log(
            f"    Resampling failed for {scenario_name}; using original training data: {exc}",
            level="WARNING",
        )
        return X_train, y_train


def _fit_ordinal_binary_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scenario_name: str,
    seed: int = RANDOM_SEED,
    fixed_params: Optional[Dict] = None,
) -> Tuple[List[Any], Dict[str, Any]]:
    X_fit, y_fit = _resample_training_data(X_train, y_train, scenario_name)
    bin_kwargs: Dict[str, Any] = {
        "objective": "binary",
        "random_state": seed,
        "verbose": -1,
        "n_jobs": getattr(Config, "MODEL_N_JOBS", 4),
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.5,
        "deterministic": True,
        "force_col_wise": True,
    }
    if Config.USE_GPU:
        bin_kwargs["device"] = "gpu"
    if fixed_params:
        allowed = {
            "n_estimators", "max_depth", "learning_rate", "num_leaves",
            "subsample", "colsample_bytree", "reg_alpha", "reg_lambda",
            "min_child_samples", "min_split_gain",
        }
        bin_kwargs.update({k: v for k, v in fixed_params.items() if k in allowed})

    bin_kwargs = _activate_lgb_subsampling(bin_kwargs)
    models: List[Any] = []
    for threshold in range(Config.NUM_CLASSES - 1):
        y_binary = (y_fit > threshold).astype(int)
        threshold_seed = seed + threshold
        threshold_kwargs = dict(bin_kwargs)
        threshold_kwargs.update({
            "random_state": threshold_seed,
            "bagging_seed": threshold_seed,
            "feature_fraction_seed": threshold_seed,
            "data_random_seed": threshold_seed,
        })
        model = lgb.LGBMClassifier(**threshold_kwargs)
        n_pos = int(y_binary.sum())
        n_neg = int(len(y_binary) - n_pos)
        if n_pos > 0 and n_neg > 0:
            sample_weight = np.where(y_binary == 1, n_neg / max(n_pos, 1), 1.0)
        else:
            sample_weight = np.ones(len(y_binary), dtype=float)
        model.fit(X_fit, y_binary, sample_weight=sample_weight)
        models.append(model)
    return models, bin_kwargs


def _predict_ordinal_probabilities(models: Sequence[Any], X: np.ndarray) -> np.ndarray:
    cumulative = np.column_stack([model.predict_proba(X)[:, 1] for model in models])
    return ordinal_cumulative_to_class_probs(cumulative)


def _predict_uncalibrated_candidate(result: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    model_name = result.get("model_name", "")
    if model_name == "OrdinalDecomp":
        return _predict_ordinal_probabilities(result["model"], X)
    return result["model"].predict_proba(X)


def _predict_calibrated_candidate(result: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    if result.get("calibration_status") != "success":
        return _predict_uncalibrated_candidate(result, X)
    if result["model_name"] == "OrdinalDecomp":
        models = result.get("calibrated_models")
        if not models or len(models) != Config.NUM_CLASSES - 1:
            raise RuntimeError("Successful ordinal calibration is missing one or more boundaries.")
        return _predict_ordinal_probabilities(models, X)
    calibrator = result.get("calibrated_model")
    if calibrator is None:
        raise RuntimeError("Calibration status is success but the calibrator is missing.")
    return calibrator.predict_proba(X)


def _register_final_test_evaluation(scenario_name: str, model_name: str) -> None:
    _test_evaluation_registry[scenario_name] += 1
    if Config.STRICT_SINGLE_TEST_EVALUATION and _test_evaluation_registry[scenario_name] > 1:
        raise RuntimeError(
            f"Main test set requested more than once for {scenario_name}. "
            f"The second request came from {model_name}."
        )
    logger.log(
        f'  FINAL TEST GATE OPENED ONCE: {scenario_name}/{model_name}; model and calibrator are fixed after selection.'
    )


def freeze_selected_model_and_calibrator(
    result: Dict[str, Any],
    X_val: np.ndarray,
    y_val: np.ndarray,
    scenario_name: str,
) -> Dict[str, Any]:
    """Fit calibration on validation data and fix the selected estimator."""
    if result.get("model_frozen", False) or result.get("calibrator_frozen", False):
        raise RuntimeError(f"{scenario_name} was already frozen before calibration.")
    _fit_selected_calibration(result, X_val, y_val, scenario_name)
    result["model_frozen"] = True
    result["calibrator_frozen"] = True
    result["frozen_before_any_test_evaluation"] = True
    logger.log(
        f"  selected model and calibration decision: {scenario_name}/{result['model_name']}; calibration_status={result.get('calibration_status', 'unknown')}"
    )
    return result


def evaluate_frozen_selected_model_on_test(
    result: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    scenario_name: str,
) -> Dict[str, Any]:
    """Evaluate the selected model and calibrator on the test set once."""
    model_name = result["model_name"]
    if not result.get("model_frozen", False) or not result.get("calibrator_frozen", False):
        raise RuntimeError(
            f"Refusing test evaluation for {scenario_name}/{model_name}: "
            "model and calibrator were not frozen first."
        )
    if Config.STRICT_SINGLE_TEST_EVALUATION and _test_evaluation_registry[scenario_name] >= 1:
        raise RuntimeError(
            f"Main test set requested more than once for {scenario_name}. "
            f"The repeated request came from {model_name}."
        )

    _register_final_test_evaluation(scenario_name, model_name)
    # Raw and calibrated outputs are generated in the same held-out final-analysis
    # stage. They cannot alter selection, fitting, or calibration.
    raw_test_probability = _predict_uncalibrated_candidate(result, X_test)
    calibrated_test_probability = _predict_calibrated_candidate(result, X_test)
    raw_test_prediction = np.argmax(raw_test_probability, axis=1)
    calibrated_test_prediction = np.argmax(calibrated_test_probability, axis=1)

    result["test_metrics_uncalibrated"] = compute_all_metrics(
        y_test, raw_test_prediction, raw_test_probability
    )
    result["metrics"] = compute_all_metrics(
        y_test, calibrated_test_prediction, calibrated_test_probability
    )
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
        f"    FINAL {model_name}: test MCC={result['metrics']['mcc']:.4f}, "
        f"QWK={result['metrics']['qwk']:.4f}; "
        f"uncalibrated MCC={result['test_metrics_uncalibrated']['mcc']:.4f}"
    )
    return result


def _parameter_candidates(model_name: str, seed: int) -> List[Dict[str, Any]]:
    spaces: Dict[str, Dict[str, Sequence[Any]]] = {
        "MultinomialLogisticRegression": {
            "logistic__C": [0.001, 0.01, 0.1, 1.0, 10.0],
            "logistic__fit_intercept": [True],
            "logistic__tol": [1e-3, 3e-3],
        },
        "ExtraTrees": {
            "n_estimators": [300, 500, 800, 1200],
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 10, 20, 40],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5],
            "max_features": ["sqrt", "log2", 0.5, 1.0],
            "bootstrap": [False, True],
        },
        "LightGBM": {
            "num_leaves": [15, 31, 63], "max_depth": [4, 6, 8, -1],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "n_estimators": [300, 500, 800, 1200],
            "min_child_samples": [10, 20, 40], "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0], "reg_alpha": [0.0, 0.1, 0.5],
            "reg_lambda": [0.1, 0.5, 1.0, 2.0],
        },
        "OrdinalDecomp": {
            "num_leaves": [15, 31, 63], "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "n_estimators": [300, 500, 800], "min_child_samples": [10, 20, 40],
            "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": [0.0, 0.1, 0.5], "reg_lambda": [0.1, 0.5, 1.0],
        },
    }
    if model_name not in spaces:
        raise ValueError(f"Unsupported model parameter space: {model_name}")
    n_iter = Config.NESTED_SEARCH_ITERATIONS[model_name]
    return list(ParameterSampler(spaces[model_name], n_iter=n_iter, random_state=seed))


def _apply_resampling_only(X: np.ndarray, y: np.ndarray, scenario: str, seed: int):
    if not HAS_IMBLEARN:
        return X, y
    old_seed = np.random.get_state()
    try:
        np.random.seed(seed)
        step = make_smote_step(scenario, y, enabled=True)
        if step is None:
            return X, y
        return step[1].fit_resample(X, y)
    except Exception:
        return X, y
    finally:
        np.random.set_state(old_seed)


def _fit_model_matrix(
    model_name: str,
    params: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    scenario: str,
    seed: int,
    imbalance_strategy: Optional[str] = None,
):
    strategy = imbalance_strategy or Config.IMBALANCE_STRATEGY
    X_fit, y_fit = X, y
    weights: Optional[np.ndarray] = None
    if strategy == "resampling":
        X_fit, y_fit = _apply_resampling_only(X, y, scenario, seed)
    elif strategy == "sample_weight":
        weights = compute_ordinal_sample_weights(y_fit)
    elif strategy != "none":
        raise ValueError(f"Unknown imbalance strategy: {strategy}")

    if model_name == "MultinomialLogisticRegression":
        model = _make_multinomial_logistic_classifier(seed=seed, **params)
        return _fit_multinomial_logistic_classifier(
            model, X_fit, y_fit, sample_weight=weights
        )
    if model_name == "ExtraTrees":
        model = _make_extra_trees_classifier(seed=seed, **params)
        model.fit(X_fit, y_fit, sample_weight=weights)
        return model
    if model_name == "LightGBM":
        model = _make_lgb_classifier(seed=seed, **params)
        model.fit(X_fit, y_fit, sample_weight=weights)
        return model
    if model_name == "OrdinalDecomp":
        models: List[Any] = []
        for threshold in range(Config.NUM_CLASSES - 1):
            y_binary = (y_fit > threshold).astype(int)
            threshold_seed = seed + threshold
            kwargs = dict(params)
            kwargs.update({
                "objective": "binary", "random_state": threshold_seed,
                "verbose": -1, "n_jobs": Config.MODEL_N_JOBS,
                "deterministic": True, "force_col_wise": True,
                "bagging_seed": threshold_seed,
                "feature_fraction_seed": threshold_seed,
                "data_random_seed": threshold_seed,
            })
            if Config.USE_GPU:
                kwargs["device"] = "gpu"
            kwargs = _activate_lgb_subsampling(kwargs)
            model = lgb.LGBMClassifier(**kwargs)
            binary_weights = None
            if strategy == "sample_weight":
                n_pos = int(y_binary.sum())
                n_neg = int(len(y_binary) - n_pos)
                binary_weights = np.where(y_binary == 1, n_neg / max(n_pos, 1), 1.0)
            model.fit(X_fit, y_binary, sample_weight=binary_weights)
            models.append(model)
        return models
    raise ValueError(f"Unsupported model: {model_name}")


def _predict_model_matrix(model_name: str, model: Any, X: np.ndarray) -> np.ndarray:
    if model_name == "OrdinalDecomp":
        return _predict_ordinal_probabilities(model, X)
    return model.predict_proba(X)


def _importance_table(model_name: str, model: Any, feature_names: List[str]) -> pd.DataFrame:
    try:
        if model_name == "OrdinalDecomp":
            values = np.mean([m.feature_importances_ for m in model], axis=0)
        elif model_name == "MultinomialLogisticRegression":
            fitted_logistic = _get_multinomial_logistic_estimator(model)
            values = np.mean(np.abs(np.asarray(fitted_logistic.coef_)), axis=0)
        else:
            values = np.asarray(model.feature_importances_)
        n = min(len(values), len(feature_names))
        return pd.DataFrame(
            {"feature": feature_names[:n], "importance": values[:n]}
        ).sort_values("importance", ascending=False)
    except Exception:
        return pd.DataFrame({"feature": feature_names, "importance": 0.0})


def _finite_metric_or_neg_inf(metrics: Dict[str, float], name: str) -> float:
    try:
        value = float(metrics.get(name, np.nan))
    except (TypeError, ValueError):
        return -np.inf
    return value if np.isfinite(value) else -np.inf


def _metric_key(metrics: Dict[str, float]) -> Tuple[float, float, float]:
    return (
        _finite_metric_or_neg_inf(metrics, "qwk"),
        _finite_metric_or_neg_inf(metrics, "mcc"),
        _finite_metric_or_neg_inf(metrics, "balanced_accuracy"),
    )


def validation_selection_score(metrics: Dict[str, float]) -> float:
    # Selection uses lexicographic ranks; this score is an interface field.
    return _finite_metric_or_neg_inf(metrics, "qwk")


def _tune_model_nested(
    model_name: str,
    fold_cache: List[Dict[str, Any]],
    scenario: str,
    seed: int,
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
                    model_name, params, item["X_train"], item["y_train"],
                    scenario, seed + candidate_index * 100 + fold
                )
                probability = _predict_model_matrix(model_name, model, item["X_eval"])
                prediction = np.argmax(probability, axis=1)
                fold_metrics.append(
                    compute_all_metrics(item["y_eval"], prediction, probability)
                )
            except Exception as exc:
                message = f"fold={fold}:{type(exc).__name__}:{exc}"
                failures.append(message)
                logger.log(
                    f"    Nested CV failed: {scenario}/{model_name}/candidate "
                    f"{candidate_index}/{message}",
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
            rows.append({
                **base_row,
                "status": "invalid_fold_failure",
                "mean_qwk": np.nan,
                "mean_mcc": np.nan,
                "mean_balanced_accuracy": np.nan,
            })
            continue

        means = {
            metric: float(np.mean([m.get(metric, np.nan) for m in fold_metrics]))
            for metric in ("qwk", "mcc", "balanced_accuracy")
        }
        if not all(np.isfinite(means[name]) for name in means):
            rows.append({
                **base_row,
                "status": "invalid_nonfinite_metric",
                "mean_qwk": means["qwk"],
                "mean_mcc": means["mcc"],
                "mean_balanced_accuracy": means["balanced_accuracy"],
            })
            continue

        key = _metric_key(means)
        rows.append({
            **base_row,
            "status": "valid",
            "mean_qwk": means["qwk"],
            "mean_mcc": means["mcc"],
            "mean_balanced_accuracy": means["balanced_accuracy"],
        })
        if key > best_key:
            best_key = key
            best_params = dict(params)

    table = pd.DataFrame(rows)
    if best_params is None:
        n_fold_failures = int((table.get("status") == "invalid_fold_failure").sum()) if not table.empty else 0
        n_metric_failures = int((table.get("status") == "invalid_nonfinite_metric").sum()) if not table.empty else 0
        raise RuntimeError(
            f"No valid hyperparameter candidate completed all {expected_folds} nested CV folds "
            f"with finite selection metrics for {scenario}/{model_name}; "
            f"fold_failures={n_fold_failures}, metric_failures={n_metric_failures}."
        )
    return best_params, table


def _ranked_probability_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    one_hot = np.eye(Config.NUM_CLASSES)[np.asarray(y_true, dtype=int)]
    pred_cumulative = np.cumsum(y_prob, axis=1)[:, :-1]
    obs_cumulative = np.cumsum(one_hot, axis=1)[:, :-1]
    return float(np.mean(np.sum((pred_cumulative - obs_cumulative) ** 2, axis=1) / (Config.NUM_CLASSES - 1)))


def compute_all_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    m: Dict[str, Any] = {}
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    labels = np.arange(Config.NUM_CLASSES)
    m["accuracy"] = float((y_true == y_pred).mean())
    m["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    m["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    m["f1"] = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    m["f1_macro"] = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    m["precision"] = float(
        precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    )
    m["recall"] = float(
        recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    )
    per_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    for i, name in enumerate(["Mild", "Moderate", "Severe", "Profound"]):
        m[f"f1_{name}"] = float(per_f1[i])

    if y_prob.ndim == 2 and y_prob.shape[1] == Config.NUM_CLASSES:
        try:
            m["auc_roc"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
            )
            m["auc_roc_macro"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
        except Exception as exc:
            logger.log(
                f"  AUC-ROC undefined: {type(exc).__name__}: {exc}", level="WARNING"
            )
            m["auc_roc"] = np.nan
            m["auc_roc_macro"] = np.nan
    else:
        logger.log(
            "  AUC-ROC undefined: probability matrix does not contain all configured classes.",
            level="WARNING",
        )
        m["auc_roc"] = np.nan
        m["auc_roc_macro"] = np.nan

    m["mae"] = float(np.abs(y_true - y_pred).mean())
    try:
        qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
        if not np.isfinite(qwk):
            raise ValueError("quadratic weighted kappa returned a non-finite value")
        m["qwk"] = qwk
    except Exception as exc:
        logger.log(f"  QWK undefined: {type(exc).__name__}: {exc}", level="WARNING")
        m["qwk"] = np.nan
    m["adjacent_accuracy"] = float((np.abs(y_true - y_pred) <= 1).mean())

    # Ordinal ranking quality. On a four-class ordinal target with an irreducible
    # within-gene noise floor, multiclass MCC understates a model that orders
    # severity correctly but places the thresholds imperfectly. Spearman rho is
    # reported on the expected-severity score when probabilities are available and
    # falls back to the hard prediction otherwise. Per-threshold AUCs report the
    # same signal at each ordinal cut.
    try:
        if y_prob.ndim == 2 and y_prob.shape[1] == Config.NUM_CLASSES:
            ordinal_score = y_prob @ np.arange(Config.NUM_CLASSES, dtype=float)
        else:
            ordinal_score = y_pred.astype(float)
        rho = spearmanr(ordinal_score, y_true).statistic
        m["spearman_rho"] = float(rho) if np.isfinite(rho) else np.nan
    except Exception as exc:
        logger.log(
            f"  Spearman rho undefined: {type(exc).__name__}: {exc}", level="WARNING"
        )
        m["spearman_rho"] = np.nan

    for threshold in range(1, Config.NUM_CLASSES):
        key = f"auc_threshold_ge{threshold}"
        try:
            binary_true = (y_true >= threshold).astype(int)
            if binary_true.min() == binary_true.max():
                m[key] = np.nan
                continue
            if y_prob.ndim == 2 and y_prob.shape[1] == Config.NUM_CLASSES:
                binary_score = y_prob[:, threshold:].sum(axis=1)
            else:
                binary_score = y_pred.astype(float)
            m[key] = float(roc_auc_score(binary_true, binary_score))
        except Exception:
            m[key] = np.nan

    m["pred_distribution"] = str(
        np.bincount(y_pred.astype(int), minlength=Config.NUM_CLASSES).tolist()
    )
    m["true_distribution"] = str(
        np.bincount(y_true.astype(int), minlength=Config.NUM_CLASSES).tolist()
    )
    metrics = m
    probability = np.asarray(y_prob, dtype=float)
    if probability.ndim == 2 and probability.shape[1] == Config.NUM_CLASSES:
        one_hot = np.eye(Config.NUM_CLASSES)[np.asarray(y_true, dtype=int)]
        metrics["multiclass_brier"] = float(np.mean(np.sum((probability - one_hot) ** 2, axis=1)))
        metrics["ranked_probability_score"] = _ranked_probability_score(y_true, probability)
    else:
        metrics["multiclass_brier"] = np.nan
        metrics["ranked_probability_score"] = np.nan
    return metrics


def _choose_calibration_method(y: np.ndarray) -> str:
    values = np.asarray(y, dtype=int)
    classes = np.unique(values)
    for class_value in classes:
        positives = int(np.sum(values == class_value))
        negatives = int(len(values) - positives)
        if positives < Config.CALIBRATION_MIN_POSITIVES or negatives < Config.CALIBRATION_MIN_NEGATIVES:
            return Config.CALIBRATION_METHOD_SMALL
    return Config.CALIBRATION_METHOD_LARGE


def calibrate_prefit_model(
    model, X_val, y_val, label: str = "model", method: Optional[str] = None
):
    selected_method = method or _choose_calibration_method(np.asarray(y_val))
    try:
        if FrozenEstimator is not None:
            calibrator = CalibratedClassifierCV(FrozenEstimator(model), method=selected_method)
        else:
            calibrator = CalibratedClassifierCV(model, method=selected_method, cv="prefit")
        calibrator.fit(X_val, y_val)
        logger.log(f"    Calibration OK ({selected_method}): {label}")
        return calibrator
    except Exception as exc:
        logger.log(
            f"    Calibration failed for {label} with {selected_method}: {type(exc).__name__}: {exc}",
            level="WARNING",
        )
        return None


def _fit_selected_calibration(
    result: Dict[str, Any], X_val: np.ndarray, y_val: np.ndarray, scenario_name: str
) -> None:
    model_name = result["model_name"]
    result["calibration_partition"] = "validation_after_model_selection"
    result["base_model_frozen_before_calibration"] = True

    if model_name == "OrdinalDecomp":
        attempted_methods: List[str] = []
        attempted_calibrators: List[Optional[Any]] = []
        failures: List[str] = []
        for threshold, model in enumerate(result["model"]):
            binary_y = (y_val > threshold).astype(int)
            method = _choose_calibration_method(binary_y)
            attempted_methods.append(method)
            calibrator = calibrate_prefit_model(
                model,
                X_val,
                binary_y,
                label=f"{scenario_name}/{model_name}/threshold_{threshold}",
                method=method,
            )
            attempted_calibrators.append(calibrator)
            if calibrator is None:
                failures.append(f"threshold_{threshold}")

        result["calibration_method_attempted"] = "|".join(attempted_methods)
        if not failures:
            result["calibrated_models"] = list(attempted_calibrators)
            result["calibrated_model"] = None
            result["calibration_method"] = result["calibration_method_attempted"]
            result["calibration_success"] = True
            result["calibration_status"] = "success"
            result["calibration_failure_details"] = ""
        else:
            # Never mix calibrated and raw cumulative thresholds.
            result["calibrated_models"] = None
            result["calibrated_model"] = None
            result["calibration_method"] = "none"
            result["calibration_success"] = False
            result["calibration_status"] = "uncalibrated_fallback"
            result["calibration_failure_details"] = "|".join(failures)
            logger.log(
                f"    Atomic ordinal calibration fallback for {scenario_name}/{model_name}; "
                f"failed boundaries={result['calibration_failure_details']}",
                level="WARNING",
            )
    else:
        method = _choose_calibration_method(y_val)
        result["calibration_method_attempted"] = method
        calibrator = calibrate_prefit_model(
            result["model"], X_val, y_val,
            label=f"{scenario_name}/{model_name}", method=method
        )
        result["calibrated_model"] = calibrator
        result["calibrated_models"] = None
        result["calibration_success"] = calibrator is not None
        if calibrator is not None:
            result["calibration_method"] = method
            result["calibration_status"] = "success"
            result["calibration_failure_details"] = ""
        else:
            result["calibration_method"] = "none"
            result["calibration_status"] = "uncalibrated_fallback"
            result["calibration_failure_details"] = "model_calibration_failed"


def _available_candidate_names() -> List[str]:
    enabled = {
        "MultinomialLogisticRegression": bool(Config.MULTINOMIAL_LOGISTIC_ENABLED),
        "ExtraTrees": bool(Config.EXTRA_TREES_ENABLED),
        "LightGBM": True,
        "OrdinalDecomp": bool(Config.ORDINAL_DECOMPOSITION_ENABLED),
    }
    return [name for name in Config.CANDIDATE_MODEL_ORDER if enabled.get(name, False)]


def _fit_final_candidate(
    model_name: str,
    params: Dict[str, Any],
    representation: Dict[str, Any],
    y_train: np.ndarray,
    y_val: np.ndarray,
    scenario: str,
    seed: int,
) -> Dict[str, Any]:
    model = _fit_model_matrix(
        model_name, params, representation["X_train"], y_train, scenario, seed
    )
    if model_name == "MultinomialLogisticRegression":
        fitted_logistic = _get_multinomial_logistic_estimator(model)
        realised_iterations = int(np.max(np.asarray(fitted_logistic.n_iter_)))
        logger.log(
            f"    Multinomial logistic converged in {realised_iterations} iterations "
            f"(limit={fitted_logistic.max_iter}, tol={fitted_logistic.tol})."
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
        (name, result) for name, result in candidates.items()
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


def _run_workflow_for_split(
    container: DataContainer,
    development_idx: np.ndarray,
    test_idx: np.ndarray,
    grouping: np.ndarray,
    analysis_label: str,
    seed: int,
    primary: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    train_idx, val_idx = _split_development_train_validation(
        container, development_idx, grouping, seed
    )
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
            container, scenario, train_idx, grouping, budget, seed,
            f"{analysis_label}_{scenario}_nested"
        )
        final_representation = _fit_scenario_representation(
            container, scenario, train_idx, val_idx, budget,
            Config.MI_LEAKAGE_THRESHOLD, f"{analysis_label}_{scenario}_final"
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
                    model_name, params, final_representation, y_train, y_val,
                    scenario, seed + model_index
                )
                candidate_results[model_name] = result
                all_results[scenario]["all"][model_name] = result
                if primary:
                    results_collector.add_feature_importance(
                        f"{scenario}_{model_name}", result["importance"]
                    )
            except Exception as exc:
                logger.log(f"  {scenario}/{model_name} failed: {exc}", level="WARNING")
        if not candidate_results:
            raise RuntimeError(
                f"{analysis_label}/{scenario}: all candidate models failed before selection."
            )
        selected_name, selected_result = _select_candidate_lexicographically(
            candidate_results, candidate_names
        )
        for model_name, result in candidate_results.items():
            selected = model_name == selected_name
            result["selected_on_validation"] = selected
            result["selection_partition"] = "outer_validation"
            result["selection_rule"] = Config.MODEL_SELECTION_RULE
            selection_rows.append({
                "Analysis": analysis_label, "Scenario": scenario, "Model": model_name,
                "selected_on_validation": selected,
                "validation_qwk": result["val_metrics"].get("qwk", np.nan),
                "validation_mcc": result["val_metrics"].get("mcc", np.nan),
                "validation_balanced_accuracy": result["val_metrics"].get("balanced_accuracy", np.nan),
                "selection_rule": Config.MODEL_SELECTION_RULE,
                "test_evaluated": False,
            })
        freeze_selected_model_and_calibrator(
            selected_result, final_representation["X_eval"], y_val,
            f"{analysis_label}:{scenario}"
        )
        if primary:
            final_representation["preprocessor"].save_audit(
                scenario, final_representation["runtime_mi_drops"]
            )
        local_artifacts[scenario] = {
            **final_representation,
            # Compatibility aliases used by the existing reporting/bundle code.
            "sel_mask": final_representation["selected_mask"],
            "sel_names": final_representation["selected_names"],
            "X_val_full": final_representation["X_eval_full"],
            "X_val": final_representation["X_eval"],
            "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
            "y_train": y_train, "y_val": y_val,
            "groups_train": grouping[train_idx],
            "selected_model": selected_name,
            "selection_budget": budget,
            "budget_selection_table": budget_table,
            "nested_hpo_sets": [
                [
                    str(name) for name in fold_item.get("selected_names", [])
                    if str(name).startswith("hpo_")
                ]
                for fold_item in fold_cache
            ],
            "tuning_table": pd.concat(tuning_tables, ignore_index=True) if tuning_tables else pd.DataFrame(),
        }
        if primary and not local_artifacts[scenario]["tuning_table"].empty:
            local_artifacts[scenario]["tuning_table"].to_csv(
                os.path.join(Config.OUTPUT_DIR, "results", f"nested_cv_tuning_{scenario}.csv"),
                index=False,
            )

    _audit_shared_hpo_selection(local_artifacts, analysis_label, write_csv=primary)

    # Every selected model/calibrator is fixed after selection before any test matrix is created.
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
                f"{scenario}_{selected_name}", y_test,
                selected_result["predictions"]["y_pred"],
                selected_result["predictions"]["y_prob"],
            )

    if primary:
        write_split_summary(container, train_idx, val_idx, test_idx, include_test_labels=True)
    return all_results, {
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
        "selection_rows": selection_rows, "artifacts": local_artifacts,
    }


def _gene_disease_component_groups(container: DataContainer) -> np.ndarray:
    metadata = container.metadata.reset_index(drop=True)
    parent: Dict[str, str] = {}
    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for _, row in metadata.iterrows():
        union(f"g:{row.get('gene_symbol', '')}", f"d:{row.get('mondo_id', '')}")
    roots: Dict[str, int] = {}
    output = []
    for _, row in metadata.iterrows():
        root = find(f"g:{row.get('gene_symbol', '')}")
        roots.setdefault(root, len(roots))
        output.append(roots[root])
    return np.asarray(output)


def run_additional_generalisation_evaluations(container: DataContainer) -> pd.DataFrame:
    if not Config.RUN_ADDITIONAL_GENERALISATION:
        return pd.DataFrame()
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
                container, development_idx, test_idx, grouping,
                mode, RANDOM_SEED + 500, primary=False
            )
            _generalisation_results[mode] = {"results": results, "details": details}
            _write_split_identifiers(
                container, details["train_idx"], details["val_idx"], details["test_idx"], mode
            )
            for scenario, model_name in _pick_best_model_per_scenario(results).items():
                result = results[scenario]["all"][model_name]
                metrics = result.get("metrics", {})
                summary_rows.append({
                    "generalisation_mode": mode, "Scenario": scenario, "Model": model_name,
                    "mcc": metrics.get("mcc", np.nan), "qwk": metrics.get("qwk", np.nan),
                    "balanced_accuracy": metrics.get("balanced_accuracy", np.nan),
                    "ranked_probability_score": metrics.get("ranked_probability_score", np.nan),
                    "n_test": len(details["test_idx"]),
                })
        except Exception as exc:
            logger.log(f"  {mode} evaluation skipped/failed: {exc}", level="WARNING")
    dataframe = pd.DataFrame(summary_rows)
    dataframe.to_csv(
        os.path.join(Config.OUTPUT_DIR, "results", "generalisation_evaluations.csv"),
        index=False,
    )
    return dataframe


def run_experiments(container: DataContainer):
    """Run the primary nested workflow, save its selected results, and perform configured generalisation checks."""
    logger.section('RUNNING NESTED complete EXPERIMENTS')
    _fold_artifacts.clear()
    _test_evaluation_registry.clear()
    _high_mi_audit_records.clear()
    _generalisation_results.clear()
    all_results, details = _run_workflow_for_split(
        container, container.train_idx, container.test_idx, container.groups,
        "primary", RANDOM_SEED, primary=True
    )
    _fold_artifacts.update(details["artifacts"])
    selection_df = pd.DataFrame(details["selection_rows"])
    selection_df.to_csv(
        os.path.join(Config.OUTPUT_DIR, "results", "model_selection_validation.csv"),
        index=False,
    )
    results_collector.save(os.path.join(Config.OUTPUT_DIR, "results", "all_results.csv"))
    with open(os.path.join(Config.OUTPUT_DIR, "results", "all_results_full.pkl"), "wb") as handle:
        pickle.dump(all_results, handle)
    if Config.RUN_ADDITIONAL_GENERALISATION:
        run_additional_generalisation_evaluations(container)
    else:
        logger.log("  Additional generalisation evaluations disabled by configuration.")
    return all_results


def _train_candidate_dispatch(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    scenario_name: str,
    seed: int = RANDOM_SEED,
    fixed_params: Optional[Dict] = None,
) -> Optional[Dict[str, Any]]:
    kwargs = dict(
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names,
        scenario_name=scenario_name,
        seed=seed,
        fixed_params=fixed_params,
    )
    if model_name == "MultinomialLogisticRegression":
        return train_multinomial_logistic_regression(**kwargs)
    if model_name == "ExtraTrees":
        return train_extra_trees(**kwargs)
    if model_name == "LightGBM":
        return train_lightgbm(**kwargs)
    if model_name == "OrdinalDecomp":
        return train_ordinal_decomposition(**kwargs)
    raise ValueError(f"Unsupported candidate model: {model_name}")


def train_lightgbm(
    X_train,
    y_train,
    groups_train,
    X_val,
    y_val,
    feature_names,
    scenario_name="default",
    seed: int = RANDOM_SEED,
    fixed_params: Optional[Dict] = None,
):
    """Tune on grouped training CV, fit on training only, score on validation."""
    logger.log("  Training LightGBM candidate (test set unavailable)...")
    if fixed_params is None:
        param_grid = {
            "max_depth": [3, 5, 7, 10, -1],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "n_estimators": [300, 500, 800, 1200],
            "num_leaves": [15, 31, 63, 127],
            "min_child_samples": [5, 10, 20],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
            "reg_alpha": [0, 0.1, 0.5, 1.0],
            "reg_lambda": [0, 0.1, 0.5, 1.0],
        }
        base = _make_lgb_classifier(seed=seed)
        best_params = _hp_search_with_pipeline(
            base, param_grid, X_train, y_train, groups_train, scenario_name, n_iter=40, cv_folds=5
        )
        logger.log(f"    Best training-CV params: {best_params}")
    else:
        best_params = dict(fixed_params)

    X_fit, y_fit = _resample_training_data(X_train, y_train, scenario_name)
    final = _make_lgb_classifier(seed=seed, **best_params)
    final.fit(X_fit, y_fit, sample_weight=compute_ordinal_sample_weights(y_fit))

    val_probability = final.predict_proba(X_val)
    val_prediction = np.argmax(val_probability, axis=1)
    val_metrics = compute_all_metrics(y_val, val_prediction, val_probability)
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": final.feature_importances_}
    ).sort_values("importance", ascending=False)
    logger.log(
        f"    LGB validation MCC={val_metrics['mcc']:.4f}, QWK={val_metrics['qwk']:.4f}"
    )
    return {
        "model_name": "LightGBM",
        "model": final,
        "calibrated_model": None,
        "val_metrics": val_metrics,
        "validation_selection_score": validation_selection_score(val_metrics),
        "best_params": best_params,
        "val_predictions": {"y_pred": val_prediction, "y_prob": val_probability},
        "importance": importance_df,
        "test_evaluated": False,
    }


def train_multinomial_logistic_regression(
    X_train,
    y_train,
    groups_train,
    X_val,
    y_val,
    feature_names,
    scenario_name="default",
    seed: int = RANDOM_SEED,
    fixed_params: Optional[Dict] = None,
):
    """Tune and evaluate the regularised multinomial linear baseline."""
    logger.log("  Training Multinomial Logistic Regression candidate (test set unavailable)...")
    if fixed_params is None:
        param_grid = {
            "logistic__C": [0.001, 0.01, 0.1, 1.0, 10.0],
            "logistic__fit_intercept": [True],
            "logistic__tol": [1e-3, 3e-3],
        }
        base = _make_multinomial_logistic_classifier(seed=seed)
        best_params = _hp_search_with_pipeline(
            base, param_grid, X_train, y_train, groups_train,
            scenario_name, n_iter=16, cv_folds=3
        )
        logger.log(f"    Best training-CV params: {best_params}")
    else:
        best_params = dict(fixed_params)

    X_fit, y_fit = _resample_training_data(X_train, y_train, scenario_name)
    final = _make_multinomial_logistic_classifier(seed=seed, **best_params)
    _fit_multinomial_logistic_classifier(
        final, X_fit, y_fit,
        sample_weight=compute_ordinal_sample_weights(y_fit),
    )
    val_probability = final.predict_proba(X_val)
    val_prediction = np.argmax(val_probability, axis=1)
    val_metrics = compute_all_metrics(y_val, val_prediction, val_probability)
    fitted_logistic = _get_multinomial_logistic_estimator(final)
    coefficient_importance = np.mean(np.abs(np.asarray(fitted_logistic.coef_)), axis=0)
    importance_df = pd.DataFrame(
        {"feature": feature_names[: len(coefficient_importance)],
         "importance": coefficient_importance[: len(feature_names)]}
    ).sort_values("importance", ascending=False)
    logger.log(
        f"    Multinomial logistic validation MCC={val_metrics['mcc']:.4f}, "
        f"QWK={val_metrics['qwk']:.4f}"
    )
    return {
        "model_name": "MultinomialLogisticRegression",
        "model": final,
        "calibrated_model": None,
        "val_metrics": val_metrics,
        "validation_selection_score": validation_selection_score(val_metrics),
        "best_params": best_params,
        "val_predictions": {"y_pred": val_prediction, "y_prob": val_probability},
        "importance": importance_df,
        "test_evaluated": False,
    }


def train_extra_trees(
    X_train,
    y_train,
    groups_train,
    X_val,
    y_val,
    feature_names,
    scenario_name="default",
    seed: int = RANDOM_SEED,
    fixed_params: Optional[Dict] = None,
):
    """Tune and evaluate the nonlinear randomized bagging baseline."""
    logger.log("  Training Extra Trees candidate (test set unavailable)...")
    if fixed_params is None:
        param_grid = {
            "n_estimators": [300, 500, 800, 1200],
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 10, 20, 40],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5],
            "max_features": ["sqrt", "log2", 0.5, 1.0],
            "bootstrap": [False, True],
        }
        base = _make_extra_trees_classifier(seed=seed)
        best_params = _hp_search_with_pipeline(
            base, param_grid, X_train, y_train, groups_train,
            scenario_name, n_iter=20, cv_folds=3
        )
        logger.log(f"    Best training-CV params: {best_params}")
    else:
        best_params = dict(fixed_params)

    X_fit, y_fit = _resample_training_data(X_train, y_train, scenario_name)
    final = _make_extra_trees_classifier(seed=seed, **best_params)
    final.fit(X_fit, y_fit, sample_weight=compute_ordinal_sample_weights(y_fit))
    val_probability = final.predict_proba(X_val)
    val_prediction = np.argmax(val_probability, axis=1)
    val_metrics = compute_all_metrics(y_val, val_prediction, val_probability)
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": final.feature_importances_}
    ).sort_values("importance", ascending=False)
    logger.log(
        f"    Extra Trees validation MCC={val_metrics['mcc']:.4f}, "
        f"QWK={val_metrics['qwk']:.4f}"
    )
    return {
        "model_name": "ExtraTrees",
        "model": final,
        "calibrated_model": None,
        "val_metrics": val_metrics,
        "validation_selection_score": validation_selection_score(val_metrics),
        "best_params": best_params,
        "val_predictions": {"y_pred": val_prediction, "y_prob": val_probability},
        "importance": importance_df,
        "test_evaluated": False,
    }


def train_ordinal_decomposition(
    X_train,
    y_train,
    groups_train,
    X_val,
    y_val,
    feature_names,
    scenario_name="default",
    seed: int = RANDOM_SEED,
    fixed_params: Optional[Dict] = None,
):
    """Fit ordinal threshold models on training only and score on validation."""
    del groups_train  # retained in the public signature for a common candidate API
    logger.log("  Training Ordinal Binary Decomposition candidate (test set unavailable)...")
    binary_models, bin_kwargs = _fit_ordinal_binary_models(
        X_train, y_train, scenario_name, seed=seed, fixed_params=fixed_params
    )
    val_probability = _predict_ordinal_probabilities(binary_models, X_val)
    val_prediction = np.argmax(val_probability, axis=1)
    val_metrics = compute_all_metrics(y_val, val_prediction, val_probability)
    average_importance = np.mean([model.feature_importances_ for model in binary_models], axis=0)
    importance_df = pd.DataFrame(
        {
            "feature": feature_names[: len(average_importance)],
            "importance": average_importance[: len(feature_names)],
        }
    ).sort_values("importance", ascending=False)
    logger.log(
        f"    Ordinal validation MCC={val_metrics['mcc']:.4f}, QWK={val_metrics['qwk']:.4f}"
    )
    return {
        "model_name": "OrdinalDecomp",
        "model": binary_models,
        "calibrated_models": None,
        "val_metrics": val_metrics,
        "validation_selection_score": validation_selection_score(val_metrics),
        "best_params": bin_kwargs,
        "val_predictions": {"y_pred": val_prediction, "y_prob": val_probability},
        "importance": importance_df,
        "test_evaluated": False,
    }


def evaluate_at_gene_level(all_results: Dict, container: DataContainer) -> pd.DataFrame:
    logger.subsection("Gene-level companion evaluation (no retraining)")
    if container.metadata is None:
        logger.log("  Container metadata unavailable; skipping.", level="WARNING")
        return pd.DataFrame()
    metadata = container.metadata.reset_index(drop=True)
    if "gene_symbol" not in metadata.columns:
        logger.log("  gene_symbol absent from metadata; skipping.", level="WARNING")
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for scenario, strata in all_results.items():
        for stratum, models in strata.items():
            for model_name, result in models.items():
                if "predictions" not in result or "y_true" not in result:
                    continue
                y_true = np.asarray(result["y_true"])
                y_pred = np.asarray(result["predictions"]["y_pred"])

                artifact = _fold_artifacts.get(scenario, {})
                test_idx = artifact.get("test_idx", container.test_idx)
                if test_idx is None:
                    continue
                test_idx = np.asarray(test_idx, dtype=int)
                if len(test_idx) != len(y_true):
                    logger.log(
                        f"  {scenario}/{model_name}: test index length {len(test_idx)} "
                        f"does not match prediction length {len(y_true)}; skipping.",
                        level="WARNING",
                    )
                    continue

                frame = pd.DataFrame({
                    "gene_symbol": metadata.iloc[test_idx]["gene_symbol"].to_numpy(),
                    "y_true": y_true.astype(float),
                    "y_pred": y_pred.astype(float),
                })
                grouped = frame.groupby("gene_symbol", sort=False).mean()
                gene_true = np.clip(
                    np.rint(grouped["y_true"].to_numpy()), 0, Config.NUM_CLASSES - 1
                ).astype(int)
                gene_pred = np.clip(
                    np.rint(grouped["y_pred"].to_numpy()), 0, Config.NUM_CLASSES - 1
                ).astype(int)
                if len(np.unique(gene_true)) < 2:
                    continue

                try:
                    qwk = float(
                        cohen_kappa_score(gene_true, gene_pred, weights="quadratic")
                    )
                except Exception:
                    qwk = np.nan
                rows.append({
                    "Scenario": scenario,
                    "Stratum": stratum,
                    "Model": model_name,
                    "evaluation_unit": "gene",
                    "aggregation": "rounded_mean_of_pair_truth_and_pair_prediction",
                    "retrained": False,
                    "n_genes": int(len(gene_true)),
                    "n_pairs": int(len(y_true)),
                    "accuracy": float((gene_true == gene_pred).mean()),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(gene_true, gene_pred)
                    ),
                    "mcc": float(matthews_corrcoef(gene_true, gene_pred)),
                    "qwk": qwk,
                    "mae": float(np.abs(gene_true - gene_pred).mean()),
                    "adjacent_accuracy": float(
                        (np.abs(gene_true - gene_pred) <= 1).mean()
                    ),
                    "frac_pairs_in_multi_severity_genes": float(
                        frame.groupby("gene_symbol")["y_true"]
                        .transform("nunique")
                        .gt(1)
                        .mean()
                    ),
                })


    gene_level_columns = [
        "Scenario", "Stratum", "Model", "evaluation_unit", "aggregation",
        "retrained", "n_genes", "n_pairs", "accuracy", "balanced_accuracy",
        "mcc", "qwk", "mae", "adjacent_accuracy",
        "frac_pairs_in_multi_severity_genes",
    ]
    dataframe = pd.DataFrame(rows, columns=gene_level_columns)
    output_path = os.path.join(
        Config.OUTPUT_DIR, "results", "gene_level_evaluation.csv"
    )
    dataframe.to_csv(output_path, index=False)
    logger.log(f"  Wrote {output_path} ({len(dataframe)} rows)")
    for _, row in dataframe.iterrows():
        logger.log(
            f"    {row['Scenario']:24s} {row['Model']:28s} "
            f"gene-level MCC={row['mcc']:.3f} QWK={row['qwk']:.3f} "
            f"BalAcc={row['balanced_accuracy']:.3f} (n_genes={row['n_genes']})"
        )
    return dataframe


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
    global_df.to_csv(
        os.path.join(Config.OUTPUT_DIR, "results", "error_decomposition.csv"), index=False
    )
    per_class_df.to_csv(
        os.path.join(Config.OUTPUT_DIR, "results", "error_decomposition_per_class.csv"), index=False
    )
    try:
        best_per_scenario = (
            global_df.sort_values("frac_correct", ascending=False).groupby("Scenario").head(1)
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(best_per_scenario))
        ax.bar(x, best_per_scenario["frac_correct"], label="Correct", color="#2ca02c")
        ax.bar(
            x,
            best_per_scenario["frac_adjacent"],
            bottom=best_per_scenario["frac_correct"],
            label="Adjacent error",
            color="#ffbb78",
        )
        ax.bar(
            x,
            best_per_scenario["frac_non_adjacent"],
            bottom=best_per_scenario["frac_correct"] + best_per_scenario["frac_adjacent"],
            label="Non-adjacent error",
            color="#d62728",
        )
        labels = [f"{r['Scenario']}\n({r['Model']})" for _, r in best_per_scenario.iterrows()]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Fraction of test pairs")
        ax.set_title("Error decomposition per scenario (best model, global)")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(
            os.path.join(Config.OUTPUT_DIR, "figures", "diagnostics", "error_decomposition.png"),
            dpi=Config.FIGURE_DPI,
        )
        plt.close()
    except Exception as e:
        logger.log(f"  Could not plot global error decomposition: {e}", level="WARNING")
    try:
        best_models = _pick_best_model_per_scenario(all_results)
        scenarios_ordered = list(best_models.keys())
        fig, axes = plt.subplots(
            1, len(scenarios_ordered), figsize=(4 * max(len(scenarios_ordered), 1), 5), sharey=True
        )
        if len(scenarios_ordered) == 1:
            axes = [axes]
        for ax, scenario in zip(axes, scenarios_ordered):
            best_name = best_models[scenario]
            sub = per_class_df[
                (per_class_df["Scenario"] == scenario) & (per_class_df["Model"] == best_name)
            ]
            sub = sub.set_index("true_class").reindex(severity_names)
            x = np.arange(len(severity_names))
            ax.bar(x, sub["frac_correct"].fillna(0), label="Correct", color="#2ca02c")
            ax.bar(
                x,
                sub["frac_adjacent"].fillna(0),
                bottom=sub["frac_correct"].fillna(0),
                label="Adjacent",
                color="#ffbb78",
            )
            ax.bar(
                x,
                sub["frac_non_adjacent"].fillna(0),
                bottom=sub["frac_correct"].fillna(0) + sub["frac_adjacent"].fillna(0),
                label="Non-adj.",
                color="#d62728",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(severity_names, rotation=30, ha="right")
            ax.set_title(f"{scenario}\n({best_name})", fontsize=9)
            ax.set_ylim(0, 1)
        axes[0].set_ylabel("Fraction within true class")
        axes[-1].legend(loc="lower right", fontsize=7)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                Config.OUTPUT_DIR, "figures", "diagnostics", "error_decomposition_per_class.png"
            ),
            dpi=Config.FIGURE_DPI,
        )
        plt.close()
    except Exception as e:
        logger.log(f"  Could not plot per-class error decomposition: {e}", level="WARNING")
    return (global_df, per_class_df)


def _calibration_bin_table(
    y_true_binary: np.ndarray,
    y_prob_class: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> pd.DataFrame:
    y = np.asarray(y_true_binary, dtype=float)
    p = np.asarray(y_prob_class, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y = y[valid]
    p = np.clip(p[valid], 0.0, 1.0)
    if len(y) == 0:
        return pd.DataFrame(
            columns=[
                "bin",
                "bin_lower",
                "bin_upper",
                "n_bin",
                "mean_predicted_probability",
                "observed_frequency",
            ]
        )
    if strategy == "quantile":
        edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 3:
            edges = np.linspace(0.0, 1.0, n_bins + 1)
        else:
            edges[0] = 0.0
            edges[-1] = 1.0
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")
    bin_idx = np.searchsorted(edges[1:-1], p, side="right")
    rows = []
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        rows.append(
            {
                "bin": b + 1,
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "n_bin": n_bin,
                "mean_predicted_probability": float(p[mask].mean()),
                "observed_frequency": float(y[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def _compute_ece_one_class(
    y_true_binary: np.ndarray,
    y_prob_class: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> float:
    bins = _calibration_bin_table(y_true_binary, y_prob_class, n_bins=n_bins, strategy=strategy)
    if bins.empty:
        return float("nan")
    total = bins["n_bin"].sum()
    gaps = np.abs(bins["observed_frequency"] - bins["mean_predicted_probability"])
    return float(np.sum(bins["n_bin"] / total * gaps))


def _brier_one_class(y_true_binary: np.ndarray, y_prob_class: np.ndarray) -> float:
    y = np.asarray(y_true_binary, dtype=float)
    p = np.asarray(y_prob_class, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    if not np.any(valid):
        return float("nan")
    return float(np.mean((p[valid] - y[valid]) ** 2))


def run_mi_threshold_sensitivity(
    container: DataContainer,
    all_results: Dict,
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Validation-only sensitivity to the minimum positive-MI selection threshold."""
    if not Config.RUN_MI_SENSITIVITY:
        return pd.DataFrame()
    if thresholds is None:
        thresholds = Config.MI_SENSITIVITY_THRESHOLDS
    logger.section(
        f"Minimum-MI selection sensitivity on validation ({len(thresholds)} values)"
    )
    best_models = _pick_best_model_per_scenario(all_results)
    rows: List[Dict[str, Any]] = []
    for scenario in [
        value for value in ["S2_Combined", "S3_Phenotype"] if value in best_models
    ]:
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
                    family_lookup=artifacts.get(
                        "family_lookup", container.feature_family_lookup
                    ),
                    budget=artifacts.get("selection_budget", Config.FEATURE_BUDGET),
                    mi_leakage_threshold=None,
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
                rows.append({
                    "Scenario": scenario,
                    "Model": model_name,
                    "evaluation_partition": "validation_only",
                    "minimum_mi_threshold": float(threshold),
                    "n_features_selected": int(mask.sum()),
                    "mcc": metrics.get("mcc", np.nan),
                    "qwk": metrics.get("qwk", np.nan),
                    "balanced_accuracy": metrics.get("balanced_accuracy", np.nan),
                })
                logger.log(
                    f"  {scenario} minimum MI={float(threshold):.4f}: "
                    f"validation MCC={metrics.get('mcc', 0):.3f}, "
                    f"QWK={metrics.get('qwk', 0):.3f}"
                )
            except Exception as exc:
                logger.log(
                    f"  {scenario} minimum MI={threshold} failed: {exc}",
                    level="WARNING",
                )
    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(
        os.path.join(Config.OUTPUT_DIR, "results", "mi_threshold_sensitivity.csv"),
        index=False,
    )
    return dataframe


def run_budget_sensitivity(
    container: DataContainer,
    all_results: Dict,
    hpo_budgets: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Validation-only sensitivity to a shared direct-HPO feature budget."""
    if not Config.RUN_BUDGET_SENSITIVITY:
        return pd.DataFrame()
    if hpo_budgets is None:
        hpo_budgets = list(Config.HPO_BUDGET_CANDIDATES)
    hpo_budgets = sorted({int(value) for value in hpo_budgets if int(value) > 0})
    logger.section(
        f"Shared HPO-budget sensitivity on validation ({len(hpo_budgets)} values)"
    )
    best_models = _pick_best_model_per_scenario(all_results)
    rows: List[Dict[str, Any]] = []

    for scenario in [s for s in ["S2_Combined", "S3_Phenotype"] if s in best_models]:
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
                    mi_leakage_threshold=Config.MI_LEAKAGE_THRESHOLD,
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
                    f"  {scenario}/{model_name} HPO budget {hpo_budget}: "
                    f"validation MCC={metrics.get('mcc', 0):.3f}, QWK={metrics.get('qwk', 0):.3f}"
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
            "mcc", "qwk", "balanced_accuracy", "f1", "auc_roc",
            "ranked_probability_score", "multiclass_brier", "spearman_rho",
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
            logger.log(
                f"  Bootstrap replicate failed: {type(exc).__name__}: {exc}",
                level="WARNING",
            )
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
            result["y_true"], result["predictions"]["y_pred"],
            result["predictions"]["y_prob"], n_bootstrap=Config.N_BOOTSTRAP,
            groups=result.get("test_gene_groups", result.get("test_groups")),
        )
        row = {"Scenario": scenario, "Stratum": "all", "Model": model_name,
               "bootstrap_unit": "gene_cluster"}
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


def _paired_contrast_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric == "mcc":
        return float(matthews_corrcoef(y_true, y_pred))
    if metric == "qwk":
        return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    raise ValueError(f"Unsupported paired-contrast metric: {metric}")


def run_paired_scenario_contrasts(
    all_results: Dict, container: DataContainer
) -> pd.DataFrame:
    """Paired gene-cluster bootstrap contrasts between adjacent scenarios."""
    logger.subsection("Paired between-scenario gene-cluster bootstrap contrasts")
    output_path = os.path.join(
        Config.OUTPUT_DIR, "results", "paired_scenario_contrasts.csv"
    )
    selected = _pick_best_model_per_scenario(all_results)

    predictions: Dict[str, Dict[str, np.ndarray]] = {}
    models: Dict[str, str] = {}
    groups_by_scenario: Dict[str, np.ndarray] = {}
    for scenario, model_name in selected.items():
        payload = results_collector.predictions.get(f"{scenario}_{model_name}")
        if not payload:
            logger.log(
                f'  No stored held-out test predictions for {scenario}_{model_name}; that scenario is excluded from the paired contrasts.',
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
            test_idx = np.asarray(
                artifact.get("test_idx", container.test_idx), dtype=int
            )
            groups = np.asarray(container.groups)[test_idx]
        groups_by_scenario[scenario] = np.asarray(groups)

    rows: List[Dict[str, Any]] = []
    if len(predictions) < 2:
        logger.log(
            '  Fewer than two scenarios have stored held-out test predictions; paired contrasts cannot be computed.',
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
                "Paired contrasts require an identical locked-test row order across "
                f"scenarios, but {scenario} has different stored true labels than "
                f"{reference}."
            )
        if not np.array_equal(
            np.asarray(groups_by_scenario[scenario], dtype=str),
            np.asarray(groups, dtype=str),
        ):
            raise RuntimeError(
                "Paired contrasts require identical test gene clusters across "
                f"scenarios, but {scenario} differs from {reference}."
            )
    if len(groups) != len(y_true):
        raise RuntimeError(
            f"Test gene-cluster vector has length {len(groups)} but the stored "
            f"predictions have length {len(y_true)}."
        )

    # One shared stream of resampled row sets, reused by every scenario/metric so
    # that each replicate scores both members of a contrast on the same rows.
    rng = np.random.RandomState(RANDOM_SEED)
    replicate_indices: List[np.ndarray] = []
    for _ in range(Config.N_BOOTSTRAP):
        idx = _cluster_bootstrap_indices(y_true, groups, rng)
        if len(idx) == 0:
            continue
        replicate_indices.append(idx)
    logger.log(
        f'  Drew {len(replicate_indices)} of {Config.N_BOOTSTRAP} non-empty severity-stratified gene-cluster replicates (seed={RANDOM_SEED}); {len(np.unique(groups))} gene clusters over {len(y_true)} held-out test pairs.'
    )

    replicate_values: Dict[Tuple[str, str], np.ndarray] = {}
    observed: Dict[Tuple[str, str], float] = {}
    for scenario, payload in predictions.items():
        for metric in PAIRED_CONTRAST_METRICS:
            observed[(scenario, metric)] = _paired_contrast_metric(
                y_true, payload["y_pred"], metric
            )
            values = np.full(len(replicate_indices), np.nan, dtype=float)
            for position, idx in enumerate(replicate_indices):
                try:
                    values[position] = _paired_contrast_metric(
                        y_true[idx], payload["y_pred"][idx], metric
                    )
                except Exception as exc:
                    logger.log(
                        f"  Paired replicate undefined for {scenario}/{metric}: "
                        f"{type(exc).__name__}: {exc}",
                        level="WARNING",
                    )
            replicate_values[(scenario, metric)] = values

    for scenario_a, scenario_b in PAIRED_SCENARIO_CONTRASTS:
        if scenario_a not in predictions or scenario_b not in predictions:
            logger.log(
                f'  Skipping contrast {scenario_a} minus {scenario_b}: one or both scenarios have no stored held-out test predictions.',
                level="WARNING",
            )
            continue
        for metric in PAIRED_CONTRAST_METRICS:
            differences = (
                replicate_values[(scenario_a, metric)]
                - replicate_values[(scenario_b, metric)]
            )
            valid = differences[np.isfinite(differences)]
            n_valid = int(len(valid))
            rows.append({
                "contrast": f"{scenario_a}_minus_{scenario_b}",
                "scenario_a": scenario_a,
                "scenario_b": scenario_b,
                "model_a": models[scenario_a],
                "model_b": models[scenario_b],
                "metric": metric,
                "observed_a": observed[(scenario_a, metric)],
                "observed_b": observed[(scenario_b, metric)],
                "observed_difference": (
                    observed[(scenario_a, metric)] - observed[(scenario_b, metric)]
                ),
                "difference_percentile_2.5": (
                    float(np.percentile(valid, 2.5)) if n_valid >= 10 else np.nan
                ),
                "difference_percentile_97.5": (
                    float(np.percentile(valid, 97.5)) if n_valid >= 10 else np.nan
                ),
                "bootstrap_tail_fraction_at_or_below_zero": (
                    float(np.mean(valid <= 0.0)) if n_valid else np.nan
                ),
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
                "tail_fraction_interpretation": (
                    "bootstrap tail fraction, not a P value"
                ),
            })

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(output_path, index=False)
    for row in rows:
        logger.log(
            f"  {row['contrast']} {row['metric']}: "
            f"delta={row['observed_difference']:+.4f} "
            f"[{row['difference_percentile_2.5']:+.4f}, "
            f"{row['difference_percentile_97.5']:+.4f}] "
            f"tail_fraction<=0={row['bootstrap_tail_fraction_at_or_below_zero']:.4f} "
            f"(n_valid={row['n_valid_replicates']})"
        )
    return dataframe


def _primary_partition_indices(container: DataContainer) -> Dict[str, np.ndarray]:
    for artifact in _fold_artifacts.values():
        if all(key in artifact for key in ("train_idx", "val_idx", "test_idx")):
            return {
                "train": np.asarray(artifact["train_idx"], dtype=int),
                "validation": np.asarray(artifact["val_idx"], dtype=int),
                "test": np.asarray(artifact["test_idx"], dtype=int),
            }
    identifiers_path = os.path.join(
        Config.OUTPUT_DIR, "results", "split_identifiers_primary.csv"
    )
    if os.path.isfile(identifiers_path):
        frame = pd.read_csv(identifiers_path)
        return {
            partition: frame.loc[
                frame["partition"] == partition, "row_index"
            ].to_numpy(dtype=int)
            for partition in ("train", "validation", "test")
        }
    raise RuntimeError(
        "Disease-overlap audit could not recover the primary train/validation/test "
        "indices from _fold_artifacts or split_identifiers_primary.csv."
    )


def run_disease_overlap_audit(container: DataContainer) -> pd.DataFrame:
    logger.subsection('Disease-overlap check across primary partitions')
    output_path = os.path.join(
        Config.OUTPUT_DIR, "results", "disease_overlap_audit.csv"
    )
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
    percent_test_pairs_in_train = (
        100.0 * test_pairs_in_train / n_test_pairs if n_test_pairs else np.nan
    )

    # Distinct direct HPO term columns flagged present, counted per disease.
    hpo_columns = [
        column for column in getattr(container, "hpo_binary_cols", [])
        if column in container.full_df.columns
    ]
    gene_counts = (
        pd.DataFrame({"mondo_id": mondo, "gene_symbol": genes})
        .groupby("mondo_id")["gene_symbol"].nunique()
    )
    reportable_diseases = sorted(
        gene_counts.index[gene_counts >= DISEASE_OVERLAP_MIN_GENES].astype(str)
    )
    hpo_present_by_disease: Dict[str, int] = {}
    if hpo_columns and reportable_diseases:
        # Restricted to the reportable diseases so the wide HPO block is only
        # materialised for the rows that appear in the per-disease table.
        reportable_mask = np.isin(mondo, reportable_diseases)
        hpo_frame = container.full_df.loc[reportable_mask, hpo_columns].apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0.0)
        present = hpo_frame.to_numpy() > 0
        subset_mondo = mondo[reportable_mask]
        for disease in reportable_diseases:
            hpo_present_by_disease[disease] = int(
                present[subset_mondo == disease].any(axis=0).sum()
            )
    elif not hpo_columns:
        logger.log(
            "  No direct HPO binary columns available; per-disease HPO counts "
            "are reported as 0.",
            level="WARNING",
        )

    rows: List[Dict[str, Any]] = []

    def summary_row(metric: str, value: Any) -> Dict[str, Any]:
        return {"record_type": "summary", "metric": metric, "value": value}

    rows.append(summary_row("n_pairs_total", int(len(metadata))))
    rows.append(summary_row("n_train_pairs", int(len(partitions["train"]))))
    rows.append(summary_row("n_validation_pairs", int(len(partitions["validation"]))))
    rows.append(summary_row("n_test_pairs", n_test_pairs))
    rows.append(summary_row(
        "n_test_pairs_with_mondo_id_also_in_train", test_pairs_in_train
    ))
    rows.append(summary_row(
        "percent_test_pairs_with_mondo_id_also_in_train",
        round(float(percent_test_pairs_in_train), 4),
    ))
    rows.append(summary_row("n_distinct_mondo_total", int(len(set(mondo)))))
    rows.append(summary_row("n_distinct_mondo_train", int(len(train_mondo))))
    rows.append(summary_row("n_distinct_mondo_validation", int(len(validation_mondo))))
    rows.append(summary_row("n_distinct_mondo_test", int(len(test_mondo))))
    rows.append(summary_row(
        "n_distinct_mondo_in_test_and_train", int(len(test_mondo & train_mondo))
    ))
    rows.append(summary_row(
        "n_distinct_mondo_in_test_and_validation",
        int(len(test_mondo & validation_mondo)),
    ))
    rows.append(summary_row(
        "percent_distinct_test_mondo_also_in_train",
        round(100.0 * len(test_mondo & train_mondo) / len(test_mondo), 4)
        if test_mondo else np.nan,
    ))
    rows.append(summary_row(
        "min_genes_for_per_disease_table", int(DISEASE_OVERLAP_MIN_GENES)
    ))
    rows.append(summary_row(
        "split_policy", "gene-disjoint; MONDO identifiers are not disjoint"
    ))

    disease_rows: List[Dict[str, Any]] = []
    for disease in reportable_diseases:
        mask = mondo == disease
        n_genes = int(len(set(genes[mask])))
        disease_labels = pd.Series(labels[mask])
        label_counts = disease_labels.value_counts()
        disease_names = pd.Series([n for n in names[mask] if n and n != "nan"])
        disease_rows.append({
            "record_type": "disease",
            "mondo_id": disease,
            "disease_name": (
                disease_names.mode().iloc[0] if not disease_names.empty else ""
            ),
            "n_genes": n_genes,
            "n_pairs": int(mask.sum()),
            "n_direct_hpo_terms": int(hpo_present_by_disease.get(disease, 0)),
            "severity_label": str(label_counts.index[0]) if len(label_counts) else "",
            "severity_label_is_unique": bool(len(label_counts) == 1),
            "severity_labels_observed": "|".join(
                f"{label}:{count}" for label, count in label_counts.items()
            ),
            "n_train_pairs": int(np.sum(partition_of[mask] == "train")),
            "n_validation_pairs": int(np.sum(partition_of[mask] == "validation")),
            "n_test_pairs": int(np.sum(partition_of[mask] == "test")),
            "spans_train_and_test": bool(
                np.any(partition_of[mask] == "train")
                and np.any(partition_of[mask] == "test")
            ),
        })
    disease_rows.sort(key=lambda item: (-item["n_genes"], item["mondo_id"]))
    rows.extend(disease_rows)

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(output_path, index=False)
    logger.log(
        f'  {test_pairs_in_train} of {n_test_pairs} held-out test pairs ({percent_test_pairs_in_train:.1f}%) carry a MONDO identifier that also appears in training; distinct MONDO ids train={len(train_mondo)}, validation={len(validation_mondo)}, test={len(test_mondo)}.'
    )
    logger.log(
        f"  Per-disease table covers {len(disease_rows)} diseases with at least "
        f"{DISEASE_OVERLAP_MIN_GENES} genes."
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
            return np.nan, np.nan
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
    ece_low, ece_high = summary(ece_values)
    brier_low, brier_high = summary(brier_values)
    ece_valid = int(np.isfinite(np.asarray(ece_values, dtype=float)).sum())
    brier_valid = int(np.isfinite(np.asarray(brier_values, dtype=float)).sum())
    return {
        "ece_ci_low": ece_low, "ece_ci_high": ece_high,
        "brier_ci_low": brier_low, "brier_ci_high": brier_high,
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
            "calibrated" if result.get("calibration_status") == "success"
            else "uncalibrated_fallback"
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
                    binary, probabilities[:, class_index], Config.CALIBRATION_N_BINS,
                    Config.CALIBRATION_BINNING
                )
                brier = _brier_one_class(binary, probabilities[:, class_index])
                ci = _bootstrap_calibration_ci(
                    binary, probabilities[:, class_index], Config.CALIBRATION_N_BINS,
                    Config.CALIBRATION_BINNING, Config.CALIBRATION_N_BOOTSTRAP,
                    RANDOM_SEED + scenario_index * 100 + class_index,
                    groups=np.asarray(groups),
                )
                bins = _calibration_bin_table(
                    binary, probabilities[:, class_index], Config.CALIBRATION_N_BINS,
                    Config.CALIBRATION_BINNING
                )
                for _, bin_row in bins.iterrows():
                    bin_rows.append({
                        "Scenario": scenario, "Model": model_name,
                        "calibration_status": status, "severity_class": class_name,
                        **bin_row.to_dict(),
                    })
                rows.append({
                    "Scenario": scenario, "Model": model_name,
                    "calibration_status": status, "severity_class": class_name,
                    "calibration_method": result.get("calibration_method", "none"),
                    "calibration_method_attempted": result.get(
                        "calibration_method_attempted", "none"
                    ),
                    "calibration_success": bool(result.get("calibration_success", False)),
                    "pipeline_calibration_status": result.get(
                        "calibration_status", "not_attempted"
                    ),
                    "bootstrap_unit": "gene_cluster", "ece": ece, **ci,
                    "brier": brier, "multiclass_brier": multiclass_brier,
                    "ranked_probability_score": rps,
                    "n_test": len(y_true), "n_test_in_class": int(binary.sum()),
                })
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
            "Scenario": scenario, "Stratum": "all", "Model": model_name,
            "Selected_on_validation": True,
            "Calibration_status": result.get("calibration_status", "not_attempted"),
            "Calibration_success": bool(result.get("calibration_success", False)),
            "Calibration_method": result.get("calibration_method", "none"),
            "Calibration_method_attempted": result.get(
                "calibration_method_attempted", "none"
            ),
            "Calibration_failure_details": result.get(
                "calibration_failure_details", ""
            ),
            "MCC": calibrated.get("mcc", np.nan), "QWK": calibrated.get("qwk", np.nan),
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
            match = bootstrap_df[(bootstrap_df["Scenario"] == scenario) & (bootstrap_df["Model"] == model_name)]
            if not match.empty:
                boot = match.iloc[0]
                for metric, label in (("mcc", "MCC"), ("qwk", "QWK"),
                                      ("ranked_probability_score", "RPS")):
                    row[f"{label}_CI_low"] = boot.get(f"{metric}_ci_low", np.nan)
                    row[f"{label}_CI_high"] = boot.get(f"{metric}_ci_high", np.nan)
        rows.append(row)
    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(os.path.join(Config.OUTPUT_DIR, "results", "publication_table.csv"), index=False)
    return dataframe


def run_imbalance_sensitivity(container: DataContainer, all_results: Dict) -> pd.DataFrame:
    if not Config.RUN_IMBALANCE_SENSITIVITY:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for scenario, model_name in _pick_best_model_per_scenario(all_results).items():
        artifacts = _fold_artifacts[scenario]
        params = all_results[scenario]["all"][model_name].get("best_params", {})
        for strategy in ("sample_weight", "resampling", "none"):
            try:
                model = _fit_model_matrix(
                    model_name, params, artifacts["X_train"], artifacts["y_train"],
                    scenario, RANDOM_SEED + 900, imbalance_strategy=strategy
                )
                probability = _predict_model_matrix(model_name, model, artifacts["X_eval"])
                prediction = np.argmax(probability, axis=1)
                metrics = compute_all_metrics(artifacts["y_val"], prediction, probability)
                rows.append({
                    "Scenario": scenario, "Model": model_name,
                    "imbalance_strategy": strategy, "evaluation_partition": "validation_only",
                    "qwk": metrics.get("qwk"), "mcc": metrics.get("mcc"),
                    "balanced_accuracy": metrics.get("balanced_accuracy"),
                    "ranked_probability_score": metrics.get("ranked_probability_score"),
                })
            except Exception as exc:
                logger.log(f"  Imbalance sensitivity failed {scenario}/{strategy}: {exc}", level="WARNING")
    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(os.path.join(Config.OUTPUT_DIR, "results", "imbalance_sensitivity.csv"), index=False)
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


def _gene_block_permutation_capacity(groups: np.ndarray) -> float:
    groups = np.asarray(groups)
    group_sizes = Counter(groups.tolist())
    genes_by_size: Dict[int, List[Any]] = defaultdict(list)
    for gene, size in group_sizes.items():
        genes_by_size[int(size)].append(gene)
    movable_rows = sum(
        size * len(genes) for size, genes in genes_by_size.items() if len(genes) >= 2
    )
    return float(movable_rows / max(len(groups), 1))


def permute_gene_label_blocks(
    y: np.ndarray,
    groups: np.ndarray,
    rng: np.random.RandomState,
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
    return permuted, float(movable_rows / max(len(y), 1))


def run_permutation_test(
    container: DataContainer,
    all_results: Dict,
    n_permutations: Optional[int] = None,
) -> pd.DataFrame:
    """Conditional validation permutation test using complete gene label blocks."""
    if not Config.RUN_PERMUTATION_TEST:
        return pd.DataFrame()
    n_permutations = int(n_permutations or Config.N_PERMUTATIONS)
    rows: List[Dict[str, Any]] = []
    candidate_names = _available_candidate_names()

    selected_by_scenario = _pick_best_model_per_scenario(
        all_results
    )
    for scenario, observed_name in selected_by_scenario.items():
        artifacts = _fold_artifacts[scenario]
        observed = float(
            all_results[scenario]["all"][observed_name]["val_metrics"].get("mcc", np.nan)
        )
        groups_train = np.asarray(artifacts["groups_train"])
        movable_fraction = _gene_block_permutation_capacity(groups_train)
        if movable_fraction < float(Config.PERMUTATION_MIN_MOVABLE_FRACTION):
            logger.log(
                f"  Skipping permutation test for {scenario}: only "
                f"{movable_fraction:.1%} of training rows can move under exact "
                "equal-size gene-block permutation.",
                level="WARNING",
            )
            continue

        parameters_by_model = {
            name: all_results[scenario]["all"][name].get("best_params", {})
            for name in candidate_names if name in all_results[scenario]["all"]
        }
        rng = np.random.RandomState(RANDOM_SEED + 700)
        null_statistics: List[float] = []
        failures = 0
        observed_movable_fractions: List[float] = []

        for permutation_index in range(n_permutations):
            permuted_y, realised_movable = permute_gene_label_blocks(
                artifacts["y_train"], groups_train, rng
            )
            observed_movable_fractions.append(realised_movable)
            try:
                if uses_grouped_mi(scenario):
                    X_train, _, _, mask, _, _ = select_features_mi_grouped(
                        artifacts["X_train_full"],
                        permuted_y,
                        artifacts["feature_names_full"],
                        family_lookup=artifacts.get(
                            "family_lookup", container.feature_family_lookup
                        ),
                        budget=artifacts["selection_budget"],
                        mi_leakage_threshold=None,
                        scenario_name=f"{scenario}_perm{permutation_index}",
                    )
                else:
                    mechanistic = "S1_Gene_Level" in scenario
                    X_train, _, _, mask, _ = select_features_mi(
                        artifacts["X_train_full"],
                        permuted_y,
                        artifacts["feature_names_full"],
                        mi_threshold=(
                            Config.MI_THRESHOLD_MECHANISTIC
                            if mechanistic else Config.MI_THRESHOLD_DEFAULT
                        ),
                        mi_leakage_threshold=None,
                        min_features=(
                            Config.MI_MIN_FEATURES_MECHANISTIC
                            if mechanistic else Config.MI_MIN_FEATURES_DEFAULT
                        ),
                        scenario_name=f"{scenario}_perm{permutation_index}",
                        family_lookup=artifacts.get(
                            "family_lookup", container.feature_family_lookup
                        ),
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
                    probability = _predict_model_matrix(
                        model_name, model, X_validation
                    )
                    prediction = np.argmax(probability, axis=1)
                    metrics = compute_all_metrics(
                        artifacts["y_val"], prediction, probability
                    )
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
            logger.log(
                f"  No successful gene-block permutations for {scenario}.",
                level="WARNING",
            )
            continue
        empirical_p = (int(np.sum(values >= observed)) + 1) / (len(values) + 1)
        rows.append({
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
        })

    dataframe = pd.DataFrame(rows)
    if not dataframe.empty:
        dataframe["bh_adjusted_p_value"] = _benjamini_hochberg(
            dataframe["empirical_p_value"].to_numpy(dtype=float)
        )
    dataframe.to_csv(
        os.path.join(Config.OUTPUT_DIR, "results", "permutation_test.csv"),
        index=False,
    )
    return dataframe


def run_split_sensitivity_fixed_configuration(
    container: DataContainer, all_results: Dict, n_seeds: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split sensitivity for the primary selected families/configurations."""
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
                f"Split-sensitivity seed {seed} reproduced the primary test holdout; "
                "refusing repeated primary-test evaluation."
            )
        train_idx, val_idx = _split_development_train_validation(
            container, development_idx, container.groups, seed
        )
        for scenario, model_name in selected.items():
            try:
                budget = _fold_artifacts[scenario].get("selection_budget", Config.FEATURE_BUDGET)
                rep = _fit_scenario_representation(
                    container, scenario, train_idx, val_idx, budget,
                    Config.MI_LEAKAGE_THRESHOLD, f"{scenario}_splitseed{seed}"
                )
                params = all_results[scenario]["all"][model_name].get("best_params", {})
                result = _fit_final_candidate(
                    model_name, params, rep, container.y[train_idx], container.y[val_idx],
                    scenario, seed
                )
                _fit_selected_calibration(result, rep["X_eval"], container.y[val_idx], f"splitseed{seed}:{scenario}")
                X_test_full = rep["preprocessor"].transform(rep["temp_df"], test_idx)
                X_test = X_test_full[:, rep["selected_mask"]]
                probability = _predict_calibrated_candidate(result, X_test)
                prediction = np.argmax(probability, axis=1)
                metrics = compute_all_metrics(container.y[test_idx], prediction, probability)
                rows.append({
                    "Scenario": scenario, "Model": model_name, "seed": seed,
                    "analysis_type": "split_sensitivity_fixed_configuration",
                    "mcc": metrics.get("mcc"), "qwk": metrics.get("qwk"),
                    "balanced_accuracy": metrics.get("balanced_accuracy"),
                    "ranked_probability_score": metrics.get("ranked_probability_score"),
                })
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
    return per_seed, summary


def run_severity_hpo_term_sensitivity(
    container: DataContainer, all_results: Dict
) -> pd.DataFrame:
    summary_path = os.path.join(
        Config.OUTPUT_DIR, "results", "severity_hpo_term_sensitivity.csv"
    )
    removed_path = os.path.join(
        Config.OUTPUT_DIR, "results", "severity_hpo_terms_removed.csv"
    )
    if not Config.RUN_SEVERITY_HPO_SENSITIVITY:
        pd.DataFrame().to_csv(summary_path, index=False)
        pd.DataFrame().to_csv(removed_path, index=False)
        return pd.DataFrame()

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
            Config.MI_LEAKAGE_THRESHOLD,
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
            Config.MI_LEAKAGE_THRESHOLD,
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
        primary = all_results[scenario]["all"][model_name].get("val_metrics", {})
        baseline = result_retained.get("val_metrics", {})
        metrics = result.get("val_metrics", {})
        retained_names = list(representation_retained.get("selected_names", []))
        severity_selected_when_retained = [
            column for column in retained_names
            if (column.startswith("hpo_") or column.startswith("feat_hpo_sys_"))
            and is_severity_hpo_column(column)
        ]
        removed = [
            item for item in representation["preprocessor"].dropped_features
            if item.get("drop_reason") == "severity_hpo_blocklist"
        ]
        for item in removed:
            provenance = _feature_provenance_for_mi(
                str(item.get("feature", "")), representation.get("family_lookup", {})
            )
            removed_rows.append({
                "Scenario": scenario,
                "feature": item.get("feature", ""),
                "drop_reason": item.get("drop_reason", ""),
                **provenance,
            })
        rows.append({
            "Scenario": scenario,
            "Model": model_name,
            "evaluation_partition": "validation_only",
            "model_family_and_hyperparameters_fixed": True,
            "severity_hpo_terms_removed": True,
            "baseline_arm": "severity_terms_retained_refit",
            "n_severity_hpo_terms_removed": len(removed),
            "n_severity_terms_selected_when_retained": len(severity_selected_when_retained),
            "severity_terms_selected_when_retained": ";".join(
                sorted(severity_selected_when_retained)
            ),
            "n_features_selected": len(representation.get("selected_names", [])),
            "n_features_selected_retained": len(retained_names),
            "mcc": metrics.get("mcc", np.nan),
            "qwk": metrics.get("qwk", np.nan),
            "balanced_accuracy": metrics.get("balanced_accuracy", np.nan),
            "baseline_validation_mcc": baseline.get("mcc", np.nan),
            "baseline_validation_qwk": baseline.get("qwk", np.nan),
            "baseline_validation_balanced_accuracy": baseline.get(
                "balanced_accuracy", np.nan
            ),
            "delta_mcc": metrics.get("mcc", np.nan) - baseline.get("mcc", np.nan),
            "delta_qwk": metrics.get("qwk", np.nan) - baseline.get("qwk", np.nan),
            "primary_run_validation_mcc": primary.get("mcc", np.nan),
            "primary_run_validation_qwk": primary.get("qwk", np.nan),
            "delta_mcc_vs_primary_run": metrics.get("mcc", np.nan)
            - primary.get("mcc", np.nan),
            "delta_qwk_vs_primary_run": metrics.get("qwk", np.nan)
            - primary.get("qwk", np.nan),
        })
        logger.log(
            f"  {scenario}: retained arm selected "
            f"{len(severity_selected_when_retained)} severity-related HPO terms "
            f"(validation MCC={baseline.get('mcc', np.nan):.4f}, "
            f"QWK={baseline.get('qwk', np.nan):.4f}); removed arm dropped "
            f"{len(removed)} such terms (validation MCC={metrics.get('mcc', np.nan):.4f}, "
            f"QWK={metrics.get('qwk', np.nan):.4f}); "
            f"delta MCC={metrics.get('mcc', np.nan) - baseline.get('mcc', np.nan):+.4f}, "
            f"delta QWK={metrics.get('qwk', np.nan) - baseline.get('qwk', np.nan):+.4f}."
        )

    dataframe = pd.DataFrame(rows)
    removed_dataframe = pd.DataFrame(removed_rows, columns=[
        "Scenario", "feature", "drop_reason", "feature_family",
        "source_dataset", "same_source_as_target_label", "provenance_status",
    ])
    dataframe.to_csv(summary_path, index=False)
    removed_dataframe.to_csv(removed_path, index=False)
    return dataframe


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
        "y_pred": majority_pred, "y_prob": majority_prob, "y_true": y_test,
    }

    class_probabilities = np.bincount(y_train, minlength=Config.NUM_CLASSES).astype(float)
    class_probabilities /= max(class_probabilities.sum(), 1.0)
    rng = np.random.RandomState(RANDOM_SEED)
    random_pred = rng.choice(Config.NUM_CLASSES, size=len(y_test), p=class_probabilities)
    random_prob = np.tile(class_probabilities, (len(y_test), 1))
    baselines["StratifiedRandom"] = {
        "metrics": compute_all_metrics(y_test, random_pred, random_prob),
        "y_pred": random_pred, "y_prob": random_prob, "y_true": y_test,
    }

    ordinal_mean = int(round(np.clip(float(y_train.mean()), 0, Config.NUM_CLASSES - 1)))
    ordinal_pred = np.full(len(y_test), ordinal_mean, dtype=int)
    ordinal_prob = np.zeros((len(y_test), Config.NUM_CLASSES), dtype=float)
    ordinal_prob[:, ordinal_mean] = 1.0
    baselines["OrdinalMean"] = {
        "metrics": compute_all_metrics(y_test, ordinal_pred, ordinal_prob),
        "y_pred": ordinal_pred, "y_prob": ordinal_prob, "y_true": y_test,
    }

    rows = []
    for name, result in baselines.items():
        row = {"Baseline": name, "training_partition": "primary_train_only"}
        row.update({k: v for k, v in result["metrics"].items() if not isinstance(v, str)})
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        os.path.join(Config.OUTPUT_DIR, "results", "naive_baselines.csv"), index=False
    )
    return baselines


def run_seed_sensitivity(container: DataContainer, all_results: Dict, n_seeds: int = None):
    if not Config.RUN_SEED_SENSITIVITY:
        return pd.DataFrame(), pd.DataFrame()
    return run_split_sensitivity_fixed_configuration(container, all_results, n_seeds)


def _normalise_multiclass_shap_values(values: Any, n_features: int) -> List[np.ndarray]:
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


def uses_grouped_mi(scenario: str) -> bool:
    return any((s in scenario for s in ["S2", "S3"]))


def _pick_best_model_per_scenario(all_results: Dict) -> Dict[str, str]:
    best: Dict[str, str] = {}
    for scenario, strata in all_results.items():
        models = strata.get("all", {})
        explicitly_selected = [
            name
            for name, result in models.items()
            if result.get("selected_on_validation", False)
        ]
        if explicitly_selected:
            best[scenario] = explicitly_selected[0]
            continue

        eligible = []
        for name, result in models.items():
            metrics = result.get("val_metrics")
            if not metrics:
                continue
            eligible.append((validation_selection_score(metrics), name))
        if eligible:
            eligible.sort(key=lambda item: (-item[0], item[1]))
            best[scenario] = eligible[0][1]
    return best


def create_performance_heatmap(results_df: pd.DataFrame):
    if len(results_df) == 0:
        return
    try:
        n_scenarios = results_df["Scenario"].nunique()
        plt.figure(figsize=(10, max(6, 0.7 * n_scenarios + 2)))
        pivot = results_df.pivot_table(
            index="Scenario", columns="Model", values="mcc", aggfunc="max"
        )
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=0, vmin=-0.2, vmax=1.0)
        plt.title("Performance heatmap (MCC)")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                Config.OUTPUT_DIR, "figures", "diagnostics", "performance_heatmap_mcc.png"
            ),
            dpi=Config.FIGURE_DPI,
        )
        plt.close()
    except Exception as e:
        logger.log(f"  Heatmap failed: {e}", level="WARNING")


def create_confusion_matrix_plot(predictions: Dict):
    if not predictions:
        return
    try:
        scenarios = ordered_scenarios_from_keys(predictions.keys())
        n = len(scenarios)
        if n == 0:
            return
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
        if n == 1:
            axes = [axes]
        else:
            axes = np.array(axes).flatten()
        for i, scenario in enumerate(scenarios):
            keys = [k for k in predictions if k.startswith(scenario)]
            if not keys:
                continue
            best_key = max(
                keys,
                key=lambda k: matthews_corrcoef(predictions[k]["y_true"], predictions[k]["y_pred"]),
            )
            yt = predictions[best_key]["y_true"]
            yp = predictions[best_key]["y_pred"]
            cm = confusion_matrix(yt, yp, labels=list(range(Config.NUM_CLASSES)))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Mild", "Moderate", "Severe", "Profound"],
                yticklabels=["Mild", "Moderate", "Severe", "Profound"],
                ax=axes[i],
                cbar=False,
            )
            axes[i].set_title(f"{scenario} ({best_key.rsplit('_', 1)[1]})")
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("True")
        for j in range(len(scenarios), len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        plt.savefig(
            os.path.join(Config.OUTPUT_DIR, "figures", "diagnostics", "confusion_matrices.png"),
            dpi=Config.FIGURE_DPI,
        )
        plt.close()
    except Exception as e:
        logger.log(f"  Confusion matrix plot failed: {e}", level="WARNING")


def _load_feature_label_map() -> Dict[str, str]:
    if _FEATURE_LABEL_LOADED["done"]:
        return _FEATURE_LABEL_CACHE
    _FEATURE_LABEL_LOADED["done"] = True
    # Built-in labels are the base layer. The external dictionaries loaded below
    # overwrite any entry here, so a run can still override without editing code.
    _FEATURE_LABEL_CACHE.update(FEATURE_DISPLAY_LABELS)
    for p in [os.path.join(Config.OUTPUT_DIR, "feature_label_dictionary.csv"),
              os.path.join(os.getcwd(), "feature_label_dictionary.csv"),
              "feature_label_dictionary.csv"]:
        if not os.path.exists(p):
            continue
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        if "feature_name" not in d.columns:
            continue
        has_label = "hpo_label" in d.columns
        has_id = "hpo_id" in d.columns
        for _, row in d.iterrows():
            fname = str(row["feature_name"])
            label = str(row["hpo_label"]).strip() if has_label and not pd.isna(row["hpo_label"]) else ""
            hpoid = str(row["hpo_id"]).strip() if has_id and not pd.isna(row["hpo_id"]) else ""
            disp = label or hpoid
            if label and hpoid:
                disp = f"{label} ({hpoid})"
            if disp:
                _FEATURE_LABEL_CACHE[fname] = disp
        break
    for p in [os.path.join(Config.OUTPUT_DIR, "feature_dictionary.csv"),
              os.path.join(os.getcwd(), "feature_dictionary.csv"),
              "feature_dictionary.csv"]:
        if not os.path.exists(p):
            continue
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        if {"feature", "readable_label"}.issubset(d.columns):
            for _, row in d.iterrows():
                fn = str(row["feature"])
                lab = str(row["readable_label"]).strip()
                if fn not in _FEATURE_LABEL_CACHE and lab:
                    _FEATURE_LABEL_CACHE[fn] = lab
        break
    return _FEATURE_LABEL_CACHE


def _load_hpo_id_name_map() -> Dict[str, str]:
    if _HPO_ID_NAME_LOADED["done"]:
        return _HPO_ID_NAME_CACHE
    _HPO_ID_NAME_LOADED["done"] = True
    import pickle as _pk
    pkl_search = [
        os.path.join(Config.OUTPUT_DIR, "hp.obo.pkl"),
        os.path.join(os.getcwd(), "hp.obo.pkl"),
        "hp.obo.pkl",
    ]
    for p in pkl_search:
        if os.path.exists(p):
            try:
                obj = _pk.load(open(p, "rb"))
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        nm = v.get("name") if isinstance(v, dict) else v
                        if str(k).startswith("HP:") and nm:
                            _HPO_ID_NAME_CACHE[str(k)] = str(nm)
                if _HPO_ID_NAME_CACHE:
                    return _HPO_ID_NAME_CACHE
            except Exception:
                pass
    obo_search = [
        getattr(Config, "HPO_OBO_PATH", "hp.obo"),
        os.path.join(os.getcwd(), "hp.obo"),
        "hp.obo",
    ]
    for p in obo_search:
        if p and os.path.exists(p):
            try:
                cur = None
                for line in open(p, encoding="utf-8", errors="ignore"):
                    line = line.strip()
                    if line == "[Term]":
                        cur = None
                    elif line.startswith("id: HP:"):
                        cur = line[4:].strip()
                    elif line.startswith("name:") and cur:
                        _HPO_ID_NAME_CACHE[cur] = line[5:].strip()
                        cur = None
                if _HPO_ID_NAME_CACHE:
                    return _HPO_ID_NAME_CACHE
            except Exception:
                pass
    return _HPO_ID_NAME_CACHE


def readable_feature_label(name, max_len: int = 48) -> str:
    name = str(name)

    # Missingness indicators are the same feature with a qualifier, not a
    # separate quantity, so resolve the base name and append the qualifier.
    missing_suffix = ""
    base = name
    if base.endswith("__missing"):
        base = base[: -len("__missing")]
        missing_suffix = " (not available)"

    m = _load_feature_label_map()
    if base in m:
        out = m[base]
    else:
        low = base.lower()
        hpo_id_match = re.match(r"^(.*?)_?hpo_HP_(\d{7})$", base)
        text_hash_match = re.match(r"^(?:feat_)?hpo_(.+?)_([0-9a-f]{6,12})$", base)
        source_prefix = next(
            (pair for pair in _FEATURE_SOURCE_PREFIXES if base.startswith(pair[0])),
            None,
        )
        if low.startswith("feat_hpo_sys_"):
            out = "System: " + base[len("feat_hpo_sys_"):].replace("_", " ").strip()
        elif low.startswith("feat_mondo_"):
            out = "MONDO: " + base[len("feat_mondo_"):].replace("_", " ").strip()
        elif low.startswith("feat_tier_"):
            out = "Tier: " + base[len("feat_tier_"):].replace("_", " ").strip()
        elif source_prefix is not None:
            prefix, human, count_label = source_prefix
            tail = base[len(prefix):]
            if tail == "count":
                out = count_label + ", count"
            else:
                # strip the 6-hex-character uniqueness hash the pipeline appends
                # to names it had to truncate, then restore readable spacing
                tail = re.sub(r"_[0-9a-f]{6}$", "…", tail)
                out = human + tail.replace("_", " ")
        elif hpo_id_match and hpo_id_match.group(1):
            # e.g. mouse_lethal_for_hpo_HP_0000252 -> "... for Microcephaly"
            id_name = _load_hpo_id_name_map()
            term = id_name.get(
                f"HP:{hpo_id_match.group(2)}", f"HP:{hpo_id_match.group(2)}"
            )
            stem = hpo_id_match.group(1)
            if stem.startswith("mouse_lethal_for"):
                out = f"Mouse-lethal knockout for {term}"
            elif stem.startswith("mouse_subviable_for"):
                out = f"Mouse-subviable knockout for {term}"
            elif stem.startswith("mouse"):
                out = f"Mouse phenotype for {term}"
            else:
                out = f"{stem.replace('_', ' ').capitalize()}: {term}"
        elif low.startswith("hpo_") or low.startswith("feat_hpo_"):
            stripped = re.sub(r"^(feat_)?hpo_", "", base)
            hp_match = re.match(r"^HP_(\d{7})$", stripped)
            id_name = _load_hpo_id_name_map()
            if hp_match and f"HP:{hp_match.group(1)}" in id_name:
                out = id_name[f"HP:{hp_match.group(1)}"]
            elif text_hash_match:
                # phenotype columns keyed by term text with a trailing hash
                term_text = text_hash_match.group(1).replace("_", " ").strip()
                out = re.sub(r"\s{2,}", ", ", term_text)
                if len(text_hash_match.group(1)) >= 33:
                    out += "…"
            else:
                out = "Phenotype: " + stripped.replace("_", " ").strip()
        elif low.startswith("has_") and low.endswith("_record"):
            out = base[len("has_"): -len("_record")].replace("_", " ") + " data available"
        elif low.startswith("feat_"):
            out = base[len("feat_"):].replace("_", " ").strip()
        else:
            out = base.replace("_", " ").strip()
        out = (out[:1].upper() + out[1:]) if out else base

    out = out + missing_suffix
    if max_len and len(out) > max_len:
        out = out[:max_len - 1].rstrip() + "…"
    return out


def write_feature_label_dictionaries(container: DataContainer) -> None:
    """Export the resolved display labels so downstream figure code can read them."""
    names: List[str] = []
    for columns in container.scenario_cols.values():
        names.extend(columns)
    names.extend(FEATURE_DISPLAY_LABELS)
    names.extend(f"{n}__missing" for n in FEATURE_DISPLAY_LABELS)
    names = sorted(dict.fromkeys(str(n) for n in names))

    rows = [(n, readable_feature_label(n, max_len=0)) for n in names]
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    for filename, columns in (
        ("feature_label_dictionary_extra.csv", ("feature_name", "label")),
        ("feature_dictionary.csv", ("feature", "readable_label")),
    ):
        path = os.path.join(Config.OUTPUT_DIR, filename)
        pd.DataFrame(rows, columns=list(columns)).to_csv(path, index=False)
    logger.log(
        f"  Wrote display labels for {len(rows)} features to "
        f"feature_dictionary.csv and feature_label_dictionary_extra.csv"
    )


def create_feature_importance_plot(feature_importances: Dict):
    if not feature_importances:
        return
    try:
        scenarios = ordered_scenarios_from_keys(feature_importances.keys())
        for scenario in scenarios:
            keys = [k for k in feature_importances if k.startswith(scenario)]
            if not keys:
                continue
            fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 8))
            if len(keys) == 1:
                axes = [axes]
            for ax, k in zip(axes, keys):
                imp_df = feature_importances[k]
                if "importance" in imp_df.columns:
                    val_col = "importance"
                elif "mean_abs_shap" in imp_df.columns:
                    val_col = "mean_abs_shap"
                else:
                    continue
                top = imp_df.nlargest(20, val_col)
                ax.barh(range(len(top)), top[val_col].values, color="steelblue")
                ax.set_yticks(range(len(top)))
                ax.set_yticklabels(
                    [readable_feature_label(f) for f in top["feature"].values], fontsize=7
                )
                ax.invert_yaxis()
                ax.set_title(k.rsplit("_", 1)[1])
                ax.set_xlabel(val_col)
            plt.suptitle(f"{scenario} top features")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    Config.OUTPUT_DIR,
                    "figures",
                    "diagnostics",
                    f"feature_importance_{scenario}.png",
                ),
                dpi=Config.FIGURE_DPI,
            )
            plt.close()
    except Exception as e:
        logger.log(f"  Feature importance plot failed: {e}", level="WARNING")


def create_calibration_plot(all_results: Dict):
    if not all_results:
        return
    try:
        best_models = _pick_best_model_per_scenario(all_results)
        scenarios = ordered_scenarios_from_keys(best_models.keys())
        n = len(scenarios)
        if n == 0:
            return
        ncols = min(2, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 5.0 * nrows), squeeze=False)
        axes = axes.flatten()
        severity_names = ["Mild", "Moderate", "Severe", "Profound"]
        for i, scenario in enumerate(scenarios):
            best_name = best_models[scenario]
            res = all_results[scenario]["all"].get(best_name)
            if res is None:
                axes[i].axis("off")
                continue
            yt = np.asarray(res["y_true"])
            ypr = np.asarray(res["predictions"]["y_prob"])
            ax = axes[i]
            for c, class_name in enumerate(severity_names):
                yt_bin = (yt == c).astype(int)
                bins = _calibration_bin_table(
                    yt_bin,
                    ypr[:, c],
                    n_bins=Config.CALIBRATION_N_BINS,
                    strategy=Config.CALIBRATION_BINNING,
                )
                if bins.empty:
                    continue
                ece = _compute_ece_one_class(
                    yt_bin,
                    ypr[:, c],
                    n_bins=Config.CALIBRATION_N_BINS,
                    strategy=Config.CALIBRATION_BINNING,
                )
                brier = _brier_one_class(yt_bin, ypr[:, c])
                marker_sizes = 25.0 + 125.0 * (bins["n_bin"].to_numpy() / bins["n_bin"].max())
                (line,) = ax.plot(
                    bins["mean_predicted_probability"],
                    bins["observed_frequency"],
                    marker="o",
                    linewidth=1.8,
                    label=f"{class_name} (ECE={ece:.3f}, Brier={brier:.3f})",
                )
                ax.scatter(
                    bins["mean_predicted_probability"],
                    bins["observed_frequency"],
                    s=marker_sizes,
                    alpha=0.65,
                    color=line.get_color(),
                )
            ax.plot([0, 1], [0, 1], "k--", alpha=0.55, linewidth=1.2)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Observed frequency")
            ax.set_title(f"{scenario} - {best_name}")
            ax.legend(fontsize=7, loc="best")
            ax.text(
                0.02,
                0.98,
                f"{Config.CALIBRATION_BINNING} bins; marker size = bin n",
                transform=ax.transAxes,
                va="top",
                fontsize=7,
            )
        for j in range(n, len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        plt.savefig(
            os.path.join(Config.OUTPUT_DIR, "figures", "diagnostics", "calibration_plots.png"),
            dpi=Config.FIGURE_DPI,
        )
        plt.close()
    except Exception as e:
        logger.log(f"  Calibration plot failed: {e}", level="WARNING")


def create_roc_curves(predictions: Dict):
    if not predictions:
        return
    try:
        scenarios = ordered_scenarios_from_keys(predictions.keys())
        n = len(scenarios)
        if n == 0:
            return
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        if n == 1:
            axes = [axes]
        else:
            axes = np.array(axes).flatten()
        for i, scenario in enumerate(scenarios):
            keys = [k for k in predictions if k.startswith(scenario)]
            if not keys:
                continue
            best_key = max(
                keys,
                key=lambda k: matthews_corrcoef(predictions[k]["y_true"], predictions[k]["y_pred"]),
            )
            yt = predictions[best_key]["y_true"]
            ypr = predictions[best_key]["y_prob"]
            ax = axes[i]
            for c in range(Config.NUM_CLASSES):
                yt_bin = (yt == c).astype(int)
                if yt_bin.sum() == 0 or yt_bin.sum() == len(yt_bin):
                    continue
                try:
                    fpr, tpr, _ = roc_curve(yt_bin, ypr[:, c])
                    auc = roc_auc_score(yt_bin, ypr[:, c])
                    ax.plot(
                        fpr,
                        tpr,
                        label=f"{['Mild', 'Moderate', 'Severe', 'Profound'][c]} AUC={auc:.2f}",
                    )
                except Exception:
                    pass
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title(scenario)
            ax.legend(fontsize=7)
        for j in range(len(scenarios), len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        plt.savefig(
            os.path.join(Config.OUTPUT_DIR, "figures", "diagnostics", "roc_curves.png"),
            dpi=Config.FIGURE_DPI,
        )
        plt.close()
    except Exception as e:
        logger.log(f"  ROC plot failed: {e}", level="WARNING")


def create_diagnostic_figures(all_results: Dict):
    logger.subsection("Generating diagnostic figures")
    results_df = results_collector.to_dataframe()
    create_performance_heatmap(results_df)
    create_confusion_matrix_plot(results_collector.predictions)
    create_feature_importance_plot(results_collector.feature_importances)
    create_calibration_plot(all_results)
    create_roc_curves(results_collector.predictions)


def _bundle_common_metadata(scenario: str, artifacts: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "scenario": scenario,
        "preprocessor": artifacts["preprocessor"],
        "sel_mask": artifacts["sel_mask"],
        "sel_names": artifacts["sel_names"],
        "selection_budget": artifacts.get("selection_budget", dict(Config.FEATURE_BUDGET)),
        "hpo_ancestor_expansion": {
            "enabled": bool(Config.HPO_ANCESTOR_EXPANSION_ENABLED),
            "max_hops": int(Config.HPO_ANCESTOR_MAX_HOPS),
            "excluded_ids": list(Config.HPO_ANCESTOR_EXCLUDED_IDS),
            "ontology_path_at_training": os.path.abspath(Config.HPO_OBO_PATH),
        },
    }


def generate_summary_report(all_results: Dict, pub_outputs: Dict):
    logger.subsection("Writing SUMMARY_REPORT.txt")
    lines = []
    lines.append(f"GENETIC DISEASE SEVERITY PREDICTION PIPELINE v{PIPELINE_VERSION}")
    lines.append(
        f"GENE-DISEASE PAIRS  -  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    lines.append("")
    lines.append("TARGET LABEL PROVENANCE")
    lines.append(f"  {Config.TARGET_LABEL_PROVENANCE}")
    lines.append("")
    lines.append("HEADLINE METRICS PER SCENARIO (validation-selected base model; one test evaluation)")
    pub_df = pub_outputs.get("publication_table", pd.DataFrame())
    best_models = _pick_best_model_per_scenario(all_results)
    for scenario, best_name in best_models.items():
        if "metrics" not in all_results[scenario]["all"].get(best_name, {}):
            continue
        m = all_results[scenario]["all"][best_name]["metrics"]
        ci_str = ""
        if len(pub_df) > 0:
            row = pub_df[(pub_df["Scenario"] == scenario) & (pub_df["Model"] == best_name)]
            if len(row) > 0 and "MCC_CI_low" in row.columns:
                r = row.iloc[0]
                if not pd.isna(r.get("MCC_CI_low", np.nan)):
                    ci_str = f"  (MCC 95% CI: [{r['MCC_CI_low']:.3f}, {r['MCC_CI_high']:.3f}])"
        lines.append(
            f"  {scenario:24s}  {best_name:14s}  MCC={m.get('mcc', 0):.3f}  QWK={m.get('qwk', 0):.3f}  BalAcc={m.get('balanced_accuracy', 0):.3f}{ci_str}"
        )
        rho = m.get("spearman_rho", np.nan)
        if rho is not None and np.isfinite(rho):
            lines.append(
                f"  {'':24s}  {'':14s}  ordinal rank Spearman rho={rho:.3f}  "
                f"AUC(>=Moderate)={m.get('auc_threshold_ge1', float('nan')):.3f}  "
                f"AUC(>=Severe)={m.get('auc_threshold_ge2', float('nan')):.3f}  "
                f"AUC(>=Profound)={m.get('auc_threshold_ge3', float('nan')):.3f}"
            )
    lines.append("")

    gene_level_df = pub_outputs.get("gene_level_evaluation", pd.DataFrame())
    if len(gene_level_df) > 0:
        lines.append("GENE-LEVEL COMPANION EVALUATION (same locked predictions, gene as the unit)")
        lines.append("-" * 80)
        lines.append(
            "  Pair-level metrics above remain the primary comparison. The rows below"
        )
        lines.append(
            "  aggregate the same test predictions to one row per gene, because the"
        )
        lines.append(
            "  non-phenotype feature blocks are gene-level and cannot separate two"
        )
        lines.append(
            "  diseases of the same gene. No model was retrained."
        )
        for scenario, best_name in best_models.items():
            row = gene_level_df[
                (gene_level_df["Scenario"] == scenario)
                & (gene_level_df["Model"] == best_name)
            ]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            lines.append(
                f"  {scenario:24s}  {best_name:14s}  MCC={r['mcc']:.3f}  "
                f"QWK={r['qwk']:.3f}  BalAcc={r['balanced_accuracy']:.3f}  "
                f"(n_genes={int(r['n_genes'])}, "
                f"{r['frac_pairs_in_multi_severity_genes']:.1%} of pairs in "
                f"multi-severity genes)"
            )
        lines.append("")

    baselines = pub_outputs.get("baselines", {})
    if baselines:
        lines.append("NAIVE BASELINES (test set)")
        lines.append("-" * 80)
        for bn, bres in baselines.items():
            m = bres["metrics"]
            lines.append(
                f"  {bn:20s}  MCC={m.get('mcc', 0):.3f}  QWK={m.get('qwk', 0):.3f}  BalAcc={m.get('balanced_accuracy', 0):.3f}"
            )
        lines.append("")
    shared_budget_df = pub_outputs.get("shared_hpo_budget_selection", pd.DataFrame())
    if len(shared_budget_df) > 0 and "selected_for_s2_and_s3" in shared_budget_df.columns:
        chosen = shared_budget_df[shared_budget_df["selected_for_s2_and_s3"].astype(bool)]
        if len(chosen) == 1:
            lines.append(
                f"SHARED S2/S3 HPO BUDGET: {int(chosen.iloc[0]['hpo_binary_budget'])} "
                "(selected by S3 training-only nested CV)"
            )
            lines.append("")
    perm_df = pub_outputs.get("permutation_test", pd.DataFrame())
    if len(perm_df) > 0:
        lines.append("PERMUTATION TEST (one-sided p, label-shuffling null)")
        lines.append("-" * 80)
        for _, r in perm_df.iterrows():
            lines.append(
                f"  {r['Scenario']:24s} observed MCC={r['observed_mcc']:.3f}  perm mean={r['permuted_mcc_mean']:.3f}  p={r['empirical_p_value']:.4f}"
            )
        lines.append("")
    seed_summary = pub_outputs.get("seed_sensitivity_summary", pd.DataFrame())
    if len(seed_summary) > 0:
        lines.append(f"SEED SENSITIVITY (n_seeds={Config.N_SEEDS_SENSITIVITY})")
        lines.append("-" * 80)
        for _, r in seed_summary.iterrows():
            lines.append(
                f"  {r['Scenario']:24s} {r['Model']:14s}  MCC={r['mcc_mean']:.3f} +/- {r['mcc_std']:.3f}   QWK={r['qwk_mean']:.3f} +/- {r['qwk_std']:.3f}"
            )
        lines.append("")
    ece_df = pub_outputs.get("ece", pd.DataFrame())
    if len(ece_df) > 0:
        lines.append("EXPECTED CALIBRATION ERROR (per class, best model per scenario)")
        lines.append("-" * 80)
        for scenario in ece_df["Scenario"].unique():
            sub = ece_df[ece_df["Scenario"] == scenario]
            if "calibration_status" in sub.columns:
                sub = sub[sub["calibration_status"] == "calibrated"]
            mn = sub["Model"].iloc[0]
            ece_str = "  ".join(
                (f"{r['severity_class']}={r['ece']:.3f}" for _, r in sub.iterrows())
            )
            lines.append(f"  {scenario:24s} {mn:14s}  {ece_str}")
        lines.append("")
    lines.append("KEY OUTPUT FILES")
    lines.append("  results/all_results.csv               selected-model test metrics only")
    lines.append("  results/model_selection_validation.csv validation-only candidate comparison")
    lines.append("  results/publication_table.csv         selected-model test metrics with bootstrap CIs")
    lines.append("  results/bootstrap_cis.csv             gene-cluster bootstrap CIs")
    lines.append("  results/error_decomposition.csv       global correct / adj / non-adj")
    lines.append("  results/error_decomposition_per_class.csv  per-true-class breakdown")
    lines.append("  results/gene_level_evaluation.csv     gene-unit companion scores (no retraining)")
    lines.append("  results/calibration_metrics.csv       class-wise ECE/Brier with bootstrap CIs")
    lines.append("  results/calibration_bins.csv          reliability-bin data for plotting")
    lines.append("  results/calibration_ece.csv           backward-compatible calibration table")
    lines.append("  results/split_sensitivity_fixed_configuration.csv  per-seed fixed-configuration metrics")
    lines.append("  results/split_sensitivity_fixed_configuration_summary.csv  mean +/- std over splits")
    lines.append("  results/mi_threshold_sensitivity.csv  validation-only S2/S3 MI sweep")
    lines.append("  results/shared_hpo_budget_training_cv_selection.csv  shared S2/S3 budget choice")
    lines.append("  results/shared_hpo_feature_audit.csv  exact S2/S3 HPO-set equality audit")
    lines.append("  results/annotation_gene_level_invariance_audit.csv  gene-level ANNOTATION collapse audit")
    lines.append("  results/permutation_test.csv          validation-only gene-block shuffled-label null")
    lines.append("  results/severity_hpo_term_sensitivity.csv  validation-only severity-HPO ablation")
    lines.append("  results/severity_hpo_terms_removed.csv exact severity-related HPO terms removed")
    lines.append("  results/high_mi_provenance_flags.csv  every high-MI feature name and provenance")
    lines.append("  results/generalisation_evaluations.csv disease- and joint-disjoint results")
    lines.append("  results/feature_audit_<scenario>.csv  drops plus MI provenance-review flags")
    lines.append("  results/split_summary.csv             realised train/val/test counts")
    lines.append(
        "  results/naive_baselines.csv           majority / stratified-random / ordinal-mean"
    )
    lines.append("  per_disease_explanations/ordinal_boundary_explanations_*.csv  boundary SHAP only")
    lines.append("  figures/diagnostics/*.png             diagnostic plots")
    lines.append("  pipeline_manifest.json                run provenance")
    lines.append("  RUN_COMPLETE.json                     successful required-artifact completion marker")
    lines.append("")
    out_path = os.path.join(Config.OUTPUT_DIR, "SUMMARY_REPORT.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.log(f"  Wrote {out_path}")


def run_publication_analyses(all_results: Dict, container: DataContainer):
    logger.section("PUBLICATION ANALYSES")
    baselines = train_naive_baselines(container)
    bootstrap_df = run_bootstrap_all(all_results)
    global_err_df, per_class_err_df = analyze_error_types(all_results)
    gene_level_df = evaluate_at_gene_level(all_results, container)
    ece_df = compute_ece_table(all_results)
    pub_table = create_publication_table(all_results, bootstrap_df)
    seed_df, seed_summary_df = run_seed_sensitivity(container, all_results)
    mi_df = run_mi_threshold_sensitivity(container, all_results)
    budget_df = run_budget_sensitivity(container, all_results)
    perm_df = run_permutation_test(container, all_results)
    create_diagnostic_figures(all_results)
    outputs = {
        "baselines": baselines,
        "bootstrap": bootstrap_df,
        "errors_global": global_err_df,
        "errors_per_class": per_class_err_df,
        "gene_level_evaluation": gene_level_df,
        "ece": ece_df,
        "publication_table": pub_table,
        "seed_sensitivity": seed_df,
        "seed_sensitivity_summary": seed_summary_df,
        "mi_threshold_sensitivity": mi_df,
        "budget_sensitivity": budget_df,
        "permutation_test": perm_df,
    }
    # Both of the following reuse stored predictions/split indices only. Neither
    # refits a model nor touches the single-test-evaluation registry.
    outputs["paired_scenario_contrasts"] = run_paired_scenario_contrasts(
        all_results, container
    )
    outputs["disease_overlap_audit"] = run_disease_overlap_audit(container)
    outputs["imbalance_sensitivity"] = run_imbalance_sensitivity(container, all_results)
    outputs["severity_hpo_term_sensitivity"] = run_severity_hpo_term_sensitivity(
        container, all_results
    )
    outputs["high_mi_provenance_flags"] = write_high_mi_provenance_report()
    outputs["generalisation_evaluations"] = pd.read_csv(
        os.path.join(Config.OUTPUT_DIR, "results", "generalisation_evaluations.csv")
    ) if os.path.exists(os.path.join(Config.OUTPUT_DIR, "results", "generalisation_evaluations.csv")) else pd.DataFrame()
    shared_budget_path = os.path.join(
        Config.OUTPUT_DIR, "results", "shared_hpo_budget_training_cv_selection.csv"
    )
    shared_audit_path = os.path.join(
        Config.OUTPUT_DIR, "results", "shared_hpo_feature_audit.csv"
    )
    outputs["shared_hpo_budget_selection"] = (
        pd.read_csv(shared_budget_path) if os.path.exists(shared_budget_path) else pd.DataFrame()
    )
    outputs["shared_hpo_feature_audit"] = (
        pd.read_csv(shared_audit_path) if os.path.exists(shared_audit_path) else pd.DataFrame()
    )
    return outputs


def _sha256_file(path: str) -> Optional[str]:
    if not path or not os.path.exists(path) or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _obo_metadata(path: str) -> Dict[str, Any]:
    metadata = {"path": os.path.abspath(path), "sha256": _sha256_file(path),
                "data_version": None, "date": None}
    if not os.path.exists(path):
        return metadata
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line_index, line in enumerate(handle):
            if line.startswith("data-version:"):
                metadata["data_version"] = line.split(":", 1)[1].strip()
            elif line.startswith("date:"):
                metadata["date"] = line.split(":", 1)[1].strip()
            if line_index > 200:
                break
    return metadata


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except Exception:
        return None


def write_pipeline_manifest(container: DataContainer, all_results: Dict, pub_outputs: Dict):
    """Record input and source hashes, fitted settings, selected features and label provenance."""
    config_dump: Dict[str, Any] = {}
    for key, value in vars(Config).items():
        if key.startswith("_") or callable(value):
            continue
        if isinstance(value, (str, int, float, bool, list, tuple, dict)) or value is None:
            config_dump[key] = value
    inputs = {
        "primary_data": Config.PRIMARY_DATA,
        "rebuilt_gene_disease_mapping": Config.REBUILT_GENE_DISEASE_MAPPING,
        "annotation_data": Config.ANNOTATION_DATA,
        "core_features": Config.CORE_FEATURES,
        "domain_features": Config.DOMAIN_FEATURES,
        "external_features": Config.EXTERNAL_FEATURES,
        "new_gene_features": Config.NEW_GENE_FEATURES,
        "mouse_lethality": Config.MOUSE_LETHALITY_FEATURES,
        "mouse_human_phenotype": Config.MOUSE_HUMAN_PHENOTYPE_FEATURES,
    }
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "source_release": "journal-cleanup-20261104",
        "source_files": {
            name: _sha256_file(os.path.join(os.path.dirname(__file__), name))
            for name in ("severity_pipeline.py", "gene_disease_mapping_filter.py",
                         "publication_figures.py", "run_report.py")
        },
        "run_timestamp": datetime.now().isoformat(),
        "git_commit_hash": get_git_commit_hash(),
        "workflow": {
            "outer_split": "optimised group-disjoint 65/15/20 train/validation/test",
            "feature_vocabulary": "learned on each training fold only",
            "annotation_representation": "gene-level union of functional annotations; disease identifiers excluded",
            "shared_hpo_budget": (
                "selected on S3 training-only nested CV and frozen identically for S2 and S3"
            ),
            "feature_selection": "repeated inside nested grouped training CV",
            "hyperparameter_tuning": "nested grouped training CV",
            "model_selection": Config.MODEL_SELECTION_RULE,
            "calibration": "adaptive isotonic-or-sigmoid fitted after validation selection; ordinal boundaries are atomic with explicit raw fallback",
            "test_policy": "single evaluation of one frozen model/calibrator per scenario",
            "bootstrap": "gene-cluster bootstrap",
            "permutation": Config.PERMUTATION_MODE,
            "missingness": (
                "pair-preserving left joins; unknown values preserved until training-fold median imputation; "
                f"missing-value indicators enabled={Config.ADD_MISSING_INDICATORS}; "
                f"source-availability indicators enabled={Config.ADD_SOURCE_AVAILABILITY_INDICATORS}"
            ),
            "mi_leakage_policy": "MI is selection/audit only; leakage exclusions use provenance rules",
            "ordinal_explanations": Config.ORDINAL_EXPLANATION_POLICY,
        },
        "target_label_provenance": Config.TARGET_LABEL_PROVENANCE,
        "input_files": {
            label: {"path": os.path.abspath(path), "sha256": _sha256_file(path)}
            for label, path in inputs.items()
        },
        "ontologies": {
            "HPO": _obo_metadata(Config.HPO_OBO_PATH),
            "MONDO": _obo_metadata(Config.MONDO_OBO_PATH),
            "automatic_download_allowed": Config.ALLOW_ONTOLOGY_DOWNLOAD,
        },
        "dataset": {
            "n_pairs_total": int(len(container.full_df)),
            "n_genes_total": int(len(np.unique(container.groups))),
            "full_class_distribution": np.bincount(container.y, minlength=Config.NUM_CLASSES).tolist(),
        },
        "selected_models": _pick_best_model_per_scenario(all_results),
        "selected_model_calibration": {
            scenario: {
                "model": model_name,
                "status": all_results[scenario]["all"][model_name].get(
                    "calibration_status", "not_attempted"
                ),
                "success": bool(
                    all_results[scenario]["all"][model_name].get(
                        "calibration_success", False
                    )
                ),
                "method": all_results[scenario]["all"][model_name].get(
                    "calibration_method", "none"
                ),
                "method_attempted": all_results[scenario]["all"][model_name].get(
                    "calibration_method_attempted", "none"
                ),
                "failure_details": all_results[scenario]["all"][model_name].get(
                    "calibration_failure_details", ""
                ),
            }
            for scenario, model_name in _pick_best_model_per_scenario(
                all_results
            ).items()
        },
        "selected_features": {
            scenario: list(artifact.get("selected_names", []))
            for scenario, artifact in _fold_artifacts.items()
        },
        "scenario_scope_exclusions": {
            "S1_Gene_Level": {
                "organ_criticality_columns_removed": (
                    list(Config.ORGAN_CRITICALITY_COLUMNS)
                ),
                "inheritance_count_proxy_columns_removed": list(
                    getattr(container, "inheritance_proxy_drops_S1", [])
                ),
                "rationale": (
                    "organ-criticality columns share a source with the severity label; "
                    "n_inheritance_modes and has_mixed_inheritance are gene-level "
                    "rollups over the gene's disease set and act as disease-count "
                    "proxies, which is out of scope for a gene-intrinsic scenario. "
                    "Mode-of-inheritance columns are retained."
                ),
            },
            "all_scenarios": {
                "duplicate_columns_removed": list(
                    getattr(container, "duplicate_column_drops", [])
                ),
                "rationale": (
                    "gene_age is an exact linear function of conservation_score. "
                    "Other broken or duplicate inputs are removed at load; see "
                    "removed_input_columns."
                ),
                "removed_input_columns": dict(
                    getattr(Config, "REMOVED_INPUT_COLUMNS", {})
                ),
                "core_inheritance_rollups_removed": list(
                    getattr(Config, "CORE_INHERITANCE_ROLLUP_COLUMNS", [])
                ),
                "inheritance": (
                    "pair-specific, from the primary file column "
                    f"{getattr(Config, 'PAIR_INHERITANCE_COLUMN', 'moi_title')}"
                    if getattr(Config, "USE_PAIR_LEVEL_INHERITANCE", False)
                    else "gene-level rollup from the CORE file"
                ),
            },
        },
        "unweighted_tuning_fallbacks": list(_UNWEIGHTED_TUNING_FALLBACKS),
        "random_seeds": {
            "primary": RANDOM_SEED,
            "split_sensitivity": list(range(
                RANDOM_SEED + Config.SPLIT_SENSITIVITY_SEED_OFFSET,
                RANDOM_SEED + Config.SPLIT_SENSITIVITY_SEED_OFFSET + Config.N_SEEDS_SENSITIVITY,
            )),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "packages": {
                name: _package_version(name) for name in [
                    "numpy", "pandas", "scikit-learn", "lightgbm",
                    "shap", "imbalanced-learn", "scipy", "matplotlib",
                ]
            },
        },
        "config": config_dump,
    }
    with open(os.path.join(Config.OUTPUT_DIR, "pipeline_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)


def save_per_disease_explanations(container: DataContainer, all_results: Dict):
    if not HAS_SHAP:
        logger.log("  SHAP not available; per-disease explanations skipped", level="WARNING")
        return
    logger.section("SAVING PER-DISEASE EXPLANATIONS FROM STORED FINAL TEST MATRICES")
    output_dir = os.path.join(Config.OUTPUT_DIR, "per_disease_explanations")
    os.makedirs(output_dir, exist_ok=True)
    metadata = container.metadata.reset_index(drop=True)
    severity_names = ["Mild", "Moderate", "Severe", "Profound"]

    for scenario, model_name in _pick_best_model_per_scenario(all_results).items():
        artifact = _fold_artifacts.get(scenario)
        result = all_results[scenario]["all"].get(model_name, {})
        if artifact is None or "X_test" not in artifact or "predictions" not in result:
            continue
        X_test = np.asarray(artifact["X_test"])
        feature_names = list(artifact["selected_names"])
        test_idx = np.asarray(artifact["test_idx"])
        model = result["model"]
        probabilities = np.asarray(result["predictions"]["y_prob"])
        predictions = np.asarray(result["predictions"]["y_pred"], dtype=int)
        true_labels = np.asarray(result["y_true"], dtype=int)
        if len(probabilities) != len(X_test):
            continue

        if model_name == "OrdinalDecomp":
            boundary_names = [
                "Moderate_or_higher",
                "Severe_or_higher",
                "Profound",
            ]
            boundary_records: List[Dict[str, Any]] = []
            try:
                for threshold, boundary_model in enumerate(model):
                    values = shap.TreeExplainer(boundary_model).shap_values(X_test)
                    if isinstance(values, list):
                        values = values[-1]
                    values = np.asarray(values)
                    if values.ndim == 3 and values.shape[-1] == 2:
                        values = values[..., 1]
                    if values.ndim != 2:
                        raise ValueError(
                            f"Unexpected ordinal-boundary SHAP shape: {values.shape}"
                        )
                    boundary_probability = boundary_model.predict_proba(X_test)[:, 1]
                    for row_number in range(len(X_test)):
                        signed = np.asarray(values[row_number])
                        absolute = np.abs(signed)
                        top_indices = np.argsort(absolute)[::-1][:10]
                        meta = metadata.iloc[test_idx[row_number]]
                        boundary_records.append({
                            "gene_symbol": meta.get("gene_symbol", ""),
                            "mondo_id": meta.get("mondo_id", ""),
                            "disease_name": meta.get("disease_name", ""),
                            "boundary_index": threshold,
                            "boundary_name": boundary_names[threshold],
                            "boundary_target": f"severity_numeric>{threshold}",
                            "boundary_probability_uncalibrated": float(
                                boundary_probability[row_number]
                            ),
                            "final_calibrated_prediction": severity_names[
                                int(predictions[row_number])
                            ],
                            "true_severity": severity_names[int(true_labels[row_number])],
                            "explanation_target": "uncalibrated_ordinal_boundary_model",
                            "top_features_json": json.dumps([
                                {
                                    "name": feature_names[index],
                                    "shap_signed": float(signed[index]),
                                    "shap_abs": float(absolute[index]),
                                    "value": float(X_test[row_number, index]),
                                }
                                for index in top_indices if index < len(feature_names)
                            ]),
                        })
            except Exception as exc:
                logger.log(
                    f"  Boundary SHAP failed for {scenario}/{model_name}: {exc}",
                    level="WARNING",
                )
                continue
            pd.DataFrame(boundary_records).to_csv(
                os.path.join(output_dir, f"ordinal_boundary_explanations_{scenario}.csv"),
                index=False,
            )
            logger.log(
                f"  Saved boundary-specific, uncalibrated ordinal explanations for {scenario}; "
                "no predicted-class SHAP file was emitted."
            )
            continue

        try:
            if model_name in ("LightGBM", "ExtraTrees"):
                tree_explainer = shap.TreeExplainer(model)
                raw_values = tree_explainer.shap_values(X_test)
                shap_per_class = _normalise_multiclass_shap_values(
                    raw_values, X_test.shape[1]
                )
                if model_name == "ExtraTrees":
                    # Extra Trees SHAP is on the probability scale, so the
                    # severity-score SHAP (save_severity_shap) reuses it.
                    _CLASS_SHAP_CACHE[scenario] = {
                        "model_name": model_name,
                        "per_class": shap_per_class,
                        "expected_value": np.asarray(
                            tree_explainer.expected_value, dtype=float
                        ).ravel(),
                    }
            elif model_name == "MultinomialLogisticRegression":
                fitted_logistic = _get_multinomial_logistic_estimator(model)
                X_train_linear = _transform_for_multinomial_logistic(
                    model, np.asarray(artifact["X_train"])
                )
                X_test_linear = _transform_for_multinomial_logistic(model, X_test)
                raw_values = shap.LinearExplainer(
                    fitted_logistic, X_train_linear
                ).shap_values(X_test_linear)
                shap_per_class = _normalise_multiclass_shap_values(
                    raw_values, X_test.shape[1]
                )
            else:
                continue
        except Exception as exc:
            logger.log(f"  SHAP failed for {scenario}/{model_name}: {exc}", level="WARNING")
            continue

        records: List[Dict[str, Any]] = []
        for row_number in range(len(X_test)):
            predicted_class = int(predictions[row_number])
            if len(shap_per_class) >= Config.NUM_CLASSES:
                signed = np.asarray(shap_per_class[predicted_class][row_number])
            else:
                signed = np.asarray(shap_per_class[0][row_number])
            absolute = np.abs(signed)
            top_indices = np.argsort(absolute)[::-1][:10]
            top_features = [
                {
                    "name": feature_names[index],
                    "shap_signed": float(signed[index]),
                    "shap_abs": float(absolute[index]),
                    "value": float(X_test[row_number, index]),
                }
                for index in top_indices if index < len(feature_names)
            ]
            meta = metadata.iloc[test_idx[row_number]]
            records.append({
                "gene_symbol": meta.get("gene_symbol", ""),
                "mondo_id": meta.get("mondo_id", ""),
                "disease_name": meta.get("disease_name", ""),
                "true_severity": severity_names[int(true_labels[row_number])],
                "predicted_severity": severity_names[predicted_class],
                "p_mild": float(probabilities[row_number, 0]),
                "p_moderate": float(probabilities[row_number, 1]),
                "p_severe": float(probabilities[row_number, 2]),
                "p_profound": float(probabilities[row_number, 3]),
                "correct": bool(predicted_class == true_labels[row_number]),
                "adjacent_correct": bool(abs(predicted_class - true_labels[row_number]) <= 1),
                "explanation_target": "uncalibrated_base_model_class_output",
                "displayed_prediction_source": "calibrated_final_model",
                "top_features_json": json.dumps(top_features),
            })
        pd.DataFrame(records).to_csv(
            os.path.join(output_dir, f"explanations_{scenario}.csv"), index=False
        )
        try:
            results_collector.add_shap_values(
                f"{scenario}_{model_name}", shap_per_class, feature_names
            )
        except Exception:
            pass


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
        "has_lethal_phenotype", "feat_lethal_x_pli", "feat_lethal_x_essential",
        "feat_has_lethal_x_pli",
    }:
        return "Mouse knockout"
    if low in {
        "feat_pli_x_dominant", "feat_pli_x_recessive",
        "feat_dominant_low_constraint", "feat_recessive_high_constraint",
    }:
        return "Constraint by inheritance"
    if low.startswith("feat_is_") or "inheritance" in low:
        return "Inheritance"
    if low.startswith(("clingen_", "hi_sensitive", "ts_sensitive")) or low in {
        "phaplo", "ptriplo", "is_haploinsufficient", "is_triplosensitive",
        "dosage_sensitive_any", "ordered_dosage_sensitive",
    }:
        return "Dosage sensitivity"
    if low in {"gnomad_gene_length", "gnomad_cds_length", "af_protein_length"}:
        return "Gene and protein size"
    if low.startswith(("gnomad_", "shet_", "gdi_")) or low in {
        "domino_score", "feat_pli_squared", "feat_missz_x_pli", "feat_missz_minus_synz",
        "feat_triple_constraint", "feat_pli_over_loeuf",
    }:
        return "Constraint"
    if low.startswith("depmap_") or low in {
        "feat_essential_constrained", "feat_essential_over_loeuf",
        "feat_central_essential", "feat_hub_essential",
    }:
        return "Cellular essentiality"
    if low.startswith(("network_", "ppi_")) or low in {
        "feat_log_ppi_degree", "feat_hub_constrained", "feat_betweenness_x_pli",
    }:
        return "Interaction network"
    if low.startswith("brainspan_"):
        return "Brain expression (BrainSpan)"
    if low.startswith(("gtex_", "hpa_")):
        return "Tissue expression (GTEx, HPA)"
    if low.startswith("af_") or low == "feat_conserved_disordered":
        return "Protein structure"
    if "ortholog" in low or low in {
        "conservation_score", "is_ancient", "gene_age", "feat_conserved_constrained",
    }:
        return "Conservation"
    if "paralog" in low or low in {"redundancy_score", "is_singleton"}:
        return "Paralogues and redundancy"
    if low.startswith(("var_", "feat_density_", "feat_prop_")) or low == "feat_ratio_lof_missense":
        return "Reported variants"
    if low.startswith(("feat_annot_", "feat_primary_", "has_annot_", "has_primary_")):
        return "Functional annotations"
    return "Other"


def _merge_missing_indicator_columns(
    shap_matrix: np.ndarray,
    value_matrix: np.ndarray,
    feature_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, List[str]]]:
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
            # Parent not among the selected features: keep the indicator as its own row.
            out_index[name] = len(out_names)
            out_names.append(name)
            shap_columns.append(np.asarray(shap_matrix[:, j], dtype=float).copy())
            value_columns.append(np.asarray(value_matrix[:, j], dtype=float))
            merged_from[name] = [name]
    return (
        np.column_stack(shap_columns),
        np.column_stack(value_columns),
        out_names,
        merged_from,
    )


def _severity_shap_matrix(
    scenario: str,
    model_name: str,
    model: Any,
    X_test: np.ndarray,
    X_train: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, Dict[str, Any]]:
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
        # Distance from the expected class of monotone probabilities; zero without crossings.
        class_probabilities = ordinal_cumulative_to_class_probs(cumulative_matrix)
        gap = np.abs(cumulative_matrix.sum(axis=1) - class_probabilities @ weights)
        return (
            total,
            np.full(len(X_test), base),
            cumulative_matrix.sum(axis=1),
            "interventional TreeExplainer (probability) summed over the three ordinal boundaries",
            {"boundary_crossing_fraction": crossing,
             "mean_abs_gap_to_expected_class": float(np.mean(gap)),
             "max_abs_gap_to_expected_class": float(np.max(gap)),
             "background_n": int(len(reference))},
        )

    if model_name == "ExtraTrees":
        cached = _CLASS_SHAP_CACHE.get(scenario)
        if (
            cached
            and cached.get("model_name") == "ExtraTrees"
            and len(cached.get("per_class", [])) == Config.NUM_CLASSES
        ):
            per_class = cached["per_class"]
            expected = np.asarray(cached["expected_value"], dtype=float).ravel()
        else:
            explainer = shap.TreeExplainer(model)
            per_class = _normalise_multiclass_shap_values(
                explainer.shap_values(X_test), X_test.shape[1]
            )
            expected = np.asarray(explainer.expected_value, dtype=float).ravel()
        values = sum(
            weights[k] * np.asarray(per_class[k], dtype=float)
            for k in range(Config.NUM_CLASSES)
        )
        base = float(np.dot(weights, expected[: Config.NUM_CLASSES]))
        explained = np.asarray(model.predict_proba(X_test), dtype=float) @ weights
        return (
            values,
            np.full(len(X_test), base),
            explained,
            "tree-path-dependent TreeExplainer (probability), expected severity class",
            {},
        )

    # Other probabilistic models (LightGBM multiclass, logistic regression): SHAP's
    # tree "probability" output is not additive for softmax models, so a
    # model-agnostic permutation explainer is used on the expected severity class.
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
    rows: List[Dict[str, Any]] = []
    for j, name in enumerate(names):
        s = np.asarray(shap_matrix[:, j], dtype=float)
        v = np.asarray(value_matrix[:, j], dtype=float)
        finite = np.isfinite(v) & np.isfinite(s)
        rho = np.nan
        if finite.sum() > 2 and np.std(v[finite]) > 0 and np.std(s[finite]) > 0:
            rho = float(pd.Series(v[finite]).corr(pd.Series(s[finite]), method="spearman"))
        high = np.zeros(len(v), dtype=bool)
        if finite.any():
            observed = v[finite]
            median = float(np.median(observed))
            high = finite & (v > median)
            if not high.any() and float(np.max(observed)) > float(np.min(observed)):
                high = finite & (v >= float(np.max(observed)))
        low = finite & ~high
        rows.append({
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
            "share_high_value_raises_severity": (
                float(np.mean(s[high] > 0)) if high.any() else np.nan
            ),
            "mean_shap_high_value": float(np.mean(s[high])) if high.any() else np.nan,
            "mean_shap_low_value": float(np.mean(s[low])) if low.any() else np.nan,
        })
    frame = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
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
    groups = np.asarray([severity_shap_group(name) for name in names])
    rows: List[Dict[str, Any]] = []
    for group in pd.unique(groups):
        columns = np.where(groups == group)[0]
        group_shap = np.asarray(shap_matrix[:, columns], dtype=float).sum(axis=1)
        members = feature_frame[feature_frame["group"] == group]
        top = members.iloc[0] if len(members) else None
        rows.append({
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
        })
    frame = pd.DataFrame(rows)
    total = float(frame["mean_abs_group_shap"].sum()) or 1.0
    frame["share_of_total"] = frame["mean_abs_group_shap"] / total
    frame = frame.sort_values("mean_abs_group_shap", ascending=False).reset_index(drop=True)
    frame.insert(3, "rank", np.arange(1, len(frame) + 1))
    return frame


def save_severity_shap(container: DataContainer, all_results: Dict) -> None:
    """Full-matrix SHAP on the expected severity class for each selected model."""
    if not HAS_SHAP or not getattr(Config, "SEVERITY_SHAP_ENABLED", False):
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
            logger.log(
                f"  Severity SHAP failed for {scenario}/{model_name}: {exc}", level="WARNING"
            )
            continue
        additivity_error = float(np.max(np.abs(values.sum(axis=1) + base - explained)))
        merged_shap, merged_values, merged_names, merged_from = _merge_missing_indicator_columns(
            values, X_test, names
        )
        feature_frame = _severity_feature_summary(
            scenario, model_name, merged_shap, merged_values, merged_names, merged_from
        )
        group_frame = _severity_group_summary(
            scenario, model_name, merged_shap, merged_names, feature_frame
        )
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
        feature_frame.to_csv(
            os.path.join(results_dir, f"severity_shap_features_{scenario}.csv"), index=False
        )
        group_frame.to_csv(
            os.path.join(results_dir, f"severity_shap_groups_{scenario}.csv"), index=False
        )
        feature_frames.append(feature_frame)
        group_frames.append(group_frame)
        seconds = (datetime.now() - started).total_seconds()
        checks.append({
            "Scenario": scenario,
            "Model": model_name,
            "method": method,
            "n_pairs": int(len(X_test)),
            "n_columns": int(len(names)),
            "n_features_after_merging_missing_flags": int(len(merged_names)),
            "max_additivity_error": additivity_error,
            "seconds": seconds,
            **extras,
        })
        logger.log(
            f"  {scenario}/{model_name}: {method}; {len(X_test)} pairs x {len(names)} columns; "
            f"max |sum(SHAP) + base - prediction| = {additivity_error:.2e}; {seconds:.0f} s"
        )
    if feature_frames:
        pd.concat(feature_frames, ignore_index=True).to_csv(
            os.path.join(results_dir, "severity_shap_features.csv"), index=False
        )
        pd.concat(group_frames, ignore_index=True).to_csv(
            os.path.join(results_dir, "severity_shap_groups.csv"), index=False
        )
        pd.DataFrame(checks).to_csv(
            os.path.join(results_dir, "severity_shap_checks.csv"), index=False
        )


def ordered_scenarios_from_keys(keys) -> List[str]:
    preferred = ["S1_Gene_Level", "S1b_Disease_Context", "S2_Combined", "S3_Phenotype"]
    found: set = set()
    for raw_key in keys:
        key = str(raw_key)
        if key in preferred:
            found.add(key)
            continue
        for scenario in preferred:
            if key.startswith(scenario + "_") or key.startswith(scenario + ":"):
                found.add(scenario)
                break
    ordered = [scenario for scenario in preferred if scenario in found]
    ordered.extend(sorted(found - set(ordered)))
    return ordered


def save_all_base_model_bundles(container: DataContainer, all_results: Dict) -> None:
    """Save individual model bundles only; no unevaluated S3 ensemble is emitted."""
    if not getattr(Config, "SAVE_ALL_BASE_MODEL_BUNDLES", True):
        return
    logger.section("SAVING INDIVIDUAL MODEL BUNDLES")
    model_dir = os.path.join(Config.OUTPUT_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    allowed = _available_candidate_names()
    for scenario, strata in all_results.items():
        artifact = _fold_artifacts.get(scenario)
        if artifact is None:
            continue
        common = _bundle_common_metadata(scenario, artifact)
        common["text_vocabulary"] = artifact.get("text_vocabulary", {})
        for model_name in allowed:
            result = strata.get("all", {}).get(model_name)
            if not result or result.get("model") is None:
                continue
            payload = dict(common)
            payload.update({
                "model": result["model"],
                "calibrated_model": result.get("calibrated_model"),
                "calibrated_models": result.get("calibrated_models"),
                "model_name": model_name,
                "validation_metrics": result.get("val_metrics", {}),
                "test_metrics_calibrated": result.get("metrics") if result.get("test_evaluated") else None,
                "test_metrics_uncalibrated": result.get("test_metrics_uncalibrated") if result.get("test_evaluated") else None,
                "selected_on_validation": result.get("selected_on_validation", False),
                "test_evaluated": result.get("test_evaluated", False),
                "best_params": result.get("best_params", {}),
                "calibration_method": result.get("calibration_method", "none"),
                "imbalance_strategy": Config.IMBALANCE_STRATEGY,
            })
            path = os.path.join(model_dir, f"{scenario}_{model_name}.pkl")
            with open(path, "wb") as handle:
                pickle.dump(payload, handle)
            logger.log(f"  Saved {path}")


def _require_fresh_output_directory() -> None:
    if getattr(Config, "ALLOW_OUTPUT_REUSE", False):
        return
    existing_files: List[str] = []
    if os.path.isdir(Config.OUTPUT_DIR):
        for root, _, files in os.walk(Config.OUTPUT_DIR):
            for filename in files:
                existing_files.append(os.path.join(root, filename))
                if len(existing_files) >= 5:
                    break
            if len(existing_files) >= 5:
                break
    if existing_files:
        raise FileExistsError(
            "Refusing to reuse a non-empty output directory. Choose a new "
            f"Config.OUTPUT_DIR before running: {Config.OUTPUT_DIR}. Sample existing "
            f"files: {existing_files}"
        )


def _validate_csv_schema(
    path: str,
    label: str,
    required_alias_groups: Optional[List[Sequence[str]]] = None,
) -> None:
    try:
        header = pd.read_csv(path, nrows=5, low_memory=False)
    except Exception as exc:
        raise RuntimeError(
            f"Preflight could not parse {label} CSV at {path}: {type(exc).__name__}: {exc}"
        ) from exc
    if len(header.columns) == 0:
        raise ValueError(f"Preflight found no columns in {label}: {path}")
    if required_alias_groups:
        available = {str(column) for column in header.columns}
        missing_groups = [
            list(group) for group in required_alias_groups
            if not any(alias in available for alias in group)
        ]
        if missing_groups:
            raise ValueError(
                f"{label} is missing required column aliases {missing_groups}; "
                f"available columns begin with {list(header.columns[:20])}."
            )


def validate_preflight_inputs_and_reporting() -> None:
    """Fail before data assembly/training when inputs or reporting code are invalid."""
    logger.section("PREFLIGHT VALIDATION")
    required_csvs = {
        "primary_data": (
            Config.PRIMARY_DATA,
            [["Gene Name", "gene_symbol"], ["MONDO_ID", "mondo_id"], ["Severity", "target_severity"]],
        ),
        "annotation_data": (Config.ANNOTATION_DATA, [["Gene Name", "gene_symbol"]]),
        "core_features": (Config.CORE_FEATURES, [["gene_symbol"]]),
    }
    optional_csvs = {
        "domain_features": (Config.DOMAIN_FEATURES, [["Gene", "gene_symbol"]]),
        "external_features": (Config.EXTERNAL_FEATURES, [["gene_symbol"]]),
        "new_gene_features": (Config.NEW_GENE_FEATURES, [["gene_symbol"]]),
        "mouse_lethality": (Config.MOUSE_LETHALITY_FEATURES, [["gene_symbol"]]),
        "mouse_human_phenotype": (
            Config.MOUSE_HUMAN_PHENOTYPE_FEATURES,
            [["gene_symbol"]],
        ),
    }
    failures: List[str] = []
    if Config.APPLY_REBUILT_GENE_DISEASE_MAPPING:
        mapping_path = _resolve_gene_disease_mapping_path(
            Config.REBUILT_GENE_DISEASE_MAPPING, Config.BASE_DIR
        )
        if not mapping_path:
            failures.append("required HPO-rebuilt gene-disease mapping is missing")
        else:
            try:
                _validate_csv_schema(mapping_path, "rebuilt_mapping", [
                    ["gene_symbol"], ["mondo_id"], ["status"], ["use_in_rebuilt_corpus"]
                ])
            except Exception as exc:
                failures.append(str(exc))
    for label, (path, aliases) in required_csvs.items():
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            failures.append(f"missing or empty required input {label}: {os.path.abspath(path)}")
            continue
        try:
            _validate_csv_schema(path, label, aliases)
            logger.log(f"  Preflight OK: {label} -> {os.path.abspath(path)}")
        except Exception as exc:
            failures.append(str(exc))
    for label, (path, aliases) in optional_csvs.items():
        if not os.path.exists(path):
            logger.log(
                f"  Optional input absent: {label} -> {os.path.abspath(path)}",
                level="WARNING",
            )
            continue
        try:
            _validate_csv_schema(path, label, aliases)
            logger.log(f"  Preflight OK: {label} -> {os.path.abspath(path)}")
        except Exception as exc:
            failures.append(str(exc))

    for label, path in {"HPO ontology": Config.HPO_OBO_PATH, "MONDO ontology": Config.MONDO_OBO_PATH}.items():
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            failures.append(f"missing or empty required {label}: {os.path.abspath(path)}")
            continue
        try:
            found_term = False
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                for line_index, line in enumerate(handle):
                    if line.strip() == "[Term]":
                        found_term = True
                        break
                    if line_index >= 5000:
                        break
            if not found_term:
                failures.append(f"{label} does not look like an OBO file: {os.path.abspath(path)}")
            else:
                logger.log(f"  Preflight OK: {label} -> {os.path.abspath(path)}")
        except Exception as exc:
            failures.append(f"could not read {label}: {type(exc).__name__}: {exc}")

    _REPORTING_MODULE_CACHE.clear()
    for module_name, callable_name in Config.REPORTING_MODULES.items():
        try:
            module = importlib.import_module(module_name)
            function = getattr(module, callable_name)
            if not callable(function):
                raise TypeError(f"{module_name}.{callable_name} is not callable")
            _REPORTING_MODULE_CACHE[module_name] = module
            logger.log(f"  Preflight OK: reporting callable {module_name}.{callable_name}")
        except Exception as exc:
            failures.append(
                f"reporting module validation failed for {module_name}.{callable_name}: "
                f"{type(exc).__name__}: {exc}"
            )

    if failures:
        for failure in failures:
            logger.log(f"  PREFLIGHT FAILURE: {failure}", level="ERROR")
        raise RuntimeError(
            "Preflight validation failed before model training:\n- " + "\n- ".join(failures)
        )
    logger.log("  Preflight validation passed.")


def validate_required_outputs(
    all_results: Dict, pub_outputs: Dict
) -> Dict[str, Any]:
    """Validate required files and scientific-control invariants before completion."""
    required_relative = [
        "results/all_results.csv",
        "results/all_results_full.pkl",
        "results/model_selection_validation.csv",
        "results/publication_table.csv",
        "results/bootstrap_cis.csv",
        "results/paired_scenario_contrasts.csv",
        "results/disease_overlap_audit.csv",
        "results/split_summary.csv",
        "results/permutation_test.csv",
        "results/split_sensitivity_fixed_configuration.csv",
        "results/split_sensitivity_fixed_configuration_summary.csv",
        "results/severity_hpo_term_sensitivity.csv",
        "results/severity_hpo_terms_removed.csv",
        "results/high_mi_provenance_flags.csv",
        "results/generalisation_evaluations.csv",
        "results/feature_provenance_table.csv",
        "results/gene_level_evaluation.csv",
        "SUMMARY_REPORT.txt",
        "pipeline_manifest.json",
    ]
    failures: List[str] = []
    file_checks: List[Dict[str, Any]] = []
    for relative in required_relative:
        path = os.path.join(Config.OUTPUT_DIR, relative)
        exists = os.path.isfile(path)
        size = os.path.getsize(path) if exists else 0
        valid = exists and size > 0
        if valid and path.endswith(".csv"):
            try:
                pd.read_csv(path, nrows=2)
            except Exception as exc:
                valid = False
                failures.append(f"unreadable required CSV {relative}: {exc}")
        if not valid and not any(relative in item for item in failures):
            failures.append(f"missing or empty required output: {relative}")
        file_checks.append({"path": relative, "exists": exists, "size_bytes": size, "valid": valid})

    selected = _pick_best_model_per_scenario(all_results)
    for scenario, model_name in selected.items():
        result = all_results[scenario]["all"][model_name]
        if not result.get("test_evaluated", False) or result.get("test_evaluation_count") != 1:
            failures.append(
                f"selected primary result {scenario}/{model_name} was not evaluated exactly once"
            )
        registry_key = f"primary:{scenario}"
        if _test_evaluation_registry.get(registry_key, 0) != 1:
            failures.append(
                f"primary test registry count for {registry_key} is "
                f"{_test_evaluation_registry.get(registry_key, 0)}, expected 1"
            )
        bundle = os.path.join(Config.OUTPUT_DIR, "models", f"{scenario}_{model_name}.pkl")
        if not os.path.isfile(bundle) or os.path.getsize(bundle) == 0:
            failures.append(f"missing selected-model bundle: {bundle}")

    try:
        split_df = pd.read_csv(os.path.join(
            Config.OUTPUT_DIR, "results", "split_sensitivity_fixed_configuration.csv"
        ))
        if "seed" in split_df and (split_df["seed"] == RANDOM_SEED).any():
            failures.append("split sensitivity contains primary seed 42")
        expected_seeds = set(range(
            RANDOM_SEED + Config.SPLIT_SENSITIVITY_SEED_OFFSET,
            RANDOM_SEED + Config.SPLIT_SENSITIVITY_SEED_OFFSET + Config.N_SEEDS_SENSITIVITY,
        ))
        observed_seeds = set(split_df.get("seed", pd.Series(dtype=int)).astype(int))
        if not expected_seeds.issubset(observed_seeds):
            failures.append(
                f"split sensitivity missing seeds: {sorted(expected_seeds - observed_seeds)}"
            )
        expected_pairs = {
            (scenario, seed) for scenario in selected for seed in expected_seeds
        }
        observed_pairs = set(zip(
            split_df.get("Scenario", pd.Series(dtype=str)).astype(str),
            split_df.get("seed", pd.Series(dtype=int)).astype(int),
        ))
        missing_pairs = expected_pairs - observed_pairs
        if missing_pairs:
            failures.append(
                f"split sensitivity missing scenario/seed results; sample={sorted(missing_pairs)[:10]}"
            )
    except Exception as exc:
        failures.append(f"could not validate split sensitivity contents: {exc}")

    try:
        perm_df = pd.read_csv(os.path.join(Config.OUTPUT_DIR, "results", "permutation_test.csv"))
        if Config.N_PERMUTATIONS < 1:
            failures.append(f"Config.N_PERMUTATIONS={Config.N_PERMUTATIONS}, expected at least 1")
        if perm_df.empty:
            failures.append("permutation output contains no scenario results")
        else:
            if (perm_df["n_permutations_requested"] < 1).any():
                failures.append("permutation output contains fewer than 1 requested permutations")
            if (perm_df["n_permutations_successful"] < 1).any():
                failures.append("fewer than 1 successful permutations for one or more scenarios")
            observed_scenarios = set(perm_df.get("Scenario", pd.Series(dtype=str)).astype(str))
            missing_scenarios = set(selected) - observed_scenarios
            if missing_scenarios:
                failures.append(
                    f"permutation output missing scenarios: {sorted(missing_scenarios)}"
                )
    except Exception as exc:
        failures.append(f"could not validate permutation contents: {exc}")

    try:
        generalisation_df = pd.read_csv(os.path.join(
            Config.OUTPUT_DIR, "results", "generalisation_evaluations.csv"
        ))
        observed_modes = set(generalisation_df.get("generalisation_mode", pd.Series(dtype=str)).astype(str))
        missing_modes = set(Config.ADDITIONAL_GENERALISATION_MODES) - observed_modes
        if Config.RUN_ADDITIONAL_GENERALISATION and missing_modes:
            failures.append(f"missing generalisation modes in results: {sorted(missing_modes)}")
        expected_pairs = {
            (mode, scenario)
            for mode in Config.ADDITIONAL_GENERALISATION_MODES
            for scenario in selected
        }
        observed_pairs = set(zip(
            generalisation_df.get("generalisation_mode", pd.Series(dtype=str)).astype(str),
            generalisation_df.get("Scenario", pd.Series(dtype=str)).astype(str),
        ))
        missing_pairs = expected_pairs - observed_pairs
        if Config.RUN_ADDITIONAL_GENERALISATION and missing_pairs:
            failures.append(
                f"generalisation results missing mode/scenario pairs; sample={sorted(missing_pairs)[:10]}"
            )
    except Exception as exc:
        failures.append(f"could not validate generalisation contents: {exc}")

    try:
        severity_df = pd.read_csv(os.path.join(
            Config.OUTPUT_DIR, "results", "severity_hpo_term_sensitivity.csv"
        ))
        expected = set(Config.SHARED_HPO_BUDGET_SCENARIOS)
        observed = set(severity_df.get("Scenario", pd.Series(dtype=str)).astype(str))
        if Config.RUN_SEVERITY_HPO_SENSITIVITY and not expected.issubset(observed):
            failures.append(
                f"severity-HPO sensitivity missing scenarios: {sorted(expected - observed)}"
            )
        if Config.RUN_SEVERITY_HPO_SENSITIVITY and not severity_df.empty:
            counts = severity_df.set_index("Scenario").get(
                "n_severity_hpo_terms_removed", pd.Series(dtype=float)
            )
            zero_or_missing = [
                scenario for scenario in expected
                if scenario not in counts.index or float(counts.loc[scenario]) <= 0
            ]
            if zero_or_missing:
                failures.append(
                    f"severity-HPO sensitivity removed no configured terms for: {sorted(zero_or_missing)}"
                )
    except Exception as exc:
        failures.append(f"could not validate severity-HPO sensitivity contents: {exc}")

    try:
        contrasts_df = pd.read_csv(os.path.join(
            Config.OUTPUT_DIR, "results", "paired_scenario_contrasts.csv"
        ))
        expected_contrasts = {
            f"{a}_minus_{b}" for a, b in PAIRED_SCENARIO_CONTRASTS
            if a in selected and b in selected
        }
        observed_contrasts = set(
            contrasts_df.get("contrast", pd.Series(dtype=str)).astype(str)
        )
        missing_contrasts = expected_contrasts - observed_contrasts
        if missing_contrasts:
            failures.append(
                f"paired scenario contrasts missing: {sorted(missing_contrasts)}"
            )
        expected_metric_rows = {
            (contrast, metric)
            for contrast in expected_contrasts
            for metric in PAIRED_CONTRAST_METRICS
        }
        observed_metric_rows = set(zip(
            contrasts_df.get("contrast", pd.Series(dtype=str)).astype(str),
            contrasts_df.get("metric", pd.Series(dtype=str)).astype(str),
        ))
        missing_metric_rows = expected_metric_rows - observed_metric_rows
        if missing_metric_rows:
            failures.append(
                f"paired scenario contrasts missing metric rows: "
                f"{sorted(missing_metric_rows)}"
            )
        if not contrasts_df.empty:
            valid_counts = contrasts_df.get("n_valid_replicates", pd.Series(dtype=float))
            if (pd.to_numeric(valid_counts, errors="coerce").fillna(0) < 10).any():
                failures.append(
                    "paired scenario contrasts contain rows with fewer than 10 "
                    "valid bootstrap replicates"
                )
    except Exception as exc:
        failures.append(f"could not validate paired scenario contrast contents: {exc}")

    try:
        overlap_df = pd.read_csv(os.path.join(
            Config.OUTPUT_DIR, "results", "disease_overlap_audit.csv"
        ))
        record_types = set(overlap_df.get("record_type", pd.Series(dtype=str)).astype(str))
        if "summary" not in record_types:
            failures.append("disease-overlap audit contains no summary rows")
        if "disease" not in record_types:
            failures.append("disease-overlap audit contains no per-disease rows")
        summary_metrics = set(
            overlap_df.loc[
                overlap_df.get("record_type", pd.Series(dtype=str)) == "summary",
                "metric",
            ].astype(str)
        ) if "metric" in overlap_df.columns else set()
        required_summary_metrics = {
            "n_test_pairs",
            "n_test_pairs_with_mondo_id_also_in_train",
            "percent_test_pairs_with_mondo_id_also_in_train",
            "n_distinct_mondo_train",
            "n_distinct_mondo_validation",
            "n_distinct_mondo_test",
        }
        missing_summary_metrics = required_summary_metrics - summary_metrics
        if missing_summary_metrics:
            failures.append(
                f"disease-overlap audit missing summary metrics: "
                f"{sorted(missing_summary_metrics)}"
            )
    except Exception as exc:
        failures.append(f"could not validate disease-overlap audit contents: {exc}")

    report = {
        "validated_at": datetime.now().isoformat(),
        "status": "passed" if not failures else "failed",
        "required_file_checks": file_checks,
        "failures": failures,
        "primary_test_registry": dict(_test_evaluation_registry),
        "n_permutations_configured": Config.N_PERMUTATIONS,
        "generalisation_enabled": Config.RUN_ADDITIONAL_GENERALISATION,
        "severity_hpo_sensitivity_enabled": Config.RUN_SEVERITY_HPO_SENSITIVITY,
    }
    validation_path = os.path.join(Config.OUTPUT_DIR, "required_output_validation.json")
    with open(validation_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    if failures:
        raise RuntimeError(
            "Required-output validation failed; RUN_COMPLETE.json was not created:\n- "
            + "\n- ".join(failures)
        )
    logger.log(f"  Required-output validation passed; wrote {validation_path}")
    return report


def _log_and_validate_runtime_identity() -> None:
    script_path = os.path.realpath(__file__)
    script_hash = _sha256_file(script_path)
    logistic_probe = _make_multinomial_logistic_classifier(seed=RANDOM_SEED)
    logistic_estimator = _get_multinomial_logistic_estimator(logistic_probe)
    logger.section("RUNTIME IDENTITY AND CONFIGURATION")
    logger.log(f"  PID: {os.getpid()}")
    logger.log(f"  Python executable: {sys.executable}")
    logger.log(f"  Script path: {script_path}")
    logger.log(f"  Script SHA256: {script_hash}")
    logger.log(f"  Command line: {' '.join(sys.argv)}")
    logger.log(
        "  Logistic constructor: "
        f"solver={logistic_estimator.solver}, max_iter={logistic_estimator.max_iter}, "
        f"tol={logistic_estimator.tol}, n_jobs={logistic_estimator.n_jobs}"
    )
    logger.log(
        "  Runtime flags: "
        f"RUN_ADDITIONAL_GENERALISATION={Config.RUN_ADDITIONAL_GENERALISATION}, "
        f"RUN_PERMUTATION_TEST={Config.RUN_PERMUTATION_TEST}, "
        f"N_PERMUTATIONS={Config.N_PERMUTATIONS}"
    )
    if int(logistic_estimator.max_iter) != 10000:
        raise RuntimeError(
            "Stale logistic constructor detected: expected max_iter=10000, "
            f"found {logistic_estimator.max_iter}."
        )
    if logistic_estimator.n_jobs is not None:
        raise RuntimeError(
            "Stale logistic constructor detected: LogisticRegression.n_jobs must be None, "
            f"found {logistic_estimator.n_jobs}."
        )


def main(smoke_test=False):
    """Run the reference workflow in a fresh output directory."""
    if smoke_test:
        from functools import lru_cache
        # Cache raw tokens; each fold still learns its own vocabulary.
        globals()['_load_text_token_maps'] = lru_cache(maxsize=1)(_load_text_token_maps)
        Config.N_PERMUTATIONS = 10
        Config.N_BOOTSTRAP = 50
        Config.CALIBRATION_N_BOOTSTRAP = 50
        Config.N_SEEDS_SENSITIVITY = 3
        Config.NESTED_SEARCH_ITERATIONS = dict.fromkeys(Config.CANDIDATE_MODEL_ORDER, 1)
        Config.HPO_BUDGET_CANDIDATES = (20,)
    _require_fresh_output_directory()
    _initialise_runtime()
    try:
        logger.section(f"PIPELINE v{PIPELINE_VERSION} START - Gene-Disease Pair Analysis")
        _log_and_validate_runtime_identity()
        validate_preflight_inputs_and_reporting()
        container = assemble_full_dataframe()
        if smoke_test:
            rng = np.random.RandomState(RANDOM_SEED)
            disease_counts = container.full_df.groupby('mondo_id')['gene_symbol'].nunique()
            shared_disease = disease_counts.idxmax()
            shared_indices = np.flatnonzero(container.full_df['mondo_id'].eq(shared_disease).to_numpy())
            sampled = []
            for label in range(Config.NUM_CLASSES):
                pool = np.flatnonzero(container.y == label)
                required = np.intersect1d(pool, shared_indices)[:100]
                remaining = np.setdiff1d(pool, required)
                extra = rng.choice(remaining, size=min(100 - len(required), len(remaining)), replace=False)
                sampled.extend(np.concatenate([required, extra]).tolist())
            indices = np.sort(np.asarray(sampled, dtype=int))
            container.full_df = container.full_df.iloc[indices].reset_index(drop=True)
            container.metadata = container.metadata.iloc[indices].reset_index(drop=True)
            container.y = container.y[indices]
            container.groups = container.groups[indices]
        # Export display labels before anything draws a figure, so both this
        # script's diagnostic plots and the publication figure module resolve
        # readable names rather than raw column names.
        write_feature_label_dictionaries(container)
        make_train_test_split(container)
        all_results = run_experiments(container)
        save_all_base_model_bundles(container, all_results)
        pub_outputs = run_publication_analyses(all_results, container)
        save_per_disease_explanations(container, all_results)
        # SHAP on the expected severity class, full matrix, used by Figure 4a-d.
        save_severity_shap(container, all_results)

        # Persist valid non-ordinal SHAP values added after model fitting.
        results_collector.save(
            os.path.join(Config.OUTPUT_DIR, "results", "all_results.csv")
        )

        figure_module = _REPORTING_MODULE_CACHE["publication_figures"]
        figure_module.generate_publication_figures(
            all_results,
            container,
            pub_outputs,
            results_collector,
            fold_artifacts=_fold_artifacts,
            output_dir=Config.OUTPUT_DIR,
            random_seed=RANDOM_SEED,
        )
        report_module = _REPORTING_MODULE_CACHE["run_report"]
        report_module.generate_manuscript_report(
            all_results,
            container,
            pub_outputs,
            output_dir=Config.OUTPUT_DIR,
            pipeline_version=PIPELINE_VERSION,
            config=Config,
        )
        generate_summary_report(all_results, pub_outputs)
        write_pipeline_manifest(container, all_results, pub_outputs)
        validate_required_outputs(all_results, pub_outputs)

        with open(
            os.path.join(Config.OUTPUT_DIR, "RUN_COMPLETE.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump({
                "pipeline_version": PIPELINE_VERSION,
                "completed_at": datetime.now().isoformat(),
                "status": "complete",
                "required_output_validation": "passed",
            }, handle, indent=2)
        logger.section(f"PIPELINE v{PIPELINE_VERSION} COMPLETE")
        logger.save()
    except Exception as exc:
        logger.log(
            f"PIPELINE FAILED: {type(exc).__name__}: {exc}",
            level="CRITICAL",
        )
        logger.save()
        raise
    print(f"\nOutputs written to: {Config.OUTPUT_DIR}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data/training')
    parser.add_argument('--output-dir', default='results/main')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    Config.OUTPUT_DIR = os.path.abspath(args.output_dir)
    os.chdir(os.path.abspath(args.data_dir))
    Config.EXTERNAL_FEATURES = next((path for path in (
        Config._EXTERNAL_V46, Config._EXTERNAL_FIXED,
        os.path.join(Config.CONSOLIDATED_DIR, 'EXTERNAL_MECHANISTIC_FEATURES.csv')
    ) if os.path.isfile(path)), Config.EXTERNAL_FEATURES)
    main(smoke_test=args.smoke_test)
