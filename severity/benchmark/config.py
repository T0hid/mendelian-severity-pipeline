"""Settings for the gene-disease severity benchmark.

Every value here is the one used for the published run. Section names follow the
Online Methods. Paths are set with ``Config.configure`` before a run.
"""

import os
import warnings

import numpy as np

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names.*",
    category=UserWarning,
    module="sklearn",
)

PIPELINE_VERSION = "4.10.0"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class Config:
    """Benchmark settings. Values are class attributes so that every module reads the same ones."""

    # ---- Input and output locations ------------------------------------------------------
    DATA_DIR = "."
    OUTPUT_DIR = "benchmark_run"
    PRIMARY_DATA = "final_output_with_severity.csv"
    GNN_DATA = "final_gnn_dataset.csv"
    REBUILT_GENE_DISEASE_MAPPING = "gene_disease_mapping_HPO_rebuilt.csv"
    CORE_FEATURES = os.path.join("CONSOLIDATED_FEATURES_v2", "CORE_MECHANISTIC_FEATURES.csv")
    EXTERNAL_FEATURES = os.path.join("CONSOLIDATED_FEATURES_v2", "EXTERNAL_MECHANISTIC_FEATURES_v4_6.csv")
    NEW_GENE_FEATURES = os.path.join("CONSOLIDATED_FEATURES_v2", "NEW_GENE_LEVEL_FEATURES.csv")
    MOUSE_LETHALITY_FEATURES = os.path.join("CONSOLIDATED_FEATURES_v2", "MOUSE_LETHALITY_FEATURES.csv")
    MOUSE_HUMAN_PHENOTYPE_FEATURES = os.path.join(
        "CONSOLIDATED_FEATURES_v2", "MOUSE_HUMAN_PHENOTYPE_FEATURES.csv"
    )
    # Optional variant table. Its columns are removed from every feature set (see
    # NON_BIOLOGICAL_PREFIXES); the published run loaded it, the public inputs do not include it.
    DOMAIN_FEATURES = os.path.join("enhanced_variant_features", "DOMAIN_CARRIER_FEATURES.csv")
    HPO_OBO_PATH = "hp.obo"
    MONDO_OBO_PATH = "mondo.obo"
    _RELATIVE_INPUTS = (
        "PRIMARY_DATA",
        "GNN_DATA",
        "REBUILT_GENE_DISEASE_MAPPING",
        "CORE_FEATURES",
        "EXTERNAL_FEATURES",
        "NEW_GENE_FEATURES",
        "MOUSE_LETHALITY_FEATURES",
        "MOUSE_HUMAN_PHENOTYPE_FEATURES",
        "DOMAIN_FEATURES",
        "HPO_OBO_PATH",
        "MONDO_OBO_PATH",
    )

    @classmethod
    def configure(cls, data_dir: str, output_dir: str) -> None:
        """Point every input path at ``data_dir`` and write results to ``output_dir``."""
        data_dir = os.path.abspath(data_dir)
        for name in cls._RELATIVE_INPUTS:
            current = getattr(cls, name)
            if not os.path.isabs(current):
                setattr(cls, name, os.path.join(data_dir, current))
        cls.DATA_DIR = data_dir
        cls.OUTPUT_DIR = os.path.abspath(output_dir)

    # ---- Data and severity labels ----------------------------------------------------------
    TARGET_MAP = {"Mild": 0, "Moderate": 1, "Severe": 2, "Profound": 3}
    NUM_CLASSES = 4
    TARGET_LABEL_PROVENANCE = (
        "Severity labels were assigned by evidence-grounded LLM agents that apply the Lazarin "
        "framework to published phenotype evidence; they are not case-by-case expert adjudications."
    )

    # ---- Data partitioning and nested cross-validation (equation 1) ----------------------
    TEST_SIZE = 0.2
    VAL_SIZE = 0.15  # fraction of all pairs
    SPLIT_SEARCH_ITERATIONS = 1000
    SPLIT_MIN_CLASS_COUNT = 1
    NESTED_CV_FOLDS = 3
    SPLIT_SENSITIVITY_SEED_OFFSET = 1  # further splits use seeds 43-47
    N_SEEDS_SENSITIVITY = 5

    # ---- Feature construction --------------------------------------------------------------
    PAIR_INHERITANCE_COLUMN = "moi_title"
    MOI_DOMINANT = ("Autosomal dominant", "Semidominant")
    MOI_RECESSIVE = ("Autosomal recessive",)
    MOI_XLINKED = ("X-linked", "X-linked recessive", "X-linked dominant")
    MOI_UNKNOWN = ("Unknown", "")
    # Gene-level inheritance summaries in the core table; inheritance is taken per pair instead.
    CORE_INHERITANCE_ROLLUP_COLUMNS = [
        "is_dominant",
        "is_recessive",
        "is_xlinked",
        "has_mixed_inheritance",
        "n_inheritance_modes",
        "all_inheritance_modes",
        "primary_inheritance",
    ]
    # Input columns removed on loading, with the reason.
    REMOVED_INPUT_COLUMNS_CORE = {
        "pLI": "gnomad_min_across_transcripts",
        "oe_lof": "gnomad_min_across_transcripts",
        "oe_mis": "gnomad_min_across_transcripts",
        "gnomad_regional_pLI": "gnomad_min_across_transcripts",
        "gnomad_regional_oe_lof": "gnomad_min_across_transcripts",
        "gnomad_regional_oe_mis": "gnomad_min_across_transcripts",
        "clingen_hi_score": "core_copy_constant",
    }
    REMOVED_INPUT_COLUMNS_EXTERNAL = {
        "collins_pHI": "unidentified_source",
        "collins_pTS": "empty_for_every_gene",
        "compensation_failure": "copy_of_collins_pHI",
        "rigid_no_backup": "copy_of_af_structural_rigidity",
        "hi_no_backup": "derived_from_constant_no_backup_flag",
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
    # Gene-annotation features: the 20 most frequent terms per category in the training pairs.
    TEXT_VOCAB_TOP_N = 20
    TEXT_PLACEHOLDER_TOKENS = ("unknown", "na", "n/a", "none", "nan", "null", "-")
    TEXT_EXCLUDED_SOURCE_COLUMNS = ("Expression_Sites", "Interacting_Genes")
    MONDO_CLASS_ROOT = "MONDO:0700096"  # "human disease"; its direct subclasses are the MONDO features
    HPO_ANCESTOR_MAX_HOPS = 0  # HPO terms enter as directly recorded terms only
    HPO_ANCESTOR_EXCLUDED_IDS = ("HP:0000001", "HP:0000118")

    # Columns kept out of every feature set.
    ORGAN_CRITICALITY_COLUMNS = [
        "critical_brain",
        "critical_heart",
        "critical_lung",
        "critical_kidney",
        "critical_liver",
        "critical_immune",
        "critical_metabolic",
        "n_critical_organs",
        "is_systemic",
        "tissue_criticality_score",
        "noncritical_skin",
        "noncritical_dental",
        "noncritical_sensory",
        "noncritical_total",
        "n_noncritical_organs",
    ]
    DUPLICATE_COLUMNS = ["gene_age"]  # equals (5 - 4 * conservation_score)
    # Features that describe how well a disease or gene is documented rather than its biology.
    RECORD_KEEPING_COLUMNS = (
        "feat_mondo_depth",
        "feat_mondo_n_classes",
        "mouse_lethal_pheno_n_hpos_with_data",
        "mouse_phenotype_count",
    )
    # Study-intensity features: variant counts, protein-interaction network features, database-presence
    # flags (has_*_record), the uncertainty of the s_het estimate and the ClinGen "curated" flag.
    NON_BIOLOGICAL_PREFIXES = ("var_", "ppi_", "network_")
    NON_BIOLOGICAL_COLUMNS = ("shet_ci_width_log10", "clingen_dosage_curated")
    # Features derived from, or nearly identical to (|Spearman rho| >= 0.9), a feature that is kept.
    REDUNDANT_FEATURE_COLUMNS = (
        "is_ancient",
        "depmap_n_essential",
        "depmap_is_pan_essential",
        "depmap_is_context_essential",
        "hpa_is_low_specificity",
        "has_yeast_ortholog",
        "has_fly_ortholog",
        "has_worm_ortholog",
        "has_zebrafish_ortholog",
        "has_mouse_ortholog",
        "af_median_plddt",
        "af_frac_confident",
        "af_frac_very_high",
        "af_structural_rigidity",
        "af_frac_disordered",
        "af_disorder_ratio",
        "gnomad_cds_length",
        "gnomad_exp_lof",
        "gnomad_exp_mis",
        "gnomad_obs_mis",
        "gnomad_syn_z",
        "feat_pli_x_dominant",
        "feat_pli_x_recessive",
    )
    # Gene-level roll-ups over a gene's diseases; removed from the gene-level set (S1) only.
    INHERITANCE_COUNT_PROXY_COLUMNS = [
        "n_inheritance_modes",
        "has_mixed_inheritance",
        "feat_has_mixed_inheritance",
    ]

    # ---- Missing values and preprocessing ------------------------------------------------
    MISSING_DROP_THRESHOLD = 0.5
    VAR_DROP_THRESHOLD = 1e-10

    # ---- Feature selection and leakage controls ------------------------------------------
    HPO_MIN_TRAIN_COUNT = 2
    HPO_MAX_TRAIN_FRACTION = 0.9
    MONDO_MIN_TRAIN_COUNT = 5
    MI_THRESHOLD_MECHANISTIC = 0.005  # S1
    MI_MIN_FEATURES_MECHANISTIC = 30
    MI_THRESHOLD_DEFAULT = 0.01  # S1b
    MI_MIN_FEATURES_DEFAULT = 15
    MI_MAX_DISCRETE_UNIQUE = 10
    MI_POSITIVE_EPS = 1e-12
    MI_HIGH_AUDIT_QUANTILE = 0.99
    MI_HIGH_AUDIT_MIN_SCORE = 0.2
    # Upper limit on selected features per feature family (S2 and S3).
    FEATURE_BUDGET = {
        "hpo_binary": 200,
        "hpo_derived": 10,
        "organ_system": 25,
        "mondo": 50,
        "inheritance": 8,
        "gnn_text": 80,
        "primary_text": 80,
        "engineered": 50,
        "mechanistic_core": 120,
        "mouse_lethality": 20,
        "mouse_phenotype": 20,
        "variant": 40,
        "external": 40,
        "missingness_indicator": 20,
    }
    # Limit on directly recorded HPO terms, chosen once for S3 by nested CV and shared with S2.
    HPO_BUDGET_ALL_ELIGIBLE = 10000  # "no limit": larger than the number of HPO columns
    HPO_BUDGET_CANDIDATES = [200, 300, 400, 600, 800, 1200, 1600, 2400, 3200, 5200, HPO_BUDGET_ALL_ELIGIBLE]
    SHARED_HPO_BUDGET_SCENARIOS = ("S2_Mech_Pheno", "S3_Phenotype")
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

    # ---- Model families, selection and calibration ---------------------------------------
    CANDIDATE_MODEL_ORDER = ("MultinomialLogisticRegression", "ExtraTrees", "LightGBM", "OrdinalDecomp")
    NESTED_SEARCH_ITERATIONS = {
        "MultinomialLogisticRegression": 10,
        "ExtraTrees": 20,
        "LightGBM": 20,
        "OrdinalDecomp": 15,
    }
    MODEL_SELECTION_RULE = "lexicographic_qwk_then_mcc_then_balanced_accuracy"
    IMBALANCE_STRATEGY = "sample_weight"
    ORDINAL_BOUNDARY_MULT = [2.5, 1.5, 1.0, 1.0]  # multipliers m in equation (3)
    CALIBRATION_METHOD_LARGE = "isotonic"
    CALIBRATION_METHOD_SMALL = "sigmoid"
    CALIBRATION_MIN_POSITIVES = 100
    CALIBRATION_MIN_NEGATIVES = 100
    MODEL_N_JOBS = 4

    # ---- Sensitivity analyses and robustness checks --------------------------------------
    MI_SENSITIVITY_THRESHOLDS = [0.0, 0.005, 0.01, 0.02]
    N_PERMUTATIONS = 100
    PERMUTATION_MODE = "gene_block_candidate_selection_fixed_hyperparameters"
    PERMUTATION_MIN_MOVABLE_FRACTION = 0.8
    ADDITIONAL_GENERALISATION_MODES = ["disease_disjoint", "gene_and_disease_disjoint"]

    # ---- SHAP (expected severity class of the uncalibrated model) -------------------------
    ORDINAL_EXPLANATION_POLICY = "uncalibrated_boundary_models_only"
    SEVERITY_SHAP_BACKGROUND_N = 100
    SEVERITY_SHAP_PERMUTATION_BACKGROUND_N = 25

    # ---- Statistics ------------------------------------------------------------------------
    N_BOOTSTRAP = 1000
    CALIBRATION_N_BINS = 5
    CALIBRATION_BINNING = "quantile"
    CALIBRATION_N_BOOTSTRAP = 1000
