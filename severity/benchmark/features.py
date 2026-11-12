"""Assembly of the feature table and the four feature sets (Feature construction).

S1 (gene-level): constraint, dosage sensitivity, essentiality, expression, conservation,
paralogues, protein structure, pair-level inheritance and engineered combinations.
S1b (disease context): S1 plus the 24 organ-system indicators and the MONDO disease classes.
S2 (combined): S1b without the organ-system indicators, plus mouse-derived features and
directly recorded HPO terms.
S3 (phenotype only): directly recorded HPO terms.

Severity-stating HPO terms are removed from S2 and S3 here (see ``severity_terms``).
Columns listed in ``Config`` as record-keeping, study-intensity or redundant features are
removed from every set. The feature provenance table written here is Supplementary Table 1.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .annotation_terms import _load_text_token_maps
from .config import Config
from .inputs import (
    build_gene_disease_target,
    load_external_features,
    load_mechanistic_features,
    load_mouse_human_phenotype_features,
    load_mouse_lethality_features,
    load_new_gene_features,
    load_variant_features,
)
from .ontology import build_mondo_features, parse_mondo_obo
from .runtime import HPO_COLUMN_TERM_NAMES, DataContainer, logger
from .severity_terms import (
    SEVERITY_HPO_IDS,
    filter_leaky_columns_for_phenotype_scenarios,
    is_severity_hpo_column,
)

# Columns that identify a pair or carry its label; never features.
METADATA_COLUMNS = [
    "gene_symbol",
    "mondo_id",
    "disease_name",
    "hgnc_id",
    "Multiclass_Target",
    "max_severity",
    "min_severity",
    "mean_severity",
    "median_severity",
    "std_severity",
    "severity_range",
    "n_diseases",
    "n_severity_levels",
    "severity_entropy",
    "original_severity",
    "target_severity",
    "primary_inheritance",
    "is_dominant",
    "is_recessive",
    "is_xlinked",
    "has_mixed_inheritance",
    "n_inheritance_modes",
    "all_inheritance_modes",
    "hpo_term",
    "parent_term",
    "hpo_token",
    "hpo_id",
]
# Clinical or label-derived columns that could reveal the severity class.
LEAKAGE_COLUMNS = [
    "total_hpo_terms",
    "hpo_n_terms",
    "n_hpo_terms",
    "has_early_onset",
    "age_of_onset",
    "onset_age",
    "early_onset",
    "late_onset",
    "adult_onset",
    "childhood_onset",
    "infantile_onset",
    "neonatal_onset",
    "congenital_onset",
    "typical_onset_age",
    "mean_onset_age",
    "median_onset_age",
    "disease_severity",
    "clinical_severity",
    "phenotype_severity",
    "n_symptoms",
    "symptom_count",
]
LEAKAGE_KEYWORDS = ["severity", "target", "class_label", "prediction_score", "probability"]
SAFE_KEYWORDS = [
    "depmap",
    "gnomad",
    "gtex",
    "ppi",
    "network",
    "paralog",
    "conservation",
    "constraint",
    "essential",
    "organ",
    "tissue",
    "alphafold",
    "af_",
    "clingen",
    "collins",
    "decipher",
    "compensation",
    "dosage",
    "backup",
    "rigid",
    "mouse",
    "lethal",
    "viability",
    "phenotype",
]
MOUSE_DERIVED_ENGINEERED = (
    "feat_lethal_x_pli",
    "feat_lethal_x_essential",
    "feat_viability_x_pli",
    "feat_has_lethal_x_pli",
)


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """The nine engineered features of Supplementary Table 24 (fixed products and differences).

    The last four use mouse data and are offered to the combined set (S2) only.
    """
    eng_df = df.copy()
    pli = eng_df["gnomad_pLI"]
    lethal = pd.to_numeric(eng_df["mouse_lethal"], errors="coerce")
    eng_df["feat_missz_x_pli"] = eng_df["gnomad_mis_z"] * pli
    eng_df["feat_missz_minus_synz"] = pd.to_numeric(eng_df["gnomad_mis_z"], errors="coerce") - pd.to_numeric(
        eng_df["gnomad_syn_z"], errors="coerce"
    )
    eng_df["feat_essential_constrained"] = eng_df["depmap_mean_effect"] * pli
    eng_df["feat_conserved_constrained"] = eng_df["conservation_score"] * pli
    eng_df["feat_constrained_no_paralog"] = pli * (eng_df["paralog_count"] == 0).astype(float)
    eng_df["feat_lethal_x_pli"] = lethal * pli
    eng_df["feat_lethal_x_essential"] = lethal * eng_df["depmap_mean_effect"]
    eng_df["feat_viability_x_pli"] = pd.to_numeric(eng_df["mouse_viability_score"], errors="coerce") * pli
    eng_df["feat_has_lethal_x_pli"] = pd.to_numeric(eng_df["has_lethal_phenotype"], errors="coerce") * pli
    new_features = [
        "feat_missz_x_pli",
        "feat_missz_minus_synz",
        "feat_essential_constrained",
        "feat_conserved_constrained",
        "feat_constrained_no_paralog",
        *MOUSE_DERIVED_ENGINEERED,
    ]
    logger.log(f"  Engineered features: {len(new_features)}")
    return eng_df, new_features


def build_inheritance_features(
    full_df: pd.DataFrame, pli_col: Optional[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """Dominant, recessive, X-linked and mixed-inheritance flags for each pair.

    Pairs without a recorded mode of inheritance are 0 on every flag.
    """
    inheritance_df = pd.DataFrame(index=full_df.index)
    recorded = (
        pd.to_numeric(full_df["is_dominant"], errors="coerce").notna()
        if "is_dominant" in full_df.columns
        else None
    )
    inheritance_features: List[str] = []
    for feature_name, source_column in {
        "feat_is_dominant": "is_dominant",
        "feat_is_recessive": "is_recessive",
        "feat_is_xlinked": "is_xlinked",
        "feat_has_mixed_inheritance": "has_mixed_inheritance",
    }.items():
        if source_column in full_df.columns:
            values = pd.to_numeric(full_df[source_column], errors="coerce").astype(float)
            inheritance_df[feature_name] = values.where(recorded, 0.0) if recorded is not None else values
        else:
            inheritance_df[feature_name] = np.nan
        inheritance_features.append(feature_name)
    if recorded is not None:
        logger.log(
            f"  Inheritance: {int((~recorded).sum())} pairs without a recorded mode coded 0 on every flag"
        )
    if pli_col and pli_col in full_df.columns:
        pli = pd.to_numeric(full_df[pli_col], errors="coerce").to_numpy(dtype=float)
        inheritance_df["feat_pli_x_dominant"] = pli * inheritance_df["feat_is_dominant"].to_numpy(dtype=float)
        inheritance_df["feat_pli_x_recessive"] = pli * inheritance_df["feat_is_recessive"].to_numpy(
            dtype=float
        )
        inheritance_features.extend(["feat_pli_x_dominant", "feat_pli_x_recessive"])
    return inheritance_df, inheritance_features


def _merge_gene_level_features(
    base_df: pd.DataFrame, source_df: pd.DataFrame, source_name: str
) -> pd.DataFrame:
    """Left-join a one-row-per-gene table; genes absent from the table keep missing values."""
    if "gene_symbol" not in source_df.columns:
        raise ValueError(f"{source_name} has no gene_symbol column.")
    if source_df["gene_symbol"].dropna().duplicated().any():
        raise ValueError(f"{source_name} has more than one row for some genes.")
    return base_df.copy().merge(source_df, on="gene_symbol", how="left", validate="m:1")


def _missing_value_semantics_for_family(family: str) -> str:
    if family in {"hpo_binary", "organ_system"}:
        return "structural_zero_means_term_absent"
    if family == "missingness_indicator":
        return "explicit_source_or_column_availability_indicator"
    if family in {"primary_text", "gnn_text"}:
        return "fold_built_tokens_with_explicit_source_availability"
    return "unknown_preserved_as_nan_then_training_fold_median_imputation"


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
    if name.endswith("__missing") or (name.startswith("has_") and name.endswith("_record")):
        return "missingness_indicator"
    if name in family_lookup:
        return family_lookup[name]
    return "unclassified"


def _drop_from_all_scenarios(container: DataContainer, should_drop) -> List[str]:
    removed = set()
    for scenario, columns in container.scenario_cols.items():
        removed.update(column for column in columns if should_drop(column))
        container.scenario_cols[scenario] = [column for column in columns if not should_drop(column)]
    return sorted(removed)


def _is_non_biological(column: str) -> bool:
    name = str(column)
    return (
        name in Config.NON_BIOLOGICAL_COLUMNS
        or name.startswith(Config.NON_BIOLOGICAL_PREFIXES)
        or (name.startswith("has_") and name.endswith("_record"))
    )


def assemble_full_dataframe() -> DataContainer:
    """Build the pair-level feature table and the candidate columns of S1, S1b, S2 and S3.

    Nothing here is fitted: frequencies, imputation values and feature selection are
    learned later inside each training fold.
    """
    logger.section("Assembling the feature table")
    container = DataContainer()
    disease_df = build_gene_disease_target(Config.PRIMARY_DATA)
    disease_df, mondo_cols = build_mondo_features(disease_df, parse_mondo_obo(Config.MONDO_OBO_PATH))
    container.mondo_cols = mondo_cols

    # Gene-level tables, joined so that every labelled pair is kept.
    n_pairs = len(disease_df)
    full_df = _merge_gene_level_features(disease_df, load_mechanistic_features(), "mechanistic")
    if len(full_df) != n_pairs:
        raise RuntimeError("Joining the gene-level features changed the number of pairs.")
    variant_df = load_variant_features()
    variant_cols: List[str] = []
    if len(variant_df) > 0:
        full_df = _merge_gene_level_features(full_df, variant_df, "variant")
        variant_cols = [column for column in variant_df.columns if column != "gene_symbol"]
    mouse_df = load_mouse_lethality_features()
    mouse_lethality_cols: List[str] = []
    if len(mouse_df) > 0:
        full_df = _merge_gene_level_features(full_df, mouse_df, "mouse_lethality")
        mouse_lethality_cols = [column for column in mouse_df.columns if column != "gene_symbol"]
    mouse_human_df = load_mouse_human_phenotype_features()
    mouse_human_phenotype_cols: List[str] = []
    if len(mouse_human_df) > 0:
        full_df = _merge_gene_level_features(full_df, mouse_human_df, "mouse_human_phenotype")
        mouse_human_phenotype_cols = [column for column in mouse_human_df.columns if column != "gene_symbol"]
    external_df, external_cols = load_external_features()
    if len(external_df) > 0:
        full_df = _merge_gene_level_features(full_df, external_df, "external")
    new_gene_df, new_gene_cols = load_new_gene_features()
    if len(new_gene_df) > 0:
        full_df = _merge_gene_level_features(full_df, new_gene_df, "new_gene")
    full_df = full_df.loc[:, ~full_df.columns.duplicated()].copy()

    # Core gene-level columns: everything not assigned to another family, minus identifiers,
    # label columns and columns whose names suggest outcome information.
    artifact_cols = [column for column in full_df.columns if "unnamed" in column.lower()]
    merge_artifact_leakage = [
        column + suffix
        for column in LEAKAGE_COLUMNS + METADATA_COLUMNS
        for suffix in ("_x", "_y")
        if column + suffix in full_df.columns and column + suffix not in LEAKAGE_COLUMNS
    ]
    exclude = set(METADATA_COLUMNS + LEAKAGE_COLUMNS + artifact_cols + merge_artifact_leakage)
    other_families = (
        set(variant_cols)
        | set(external_cols)
        | set(mondo_cols)
        | set(mouse_lethality_cols)
        | set(mouse_human_phenotype_cols)
    )
    potential_mechanistic = [
        column
        for column in full_df.columns
        if column not in other_families and not column.startswith("hpo_") and not column.startswith("feat_")
    ]
    cleaned_mechanistic: List[str] = []
    provenance_exclusions: Dict[str, str] = {}
    for column in potential_mechanistic:
        lower = column.lower()
        if column in exclude:
            provenance_exclusions[column] = "identifier_target_or_prespecified_prohibited_column"
        elif any(keyword in lower for keyword in LEAKAGE_KEYWORDS) and not any(
            safe in lower for safe in SAFE_KEYWORDS
        ):
            provenance_exclusions[column] = "name_indicates_target_or_post_outcome_information"
        else:
            cleaned_mechanistic.append(column)
    for column in LEAKAGE_COLUMNS:
        if column in full_df.columns:
            provenance_exclusions[column] = "prespecified_clinical_or_target_proxy"
    for column in METADATA_COLUMNS:
        if column in full_df.columns and column not in provenance_exclusions:
            provenance_exclusions[column] = "metadata_identifier_or_target"
    for column in artifact_cols + merge_artifact_leakage:
        provenance_exclusions[column] = "merge_or_file_artifact"

    container.metadata = full_df[[column for column in METADATA_COLUMNS if column in full_df.columns]].copy()
    container.y = full_df["Multiclass_Target"].to_numpy(dtype=int)
    container.groups = full_df["gene_symbol"].to_numpy()

    # Inheritance and engineered features.
    pli_column = "gnomad_pLI" if "gnomad_pLI" in full_df.columns and "gnomad_pLI" not in exclude else None
    for column in [] if pli_column else full_df.columns:
        if re.search("(^|[^a-z0-9])pli([^a-z0-9]|$)", column.lower()) and column not in exclude:
            if pd.to_numeric(full_df[column], errors="coerce").notna().sum() > 50:
                pli_column = column
                break
    logger.log(f"  pLI column used for inheritance interactions: {pli_column}")
    inheritance_df, inheritance_features = build_inheritance_features(full_df, pli_column)
    for column in inheritance_features:
        if column not in full_df.columns:
            full_df[column] = inheritance_df[column].to_numpy()
    full_df, engineered_features = engineer_features(full_df)
    mouse_engineered = [column for column in MOUSE_DERIVED_ENGINEERED if column in engineered_features]
    engineered_features = [column for column in engineered_features if column not in mouse_engineered]

    organ_system_features = [column for column in full_df.columns if column.startswith("feat_hpo_sys_")]
    hpo_binary_cols = [
        column
        for column in full_df.columns
        if column.startswith("hpo_")
        and column != "hpo_term"
        and pd.api.types.is_numeric_dtype(full_df[column])
    ]
    container.hpo_binary_cols = hpo_binary_cols

    # The four feature sets.
    s1_cols = [
        column
        for column in cleaned_mechanistic
        + external_cols
        + new_gene_cols
        + variant_cols
        + inheritance_features
        + engineered_features
        if column in full_df.columns
        and not column.startswith(("hpo_", "feat_hpo_sys_", "feat_mondo_"))
        and column != "has_mondo_record"
    ]
    mouse_cols = mouse_lethality_cols + mouse_human_phenotype_cols + mouse_engineered
    s1b_cols = [
        column
        for column in s1_cols + organ_system_features + mondo_cols
        if column in full_df.columns and not column.startswith("hpo_")
    ]
    s2_raw = [
        column
        for column in s1b_cols + hpo_binary_cols + mouse_cols
        if column in full_df.columns and not column.startswith("feat_hpo_sys_")
    ]
    s3_raw = [column for column in hpo_binary_cols if column in full_df.columns]
    s2_cols, s2_dropped = filter_leaky_columns_for_phenotype_scenarios(s2_raw, "S2_Mech_Pheno")
    s3_cols, s3_dropped = filter_leaky_columns_for_phenotype_scenarios(s3_raw, "S3_Phenotype")
    container.hpo_column_term_names = dict(HPO_COLUMN_TERM_NAMES)
    severity_rows = []
    for scenario, columns, dropped in (
        ("S2_Mech_Pheno", s2_cols, s2_dropped),
        ("S3_Phenotype", s3_cols, s3_dropped),
    ):
        container.scenario_static_drops[scenario] = dropped
        container.severity_hpo_candidates[scenario] = [
            c for c, reason in dropped if reason == "severity_hpo_blocklist"
        ]
        leftover = [
            column for column in columns if column.startswith("hpo_") and is_severity_hpo_column(column)
        ]
        if leftover:
            raise RuntimeError(f"{scenario}: severity-term columns left after filtering: {leftover[:10]}")
    for scenario in ("S2_Mech_Pheno", "S3_Phenotype"):
        for column in container.severity_hpo_candidates[scenario]:
            match = re.search("HP_(\\d{7})", column)
            hpo_id = f"HP:{match.group(1)}" if match else ""
            severity_rows.append(
                {
                    "Scenario": scenario,
                    "feature": column,
                    "hpo_id": hpo_id,
                    "hpo_term_name": HPO_COLUMN_TERM_NAMES.get(column, ""),
                    "rule": "identifier_list" if hpo_id in SEVERITY_HPO_IDS else "severity_word_in_name",
                }
            )
    pd.DataFrame(severity_rows).to_csv(
        os.path.join(Config.OUTPUT_DIR, "results", "severity_hpo_terms_removed_at_build.csv"), index=False
    )
    logger.log(
        f"  Severity-term rules removed {len(container.severity_hpo_candidates['S3_Phenotype'])} HPO columns "
        f"from S3 and {len(container.severity_hpo_candidates['S2_Mech_Pheno'])} from S2"
    )
    container.scenario_cols = {
        "S1_Mechanistic": list(dict.fromkeys(s1_cols)),
        "S1b_Disease_Context": list(dict.fromkeys(s1b_cols)),
        "S2_Mech_Pheno": list(dict.fromkeys(s2_cols)),
        "S3_Phenotype": list(dict.fromkeys(s3_cols)),
    }

    # Columns removed from every feature set (see Config for the reasons).
    _drop_from_all_scenarios(container, lambda c: c in set(Config.ORGAN_CRITICALITY_COLUMNS))
    container.duplicate_column_drops = _drop_from_all_scenarios(
        container, lambda c: c in set(Config.DUPLICATE_COLUMNS)
    )
    _drop_from_all_scenarios(container, lambda c: c == "has_mondo_record")
    record_drops = _drop_from_all_scenarios(container, lambda c: c in set(Config.RECORD_KEEPING_COLUMNS))
    non_biological = _drop_from_all_scenarios(container, _is_non_biological)
    redundant = _drop_from_all_scenarios(container, lambda c: c in set(Config.REDUNDANT_FEATURE_COLUMNS))
    logger.log(
        f"  Removed from all feature sets: record-keeping {record_drops}; study-intensity {non_biological}; redundant {redundant}"
    )
    proxies = set(Config.INHERITANCE_COUNT_PROXY_COLUMNS)
    container.inheritance_proxy_drops_S1 = [
        c for c in container.scenario_cols["S1_Mechanistic"] if c in proxies
    ]
    container.scenario_cols["S1_Mechanistic"] = [
        c for c in container.scenario_cols["S1_Mechanistic"] if c not in proxies
    ]

    # Feature family of each column (later assignments take precedence).
    family_lookup: Dict[str, str] = {}
    for columns, family in (
        (cleaned_mechanistic, "mechanistic_core"),
        (external_cols, "external"),
        (new_gene_cols, "gene_constraint_v2"),
        (variant_cols, "variant"),
        (inheritance_features, "inheritance"),
        (engineered_features, "engineered"),
        (mouse_engineered, "engineered"),
        (organ_system_features, "organ_system"),
    ):
        for column in columns:
            family_lookup[column] = family
    for column in mondo_cols:
        family_lookup[column] = "missingness_indicator" if column == "has_mondo_record" else "mondo"
    for columns, family in (
        (hpo_binary_cols, "hpo_binary"),
        (mouse_lethality_cols, "mouse_lethality"),
        (mouse_human_phenotype_cols, "mouse_phenotype"),
    ):
        for column in columns:
            family_lookup[column] = family
    container.feature_family_lookup = family_lookup
    container.full_df = full_df
    for scenario, columns in container.scenario_cols.items():
        logger.log(f"  Scenario {scenario}: {len(columns)} candidate columns")
    _write_feature_provenance_table(container, provenance_exclusions, disease_df.columns)
    container.text_token_maps = _load_text_token_maps()
    return container


def _write_feature_provenance_table(
    container: DataContainer, provenance_exclusions: Dict[str, str], label_source_columns
) -> None:
    """Supplementary Table 1: every candidate feature with its family, source and feature sets."""
    family_lookup = container.feature_family_lookup
    risk_by_family = {
        "mechanistic_core": "low_mechanistic",
        "external": "low_to_moderate_external_mechanistic",
        "variant": "moderate_variant_annotation",
        "inheritance": "moderate_clinical_annotation",
        "engineered": "derived_from_allowed_inputs",
        "primary_text": "moderate_text_annotation_same_source_review",
        "gnn_text": "moderate_text_annotation",
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
        "gnn_text": "gnn_dataset",
        "organ_system": "primary_label_source",
        "mondo": "MONDO_ontology",
        "hpo_binary": "primary_label_source_and_HPO",
        "mouse_lethality": "mouse_lethality_features",
        "mouse_phenotype": "mouse_human_phenotype_features",
        "missingness_indicator": "source_availability",
    }
    rows: List[Dict[str, Any]] = []
    active_features = sorted(set().union(*(set(value) for value in container.scenario_cols.values())))
    for feature in active_features:
        family = family_lookup.get(feature, _classify_feature_family(feature, family_lookup))
        rows.append(
            {
                "feature": feature,
                "family": family,
                "source_dataset": source_by_family.get(family, "manual_review_required"),
                "used_in_scenarios": "|".join(
                    s for s, columns in container.scenario_cols.items() if feature in columns
                ),
                "provenance_risk": risk_by_family.get(family, "unclassified_review_manually"),
                "available_at_prediction_time": True,
                "derived_from_target": False,
                "same_source_as_target_label": family in {"primary_text", "organ_system", "hpo_binary"},
                "missing_value_semantics": _missing_value_semantics_for_family(family),
                "included_in_main_clean_run": True,
                "exclusion_reason": "",
                "mi_used_to_declare_leakage": False,
                "manual_review_status": "rule_based_inclusion_requires_domain_confirmation",
            }
        )
    for feature, reason in sorted(provenance_exclusions.items()):
        rows.append(
            {
                "feature": feature,
                "family": "excluded",
                "source_dataset": "assembled_input",
                "used_in_scenarios": "",
                "provenance_risk": "excluded_by_provenance_rule",
                "available_at_prediction_time": False,
                "derived_from_target": "target" in reason
                or feature in {"Multiclass_Target", "target_severity"},
                "same_source_as_target_label": feature in label_source_columns,
                "missing_value_semantics": "not_applicable",
                "included_in_main_clean_run": False,
                "exclusion_reason": reason,
                "mi_used_to_declare_leakage": False,
                "manual_review_status": "excluded",
            }
        )
    for scenario, drops in container.scenario_static_drops.items():
        for feature, reason in drops:
            rows.append(
                {
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
                }
            )
    path = os.path.join(Config.OUTPUT_DIR, "results", "feature_provenance_table.csv")
    pd.DataFrame(rows).drop_duplicates(
        subset=["feature", "used_in_scenarios", "included_in_main_clean_run"]
    ).to_csv(path, index=False)
    logger.log(f"  Wrote feature provenance table: {path}")
