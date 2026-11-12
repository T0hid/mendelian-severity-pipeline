"""Input tables (Data and severity labels; Gene-level and cross-species features).

Loads the severity-labelled gene-disease pairs and the gene-level feature tables, applies the
HPO-verified gene-disease mapping (59 of 6,684 candidate pairs removed) and builds one row per
pair with its HPO term indicators and pair-level inheritance.
"""

import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..mapping import apply_rebuilt_mapping
from .config import Config
from .ontology import (
    parse_hpo_obo,
    resolve_hpo_terms,
    safe_feature_name,
    safe_hpo_colname,
)
from .runtime import HPO_COLUMN_TERM_NAMES, logger


def load_mechanistic_features() -> pd.DataFrame:
    logger.log("Loading mechanistic features...")
    if not os.path.exists(Config.CORE_FEATURES):
        raise FileNotFoundError(Config.CORE_FEATURES)
    df = pd.read_csv(Config.CORE_FEATURES)
    logger.log(f"  Loaded {len(df)} genes x {len(df.columns)} columns")
    df = _drop_removed_input_columns(df, "CORE")
    rollups = [c for c in Config.CORE_INHERITANCE_ROLLUP_COLUMNS if c in df.columns]
    if rollups:
        df = df.drop(columns=rollups)
        logger.log(f"  Removed gene-level inheritance rollups (pair-level inheritance is used): {rollups}")
    return df


def _drop_removed_input_columns(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Drop the broken or duplicated input columns listed in Config for this source table."""
    listed = getattr(Config, f"REMOVED_INPUT_COLUMNS_{source_name}")
    removed = {column: reason for column, reason in listed.items() if column in df.columns}
    if removed:
        df = df.drop(columns=list(removed))
        logger.log(f"  {source_name}: removed {len(removed)} input columns: {removed}")
    return df


def load_variant_features() -> pd.DataFrame:
    """Load the optional variant table. Its columns are excluded from every feature set."""
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
    """Load mouse viability and lethality features (combined feature set only)."""
    logger.subsection("Loading Mouse Lethality Features")
    if not os.path.exists(Config.MOUSE_LETHALITY_FEATURES):
        logger.log("  Mouse lethality features file not found", level="WARNING")
        return pd.DataFrame()
    df = pd.read_csv(Config.MOUSE_LETHALITY_FEATURES)
    for col in [c for c in df.columns if c != "gene_symbol"]:
        df[col] = df[col].replace({"True": 1.0, "False": 0.0, True: 1.0, False: 0.0})
        df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.log(
        f"  Loaded mouse lethality features for {len(df)} genes ({len([c for c in df.columns if c != 'gene_symbol'])} columns)"
    )
    return df


def load_mouse_human_phenotype_features() -> pd.DataFrame:
    """Load features linking mouse lethality to 20 human clinical findings (combined feature set only)."""
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
    """Load s_het (GeneBayes), Human Protein Atlas specificity, BrainSpan, DOMINO and GDI features."""
    if not os.path.exists(Config.NEW_GENE_FEATURES):
        logger.log("  New gene-level features file not found, continuing without it", level="WARNING")
        return (pd.DataFrame(), [])
    logger.subsection("Loading New Gene-Level Features (s_het / HPA / BrainSpan / DOMINO / GDI)")
    new_df = pd.read_csv(Config.NEW_GENE_FEATURES)
    identifier_columns = [c for c in ("hgnc_id",) if c in new_df.columns]
    if identifier_columns:
        logger.log(f"  Dropping identifier columns: {identifier_columns}")
        new_df = new_df.drop(columns=identifier_columns)
    drop_text = [c for c in new_df.columns if new_df[c].dtype == "object" and c != "gene_symbol"]
    if drop_text:
        logger.log(f"  Dropping text columns: {drop_text}")
        new_df = new_df.drop(columns=drop_text)
    new_df = new_df.drop_duplicates(subset="gene_symbol")
    new_cols = [c for c in new_df.columns if c != "gene_symbol"]
    logger.log(f"  New gene-level features: {len(new_cols)} columns for {len(new_df)} genes")
    return (new_df, new_cols)


def _add_pair_level_inheritance(disease_df: pd.DataFrame, moi_column: str) -> pd.DataFrame:
    """Derive inheritance flags for each gene-disease pair from its own mode(s) of inheritance.

    Pairs with only unknown or blank modes get NaN flags. A known mode outside the
    dominant / recessive / X-linked groups (e.g. Mitochondrial) gives all three flags 0.
    """
    dominant = set(Config.MOI_DOMINANT)
    recessive = set(Config.MOI_RECESSIVE)
    xlinked = set(Config.MOI_XLINKED)
    unknown = {value.lower() for value in Config.MOI_UNKNOWN}
    rows = []
    for modes in disease_df[moi_column]:
        known = sorted({str(m).strip() for m in modes or [] if str(m).strip().lower() not in unknown})
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
        rows.append(
            (
                float("dominant" in groups),
                float("recessive" in groups),
                float("xlinked" in groups),
                float(len(groups) > 1),
                float(len(groups)),
                "|".join(known),
            )
        )
    flags = pd.DataFrame(
        rows,
        columns=[
            "is_dominant",
            "is_recessive",
            "is_xlinked",
            "has_mixed_inheritance",
            "n_inheritance_modes",
            "all_inheritance_modes",
        ],
        index=disease_df.index,
    )
    disease_df = pd.concat([disease_df, flags], axis=1)
    n_unknown = int(flags["is_dominant"].isna().sum())
    logger.log(
        f"  Pair-level inheritance from '{moi_column}': dominant={int(flags['is_dominant'].sum())}, recessive={int(flags['is_recessive'].sum())}, X-linked={int(flags['is_xlinked'].sum())}, mixed={int(flags['has_mixed_inheritance'].sum())}, unknown (NaN)={n_unknown} of {len(flags)} pairs"
    )
    return disease_df


def build_gene_disease_target(primary_data_path: str) -> pd.DataFrame:
    """One row per gene-disease pair with its severity class, HPO term indicators and inheritance."""
    logger.subsection("Building gene-disease pair targets")
    df = pd.read_csv(primary_data_path, low_memory=False)
    col_map = {
        "Gene Name": "gene_symbol",
        "MONDO_ID": "mondo_id",
        "Severity": "target_severity",
        "HPO_ID": "hpo_id",
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
    logger.log("  Applying HPO-verified gene-disease mapping")
    df = apply_rebuilt_mapping(
        df,
        gene_col="gene_symbol",
        mondo_col="mondo_id",
        mapping_path=Config.REBUILT_GENE_DISEASE_MAPPING,
        log=logger.log,
        audit_dir=os.path.join(Config.OUTPUT_DIR, "results"),
        audit_name="gene_disease_mapping_filter_audit.csv",
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
        f"  Severity-label conflict audit: {n_conflict} conflicting gene-disease pairs; wrote {conflict_path}"
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
        df = df.loc[~df[["gene_symbol", "mondo_id"]].apply(tuple, axis=1).isin(conflict_keys)].copy()
        logger.log(
            f"  Excluding {before - len(df)} source rows from {n_conflict} gene-disease pairs with conflicting severity labels"
        )
    if df.empty:
        raise ValueError("No labelled rows remain after excluding conflicting severity labels.")

    def get_list(series):
        values = set()
        for value in series.dropna():
            values.update((item.strip() for item in str(value).split("|") if item.strip()))
        return sorted(values)

    def first_mode(series):
        mode = series.mode(dropna=True)
        return mode.iloc[0] if len(mode) else series.iloc[0]

    agg_dict = {"target_severity": first_mode, "severity_numeric": first_mode}
    if "disease_name" in df.columns:
        agg_dict["disease_name"] = "first"
    if "hpo_term" in df.columns:
        agg_dict["hpo_term"] = get_list
    source_id_to_label: Dict[str, str] = {}
    if "hpo_id" in df.columns:
        valid_id = df["hpo_id"].astype(str).str.strip().str.fullmatch("HP:\\d{7}")
        label = df["hpo_term"] if "hpo_term" in df.columns else pd.Series("", index=df.index)
        df["hpo_token"] = np.where(valid_id, df["hpo_id"].astype(str).str.strip(), label)
        n_label_only = int((~valid_id & label.notna()).sum())
        logger.log(
            f"  HPO features keyed by HPO_ID; {n_label_only} source rows without a valid HPO_ID fall back to the term label"
        )
        for hpo_id_value, label_value in zip(df.loc[valid_id, "hpo_id"], label[valid_id]):
            if isinstance(label_value, str) and label_value.strip():
                source_id_to_label.setdefault(str(hpo_id_value).strip(), label_value.strip())
        agg_dict["hpo_token"] = get_list
    if "parent_term" in df.columns:
        agg_dict["parent_term"] = get_list
    moi_column = Config.PAIR_INHERITANCE_COLUMN
    if moi_column not in df.columns:
        raise ValueError(f"The primary data has no '{moi_column}' column.")
    agg_dict[moi_column] = get_list
    disease_df = df.groupby(["gene_symbol", "mondo_id"]).agg(agg_dict).reset_index()
    disease_df = disease_df.rename(columns={"severity_numeric": "Multiclass_Target"})
    disease_df["Multiclass_Target"] = disease_df["Multiclass_Target"].astype(int)
    disease_df = _add_pair_level_inheritance(disease_df, moi_column)
    disease_df = disease_df.drop(columns=[moi_column])
    logger.log("  Building HPO term indicators")
    hpo_terms, hpo_alt_to_id, hpo_name_to_id = parse_hpo_obo(Config.HPO_OBO_PATH)
    feature_rows = []
    for _, row in disease_df.iterrows():
        feature_row = {"gene_symbol": row["gene_symbol"], "mondo_id": row["mondo_id"]}
        token_column = "hpo_token" if "hpo_token" in row else "hpo_term"
        raw_hpo = row[token_column] if token_column in row and isinstance(row[token_column], list) else []
        direct_ids, unresolved_terms = resolve_hpo_terms(raw_hpo, hpo_alt_to_id, hpo_name_to_id)
        for hpo_id in sorted(direct_ids):
            column = safe_hpo_colname(hpo_id)
            feature_row[column] = 1
            if column not in HPO_COLUMN_TERM_NAMES:
                HPO_COLUMN_TERM_NAMES[column] = hpo_terms.get(hpo_id, {}).get(
                    "name"
                ) or source_id_to_label.get(hpo_id, "")
        for raw_term in sorted(unresolved_terms):
            if raw_term:
                column = safe_hpo_colname(raw_term)
                feature_row[column] = 1
                HPO_COLUMN_TERM_NAMES.setdefault(column, str(raw_term))
        if "parent_term" in row and isinstance(row["parent_term"], list):
            # Top-level HPO categories of the recorded terms: the organ-system indicators.
            for parent_term in row["parent_term"]:
                column = safe_feature_name("feat_hpo_sys", str(parent_term).lower(), max_len=50)
                feature_row[column] = 1
        feature_rows.append(feature_row)
    features_df = pd.DataFrame(feature_rows).fillna(0)
    n_label_columns = sum(
        (
            1
            for column in features_df.columns
            if column.startswith("hpo_") and (not re.fullmatch("hpo_HP_\\d{7}", column))
        )
    )
    logger.log(f"  HPO feature columns named by label rather than HPO ID: {n_label_columns}")
    pd.DataFrame(sorted(HPO_COLUMN_TERM_NAMES.items()), columns=["column", "hpo_term_name"]).to_csv(
        os.path.join(Config.OUTPUT_DIR, "results", "hpo_column_term_names.csv"), index=False
    )
    disease_df = disease_df.drop(columns=[c for c in ("hpo_token",) if c in disease_df.columns])
    disease_df = disease_df.merge(features_df, on=["gene_symbol", "mondo_id"], how="left", validate="1:1")
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
