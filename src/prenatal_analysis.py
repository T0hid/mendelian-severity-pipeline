#!/usr/bin/env python3
"""Apply the disease-level phenotype scorer to prenatal profiles and analyse outcomes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import re
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_VERSION = "1.0.0"

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "phenotype-matplotlib-cache")
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats

try:
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    HAS_SM = True
except Exception:
    HAS_SM = False


# outcome mapping
LIVE_BIRTH = {
    "livebirth",
    "live birth",
    "live born child - no nicu/special care required",
    "live born child - nicu/special care required",
    "live born child - special care nursery at birth",
    "live born child - nicu 1-7 days",
    "live born child - nicu 2-4 weeks",
    "live born child - nicu 4+ weeks",
}
NICU_SET = {
    # Special-care nursery is grouped with NICU because it represents
    # escalated neonatal care.
    "live born child - special care nursery at birth",
    "live born child - nicu/special care required",
    "live born child - nicu 1-7 days",
    "live born child - nicu 2-4 weeks",
    "live born child - nicu 4+ weeks",
}
TOP_SET = {
    "top",
    "top following genomic result",
    "top before interview",
    "termination of pregnancy",
}
DEATH_SET = {
    "stillbirth",
    "stillbirth (fdiu from 20 weeks to birth)",
    "neonatal death",
    "neonatal death (following birth)",
    "miscarriage",
    "live born child - death after 28 days",
}

LIVE_BIRTH_NO_NICU = LIVE_BIRTH - NICU_SET
OUTCOME_LABELS = {0: "Live birth", 1: "TOP", 2: "Death"}
OUTCOME4_LABELS = {
    0: "Live birth (no NICU)",
    1: "NICU / special care",
    2: "TOP",
    3: "Death",
}
OUTCOME4_PLOT_LABELS = {
    0: "Live birth\n(no NICU)",
    1: "NICU /\nspecial care",
    2: "TOP",
    3: "Death",
}
HPO_RE = re.compile(r"HP[:_]?([0-9]{7})", re.IGNORECASE)


SCORE_CHANNEL_LABELS = {
    "combined": "Combined multichannel severity score",
    "phenotype": "Phenotype-derived severity score",
    "headline": "Model headline severity score",
}


def _score_channel_label(score_channel: str) -> str:
    return SCORE_CHANNEL_LABELS.get(
        str(score_channel),
        f"{str(score_channel).replace('_', ' ').title()} severity score",
    )


def _normalise_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def _is_death_outcome(text: str) -> bool:
    return text in DEATH_SET or any(token in text for token in (
        "miscarriage",
        "stillbirth",
        "still birth",
        "fdiu",
        "fetal death",
        "foetal death",
        "neonatal death",
        "postnatal death",
        "death after",
        "died",
    ))


def _is_top_outcome(text: str) -> bool:
    return (
        text in TOP_SET
        or text.startswith("top ")
        or "termination of pregnancy" in text
    )


def map_outcome_3level(value: Any) -> float:
    """Map to Live birth (0), TOP (1), or Death (2)."""
    text = _normalise_text(value)
    if not text:
        return np.nan
    # Death must be checked before generic live-birth wording.
    if _is_death_outcome(text):
        return 2.0
    if _is_top_outcome(text):
        return 1.0
    if (
        text in LIVE_BIRTH
        or "live birth" in text
        or "livebirth" in text
        or text.startswith("live born child")
    ):
        return 0.0
    return np.nan


def map_outcome_4level(value: Any) -> float:
    """Map to no-NICU live birth (0), NICU/special care (1), TOP (2), Death (3)."""
    text = _normalise_text(value)
    if not text:
        return np.nan
    if _is_death_outcome(text):
        return 3.0
    if _is_top_outcome(text):
        return 2.0


    if text in LIVE_BIRTH_NO_NICU or "no nicu/special care required" in text:
        return 0.0
    if text in NICU_SET:
        return 1.0
    if text.startswith("live born child") and (
        "nicu" in text
        or "special care nursery" in text
        or "special care required" in text
    ):
        return 1.0
    if text in LIVE_BIRTH or "live birth" in text or "livebirth" in text:
        return 0.0
    return np.nan


def parse_hpo(cell: Any) -> List[str]:
    if pd.isna(cell):
        return []
    terms: List[str] = []
    seen = set()
    for match in HPO_RE.finditer(str(cell)):
        term = f"HP:{match.group(1)}"
        if term not in seen:
            seen.add(term)
            terms.append(term)
    return terms

# load + aggregate cohort  (cohort construction only; no study statistics)
def _unique_nonempty(values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = "" if pd.isna(value) else re.sub(r"\s+", " ", str(value).strip())
        if not text:
            continue
        key = text.casefold()
        if key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _join_unique_text(values: Sequence[Any], separator: str = " | ") -> str:
    return separator.join(_unique_nonempty(values))


def _union_hpo(values: Sequence[Any]) -> List[str]:
    terms: List[str] = []
    seen = set()
    for value in values:
        for term in parse_hpo(value):
            if term not in seen:
                seen.add(term)
                terms.append(term)
    return terms


def _resolve_family_outcome(values: Sequence[Any]) -> Dict[str, Any]:
    labels = _unique_nonempty(values)
    mapped3 = [map_outcome_3level(label) for label in labels]
    mapped4 = [map_outcome_4level(label) for label in labels]
    levels3 = sorted({int(v) for v in mapped3 if np.isfinite(v)})
    levels4 = sorted({int(v) for v in mapped4 if np.isfinite(v)})
    n_unmapped = sum(
        1 for v3, v4 in zip(mapped3, mapped4)
        if not np.isfinite(v3) and not np.isfinite(v4)
    )
    conflict = (
        len(levels3) > 1
        or len(levels4) > 1
        or (n_unmapped > 0 and (levels3 or levels4))
    )
    return {
        "selected_label": labels[0] if labels and not conflict else "",
        "all_labels": " | ".join(labels),
        "outcome_3level": float(levels3[0]) if len(levels3) == 1 and not conflict else np.nan,
        "outcome_4level": float(levels4[0]) if len(levels4) == 1 and not conflict else np.nan,
        "outcome_conflict": bool(conflict),
        "n_unmapped_labels": int(n_unmapped),
    }


def _aggregate_by_family(
    df: pd.DataFrame,
    family_col: str,
    outcome_col: str,
    hpo_col: str,
    absent_hpo_col: Optional[str],
    study_col: str,
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    special_cols = {family_col, outcome_col, hpo_col}
    if absent_hpo_col:
        special_cols.add(absent_hpo_col)

    for family_id, group in df.groupby(family_col, sort=False, dropna=False):
        record: Dict[str, Any] = {family_col: str(family_id)}
        for column in df.columns:
            if column in special_cols:
                continue
            record[column] = _join_unique_text(group[column].tolist())

        hpo_terms = _union_hpo(group[hpo_col].tolist())
        absent_terms = (
            _union_hpo(group[absent_hpo_col].tolist())
            if absent_hpo_col
            else []
        )
        resolved = _resolve_family_outcome(group[outcome_col].tolist())

        record[hpo_col] = ";".join(hpo_terms)
        if absent_hpo_col:
            record[absent_hpo_col] = ";".join(absent_terms)
        record[outcome_col] = resolved["selected_label"]
        record["pregnancy_outcome_all"] = resolved["all_labels"]
        record["outcome_3level"] = resolved["outcome_3level"]
        record["outcome_4level"] = resolved["outcome_4level"]
        record["outcome_conflict"] = resolved["outcome_conflict"]
        record["n_unmapped_outcome_labels"] = resolved["n_unmapped_labels"]
        record["source_row_count"] = int(len(group))
        record["source_studies"] = (
            _join_unique_text(group[study_col].tolist())
            if study_col in group.columns
            else ""
        )
        records.append(record)

    return pd.DataFrame.from_records(records)


def load_cohort(path: str, exclude_studies: Sequence[str],
                livebirth_only_studies: Sequence[str],
                family_col: str = "family_id",
                outcome_col: str = "pregnancy_outcome",
                hpo_col: str = "hpo_ids",
                absent_hpo_col: Optional[str] = None,
                study_col: str = "study") -> Tuple[pd.DataFrame, Dict]:
    df = pd.read_csv(path, low_memory=False)
    audit: Dict[str, Any] = {"n_start_rows": int(len(df))}

    required = [outcome_col, hpo_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"Cohort is missing required columns: {missing}. "
            f"Available columns are: {list(df.columns)}"
        )
    if absent_hpo_col and absent_hpo_col not in df.columns:
        raise ValueError(
            f"Requested absent-HPO column {absent_hpo_col!r} is missing. "
            f"Available columns are: {list(df.columns)}"
        )

    # family_id is useful but not required. Missing IDs are made row-specific so
    # unrelated records are never merged accidentally.
    if family_col not in df.columns:
        df[family_col] = [f"ROW_{i+1:06d}" for i in range(len(df))]
        audit["family_id_generated_from_rows"] = True
    else:
        audit["family_id_generated_from_rows"] = False
        raw_family = df[family_col]
        missing_family = raw_family.isna() | raw_family.astype(str).str.strip().str.casefold().isin(
            {"", "nan", "none", "null"}
        )
        df[family_col] = raw_family.astype(str).str.strip()
        if missing_family.any():
            replacements = [
                f"ROW_MISSING_{i+1:06d}"
                for i in range(int(missing_family.sum()))
            ]
            df.loc[missing_family, family_col] = replacements
    audit["n_missing_family_ids_replaced"] = int(
        sum(str(v).startswith("ROW_MISSING_") for v in df[family_col])
    )

    has_study = study_col in df.columns
    if has_study and exclude_studies:
        wanted = {_normalise_text(v) for v in exclude_studies}
        df = df[~df[study_col].map(_normalise_text).isin(wanted)].copy()
    audit["n_after_exclude_studies_rows"] = int(len(df))


    lb_only = {_normalise_text(v) for v in livebirth_only_studies} if has_study else set()
    if lb_only:
        is_lb_only = df[study_col].map(_normalise_text).isin(lb_only)
        mapped = df[outcome_col].map(map_outcome_3level)
        is_live = mapped.eq(0.0)
        audit["n_nonlivebirth_dropped_from_explicit_lb_only_studies"] = int(
            (is_lb_only & ~is_live).sum()
        )
        df = df[~(is_lb_only & ~is_live)].copy()
    else:
        audit["n_nonlivebirth_dropped_from_explicit_lb_only_studies"] = 0
    audit["n_after_livebirth_only_rule_rows"] = int(len(df))

    # Do not discard repeated families. Count exact duplicates for transparency,
    # then combine every row belonging to the same family.
    audit["n_exact_duplicate_rows_observed"] = int(df.duplicated().sum())
    family_sizes = df.groupby(family_col, sort=False).size()
    audit["n_multirow_families"] = int((family_sizes > 1).sum())
    audit["n_rows_in_multirow_families"] = int(family_sizes[family_sizes > 1].sum())

    family_df = _aggregate_by_family(
        df,
        family_col=family_col,
        outcome_col=outcome_col,
        hpo_col=hpo_col,
        absent_hpo_col=absent_hpo_col,
        study_col=study_col,
    )
    family_df["hpo_list"] = family_df[hpo_col].apply(parse_hpo)
    if absent_hpo_col:
        family_df["absent_hpo_list"] = family_df[absent_hpo_col].apply(parse_hpo)
    else:
        family_df["absent_hpo_list"] = [[] for _ in range(len(family_df))]
    family_df["n_hpo"] = family_df["hpo_list"].apply(len)
    family_df["n_absent_hpo"] = family_df["absent_hpo_list"].apply(len)

    # Keep every family profile. Missing outcomes and empty HPO profiles are not
    # deleted; they are flagged and remain in severity_by_family.csv.
    family_df["has_mapped_outcome_3level"] = family_df["outcome_3level"].notna()
    family_df["has_mapped_outcome_4level"] = family_df["outcome_4level"].notna()
    family_df["has_hpo"] = family_df["n_hpo"] > 0

    audit["n_family_profiles"] = int(len(family_df))
    audit["n_final"] = int(len(family_df))  # backwards-compatible check field
    audit["n_outcome_conflict_families"] = int(family_df["outcome_conflict"].sum())
    audit["n_with_hpo"] = int(family_df["has_hpo"].sum())
    audit["n_without_hpo"] = int((~family_df["has_hpo"]).sum())
    audit["n_mapped_to_3group"] = int(family_df["has_mapped_outcome_3level"].sum())
    audit["n_mapped_to_4group"] = int(family_df["has_mapped_outcome_4level"].sum())
    audit["n_missing_or_unmapped_outcome"] = int(
        (~family_df["has_mapped_outcome_3level"]).sum()
    )
    audit["outcome_counts"] = {
        OUTCOME_LABELS[g]: int((family_df["outcome_3level"] == g).sum())
        for g in OUTCOME_LABELS
    }
    audit["outcome_4group_counts"] = {
        OUTCOME4_LABELS[g]: int((family_df["outcome_4level"] == g).sum())
        for g in OUTCOME4_LABELS
    }
    return family_df.reset_index(drop=True), audit


def _load_scorer_module(scorer_code: Optional[str]):
    known_module_names = (
        "phenotype_scorer",
        "phenotype_severity_scorer_v3_3_monotone_ontology",
        "phenotype_severity_scorer_v3_1_visual_dashboard",
        "phenotype_severity_scorer_v3",
        "phenotype_severity_scorer_updated",
        "phenotype_severity_scorer",
    )

    if scorer_code:
        if not os.path.isfile(scorer_code):
            raise FileNotFoundError(f"Scorer code not found: {scorer_code}")
        module_name = "phenotype_severity_scorer_v3_3_monotone_ontology"
        spec = importlib.util.spec_from_file_location(module_name, scorer_code)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import scorer code from {scorer_code}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        module = None
        import_errors = []
        for module_name in known_module_names:
            try:
                module = __import__(module_name)
                break
            except ImportError as exc:
                import_errors.append(f"{module_name}: {exc}")
        if module is None:
            raise ImportError(
                "Could not import a phenotype severity scorer module. Tried: "
                + "; ".join(import_errors)
            )

    # Make all historical/current scorer module paths resolve to the loaded code.
    for alias in (*known_module_names, "phenotype_severity_scorer_v3_3_monotone_ontology"):
        sys.modules[alias] = module

    legacy_aliases = {
        "ProportionalOddsModel": "OrdinalThresholdModel",
        "OrdinalModel": "OrdinalThresholdModel",
    }
    for old_name, current_name in legacy_aliases.items():
        if not hasattr(module, old_name) and hasattr(module, current_name):
            setattr(module, old_name, getattr(module, current_name))

    # Command-line training can pickle scorer classes under __main__.
    main_mod = sys.modules["__main__"]
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type):
            setattr(main_mod, name, obj)
    for old_name, current_name in legacy_aliases.items():
        if hasattr(module, current_name):
            setattr(main_mod, old_name, getattr(module, current_name))
    return module


class _ScorerCompatUnpickler(pickle.Unpickler):

    def __init__(self, file_obj, scorer_module):
        super().__init__(file_obj)
        self.scorer_module = scorer_module
        self.aliases = {
            "ProportionalOddsModel": "OrdinalThresholdModel",
            "OrdinalModel": "OrdinalThresholdModel",
        }

    def find_class(self, module_name, class_name):
        is_scorer_module = (
            module_name == "__main__"
            or module_name == "phenotype_scorer"
            or module_name == "phenotype_severity_scorer_v3_3_monotone_ontology"
            or module_name.startswith("phenotype_severity_scorer")
        )
        if is_scorer_module:
            target_name = self.aliases.get(class_name, class_name)
            if hasattr(self.scorer_module, target_name):
                return getattr(self.scorer_module, target_name)
            if hasattr(self.scorer_module, class_name):
                return getattr(self.scorer_module, class_name)
        return super().find_class(module_name, class_name)


def _upgrade_loaded_scorer_state(scorer, scorer_module) -> List[str]:
    repairs: List[str] = []
    expander = getattr(scorer, "expander", None)
    builder = getattr(scorer, "builder", None)
    vectorizer = getattr(scorer, "vectorizer", None)


    # a compatibility alias. Recreate only the alias if it is absent.
    if vectorizer is None and builder is not None and hasattr(builder, "phenotype"):
        scorer.vectorizer = builder.phenotype
        vectorizer = scorer.vectorizer
        repairs.append("scorer.vectorizer_alias")

    if expander is not None:
        defaults = {
            "parents": {},
            "alt_to_id": {},
            "excluded": {"HP:0000001", "HP:0000118"},
        }
        for attribute, default in defaults.items():
            if not hasattr(expander, attribute):
                setattr(
                    expander,
                    attribute,
                    default.copy() if hasattr(default, "copy") else default,
                )
                repairs.append(f"expander.{attribute}")
        if not hasattr(expander, "names"):
            fallback_names = dict(getattr(vectorizer, "id_to_name", {}) or {})
            expander.names = fallback_names
            repairs.append("expander.names")
        if not hasattr(expander, "enabled"):
            expander.enabled = bool(getattr(expander, "parents", {}))
            repairs.append("expander.enabled")

    if vectorizer is not None:
        if not hasattr(vectorizer, "id_to_name"):
            vectorizer.id_to_name = {}
            repairs.append("vectorizer.id_to_name")
        if not hasattr(vectorizer, "index"):
            vectorizer.index = {
                term: i for i, term in enumerate(getattr(vectorizer, "vocab", []))
            }
            repairs.append("vectorizer.index")

    if not hasattr(scorer, "config"):
        config_class = getattr(scorer_module, "ModelConfig", None)
        if config_class is not None:
            kwargs = {
                "C": float(getattr(scorer, "C", 1.0)),
                "min_count": int(getattr(vectorizer, "min_count", 2)),
                "max_freq": float(getattr(vectorizer, "max_freq", 0.95)),
                "max_hops": int(getattr(expander, "max_hops", 3)),
                "ancestor_decay": float(
                    getattr(vectorizer, "ancestor_decay", 1.0)
                ),
                "ic_weighting": bool(
                    getattr(vectorizer, "ic_weighting", False)
                ),
            }


            try:
                scorer.config = config_class(**kwargs)
                repairs.append("scorer.config")
            except TypeError:
                pass

    scorer_defaults = {
        "metadata": {},
        "term_weight_by_threshold": {},
        "min_supported_direct_terms": 1,
        "min_recognised_expanded_terms": 1,
    }
    if builder is not None:
        scorer_defaults.update({
            "combined_feature_weights": {},
            "dimension_models": {},
            "dimension_model_calibrated": {},
        })
    else:

        scorer_defaults.update({
            "use_augmented_headline": False,
            "use_mechanistic": False,
            "mech_mean": {},
            "mech_std": {},
            "feature_source": {},
            "aug_term_coef_by_threshold": {},
            "aug_mech_coef_by_threshold": {},
        })

    for attribute, default in scorer_defaults.items():
        if not hasattr(scorer, attribute):
            setattr(
                scorer,
                attribute,
                default.copy() if hasattr(default, "copy") else default,
            )
            repairs.append(f"scorer.{attribute}")

    for model_attribute in ("pheno_model", "combined_model", "aug_model"):
        model = getattr(scorer, model_attribute, None)
        if model is not None and not hasattr(model, "calibrator"):
            model.calibrator = None
            repairs.append(f"{model_attribute}.calibrator")
    for dimension, model in (getattr(scorer, "dimension_models", {}) or {}).items():
        if model is not None and not hasattr(model, "calibrator"):
            model.calibrator = None
            repairs.append(f"dimension_models.{dimension}.calibrator")

    return repairs


def _validate_model_compatibility(scorer) -> str:
    if not hasattr(scorer, "score_profile"):
        raise RuntimeError(
            "The loaded scorer code does not provide score_profile(); use the "
            "same scorer script that created model.pkl."
        )

    builder = getattr(scorer, "builder", None)
    vectorizer = getattr(scorer, "vectorizer", None)
    pheno_model = getattr(scorer, "pheno_model", None)
    combined_model = getattr(scorer, "combined_model", None)

    if builder is not None and combined_model is not None:
        phenotype = getattr(builder, "phenotype", vectorizer)
        if phenotype is None:
            raise ValueError("The v3 model is missing its phenotype vectorizer.")

        if pheno_model is not None:
            expected_pheno = (
                len(getattr(phenotype, "vocab", []))
                + len(getattr(getattr(builder, "explicit", None), "vocab", []))
            )

            actual_pheno = int(getattr(pheno_model, "n_features", -1))

            if actual_pheno != expected_pheno:
                raise RuntimeError(
                    "The v3 phenotype-channel layout is inconsistent: "
                    f"{actual_pheno} fitted features versus {expected_pheno} "
                    "phenotype + explicit features."
                )
        expected_combined = len(getattr(builder, "feature_names", []))
        actual_combined = int(getattr(combined_model, "n_features", -1))
        if actual_combined != expected_combined:
            raise RuntimeError(
                "The v3 combined-channel layout is inconsistent: "
                f"{actual_combined} fitted features versus {expected_combined} "
                "builder feature names. Use the scorer code that created the model."
            )
        return "v3_multichannel"

    if vectorizer is None or pheno_model is None:
        raise ValueError(
            "Saved artifact is missing a supported phenotype vectorizer/model "
            "or v3 combined model."
        )
    expected_features = len(getattr(vectorizer, "vocab", []))
    actual_features = int(getattr(pheno_model, "n_features", -1))
    if actual_features != expected_features:
        raise RuntimeError(
            "The saved phenotype model uses an unsupported legacy feature layout "
            f"({actual_features} fitted features versus {expected_features} HPO "
            "features). Retrain it with the revised scorer."
        )
    return "v2_phenotype_only"


def _probability_value(probabilities: Any, class_name: str) -> float:
    if not isinstance(probabilities, dict):
        return float("nan")
    value = pd.to_numeric(probabilities.get(class_name), errors="coerce")
    return float(value) if np.isfinite(value) else float("nan")


def _safe_int(value: Any, default: int = 0) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    return int(numeric) if np.isfinite(numeric) else int(default)


def _json_compact(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except Exception:
        return "{}"


def score_families(
    df: pd.DataFrame,
    model_path: str,
    scorer_code: Optional[str] = None,
    score_channel: str = "combined",
    allow_low_information: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """Score family profiles and export the phenotype and combined channels."""
    if score_channel not in SCORE_CHANNEL_LABELS:
        raise ValueError(
            f"Unknown score channel {score_channel!r}; choose from "
            f"{sorted(SCORE_CHANNEL_LABELS)}."
        )

    scorer_module = _load_scorer_module(scorer_code)
    with open(model_path, "rb") as fh:
        scorer = _ScorerCompatUnpickler(fh, scorer_module).load()
    repairs = _upgrade_loaded_scorer_state(scorer, scorer_module)
    if repairs:
        print("[model] compatibility fields restored: " + ", ".join(repairs))
    model_generation = _validate_model_compatibility(scorer)

    rows: List[Dict[str, Any]] = []
    for terms, absent_terms in zip(df["hpo_list"], df["absent_hpo_list"]):
        terms = list(terms)
        absent_terms = list(absent_terms)

        if len(terms) == 0 and len(absent_terms) == 0:
            rows.append({
                "severity_score": np.nan,
                "severity_score_headline": np.nan,
                "severity_score_combined": np.nan,
                "severity_score_phenotype": np.nan,
                "expected_severity_0_3": np.nan,
                "combined_expected_severity_0_3": np.nan,
                "phenotype_expected_severity_0_3": np.nan,
                "score_channel_used": score_channel,
                "severity_score_source": "",
                "score_status": "no_hpo_terms",
                "score_warnings": "No valid present or absent HPO IDs were available for severity scoring.",
                "n_terms_recognised": 0,
                "n_terms_recognised_expanded": 0,
                "n_terms_supported": 0,
                "n_valid_hpo_input": 0,
                "n_explicit_severity_terms": 0,
                "n_absent_terms_accepted": 0,
                "supported_fraction": np.nan,
                "learned_absent_channel_available": False,
                "explicit_annotation_expected_0_3": np.nan,
                "explicit_annotation_maximum_0_3": np.nan,
                "organ_burden_score_0_3": np.nan,
                "n_affected_organ_systems": 0,
                "affected_organ_systems": "",
                "class_probability_mild": np.nan,
                "class_probability_moderate": np.nan,
                "class_probability_severe": np.nan,
                "class_probability_profound": np.nan,
                "phenotype_probability_mild": np.nan,
                "phenotype_probability_moderate": np.nan,
                "phenotype_probability_severe": np.nan,
                "phenotype_probability_profound": np.nan,
                "severity_dimensions_json": "{}",
                "direct_term_effects_json": "[]",
                "absent_term_effects_json": "[]",
            })
            continue

        result = scorer.score_profile(
            terms,
            absent_hpo_terms=absent_terms,
            allow_low_information=allow_low_information,
        )

        headline_score = pd.to_numeric(
            result.get("severity_score_0_1"), errors="coerce"
        )
        phenotype_expected = pd.to_numeric(
            result.get("phenotype_only_expected_severity_0_3"),
            errors="coerce",
        )
        combined_expected = pd.to_numeric(
            result.get("combined_expected_severity_0_3"),
            errors="coerce",
        )
        headline_expected = pd.to_numeric(
            result.get("expected_severity_0_3"), errors="coerce"
        )

        phenotype_score = (
            float(phenotype_expected) / 3.0
            if np.isfinite(phenotype_expected)
            else np.nan
        )
        combined_score = (
            float(combined_expected) / 3.0
            if np.isfinite(combined_expected)
            else np.nan
        )


        if not np.isfinite(phenotype_score) and model_generation == "v2_phenotype_only":
            phenotype_score = float(headline_score) if np.isfinite(headline_score) else np.nan
            phenotype_expected = (
                float(headline_score) * 3.0 if np.isfinite(headline_score) else np.nan
            )
        if not np.isfinite(combined_score):
            combined_score = float(headline_score) if np.isfinite(headline_score) else phenotype_score
            combined_expected = (
                combined_score * 3.0 if np.isfinite(combined_score) else np.nan
            )

        if score_channel == "phenotype":
            selected_score = phenotype_score
        elif score_channel == "headline":
            selected_score = float(headline_score) if np.isfinite(headline_score) else combined_score
        else:
            selected_score = combined_score

        accepted_present = result.get(
            "accepted_present_terms",
            result.get("accepted_direct_terms", terms),
        ) or []
        phenotype_direct = result.get(
            "phenotype_direct_terms",
            result.get("accepted_direct_terms", accepted_present),
        ) or []
        explicit_terms = result.get("explicit_severity_terms", []) or []
        accepted_absent = result.get("accepted_absent_terms", absent_terms) or []

        n_supported = _safe_int(
            result.get(
                "n_supported_phenotype_direct_terms",
                result.get(
                    "n_supported_direct_terms",
                    result.get("n_recognised_direct_terms", 0),
                ),
            )
        )
        n_recognised = _safe_int(
            result.get("n_recognised_direct_terms", len(accepted_present)),
            default=len(accepted_present),
        )
        n_expanded = _safe_int(result.get("n_recognised_expanded_terms", 0))
        denominator = len(phenotype_direct)
        supported_fraction = (
            float(n_supported / denominator) if denominator > 0 else np.nan
        )

        dimensions = result.get("severity_dimensions", {}) or {}
        explicit_signal = dimensions.get("explicit_annotation_signal", {}) or {}
        organ_burden = dimensions.get("organ_system_burden", {}) or {}
        systems = organ_burden.get("affected_systems", []) or []
        learned_dimensions = (
            dimensions.get("learned_clinical_dimensions", {}) or {}
        )

        headline_probs = result.get("class_probabilities", {}) or {}
        phenotype_probs = result.get(
            "phenotype_only_class_probabilities", {}
        ) or {}

        row: Dict[str, Any] = {
            "severity_score": selected_score,
            "severity_score_headline": headline_score,
            "severity_score_combined": combined_score,
            "severity_score_phenotype": phenotype_score,
            "expected_severity_0_3": headline_expected,
            "combined_expected_severity_0_3": combined_expected,
            "phenotype_expected_severity_0_3": phenotype_expected,
            "score_channel_used": score_channel,
            "severity_score_source": str(
                result.get("severity_score_source", model_generation)
            ),
            "score_status": str(result.get("status", "unknown")),
            "score_warnings": " | ".join(
                map(str, result.get("warnings", []))
            ),
            "n_terms_recognised": n_recognised,
            "n_terms_recognised_expanded": n_expanded,
            "n_terms_supported": n_supported,
            "n_valid_hpo_input": len(accepted_present),
            "n_explicit_severity_terms": len(explicit_terms),
            "n_absent_terms_accepted": len(accepted_absent),
            "supported_fraction": supported_fraction,
            "learned_absent_channel_available": bool(
                result.get("learned_absent_channel_available", False)
            ),
            "explicit_annotation_expected_0_3": pd.to_numeric(
                explicit_signal.get("expected_severity_0_3"), errors="coerce"
            ),
            "explicit_annotation_maximum_0_3": pd.to_numeric(
                explicit_signal.get("maximum_severity_0_3"), errors="coerce"
            ),
            "organ_burden_score_0_3": pd.to_numeric(
                organ_burden.get("score_0_3"), errors="coerce"
            ),
            "n_affected_organ_systems": _safe_int(
                organ_burden.get("n_affected_systems", len(systems)),
                default=len(systems),
            ),
            "affected_organ_systems": ";".join(map(str, systems)),
            "class_probability_mild": _probability_value(headline_probs, "Mild"),
            "class_probability_moderate": _probability_value(headline_probs, "Moderate"),
            "class_probability_severe": _probability_value(headline_probs, "Severe"),
            "class_probability_profound": _probability_value(headline_probs, "Profound"),
            "phenotype_probability_mild": _probability_value(phenotype_probs, "Mild"),
            "phenotype_probability_moderate": _probability_value(phenotype_probs, "Moderate"),
            "phenotype_probability_severe": _probability_value(phenotype_probs, "Severe"),
            "phenotype_probability_profound": _probability_value(phenotype_probs, "Profound"),
            "severity_dimensions_json": _json_compact(dimensions),
            "direct_term_effects_json": _json_compact(
                result.get("direct_term_leave_one_out_effects", [])
            ),
            "absent_term_effects_json": _json_compact(
                result.get("absent_term_leave_one_out_effects", [])
            ),
        }
        for dimension, payload in learned_dimensions.items():
            if isinstance(payload, dict):
                row[f"dimension_{dimension}_expected_0_3"] = pd.to_numeric(
                    payload.get("expected_severity_0_3"), errors="coerce"
                )
        rows.append(row)

    score_table = pd.DataFrame.from_records(rows, index=df.index)
    out = pd.concat([df.copy(), score_table], axis=1)

    integer_columns = [
        "n_terms_recognised",
        "n_terms_recognised_expanded",
        "n_terms_supported",
        "n_valid_hpo_input",
        "n_explicit_severity_terms",
        "n_absent_terms_accepted",
        "n_affected_organ_systems",
    ]
    for column in integer_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype(int)

    numeric_columns = [
        column for column in out.columns
        if (
            column.startswith("severity_score")
            or column.endswith("_expected_0_3")
            or column.endswith("_maximum_0_3")
            or column.startswith("class_probability_")
            or column.startswith("phenotype_probability_")
            or column in {"supported_fraction", "organ_burden_score_0_3"}
        )
    ]
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out["score_available"] = np.isfinite(out["severity_score"])
    # Backwards-compatible column name. Outcome availability remains separate.
    out["score_included_in_analysis"] = out["score_available"]

    metadata = getattr(scorer, "metadata", {}) or {}
    src = metadata.get(
        "pipeline_version",
        metadata.get("pipeline", model_generation),
    )
    return out, str(src)

# correlation statistics  (single cohort, no clustering)
def _pairwise_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    combined = np.concatenate([pos, neg])
    ranks = stats.rankdata(combined)
    u = ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _concordance_fast(score: np.ndarray, outcome: np.ndarray) -> float:
    s = np.asarray(score, float)
    y = np.asarray(outcome, float)
    levels = np.unique(y[np.isfinite(y)])
    num = den = 0.0
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            lo = s[y == levels[i]]
            hi = s[y == levels[j]]
            if lo.size == 0 or hi.size == 0:
                continue
            weight = float(lo.size * hi.size)
            num += weight * _pairwise_auc(hi, lo)
            den += weight
    return float("nan") if den == 0 else num / den


def format_permutation_p(perm_p: float, n_perm_used: int, unicode_times: bool = True) -> str:
    """Render a permutation P value without collapsing the attainable floor to zero."""
    if perm_p is None or not np.isfinite(perm_p):
        return "not available"
    floor = 1.0 / (int(n_perm_used) + 1) if int(n_perm_used) > 0 else 0.0
    times = " × " if unicode_times else " x "
    def _scientific(value: float) -> str:
        exponent = int(np.floor(np.log10(value)))
        mantissa = value / (10.0 ** exponent)
        if unicode_times:
            superscripts = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
            return f"{mantissa:.1f}{times}10{str(exponent).translate(superscripts)}"
        return f"{mantissa:.1f}{times}10^{exponent}"
    if floor > 0 and perm_p <= floor * (1.0 + 1e-9):
        return _scientific(floor)
    if perm_p < 0.001:
        return _scientific(perm_p)
    return f"{perm_p:.3f}"
def correlation_analysis(df: pd.DataFrame, n_boot: int, n_perm: int,
                         seed: int = 0, score_col: str = "severity_score",
                         outcome_col: str = "outcome_3level") -> Dict:
    """Spearman rho, Kendall tau-b, ordinal concordance, bootstrap CI and
    label-permutation p-value for the requested ordered outcome."""
    rng = np.random.RandomState(seed)
    s = df[score_col].to_numpy(float)
    y = df[outcome_col].to_numpy(float)
    n = len(s)

    rho, _ = stats.spearmanr(s, y)
    tau, _ = stats.kendalltau(s, y, variant="b")
    conc = _concordance_fast(s, y)

    boot = []
    conc_boot = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        r, _ = stats.spearmanr(s[idx], y[idx])
        if np.isfinite(r):
            boot.append(r)
        c = _concordance_fast(s[idx], y[idx])
        if np.isfinite(c):
            conc_boot.append(c)
    boot = np.array(boot)
    conc_boot = np.array(conc_boot)
    ci = (
        (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
        if boot.size
        else (float("nan"), float("nan"))
    )

    null = []
    conc_null = []
    for _ in range(n_perm):
        shuffled = rng.permutation(y)
        r, _ = stats.spearmanr(s, shuffled)
        if np.isfinite(r):
            null.append(r)
        c = _concordance_fast(s, shuffled)
        if np.isfinite(c):
            conc_null.append(c)
    null = np.array(null)
    conc_null = np.array(conc_null)
    perm_p = (
        float((np.sum(np.abs(null) >= abs(rho)) + 1) / (null.size + 1))
        if null.size
        else float("nan")
    )
    # The concordance index is centred on 0.5 under the null, so its two-sided
    # permutation test compares |c - 0.5| rather than |c|.
    conc_perm_p = (
        float(
            (np.sum(np.abs(conc_null - 0.5) >= abs(conc - 0.5)) + 1)
            / (conc_null.size + 1)
        )
        if conc_null.size
        else float("nan")
    )
    conc_ci = (
        (
            float(np.percentile(conc_boot, 2.5)),
            float(np.percentile(conc_boot, 97.5)),
        )
        if conc_boot.size
        else (float("nan"), float("nan"))
    )

    return {
        "spearman_rho": float(rho),
        "kendall_tau_b": float(tau),
        "ordinal_concordance": float(conc),
        "ordinal_concordance_ci95_low": conc_ci[0],
        "ordinal_concordance_ci95_high": conc_ci[1],
        "ordinal_concordance_perm_p": conc_perm_p,
        "ci95_low": ci[0],
        "ci95_high": ci[1],
        "perm_p": perm_p,
        "n_boot_used": int(boot.size),
        "n_perm_used": int(null.size),
        "n": int(n),
        "score_col": score_col,
        "outcome_col": outcome_col,
    }


def proportional_odds(df: pd.DataFrame,
                      outcome_col: str = "outcome_3level",
                      score_col: str = "severity_score") -> Dict:
    """Ordinal regression of outcome on standardised severity. OR is per +1 SD."""
    s = df[score_col].to_numpy(float)
    sd = float(s.std())
    z = (s - s.mean()) / (sd if sd > 0 else 1.0)
    y = df[outcome_col].to_numpy(int)

    if HAS_SM:
        try:
            res = OrderedModel(y, z.reshape(-1, 1), distr="logit").fit(
                method="bfgs", disp=False
            )
            beta, se = float(res.params[0]), float(res.bse[0])
            lo = float(np.exp(beta - 1.96 * se))
            hi = float(np.exp(beta + 1.96 * se))
            p = (
                float(2 * (1 - stats.norm.cdf(abs(beta / se))))
                if se > 0
                else float("nan")
            )
            return {
                "available": True,
                "method": "statsmodels_ordered_logit",
                "score_col": score_col,
                "or_per_sd": float(np.exp(beta)),
                "ci95": [lo, hi],
                "p_value": p,
                "beta": beta,
                "se": se,
                "outcome_col": outcome_col,
            }
        except Exception as exc:
            statsmodels_reason = f"statsmodels failed: {exc}"
    else:
        statsmodels_reason = "statsmodels OrderedModel is unavailable"

    # Fallback: common-slope estimate from pooled "outcome >= k" logits.
    thresholds = sorted(np.unique(y))[1:]

    def _slope(zz, yy):
        slopes = []
        for k in thresholds:
            lab = (yy >= k).astype(int)
            if len(np.unique(lab)) < 2:
                continue
            model = LogisticRegression(max_iter=2000).fit(
                zz.reshape(-1, 1), lab
            )
            slopes.append(float(model.coef_[0, 0]))
        return float(np.mean(slopes)) if slopes else float("nan")

    beta = _slope(z, y)
    rng = np.random.RandomState(0)
    boot = []
    for _ in range(1000):
        idx = rng.randint(0, len(z), len(z))
        b = _slope(z[idx], y[idx])
        if np.isfinite(b):
            boot.append(b)
    boot = np.array(boot)
    if not np.isfinite(beta) or boot.size == 0:
        return {
            "available": False,
            "reason": f"{statsmodels_reason}; insufficient variation for fallback fit",
            "outcome_col": outcome_col,
        }
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = float(
        (np.sum((boot <= 0) if beta > 0 else (boot >= 0)) + 1)
        / (boot.size + 1)
    ) * 2
    return {
        "available": True,
        "method": "bootstrap_pooled_threshold_logit",
        "score_col": score_col,
        "or_per_sd": float(np.exp(beta)),
        "ci95": [float(np.exp(lo)), float(np.exp(hi))],
        "p_value": min(p, 1.0),
        "beta": float(beta),
        "outcome_col": outcome_col,
        "statsmodels_note": statsmodels_reason,
    }


def mean_ci_by_outcome(
    df: pd.DataFrame,
    outcome_col: str,
    labels: Dict[int, str],
    score_col: str = "severity_score",
) -> pd.DataFrame:
    rows = []
    for level, label in labels.items():
        sub = df[df[outcome_col] == level][score_col].to_numpy(float)
        if sub.size == 0:
            continue
        mean = float(sub.mean())
        se = float(sub.std(ddof=1) / np.sqrt(len(sub))) if len(sub) > 1 else 0.0
        rows.append({
            "outcome": label,
            "level": int(level),
            "n": int(len(sub)),
            "mean": mean,
            "ci_low": mean - 1.96 * se,
            "ci_high": mean + 1.96 * se,
        })
    return pd.DataFrame(rows)


def _binary_roc_contrasts(
    df: pd.DataFrame,
    outcome_col: str,
    specifications: Sequence[Tuple[str, Any, Any]],
    score_col: str = "severity_score",
) -> List[Dict]:
    score = df[score_col].to_numpy(float)
    outcome = df[outcome_col].to_numpy(int)
    results: List[Dict] = []

    for name, positive_rule, negative_rule in specifications:
        positive = np.asarray(positive_rule(outcome), dtype=bool)
        negative = np.asarray(negative_rule(outcome), dtype=bool)
        selected = positive | negative
        if positive.sum() == 0 or negative.sum() == 0:
            continue

        labels = positive[selected].astype(int)
        selected_scores = score[selected]
        fpr, tpr, _ = roc_curve(labels, selected_scores)
        results.append({
            "name": name,
            "score_col": score_col,
            "auc": float(roc_auc_score(labels, selected_scores)),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "n_pos": int(positive.sum()),
            "n_neg": int(negative.sum()),
        })

    return results


def clinical_contrasts_3group(df: pd.DataFrame,
                              score_col: str = "severity_score") -> List[Dict]:
    return _binary_roc_contrasts(
        df,
        score_col=score_col,
        outcome_col="outcome_3level",
        specifications=[
            ("(TOP+Death) vs live birth", lambda y: y >= 1, lambda y: y == 0),
            ("TOP vs live birth", lambda y: y == 1, lambda y: y == 0),
            ("Death vs rest", lambda y: y == 2, lambda y: y < 2),
            ("Death vs live birth", lambda y: y == 2, lambda y: y == 0),
        ],
    )


def clinical_contrasts_4group(df: pd.DataFrame,
                              score_col: str = "severity_score") -> List[Dict]:
    """Three clinically direct pairwise ROC contrasts for the 4-group outcome."""
    return _binary_roc_contrasts(
        df,
        score_col=score_col,
        outcome_col="outcome_4level",
        specifications=[
            (
                "Death vs live birth (no NICU)",
                lambda y: y == 3,
                lambda y: y == 0,
            ),
            (
                "TOP vs live birth (no NICU)",
                lambda y: y == 2,
                lambda y: y == 0,
            ),
            (
                "Death vs NICU / special care",
                lambda y: y == 3,
                lambda y: y == 1,
            ),
        ],
    )


CONTRIBUTING_SOURCE_RULES = (
    ("PreGen (in-house)", ("PreGen",)),
    ("PED_cohort", ("PED",)),
    ("ES_negative_cohort", ("ES_negative",)),
    ("EP_* (progressive pES)", ("EP_",)),
)


def assign_contributing_source(study: Any) -> str:
    """Families whose records span two EP_* sub-tables carry a compound study label
    and are counted once under EP_*."""
    text = str(study).strip()
    for label, prefixes in CONTRIBUTING_SOURCE_RULES:
        if any(text.startswith(prefix) for prefix in prefixes):
            return label
    return f"unassigned: {text}"


def source_sensitivity(
    df: pd.DataFrame,
    outcome_col: str,
    score_col: str = "severity_score",
    study_col: str = "study",
) -> Dict[str, pd.DataFrame]:
    """The three prespecified secondary analyses for contributing-source confounding."""
    work = df.copy()
    work["contributing_source"] = work[study_col].map(assign_contributing_source)
    score = work[score_col].to_numpy(float)
    outcome = work[outcome_col].to_numpy(float)
    levels = sorted(int(v) for v in np.unique(outcome[np.isfinite(outcome)]))

    composition = (
        work.groupby("contributing_source")[outcome_col]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=levels, fill_value=0)
        .reset_index()
    )
    composition.columns = ["contributing_source"] + [f"outcome_level_{k}" for k in levels]
    composition["n_families"] = composition[
        [f"outcome_level_{k}" for k in levels]
    ].sum(axis=1)

    rank_rows = []
    for name, group in work.groupby("contributing_source"):
        n_levels = int(group[outcome_col].nunique())
        if len(group) >= 3 and n_levels > 1:
            rho, p = stats.spearmanr(group[score_col], group[outcome_col])
        else:
            rho, p = float("nan"), float("nan")
        rank_rows.append({
            "analysis": "within-source",
            "stratum": name,
            "n": int(len(group)),
            "n_outcome_levels": n_levels,
            "spearman_rho": float(rho),
            "p_value": float(p),
        })
    finite = [r for r in rank_rows if np.isfinite(r["spearman_rho"])]
    if finite:
        total = sum(r["n"] for r in finite)
        rank_rows.append({
            "analysis": "within-source",
            "stratum": "sample-size-weighted mean",
            "n": int(total),
            "n_outcome_levels": np.nan,
            "spearman_rho": float(
                sum(r["n"] * r["spearman_rho"] for r in finite) / total
            ),
            "p_value": float("nan"),
        })
    for name in sorted(work["contributing_source"].unique()):
        group = work[work["contributing_source"] != name]
        if len(group) >= 3 and group[outcome_col].nunique() > 1:
            rho, p = stats.spearmanr(group[score_col], group[outcome_col])
        else:
            rho, p = float("nan"), float("nan")
        rank_rows.append({
            "analysis": "leave-one-source-out",
            "stratum": f"excluding {name}",
            "n": int(len(group)),
            "n_outcome_levels": int(group[outcome_col].nunique()),
            "spearman_rho": float(rho),
            "p_value": float(p),
        })
    rho, p = stats.spearmanr(score, outcome)
    rank_rows.append({
        "analysis": "pooled (primary)",
        "stratum": "all sources",
        "n": int(len(work)),
        "n_outcome_levels": len(levels),
        "spearman_rho": float(rho),
        "p_value": float(p),
    })

    sd = float(score.std())
    z = (score - score.mean()) / (sd if sd > 0 else 1.0)
    y = outcome.astype(int)
    order = sorted(work["contributing_source"].unique())
    dummies = np.column_stack(
        [(work["contributing_source"] == name).astype(float).to_numpy()
         for name in order[1:]]
    ) if len(order) > 1 else None

    def _fit(design: np.ndarray, label: str) -> Dict[str, Any]:
        if not HAS_SM:
            return {"model": label, "available": False,
                    "reason": "statsmodels OrderedModel is unavailable"}
        try:
            res = OrderedModel(y, design, distr="logit").fit(method="bfgs", disp=False)
            beta, se = float(res.params[0]), float(res.bse[0])
            return {
                "model": label,
                "available": True,
                "or_per_sd": float(np.exp(beta)),
                "ci95_low": float(np.exp(beta - 1.96 * se)),
                "ci95_high": float(np.exp(beta + 1.96 * se)),
                "p_value": float(2 * (1 - stats.norm.cdf(abs(beta / se)))) if se > 0 else float("nan"),
                "beta": beta,
                "se": se,
                "n": int(len(y)),
            }
        except Exception as exc:
            return {"model": label, "available": False, "reason": f"fit failed: {exc}"}

    odds = [_fit(z.reshape(-1, 1), "severity score only (unadjusted, primary)")]
    if dummies is not None:
        odds.append(_fit(np.column_stack([z, dummies]),
                         "severity score + contributing source (adjusted)"))

    return {
        "composition": composition,
        "rank": pd.DataFrame(rank_rows),
        "odds": pd.DataFrame(odds),
    }


def term_count_baseline(
    df: pd.DataFrame,
    outcome_col: str = "outcome_4level",
    count_col: str = "n_hpo",
    score_col: str = "severity_score",
    n_boot: int = 2000,
    n_perm: int = 2000,
    seed: int = 2,
    termination_level: Optional[int] = 2,
) -> Dict[str, pd.DataFrame]:
    """Compare the severity score with a baseline that needs no training: the
    number of distinct HPO terms recorded for each family (anomaly burden)."""
    work = df[[score_col, count_col, outcome_col]].dropna().reset_index(drop=True)

    def _partial_rho(ss: np.ndarray, cc: np.ndarray, yy: np.ndarray) -> float:
        rs, rc, ry = (stats.rankdata(v) for v in (ss, cc, yy))
        design = np.column_stack([np.ones_like(rc), rc])
        res_s = rs - design @ np.linalg.lstsq(design, rs, rcond=None)[0]
        res_y = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
        if np.std(res_s) == 0 or np.std(res_y) == 0:
            return float("nan")
        return float(np.corrcoef(res_s, res_y)[0, 1])

    subsets = [("all outcomes", work)]
    if termination_level is not None:
        subsets.append((
            "terminations excluded",
            work[work[outcome_col] != termination_level].reset_index(drop=True),
        ))

    rank_rows: List[Dict[str, Any]] = []
    for offset, (label, frame) in enumerate(subsets):
        if len(frame) < 4 or frame[outcome_col].nunique() < 2:
            continue
        for k, (name, col) in enumerate(
            (("severity score", score_col), ("HPO term count", count_col))
        ):
            res = correlation_analysis(
                frame, n_boot, n_perm, seed=seed + 10 * offset + k,
                score_col=col, outcome_col=outcome_col,
            )
            rank_rows.append({
                "subset": label, "n": res["n"], "measure": name,
                "spearman_rho": res["spearman_rho"],
                "ci95_low": res["ci95_low"], "ci95_high": res["ci95_high"],
                "perm_p": res["perm_p"],
            })
        ss = frame[score_col].to_numpy(float)
        cc = frame[count_col].to_numpy(float)
        yy = frame[outcome_col].to_numpy(float)
        rank_rows.append({
            "subset": label, "n": int(len(frame)),
            "measure": "severity score vs HPO term count",
            "spearman_rho": float(stats.spearmanr(ss, cc)[0]),
            "ci95_low": float("nan"), "ci95_high": float("nan"),
            "perm_p": float("nan"),
        })
        rng = np.random.RandomState(seed + 100 + offset)
        partial = _partial_rho(ss, cc, yy)
        boot = []
        for _ in range(n_boot):
            idx = rng.randint(0, len(frame), len(frame))
            if len(np.unique(yy[idx])) < 2:
                continue
            value = _partial_rho(ss[idx], cc[idx], yy[idx])
            if np.isfinite(value):
                boot.append(value)
        rank_rows.append({
            "subset": label, "n": int(len(frame)),
            "measure": "severity score given HPO term count (partial)",
            "spearman_rho": partial,
            "ci95_low": float(np.percentile(boot, 2.5)) if boot else float("nan"),
            "ci95_high": float(np.percentile(boot, 97.5)) if boot else float("nan"),
            "perm_p": float("nan"),
        })

    odds_rows: List[Dict[str, Any]] = []
    y = work[outcome_col].to_numpy(int)

    def _z(values: np.ndarray) -> np.ndarray:
        sd = float(values.std())
        return (values - values.mean()) / (sd if sd > 0 else 1.0)

    z_score = _z(work[score_col].to_numpy(float))
    z_count = _z(work[count_col].to_numpy(float))
    fits: Dict[str, Any] = {}
    if HAS_SM:
        for label, design, names in (
            ("score only", z_score.reshape(-1, 1), ["severity score"]),
            ("term count only", z_count.reshape(-1, 1), ["HPO term count"]),
            ("score + term count", np.column_stack([z_score, z_count]),
             ["severity score", "HPO term count"]),
        ):
            try:
                res = OrderedModel(y, design, distr="logit").fit(method="bfgs", disp=False)
                fits[label] = res
                for j, name in enumerate(names):
                    beta, se = float(res.params[j]), float(res.bse[j])
                    odds_rows.append({
                        "model": label, "covariate": name, "available": True,
                        "or_per_sd": float(np.exp(beta)),
                        "ci95_low": float(np.exp(beta - 1.96 * se)),
                        "ci95_high": float(np.exp(beta + 1.96 * se)),
                        "p_value": float(2 * (1 - stats.norm.cdf(abs(beta / se)))) if se > 0 else float("nan"),
                        "log_likelihood": float(res.llf), "n": int(len(y)),
                    })
            except Exception as exc:
                odds_rows.append({"model": label, "covariate": "", "available": False,
                                  "reason": f"fit failed: {exc}", "n": int(len(y))})
        both = fits.get("score + term count")
        for reduced, added in (("term count only", "severity score"),
                               ("score only", "HPO term count")):
            if both is not None and reduced in fits:
                lr = float(2 * (both.llf - fits[reduced].llf))
                odds_rows.append({
                    "model": f"likelihood-ratio test: {added} added to {reduced}",
                    "covariate": added, "available": True,
                    "lr_statistic": lr, "df": 1,
                    "p_value": float(stats.chi2.sf(max(lr, 0.0), 1)),
                    "n": int(len(y)),
                })
    else:
        odds_rows.append({"model": "all", "available": False,
                          "reason": "statsmodels OrderedModel is unavailable"})

    return {"rank": pd.DataFrame(rank_rows), "odds": pd.DataFrame(odds_rows)}


def adjacent_level_separation(
    df: pd.DataFrame,
    outcome_col: str,
    labels: Dict[int, str],
    n_boot: int = 2000,
    seed: int = 7,
    score_col: str = "severity_score",
) -> pd.DataFrame:
    """Does the score separate each neighbouring pair of outcome levels?"""
    rng = np.random.RandomState(seed)
    score = df[score_col].to_numpy(float)
    outcome = df[outcome_col].to_numpy(float)
    levels = sorted(int(v) for v in np.unique(outcome[np.isfinite(outcome)]))
    rows = []
    for lower, upper in zip(levels[:-1], levels[1:]):
        lo = score[outcome == lower]
        hi = score[outcome == upper]
        if lo.size == 0 or hi.size == 0:
            continue
        observed = _pairwise_auc(hi, lo)
        draws = []
        for _ in range(n_boot):
            lo_b = lo[rng.randint(0, lo.size, lo.size)]
            hi_b = hi[rng.randint(0, hi.size, hi.size)]
            value = _pairwise_auc(hi_b, lo_b)
            if np.isfinite(value):
                draws.append(value)
        draws = np.array(draws)
        ci_low, ci_high = (
            (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))
            if draws.size
            else (float("nan"), float("nan"))
        )
        rows.append({
            "boundary": f"{labels.get(lower, lower)} vs {labels.get(upper, upper)}",
            "lower_level": lower,
            "upper_level": upper,
            "n_lower": int(lo.size),
            "n_upper": int(hi.size),
            "mean_lower": float(lo.mean()),
            "mean_upper": float(hi.mean()),
            "mean_difference": float(hi.mean() - lo.mean()),
            "prob_more_severe_scores_higher": float(observed),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "resolved_at_95pct": bool(np.isfinite(ci_low) and ci_low > 0.5),
            "n_boot_used": int(draws.size),
            "score_col": score_col,
        })
    return pd.DataFrame(rows)


# Backwards-compatible name used by older external imports.
# figures
def _draw_group_distribution(
    ax,
    df: pd.DataFrame,
    outcome_col: str,
    levels: Sequence[int],
    labels: Sequence[str],
    colors: Sequence[str],
    title: str,
    score_label: str = "Severity score",
) -> None:
    data = [
        df[df[outcome_col] == level]["severity_score"].to_numpy(float)
        for level in levels
    ]
    nonempty = [(level, values) for level, values in zip(levels, data) if len(values)]
    violin_ready = [(level, values) for level, values in nonempty if len(values) >= 2]

    if violin_ready:
        parts = ax.violinplot(
            [values for _, values in violin_ready],
            positions=[level for level, _ in violin_ready],
            widths=0.8,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor("#a6cee3")
            body.set_alpha(0.45)

    if nonempty:
        ax.boxplot(
            [values for _, values in nonempty],
            positions=[level for level, _ in nonempty],
            widths=0.25,
            showfliers=False,
            medianprops=dict(color="black"),
        )

    rng = np.random.RandomState(0)
    for level, values, color in zip(levels, data, colors):
        if len(values):
            ax.scatter(
                level + rng.uniform(-0.09, 0.09, len(values)),
                values,
                s=14,
                alpha=0.55,
                color=color,
                edgecolors="none",
                zorder=3,
            )

    means = np.array(
        [values.mean() if len(values) else np.nan for values in data],
        dtype=float,
    )
    valid = np.isfinite(means)
    if valid.any():
        level_array = np.asarray(levels)
        ax.plot(level_array[valid], means[valid], "--", color="#b22222", zorder=4)
        ax.scatter(
            level_array[valid],
            means[valid],
            marker="D",
            s=90,
            color="#b22222",
            edgecolors="white",
            zorder=5,
            label="Group mean",
        )

    counts = [len(values) for values in data]
    ax.set_xticks(levels)
    ax.set_xticklabels([f"{label}\n(n={n})" for label, n in zip(labels, counts)])
    ax.set_ylabel(f"{score_label} (0-1)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylim(-0.02, 1.02)
    if valid.any():
        ax.legend(loc="upper left")


def _draw_mean_panel(
    ax,
    mean_tbl: pd.DataFrame,
    levels: Sequence[int],
    labels: Sequence[str],
    corr: Dict,
    title: str,
) -> None:
    ax.errorbar(
        mean_tbl["level"],
        mean_tbl["mean"],
        yerr=[
            mean_tbl["mean"] - mean_tbl["ci_low"],
            mean_tbl["ci_high"] - mean_tbl["mean"],
        ],
        fmt="o-",
        color="#1f77b4",
        capsize=5,
        markersize=9,
        linewidth=2,
    )
    for _, row in mean_tbl.iterrows():
        ax.annotate(
            f"n={int(row['n'])}",
            (row["level"], row["ci_high"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
        )
    ax.set_xticks(levels)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean severity score (95% CI)")
    ax.set_title(title, loc="left", fontweight="bold")
    text = (
        f"Spearman rho = {corr['spearman_rho']:+.3f}\n"
        f"(95% CI {corr['ci95_low']:+.3f}, {corr['ci95_high']:+.3f})\n"
        f"permutation p = "
        f"{format_permutation_p(corr['perm_p'], corr.get('n_perm_used', 0))}"
    )
    ax.text(
        0.97,
        0.04,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round", fc="white", ec="gray"),
    )


def build_3group_figure(
    df,
    mean_tbl,
    corr,
    contrasts,
    out_png,
    score_label: str = "Phenotype-derived severity score",
) -> None:
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(18, 6.2))
    levels = list(OUTCOME_LABELS)
    labels = [OUTCOME_LABELS[level] for level in levels]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    _draw_group_distribution(
        ax_a,
        df,
        "outcome_3level",
        levels,
        labels,
        colors,
        "a  Severity score by 3-level prenatal outcome",
        score_label,
    )
    _draw_mean_panel(
        ax_b,
        mean_tbl,
        levels,
        labels,
        corr,
        "b  Mean score by 3-level outcome",
    )

    for contrast, color in zip(
        contrasts,
        ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    ):
        ax_c.plot(
            contrast["fpr"],
            contrast["tpr"],
            color=color,
            linewidth=2,
            label=f"{contrast['name']}  AUC={contrast['auc']:.3f}",
        )
    ax_c.plot([0, 1], [0, 1], "--", color="gray")
    ax_c.set_xlabel("False-positive rate")
    ax_c.set_ylabel("True-positive rate")
    ax_c.set_title("c  ROC for clinical contrasts", loc="left", fontweight="bold")
    if contrasts:
        ax_c.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        f"{score_label} vs prenatal outcome (3 groups)",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Backwards-compatible function name.
def _draw_class_probability_panel(
    ax,
    df: pd.DataFrame,
    outcome_col: str = "outcome_4level",
) -> None:
    prefix = "phenotype_probability_" if getattr(ax, "_score_channel", "phenotype") == "phenotype" else "class_probability_"
    probability_columns = [prefix + n for n in ("mild", "moderate", "severe", "profound")]
    class_labels = ["Mild", "Moderate", "Severe", "Profound"]
    missing = [column for column in probability_columns if column not in df.columns]
    if missing:
        ax.text(
            0.5,
            0.5,
            "Calibrated class probabilities are not available\n"
            + ", ".join(missing),
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return

    levels = list(OUTCOME4_LABELS)
    plot_labels = [OUTCOME4_PLOT_LABELS[level] for level in levels]
    matrix: List[List[float]] = []
    counts: List[int] = []

    for level in levels:
        sub = df[df[outcome_col] == level]
        probabilities = sub[probability_columns].apply(pd.to_numeric, errors="coerce")
        complete = probabilities.dropna(how="all")
        counts.append(int(len(complete)))
        matrix.append([
            float(complete[column].mean()) if len(complete) else 0.0
            for column in probability_columns
        ])

    matrix_array = np.asarray(matrix, dtype=float)
    totals = matrix_array.sum(axis=1, keepdims=True)
    matrix_array = np.divide(
        matrix_array,
        totals,
        out=np.zeros_like(matrix_array),
        where=totals > 0,
    )

    bottom = np.zeros(len(levels), dtype=float)

    # Mild = plain green
    # Moderate = orange
    # Severe = very orange
    # Profound = red
    probability_colors = [
        "#2ca02c",  # Mild
        "#ffb347",  # Moderate
        "#ff7f0e",  # Severe
        "#d62728",  # Profound
    ]

    for index, (label, color) in enumerate(zip(class_labels, probability_colors)):
        ax.bar(
            levels,
            matrix_array[:, index],
            bottom=bottom,
            width=0.68,
            label=label,
            color=color,
        )
        bottom += matrix_array[:, index]

    for level, count in zip(levels, counts):
        ax.annotate(
            f"n={count}",
            (level, 1.0),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(levels)
    ax.set_xticklabels(plot_labels)
    ax.set_ylim(0, 1.20)
    ax.set_ylabel("Mean calibrated class probability")
    ax.set_title("d  Mean predicted severity-class probabilities by prenatal outcome", loc="left", fontweight="bold")
    # Legend sits inside the axes headroom above the bars, which reach 1.0, so
    # that it cannot collide with the panel title.
    ax.legend(
        ncol=4,
        loc="upper center",
        fontsize=8,
        frameon=False,
    )


# Scorer held-out validation panels (attached to the prenatal figure).
# These read PHENOTYPE_SEVERITY_MODEL_V*/held_out_test_predictions.csv and need
# no model object. Panel a is the held-out confusion matrix of the combined
# multichannel score, the only channel for which class probabilities are stored.
# Panel b is expected severity by observed class for both channels, the
# continuous quantity the prenatal analysis actually uses, since the reported
# severity index equals expected severity divided by three.

CLASS_NAMES = ["Mild", "Moderate", "Severe", "Profound"]

# A single blue ramp keeps the confusion-matrix values perceptually ordered.
CONFUSION_CMAP = "Blues"


def load_scorer_heldout(path):
    """Load scorer held-out predictions, or return None if unavailable."""
    if not path or not os.path.exists(path):
        return None
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    required = {"observed", "predicted_class", "phenotype_expected", "combined_expected"}
    if not required.issubset(frame.columns):
        return None
    return frame


def _draw_scorer_confusion(ax, heldout, score_channel="phenotype", title="a  Scorer held-out confusion matrix"):
    observed = heldout["observed"].astype(int).to_numpy()
    if score_channel == "phenotype" and "phenotype_expected" in heldout.columns:
        predicted = np.clip(np.round(heldout["phenotype_expected"].to_numpy()), 0, len(CLASS_NAMES) - 1).astype(int)
    else:
        predicted = heldout["predicted_class"].astype(int).to_numpy()
    n_classes = len(CLASS_NAMES)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true_value, pred_value in zip(observed, predicted):
        if 0 <= true_value < n_classes and 0 <= pred_value < n_classes:
            matrix[true_value, pred_value] += 1
    row_totals = matrix.sum(axis=1, keepdims=True)
    proportions = np.divide(matrix, np.where(row_totals == 0, 1, row_totals), dtype=float)
    image = ax.imshow(proportions, cmap=CONFUSION_CMAP, vmin=0.0, vmax=1.0)
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j, i, "{:,}\n{:.0f}%".format(matrix[i, j], proportions[i, j] * 100),
                ha="center", va="center", fontsize=9,
                color="white" if proportions[i, j] > 0.55 else "#222222",
            )
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(CLASS_NAMES, rotation=20, ha="right")
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(
        ["{}\n(n={})".format(name, int(row_totals[i, 0])) for i, name in enumerate(CLASS_NAMES)]
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Observed class")
    ax.set_title(title, loc="left", fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label("Within-observed-class proportion", fontsize=8)
    bar.ax.tick_params(labelsize=8)


def _draw_scorer_expected_severity(
    ax, heldout, title="b  Scorer held-out expected severity by observed class"
):
    rng = np.random.RandomState(0)
    channels = [
        ("phenotype_expected", "Phenotype channel", "#1f77b4", -0.16, "o"),
        ("combined_expected", "Combined multichannel", "#ff7f0e", 0.16, "x"),
    ]
    for column, label, color, offset, marker in channels:
        for class_index in range(len(CLASS_NAMES)):
            values = heldout.loc[
                heldout["observed"].astype(int) == class_index, column
            ].to_numpy(dtype=float)
            if values.size == 0:
                continue
            jitter = rng.uniform(-0.055, 0.055, size=values.size)
            ax.scatter(
                np.full(values.size, class_index + offset) + jitter, values,
                s=9, alpha=0.35, color=color, marker=marker, linewidths=0.6,
                label=label if class_index == 0 else None,
            )
            ax.hlines(
                float(np.mean(values)), class_index + offset - 0.13,
                class_index + offset + 0.13, color=color, linewidth=2.4, zorder=5,
            )
    ax.plot(
        range(len(CLASS_NAMES)), range(len(CLASS_NAMES)), "--",
        color="gray", linewidth=1.2, label="Ideal class centre",
    )
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_xlim(-0.55, len(CLASS_NAMES) - 0.45)
    ax.set_ylim(-0.12, 3.12)
    ax.set_xlabel("Observed severity class")
    ax.set_ylabel("Expected severity (0-3)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)


def build_4group_figure(
    df,
    mean_tbl,
    corr,
    contrasts,
    out_png,
    score_label: str = "Phenotype-derived severity score",
    scorer_heldout=None,
    score_channel="phenotype",
) -> None:
    """Scorer held-out validation plus the live birth/NICU/TOP/death summary."""
    with_scorer = scorer_heldout is not None and len(scorer_heldout) > 0
    if with_scorer:
        fig, axes = plt.subplots(3, 2, figsize=(15.5, 17.0), constrained_layout=True)
        ax_s1, ax_s2 = axes[0]
        ax_a, ax_b = axes[1]
        ax_c, ax_d = axes[2]
        _draw_scorer_confusion(ax_s1, scorer_heldout, score_channel=score_channel)
        _draw_scorer_expected_severity(ax_s2, scorer_heldout)
        panel_letters = ("c", "d", "e", "f")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.5), constrained_layout=True)
        ax_a, ax_b = axes[0]
        ax_c, ax_d = axes[1]
        panel_letters = ("a", "b", "c", "d")

    levels = list(OUTCOME4_LABELS)
    distribution_labels = [OUTCOME4_PLOT_LABELS[level] for level in levels]
    mean_labels = [OUTCOME4_PLOT_LABELS[level] for level in levels]
    colors = ["#1f77b4", "#17becf", "#ff7f0e", "#d62728"]

    _draw_group_distribution(
        ax_a,
        df,
        "outcome_4level",
        levels,
        distribution_labels,
        colors,
        f"{panel_letters[0]}  Severity score by 4-level prenatal outcome",
        score_label,
    )
    _draw_mean_panel(
        ax_b,
        mean_tbl,
        levels,
        mean_labels,
        corr,
        f"{panel_letters[1]}  Mean score by 4-level outcome",
    )

    roc_colors = ["#1f77b4", "#ff7f0e", "#d62728"]
    for contrast, color in zip(contrasts, roc_colors):
        ax_c.plot(
            contrast["fpr"],
            contrast["tpr"],
            color=color,
            linewidth=2,
            label=(
                f"{contrast['name']}  AUC={contrast['auc']:.3f} "
                f"(n={contrast['n_pos']} vs {contrast['n_neg']})"
            ),
        )
    ax_c.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.2)
    ax_c.set_xlim(-0.02, 1.02)
    ax_c.set_ylim(-0.02, 1.02)
    ax_c.set_xlabel("False-positive rate")
    ax_c.set_ylabel("True-positive rate")
    ax_c.set_title(f"{panel_letters[2]}  ROC for clinical contrasts", loc="left", fontweight="bold")
    if contrasts:
        ax_c.legend(loc="lower right", fontsize=8)

    ax_d._score_channel = score_channel
    _draw_class_probability_panel(ax_d, df, outcome_col="outcome_4level")
    ax_d.set_title(
        f"{panel_letters[3]}  Mean predicted severity-class probabilities by prenatal outcome",
        loc="left", fontweight="bold",
    )

    if with_scorer:
        suptitle = (
            "Deployable phenotype severity scorer and its application to an "
            "independent prenatal cohort"
        )
    else:
        suptitle = f"{score_label}: live birth, NICU, TOP, and death"
    fig.suptitle(suptitle, fontsize=16, fontweight="bold")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    root, extension = os.path.splitext(out_png)
    if extension.lower() != ".pdf":
        fig.savefig(root + ".pdf", bbox_inches="tight")
    plt.close(fig)

def write_paragraph(
    path,
    audit,
    corr3,
    po3,
    mean3,
    contrasts3,
    corr4,
    po4,
    mean4,
    contrasts4,
    score_label: str = "Phenotype-derived severity score",
) -> None:
    counts3 = audit["outcome_counts_analyzed_3group"]
    counts4 = audit["outcome_counts_analyzed_4group"]
    lines = [
        (
            f"The selected analysis measure was the {score_label.lower()} (0-1), "
            f"using scorer channel {audit.get('score_channel', 'unspecified')!r}."
        ),
        (
            f"The input contained {audit['n_start_rows']} rows, which were combined into "
            f"{audit['n_family_profiles']} family-level profiles without discarding "
            f"repeated-family records. All family profiles are retained in the main "
            f"output. {audit['n_missing_or_unmapped_outcome']} families had no mapped "
            f"pregnancy outcome and {audit['n_without_hpo']} had no valid HPO IDs; these "
            f"families remain in the audit output but cannot contribute to the relevant "
            f"outcome analysis or severity model."
        ),
        (
            f"The 3-group analysis included {audit['n_analyzed_3group']} scoreable "
            f"families (Live birth {counts3['Live birth']}, TOP {counts3['TOP']}, "
            f"Death {counts3['Death']}). Severity alignment with the ordered outcome "
            f"(Live birth < TOP < Death) was Spearman rho = "
            f"{corr3['spearman_rho']:+.3f} (bootstrap 95% CI "
            f"{corr3['ci95_low']:+.3f} to {corr3['ci95_high']:+.3f}; permutation "
            f"p = {format_permutation_p(corr3['perm_p'], corr3.get('n_perm_used', 0))}"
            f"), Kendall tau-b = "
            f"{corr3['kendall_tau_b']:+.3f}, and ordinal concordance = "
            f"{corr3['ordinal_concordance']:.3f}."
        ),
        (
            f"The 4-group analysis included {audit['n_analyzed_4group']} scoreable "
            f"families (Live birth without NICU {counts4['Live birth (no NICU)']}, "
            f"NICU/special care {counts4['NICU / special care']}, TOP "
            f"{counts4['TOP']}, Death {counts4['Death']}). Severity alignment with "
            f"the ordered outcome (no-NICU live birth < NICU/special care < TOP < "
            f"Death) was Spearman rho = {corr4['spearman_rho']:+.3f} "
            f"(bootstrap 95% CI {corr4['ci95_low']:+.3f} to "
            f"{corr4['ci95_high']:+.3f}; permutation p = "
            f"{format_permutation_p(corr4['perm_p'], corr4.get('n_perm_used', 0))})."
        ),
    ]
    if po3.get("available"):
        lines.append(
            f"In the 3-group proportional-odds model, each +1 SD of severity was "
            f"associated with an odds ratio of {po3['or_per_sd']:.2f} for a more "
            f"severe outcome (95% CI {po3['ci95'][0]:.2f} to "
            f"{po3['ci95'][1]:.2f}; p = {po3['p_value']:.3g})."
        )
    if po4.get("available"):
        lines.append(
            f"In the 4-group proportional-odds model, each +1 SD of severity was "
            f"associated with an odds ratio of {po4['or_per_sd']:.2f} for a more "
            f"severe outcome (95% CI {po4['ci95'][0]:.2f} to "
            f"{po4['ci95'][1]:.2f}; p = {po4['p_value']:.3g})."
        )
    if contrasts3:
        auc_text = "; ".join(
            f"{item['name']} AUC {item['auc']:.3f}" for item in contrasts3
        )
        lines.append(f"Discrimination for 3-group clinical contrasts: {auc_text}.")
    if contrasts4:
        auc_text = "; ".join(
            f"{item['name']} AUC {item['auc']:.3f}" for item in contrasts4
        )
        lines.append(f"Discrimination for 4-group clinical contrasts: {auc_text}.")
    lines.append(
        f"The {score_label.lower()} is a disease-severity score, not a calibrated "
        "probability of a pregnancy outcome. Families with missing outcomes are "
        "retained and scored when possible, but are not assigned to an outcome group."
    )
    with open(path, "w") as handle:
        handle.write("\n\n".join(lines) + "\n")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="prenatal_cohort_merged.csv")
    ap.add_argument("--model", default="results/scorer/model.pkl")
    ap.add_argument(
        "--scorer-code",
        default="src/phenotype_scorer.py",
        help=(
            "Python file used to create the saved scorer model. The same file "
            "should normally be supplied for v3 artifacts."
        ),
    )
    ap.add_argument(
        "--score-channel",
        choices=sorted(SCORE_CHANNEL_LABELS),
        default="phenotype",
        help=(
            "Score used for prenatal-outcome analyses. 'combined' uses the v3.2 "
            "multichannel score; 'phenotype' preserves the phenotype-derived "
            "analysis; 'headline' follows the model artifact's headline score."
        ),
    )
    ap.add_argument(
        "--allow-low-information",
        action="store_true",
        help=(
            "Allow the scorer's intercept-only/low-information override. By "
            "default model abstentions remain excluded from outcome analyses."
        ),
    )
    ap.add_argument(
        "--scorer-heldout",
        default="results/scorer/held_out_test_predictions.csv",
        help=(
            "Held-out predictions written by the phenotype scorer. When present, "
            "two scorer-validation panels are prepended to the prenatal figure. "
            "Pass an empty string for the four-panel prenatal figure only."
        ),
    )
    ap.add_argument("--output", default="results/prenatal")
    ap.add_argument("--family-col", default="family_id")
    ap.add_argument("--outcome-col", default="pregnancy_outcome")
    ap.add_argument("--hpo-col", default="hpo_ids")
    ap.add_argument(
        "--absent-hpo-col",
        default=None,
        help="Optional column containing explicitly absent HPO terms.",
    )
    ap.add_argument("--study-col", default="study")
    ap.add_argument(
        "--exclude-studies",
        nargs="*",
        default=[],
        help="Optional studies to drop. By default no study is excluded.",
    )
    ap.add_argument(
        "--livebirth-only-studies",
        nargs="*",
        default=[],
        help=(
            "Optional studies to restrict to live births. This is disabled by "
            "default so the full cohort is used."
        ),
    )
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=2000)
    args = ap.parse_args()
    print(f"[version] {SCRIPT_VERSION}")
    os.makedirs(args.output, exist_ok=True)

    family_profiles, audit = load_cohort(
        args.cohort,
        args.exclude_studies,
        args.livebirth_only_studies,
        family_col=args.family_col,
        outcome_col=args.outcome_col,
        hpo_col=args.hpo_col,
        absent_hpo_col=args.absent_hpo_col,
        study_col=args.study_col,
    )
    print(
        f"[cohort] rows={audit['n_start_rows']} families={audit['n_family_profiles']} "
        f"3-group={audit['outcome_counts']} 4-group={audit['outcome_4group_counts']} "
        f"missing/unmapped outcomes={audit['n_missing_or_unmapped_outcome']}"
    )

    all_profiles, model_src = score_families(
        family_profiles,
        args.model,
        args.scorer_code,
        score_channel=args.score_channel,
        allow_low_information=args.allow_low_information,
    )
    score_label = _score_channel_label(args.score_channel)
    audit["score_channel"] = args.score_channel
    audit["score_label"] = score_label
    all_profiles["included_3group_analysis"] = (
        all_profiles["score_available"]
        & all_profiles["outcome_3level"].notna()
    )
    all_profiles["included_4group_analysis"] = (
        all_profiles["score_available"]
        & all_profiles["outcome_4level"].notna()
    )

    analysis3 = all_profiles[
        all_profiles["included_3group_analysis"]
    ].copy().reset_index(drop=True)
    analysis4 = all_profiles[
        all_profiles["included_4group_analysis"]
    ].copy().reset_index(drop=True)
    scoring_excluded = all_profiles[
        ~all_profiles["score_available"]
    ].copy().reset_index(drop=True)
    missing_outcome = all_profiles[
        ~all_profiles["has_mapped_outcome_3level"]
    ].copy().reset_index(drop=True)

    audit["n_scoreable"] = int(all_profiles["score_available"].sum())
    audit["n_not_scoreable"] = int((~all_profiles["score_available"]).sum())
    audit["n_analyzed_3group"] = int(len(analysis3))
    audit["n_analyzed_4group"] = int(len(analysis4))
    audit["scoring_status_counts"] = {
        str(key): int(value)
        for key, value in all_profiles["score_status"].value_counts(
            dropna=False
        ).items()
    }
    audit["outcome_counts_analyzed_3group"] = {
        OUTCOME_LABELS[level]: int((analysis3["outcome_3level"] == level).sum())
        for level in OUTCOME_LABELS
    }
    audit["outcome_counts_analyzed_4group"] = {
        OUTCOME4_LABELS[level]: int((analysis4["outcome_4level"] == level).sum())
        for level in OUTCOME4_LABELS
    }

    print(
        f"[score] scoreable {audit['n_scoreable']}/{len(all_profiles)} families "
        f"with model '{model_src}' using channel '{args.score_channel}'; "
        f"3-group n={len(analysis3)}, 4-group n={len(analysis4)}; "
        f"statuses={audit['scoring_status_counts']}"
    )
    if len(analysis3) < 3 or analysis3["outcome_3level"].nunique() < 2:
        raise ValueError(
            "Not enough scoreable families with mapped 3-level outcomes for analysis."
        )
    if len(analysis4) < 4 or analysis4["outcome_4level"].nunique() < 2:
        raise ValueError(
            "Not enough scoreable families with mapped 4-level outcomes for analysis."
        )

    corr3 = correlation_analysis(
        analysis3,
        args.n_boot,
        args.n_perm,
        outcome_col="outcome_3level",
    )
    po3 = proportional_odds(analysis3, outcome_col="outcome_3level")
    mean3 = mean_ci_by_outcome(
        analysis3, "outcome_3level", OUTCOME_LABELS
    )
    contrasts3 = clinical_contrasts_3group(analysis3)

    corr4 = correlation_analysis(
        analysis4,
        args.n_boot,
        args.n_perm,
        seed=1,
        outcome_col="outcome_4level",
    )
    po4 = proportional_odds(analysis4, outcome_col="outcome_4level")
    mean4 = mean_ci_by_outcome(
        analysis4, "outcome_4level", OUTCOME4_LABELS
    )
    contrasts4 = clinical_contrasts_4group(analysis4)
    adjacent4 = adjacent_level_separation(
        analysis4,
        outcome_col="outcome_4level",
        labels=OUTCOME4_LABELS,
        n_boot=args.n_boot,
    )
    adjacent3 = adjacent_level_separation(
        analysis3,
        outcome_col="outcome_3level",
        labels=OUTCOME_LABELS,
        n_boot=args.n_boot,
    )
    source4 = source_sensitivity(analysis4, outcome_col="outcome_4level", study_col=args.study_col)
    # Baseline without training: does the score add information beyond the
    # number of recorded HPO terms (anomaly burden)?
    termcount4 = term_count_baseline(
        analysis4,
        outcome_col="outcome_4level",
        n_boot=args.n_boot,
        n_perm=args.n_perm,
        seed=2,
        termination_level=2,
    )
    print("[term count] rank comparison (4-group):")
    print(termcount4["rank"].to_string(index=False))

    def exportable(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.drop(
            columns=["hpo_list", "absent_hpo_list"],
            errors="ignore",
        )


    exportable(all_profiles).to_csv(
        os.path.join(args.output, "severity_by_family.csv"),
        index=False,
    )
    exportable(analysis3).to_csv(
        os.path.join(args.output, "severity_analysis_3group.csv"),
        index=False,
    )
    exportable(analysis4).to_csv(
        os.path.join(args.output, "severity_analysis_4group.csv"),
        index=False,
    )
    if len(scoring_excluded):
        exportable(scoring_excluded).to_csv(
            os.path.join(args.output, "severity_scoring_exclusions.csv"),
            index=False,
        )
    if len(missing_outcome):
        exportable(missing_outcome).to_csv(
            os.path.join(args.output, "families_missing_or_unmapped_outcome.csv"),
            index=False,
        )

    aggregation_audit = all_profiles[
        (all_profiles["source_row_count"] > 1)
        | all_profiles["outcome_conflict"]
    ].copy()
    if len(aggregation_audit):
        exportable(aggregation_audit).to_csv(
            os.path.join(args.output, "family_aggregation_audit.csv"),
            index=False,
        )

    mean3.to_csv(
        os.path.join(args.output, "mean_by_outcome_3group.csv"),
        index=False,
    )
    mean3.to_csv(
        os.path.join(args.output, "mean_by_outcome.csv"),
        index=False,
    )
    mean4.to_csv(
        os.path.join(args.output, "mean_by_outcome_4group.csv"),
        index=False,
    )
    roc_table = pd.DataFrame([
        {
            "contrast": item["name"],
            "auc": item["auc"],
            "n_pos": item["n_pos"],
            "n_neg": item["n_neg"],
        }
        for item in contrasts3
    ])
    roc_table.to_csv(
        os.path.join(args.output, "roc_contrasts_3group.csv"),
        index=False,
    )
    roc_table.to_csv(
        os.path.join(args.output, "roc_contrasts.csv"),
        index=False,
    )
    pd.DataFrame([
        {
            "contrast": item["name"],
            "auc": item["auc"],
            "n_pos": item["n_pos"],
            "n_neg": item["n_neg"],
        }
        for item in contrasts4
    ]).to_csv(
        os.path.join(args.output, "roc_contrasts_4group.csv"),
        index=False,
    )
    adjacent4.to_csv(
        os.path.join(args.output, "adjacent_level_separation_4group.csv"),
        index=False,
    )
    adjacent3.to_csv(
        os.path.join(args.output, "adjacent_level_separation_3group.csv"),
        index=False,
    )
    source4["composition"].to_csv(
        os.path.join(args.output, "source_composition_4group.csv"),
        index=False,
    )
    source4["rank"].to_csv(
        os.path.join(args.output, "source_rank_sensitivity_4group.csv"),
        index=False,
    )
    source4["odds"].to_csv(
        os.path.join(args.output, "source_adjusted_proportional_odds_4group.csv"),
        index=False,
    )
    termcount4["rank"].to_csv(
        os.path.join(args.output, "term_count_baseline_rank_4group.csv"),
        index=False,
    )
    termcount4["odds"].to_csv(
        os.path.join(args.output, "term_count_baseline_odds_4group.csv"),
        index=False,
    )

    with open(os.path.join(args.output, "statistics.json"), "w") as handle:
        json.dump(
            {
                "audit": audit,
                "correlation_3group": corr3,
                "proportional_odds_3group": po3,
                "correlation_4group": corr4,
                "proportional_odds_4group": po4,
                "roc_contrasts_3group": [
                    {key: item[key] for key in ("name", "auc", "n_pos", "n_neg")}
                    for item in contrasts3
                ],
                "roc_contrasts_4group": [
                    {key: item[key] for key in ("name", "auc", "n_pos", "n_neg")}
                    for item in contrasts4
                ],
                "adjacent_level_separation_3group":
                    adjacent3.to_dict(orient="records"),
                "adjacent_level_separation_4group":
                    adjacent4.to_dict(orient="records"),
                "source_composition_4group":
                    source4["composition"].to_dict(orient="records"),
                "source_rank_sensitivity_4group":
                    source4["rank"].to_dict(orient="records"),
                "source_adjusted_proportional_odds_4group":
                    source4["odds"].to_dict(orient="records"),
                "term_count_baseline_rank_4group":
                    termcount4["rank"].to_dict(orient="records"),
                "term_count_baseline_odds_4group":
                    termcount4["odds"].to_dict(orient="records"),
                "model": model_src,
                "score_channel": args.score_channel,
                "score_label": score_label,
            },
            handle,
            indent=2,
        )

    build_3group_figure(
        analysis3,
        mean3,
        corr3,
        contrasts3,
        os.path.join(args.output, "prenatal_severity_figure.png"),
        score_label=score_label,
    )
    four_group_roc_figure = os.path.join(
        args.output, "prenatal_severity_figure_4group_WITH_ROC.png"
    )
    scorer_heldout = load_scorer_heldout(args.scorer_heldout)
    if scorer_heldout is not None:
        print(
            f"[figure] scorer validation panels from {args.scorer_heldout} "
            f"(n={len(scorer_heldout)})"
        )
    elif args.scorer_heldout:
        print(
            f"[figure] scorer held-out predictions not usable at "
            f"'{args.scorer_heldout}'; writing the prenatal panels only"
        )
    build_4group_figure(
        analysis4,
        mean4,
        corr4,
        contrasts4,
        four_group_roc_figure,
        score_label=score_label,
        scorer_heldout=scorer_heldout,
        score_channel=args.score_channel,
    )
    # Both filenames contain the same three-panel figure for downstream compatibility.
    import shutil
    shutil.copy2(
        four_group_roc_figure,
        os.path.join(args.output, "prenatal_severity_figure_4group.png"),
    )
    print(
        f"[figure] wrote 3-panel 4-group figure with ROC: {four_group_roc_figure}"
    )
    write_paragraph(
        os.path.join(args.output, "results_paragraph.txt"),
        audit,
        corr3,
        po3,
        mean3,
        contrasts3,
        corr4,
        po4,
        mean4,
        contrasts4,
        score_label=score_label,
    )

    print(
        f"[done] 3-group rho={corr3['spearman_rho']:+.3f} "
        f"CI=({corr3['ci95_low']:+.3f},{corr3['ci95_high']:+.3f}) "
        f"p={format_permutation_p(corr3['perm_p'], corr3.get('n_perm_used', 0), unicode_times=False)}"
    )
    print(
        f"[done] 4-group rho={corr4['spearman_rho']:+.3f} "
        f"CI=({corr4['ci95_low']:+.3f},{corr4['ci95_high']:+.3f}) "
        f"p={format_permutation_p(corr4['perm_p'], corr4.get('n_perm_used', 0), unicode_times=False)}"
    )
    print(f"[done] artifacts in {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()
