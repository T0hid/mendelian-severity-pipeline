"""Prespecified rules that keep severity-stating HPO terms out of S2 and S3
(Feature selection and leakage controls; Supplementary Table 13).

Rule 1 removes the 17 listed HPO identifiers. Rule 2 removes any HPO feature whose term
name contains mild, moderate, severe or profound as a separate word.
"""

import re
from typing import List, Optional, Tuple

from .runtime import HPO_COLUMN_TERM_NAMES

# Rule 1. Names are from hp.obo, release 2025-11-24.
SEVERITY_HPO_IDS = {
    "HP:0001522",  # Death in infancy
    "HP:0003811",  # Neonatal death
    "HP:0011421",  # Death in adolescence
    "HP:0011420",  # Age of death
    "HP:0003819",  # Death in childhood
    "HP:0034241",  # Prenatal death
    "HP:0002187",  # Profound intellectual disability
    "HP:0011344",  # Severe global developmental delay
    "HP:0011342",  # Mild global developmental delay
    "HP:0011343",  # Moderate global developmental delay
    "HP:0001263",  # Global developmental delay
    "HP:0006887",  # Intellectual disability, progressive (obsolete)
    "HP:0010864",  # Severe intellectual disability
    "HP:0002342",  # Moderate intellectual disability
    "HP:0001256",  # Mild intellectual disability
    "HP:0001249",  # Intellectual disability
    "HP:0003828",  # Variable expressivity
}
# Rule 2.
SEVERITY_HPO_KEYWORDS = ["profound", "severe", "mild", "moderate"]


def _has_severity_keyword(text: str) -> bool:
    words = str(text).lower().replace("_", " ")
    return any(
        re.search(f"(^|[^a-z]){re.escape(keyword)}([^a-z]|$)", words) for keyword in SEVERITY_HPO_KEYWORDS
    )


def is_severity_hpo_column(colname: str) -> bool:
    """True if a feature column falls under rule 1 or rule 2."""
    if not isinstance(colname, str):
        return False
    match = re.search("(HP[_:]\\d{7})", colname)
    if match and match.group(1).replace("_", ":") in SEVERITY_HPO_IDS:
        return True
    if _has_severity_keyword(colname):
        return True
    term_name = HPO_COLUMN_TERM_NAMES.get(colname)
    return bool(term_name and _has_severity_keyword(term_name))


def filter_leaky_columns_for_phenotype_scenarios(
    cols: List[str], scenario_name: str, drop_severity_hpo_terms: Optional[bool] = None
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Apply both rules to the S2 and S3 candidates. ``drop_severity_hpo_terms=False`` is the
    validation-only check in which the removed terms are returned to the candidate pool."""
    if not any(s in scenario_name for s in ("S2", "S3")):
        return cols, []
    drop_severity = True if drop_severity_hpo_terms is None else bool(drop_severity_hpo_terms)
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
        kept.append(column)
    return kept, dropped
