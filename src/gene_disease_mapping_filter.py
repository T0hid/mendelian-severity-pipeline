"""
Shared filter that restricts a gene-disease corpus to the HPO-verified mapping.

Background
----------
The original corpus attached whole gene panels to single disease terms. For
example MONDO:0007542 (Camurati-Engelmann disease) carried 571 genes where HPO
asserts only TGFB1, and MONDO:0014343 (Desbuquois dysplasia 2) carried the
identical 571-gene set. Those duplicated rows share one feature vector and one
label, so they are memorised rather than predicted.

gene_disease_mapping_HPO_rebuilt.csv checks every gene-disease edge against the
HPO curated gene-to-disease file (release 2025-11-24), joined on the disease's
OMIM/ORPHA identifier. This module applies that mapping at the single point
where each pipeline reads its primary corpus.

Behaviour
---------
If the mapping file is absent the corpus is returned unchanged and a warning is
logged, so an existing checkout still runs.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Optional, Set, Tuple

import pandas as pd

DEFAULT_MAPPING_FILENAME = "gene_disease_mapping_HPO_rebuilt.csv"

# Pairs HPO asserts that were absent from the original corpus. Enabling this
# requires synthesising source rows from the disease's existing rows, so it is
# off by default. Turn it on only after inspecting the ADD rows of the mapping.
INCLUDE_HPO_ADDED_PAIRS = False


def _normalise_gene(value: Any) -> str:
    return str(value).strip().upper()


def _normalise_mondo(value: Any) -> str:
    return str(value).strip().upper()


def resolve_mapping_path(explicit: Optional[str], *search_dirs: str) -> Optional[str]:
    """Return the first readable mapping file, or None."""
    candidates = []
    if explicit:
        candidates.append(explicit)
    for directory in search_dirs:
        if directory:
            candidates.append(os.path.join(directory, DEFAULT_MAPPING_FILENAME))
            candidates.append(os.path.join(directory, "..", DEFAULT_MAPPING_FILENAME))
    for path in candidates:
        if path and os.path.exists(path):
            return os.path.abspath(path)
    return None


def load_allowed_pairs(mapping_path: str,
                       include_added: bool = INCLUDE_HPO_ADDED_PAIRS
                       ) -> Tuple[Set[Tuple[str, str]], dict]:
    """Read the mapping and return the permitted (gene, mondo) keys."""
    mapping = pd.read_csv(mapping_path, low_memory=False)
    for column in ("gene_symbol", "mondo_id", "status", "use_in_rebuilt_corpus"):
        if column not in mapping.columns:
            raise ValueError(
                f"{mapping_path} is missing the required column {column!r}. "
                "Regenerate it with rebuild_gene_disease_from_hpo.py."
            )
    keep = mapping["use_in_rebuilt_corpus"].astype(str).str.lower().isin(("true", "1", "yes"))
    if not include_added:
        keep &= mapping["status"].eq("KEPT")
    selected = mapping.loc[keep]
    allowed = {
        (_normalise_gene(g), _normalise_mondo(m))
        for g, m in zip(selected.gene_symbol, selected.mondo_id)
    }
    summary = {
        "mapping_path": mapping_path,
        "mapping_rows": int(len(mapping)),
        "include_added_pairs": bool(include_added),
        "allowed_pairs": len(allowed),
        "status_counts": mapping.status.value_counts().to_dict(),
    }
    return allowed, summary


def apply_rebuilt_mapping(df: pd.DataFrame,
                          gene_col: str,
                          mondo_col: str,
                          mapping_path: Optional[str],
                          log: Optional[Callable[..., None]] = None,
                          audit_dir: Optional[str] = None,
                          audit_name: str = "gene_disease_mapping_filter_audit.csv",
                          include_added: bool = INCLUDE_HPO_ADDED_PAIRS
                          ) -> pd.DataFrame:
    """Restrict df to gene-disease pairs HPO verifies. Returns the filtered frame."""
    def emit(message: str, level: str = "INFO") -> None:
        if log is None:
            print(f"[{level}] {message}")
        else:
            try:
                log(message, level=level)
            except TypeError:
                log(message)

    if not mapping_path or not os.path.exists(mapping_path):
        emit(
            "HPO-rebuilt gene-disease mapping not found; corpus used unfiltered. "
            "The panel-collapse duplication described in gene_disease_mapping_filter "
            "is therefore still present.",
            level="WARNING",
        )
        return df

    allowed, summary = load_allowed_pairs(mapping_path, include_added=include_added)
    key = pd.Series(
        list(zip(df[gene_col].map(_normalise_gene), df[mondo_col].map(_normalise_mondo))),
        index=df.index,
    )
    mask = key.isin(allowed)

    before_rows = int(len(df))
    before_pairs = int(key.nunique())
    filtered = df.loc[mask].copy()
    after_rows = int(len(filtered))
    after_key = key.loc[mask]
    after_pairs = int(after_key.nunique())

    emit(f"  HPO-rebuilt mapping: {summary['mapping_path']}")
    emit(f"  Allowed gene-disease pairs in mapping: {summary['allowed_pairs']}"
         f" (include_added={summary['include_added_pairs']})")
    emit(f"  Corpus pairs {before_pairs} -> {after_pairs}"
         f" ({before_pairs - after_pairs} removed as panel-collapse artefacts)")
    emit(f"  Corpus source rows {before_rows} -> {after_rows}")

    if after_pairs == 0:
        raise ValueError(
            "The HPO-rebuilt mapping removed every gene-disease pair. Check that the "
            "gene symbols and MONDO identifiers in the corpus match the mapping file."
        )

    if audit_dir:
        os.makedirs(audit_dir, exist_ok=True)
        removed = df.loc[~mask, [gene_col, mondo_col]].drop_duplicates()
        removed = removed.rename(columns={gene_col: "gene_symbol", mondo_col: "mondo_id"})
        removed["removed_reason"] = "not verified by HPO gene-to-disease mapping"
        audit_path = os.path.join(audit_dir, audit_name)
        removed.to_csv(audit_path, index=False)
        emit(f"  Wrote removed-pair audit: {audit_path}")

    return filtered
