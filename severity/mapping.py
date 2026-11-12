"""HPO-verified gene-disease mapping (Data and severity labels).

Every gene-disease pair is checked against the HPO gene-to-disease annotations
(release 2025-11-24) using the disease's OMIM or Orphanet identifier. Pairs that HPO
does not support are removed before any feature is built; this removes 59 of the
6,684 candidate pairs. ``gene_disease_mapping_HPO_rebuilt.csv`` records the decision
for each pair (``status`` and ``use_in_rebuilt_corpus``).
"""

import os
from typing import Any, Callable, Optional, Set, Tuple

import pandas as pd


def _normalise(value: Any) -> str:
    return str(value).strip().upper()


def load_allowed_pairs(mapping_path: str) -> Tuple[Set[Tuple[str, str]], dict]:
    """Return the (gene, MONDO) pairs kept in the mapping, with a short summary."""
    mapping = pd.read_csv(mapping_path, low_memory=False)
    for column in ("gene_symbol", "mondo_id", "status", "use_in_rebuilt_corpus"):
        if column not in mapping.columns:
            raise ValueError(f"{mapping_path} has no column {column!r}.")
    keep = mapping["use_in_rebuilt_corpus"].astype(str).str.lower().isin(("true", "1", "yes"))
    keep &= mapping["status"].eq("KEPT")
    selected = mapping.loc[keep]
    allowed = {(_normalise(g), _normalise(m)) for g, m in zip(selected.gene_symbol, selected.mondo_id)}
    summary = {"mapping_path": mapping_path, "allowed_pairs": len(allowed)}
    return allowed, summary


def apply_rebuilt_mapping(
    df: pd.DataFrame,
    gene_col: str,
    mondo_col: str,
    mapping_path: str,
    log: Optional[Callable[..., None]] = None,
    audit_dir: Optional[str] = None,
    audit_name: str = "gene_disease_mapping_filter_audit.csv",
) -> pd.DataFrame:
    """Keep only the rows of ``df`` whose gene-disease pair is supported by HPO."""

    def emit(message: str) -> None:
        if log is not None:
            log(message)

    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Gene-disease mapping not found: {mapping_path}")
    allowed, summary = load_allowed_pairs(mapping_path)
    key = pd.Series(
        list(zip(df[gene_col].map(_normalise), df[mondo_col].map(_normalise))),
        index=df.index,
    )
    mask = key.isin(allowed)
    filtered = df.loc[mask].copy()
    before_pairs, after_pairs = int(key.nunique()), int(key.loc[mask].nunique())
    emit(f"  HPO-verified mapping: {summary['mapping_path']} ({summary['allowed_pairs']} pairs)")
    emit(f"  Gene-disease pairs {before_pairs} -> {after_pairs}; source rows {len(df)} -> {len(filtered)}")
    if after_pairs == 0:
        raise ValueError(
            "The gene-disease mapping removed every pair; check gene symbols and MONDO identifiers."
        )
    if audit_dir:
        os.makedirs(audit_dir, exist_ok=True)
        removed = df.loc[~mask, [gene_col, mondo_col]].drop_duplicates()
        removed = removed.rename(columns={gene_col: "gene_symbol", mondo_col: "mondo_id"})
        removed["removed_reason"] = "not verified by HPO gene-to-disease mapping"
        removed.to_csv(os.path.join(audit_dir, audit_name), index=False)
    return filtered
