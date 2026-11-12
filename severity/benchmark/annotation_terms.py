"""Gene-annotation features (Feature construction).

Pathways, Gene Ontology biological processes and Gene Ontology molecular functions are
open-ended term lists. For each fold, the 20 most frequent terms per category are counted in
the training pairs only and turned into binary features (up to 60), which then go through the
same preprocessing and feature selection as every other feature. Annotations are combined to
one profile per gene, so disease identity cannot change a gene-level feature.
"""

from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .ontology import safe_feature_name
from .runtime import DataContainer

# Annotation columns of the gene annotation table that are used.
_TEXT_STRUCT_COLS = [
    column
    for column in [
        "Pathways",
        "Biological_Processes",
        "Molecular_Activities",
        "Interacting_Genes",
        "Expression_Sites",
    ]
    if column not in Config.TEXT_EXCLUDED_SOURCE_COLUMNS
]

# Feature sets that receive annotation-term features. "gnn" is the name of the gene
# annotation table (final_gnn_dataset.csv); no graph neural network is used.
SCENARIO_TEXT_SOURCES: Dict[str, Tuple[str, ...]] = {
    "S1_Mechanistic": ("gnn",),
    "S1b_Disease_Context": ("gnn",),
    "S2_Mech_Pheno": ("gnn",),
    "S3_Phenotype": (),
}


def _tokenise_pipe_value(value: Any) -> set:
    """Split a pipe-separated annotation string; placeholders such as "Unknown" count as no annotation."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    placeholders = set(Config.TEXT_PLACEHOLDER_TOKENS)
    return {
        item.strip()
        for item in str(value).split("|")
        if item.strip() and item.strip().lower() not in placeholders
    }


def _gene_key(gene: Any) -> str:
    return str(gene).strip().upper()


def _collapse_gnn_to_gene_level(source: pd.DataFrame) -> pd.DataFrame:
    """One union-of-annotations profile per gene."""
    frame = source.copy()
    frame["gene_symbol"] = frame["gene_symbol"].map(_gene_key)
    text_columns = [column for column in _TEXT_STRUCT_COLS if column in frame.columns]
    rows: List[Dict[str, Any]] = []
    for gene, group in frame.groupby("gene_symbol", sort=True):
        row: Dict[str, Any] = {"gene_symbol": gene}
        for column in text_columns:
            tokens: set = set()
            for value in group[column].dropna():
                tokens.update(_tokenise_pipe_value(value))
            row[column] = "|".join(sorted(tokens))
        rows.append(row)
    return pd.DataFrame(rows)


def _load_text_token_maps() -> Dict[str, Dict[str, Dict[str, set]]]:
    """Gene -> set of annotation terms, for each annotation column."""
    source = pd.read_csv(Config.GNN_DATA, low_memory=False).rename(columns={"Gene Name": "gene_symbol"})
    source = _collapse_gnn_to_gene_level(source)
    output: Dict[str, Dict[str, set]] = {}
    for column in _TEXT_STRUCT_COLS:
        if column not in source.columns:
            continue
        mapping: Dict[str, set] = {}
        for gene, values in source.groupby("gene_symbol")[column]:
            tokens: set = set()
            for value in values.dropna():
                tokens.update(_tokenise_pipe_value(value))
            mapping[_gene_key(gene)] = tokens
        output[column] = mapping
    return {"gnn": output}


def _learn_text_vocabulary(
    container: DataContainer, train_idx: np.ndarray
) -> Dict[str, Dict[str, List[str]]]:
    """The 20 most frequent terms per annotation column among the training pairs (ties broken by name)."""
    genes = container.full_df.iloc[np.asarray(train_idx, dtype=int)]["gene_symbol"].astype(str).to_numpy()
    vocabulary: Dict[str, Dict[str, List[str]]] = {}
    for source_label, column_maps in container.text_token_maps.items():
        vocabulary[source_label] = {}
        for column, token_map in column_maps.items():
            counter: Counter = Counter()
            for gene in genes:
                counter.update(token_map.get(_gene_key(gene), set()))
            ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
            vocabulary[source_label][column] = [token for token, _ in ranked[: Config.TEXT_VOCAB_TOP_N]]
    return vocabulary


def _add_training_text_features(
    container: DataContainer,
    indices: np.ndarray,
    vocabulary: Dict[str, Dict[str, List[str]]],
    family_lookup: Dict[str, str],
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    """Add one binary column per vocabulary term (1 = the gene carries the term)."""
    frame = container.full_df.iloc[indices].copy()
    genes = frame["gene_symbol"].astype(str).values
    generated: List[str] = []
    new_columns: Dict[str, np.ndarray] = {}
    for source_label, columns in vocabulary.items():
        for source_column, tokens in columns.items():
            token_map = container.text_token_maps.get(source_label, {}).get(source_column, {})
            gene_tokens = [token_map.get(_gene_key(gene)) for gene in genes]
            for token in tokens:
                name = safe_feature_name(f"feat_{source_label}_{source_column.lower()}", token, max_len=30)
                if name in frame.columns or name in new_columns:
                    continue
                new_columns[name] = np.fromiter(
                    (1.0 if values and token in values else 0.0 for values in gene_tokens),
                    dtype=float,
                    count=len(gene_tokens),
                )
                generated.append(name)
                family_lookup[name] = f"{source_label}_text"
    if new_columns:
        frame = pd.concat([frame, pd.DataFrame(new_columns, index=frame.index)], axis=1).copy()
    return frame, generated, family_lookup
