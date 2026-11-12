"""Data partitioning and nested cross-validation.

The test set (20% of pairs) is chosen first and the validation set (15% of all pairs) second,
from the remaining genes. At each stage 1,000 random gene-disjoint candidates are generated
and the one with the lowest equation (1) score is kept. Only pair and class counts are used.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import RANDOM_SEED, Config
from .runtime import DataContainer, _fold_artifacts, logger


def _distribution(values: np.ndarray) -> np.ndarray:
    counts = np.bincount(values.astype(int), minlength=Config.NUM_CLASSES).astype(float)
    return counts / max(counts.sum(), 1.0)


def _split_objective(
    selected_idx: np.ndarray, all_idx: np.ndarray, y: np.ndarray, target_fraction: float
) -> float:
    """Equation (1): distance of a candidate held-out set S from its target size and class mix."""
    if len(selected_idx) == 0 or len(selected_idx) == len(all_idx):
        return float("inf")
    fraction_error = abs(len(selected_idx) / len(all_idx) - target_fraction)
    distribution_error = float(np.abs(_distribution(y[selected_idx]) - _distribution(y[all_idx])).sum())
    class_counts = np.bincount(y[selected_idx], minlength=Config.NUM_CLASSES)
    missing_penalty = 10.0 * float(np.sum(class_counts < Config.SPLIT_MIN_CLASS_COUNT))
    return 4.0 * fraction_error + distribution_error + missing_penalty


def _optimised_group_holdout(
    indices: np.ndarray, y: np.ndarray, group_values: np.ndarray, target_fraction: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw gene-disjoint candidate held-out sets and keep the one with the lowest equation (1) score.

    Returns (retained indices, held-out indices).
    """
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
    return (retained, best_selected)


def make_train_test_split(container: DataContainer, seed: int = RANDOM_SEED):
    """Stage 1: hold out 20% of pairs as the test set, keeping all pairs of a gene together."""
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
        f"  Development={len(development_idx)} pairs/{len(train_genes)} genes; test={len(test_idx)} pairs/{len(test_genes)} genes."
    )


def _split_development_train_validation(
    container: DataContainer, development_idx: np.ndarray, grouping: np.ndarray, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Stage 2: validation set of 15% of all pairs (0.1875 of the development pairs)."""
    relative_fraction = Config.VAL_SIZE / max(1.0 - Config.TEST_SIZE, 1e-12)
    train_idx, val_idx = _optimised_group_holdout(
        development_idx, container.y, grouping, relative_fraction, seed + 17
    )
    return (train_idx, val_idx)


def _nested_splits(
    train_idx: np.ndarray, y: np.ndarray, grouping: np.ndarray, seed: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Three folds of the training data, grouped by gene and stratified by severity."""
    local_y = y[train_idx]
    local_groups = grouping[train_idx]
    n_groups = len(np.unique(local_groups))
    n_splits = max(2, min(Config.NESTED_CV_FOLDS, n_groups))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for tr_local, va_local in splitter.split(np.zeros(len(train_idx)), local_y, local_groups):
        splits.append((train_idx[tr_local], train_idx[va_local]))
    return splits


def _gene_disease_component_groups(container: DataContainer) -> np.ndarray:
    """Groups of pairs connected through a shared gene or disease (gene-and-disease-disjoint evaluation)."""
    metadata = container.metadata.reset_index(drop=True)
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = (find(a), find(b))
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


def write_split_summary(
    container: DataContainer,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    out_path: Optional[str] = None,
    include_test_labels: bool = False,
) -> pd.DataFrame:
    """Pair, gene and class counts per partition; test class counts stay hidden until evaluation."""
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
                (
                    f"c{severity_class}={int(row[f'n_class_{severity_class}'])}"
                    for severity_class in range(Config.NUM_CLASSES)
                )
            )
        else:
            per_class = "class counts sealed"
        logger.log(
            f"  {row['partition']:6s} pairs={int(row['n_pairs']):5d} genes={int(row['n_genes']):5d} [{per_class}]"
        )
    return dataframe


def _write_split_identifiers(
    container: DataContainer,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    prefix: str = "primary",
) -> None:
    """Gene and MONDO identifiers of every pair in each partition."""
    metadata = container.metadata.reset_index(drop=True)
    rows: List[Dict[str, Any]] = []
    for partition, idx in (("train", train_idx), ("validation", val_idx), ("test", test_idx)):
        for row_idx in idx:
            rows.append(
                {
                    "analysis": prefix,
                    "partition": partition,
                    "row_index": int(row_idx),
                    "gene_symbol": metadata.iloc[row_idx].get("gene_symbol", ""),
                    "mondo_id": metadata.iloc[row_idx].get("mondo_id", ""),
                }
            )
    pd.DataFrame(rows).to_csv(
        os.path.join(Config.OUTPUT_DIR, "results", f"split_identifiers_{prefix}.csv"), index=False
    )


def _primary_partition_indices(container: DataContainer) -> Dict[str, np.ndarray]:
    """Row indices of the primary training, validation and test sets."""
    for artifact in _fold_artifacts.values():
        if all((key in artifact for key in ("train_idx", "val_idx", "test_idx"))):
            return {
                "train": np.asarray(artifact["train_idx"], dtype=int),
                "validation": np.asarray(artifact["val_idx"], dtype=int),
                "test": np.asarray(artifact["test_idx"], dtype=int),
            }
    identifiers_path = os.path.join(Config.OUTPUT_DIR, "results", "split_identifiers_primary.csv")
    if os.path.isfile(identifiers_path):
        frame = pd.read_csv(identifiers_path)
        return {
            partition: frame.loc[frame["partition"] == partition, "row_index"].to_numpy(dtype=int)
            for partition in ("train", "validation", "test")
        }
    raise RuntimeError(
        "Disease-overlap audit could not recover the primary train/validation/test indices from _fold_artifacts or split_identifiers_primary.csv."
    )
