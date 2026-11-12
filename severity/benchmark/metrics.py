"""Performance metrics: QWK (equation 9), MCC, balanced accuracy, AUC-ROC, F1, the ranked
probability score, the multiclass Brier score and expected calibration error (equation 10).
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import Config
from .runtime import logger


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
    m["recall"] = float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    per_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    for i, name in enumerate(["Mild", "Moderate", "Severe", "Profound"]):
        m[f"f1_{name}"] = float(per_f1[i])
    if y_prob.ndim == 2 and y_prob.shape[1] == Config.NUM_CLASSES:
        try:
            m["auc_roc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
            m["auc_roc_macro"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception as exc:
            logger.log(f"  AUC-ROC undefined: {type(exc).__name__}: {exc}", level="WARNING")
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
    try:
        if y_prob.ndim == 2 and y_prob.shape[1] == Config.NUM_CLASSES:
            ordinal_score = y_prob @ np.arange(Config.NUM_CLASSES, dtype=float)
        else:
            ordinal_score = y_pred.astype(float)
        rho = spearmanr(ordinal_score, y_true).statistic
        m["spearman_rho"] = float(rho) if np.isfinite(rho) else np.nan
    except Exception as exc:
        logger.log(f"  Spearman rho undefined: {type(exc).__name__}: {exc}", level="WARNING")
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
    m["pred_distribution"] = str(np.bincount(y_pred.astype(int), minlength=Config.NUM_CLASSES).tolist())
    m["true_distribution"] = str(np.bincount(y_true.astype(int), minlength=Config.NUM_CLASSES).tolist())
    probability = np.asarray(y_prob, dtype=float)
    if probability.ndim == 2 and probability.shape[1] == Config.NUM_CLASSES:
        one_hot = np.eye(Config.NUM_CLASSES)[np.asarray(y_true, dtype=int)]
        m["multiclass_brier"] = float(np.mean(np.sum((probability - one_hot) ** 2, axis=1)))
        m["ranked_probability_score"] = _ranked_probability_score(y_true, probability)
    else:
        m["multiclass_brier"] = np.nan
        m["ranked_probability_score"] = np.nan
    return m


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
    return _finite_metric_or_neg_inf(metrics, "qwk")


def _ranked_probability_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    one_hot = np.eye(Config.NUM_CLASSES)[np.asarray(y_true, dtype=int)]
    pred_cumulative = np.cumsum(y_prob, axis=1)[:, :-1]
    obs_cumulative = np.cumsum(one_hot, axis=1)[:, :-1]
    return float(np.mean(np.sum((pred_cumulative - obs_cumulative) ** 2, axis=1) / (Config.NUM_CLASSES - 1)))


def _paired_contrast_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    """Metric definitions identical to those used by compute_all_metrics."""
    if metric == "mcc":
        return float(matthews_corrcoef(y_true, y_pred))
    if metric == "qwk":
        return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    raise ValueError(f"Unsupported paired-contrast metric: {metric}")
