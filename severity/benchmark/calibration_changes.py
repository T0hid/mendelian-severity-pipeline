"""Predicted classes before and after calibration (Extended Data Fig. 3).

For the selected model of each feature set, the test-set classes predicted from the
uncalibrated and from the calibrated probabilities are compared: confusion matrices, the
number of pairs whose class changed, the direction of the change and the pairs that became
correct or incorrect. Nothing is refitted and no further test evaluation takes place.
"""

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, matthews_corrcoef

CLASS_NAMES = ["Mild", "Moderate", "Severe", "Profound"]
CLASSES = list(range(len(CLASS_NAMES)))
STAGES = ("before_calibration", "after_calibration")

CONFUSION_FILE = "extended_data_fig_3_confusion_before_after.csv"
CHANGES_FILE = "extended_data_fig_3_class_changes.csv"
TRANSITIONS_FILE = "extended_data_fig_3_transitions.csv"


def _qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(cohen_kappa_score(y_true, y_pred, labels=CLASSES, weights="quadratic"))


def calibration_change_tables(
    records: Iterable[Tuple[str, str, np.ndarray, np.ndarray, np.ndarray]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the three Extended Data Fig. 3 tables.

    ``records`` holds one ``(scenario, model, y_true, y_pred_before, y_pred_after)`` tuple per
    feature set. Returns the confusion counts (long format), the per-model summary of class
    changes and, for the pairs that changed, the counts of each class-before/class-after pair.
    """
    confusion_rows, change_rows, transition_rows = [], [], []
    for scenario, model, y_true, before, after in records:
        y_true = np.asarray(y_true).astype(int)
        before = np.asarray(before).astype(int)
        after = np.asarray(after).astype(int)
        if not (len(y_true) == len(before) == len(after)):
            raise ValueError(f"{scenario}: prediction lengths differ.")
        changed = before != after
        correct_before, correct_after = before == y_true, after == y_true
        change_rows.append(
            {
                "Scenario": scenario,
                "Model": model,
                "n_test": int(len(y_true)),
                "n_changed": int(changed.sum()),
                "share_changed": float(changed.mean()),
                "n_changed_to_more_severe": int((after > before).sum()),
                "n_changed_to_milder": int((after < before).sum()),
                "n_became_correct": int((~correct_before & correct_after).sum()),
                "n_became_incorrect": int((correct_before & ~correct_after).sum()),
                "n_correct_before": int(correct_before.sum()),
                "n_correct_after": int(correct_after.sum()),
                "mcc_before": float(matthews_corrcoef(y_true, before)),
                "mcc_after": float(matthews_corrcoef(y_true, after)),
                "qwk_before": _qwk(y_true, before),
                "qwk_after": _qwk(y_true, after),
            }
        )
        for b in CLASSES:
            for a in CLASSES:
                n = int(((before == b) & (after == a)).sum()) if a != b else 0
                if n:
                    transition_rows.append(
                        {
                            "Scenario": scenario,
                            "Model": model,
                            "class_before": CLASS_NAMES[b],
                            "class_after": CLASS_NAMES[a],
                            "n_pairs": n,
                        }
                    )
        for stage, pred in zip(STAGES, (before, after)):
            cm = confusion_matrix(y_true, pred, labels=CLASSES)
            norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
            for r in CLASSES:
                for c in CLASSES:
                    confusion_rows.append(
                        {
                            "Scenario": scenario,
                            "Model": model,
                            "stage": stage,
                            "true_class": CLASS_NAMES[r],
                            "predicted_class": CLASS_NAMES[c],
                            "count": int(cm[r, c]),
                            "row_fraction": float(norm[r, c]),
                        }
                    )
    return pd.DataFrame(confusion_rows), pd.DataFrame(change_rows), pd.DataFrame(transition_rows)


def confusion_counts_match(recomputed: pd.DataFrame, saved: pd.DataFrame) -> bool:
    """True when two confusion tables hold the same counts for the same cells."""
    keys = ["Scenario", "Model", "stage", "true_class", "predicted_class"]
    merged = recomputed[keys + ["count"]].merge(
        saved[keys + ["count"]], on=keys, how="outer", suffixes=("", "_saved"), indicator=True
    )
    return bool((merged["_merge"] == "both").all() and (merged["count"] == merged["count_saved"]).all())
