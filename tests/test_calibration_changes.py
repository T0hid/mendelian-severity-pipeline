"""Tables behind Extended Data Fig. 3 (predicted classes before and after calibration)."""

import json

import numpy as np
import pandas as pd
from conftest import ROOT

from severity.benchmark.calibration_changes import (
    CHANGES_FILE,
    CONFUSION_FILE,
    calibration_change_tables,
    confusion_counts_match,
)

FIGURE_DATA = ROOT / "published_run" / "benchmark" / "figures" / "publication" / "figure_data"


def test_change_counts_on_a_small_example():
    y_true = np.array([0, 1, 2, 3])
    before = np.array([0, 1, 1, 3])
    after = np.array([0, 2, 2, 3])
    confusion, changes, transitions = calibration_change_tables([("S", "M", y_true, before, after)])
    row = changes.iloc[0]
    assert row["n_changed"] == 2
    assert row["n_changed_to_more_severe"] == 2
    assert row["n_became_correct"] == 1
    assert row["n_became_incorrect"] == 1
    assert len(confusion) == 32
    assert transitions[["class_before", "class_after", "n_pairs"]].values.tolist() == [
        ["Moderate", "Severe", 2]
    ]
    assert confusion_counts_match(confusion, confusion)
    altered = confusion.copy()
    altered.loc[0, "count"] += 1
    assert not confusion_counts_match(confusion, altered)


def test_saved_figure_data_agree_with_fig_3_and_table_1():
    confusion = pd.read_csv(FIGURE_DATA / CONFUSION_FILE)
    changes = pd.read_csv(FIGURE_DATA / CHANGES_FILE)
    fig_3 = pd.read_csv(FIGURE_DATA / "figure_3_confusion_matrices.csv")
    after = confusion[confusion["stage"] == "after_calibration"]
    keys = ["Scenario", "Model", "true_class", "predicted_class"]
    merged = after.merge(fig_3, on=keys, suffixes=("", "_fig3"))
    assert len(merged) == 64
    assert (merged["count"] == merged["count_fig3"]).all()
    table_1 = pd.read_csv(ROOT / "published_run" / "benchmark" / "results" / "publication_table.csv")
    merged = changes.merge(table_1, on=["Scenario", "Model"])
    assert len(merged) == 4
    assert np.allclose(merged["mcc_before"], merged["Uncalibrated_MCC"], atol=1e-9)
    assert np.allclose(merged["mcc_after"], merged["MCC"], atol=1e-9)


def test_saved_change_counts():
    changes = pd.read_csv(FIGURE_DATA / CHANGES_FILE).set_index("Scenario")
    assert changes["n_test"].eq(1325).all()
    assert json.loads(changes["n_changed"].to_json()) == {
        "S1_Mechanistic": 182,
        "S1b_Disease_Context": 134,
        "S2_Mech_Pheno": 91,
        "S3_Phenotype": 186,
    }
