"""Tables computed from the stored test predictions: the moderate/severe cut-off
(Supplementary Table 27) and the comparison by number of recorded HPO terms (Supplementary
Table 28)."""

import numpy as np
import pandas as pd
import pytest
from conftest import ROOT

from severity.benchmark.evaluation import cutoff_statistics, hpo_term_count_comparison

RESULTS = ROOT / "published_run" / "benchmark" / "results"


def test_cutoff_statistics_counts():
    # true: two negatives (Mild, Moderate), two positives (Severe, Profound)
    stats = cutoff_statistics([0, 1, 2, 3], [0, 2, 2, 1])
    assert stats["sensitivity"] == 0.5
    assert stats["specificity"] == 0.5
    assert stats["ppv"] == 0.5
    assert stats["npv"] == 0.5


def test_saved_cutoff_point_estimates():
    table = pd.read_csv(RESULTS / "moderate_severe_cutoff.csv").set_index("Scenario")
    assert (table["n_test"] == 1261).all()
    assert table.loc["S2_Mech_Pheno", "sensitivity"] == pytest.approx(872 / 941, abs=1e-12)
    assert table.loc["S2_Mech_Pheno", "specificity"] == pytest.approx(280 / 320, abs=1e-12)


def test_hpo_term_count_groups():
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    genes = np.array(["A", "A", "B", "B", "C", "C", "D", "D"])
    n_terms = np.array([1, 2, 7, 8, 30, 31, 3, 4])
    table = hpo_term_count_comparison(y, y, y, genes, n_terms, n_bootstrap=10).set_index("group")
    assert table["n_test_pairs"].to_dict() == {
        "1-5 HPO terms": 4, "6-20 HPO terms": 2, "more than 20 HPO terms": 2
    }
    assert table["difference_S2_minus_S3"].eq(0).all()
