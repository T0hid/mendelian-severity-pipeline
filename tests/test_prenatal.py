"""Prenatal analysis on the public part of the cohort (Fig. 5c-f; Supplementary Tables 20-23)."""

import json

import pytest
from conftest import ROOT

from severity.prenatal.analysis import ordinal_concordance, run

PUBLISHED = ROOT / "published_run" / "scorer"


def test_ordinal_concordance_extremes():
    assert ordinal_concordance([0.1, 0.2, 0.3, 0.4], [0, 1, 2, 3]) == 1.0
    assert ordinal_concordance([0.4, 0.3, 0.2, 0.1], [0, 1, 2, 3]) == 0.0
    assert ordinal_concordance([0.5, 0.5, 0.5, 0.5], [0, 1, 2, 3]) == 0.5


def test_public_cohort(tmp_path):
    statistics = run(
        str(ROOT / "data" / "prenatal_cohort_public.csv"),
        str(PUBLISHED / "model.pkl"),
        str(PUBLISHED / "held_out_test_predictions.csv"),
        str(tmp_path),
        n_boot=20,
        n_perm=20,
    )
    audit = statistics["audit"]
    assert audit["n_start_rows"] == 275
    assert audit["n_family_profiles"] == 267
    assert audit["n_analyzed"] == 241
    assert statistics["correlation"]["spearman_rho"] == pytest.approx(0.19474377502903953, abs=1e-12)
    assert (tmp_path / "prenatal_severity_figure.png").exists()
    saved = json.loads((tmp_path / "statistics.json").read_text())
    assert saved["correlation"]["n"] == 241
