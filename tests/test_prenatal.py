"""Prenatal analysis on the public part of the cohort (Fig. 5c-f; Supplementary Tables 20-22, 25 and 26)."""

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
    assert statistics["correlation"]["spearman_rho"] == pytest.approx(0.20088677031026453, abs=1e-12)
    assert (tmp_path / "prenatal_severity_figure.png").exists()
    saved = json.loads((tmp_path / "statistics.json").read_text())
    assert saved["correlation"]["n"] == 241
    counts = statistics["counts_and_continued_pregnancies"]
    assert counts["n_families"] == 241
    assert counts["n_genomic_autopsy"] == 181
    assert counts["n_pregen"] == 0
    assert counts["n_unknown_nicu"] == 21
    for name in (
        "score_vs_counts_rank.csv",
        "termination_reported_separately.csv",
        "continued_pregnancies_auc.csv",
        "unknown_nicu_sensitivity.csv",
        "scorer_identity.json",
    ):
        assert (tmp_path / name).exists()
    identity = json.loads((tmp_path / "scorer_identity.json").read_text())
    published = json.loads((ROOT / "published_run" / "prenatal" / "scorer_identity.json").read_text())
    assert identity["model_sha256"] == published["model_sha256"]
    assert identity["scorer_version"] == published["scorer_version"]


def test_public_cohort_without_autopsy_cohort(tmp_path):
    """--exclude-studies PED_cohort leaves the 60 published families with antenatal findings only."""
    statistics = run(
        str(ROOT / "data" / "prenatal_cohort_public.csv"),
        str(PUBLISHED / "model.pkl"),
        str(PUBLISHED / "held_out_test_predictions.csv"),
        str(tmp_path),
        n_boot=20,
        n_perm=20,
        exclude_studies=["PED_cohort"],
    )
    audit = statistics["audit"]
    assert audit["n_start_rows"] == 275
    assert audit["n_after_exclude_studies_rows"] == 80
    assert audit["excluded_studies"] == ["PED_cohort"]
    assert audit["n_analyzed"] == 60
    assert statistics["correlation"]["spearman_rho"] == pytest.approx(0.3622157541231031, abs=1e-9)
