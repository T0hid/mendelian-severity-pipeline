"""Disease-level scorer (Fig. 5a,b; Supplementary Tables 17-19)."""

import pandas as pd
import pytest
from conftest import ROOT

from run import prepare_inputs
from severity.scorer import load_scorer
from severity.scorer.model import (
    RANDOM_SEED,
    OntologyExpander,
    load_pairs,
    optimised_disease_disjoint_split,
    prepare_terms,
)

PUBLISHED = ROOT / "published_run" / "scorer"


@pytest.fixture(scope="module")
def scorer():
    return load_scorer(str(PUBLISHED / "model.pkl"))


def test_disease_split_matches_published(tmp_path):
    inputs = prepare_inputs()
    pairs, names, _ = load_pairs(str(inputs / "final_output_with_severity.csv"), str(tmp_path))
    expander = OntologyExpander().load(str(inputs / "hp.obo"))
    pairs, _ = prepare_terms(pairs, expander, names, severity_term_policy="phenotype")
    assert len(pairs) == 6204
    _, test = optimised_disease_disjoint_split(pairs, 0.2, RANDOM_SEED)
    published = pd.read_csv(PUBLISHED / "disease_split_membership.csv")
    assert set(test["mondo_id"]) == set(published.loc[published["partition"] == "test", "mondo_id"])
    assert len(test) == 1241


def test_example_profiles_are_scored(scorer):
    profiles = pd.read_csv(ROOT / "data" / "example_profiles.csv")
    for terms in profiles["hpo_terms"]:
        result = scorer.score_profile(terms.split(";"))
        assert result["status"] == "scored"
        probabilities = result["phenotype_only_class_probabilities"]
        assert abs(sum(probabilities.values()) - 1.0) < 1e-9
        assert 0.0 <= result["phenotype_only_expected_severity_0_3"] <= 3.0
