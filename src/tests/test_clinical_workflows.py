"""Checks for source preservation, family aggregation and outcome handling."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime_paths import reference_directory
REFERENCE_DIR = reference_directory(ROOT)
from verify_clinical import (PRENATAL_FILE, SCORER_FILE, load_module, semantic_hash,
                             verify_clinical_source, verify_prenatal_aggregate, verify_scorer_results)


class ClinicalWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prenatal = load_module(ROOT / PRENATAL_FILE, 'prenatal_severity_analysis_v3_multichannel')
        cls.scorer = load_module(ROOT / SCORER_FILE, 'phenotype_severity_scorer_v3_3_monotone_ontology')

    def test_all_original_clinical_definitions_and_globals_are_preserved(self):
        result = verify_clinical_source()
        self.assertEqual(sum(item['functions_retained'] for item in result.values()), 90)
        self.assertEqual(sum(item['classes_retained'] for item in result.values()), 15)

    def test_mutated_scorer_computation_is_detected(self):
        text = (ROOT / SCORER_FILE).read_text(encoding='utf-8')
        changed = text.replace('RANDOM_SEED = 42', 'RANDOM_SEED = 43')
        self.assertNotEqual(semantic_hash(text), semantic_hash(changed))

    def test_scorer_saved_holdout_is_reproduced(self):
        result = verify_scorer_results(REFERENCE_DIR / 'scorer', self.scorer)
        self.assertEqual(result['held_out_diseases'], 1241)
        self.assertEqual(result['metric_comparisons'], 22)

    def test_conflicting_family_outcomes_are_flagged(self):
        result = self.prenatal._resolve_family_outcome(['Live birth', 'Neonatal death'])
        self.assertTrue(result['outcome_conflict'])
        self.assertTrue(pd.isna(result['outcome_4level']))

    def test_repeated_rows_combine_into_one_family(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic_cohort.csv'
            pd.DataFrame([
                {'family_id': 'synthetic_A', 'pregnancy_outcome': 'Live birth', 'hpo_ids': 'HP:0001250', 'study': 'synthetic'},
                {'family_id': 'synthetic_A', 'pregnancy_outcome': 'Live birth', 'hpo_ids': 'HP:0000252', 'study': 'synthetic'},
            ]).to_csv(path, index=False)
            families, audit = self.prenatal.load_cohort(str(path), [], [])
            self.assertEqual(len(families), 1)
            self.assertEqual(set(families.iloc[0]['hpo_list']), {'HP:0001250', 'HP:0000252'})
            self.assertEqual(audit['n_multirow_families'], 1)

    def test_no_nicu_phrase_is_not_classified_as_nicu(self):
        self.assertEqual(self.prenatal.map_outcome_4level('Live born child - no NICU/special care required'), 0)
        self.assertEqual(self.prenatal.map_outcome_4level('Live born child - special care nursery at birth'), 1)

    def test_missing_outcome_is_retained_as_missing(self):
        self.assertTrue(pd.isna(self.prenatal.map_outcome_4level(None)))

    def test_missing_termcount_outputs_are_reported_without_fabrication(self):
        result = verify_prenatal_aggregate(REFERENCE_DIR / 'prenatal')
        self.assertEqual(len(result['current_source_outputs_missing_from_saved_run']), 2)


if __name__ == '__main__':
    unittest.main()
