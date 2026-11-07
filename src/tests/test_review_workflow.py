"""Regression checks for result preservation and workflow entry points."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR))
from runtime_paths import reference_directory
REFERENCE_DIR = reference_directory(PACKAGE_DIR)
from run_report import generate_manuscript_report
from run_pipeline import require_fresh_output
from verify_results import (CheckFailure, check_model_selection, check_split_frame,
                            compare_metrics, load_predictions, source_functions,
                            validate_prediction_arrays)
from verify_source import verify_source


class ReviewWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((REFERENCE_DIR / 'pipeline_manifest.json').read_text())
        cls.helpers = source_functions(['compute_all_metrics', '_select_candidate_lexicographically',
                                       '_validate_csv_schema'], cls.manifest['config'])

    def test_definitions_match_documented_merges(self):
        result = verify_source(PACKAGE_DIR)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['original_functions'], 166)
        self.assertEqual(len(result['merged_definitions']), 5)
        self.assertEqual(result['original_classes'], 5)

    def test_changed_scoring_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            (target / 'reference').mkdir()
            shutil.copy2(REFERENCE_DIR / 'function_fingerprints.json', target / 'reference/function_fingerprints.json')
            shutil.copy2(REFERENCE_DIR / 'approved_cleanup_fingerprints.json', target / 'reference/approved_cleanup_fingerprints.json')
            source = (PACKAGE_DIR / 'severity_pipeline.py').read_text(encoding='utf-8')
            original = 'm["accuracy"] = float((y_true == y_pred).mean())'
            self.assertIn(original, source)
            (target / 'severity_pipeline.py').write_text(source.replace(original, 'm["accuracy"] = 0.0'), encoding='utf-8')
            result = verify_source(target)
            self.assertEqual(result['status'], 'failed')
            self.assertIn('compute_all_metrics', result['unexpected_changes'])

    def test_modified_stored_metric_is_rejected(self):
        predictions, _ = load_predictions(REFERENCE_DIR)
        table = pd.read_csv(REFERENCE_DIR / 'results/all_results.csv')
        table.loc[0, 'mcc'] += 0.01
        with self.assertRaisesRegex(CheckFailure, 'Metric mismatch'):
            compare_metrics(predictions, table, self.helpers)

    def test_gene_overlap_is_rejected(self):
        frame = pd.DataFrame({'row_index': [0, 1, 2], 'partition': ['train', 'validation', 'test'],
                              'gene_symbol': ['GENE_A', 'GENE_B', 'GENE_A'],
                              'mondo_id': ['MONDO:1', 'MONDO:2', 'MONDO:3']})
        with self.assertRaisesRegex(CheckFailure, 'overlaps'):
            check_split_frame(frame, 'primary')

    def test_incorrect_validation_selection_is_rejected(self):
        table = pd.read_csv(REFERENCE_DIR / 'results/model_selection_validation.csv')
        scenario = table['Scenario'].iloc[0]
        block = table['Scenario'] == scenario
        table.loc[block, 'selected_on_validation'] = False
        table.loc[block & table['Model'].eq('LightGBM'), 'selected_on_validation'] = True
        with self.assertRaisesRegex(CheckFailure, 'selection mismatch'):
            check_model_selection(table, self.manifest, self.helpers)

    def test_non_normalised_probabilities_are_rejected(self):
        with self.assertRaisesRegex(CheckFailure, 'sum to one'):
            validate_prediction_arrays({'y_true': np.array([0]), 'y_pred': np.array([0]),
                                        'y_prob': np.array([[0.2, 0.2, 0.2, 0.2]])}, 4)

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            sentinel = target / 'pipeline_log.txt'
            sentinel.write_text('keep the original log')
            with self.assertRaises(FileExistsError):
                require_fresh_output(target)
            self.assertEqual(sentinel.read_text(), 'keep the original log')

    def test_missing_mapping_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mapping.csv'
            path.write_text('gene_symbol,mondo_id\nGENE_A,MONDO:1\n')
            with self.assertRaises(ValueError):
                self.helpers['_validate_csv_schema'](str(path), 'mapping',
                    [['gene_symbol'], ['mondo_id'], ['status'], ['use_in_rebuilt_corpus']])

    def test_report_uses_actual_provenance_and_permutation_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.md'
            generate_manuscript_report(None, None, None,
                output_dir=REFERENCE_DIR, pipeline_version=self.manifest['pipeline_version'],
                config=SimpleNamespace(**self.manifest['config']), report_path=path)
            text = path.read_text()
            self.assertIn(self.manifest['config']['TARGET_LABEL_PROVENANCE'], text)
            self.assertEqual(self.manifest['config']['N_PERMUTATIONS'], 100)
            self.assertIn('Configured permutations: 100', text)
            self.assertIn('5200', text)
            self.assertNotIn('assigned by expert curators', text)


if __name__ == '__main__':
    unittest.main()
