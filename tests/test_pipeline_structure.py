import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'src'


class PipelineStructureTests(unittest.TestCase):
    def test_analysis_entry_points_and_supported_paths(self):
        source = (ROOT / 'severity_pipeline.py').read_text()
        names = {node.name for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef)}
        for name in ['parse_hpo_obo', 'parse_mondo_obo', 'assemble_full_dataframe',
                     'compute_all_metrics', 'run_publication_analyses']:
            self.assertIn(name, names)
        for name in ['_read_hpo_ontology', '_read_mondo_ontology', '_assemble_source_dataframe',
                     '_compute_discrimination_metrics', '_run_core_publication_analyses']:
            self.assertNotIn(name, names)
        self.assertNotIn('StackEnsemble', source)
        self.assertNotIn('self.pca', source)
        source = (ROOT / 'phenotype_scorer.py').read_text()
        self.assertNotIn('write_patient_visualization_dashboard', source)
        self.assertNotIn('--score-dashboard', source)
        self.assertNotIn('split_severity_terms', source)
        self.assertNotIn('scorer.dimension_models[', source)


if __name__ == '__main__':
    unittest.main()
