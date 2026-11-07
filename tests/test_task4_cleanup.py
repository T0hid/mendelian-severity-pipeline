import ast
import io
import re
import tokenize
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'src'
FILES = ['severity_pipeline.py', 'phenotype_scorer.py', 'prenatal_analysis.py',
         'publication_figures.py', 'run_report.py']


class CleanupTests(unittest.TestCase):
    def test_comments_and_docstrings_have_no_patch_notes(self):
        pattern = re.compile(r'\b(locked|frozen|untouched|headline|audit|reviewers?|manuscript|paper|methods|previously|previous|former|legacy|patch|explicit|robust|ensures|comprehensive|safeguard|leverage|seamless)\b|manuscript-complete|is now|no longer|Note:|Important:|This function|\bv\d+\.\d+|\b\d{1,2} Nov 2026|aesthetic v\d+', re.I)
        for name in FILES:
            source = (ROOT / name).read_text()
            for token in tokenize.generate_tokens(io.StringIO(source).readline):
                if token.type == tokenize.COMMENT:
                    self.assertTrue(token.string.isascii(), (name, token.start))
                    self.assertIsNone(pattern.search(token.string), (name, token.start))
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Call):
                    func = node.func
                    log_call = (isinstance(func, ast.Name) and func.id in {'print', 'logger'}) or (
                        isinstance(func, ast.Attribute) and func.attr in
                        {'log', 'section', 'subsection', 'info', 'warning', 'debug'})
                    if log_call:
                        for value in ast.walk(node):
                            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                                self.assertIsNone(pattern.search(value.value), (name, node.lineno))
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                    text = ast.get_docstring(node)
                    if text:
                        self.assertIsNone(pattern.search(text), (name, getattr(node, 'lineno', 1)))

    def test_wrapper_bases_and_disabled_paths_are_absent(self):
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
