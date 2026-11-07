import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('compare_runs', ROOT / 'tests/compare_runs.py')
compare_runs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_runs)


class ComparisonTests(unittest.TestCase):
    def prepare(self, destination):
        reference = ROOT / 'data/reference'
        (destination / 'main/results').mkdir(parents=True)
        (destination / 'prenatal').mkdir()
        for name in ['publication_table.csv', 'model_selection_validation.csv']:
            text = (reference / 'results' / name).read_text()
            text = text.replace('S1_Mechanistic', 'S1_Gene_Level').replace('S2_Mech_Pheno', 'S2_Combined')
            (destination / 'main/results' / name).write_text(text)
        shutil.copy2(reference / 'prenatal/statistics.json', destination / 'prenatal/statistics.json')
        return reference

    def test_label_changes_do_not_hide_numeric_differences(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            reference = self.prepare(destination)
            self.assertFalse(any(compare_runs.compare(reference, destination).values()))
            path = destination / 'main/results/publication_table.csv'
            text = path.read_text().replace('0.2777218273381103', '0.2777218273381104')
            path.write_text(text)
            report = compare_runs.compare(reference, destination)
            self.assertTrue(report['main/results/publication_table.csv'])

    def test_missing_prenatal_statistics_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            reference = self.prepare(destination)
            (destination / 'prenatal/statistics.json').unlink()
            self.assertTrue(compare_runs.compare(reference, destination)['prenatal/statistics.json'])


if __name__ == '__main__':
    unittest.main()
