"""Integrity, safe export and plotting interactions for the reviewer package."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from review_bundle import extract_assets, write_source_workbook
from replot_saved import _saved_model_results


class ReviewBundleTests(unittest.TestCase):
    def test_saved_prediction_adapter_keeps_validation_selection(self):
        results, outputs = _saved_model_results(SOURCE.parent / 'data/reference')
        self.assertEqual(len(outputs['selected_models']), 4)
        for scenario, model in outputs['selected_models'].items():
            selected = results[scenario]['all'][model]
            self.assertTrue(selected['selected_on_validation'])
            self.assertEqual(len(selected['y_true']), 1325)
            self.assertEqual(selected['predictions']['y_prob'].shape, (1325, 4))
            self.assertEqual(sum(bool(item['selected_on_validation'])
                                 for item in results[scenario]['all'].values()), 1)

    def test_changed_publication_archive_is_rejected_before_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'publication_assets.zip').write_bytes(b'changed')
            (root / 'publication_assets.json').write_text(json.dumps({
                'archives': [{'file': 'publication_assets.zip', 'sha256': '0' * 64}], 'files': []}))
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                extract_assets(root, root / 'output')
            self.assertFalse((root / 'output').exists())

    def test_manifest_cannot_authorise_parent_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'publication_assets.zip'
            with zipfile.ZipFile(archive, 'w') as handle:
                handle.writestr('../outside.txt', b'unsafe')
            (root / 'publication_assets.json').write_text(json.dumps({
                'archives': [{'file': archive.name, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}],
                'files': [{'path': '../outside.txt', 'archive': archive.name,
                           'sha256': hashlib.sha256(b'unsafe').hexdigest()}]}))
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                extract_assets(root, root / 'output')
            self.assertFalse((root / 'outside.txt').exists())

    def test_workbook_keeps_numbers_and_formula_like_text_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.xlsx'
            write_source_workbook(path, [('Values', [['value', 'identifier'], ['0.721', '=1+1'], ['90', '0012']])])
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    if name.endswith(('.xml', '.rels')):
                        ET.fromstring(archive.read(name))
                xml = archive.read('xl/worksheets/sheet1.xml').decode()
            self.assertIn('<v>0.721</v>', xml)
            self.assertIn('<v>90</v>', xml)
            self.assertIn('=1+1</t>', xml)
            self.assertIn('0012</t>', xml)
            self.assertNotIn('<f>', xml)


if __name__ == '__main__':
    unittest.main()
