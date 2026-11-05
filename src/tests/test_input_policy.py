"""Check that excluded optional inputs cannot block final-model reproduction."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime_paths import reference_directory
REFERENCE_DIR = reference_directory(ROOT)
import run_pipeline


class InputPolicyTests(unittest.TestCase):
    def check_fixture(self, optional_contents=None, missing_primary=False):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            package=base/'package';(package/'reference').mkdir(parents=True)
            data=base/'data';data.mkdir()
            required={'label':'primary_data','relative_path':'required.csv',
                      'expected_sha256':hashlib.sha256(b'expected').hexdigest(),
                      'required_for_fitted_models':True}
            optional={'label':'domain_features','relative_path':'optional.csv',
                      'expected_sha256':hashlib.sha256(b'expected').hexdigest(),
                      'required_for_fitted_models':False}
            (package/'reference/input_manifest.json').write_text(json.dumps([required,optional]),encoding='utf-8')
            if not missing_primary:(data/'required.csv').write_bytes(b'expected')
            if optional_contents is not None:(data/'optional.csv').write_bytes(optional_contents)
            with patch.object(run_pipeline,'PACKAGE_DIR',package), patch.object(run_pipeline.importlib.util,'find_spec',return_value=object()):
                return run_pipeline.check_inputs(data)

    def test_absent_excluded_domain_file_does_not_block(self):
        result=self.check_fixture()
        self.assertEqual(result['status'],'passed')
        self.assertEqual(result['inputs'][1]['status'],'optional_excluded_not_provided')

    def test_missing_required_input_still_blocks(self):
        result=self.check_fixture(missing_primary=True)
        self.assertEqual(result['status'],'blocked')

    def test_unexpected_optional_file_is_still_checked(self):
        result=self.check_fixture(optional_contents=b'unexpected replacement')
        self.assertEqual(result['status'],'blocked')
        self.assertEqual(result['inputs'][1]['status'],'hash_mismatch')


if __name__=='__main__':
    unittest.main()
