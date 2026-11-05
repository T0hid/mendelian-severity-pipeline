"""Integrity checks for portable restoration of original research inputs."""
import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_inputs import prepare_training_inputs


class TrainingInputTests(unittest.TestCase):
    def make_archive(self, directory, name='input.csv'):
        content=b'gene,value\nGENE1,1\n'
        archive=directory/'training_inputs.zip'
        with zipfile.ZipFile(archive,'w') as handle:
            handle.writestr(name,content)
        metadata={'archive':archive.name,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
                  'files':{name:{'sha256':hashlib.sha256(content).hexdigest()}},
                  'mapping_provenance':'synthetic test input'}
        (directory/'training_inputs_metadata.json').write_text(json.dumps(metadata),encoding='utf-8')
        return content

    def test_restore_preserves_modified_existing_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            expected=self.make_archive(folder)
            restored,report=prepare_training_inputs(folder)
            self.assertEqual((restored/'input.csv').read_bytes(),expected)
            self.assertEqual(report['required_files_checked'],1)
            (restored/'input.csv').write_bytes(b'author changes')
            with self.assertRaisesRegex(ValueError,'has changed'):
                prepare_training_inputs(folder)
            self.assertEqual((restored/'input.csv').read_bytes(),b'author changes')

    def test_archive_path_cannot_escape_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            self.make_archive(folder,'../escape.csv')
            with self.assertRaisesRegex(ValueError,'Unsafe path'):
                prepare_training_inputs(folder)
            self.assertFalse((folder/'escape.csv').exists())


if __name__=='__main__':
    unittest.main()
