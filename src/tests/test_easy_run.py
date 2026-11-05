"""Input and output protections for the public prediction interface."""
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import run


class EasyRunTests(unittest.TestCase):
    def test_bundled_csv_is_ready_without_editing(self):
        profiles=run.read_profiles(ROOT/'data/example_profiles.csv')
        self.assertEqual(len(profiles),2)
        self.assertEqual(profiles[0]['present'],['HP:0001250','HP:0000252'])

    def test_bad_hpo_format_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Invalid HPO'):
            run.parse_terms('HP:123',2)

    def test_hpo_delimiters_and_duplicates_are_normalised(self):
        self.assertEqual(run.parse_terms('hp_0001250|HP:0001250 HP:0000252',2),
                         ['HP:0001250','HP:0000252'])

    def test_duplicate_profiles_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'input.csv'
            path.write_text('sample_id,hpo_terms\na,HP:0001250\na,HP:0000252\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'duplicate'):
                run.read_profiles(path)

    def test_wrong_model_is_rejected_before_unpickling(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            (folder/'model.pkl').write_bytes(b'not the model')
            (folder/'model_metadata.json').write_text(json.dumps({'model_sha256':hashlib.sha256(b'expected').hexdigest()}),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'has changed'):
                run.check_model_file(folder)

    def test_input_csv_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);path=folder/'predictions.csv'
            path.write_text('unchanged',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'different files'):
                run.predict(path,folder,ROOT/'data')
            self.assertEqual(path.read_text(encoding='utf-8'),'unchanged')

    def test_full_workflow_uses_newly_trained_model_and_holdout(self):
        sys.path.insert(0,str(ROOT/'src'))
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            cohort=folder/'cohort.csv'
            cohort.write_text('pregnancy_outcome,hpo_ids\n',encoding='utf-8')
            output=folder/'new_results'
            output.mkdir()
            readiness={'inputs':[],'dependencies':{}}
            with patch('run_pipeline.check_inputs',return_value=readiness), \
                    patch.object(run.importlib.util,'find_spec',return_value=object()), \
                    patch.object(run.subprocess,'run') as execute:
                run.reproduce(folder,cohort,output)
            commands=[call.args[0] for call in execute.call_args_list]
            self.assertEqual(len(commands),3)
            prenatal=commands[2]
            self.assertEqual(prenatal[prenatal.index('--model')+1],str(output/'scorer/model.pkl'))
            self.assertEqual(prenatal[prenatal.index('--scorer-heldout')+1],
                             str(output/'scorer/held_out_test_predictions.csv'))


if __name__=='__main__':
    unittest.main()
