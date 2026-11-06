"""Run the bundled example, predict from a CSV, or reproduce the full study."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'src'


def default_data_directory():
    for directory in [ROOT / 'data', ROOT.parent / 'data']:
        if (directory / 'model.pkl').is_file():
            return directory
    raise ValueError('Bundled model not found. Keep the supplied data folder with the code.')


def file_hash(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_model_file(data_directory):
    path=data_directory/'model.pkl'
    metadata=json.loads((data_directory/'model_metadata.json').read_text(encoding='utf-8'))
    if not path.is_file() or file_hash(path)!=metadata['model_sha256']:
        raise ValueError('The bundled model is missing or has changed. Restore data/model.pkl from the package.')
    return path,metadata


def parse_terms(value, row_number):
    result=[]
    for token in re.split(r'[;,|\s]+', value or ''):
        if not token:
            continue
        match=re.fullmatch(r'HP[:_](\d{7})',token,re.IGNORECASE)
        if not match:
            raise ValueError(f'Invalid HPO identifier at CSV row {row_number}; use identifiers such as HP:0001250.')
        term='HP:'+match.group(1)
        if term not in result:
            result.append(term)
    return result


def read_profiles(path):
    profiles=[]
    with path.open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle)
        columns=reader.fieldnames or []
        if len(columns)!=len(set(columns)):
            raise ValueError('CSV column names must be unique.')
        id_column=next((name for name in ['sample_id','family_id'] if name in columns),None)
        hpo_column=next((name for name in ['hpo_terms','hpo_ids'] if name in columns),None)
        if id_column is None or hpo_column is None:
            raise ValueError('Input CSV needs sample_id and hpo_terms columns. Use data/example_profiles.csv as the template.')
        seen=set()
        for row_number,row in enumerate(reader,2):
            if None in row:
                raise ValueError(f'Unexpected extra CSV columns at row {row_number}. Separate HPO terms with semicolons.')
            identifier=(row.get(id_column) or '').strip()
            if not identifier or identifier in seen:
                raise ValueError(f'Missing or duplicate sample identifier at CSV row {row_number}; use one row per profile.')
            seen.add(identifier)
            profiles.append({'sample_id':identifier,'present':parse_terms(row.get(hpo_column),row_number),
                             'absent':parse_terms(row.get('absent_hpo_terms'),row_number)})
    if not profiles:
        raise ValueError('Input CSV has no profiles.')
    return profiles


def predict(input_path, output_directory, data_directory):
    output_path=output_directory/'predictions.csv'
    if input_path.resolve()==output_path.resolve():
        raise ValueError('Input and output must be different files. Choose another --output-dir.')
    profiles=read_profiles(input_path)
    model_path,metadata=check_model_file(data_directory)
    from verify_clinical import PRENATAL_FILE, load_module, load_scorer
    import numpy as np

    loader=load_module(SOURCE/PRENATAL_FILE,'prenatal_severity_analysis_v3_multichannel')
    model=load_scorer(model_path,loader)
    rows=[]
    for profile in profiles:
        result=model.score_profile(profile['present'],absent_hpo_terms=profile['absent'])
        expected=result.get('phenotype_only_expected_severity_0_3')
        probabilities=result.get('phenotype_only_class_probabilities')
        row={'sample_id':profile['sample_id'],'status':result['status'],
             'severity_score':'','severity_class':'',
             'prob_mild':'','prob_moderate':'','prob_severe':'','prob_profound':''}
        if expected is not None and probabilities is not None:
            values=np.asarray([probabilities[name.title()] for name in metadata['classes']],dtype=float)
            if (not np.isfinite(values).all() or (values<0).any() or (values>1).any()
                    or not np.isclose(values.sum(),1,atol=1e-10)):
                raise ValueError('The model returned invalid class probabilities.')
            row['severity_score']=float(expected)/3.0
            row['severity_class']=metadata['classes'][int(np.clip(np.rint(expected),0,3))]
            for name,value in zip(metadata['classes'],values):
                row['prob_'+name]=float(value)
        else:
            row['status']='insufficient_information'
        rows.append(row)
    output_directory.mkdir(parents=True,exist_ok=True)
    with output_path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'mode':'prediction','channel':'phenotype','profiles':len(rows),
             'status_counts':dict(Counter(row['status'] for row in rows)),
             'model_sha256':metadata['model_sha256'],'model_training_performed':False}
    (output_directory/'run_info.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    return rows


def check_reference_results(model_path):
    from verify_results import verify as verify_benchmark
    from verify_clinical import verify as verify_clinical
    from runtime_paths import reference_directory

    benchmark=verify_benchmark(reference_directory(SOURCE))
    clinical=verify_clinical(model_path=model_path)
    functions=benchmark['source_preservation']['original_functions']+sum(
        item['functions_retained'] for item in clinical['source'].values())
    classes=benchmark['source_preservation']['original_classes']+sum(
        item['classes_retained'] for item in clinical['source'].values())
    return {'status':'passed','scope':'Saved-result replay and frozen-model checks; no retraining or new inference analyses.',
            'benchmark_metric_checks':benchmark['metric_checks'],
            'benchmark_permutations_per_feature_set':benchmark['permutations_recorded'][0],
            'benchmark_permutation_adjusted_p_values':benchmark['permutation_adjusted_p_values'],
            'scorer_metric_checks':clinical['scorer']['metric_comparisons'],
            'prenatal_aggregate_checks':clinical['prenatal_aggregate']['aggregate_numeric_checks'],
            'synthetic_model_profiles_checked':clinical['deployment_model']['synthetic_profiles_checked'],
            'original_functions_retained':functions,'original_classes_retained':classes,
            'model_training_performed':False,'individual_prenatal_data_included':False}


def reproduce(training_data, cohort, output_directory, permutations=None):
    if training_data is None or cohort is None:
        raise ValueError('--full needs --training-data and --cohort. Use the default demo to test the software without these inputs.')
    from run_pipeline import check_inputs, require_fresh_output

    readiness=check_inputs(training_data)
    missing=[item['relative_path'] for item in readiness['inputs'] if item['status'] in {'missing','hash_mismatch'}]
    libraries=[name for name,found in readiness['dependencies'].items() if not found]
    if importlib.util.find_spec('statsmodels') is None:
        libraries.append('statsmodels')
    if missing or libraries:
        raise ValueError('Full reproduction prerequisites are incomplete. Inputs: '+(', '.join(missing) or 'none')+
                         '. Libraries: '+(', '.join(libraries) or 'none')+
                         '. See src/INPUTS.md and install src/requirements-full.txt.')
    if not cohort.is_file():
        raise ValueError('Prenatal cohort CSV not found.')
    with cohort.open(encoding='utf-8-sig',newline='') as handle:
        columns=next(csv.reader(handle),[])
    if not {'pregnancy_outcome','hpo_ids'}.issubset(columns):
        raise ValueError('The full prenatal cohort needs pregnancy_outcome and hpo_ids columns.')
    require_fresh_output(output_directory)
    steps=[
        ['benchmark','--data-dir',str(training_data),'--output-dir',str(output_directory/'benchmark')],
        ['train-scorer','--data-dir',str(training_data),'--output-dir',str(output_directory/'scorer')],
        ['prenatal','--cohort',str(cohort),'--model',str(output_directory/'scorer/model.pkl'),
         '--scorer-heldout',str(output_directory/'scorer/held_out_test_predictions.csv'),
         '--output-dir',str(output_directory/'prenatal'),'--score-channel','phenotype'],
    ]
    if permutations is not None:
        if permutations<1:
            raise ValueError('--permutations must be positive.')
        steps[0]+=['--permutations',str(permutations)]
    for step in steps:
        subprocess.run([sys.executable,str(SOURCE/'run_study.py')]+step,check=True)
    (output_directory/'workflow_summary.json').write_text(json.dumps({'status':'complete','steps':['benchmark','scorer','prenatal']},indent=2)+'\n',encoding='utf-8')


def main(argv=None):
    started = time.perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,help='CSV with sample_id and semicolon-separated hpo_terms. Omit to run the bundled example.')
    parser.add_argument('--data-dir',type=Path,help='Bundled model/example/reference data folder; normally detected automatically.')
    parser.add_argument('--output-dir',type=Path,help='Output folder; defaults to results.')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--full',action='store_true',help='Run all three research workflows; requires an authorised --cohort.')
    modes.add_argument('--benchmark',action='store_true',help='Train and evaluate the gene-disease benchmark with the bundled inputs.')
    modes.add_argument('--prenatal',action='store_true',help='Replay the prenatal analysis with the bundled scorer and a cohort CSV.')
    modes.add_argument('--train-scorer',action='store_true',help='Train and refit the disease-level phenotype scorer with the bundled inputs.')
    modes.add_argument('--check-inputs',action='store_true',help='Restore and verify the ten bundled training inputs without training.')
    modes.add_argument('--figures',action='store_true',help='Export original figures and source data, and replot saved benchmark results without training.')
    parser.add_argument('--training-data',type=Path,help='Original training inputs for --full.')
    parser.add_argument('--cohort',type=Path,help='Prenatal cohort CSV for --prenatal or --full.')
    parser.add_argument('--permutations',type=int,help='Explicit benchmark permutation count for --full; otherwise preserve the source configuration.')
    args=parser.parse_args(argv)
    try:
        data=(args.data_dir or default_data_directory()).resolve()
        os.environ['SEVERITY_DATA_DIR']=str(data)
        if str(SOURCE) not in sys.path:
            sys.path.insert(0,str(SOURCE))
        default_results=ROOT/'results' if data==ROOT/'data' else data.parent/'results'
        output=(args.output_dir or (default_results/'full_reproduction' if args.full else default_results)).resolve()
        if args.prenatal:
            if args.cohort is None or not args.cohort.is_file():
                parser.error('--prenatal requires an existing --cohort CSV.')
            if args.input or args.training_data or args.permutations is not None:
                parser.error('--prenatal uses --cohort, --data-dir and --output-dir.')
            model_path,_=check_model_file(data)
            destination=output/'prenatal' if args.output_dir is None else output
            subprocess.run([sys.executable,str(SOURCE/'run_study.py'),'prenatal',
                            '--cohort',str(args.cohort.resolve()),'--model',str(model_path),
                            '--scorer-heldout',str(data/'reference/scorer/held_out_test_predictions.csv'),
                            '--output-dir',str(destination),'--score-channel','phenotype'],check=True)
            return 0
        if args.figures:
            if args.input or args.cohort or args.permutations is not None:
                raise ValueError('--figures reads the bundled saved results; omit prediction/training arguments.')
            from review_bundle import build_review_bundle
            from run_pipeline import require_fresh_output
            require_fresh_output(output)
            build_review_bundle(data, output)
            print(f'Figures and source data exported. Open {output/"index.html"}')
            return 0
        if args.full or args.benchmark or args.train_scorer or args.check_inputs:
            if args.input:
                raise ValueError('--input is for prediction; omit it for training and input verification.')
            from training_inputs import prepare_training_inputs
            training,readiness=prepare_training_inputs(data,args.training_data.resolve() if args.training_data else None)
            if args.check_inputs:
                print(json.dumps(readiness,indent=2))
                return 0
            if args.benchmark or args.train_scorer:
                mode='benchmark' if args.benchmark else 'train-scorer'
                destination=output/mode if args.output_dir is None else output
                command=[sys.executable,str(SOURCE/'run_study.py'),mode,'--data-dir',str(training),
                         '--output-dir',str(destination)]
                if args.permutations is not None:
                    if not args.benchmark or args.permutations<1:
                        raise ValueError('--permutations must be positive and is only for the benchmark.')
                    command+=['--permutations',str(args.permutations)]
                subprocess.run(command,check=True)
                return 0
        if args.full:
            if args.input:
                raise ValueError('--input is for prediction; use --cohort for full reproduction.')
            reproduce(training,
                      args.cohort.resolve() if args.cohort else None,output,args.permutations)
            print(f'Full workflow complete. Results: {output}')
            return 0
        input_path=(args.input or data/'example_profiles.csv').resolve()
        if not args.input:
            from run_pipeline import require_fresh_output
            require_fresh_output(output)
            model_path,_=check_model_file(data)
            verification=check_reference_results(model_path)
        rows=predict(input_path,output,data)
        if not args.input:
            (output/'verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
            from review_bundle import build_review_bundle
            review = build_review_bundle(data, output)
            (output/'execution.json').write_text(json.dumps({
                'elapsed_seconds': time.perf_counter() - started,
                'environment': review['environment'], 'model_training_performed': False}, indent=2)+'\n',encoding='utf-8')
            print('Demo and reference checks passed. No model was retrained.')
            print(f'Open {output/"index.html"} to view figures and source-data downloads.')
        print(f'Wrote {len(rows)} predictions to {output/"predictions.csv"}')
        return 0
    except (ValueError,OSError,KeyError,ImportError,subprocess.CalledProcessError) as exc:
        print(f'Cannot run: {exc}',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
