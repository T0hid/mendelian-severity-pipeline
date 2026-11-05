# Mendelian severity pipeline

[![Run pipeline demo](https://github.com/T0hid/mendelian-severity-pipeline/actions/workflows/demo.yml/badge.svg)](https://github.com/T0hid/mendelian-severity-pipeline/actions/workflows/demo.yml)

Predict severity from HPO profiles, check the supplied study results, or run the original research workflows. The trained phenotype model, synthetic examples, original public training inputs and pinned ontologies are included.

## Run on GitHub

Open **Actions → Run pipeline demo → Run workflow**. The run checks all ten training inputs, runs the regression tests, replays the saved results and scores the example profiles. Download **severity-demo-results** from the completed run for predictions and the verification report. Model execution runs in a container with networking disabled; dependencies are installed during the environment build.

## Run on your computer

Use Python 3.11. Download or clone this repository, open a terminal in its folder, then run:

```shell
python -m pip install -r requirements.txt
python run.py
```

Open `results/predictions.csv`. The verification summary is in `results/verification.json`. This quick run uses the frozen phenotype model and replays saved results. It does not retrain models or repeat bootstrap/permutation analyses.

## Predict your own profiles

Use `data/example_profiles.csv` as the template, with one row per profile:

```csv
sample_id,hpo_terms
sample_1,HP:0001250;HP:0000252
sample_2,HP:0001629
```

```shell
python run.py --input my_profiles.csv
```

The output contains a phenotype-derived severity score from 0 to 1, class probabilities, the reported class and scoring status. The original rounded-expected-severity class rule is preserved. Empty or unrecognised evidence returns `insufficient_information`. These scores do not estimate pregnancy-outcome probabilities. Keep individual-level inputs and outputs in your authorised research environment.

## Run the research workflows

The two public training workflows can run independently:

```shell
python -m pip install -r src/requirements-full.txt
python run.py --benchmark
python run.py --train-scorer
```

The original inputs are restored automatically from `data/training_tables.zip` and `data/ontologies.zip` into `data/training/`. Every file is checked against its recorded SHA-256 checksum. Check or restore them without training with `python run.py --check-inputs`.

For the complete benchmark, scorer training and prenatal workflow, supply an authorised cohort:

```shell
python run.py --full --cohort PRIVATE_COHORT.csv --output-dir NEW_RESULTS
```

Research training can take hours or days. The original numerical source and configuration are preserved. The supplied benchmark used one permutation; use `--permutations` only when deliberately generating a new benchmark analysis. Original prenatal patient rows are excluded from this repository. See [input details](src/INPUTS.md) and [release status](src/RELEASE_STATUS.md) for data access and unresolved manuscript items.

## Code Ocean

Upload `run`, `run.py`, `Dockerfile`, `requirements.txt` and `src/` to **Code**, and the contents of `data/` to **Data**. Configure Python 3.11 with the packages in `requirements.txt`, then select `run` as the run file. The default demonstration writes to `/results` and needs no interactive choices. A hosted Code Ocean capsule has not been created or verified here. [Official run-file instructions](https://docs.codeocean.com/osl-guide/user-manual/reproducible-code-execution/how-to-write-and-set-a-run-script).

## What is preserved

The benchmark, disease-level phenotype scorer and prenatal analysis remain separate numerical modules in `src/`, connected by `run.py`. Source-preservation checks cover all 261 original functions and 20 classes. The deployed model is trained on diseases, not prenatal outcomes. The public snapshot contains disease-level predictions and aggregate prenatal tables.

The newly supplied mapping has a release checksum and includes every pair in the saved benchmark partitions. Its historical checksum was not recorded, so byte identity with the historical mapping cannot be established. See [input audit](src/validation/input_audit.json).

Citation metadata is in `CITATION.cff`. A software licence has not yet been selected by the project owner; no licence grant is implied. The complete submission archive separately retains full-resolution publication assets.
