# Mendelian severity pipeline

[![Run pipeline demo](https://github.com/T0hid/mendelian-severity-pipeline/actions/workflows/demo.yml/badge.svg)](https://github.com/T0hid/mendelian-severity-pipeline/actions/workflows/demo.yml)

Version **1.3.1**. Run the trained phenotype scorer, check the supplied results and view the study figures with one command. The original benchmark, scorer training, prenatal analysis and publication plotting code are included.

## Run and see the results

Use Python 3.11 in a fresh environment. In the downloaded repository folder:

```shell
python -m pip install -r requirements.txt
python run.py
```

Open **`results/index.html`** in your browser. It contains the saved study figures, editable exports, Excel source data and plots rendered during this run. Open **`results/predictions.csv`** in Excel for the two synthetic example predictions. **`results/verification.json`** records the numerical checks.

The command uses the frozen model and saved numeric results. Training, feature selection, PCA and SHAP are not repeated. For another run, use `python run.py --output-dir results_second_run`.

## Run on GitHub

Open **Actions → Run pipeline demo → Run workflow**. After the green tick appears, open the completed run, scroll to **Artifacts**, and download **severity-demo-results**. Extract the ZIP and open **`index.html`**. The download includes predictions, figures, source tables, workbooks, checksums and execution details. Other users can fork the repository and enable Actions in their fork first.

## Predict your own profiles

Use `data/example_profiles.csv` as the template:

```csv
sample_id,hpo_terms
sample_1,HP:0001250;HP:0000252
sample_2,HP:0001629
```

```shell
python run.py --input my_profiles.csv --output-dir my_predictions
```

The output contains a 0–1 severity score, four class probabilities, the original rounded-expected-severity class and scoring status. Empty or unrecognised evidence returns `insufficient_information`. Optional `absent_hpo_terms` are accepted. See [input/output definitions](docs/INPUT_OUTPUT.md).

## Figures without prediction

```shell
python run.py --figures --output-dir figure_review
```

Open `figure_review/index.html`. Original images and source tables are exported unchanged. The original plotting functions also render Figures 2–4 and Extended Data Figure 2 from saved inputs. Extended Data Figure 1 has a compact replay from its original count tables. Figure 5 is supplied as its original composite with the available public source tables; replotting individual-family panels uses the authorised cohort through the original prenatal code.

The [output map](docs/OUTPUT_MAP.csv) links manuscript items to files, functions and commands. Figure 1 is an author-produced conceptual schematic outside the analytical code.

## Full research workflows

```shell
python -m pip install -r src/requirements-full.txt
python run.py --check-inputs
python run.py --benchmark --output-dir new_benchmark
python run.py --train-scorer --output-dir new_scorer
```

All ten public training inputs are bundled and unpacked automatically. For all three workflows with an authorised cohort:

```shell
python run.py --full --cohort PRIVATE_COHORT.csv --output-dir new_full_run
```

The benchmark defaults match the completed 5 November 2026 run, including 100 gene-block permutations per feature set. Explicit parameter changes create a new run. See [reproduction instructions](docs/REPRODUCE.md).

## Documentation

| Document | Contents |
|---|---|
| [Installation](docs/INSTALL.md) | Systems, dependencies, resources and runtime |
| [Input/output guide](docs/INPUT_OUTPUT.md) | Schemas and expected files |
| [Reproduction](docs/REPRODUCE.md) | Training, plotting and cohort commands |
| [Output map](docs/OUTPUT_MAP.csv) | Figures, source tables and generating code |
| [Software checklist](docs/SOFTWARE_SUBMISSION_CHECKLIST.md) | Code-specific answers for editorial assessment |
| [Code availability](docs/CODE_AVAILABILITY.md) | Version statement and permanent-deposit steps |
| [Licensing](docs/LICENSING.md) | Current licence status and third-party terms |
| [Release status](src/RELEASE_STATUS.md) | Verified scope and execution limits |

The package follows Nature Medicine's [software policy](https://www.nature.com/nm/editorial-policies) and [code-submission guidance](https://media.nature.com/full/nature-cms/documents/GuidelinesCodePublication.pdf). The rights holder must supply the authorised licence and permanent release DOI when finalising the deposit. No DOI or licence grant is invented.

## Code Ocean

Place `run`, `run.py`, `Dockerfile`, `requirements.txt`, `src/` and `docs/` in **Code**, and the contents of `data/` in **Data**. Use Python 3.11 with `requirements.txt` and select `run` as the run file. The launcher detects the capsule data folder and writes to `/results`. A hosted capsule has not been verified. [Official run-file instructions](https://docs.codeocean.com/osl-guide/user-manual/reproducible-code-execution/how-to-write-and-set-a-run-script).

Citation metadata is in [CITATION.cff](CITATION.cff). Reproducible software reports can be submitted through [GitHub Issues](https://github.com/T0hid/mendelian-severity-pipeline/issues).
