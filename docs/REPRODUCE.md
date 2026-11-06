# Reproduce the analyses

## Plot saved results

```shell
python run.py --figures --output-dir figure_results
```

Open `figure_results/index.html`. The command exports the saved figures and regenerates Figures 2–4 and Extended Data Fig. 2 from saved results. The PCA plot uses the saved coordinates, and the SHAP plots use saved arrays. Extended Data Fig. 1 is also redrawn from its count tables.

Figure 1 is a schematic. Figure 5 is supplied as a composite with aggregate source tables. Regenerating its family-level panels requires authorised cohort data. The [figure map](OUTPUT_MAP.csv) gives the source files and functions for each figure.

## Benchmark

```shell
python -m pip install -r src/requirements-full.txt
python run.py --check-inputs
python run.py --benchmark --output-dir benchmark_results
```

The benchmark uses the 5 November 2026 configuration, including 100 gene-block permutations per feature set. The demo checks the permutation counts and adjusted P values against the saved results. Use `--permutations` with a fresh output directory to run a different number.

## Train the disease-level scorer

```shell
python run.py --train-scorer --output-dir scorer_results
```

This fits and evaluates the disease-level phenotype scorer and saves the refitted model for scoring new profiles.

## Run the prenatal analysis

Supply a local copy of the cohort CSV. The input schema is defined by `load_cohort` in the [prenatal analysis script](../src/prenatal_severity_analysis_v3_multichannel.py).

```shell
python -m pip install -r src/requirements-full.txt
python run.py --prenatal --cohort prenatal_cohort_merged.csv --output-dir prenatal_results
```

This uses the bundled frozen scorer and regenerates the prenatal analyses and plots. Compare `statistics.json` and the aggregate CSVs with [the saved outputs](../data/reference/prenatal/). The output directory also contains family-level scores and should be kept local.

## Run the full pipeline

```shell
python run.py --full --cohort cohort.csv --output-dir full_results
```

This runs the benchmark, trains the disease-level scorer and uses the newly trained scorer for the prenatal analysis. The three workflows write to separate subdirectories. Some package versions from the original scorer and prenatal runs were unrecorded. New runs record their commands, source hashes and Python version in `workflow_execution.json`.

## Code Ocean

Place `run`, `run.py`, `Dockerfile`, `requirements.txt`, `src/` and `docs/` in Code. Copy the contents of `data/` to Data. Use Python 3.11 and `requirements.txt`, and select `run` as the run file. Outputs go to `/results`.
