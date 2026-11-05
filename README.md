# Mendelian severity pipeline

This repository contains the gene–disease severity benchmark, the disease-level phenotype scorer and the prenatal analysis.

[![Demo](https://github.com/T0hid/mendelian-severity-pipeline/actions/workflows/demo.yml/badge.svg)](https://github.com/T0hid/mendelian-severity-pipeline/actions/workflows/demo.yml)

## Run the demo

Use Python 3.11 in a fresh environment. Run these commands from the repository folder.

```shell
python -m pip install -r requirements.txt
python run.py
```

Open `results/index.html` to view the study figures and download the source tables. The demo scores two synthetic profiles with the trained model and checks the saved benchmark results. See `results/predictions.csv` and `results/verification.json`.

To view a GitHub run, open [Run pipeline demo](https://github.com/T0hid/mendelian-severity-pipeline/actions/workflows/demo.yml) and select a successful run on `main`. Download `severity-demo-results`, extract it and open `index.html`.

## Score HPO profiles

Use [example_profiles.csv](data/example_profiles.csv) as a template. The output includes a severity score on a 0–1 scale, the severity class and the probability of each class.

```shell
python run.py --input profiles.csv --output-dir profile_results
```

## Export figures

```shell
python run.py --figures --output-dir figure_results
```

Open `figure_results/index.html`. The [figure map](docs/OUTPUT_MAP.csv) lists the data and plotting functions for each panel.

## Reproduce the benchmark

```shell
python -m pip install -r src/requirements-full.txt
python run.py --check-inputs
python run.py --benchmark --output-dir benchmark_results
```

The benchmark uses the 5 November 2026 configuration, with 100 gene-block permutations per feature set. For scorer training and the prenatal analysis, see the [reproduction guide](docs/REPRODUCE.md). Prenatal analysis requires an authorised local cohort.

[Installation](docs/INSTALL.md) · [Inputs and outputs](docs/INPUT_OUTPUT.md) · [Code availability](docs/CODE_AVAILABILITY.md) · [Citation](CITATION.cff)
