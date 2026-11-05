# Reproduction commands

## Demonstration and verification

```shell
python -m pip install -r requirements.txt
python run.py --output-dir review_results
```

Open `review_results/index.html`. This checks the saved results and original computational definitions, predicts the invented examples, exports original figures/source tables and renders benchmark plots from saved numeric inputs. It performs no training or new inferential analysis.

## Saved-data plotting

```shell
python run.py --figures --output-dir figure_results
```

Original plotting functions render Figures 2–4 and Extended Data Figure 2. PCA reads the saved projected coordinates; SHAP reads saved arrays. Extended Data Figure 1 uses its saved descriptive tables in a compact layout. Original PNG/PDF/SVG/TIFF exports are separate, preserving them despite environment-dependent font differences.

Figure 5's original composite and public source tables are exported. Its individual-family panels use the original prenatal workflow with authorised family data. Figure 1 is an independently prepared conceptual schematic with no analytical generation command.

## Benchmark

```shell
python -m pip install -r src/requirements-full.txt
python run.py --check-inputs
python run.py --benchmark --output-dir benchmark_results
```

The scenarios, model selection, evaluation, sensitivity analyses, explanations and publication rendering remain in `src/pipeline.py` and `src/publication_figures_v2.py`. The supplied configuration uses one permutation. An explicit override creates a separate run:

```shell
python run.py --benchmark --permutations 99 --output-dir benchmark_99_results
```

The frozen reference snapshot and existing output folders are preserved.

## Disease-level scorer

```shell
python run.py --train-scorer --output-dir scorer_results
```

This trains/evaluates the separate disease-level scorer and saves its refitted deployment model. It is distinct from the benchmark S3 model.

## Original prenatal workflow

After installing the full requirements, use an authorised local cohort:

```shell
python src/run_study.py prenatal --cohort PRIVATE_COHORT.csv --model data/model.pkl --scorer-heldout data/reference/scorer/held_out_test_predictions.csv --output-dir prenatal_results --score-channel phenotype
```

The schema and aliases are defined by `load_cohort` in the prenatal source. Family data remain in the authorised environment. Public outputs contain aggregate prenatal tables.

## All three workflows

```shell
python run.py --full --cohort PRIVATE_COHORT.csv --output-dir full_results
```

The order is benchmark, scorer training, then prenatal application using that newly trained scorer and its held-out predictions. Workflows write separate directories and record commands/source hashes.

`src/FUNCTION_INVENTORY.csv` and `src/CLINICAL_FUNCTION_INVENTORY.csv` identify retained definitions. Checks cover 261 original functions and 20 classes. New reviewer interfaces are separate from the numerical modules.

Saved-result replay checks available predictions, tables and arrays. It does not claim numerical identity for future training under every platform/version. The scorer/prenatal historical environment was not completely recorded. All ten public training inputs are bundled; individual prenatal data follow the existing access arrangements.
