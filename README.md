# Predicting severity in Mendelian disorders

Code, data and trained models for the manuscript "Development and evaluation of machine-learning models for Mendelian disease severity".

The repository has three parts, in the order of the manuscript:

| Part | Manuscript | Code |
|---|---|---|
| Gene-disease severity benchmark: four feature sets, four model families, 6,298 gene-disease pairs | Figs 1-4, Table 1, Extended Data Figs 1-3, Supplementary Tables 1-16, 23, 24, 27, 28 | `severity/benchmark`, `severity/figures` |
| Disease-level phenotype scorer: 5,970 diseases, 1,194 held out | Fig. 5a,b, Supplementary Tables 17-19 | `severity/scorer` |
| Application to a prenatal cohort | Fig. 5c-f, Supplementary Tables 20-22, 25, 26 | `severity/prenatal` |

## Install

Python 3.11 is required. The versions in `requirements.txt` are the ones used for the published run.

```shell
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Or with Docker:

```shell
docker build -t severity .
docker run --rm -v "$PWD:/work" severity python run.py verify
```

Installation takes a few minutes. The published run was made on Linux (x86-64) with Python 3.11.15. No GPU is needed.

## Check the published results (minutes)

These commands use the models and results saved from the run reported in the paper. Nothing is retrained. Each writes to a new folder under `outputs/`.

| Command | What it does | Time | Output |
|---|---|---|---|
| `python run.py verify` | Rebuilds the feature table from the raw inputs, draws the split again, applies the four selected models to the 1,261 test pairs with and without their calibrators, and compares the results with the saved predictions, with Table 1 and with the values before calibration (Supplementary Table 5). It then recomputes the tables that the run derives from its stored test predictions and compares them with the saved ones | about 20 min, 6 GB RAM | `outputs/verify/verification.json`, `verification_metrics.csv`, `extended_data_fig_3_*.csv`, `table_1_before_after_calibration.csv`, `recomputed/` |
| `python run.py figures` | Redraws Figs 2-4, Extended Data Figs 1-3 and Table 1 from the saved results | under 1 min | `outputs/figures/main`, panel data in `outputs/figures/figure_data` |
| `python run.py prenatal` | Applies the saved scorer to the public part of the prenatal cohort and draws Fig. 5. Add `--exclude-studies PED_cohort` to leave out the genomic-autopsy cohort | under 1 min | `outputs/prenatal` |
| `python run.py score --input my_profiles.csv` | Scores your own HPO profiles (columns `sample_id`, `hpo_terms` separated by `;`) | seconds | `outputs/scores.csv` |

`verify` ends with `PASSED` when:

- the rebuilt feature inventory (Supplementary Table 1) and the curation tables written while building it are identical to the saved ones, and the split matches pair for pair;
- the predicted probabilities match the saved ones, and all 28 Table 1 values (seven metrics for four models) and the 20 values before calibration (five metrics for four models, Supplementary Table 5) are reproduced to within 0.00001;
- the confusion matrices before and after calibration match the saved Extended Data Fig. 3 data;
- the tables computed from the stored test predictions are reproduced to within 0.000000001: metrics before and after calibration (`raw_vs_calibrated.csv`), the moderate/severe boundary (`moderate_severe_cutoff*.csv`, Supplementary Table 27), the comparison by number of recorded HPO terms (`hpo_term_count_s2_vs_s3.csv`, Supplementary Table 28) and the disease-disjoint and gene-and-disease-disjoint evaluations (`generalisation_evaluations.csv`, `generalisation_paired_contrasts.csv`; Table 1b, Supplementary Table 16).

The paper reports three decimals; the differences seen in practice are below 0.000001 and come from floating-point rounding, which differs slightly between operating systems.

The prenatal cohort in the paper has 531 analysed families. 290 of them come from an in-house cohort that cannot be shared, so `data/prenatal_cohort_public.csv` holds only the published cases (275 rows, 241 analysed families). Results from the public subset therefore differ from the paper (for example, Spearman's rho is 0.201 rather than 0.385). Applied to these cases, the saved scorer gives the same score for every family as in the full analysis. The full-cohort results are in `published_run/prenatal`.

The paper also reports the prenatal analysis without the genomic-autopsy cohort, whose findings include post-mortem examinations and so are not all available before the outcome: `python run.py prenatal --exclude-studies PED_cohort`. On the public subset this leaves 60 families. For the full cohort (350 families), Spearman's rho, its interval and the proportional-odds model are the "without genomic-autopsy cohort" rows of `published_run/prenatal/score_vs_counts_rank.csv` and `score_vs_counts_odds.csv`, and the group means and neighbouring-group comparisons are in `published_run/prenatal/without_autopsy_cohort`.

`run.py prenatal` also writes `scorer_identity.json` (scorer file, SHA-256, version and training size) and compares the score with a count of recorded HPO terms and a count of affected organ systems, reports termination separately, analyses continued pregnancies only, and repeats the analysis without the live births whose NICU status was not recorded. On the public subset these tables are limited, because NICU admission was recorded only in PreGen; the full-cohort tables (Supplementary Tables 25 and 26) are in `published_run/prenatal`.

Fig. 5a,b and Supplementary Tables 17-19 come from scorer training. The saved scorer (`published_run/scorer/model.pkl`) is the final model refitted on all 5,970 diseases; the held-out predictions were made by the model fitted on the 4,776 development diseases, so reproducing them requires `train-scorer`. `run.py prenatal` redraws Fig. 5a,b from the saved held-out predictions.

## Rerun everything (hours)

```shell
PYTHONHASHSEED=0 python run.py train-benchmark --output outputs/benchmark   # about 23 h on one CPU workstation
python run.py train-scorer --output outputs/scorer
```

`train-benchmark` repeats feature construction, splitting, grouped three-fold cross-validation, model selection, calibration, the single test evaluation, the tables computed from the stored test predictions, all sensitivity analyses, SHAP and the figures. The settings are those of the published run (`severity/benchmark/config.py`). A rerun rebuilds the same feature table and the same split, but model scores may differ slightly from the published values. The saved models in `published_run` are the ones behind the reported numbers, and `verify` checks them.

## Where each result comes from

Feature sets have short internal names in the code and output files:

| Paper | Code |
|---|---|
| S1 gene-level | `S1_Mechanistic` |
| S1b disease-context | `S1b_Disease_Context` |
| S2 combined | `S2_Mech_Pheno` |
| S3 phenotype-only | `S3_Phenotype` |
| ordinal learner | `OrdinalDecomp` |

Figures:

| Figure | Function | File after `run.py figures` |
|---|---|---|
| Fig. 1 | schematic, not generated by code | |
| Fig. 2 | `severity/figures/panels.py: figure_2_model_comparison` | `main/figure_2_model_comparison.*` |
| Fig. 3 | `figure_3_prediction_calibration` | `main/figure_3_prediction_and_calibration.*` |
| Fig. 4 | `figure_4_interpretation` | `main/figure_4_interpretation_and_biology.*` |
| Fig. 5 | `severity/prenatal/analysis.py: build_figure` | `outputs/prenatal/prenatal_severity_figure.*` |
| Extended Data Fig. 1 | `extended_data_fig_1_cohort` | `main/extended_data_fig_1_cohort_composition.*` |
| Extended Data Fig. 2 | `extended_data_fig_2_robustness` | `main/extended_data_fig_2_robustness_and_audits.*` |
| Extended Data Fig. 3 | `extended_data_fig_3_calibration` | `main/extended_data_fig_3_calibration_confusion.*` |
| Table 1 | `table_1_headline_metrics` | `main/table_1_headline_metrics.csv` |

The numbers behind every panel are in `figure_data`. Those files keep the numbering used while drafting: `figure_1*` is Extended Data Fig. 1, `figure_4*` is Extended Data Fig. 2 and `figure_5*` is Fig. 4; `extended_data_fig_3*` is Extended Data Fig. 3.

Supplementary Tables. Benchmark files are in `published_run/benchmark/results` (and in `results/` of a new run), scorer files in `published_run/scorer`, prenatal files in `published_run/prenatal`.

| Table | Content | File | Code |
|---|---|---|---|
| 1 | Candidate feature inventory | `feature_provenance_table.csv` | `benchmark/features.py` |
| 2 | Shared HPO term limit | `shared_hpo_budget_training_cv_selection.csv` | `benchmark/workflow.py` |
| 3 | Same HPO terms in S2 and S3 | `shared_hpo_feature_audit.csv` | `benchmark/workflow.py` |
| 4 | Validation results, all model families | `model_selection_validation.csv`, `nested_cv_tuning_*.csv` (grouped three-fold cross-validation) | `benchmark/workflow.py` |
| 5 | Test results, bootstrap CIs, naive baselines, values before calibration | `publication_table.csv`, `bootstrap_cis.csv`, `naive_baselines.csv`, `raw_vs_calibrated.csv` | `benchmark/evaluation.py` |
| 6 | Paired comparisons between feature sets | `paired_scenario_contrasts.csv` | `benchmark/evaluation.py` |
| 7 | Correct, adjacent and non-adjacent predictions | `error_decomposition.csv`, `error_decomposition_per_class.csv` | `benchmark/evaluation.py` |
| 8 | Calibration by class, before and after calibration | `calibration_metrics.csv`, `calibration_bins.csv`, `calibration_ece.csv` | `benchmark/evaluation.py` |
| 9 | Five further gene-disjoint splits | `split_sensitivity_fixed_configuration*.csv` | `benchmark/sensitivity.py` |
| 10 | Mutual-information threshold | `mi_threshold_sensitivity.csv` | `benchmark/sensitivity.py` |
| 11 | Class-imbalance strategies | `imbalance_sensitivity.csv` | `benchmark/sensitivity.py` |
| 12 | Gene-block permutation tests | `permutation_test.csv` | `benchmark/sensitivity.py` |
| 13 | Severity-term check | `severity_hpo_term_sensitivity.csv`, `severity_hpo_terms_removed*.csv` | `benchmark/sensitivity.py` |
| 14 | High-MI features flagged for review | `high_mi_provenance_flags.csv` | `benchmark/selection.py` |
| 15 | Diseases shared across the split | `disease_overlap_audit.csv` | `benchmark/sensitivity.py` |
| 16 | Disease-disjoint and gene-and-disease-disjoint evaluations | `generalisation_evaluations.csv`, `generalisation_paired_contrasts.csv`, `generalisation_test_predictions_*.csv` | `benchmark/workflow.py`, `benchmark/evaluation.py` |
| 17 | Scorer: held-out performance, curation, configuration | `metadata.json`, `held_out_test_predictions.csv`, `input_curation_summary.csv`, `hpo_curation_summary.csv` | `scorer/model.py` |
| 18 | Scorer: configuration selection | `model_selection_results.csv` | `scorer/model.py` |
| 19 | Scorer: feature weights and held-out SHAP | `phenotype_feature_weights.csv`, `combined_feature_weights.csv`, `held_out_ordinal_shap_importance.csv` | `scorer/model.py` |
| 20 | Prenatal: family records and inclusion | `statistics.json` (`audit`) | `prenatal/analysis.py: load_cohort` |
| 21 | Prenatal: association with outcome | `statistics.json` (`correlation`, `proportional_odds`, `proportional_odds_cohort_adjusted`), `mean_by_outcome.csv`, `adjacent_level_separation.csv`, `within_cohort_spearman.csv`, `without_autopsy_cohort/` | `prenatal/analysis.py` |
| 22 | Prenatal: discrimination between outcome groups | `roc_contrasts.csv` | `prenatal/analysis.py` |
| 23 | Engineered feature definitions | | `benchmark/features.py: engineer_features` |
| 24 | Hyperparameter search spaces | | `benchmark/models.py: _parameter_candidates` |
| 25 | Prenatal: score against counts of recorded HPO terms and affected organ systems | `score_vs_counts_rank.csv`, `score_vs_counts_odds.csv` | `prenatal/analysis.py: count_and_outcome_analyses` |
| 26 | Prenatal: termination reported separately, continued pregnancies, unknown NICU status | `termination_reported_separately.csv`, `continued_pregnancies_auc.csv`, `unknown_nicu_sensitivity.csv`, `outcome_labels_by_source.csv` | `prenatal/analysis.py: count_and_outcome_analyses` |
| 27 | Sensitivity, specificity, PPV and NPV at the moderate/severe boundary | `moderate_severe_cutoff.csv`, `moderate_severe_cutoff_paired_contrasts.csv` | `benchmark/evaluation.py: moderate_severe_cutoff` |
| 28 | Combined and phenotype-only models by number of recorded HPO terms | `hpo_term_count_s2_vs_s3.csv` | `benchmark/evaluation.py: hpo_term_count_comparison` |

Equations in the Online Methods and where they are implemented:

| Equation | Code |
|---|---|
| (1) gene-disjoint split objective | `benchmark/partition.py` |
| (2) ordinal decomposition | `benchmark/models.py` |
| (3) class weights | `benchmark/models.py` |
| (4)-(8) disease-level scorer | `scorer/model.py` |
| (9) quadratic weighted kappa, (10) expected calibration error | `benchmark/metrics.py` |
| (11) permutation P value | `benchmark/sensitivity.py` |
| (12) ordinal concordance index | `prenatal/analysis.py: ordinal_concordance` |
| (13) proportional-odds model | `prenatal/analysis.py: proportional_odds` |

## Repository layout

```
run.py                     command-line entry point
severity/
  mapping.py               gene-disease mapping filter shared by the benchmark and the scorer
  benchmark/               feature construction, split, models, evaluation, sensitivity analyses, SHAP
    pipeline.py            runs the benchmark end to end; the order of steps follows the Online Methods
    verify.py              re-scores the test set with the saved models, before and after calibration,
                           and recomputes the tables derived from the stored test predictions
  figures/                 Figs 2-4, Extended Data Figs 1-3, Table 1
  scorer/                  disease-level phenotype scorer
  prenatal/                prenatal cohort analysis and Fig. 5
data/                      training tables, ontologies, public prenatal cohort, example profiles
published_run/             saved outputs of the run reported in the paper
tests/                     quick tests run by GitHub Actions
```

## Data

`data/training_tables.zip` and `data/ontologies.zip` hold every input of the benchmark and the scorer; `run.py` unpacks them into `data/inputs` and checks each file against `data/checksums.json`. The ontology releases are recorded in `published_run/benchmark/pipeline_manifest.json`. How the 6,298 gene-disease pairs and their severity labels were obtained from 8,467 candidate pairs is described under "Data and severity labels" in the Online Methods; `gene_disease_mapping_HPO_rebuilt.csv` records the decision for every candidate pair. Feature sources are listed in Supplementary Table 1. `final_gnn_dataset.csv` holds gene-level annotation terms; despite its name, no graph neural network is used.

`published_run/benchmark/models` holds the four validation-selected models, compressed with xz. `published_run/benchmark/results/all_results.pkl` holds their test-set predictions and metrics.

## Tests

```shell
python -m pytest tests
```

The tests check that the published split is gene-disjoint and has the reported sizes, that the scorer's disease split is reproduced from the training table, that the saved scorer returns valid probabilities, that the saved Extended Data Fig. 3 data agree with Fig. 3 and Table 1, that the prenatal analysis gives the expected counts and Spearman's rho on the public cohort, with and without the genomic-autopsy cohort, and that the moderate/severe boundary and HPO term-count tables are computed as described. GitHub Actions runs them, together with `run.py verify`, `run.py figures`, `run.py prenatal` and `run.py score`, on every push, and keeps the outputs as a downloadable artifact.

## Citation

See `CITATION.cff`.
