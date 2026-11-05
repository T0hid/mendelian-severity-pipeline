# Inputs and outputs

One UTF-8 prediction CSV row represents one HPO profile.

| Input column | Requirement | Meaning |
|---|---|---|
| `sample_id` | Required, unique | User-selected identifier |
| `hpo_terms` | Required column | Present HPO IDs separated by semicolons; empty values allowed |
| `absent_hpo_terms` | Optional | Explicitly absent HPO IDs |

Aliases `family_id` and `hpo_ids` remain supported. IDs use `HP:` followed by seven digits. Duplicate HPO IDs within a profile are normalised. Duplicate profile IDs, malformed HPO IDs and extra CSV fields are rejected.

| Prediction column | Meaning |
|---|---|
| `sample_id` | Original identifier |
| `status` | Original scorer status or `insufficient_information` |
| `severity_score` | Phenotype-only expected class divided by three, on 0–1 |
| `severity_class` | Original rounding rule applied to mild, moderate, severe, profound |
| `prob_mild`, `prob_moderate`, `prob_severe`, `prob_profound` | Frozen phenotype-channel probabilities |

Score/class/probability cells stay empty when evidence is insufficient. Finite probabilities and their sum are validated before export. `run_info.json` records the model checksum and profile count.

## Reviewer outputs

| Path below the output directory | Contents |
|---|---|
| `index.html` | Local figure gallery and download links |
| `predictions.csv` | Two synthetic example predictions |
| `verification.json` | Numerical/source-preservation checks |
| `execution.json` | Measured runtime and environment |
| `review_manifest.json` | Asset hashes, exported files and replay scope |
| `publication/benchmark/` | Original benchmark figures and source tables |
| `publication/prenatal_aggregate/` | Original Figure 5 composite and aggregate tables |
| `publication/scorer/` | Saved scorer plots and disease-level tables |
| `publication/source_data/` | One Excel workbook per calculated figure/Extended Data figure |
| `publication/figure_map.csv` | Figure-to-code/data mapping |
| `replotted/` | Plots rendered during this run and panel CSVs |

Each values-only Excel workbook has a source index and the available original CSV tables. There are no formulas, external links or macros. The original CSV bytes remain authoritative and available separately. Original figure exports remain separate from replotted files.

## Training inputs

`data/training_tables.zip` and `data/ontologies.zip` restore the ten files in `src/INPUTS.md`. Hashes are checked before use. The domain-carrier CSV is optional and excluded from required inputs. The supplied rebuilt mapping has a release hash; its historical hash was not recorded.

Individual prenatal records are excluded. Public aggregate outputs and invented examples are supplied. Use the original cohort through `--cohort` within its authorised environment.
