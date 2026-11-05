# Exact input files

These are the original inputs for full research reproduction. Demo and CSV prediction need none of them. All ten required inputs are bundled in two ZIP archives under `data/`. Paths are relative to `--training-data` in `python run.py --full`. Known hashes are recorded in `data/reference/input_manifest.json`; scorer-specific hashes are also in `data/reference/scorer/metadata.json`.

| File | Used by | Supplied in |
|---|---|---|
| `final_output_with_severity.csv` | Benchmark and scorer training | `training_tables.zip` |
| `final_gnn_dataset.csv` | Benchmark | `training_tables.zip` |
| `CONSOLIDATED_FEATURES_v2/CORE_MECHANISTIC_FEATURES.csv` | Benchmark | `training_tables.zip` |
| `CONSOLIDATED_FEATURES_v2/EXTERNAL_MECHANISTIC_FEATURES_v4_6.csv` | Benchmark | `training_tables.zip` |
| `CONSOLIDATED_FEATURES_v2/NEW_GENE_LEVEL_FEATURES.csv` | Benchmark | `training_tables.zip` |
| `CONSOLIDATED_FEATURES_v2/MOUSE_LETHALITY_FEATURES.csv` | Benchmark | `training_tables.zip` |
| `CONSOLIDATED_FEATURES_v2/MOUSE_HUMAN_PHENOTYPE_FEATURES.csv` | Benchmark | `training_tables.zip` |
| `hp.obo` | Benchmark and scorer training | `ontologies.zip` |
| `mondo.obo` | Benchmark | `ontologies.zip` |
| `gene_disease_mapping_HPO_rebuilt.csv` | Benchmark and scorer training | `training_tables.zip` |

Run `python run.py --check-inputs` to restore and verify these inputs. The release records every file hash in `data/training_inputs_metadata.json`; nine hashes match the historical run. The newly supplied mapping includes every saved benchmark pair, but its historical hash was not recorded. Its current hash is recorded without claiming historical byte identity.

The saved scorer contains its fitted ontology/vectorizer state and can score new profiles without separately loading `hp.obo`. This does not remove the pinned ontology requirement for retraining.

`DOMAIN_CARRIER_FEATURES.csv` was read for 4,733 genes in the historical run. Its six `var_*` fields and their variant-derived ratios/densities were removed from all candidate sets by the non-biological-feature filter. No variant/domain/PTM-derived fields appear in the final provenance table or selected features. The input checker therefore accepts its absence; if a file is supplied, its recorded hash is still checked. Omitting it can change loading logs and pre-filter audit inventories, but it is not a dependency of the final fitted feature sets. The optional loader is retained for compatibility.

For prenatal execution, supply the original local `prenatal_cohort_merged.csv` (or another authorised cohort following the script schema), the frozen scorer `model.pkl`, the scorer Python module and, optionally, held-out scorer predictions for Figure 5 panels a/b. The packaged snapshot supplies the latter file. The cohort is not part of the public upload.

The full workflow requirements preserve the versions recorded by the benchmark. The original scorer/prenatal outputs did not capture a separate complete environment, including the statsmodels version. Those uncaptured packages are explicitly identified in requirements rather than assigned invented versions. New launcher runs record the command, source hashes and Python version in `workflow_execution.json`; native outputs retain their original metadata.
