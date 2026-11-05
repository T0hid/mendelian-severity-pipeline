# Training inputs

The benchmark and scorer use the ten files below. Run `python run.py --check-inputs` to unpack and verify them. Paths are relative to the training directory supplied with `--training-data`.

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

Nine checksums match the historical run. The supplied mapping contains all saved benchmark pairs. Its historical checksum is unavailable. The checksums of the bundled files are in [training_inputs_metadata.json](../data/training_inputs_metadata.json).

`DOMAIN_CARRIER_FEATURES.csv` is optional because its features were filtered out of the candidate sets. If supplied, the file is checked against its recorded checksum.

The fitted scorer includes its ontology and vectoriser. Retraining uses the bundled `hp.obo`.

Prenatal analysis takes an authorised local cohort. See the [reproduction guide](../docs/REPRODUCE.md) for the scorer and held-out prediction inputs.
