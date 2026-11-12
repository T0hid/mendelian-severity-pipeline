# Data

| File | Content |
|---|---|
| `training_tables.zip` | Inputs of the benchmark and the scorer (below) |
| `ontologies.zip` | `hp.obo` (HPO release 2025-11-24) and `mondo.obo` (MONDO release 2026-03-03), as used for the published run |
| `checksums.json` | SHA-256 of every unpacked file; `run.py` checks them before use |
| `prenatal_cohort_public.csv` | Published prenatal cases (275 rows). The in-house part of the cohort is not shared |
| `example_profiles.csv` | Two synthetic HPO profiles for `run.py score` |

Contents of `training_tables.zip`:

| File | Content |
|---|---|
| `final_output_with_severity.csv` | Gene, MONDO disease, HPO terms and severity label of each gene-disease pair |
| `gene_disease_mapping_HPO_rebuilt.csv` | The rebuilt gene-disease mapping; only pairs marked `KEPT` are used |
| `final_gnn_dataset.csv` | Gene-level annotation terms. No graph neural network is used; the name is kept so that the checksums match the published run |
| `CONSOLIDATED_FEATURES_v2/*.csv` | Gene-level constraint, expression, paralogue and mouse features |

The source of every feature is listed in Supplementary Table 1. The published run also read a variant-level table (`DOMAIN_CARRIER_FEATURES.csv`) whose columns were removed from every feature set; it is not included and is not needed.

`run.py` unpacks both archives into `data/inputs`. To run the package code directly, point `Config.configure` (benchmark) or `--severity-csv` and `--hpo-obo` (scorer) at that folder.

## Prenatal cohort columns

`family_id`, `study`, `hpo_ids` (HPO identifiers separated by `;`) and `pregnancy_outcome` are used. Rows with the same `family_id` are combined into one profile. Outcome labels are mapped to four ordered groups in `severity/prenatal/analysis.py: map_outcome`.

The gene annotation table and the two ontology files are also archived at Zenodo: [10.5281/zenodo.24026146](https://doi.org/10.5281/zenodo.24026146) (CC BY 4.0).
