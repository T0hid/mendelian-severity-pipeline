# Input data

Data archive: https://doi.org/10.5281/zenodo.24026146. The DOI was supplied by the author; its download contents have not been verified here. Files over 50 MB should be downloaded from that archive.

Bundled inputs are extracted and checked by `python run.py --check-inputs`. Paths below are relative to `data/training/`.

| Input | Source | SHA-256 from the reference run |
|---|---|---|
| `final_output_with_severity.csv` | training_tables.zip | `da7dac61447a4be2e18b69e52063c5cde33996754eb14856e94b77977df28f84` |
| `gene_annotations.csv` | training_tables.zip | `8de956183594d5aa5f0db50d767c87439db3af374372194132b139c5fd469f6a` |
| `CONSOLIDATED_FEATURES_v2/CORE_MECHANISTIC_FEATURES.csv` | training_tables.zip | `1721d82276b26274da1d808205e523b088837dbd1e8cf4fd6dfb47d16d427314` |
| `enhanced_variant_features/DOMAIN_CARRIER_FEATURES.csv` | Not bundled; optional input excluded from fitted feature sets | `fbe2fb0b06e917815f0f492c8b6ecf715541f8f08737519af7232900f57ed786` |
| `CONSOLIDATED_FEATURES_v2/EXTERNAL_MECHANISTIC_FEATURES_v4_6.csv` | training_tables.zip | `9cd15c54eb4e8b70cc44f42dafd4b207682c9408a4d40d3635f51893d1d7590d` |
| `CONSOLIDATED_FEATURES_v2/NEW_GENE_LEVEL_FEATURES.csv` | training_tables.zip | `ae11ccc9bdab89f32ad53cdccd95d05eaa75a1414bd0e4ba72caff3ed4006845` |
| `CONSOLIDATED_FEATURES_v2/MOUSE_LETHALITY_FEATURES.csv` | training_tables.zip | `2c6285e75aaae125e1fd3c7dcd1458f8e0e76cc170aad0954b0915771d12230b` |
| `CONSOLIDATED_FEATURES_v2/MOUSE_HUMAN_PHENOTYPE_FEATURES.csv` | training_tables.zip | `2496de7a66331128eba90a3f2a6d75a638b56811fb1dd2ff8963f34bc89593eb` |
| `hp.obo` | ontologies.zip | Not recorded in the last-run manifest |
| `mondo.obo` | ontologies.zip | Not recorded in the last-run manifest |
| `gene_disease_mapping_HPO_rebuilt.csv` | training_tables.zip | Not recorded in the last-run manifest |

The supplied mapping checksum is recorded separately in `training_inputs_metadata.json`; it is not a checksum from the last run.

`run_report.py` is present. Its docstring identifies it as a reporting replacement from an earlier repository preparation; its provenance has not been verified against the author's local original.

Referenced feature builders `build_clingen_v46.py` and `build_new_gene_features.py` are not bundled. Their generated feature tables are bundled. The other feature-builder names are not recorded in this repository.

`prenatal_cohort_merged.csv` is not bundled. Supply it with `--cohort` for prenatal replay. No replacement data are generated.
