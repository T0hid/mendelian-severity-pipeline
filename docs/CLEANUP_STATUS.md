# Validation status

The pipeline is maintained on `main`.

## Completed checks

- The benchmark smoke test completed in 4 minutes 7 seconds with 400 pairs, 10 permutations, 50 bootstrap replicates and three split seeds. Required-output validation passed.
- All ten bundled input checksums pass.
- Reduced scorer training completed with one candidate configuration and two cross-validation folds. This checks execution, not the full published model-selection run.
- Both the bundled scorer and the newly trained scorer load through the prenatal loader and score profiles.
- The demo, saved-result verification and figure export complete.
- Source fingerprints agree between Python 3.11 and 3.12 by omitting empty generic-parameter fields.

## Numerical comparison

Reduced scorer retraining produces identical held-out predictions and model-selection tables before and after the source cleanup. Saved-model profile predictions also match.

Prenatal statistics match exactly on the available cohort with 50 bootstrap replicates and 10 permutations. That cohort has 275 input rows and 241 analysed profiles, giving Spearman rho about 0.195. The reference run starts with 614 rows and analyses 531 profiles; its rho of 0.383 requires the complete cohort.

The controlled benchmark comparison uses `PYTHONHASHSEED=0` for both versions. Model selection and classification metrics match. Four probability-metric fields differ in their final floating-point digits: `RPS_CI_high`, `RPS`, `Uncalibrated_RPS` and `Multiclass_Brier`. The largest absolute difference is 1.6653345369377348e-16.

Token vocabularies iterate Python sets. With `PYTHONHASHSEED` unset, tied token counts can have different orders between processes and affect tree-model results. The pipeline default remains unchanged.

The full benchmark, which takes about 17.5 hours, has not been rerun after cleanup. Use the reference Python 3.11.15 environment for full reproduction, then run:

```shell
python tests/compare_runs.py --previous data/reference --current results
```

Source hashes in `data/cleanup_source_changes.json` identify the current code. The original fingerprint inventories and result tables are retained. Hash checks establish source identity; numerical equivalence requires a run comparison.

## Compatibility and provenance

The five ontology, assembly, metric and analysis functions each have one implementation. Unused stacking, HPO PCA, patient-dashboard and dimension-model paths are removed. Existing output fields and serialised model identities are retained.

Scorer compatibility loading is required because the bundled and retrained models contain classes saved under `__main__`.

The report generator is a replacement from repository preparation. Equivalence to the original local report generator has not been checked.

A project-wide code licence is pending. The contents of the linked Zenodo archive have not been independently verified against the bundled files.

## Additional inputs

| File | Purpose |
|---|---|
| Complete 614-row reference prenatal cohort | Reproduce the 531-profile prenatal result and rho 0.383. |
| `enhanced_variant_features/DOMAIN_CARRIER_FEATURES.csv` | Optional input whose features are excluded from fitted candidate sets. |
| `build_new_gene_features.py` | Source feature builder; its generated table is bundled. |
| `build_clingen_v46.py` | Source feature builder; its generated table is bundled. |

Individual prenatal records are used locally and are not included in the repository.
