# Cleanup status

Work is on `paper-cleanup`. Nothing is merged into main. The existing `cleanup/reviewer-pipeline` ref prevents creating the shorter `cleanup` ref.

## Changes by task

| Task | Diff |
|---|---|
| 1 | Added input sources and reference checksums, linked Zenodo, pinned the full environment, and documented the run order and output directories. |
| 2 | Corrected the publication layout docstring and the S3-to-S2 contrast comment. The permutation count was already 100 without the stale comment. |
| 3 | Renamed the scripts, S1/S2 scenarios, annotation features and output paths. Input preparation creates a byte-identical `gene_annotations.csv` from the original archive member. Reference archives and saved result tables remain unchanged. |
| 4 | Removed patch-history comments and inactive stacking/PCA branches; made organ-criticality exclusion unconditional, matching its previous disabled default. The five wrapper/helper pairs are merged into single functions in the Task 4 follow-up. |
| 5 | Added a reduced main workflow, exact result comparison, comparison tests, updated imports and saved-figure replay, and adapted source checks to the documented names and cleanup edits. |

## Validation

- The complete benchmark smoke test passed in 4 minutes 7 seconds: 400 pairs, 10 permutations, 50 bootstrap replicates and three split seeds. Required-output validation passed and RUN_COMPLETE.json was written.
- All 37 tests pass, including result-comparison and Task 4 source checks.
- All ten bundled input checksums pass.
- Scorer training completed with one candidate configuration and two cross-validation folds. This was a reduced configuration, not the published model-selection run.
- Both the supplied model and the newly trained smoke model load through the prenatal loader and score profiles.
- The demo, saved-result checks and publication-figure replay complete.
- Original and cleaned prenatal scripts produce identical numeric statistics on the available cohort, using 50 bootstrap replicates and 10 permutations.
- The available cohort has 275 input rows and 241 analysed profiles. Both scripts produce rho about 0.195 on that file. The stored reference run starts with 614 rows and analyses 531 profiles. Its rho of 0.383 cannot be reproduced from the smaller supplied file.
- The full 17.5-hour benchmark has not been rerun. Run `python tests/compare_runs.py --previous data/reference --current results` after completing the full workflow.

Tests ran under the available Python 3.12.14 runtime, rather than the reference Python 3.11.15 environment. Use the documented reference environment for the full reproduction run.

`data/cleanup_source_changes.json` records source integrity after the cleanup. These hashes identify the edited source; they do not prove numerical equivalence to a full reference training run. The old fingerprint inventories remain intact. A separate AST inspection, ignoring the documented names and docstrings, identifies changes in Config, source assembly, the fold preprocessor, the unused stacking branch, manifest labels and the main entry point. The other benchmark definitions retain their computational structure under that name mapping.

## Retained or unresolved

- The patient dashboard, optional dimension-model fitting/prediction and separate-severity routing have been removed. Empty model-state fields and existing output columns remain so file schemas stay unchanged. Compatibility loading remains: both supplied and newly trained models fail ordinary loading in a fresh prenatal process because their classes were saved under __main__.
- The repository's reporting module describes itself, before this cleanup, as a replacement produced during earlier repository preparation. It has been retained and renamed, but has not been checked against the author's local original.
- No code licence has been selected. MIT is suggested; a LICENSE file has not been added without that selection.
- The Zenodo DOI is linked as supplied. Its downloadable contents could not be inspected in this session.
- Existing scientific warnings and unrelated bugs have not been repaired.

## Files still needed

| File | Purpose |
|---|---|
| Complete 614-row reference prenatal cohort | Reproduce the saved 531-profile prenatal result and rho 0.383. The supplied 275-row file is sufficient for execution testing. |
| `enhanced_variant_features/DOMAIN_CARRIER_FEATURES.csv` | Optional historical input; its features are excluded from the fitted candidate sets. Not needed for the checked default-input preflight. |
| `build_new_gene_features.py` | Referenced feature builder, absent from the repository; its generated table is included. |
| `build_clingen_v46.py` | Referenced feature builder, absent from the repository; its generated table is included. The author previously asked to ignore the ClinGen build. |

No replacement input files have been invented. The supplied prenatal records were used locally for checks and have not been added to GitHub.

## Task 4 follow-up

All five wrapper pairs are merged; their private base functions are removed. Remaining disabled stacking filters and HPO PCA transform/construction code are removed. Public scorer training accepts only the default phenotype severity-term policy. Severity-term detection remains because its annotations are part of the existing profile output.

Comments and docstrings in the five analysis modules pass checks for patch-history wording, publishing references, the prohibited jargon and non-ASCII comments. Log messages use plain terms while output filenames, columns and serialized identities remain unchanged. Decorative log/report separators are removed.

Reduced scorer retraining matches the previous held-out predictions and model-selection CSVs exactly. Prenatal statistics match the previous run exactly on the same available cohort. The demo and saved-result checks pass. Tests include checks that reject reintroduced wrapper layers and unused options.

The existing token-vocabulary construction iterates Python sets. Separate processes with an unset PYTHONHASHSEED can therefore choose a different order among tied token counts, changing some tree-model results. This pre-existing behaviour is left unchanged under the instruction to preserve known bugs. Before/after numerical checks use the same explicit PYTHONHASHSEED=0 for both versions; this is a verification setting, not a new pipeline default.

The controlled benchmark comparison matches model selection and classification metrics. Strict comparison detects four final-digit differences in RPS_CI_high, RPS, Uncalibrated_RPS and Multiclass_Brier, at most 1.6653345369377348e-16; exact bitwise reproduction is not claimed.

The final wording pass removes residual dated notes, patch markers and incomplete comment fragments. An AST comparison against the preceding Task 4 commit is identical after excluding docstrings and four specified log-message edits. It changes no calculations, options, output names or schemas. The source checks also inspect log literals and dated comments.
