# Release status and unresolved items

Version 1.2.0 includes a runnable public entry point, the frozen phenotype model, all ten required original training inputs and GitHub demo automation. Nine input hashes match the saved run. The author supplied the rebuilt mapping on 5 November 2026; its schema and membership of all saved benchmark pairs were checked, and its release hash is recorded. The historical run did not record that mapping hash, so historical byte identity cannot be claimed. Numerical source and existing results are unchanged.

The quick GitHub workflow performs input integrity checks, regression tests and frozen-model/saved-result verification. Full retraining and original prenatal inference are separate, expensive workflows. The original prenatal cohort is governed and is excluded from the public repository.

## Benchmark permutation mismatch

The supplied v4.9.1 script, manifest, log and result table record **one** permutation per scenario. The current manuscript describes **99**. It also reports adjusted P = 0.001 for this analysis, while its own plus-one formula and 99 permutations imply a minimum unadjusted P = 0.01. These statements need reconciliation with the actual run. The existing code/results were not silently changed and no permutation analysis was rerun.

Do not substitute permutation results from the differently versioned older benchmark folder. A new `--permutations 99` run would generate new outputs that need consistent reporting. Confirm the intended run and actual permutation count before final manuscript submission.

## Current prenatal source versus supplied saved outputs

The current source includes term-count baseline analyses, but the supplied prenatal results do not contain:

- `term_count_baseline_rank_4group.csv`
- `term_count_baseline_odds_4group.csv`

Those functions are preserved. These two new outputs were not fabricated or generated during cleanup. Existing scorer/outcome results were checked without running new inference. A future full prenatal run of the current script will also produce these tables. Their absence is reported by `verify_clinical.py`.

## Existing interpretation diagnostics

The benchmark's ordinal SHAP diagnostics include boundary-crossing/gap values and an S2 maximum additivity error of about 0.14076. Replaying saved arrays confirms their aggregates, not the exact equivalence of boundary-summed explanations to repaired/calibrated predictions. Check interpretation wording against the original diagnostic table before describing these as exact explanations of final predictions.

The scorer's saved CSV contains combined-channel class probabilities and both channels' expected severities. Combined held-out probability metrics and both channels' available expected-severity metrics were replayed. Phenotype-channel log-loss/probability calibration metrics could not be independently reconstructed from that CSV. The refitted deployment model was not used to regenerate held-out metrics, because it was trained on all diseases.

## Public deposit scope

This repository includes the code for all three workflows, disease-level review predictions, aggregate prenatal tables, a disease-trained deployment model and invented demonstration profiles. Full-resolution publication assets remain in the separate complete submission archive. Individual prenatal input/output rows and family identifiers are excluded. Access to the original prenatal cohort remains through the manuscript's corresponding-author/governance process.

The source does not supply an existing software licence. Select and add the licence authorised for this project before public release. No licence grant, repository URL or DOI was invented. The original trained scorer/prenatal environments were not fully recorded; uncaptured dependency versions are not claimed as pinned. A hosted Code Ocean run has not been verified here; input checks are recorded in `src/validation/input_audit.json`, and each GitHub run retains its execution log and result artifact.
