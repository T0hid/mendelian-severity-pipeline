# Software release status

Version 1.3.1 packages the completed 5 November 2026 benchmark and the corresponding publication exports. The frozen disease-level scorer and prenatal aggregate files are retained.

## Verified scope

The demo checks ten bundled public inputs, original computational definitions, saved-result snapshots and frozen-model predictions for invented profiles. Plotting reads saved inputs without training, SHAP recalculation or PCA refitting. Original images remain separate from newly rendered plots.

The three original research workflows remain executable. The demo runs from saved outputs and the frozen scorer. The original prenatal cohort is excluded; its authorised local command is documented.

Checks cover 261 original functions and 20 classes. The benchmark configuration and its saved outputs were aligned with the run recorded in the benchmark manifest. Nine input hashes match the saved run. The rebuilt mapping includes all saved benchmark pairs and has a release checksum; its historical hash was not recorded.

The saved benchmark records 100 successful gene-block permutations per feature set, with adjusted P = 0.00990099 for each. The reviewer demo verifies these counts and the adjustment. Explicit overrides create separate runs. The frozen prenatal snapshot lacks two term-count tables produced by the current source; the verifier reports their absence without fabricating outputs.

## Figure coverage

Figures 2–4 and Extended Data Figure 2 can be rendered by original plotting functions from saved data. Extended Data Figure 1 has a compact replay of its count tables. Original exports are bundled. Figure 5's composite is included; its family-level panels require authorised data for replotting. Figure 1 is an author-produced conceptual schematic outside the analytical code.

## Final metadata

The rights holder must supply the authorised licence and permanent DOI. See `docs/LICENSING.md` and `docs/CODE_AVAILABILITY.md`. No licence, DOI, independent colleague test or hosted Code Ocean execution is claimed without evidence. Some historical scorer/prenatal dependency versions were not recorded.
