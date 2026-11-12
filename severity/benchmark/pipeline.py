"""Run the complete gene-disease severity benchmark.

The steps follow the Online Methods:

1. assemble the feature table and the four feature sets (``features``);
2. split the pairs into training, validation and test sets (``partition``);
3. for each feature set, tune, select, calibrate and test the model families, and repeat the
   whole workflow for the disease-disjoint and gene-and-disease-disjoint evaluations (``workflow``);
4. bootstrap confidence intervals, paired comparisons and calibration tables (``evaluation``);
5. sensitivity analyses and robustness checks (``sensitivity``);
6. SHAP feature attributions (``interpretation``);
7. figures (``severity.figures``);
8. save the models and the run manifest (``provenance``).
"""

import json
import os
from datetime import datetime
from typing import Dict

from ..figures.render import render_in_pipeline
from .config import PIPELINE_VERSION, RANDOM_SEED, Config
from .evaluation import (
    analyze_error_types,
    compute_ece_table,
    create_publication_table,
    run_bootstrap_all,
    run_paired_scenario_contrasts,
)
from .features import assemble_full_dataframe
from .interpretation import save_severity_shap
from .labels import write_feature_label_dictionaries
from .partition import make_train_test_split
from .provenance import save_all_base_model_bundles, write_pipeline_manifest
from .runtime import (
    DataContainer,
    _fold_artifacts,
    _require_fresh_output_directory,
    logger,
    results_collector,
)
from .selection import write_high_mi_provenance_report
from .sensitivity import (
    run_budget_sensitivity,
    run_disease_overlap_audit,
    run_imbalance_sensitivity,
    run_mi_threshold_sensitivity,
    run_permutation_test,
    run_severity_hpo_term_sensitivity,
    run_split_sensitivity_fixed_configuration,
)
from .workflow import run_experiments, train_naive_baselines


def run_publication_analyses(all_results: Dict, container: DataContainer) -> None:
    """Statistics, sensitivity analyses and checks reported in the manuscript."""
    logger.section("Test-set statistics and sensitivity analyses")
    train_naive_baselines(container)  # Table 1, baselines
    bootstrap_df = run_bootstrap_all(all_results)  # Table 1 and Fig. 2d confidence intervals
    analyze_error_types(all_results)  # Fig. 3i, Supplementary Table 7
    compute_ece_table(all_results)  # Fig. 3j, Supplementary Table 8
    create_publication_table(all_results, bootstrap_df)  # Table 1, Supplementary Table 5
    run_split_sensitivity_fixed_configuration(container, all_results)  # Extended Data Fig. 2a
    run_mi_threshold_sensitivity(container, all_results)  # Extended Data Fig. 2b
    run_budget_sensitivity(container, all_results)  # Extended Data Fig. 2d
    run_permutation_test(container, all_results)  # Extended Data Fig. 2c
    run_paired_scenario_contrasts(all_results, container)  # Fig. 2f, Supplementary Table 6
    run_disease_overlap_audit(container)  # Supplementary Table 15
    run_imbalance_sensitivity(container, all_results)  # Extended Data Fig. 2e
    run_severity_hpo_term_sensitivity(container, all_results)  # Supplementary Table 13
    write_high_mi_provenance_report()  # Supplementary Table 14


def main() -> None:
    """Run the benchmark end to end. Set paths first with ``Config.configure``."""
    _require_fresh_output_directory()
    for sub in ("", "models", "results", "per_disease_explanations"):
        os.makedirs(os.path.join(Config.OUTPUT_DIR, sub), exist_ok=True)
    logger.log_file = os.path.join(Config.OUTPUT_DIR, "pipeline_log.txt")
    try:
        logger.section(f"Severity benchmark {PIPELINE_VERSION}")
        container = assemble_full_dataframe()
        write_feature_label_dictionaries(container)
        make_train_test_split(container)
        all_results = run_experiments(container)
        save_all_base_model_bundles(container, all_results)
        run_publication_analyses(all_results, container)
        save_severity_shap(container, all_results)
        results_collector.save(os.path.join(Config.OUTPUT_DIR, "results", "all_results.csv"))
        render_in_pipeline(
            all_results, container, results_collector, _fold_artifacts, Config.OUTPUT_DIR, RANDOM_SEED
        )
        write_pipeline_manifest(container, all_results)
        with open(os.path.join(Config.OUTPUT_DIR, "RUN_COMPLETE.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "pipeline_version": PIPELINE_VERSION,
                    "completed_at": datetime.now().isoformat(),
                    "status": "complete",
                },
                handle,
                indent=2,
            )
        logger.section("Severity benchmark complete")
    finally:
        logger.save()
    print(f"Outputs written to: {Config.OUTPUT_DIR}")
