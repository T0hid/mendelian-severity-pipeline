"""Saved models and the run manifest (input checksums, package versions, seeds and settings)."""

import importlib.metadata as importlib_metadata
import json
import os
import pickle
import platform
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from .config import PIPELINE_VERSION, RANDOM_SEED, Config
from .models import _available_candidate_names
from .runtime import (
    DataContainer,
    _fold_artifacts,
    _sha256_file,
    logger,
)
from .workflow import _pick_best_model_per_scenario


def get_git_commit_hash() -> str:
    """Commit of the code that produced the run ("-dirty" if there were uncommitted changes)."""
    here = os.path.dirname(os.path.realpath(__file__))
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=here, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "not_available"
    return f"{commit}-dirty" if status else commit


def _bundle_common_metadata(scenario: str, artifacts: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "scenario": scenario,
        "preprocessor": artifacts["preprocessor"],
        "sel_mask": artifacts["sel_mask"],
        "sel_names": artifacts["sel_names"],
        "selection_budget": artifacts.get("selection_budget", dict(Config.FEATURE_BUDGET)),
    }


def _obo_metadata(path: str) -> Dict[str, Any]:
    metadata = {
        "path": os.path.abspath(path),
        "sha256": _sha256_file(path),
        "data_version": None,
        "date": None,
    }
    if not os.path.exists(path):
        return metadata
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line_index, line in enumerate(handle):
            if line.startswith("data-version:"):
                metadata["data_version"] = line.split(":", 1)[1].strip()
            elif line.startswith("date:"):
                metadata["date"] = line.split(":", 1)[1].strip()
            if line_index > 200:
                break
    return metadata


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except Exception:
        return None


def write_pipeline_manifest(container: DataContainer, all_results: Dict):
    """pipeline_manifest.json: input checksums, ontology releases, selected models and features, seeds, settings."""
    config_dump: Dict[str, Any] = {}
    for key, value in vars(Config).items():
        if key.startswith("_") or callable(value):
            continue
        if isinstance(value, (str, int, float, bool, list, tuple, dict)) or value is None:
            config_dump[key] = value
    inputs = {
        "primary_data": Config.PRIMARY_DATA,
        "gnn_data": Config.GNN_DATA,
        "core_features": Config.CORE_FEATURES,
        "domain_features": Config.DOMAIN_FEATURES,
        "external_features": Config.EXTERNAL_FEATURES,
        "new_gene_features": Config.NEW_GENE_FEATURES,
        "mouse_lethality": Config.MOUSE_LETHALITY_FEATURES,
        "mouse_human_phenotype": Config.MOUSE_HUMAN_PHENOTYPE_FEATURES,
    }
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "run_timestamp": datetime.now().isoformat(),
        "git_commit_hash": get_git_commit_hash(),
        "workflow": {
            "outer_split": "optimised gene-disjoint 65/15/20 train/validation/test split",
            "feature_vocabulary": "annotation terms learned on each training fold only",
            "shared_hpo_budget": "selected on S3 training-only nested CV and used for both S2 and S3",
            "feature_selection": "repeated inside nested grouped training CV",
            "hyperparameter_tuning": "nested grouped training CV",
            "model_selection": Config.MODEL_SELECTION_RULE,
            "calibration": "isotonic or Platt scaling, chosen from validation class counts, after model selection",
            "test_policy": "single evaluation of one frozen model and calibrator per feature set",
            "bootstrap": "gene-cluster bootstrap stratified by severity",
            "permutation": Config.PERMUTATION_MODE,
            "missingness": "left joins keep every pair; missing values imputed with the training-fold median",
            "mi_leakage_policy": "MI is used for selection and review only; exclusions follow prespecified rules",
            "ordinal_explanations": Config.ORDINAL_EXPLANATION_POLICY,
        },
        "target_label_provenance": Config.TARGET_LABEL_PROVENANCE,
        "input_files": {
            label: {"path": os.path.abspath(path), "sha256": _sha256_file(path)}
            for label, path in inputs.items()
        },
        "ontologies": {
            "HPO": _obo_metadata(Config.HPO_OBO_PATH),
            "MONDO": _obo_metadata(Config.MONDO_OBO_PATH),
            "automatic_download_allowed": False,
        },
        "dataset": {
            "n_pairs_total": int(len(container.full_df)),
            "n_genes_total": int(len(np.unique(container.groups))),
            "full_class_distribution": np.bincount(container.y, minlength=Config.NUM_CLASSES).tolist(),
        },
        "selected_models": _pick_best_model_per_scenario(all_results),
        "selected_model_calibration": {
            scenario: {
                "model": model_name,
                "status": all_results[scenario]["all"][model_name].get("calibration_status", "not_attempted"),
                "success": bool(all_results[scenario]["all"][model_name].get("calibration_success", False)),
                "method": all_results[scenario]["all"][model_name].get("calibration_method", "none"),
                "method_attempted": all_results[scenario]["all"][model_name].get(
                    "calibration_method_attempted", "none"
                ),
                "failure_details": all_results[scenario]["all"][model_name].get(
                    "calibration_failure_details", ""
                ),
            }
            for scenario, model_name in _pick_best_model_per_scenario(all_results).items()
        },
        "selected_features": {
            scenario: list(artifact.get("selected_names", []))
            for scenario, artifact in _fold_artifacts.items()
        },
        "scenario_scope_exclusions": {
            "S1_Mechanistic": {
                "organ_criticality_columns_removed": list(Config.ORGAN_CRITICALITY_COLUMNS),
                "inheritance_count_proxy_columns_removed": list(
                    getattr(container, "inheritance_proxy_drops_S1", [])
                ),
                "rationale": "n_inheritance_modes and has_mixed_inheritance summarise a gene's diseases and act as disease-count proxies.",
            },
            "all_scenarios": {
                "duplicate_columns_removed": list(getattr(container, "duplicate_column_drops", [])),
                "rationale": "gene_age is an exact linear function of conservation_score; see removed_input_columns for columns removed at load.",
                "removed_input_columns": dict(Config.REMOVED_INPUT_COLUMNS),
                "core_inheritance_rollups_removed": list(Config.CORE_INHERITANCE_ROLLUP_COLUMNS),
                "inheritance": f"pair-specific, from the primary file column {Config.PAIR_INHERITANCE_COLUMN}",
            },
        },
        "random_seeds": {
            "primary": RANDOM_SEED,
            "split_sensitivity": list(
                range(
                    RANDOM_SEED + Config.SPLIT_SENSITIVITY_SEED_OFFSET,
                    RANDOM_SEED + Config.SPLIT_SENSITIVITY_SEED_OFFSET + Config.N_SEEDS_SENSITIVITY,
                )
            ),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "packages": {
                name: _package_version(name)
                for name in [
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "lightgbm",
                    "shap",
                    "imbalanced-learn",
                    "scipy",
                    "matplotlib",
                ]
            },
        },
        "config": config_dump,
    }
    with open(os.path.join(Config.OUTPUT_DIR, "pipeline_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)


def save_all_base_model_bundles(container: DataContainer, all_results: Dict) -> None:
    """Save individual model bundles only; no unevaluated S3 ensemble is emitted."""
    logger.section("SAVING INDIVIDUAL MODEL BUNDLES")
    model_dir = os.path.join(Config.OUTPUT_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    allowed = _available_candidate_names()
    for scenario, strata in all_results.items():
        artifact = _fold_artifacts.get(scenario)
        if artifact is None:
            continue
        common = _bundle_common_metadata(scenario, artifact)
        common["text_vocabulary"] = artifact.get("text_vocabulary", {})
        for model_name in allowed:
            result = strata.get("all", {}).get(model_name)
            if not result or result.get("model") is None:
                continue
            payload = dict(common)
            payload.update(
                {
                    "model": result["model"],
                    "calibrated_model": result.get("calibrated_model"),
                    "calibrated_models": result.get("calibrated_models"),
                    "model_name": model_name,
                    "validation_metrics": result.get("val_metrics", {}),
                    "test_metrics_calibrated": result.get("metrics")
                    if result.get("test_evaluated")
                    else None,
                    "test_metrics_uncalibrated": result.get("test_metrics_uncalibrated")
                    if result.get("test_evaluated")
                    else None,
                    "selected_on_validation": result.get("selected_on_validation", False),
                    "test_evaluated": result.get("test_evaluated", False),
                    "best_params": result.get("best_params", {}),
                    "calibration_method": result.get("calibration_method", "none"),
                    "imbalance_strategy": Config.IMBALANCE_STRATEGY,
                }
            )
            path = os.path.join(model_dir, f"{scenario}_{model_name}.pkl")
            with open(path, "wb") as handle:
                pickle.dump(payload, handle)
            logger.log(f"  Saved {path}")
