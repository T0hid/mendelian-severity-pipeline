"""Run-wide objects shared by the benchmark modules: the log, the result store and the
feature table container."""

import hashlib
import os
import pickle
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import PIPELINE_VERSION, Config


class Logger:
    """Prints progress and keeps a copy that is written to ``pipeline_log.txt``."""

    def __init__(self, log_file: str):
        self.log_file = log_file
        self.logs: List[str] = []
        self.start_time = datetime.now()

    def log(self, msg: str, level: str = "INFO", print_msg: bool = True):
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [{level}] {msg}"
        self.logs.append(line)
        if print_msg:
            print(line)

    def section(self, title: str):
        self.log("=" * 70)
        self.log(title)
        self.log("=" * 70)

    def subsection(self, title: str):
        self.log("-" * 50)
        self.log(title)

    def save(self):
        self.log(f"Total runtime: {datetime.now() - self.start_time}")
        with open(self.log_file, "w", encoding="utf-8") as handle:
            handle.write(f"Severity benchmark {PIPELINE_VERSION}\n\n")
            handle.write("\n".join(self.logs) + "\n")


logger = Logger(os.path.join(Config.OUTPUT_DIR, "pipeline_log.txt"))


class ResultsCollector:
    """Test-set metrics, predictions and feature importances of every fitted candidate."""

    def __init__(self):
        self.results = defaultdict(lambda: defaultdict(dict))
        self.predictions = defaultdict(dict)
        self.feature_importances = defaultdict(dict)
        self.shap_values = defaultdict(dict)

    def add_result(self, scenario, stratum, model, metrics):
        self.results[scenario][stratum][model] = metrics

    def add_predictions(self, key, y_true, y_pred, y_prob):
        self.predictions[key] = {
            "y_true": np.array(y_true),
            "y_pred": np.array(y_pred),
            "y_prob": np.array(y_prob),
        }

    def add_feature_importance(self, key, importance_df):
        self.feature_importances[key] = importance_df

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for scenario, strata in self.results.items():
            for stratum, models in strata.items():
                for model, metrics in models.items():
                    rows.append({"Scenario": scenario, "Stratum": stratum, "Model": model, **metrics})
        return pd.DataFrame(rows)

    def save(self, filepath: str):
        self.to_dataframe().to_csv(filepath, index=False)
        with open(filepath.replace(".csv", ".pkl"), "wb") as handle:
            pickle.dump(
                {
                    "results": dict(self.results),
                    "predictions": dict(self.predictions),
                    "feature_importances": dict(self.feature_importances),
                    "shap_values": dict(self.shap_values),
                },
                handle,
            )


results_collector = ResultsCollector()


class DataContainer:
    """The assembled feature table, labels, gene groups and the candidate columns of each feature set."""

    def __init__(self):
        self.full_df: Optional[pd.DataFrame] = None
        self.y: Optional[np.ndarray] = None
        self.groups: Optional[np.ndarray] = None
        self.train_idx: Optional[np.ndarray] = None
        self.test_idx: Optional[np.ndarray] = None
        self.metadata: Optional[pd.DataFrame] = None
        self.scenario_cols: Dict[str, List[str]] = {}
        self.hpo_binary_cols: List[str] = []
        self.mondo_cols: List[str] = []
        self.scenario_static_drops: Dict[str, List[Tuple[str, str]]] = {}
        self.severity_hpo_candidates: Dict[str, List[str]] = {}
        self.hpo_column_term_names: Dict[str, str] = {}
        self.feature_family_lookup: Dict[str, str] = {}
        self.inheritance_proxy_drops_S1: List[str] = []


# Stores filled during a run.
HPO_COLUMN_TERM_NAMES: Dict[str, str] = {}  # HPO feature column -> HPO term name
_high_mi_audit_records: List[Dict[str, Any]] = []
_test_evaluation_registry: Dict[str, int] = defaultdict(int)  # guards the single test evaluation
_fold_artifacts: Dict[str, Dict[str, Any]] = {}  # fitted representation of each primary feature set
_generalisation_results: Dict[str, Any] = {}


def _sha256_file(path: str) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_fresh_output_directory() -> None:
    """Refuse to write into an output folder that already holds files."""
    if os.path.isdir(Config.OUTPUT_DIR) and any(files for _, _, files in os.walk(Config.OUTPUT_DIR)):
        raise FileExistsError(f"The output folder is not empty: {Config.OUTPUT_DIR}")
