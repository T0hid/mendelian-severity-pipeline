"""Missing values and preprocessing.

All statistics are learned on the training pairs of a fold and applied unchanged to the
validation and test pairs:

1. features missing in more than 50% of training pairs are removed;
2. remaining missing values are replaced by the training median;
3. features with variance below 1e-10 after imputation are removed;
4. MONDO subclasses must occur in at least five training pairs, and HPO terms in at least two
   training pairs and in no more than 90% of them;
5. each feature is centred on its training median and divided by its interquartile range
   (RobustScaler; only centred when the range is zero).
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from .config import Config
from .runtime import logger


class FoldPreprocessor:
    """Training-fold filters, median imputation and robust scaling for one feature set."""

    def __init__(
        self,
        scenario_name: str,
        hpo_binary_cols: List[str],
        mondo_cols: List[str],
        static_drops: Optional[List[Tuple[str, str]]] = None,
    ):
        self.scenario_name = scenario_name
        self.hpo_binary_cols = list(hpo_binary_cols)
        self.mondo_cols = list(mondo_cols)
        self.kept_cols: List[str] = []
        self.kept_mondo_cols: List[str] = []
        self.kept_hpo_cols: List[str] = []
        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[RobustScaler] = None
        self.final_feature_names: List[str] = []
        self.dropped_features: List[Dict[str, Any]] = [
            {"feature": name, "audit_action": "drop", "drop_reason": reason, "drop_statistic": ""}
            for name, reason in (static_drops or [])
        ]

    def _record_drop(self, feature: str, reason: str, statistic: Any) -> None:
        self.dropped_features.append(
            {"feature": feature, "audit_action": "drop", "drop_reason": reason, "drop_statistic": statistic}
        )

    def fit_transform(
        self, full_df: pd.DataFrame, train_idx: np.ndarray, candidate_cols: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        train_df = full_df.iloc[train_idx]
        candidate_cols = [column for column in candidate_cols if column in full_df.columns]
        if not candidate_cols:
            raise ValueError(f"{self.scenario_name}: no candidate features are available.")
        train_subset = train_df[candidate_cols].apply(pd.to_numeric, errors="coerce")

        # 1. Missingness.
        missing_fraction = train_subset.isna().mean()
        keep = missing_fraction[missing_fraction <= Config.MISSING_DROP_THRESHOLD].index.tolist()
        for column in candidate_cols:
            if column not in keep:
                self._record_drop(
                    column, "high_missing_training_fold", float(missing_fraction.get(column, 1.0))
                )
        if not keep:
            raise ValueError(f"{self.scenario_name}: all features failed the missingness filter.")

        # 2-3. Variance after median imputation.
        variances = train_subset[keep].fillna(train_subset[keep].median(axis=0, skipna=True)).var(axis=0)
        keep_after_variance = variances[variances >= Config.VAR_DROP_THRESHOLD].index.tolist()
        for column in keep:
            if column not in keep_after_variance:
                self._record_drop(column, "low_variance_training_fold", float(variances.get(column, 0.0)))
        keep = keep_after_variance
        if not keep:
            raise ValueError(f"{self.scenario_name}: all features failed the variance filter.")

        # 4a. MONDO subclasses: at least five training pairs.
        mondo_in_keep = [column for column in self.mondo_cols if column in keep]
        kept_mondo: List[str] = []
        if mondo_in_keep:
            positive_counts = (train_df[mondo_in_keep].apply(pd.to_numeric, errors="coerce") > 0).sum(axis=0)
            kept_mondo = positive_counts[positive_counts >= Config.MONDO_MIN_TRAIN_COUNT].index.tolist()
            for required_column in ["feat_mondo_depth", "feat_mondo_n_classes", "has_mondo_record"]:
                if required_column in keep and required_column not in kept_mondo:
                    kept_mondo.append(required_column)
            for column in mondo_in_keep:
                if column not in kept_mondo:
                    self._record_drop(
                        column, "mondo_low_frequency_training_fold", int(positive_counts.get(column, 0))
                    )
            keep = [column for column in keep if column not in mondo_in_keep] + kept_mondo
        self.kept_mondo_cols = kept_mondo

        # 4b. HPO terms: at least two training pairs and at most 90% of them.
        hpo_in_keep = [column for column in self.hpo_binary_cols if column in keep]
        kept_hpo: List[str] = []
        if hpo_in_keep:
            hpo_training = train_df[hpo_in_keep].apply(pd.to_numeric, errors="coerce")
            positive_counts = (hpo_training.fillna(0) > 0).sum(axis=0)
            max_count = Config.HPO_MAX_TRAIN_FRACTION * len(hpo_training)
            in_band = (positive_counts >= Config.HPO_MIN_TRAIN_COUNT) & (positive_counts <= max_count)
            kept_hpo = positive_counts[in_band].index.tolist()
            for column in hpo_in_keep:
                if column not in kept_hpo:
                    count = int(positive_counts.get(column, 0))
                    reason = (
                        "hpo_high_frequency_training_fold"
                        if count > max_count
                        else "hpo_low_frequency_training_fold"
                    )
                    self._record_drop(column, reason, count)
            keep = [column for column in keep if column not in hpo_in_keep] + kept_hpo
        self.kept_hpo_cols = kept_hpo
        keep_set = set(keep)
        self.kept_cols = [column for column in candidate_cols if column in keep_set]
        if self.kept_hpo_cols:
            logger.log(f"    {self.scenario_name}: retaining {len(self.kept_hpo_cols)} binary HPO columns.")

        # 2 and 5. Median imputation and robust scaling, fitted on the training pairs.
        training_base = train_df[self.kept_cols].apply(pd.to_numeric, errors="coerce")
        self.imputer = SimpleImputer(strategy="median", add_indicator=False, keep_empty_features=True)
        imputed = self.imputer.fit_transform(training_base.to_numpy(dtype=float))
        self.scaler = RobustScaler()
        scaled = self.scaler.fit_transform(imputed)
        self.final_feature_names = list(self.kept_cols)
        return scaled, self.final_feature_names

    def transform(self, full_df: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
        if self.imputer is None or self.scaler is None:
            raise RuntimeError("FoldPreprocessor must be fitted before transform().")
        base = full_df.iloc[idx][self.kept_cols].apply(pd.to_numeric, errors="coerce")
        return self.scaler.transform(self.imputer.transform(base.to_numpy(dtype=float)))

    def save_audit(self, scenario_name: str, runtime_mi_drops: Optional[List[Dict]] = None):
        """Every feature removed for this feature set and why (Extended Data Fig. 2f)."""
        rows = list(self.dropped_features) + list(runtime_mi_drops or [])
        if not rows:
            return
        for row in rows:
            row.setdefault("audit_action", "drop")
            row.setdefault("drop_reason", "unspecified")
            row.setdefault("drop_statistic", "")
        dataframe = pd.DataFrame(rows)
        safe_scenario = re.sub("[^A-Za-z0-9_.-]+", "_", str(scenario_name))
        dataframe.to_csv(
            os.path.join(Config.OUTPUT_DIR, "results", f"feature_audit_{safe_scenario}.csv"), index=False
        )
        logger.log(
            f"    Feature audit ({scenario_name}): {len(dataframe)} entries; "
            f"reasons={dataframe['drop_reason'].value_counts().to_dict()}"
        )
