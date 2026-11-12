"""The four model families (Model families, selection and calibration).

Multinomial logistic regression (with its own standardisation), ExtraTrees, LightGBM and the
ordinal learner, which fits one LightGBM classifier per class boundary and converts the three
boundary probabilities to class probabilities with equation (2). Training pairs are weighted
with equation (3); the ordinal learner instead balances the two sides of each boundary.
Oversampling (ADASYN or SMOTE-Tomek) is used only in the imbalance sensitivity analysis.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_SEED, Config
from .metrics import compute_all_metrics, validation_selection_score
from .runtime import logger


def ordinal_cumulative_to_class_probs(cumulative_probs: np.ndarray) -> np.ndarray:
    cum = np.asarray(cumulative_probs, dtype=float)
    expected_cols = Config.NUM_CLASSES - 1
    if cum.ndim != 2 or cum.shape[1] != expected_cols:
        raise ValueError(
            f"Expected cumulative probabilities with shape (n, {expected_cols}); received {cum.shape}."
        )
    cum = np.clip(cum, 0.0, 1.0)
    cum = np.minimum.accumulate(cum, axis=1)
    probs = np.zeros((cum.shape[0], Config.NUM_CLASSES), dtype=float)
    probs[:, 0] = 1.0 - cum[:, 0]
    for k in range(1, expected_cols):
        probs[:, k] = cum[:, k - 1] - cum[:, k]
    probs[:, -1] = cum[:, -1]
    probs = np.clip(probs, 0.0, 1.0)
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return probs / row_sums


def compute_ordinal_sample_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=Config.NUM_CLASSES).astype(float)
    inv_freq = 1.0 / (counts + 1)
    inv_freq = inv_freq / inv_freq.sum() * Config.NUM_CLASSES
    boundary_mult = np.array(Config.ORDINAL_BOUNDARY_MULT)
    combined = inv_freq * boundary_mult
    combined = combined / combined.mean()
    return combined[y]


def make_smote_step(scenario_name: str, y_train: np.ndarray):
    """Oversampler for the mild and moderate classes: ADASYN for S1, SMOTE with Tomek links otherwise."""
    counts = np.bincount(y_train, minlength=Config.NUM_CLASSES)
    largest = max(counts) if len(counts) else 0
    is_mechanistic = (
        ("mechanistic" in scenario_name.lower() or "S1" in scenario_name)
        and "S1b" not in scenario_name
        and ("pheno" not in scenario_name.lower())
    )
    if is_mechanistic:
        targets = {
            0: max(int(largest * 0.35), counts[0] * 3),
            1: max(int(largest * 0.5), counts[1] * 2),
        }
    else:
        targets = {
            0: max(int(largest * 0.6), counts[0] * 5),
            1: max(int(largest * 0.7), counts[1] * 2),
        }
    sampling_strategy = {cls: t for cls, t in targets.items() if counts[cls] < t}
    if not sampling_strategy:
        return None
    min_class_size = min(counts[counts > 0])
    k_neighbors = max(1, min(5, min_class_size - 1))
    try:
        if is_mechanistic:
            return (
                "resample",
                ADASYN(
                    sampling_strategy=sampling_strategy,
                    random_state=RANDOM_SEED,
                    n_neighbors=k_neighbors,
                ),
            )
        else:
            return (
                "resample",
                SMOTETomek(
                    smote=SMOTE(
                        sampling_strategy=sampling_strategy,
                        random_state=RANDOM_SEED,
                        k_neighbors=k_neighbors,
                    ),
                    random_state=RANDOM_SEED,
                ),
            )
    except Exception as e:
        logger.log(f"    Could not build resampler: {e}", level="WARNING")
        return None


def _activate_lgb_subsampling(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Make LightGBM's subsample/bagging fraction operational and reproducible."""
    normalised = dict(parameters)
    try:
        fraction = float(normalised.get("subsample", normalised.get("bagging_fraction", 1.0)))
    except (TypeError, ValueError):
        fraction = 1.0
    if fraction < 1.0:
        normalised.setdefault("subsample_freq", 1)
    else:
        normalised.setdefault("subsample_freq", 0)
    return normalised


def _make_lgb_classifier(seed: int = RANDOM_SEED, **extra):
    base_kwargs = dict(
        objective="multiclass",
        num_class=Config.NUM_CLASSES,
        class_weight=None,
        random_state=seed,
        verbose=-1,
        n_jobs=Config.MODEL_N_JOBS,
        importance_type="gain",
        deterministic=True,
        force_col_wise=True,
        bagging_seed=seed,
        feature_fraction_seed=seed,
        data_random_seed=seed,
    )
    base_kwargs.update(extra)
    base_kwargs = _activate_lgb_subsampling(base_kwargs)
    return lgb.LGBMClassifier(**base_kwargs)


def _make_multinomial_logistic_classifier(seed: int = RANDOM_SEED, **extra):
    """Create a numerically stable multinomial linear baseline.

    The scenario preprocessor already performs training-only imputation and
    robust scaling. A second, model-specific StandardScaler is intentionally
    fitted inside each logistic-regression training fold because optimisation
    is sensitive to residual scale differences and extreme robust-scaled
    values. Keeping this scaler inside the estimator prevents leakage.
    """
    logistic_overrides: Dict[str, Any] = {}
    for key, value in extra.items():
        normalised_key = key.split("logistic__", 1)[1] if key.startswith("logistic__") else key
        logistic_overrides[normalised_key] = value
    base_kwargs = dict(
        solver="lbfgs", C=1.0, fit_intercept=True, max_iter=10000, tol=0.001, random_state=seed
    )
    base_kwargs.update(logistic_overrides)
    logistic = LogisticRegression(**base_kwargs)
    return SkPipeline([("standardize", StandardScaler()), ("logistic", logistic)])


def _fit_multinomial_logistic_classifier(
    model: SkPipeline, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
) -> SkPipeline:
    """Fit the scaled logistic pipeline with correctly routed sample weights."""
    fit_kwargs: Dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["logistic__sample_weight"] = sample_weight
    model.fit(X, y, **fit_kwargs)
    return model


def _get_multinomial_logistic_estimator(model: Any) -> LogisticRegression:
    """Return the fitted logistic step from the model-specific pipeline."""
    if isinstance(model, SkPipeline) and "logistic" in model.named_steps:
        estimator = model.named_steps["logistic"]
        if isinstance(estimator, LogisticRegression):
            return estimator
    if isinstance(model, LogisticRegression):
        return model
    raise TypeError(f"Unexpected multinomial logistic model type: {type(model).__name__}")


def _transform_for_multinomial_logistic(model: Any, X: np.ndarray) -> np.ndarray:
    """Apply the fitted model-specific scaler for coefficient/SHAP analyses."""
    if isinstance(model, SkPipeline) and "standardize" in model.named_steps:
        return np.asarray(model.named_steps["standardize"].transform(X), dtype=float)
    return np.asarray(X, dtype=float)


def _make_extra_trees_classifier(seed: int = RANDOM_SEED, **extra):
    """Create the nonlinear, non-boosting tree-ensemble candidate."""
    base_kwargs = dict(
        n_estimators=500,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=False,
        random_state=seed,
        n_jobs=Config.MODEL_N_JOBS,
    )
    base_kwargs.update(extra)
    return ExtraTreesClassifier(**base_kwargs)


def _fit_ordinal_binary_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scenario_name: str,
    seed: int = RANDOM_SEED,
    fixed_params: Optional[Dict] = None,
) -> Tuple[List[Any], Dict[str, Any]]:
    """Fit ordinal threshold models using training data only."""
    X_fit, y_fit = X_train, y_train
    bin_kwargs: Dict[str, Any] = {
        "objective": "binary",
        "random_state": seed,
        "verbose": -1,
        "n_jobs": Config.MODEL_N_JOBS,
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.5,
        "deterministic": True,
        "force_col_wise": True,
    }
    if fixed_params:
        allowed = {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "num_leaves",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
            "min_child_samples",
            "min_split_gain",
        }
        bin_kwargs.update({k: v for k, v in fixed_params.items() if k in allowed})
    bin_kwargs = _activate_lgb_subsampling(bin_kwargs)
    models: List[Any] = []
    for threshold in range(Config.NUM_CLASSES - 1):
        y_binary = (y_fit > threshold).astype(int)
        threshold_seed = seed + threshold
        threshold_kwargs = dict(bin_kwargs)
        threshold_kwargs.update(
            {
                "random_state": threshold_seed,
                "bagging_seed": threshold_seed,
                "feature_fraction_seed": threshold_seed,
                "data_random_seed": threshold_seed,
            }
        )
        model = lgb.LGBMClassifier(**threshold_kwargs)
        n_pos = int(y_binary.sum())
        n_neg = int(len(y_binary) - n_pos)
        if n_pos > 0 and n_neg > 0:
            sample_weight = np.where(y_binary == 1, n_neg / max(n_pos, 1), 1.0)
        else:
            sample_weight = np.ones(len(y_binary), dtype=float)
        model.fit(X_fit, y_binary, sample_weight=sample_weight)
        models.append(model)
    return (models, bin_kwargs)


def _predict_ordinal_probabilities(models: Sequence[Any], X: np.ndarray) -> np.ndarray:
    cumulative = np.column_stack([model.predict_proba(X)[:, 1] for model in models])
    return ordinal_cumulative_to_class_probs(cumulative)


def _train_candidate_dispatch(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    scenario_name: str,
    seed: int = RANDOM_SEED,
    fixed_params: Optional[Dict] = None,
) -> Optional[Dict[str, Any]]:
    kwargs = dict(
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names,
        scenario_name=scenario_name,
        seed=seed,
        fixed_params=fixed_params,
    )
    if model_name == "MultinomialLogisticRegression":
        return train_multinomial_logistic_regression(**kwargs)
    if model_name == "ExtraTrees":
        return train_extra_trees(**kwargs)
    if model_name == "LightGBM":
        return train_lightgbm(**kwargs)
    if model_name == "OrdinalDecomp":
        return train_ordinal_decomposition(**kwargs)
    raise ValueError(f"Unsupported candidate model: {model_name}")


def train_lightgbm(
    X_train,
    y_train,
    groups_train,
    X_val,
    y_val,
    feature_names,
    scenario_name="default",
    seed: int = RANDOM_SEED,
    fixed_params: Optional[Dict] = None,
):
    """Refit with fixed hyperparameters on the training pairs and score the validation pairs."""
    logger.log("  Training LightGBM candidate (test set unavailable)...")
    best_params = dict(fixed_params)
    X_fit, y_fit = X_train, y_train
    final = _make_lgb_classifier(seed=seed, **best_params)
    final.fit(X_fit, y_fit, sample_weight=compute_ordinal_sample_weights(y_fit))
    val_probability = final.predict_proba(X_val)
    val_prediction = np.argmax(val_probability, axis=1)
    val_metrics = compute_all_metrics(y_val, val_prediction, val_probability)
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": final.feature_importances_}
    ).sort_values("importance", ascending=False)
    logger.log(f"    LGB validation MCC={val_metrics['mcc']:.4f}, QWK={val_metrics['qwk']:.4f}")
    return {
        "model_name": "LightGBM",
        "model": final,
        "calibrated_model": None,
        "val_metrics": val_metrics,
        "validation_selection_score": validation_selection_score(val_metrics),
        "best_params": best_params,
        "val_predictions": {"y_pred": val_prediction, "y_prob": val_probability},
        "importance": importance_df,
        "test_evaluated": False,
    }


def train_multinomial_logistic_regression(
    X_train,
    y_train,
    groups_train,
    X_val,
    y_val,
    feature_names,
    scenario_name="default",
    seed: int = RANDOM_SEED,
    fixed_params: Optional[Dict] = None,
):
    """Refit with fixed hyperparameters on the training pairs and score the validation pairs."""
    logger.log("  Training Multinomial Logistic Regression candidate (test set unavailable)...")
    best_params = dict(fixed_params)
    X_fit, y_fit = X_train, y_train
    final = _make_multinomial_logistic_classifier(seed=seed, **best_params)
    _fit_multinomial_logistic_classifier(
        final, X_fit, y_fit, sample_weight=compute_ordinal_sample_weights(y_fit)
    )
    val_probability = final.predict_proba(X_val)
    val_prediction = np.argmax(val_probability, axis=1)
    val_metrics = compute_all_metrics(y_val, val_prediction, val_probability)
    fitted_logistic = _get_multinomial_logistic_estimator(final)
    coefficient_importance = np.mean(np.abs(np.asarray(fitted_logistic.coef_)), axis=0)
    importance_df = pd.DataFrame(
        {
            "feature": feature_names[: len(coefficient_importance)],
            "importance": coefficient_importance[: len(feature_names)],
        }
    ).sort_values("importance", ascending=False)
    logger.log(
        f"    Multinomial logistic validation MCC={val_metrics['mcc']:.4f}, QWK={val_metrics['qwk']:.4f}"
    )
    return {
        "model_name": "MultinomialLogisticRegression",
        "model": final,
        "calibrated_model": None,
        "val_metrics": val_metrics,
        "validation_selection_score": validation_selection_score(val_metrics),
        "best_params": best_params,
        "val_predictions": {"y_pred": val_prediction, "y_prob": val_probability},
        "importance": importance_df,
        "test_evaluated": False,
    }


def train_extra_trees(
    X_train,
    y_train,
    groups_train,
    X_val,
    y_val,
    feature_names,
    scenario_name="default",
    seed: int = RANDOM_SEED,
    fixed_params: Optional[Dict] = None,
):
    """Refit with fixed hyperparameters on the training pairs and score the validation pairs."""
    logger.log("  Training Extra Trees candidate (test set unavailable)...")
    best_params = dict(fixed_params)
    X_fit, y_fit = X_train, y_train
    final = _make_extra_trees_classifier(seed=seed, **best_params)
    final.fit(X_fit, y_fit, sample_weight=compute_ordinal_sample_weights(y_fit))
    val_probability = final.predict_proba(X_val)
    val_prediction = np.argmax(val_probability, axis=1)
    val_metrics = compute_all_metrics(y_val, val_prediction, val_probability)
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": final.feature_importances_}
    ).sort_values("importance", ascending=False)
    logger.log(f"    Extra Trees validation MCC={val_metrics['mcc']:.4f}, QWK={val_metrics['qwk']:.4f}")
    return {
        "model_name": "ExtraTrees",
        "model": final,
        "calibrated_model": None,
        "val_metrics": val_metrics,
        "validation_selection_score": validation_selection_score(val_metrics),
        "best_params": best_params,
        "val_predictions": {"y_pred": val_prediction, "y_prob": val_probability},
        "importance": importance_df,
        "test_evaluated": False,
    }


def train_ordinal_decomposition(
    X_train,
    y_train,
    groups_train,
    X_val,
    y_val,
    feature_names,
    scenario_name="default",
    seed: int = RANDOM_SEED,
    fixed_params: Optional[Dict] = None,
):
    """Fit ordinal threshold models on training only and score on validation."""
    del groups_train
    logger.log("  Training Ordinal Binary Decomposition candidate (test set unavailable)...")
    binary_models, bin_kwargs = _fit_ordinal_binary_models(
        X_train, y_train, scenario_name, seed=seed, fixed_params=fixed_params
    )
    val_probability = _predict_ordinal_probabilities(binary_models, X_val)
    val_prediction = np.argmax(val_probability, axis=1)
    val_metrics = compute_all_metrics(y_val, val_prediction, val_probability)
    average_importance = np.mean([model.feature_importances_ for model in binary_models], axis=0)
    importance_df = pd.DataFrame(
        {
            "feature": feature_names[: len(average_importance)],
            "importance": average_importance[: len(feature_names)],
        }
    ).sort_values("importance", ascending=False)
    logger.log(f"    Ordinal validation MCC={val_metrics['mcc']:.4f}, QWK={val_metrics['qwk']:.4f}")
    return {
        "model_name": "OrdinalDecomp",
        "model": binary_models,
        "calibrated_models": None,
        "val_metrics": val_metrics,
        "validation_selection_score": validation_selection_score(val_metrics),
        "best_params": bin_kwargs,
        "val_predictions": {"y_pred": val_prediction, "y_prob": val_probability},
        "importance": importance_df,
        "test_evaluated": False,
    }


def _parameter_candidates(model_name: str, seed: int) -> List[Dict[str, Any]]:
    spaces: Dict[str, Dict[str, Sequence[Any]]] = {
        "MultinomialLogisticRegression": {
            "logistic__C": [0.001, 0.01, 0.1, 1.0, 10.0],
            "logistic__fit_intercept": [True],
            "logistic__tol": [0.001, 0.003],
        },
        "ExtraTrees": {
            "n_estimators": [300, 500, 800, 1200],
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 10, 20, 40],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5],
            "max_features": ["sqrt", "log2", 0.5, 1.0],
            "bootstrap": [False, True],
        },
        "LightGBM": {
            "num_leaves": [15, 31, 63],
            "max_depth": [4, 6, 8, -1],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "n_estimators": [300, 500, 800, 1200],
            "min_child_samples": [10, 20, 40],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": [0.0, 0.1, 0.5],
            "reg_lambda": [0.1, 0.5, 1.0, 2.0],
        },
        "OrdinalDecomp": {
            "num_leaves": [15, 31, 63],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "n_estimators": [300, 500, 800],
            "min_child_samples": [10, 20, 40],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": [0.0, 0.1, 0.5],
            "reg_lambda": [0.1, 0.5, 1.0],
        },
    }
    if model_name not in spaces:
        raise ValueError(f"Unsupported model parameter space: {model_name}")
    n_iter = Config.NESTED_SEARCH_ITERATIONS[model_name]
    return list(ParameterSampler(spaces[model_name], n_iter=n_iter, random_state=seed))


def _apply_resampling_only(X: np.ndarray, y: np.ndarray, scenario: str, seed: int):
    if False:
        return (X, y)
    old_seed = np.random.get_state()
    try:
        np.random.seed(seed)
        step = make_smote_step(scenario, y)
        if step is None:
            return (X, y)
        return step[1].fit_resample(X, y)
    except Exception:
        return (X, y)
    finally:
        np.random.set_state(old_seed)


def _fit_model_matrix(
    model_name: str,
    params: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    scenario: str,
    seed: int,
    imbalance_strategy: Optional[str] = None,
):
    strategy = imbalance_strategy or Config.IMBALANCE_STRATEGY
    X_fit, y_fit = (X, y)
    weights: Optional[np.ndarray] = None
    if strategy == "resampling":
        X_fit, y_fit = _apply_resampling_only(X, y, scenario, seed)
    elif strategy == "sample_weight":
        weights = compute_ordinal_sample_weights(y_fit)
    elif strategy != "none":
        raise ValueError(f"Unknown imbalance strategy: {strategy}")
    if model_name == "MultinomialLogisticRegression":
        model = _make_multinomial_logistic_classifier(seed=seed, **params)
        return _fit_multinomial_logistic_classifier(model, X_fit, y_fit, sample_weight=weights)
    if model_name == "ExtraTrees":
        model = _make_extra_trees_classifier(seed=seed, **params)
        model.fit(X_fit, y_fit, sample_weight=weights)
        return model
    if model_name == "LightGBM":
        model = _make_lgb_classifier(seed=seed, **params)
        model.fit(X_fit, y_fit, sample_weight=weights)
        return model
    if model_name == "OrdinalDecomp":
        models: List[Any] = []
        for threshold in range(Config.NUM_CLASSES - 1):
            y_binary = (y_fit > threshold).astype(int)
            threshold_seed = seed + threshold
            kwargs = dict(params)
            kwargs.update(
                {
                    "objective": "binary",
                    "random_state": threshold_seed,
                    "verbose": -1,
                    "n_jobs": Config.MODEL_N_JOBS,
                    "deterministic": True,
                    "force_col_wise": True,
                    "bagging_seed": threshold_seed,
                    "feature_fraction_seed": threshold_seed,
                    "data_random_seed": threshold_seed,
                }
            )
            kwargs = _activate_lgb_subsampling(kwargs)
            model = lgb.LGBMClassifier(**kwargs)
            binary_weights = None
            if strategy == "sample_weight":
                n_pos = int(y_binary.sum())
                n_neg = int(len(y_binary) - n_pos)
                binary_weights = np.where(y_binary == 1, n_neg / max(n_pos, 1), 1.0)
            model.fit(X_fit, y_binary, sample_weight=binary_weights)
            models.append(model)
        return models
    raise ValueError(f"Unsupported model: {model_name}")


def _predict_model_matrix(model_name: str, model: Any, X: np.ndarray) -> np.ndarray:
    if model_name == "OrdinalDecomp":
        return _predict_ordinal_probabilities(model, X)
    return model.predict_proba(X)


def _importance_table(model_name: str, model: Any, feature_names: List[str]) -> pd.DataFrame:
    try:
        if model_name == "OrdinalDecomp":
            values = np.mean([m.feature_importances_ for m in model], axis=0)
        elif model_name == "MultinomialLogisticRegression":
            fitted_logistic = _get_multinomial_logistic_estimator(model)
            values = np.mean(np.abs(np.asarray(fitted_logistic.coef_)), axis=0)
        else:
            values = np.asarray(model.feature_importances_)
        n = min(len(values), len(feature_names))
        return pd.DataFrame({"feature": feature_names[:n], "importance": values[:n]}).sort_values(
            "importance", ascending=False
        )
    except Exception:
        return pd.DataFrame({"feature": feature_names, "importance": 0.0})


def _available_candidate_names() -> List[str]:
    """Return the single authoritative, ordered set of headline candidates."""
    enabled = {
        "MultinomialLogisticRegression": True,
        "ExtraTrees": True,
        "LightGBM": True,
        "OrdinalDecomp": True,
    }
    return [name for name in Config.CANDIDATE_MODEL_ORDER if enabled.get(name, False)]
