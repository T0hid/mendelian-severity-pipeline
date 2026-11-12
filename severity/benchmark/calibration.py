"""Probability calibration (Model families, selection and calibration).

Only the selected model of each feature set is calibrated, on the validation pairs. Isotonic
regression is used when every calibrated group has at least 100 validation pairs on each side,
Platt scaling otherwise. Four-class models are calibrated one-versus-rest with one method; the
ordinal learner is calibrated separately at each of its three boundaries. The fitted model and
calibrator are then frozen before the test set is used.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from .config import Config
from .models import _predict_ordinal_probabilities
from .runtime import logger


def _predict_uncalibrated_candidate(result: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    model_name = result.get("model_name", "")
    if model_name == "OrdinalDecomp":
        return _predict_ordinal_probabilities(result["model"], X)
    return result["model"].predict_proba(X)


def _predict_calibrated_candidate(result: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    if result.get("calibration_status") != "success":
        return _predict_uncalibrated_candidate(result, X)
    if result["model_name"] == "OrdinalDecomp":
        models = result.get("calibrated_models")
        if not models or len(models) != Config.NUM_CLASSES - 1:
            raise RuntimeError("Successful ordinal calibration is missing one or more boundaries.")
        return _predict_ordinal_probabilities(models, X)
    calibrator = result.get("calibrated_model")
    if calibrator is None:
        raise RuntimeError("Calibration status is success but the calibrator is missing.")
    return calibrator.predict_proba(X)


def freeze_selected_model_and_calibrator(
    result: Dict[str, Any], X_val: np.ndarray, y_val: np.ndarray, scenario_name: str
) -> Dict[str, Any]:
    """Fit validation-only calibration and lock the selected estimator."""
    if result.get("model_frozen", False) or result.get("calibrator_frozen", False):
        raise RuntimeError(f"{scenario_name} was already frozen before calibration.")
    _fit_selected_calibration(result, X_val, y_val, scenario_name)
    result["model_frozen"] = True
    result["calibrator_frozen"] = True
    result["frozen_before_any_test_evaluation"] = True
    logger.log(
        f"  Frozen selected model and calibration decision: {scenario_name}/{result['model_name']}; calibration_status={result.get('calibration_status', 'unknown')}"
    )
    return result


def _choose_calibration_method(y: np.ndarray) -> str:
    values = np.asarray(y, dtype=int)
    classes = np.unique(values)
    for class_value in classes:
        positives = int(np.sum(values == class_value))
        negatives = int(len(values) - positives)
        if positives < Config.CALIBRATION_MIN_POSITIVES or negatives < Config.CALIBRATION_MIN_NEGATIVES:
            return Config.CALIBRATION_METHOD_SMALL
    return Config.CALIBRATION_METHOD_LARGE


def calibrate_prefit_model(model, X_val, y_val, label: str = "model", method: Optional[str] = None):
    selected_method = method or _choose_calibration_method(np.asarray(y_val))
    try:
        if True:
            calibrator = CalibratedClassifierCV(FrozenEstimator(model), method=selected_method)
        else:
            calibrator = CalibratedClassifierCV(model, method=selected_method, cv="prefit")
        calibrator.fit(X_val, y_val)
        logger.log(f"    Calibration OK ({selected_method}): {label}")
        return calibrator
    except Exception as exc:
        logger.log(
            f"    Calibration failed for {label} with {selected_method}: {type(exc).__name__}: {exc}",
            level="WARNING",
        )
        return None


def _fit_selected_calibration(
    result: Dict[str, Any], X_val: np.ndarray, y_val: np.ndarray, scenario_name: str
) -> None:
    model_name = result["model_name"]
    result["calibration_partition"] = "validation_after_model_selection"
    result["base_model_frozen_before_calibration"] = True
    if model_name == "OrdinalDecomp":
        attempted_methods: List[str] = []
        attempted_calibrators: List[Optional[Any]] = []
        failures: List[str] = []
        for threshold, model in enumerate(result["model"]):
            binary_y = (y_val > threshold).astype(int)
            method = _choose_calibration_method(binary_y)
            attempted_methods.append(method)
            calibrator = calibrate_prefit_model(
                model,
                X_val,
                binary_y,
                label=f"{scenario_name}/{model_name}/threshold_{threshold}",
                method=method,
            )
            attempted_calibrators.append(calibrator)
            if calibrator is None:
                failures.append(f"threshold_{threshold}")
        result["calibration_method_attempted"] = "|".join(attempted_methods)
        if not failures:
            result["calibrated_models"] = list(attempted_calibrators)
            result["calibrated_model"] = None
            result["calibration_method"] = result["calibration_method_attempted"]
            result["calibration_success"] = True
            result["calibration_status"] = "success"
            result["calibration_failure_details"] = ""
        else:
            result["calibrated_models"] = None
            result["calibrated_model"] = None
            result["calibration_method"] = "none"
            result["calibration_success"] = False
            result["calibration_status"] = "uncalibrated_fallback"
            result["calibration_failure_details"] = "|".join(failures)
            logger.log(
                f"    Atomic ordinal calibration fallback for {scenario_name}/{model_name}; failed boundaries={result['calibration_failure_details']}",
                level="WARNING",
            )
    else:
        method = _choose_calibration_method(y_val)
        result["calibration_method_attempted"] = method
        calibrator = calibrate_prefit_model(
            result["model"], X_val, y_val, label=f"{scenario_name}/{model_name}", method=method
        )
        result["calibrated_model"] = calibrator
        result["calibrated_models"] = None
        result["calibration_success"] = calibrator is not None
        if calibrator is not None:
            result["calibration_method"] = method
            result["calibration_status"] = "success"
            result["calibration_failure_details"] = ""
        else:
            result["calibration_method"] = "none"
            result["calibration_status"] = "uncalibrated_fallback"
            result["calibration_failure_details"] = "model_calibration_failed"
