"""
Normalized Split Conformal Prediction for Healthcare Cost Prediction.

Implements finite-sample valid prediction intervals with conditional residual
scaling (Papadopoulos et al., 2002; Lei et al., 2018; Angelopoulos & Bates, 2021).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error


class NormalizedSplitConformalPredictor:
    """
    Normalized Split Conformal Predictor for regression with non-negative targets.

    Parameters
    ----------
    point_model : BaseEstimator
        Regression model predicting the central response on the transformed scale.
    scale_model : BaseEstimator
        Dispersion model predicting conditional residual scale on the training residuals.
    target_transform : str, default='log1p'
        Transformation applied to the target variable: 'log1p' or 'none'.
    scale_floor_percentile : float, default=5.0
        Percentile of training scale estimates used as positive regularizer floor.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        point_model: BaseEstimator,
        scale_model: BaseEstimator,
        target_transform: str = "log1p",
        scale_floor_percentile: float = 5.0,
        random_state: int = 42
    ):
        self.point_model = point_model
        self.scale_model = scale_model
        self.target_transform = target_transform
        self.scale_floor_percentile = scale_floor_percentile
        self.random_state = random_state

        self.scale_floor: float = 1.0
        self.calibration_scores: np.ndarray = np.array([])
        self.feature_names: List[str] = []
        self.categorical_features: List[str] = []
        self.n_train: int = 0
        self.n_cal: int = 0
        self.is_fitted: bool = False
        self.is_calibrated: bool = False

    def _transform_y(self, y: np.ndarray) -> np.ndarray:
        if self.target_transform == "log1p":
            return np.log1p(y)
        return y

    def _inverse_transform_y(self, y_trans: np.ndarray) -> np.ndarray:
        if self.target_transform == "log1p":
            return np.maximum(0.0, np.expm1(y_trans))
        return np.maximum(0.0, y_trans)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None
    ) -> "NormalizedSplitConformalPredictor":
        """
        Fits the point prediction model and residual-scale model on training data only.
        """
        self.feature_names = feature_names or list(X_train.columns)
        self.categorical_features = categorical_features or []
        self.n_train = len(X_train)

        y_train_arr = np.asarray(y_train, dtype=float)
        y_trans = self._transform_y(y_train_arr)

        # 1. Fit central point model f(x)
        self.point_model.fit(X_train, y_trans)
        train_pred_trans = self.point_model.predict(X_train)

        # 2. Compute in-sample training absolute residuals
        train_abs_res = np.abs(y_trans - train_pred_trans)

        # 3. Fit scale model s(x) to predict log-dispersion
        self.scale_model.fit(X_train, np.log(train_abs_res + 0.05))

        # 4. Compute empirical scale floor from training scale distribution
        train_scales = np.exp(self.scale_model.predict(X_train))
        self.scale_floor = float(np.percentile(train_scales, self.scale_floor_percentile))
        self.scale_floor = max(1e-4, self.scale_floor)

        self.is_fitted = True
        return self

    def calibrate(
        self,
        X_cal: pd.DataFrame,
        y_cal: np.ndarray
    ) -> "NormalizedSplitConformalPredictor":
        """
        Calibrates the nonconformity scores on an independent calibration set.
        """
        if not self.is_fitted:
            raise ValueError("Predictor must be fitted before calibration.")

        self.n_cal = len(X_cal)
        y_cal_arr = np.asarray(y_cal, dtype=float)
        y_cal_trans = self._transform_y(y_cal_arr)

        cal_pred_trans = self.point_model.predict(X_cal)
        cal_scale = np.maximum(np.exp(self.scale_model.predict(X_cal)), self.scale_floor)

        # Normalized nonconformity score: |y - f(x)| / max(s(x), floor)
        self.calibration_scores = np.abs(y_cal_trans - cal_pred_trans) / cal_scale
        self.is_calibrated = True
        return self

    def get_conformal_quantile(self, confidence_level: float = 0.90) -> float:
        """
        Computes finite-sample conformal quantile:
        k = ceil((n + 1) * (1 - alpha))
        q = Quantile(k / n)
        """
        if not self.is_calibrated or len(self.calibration_scores) == 0:
            raise ValueError("Predictor has not been calibrated with an independent dataset.")

        n = len(self.calibration_scores)
        alpha = 1.0 - confidence_level
        k = int(np.ceil((n + 1) * (1.0 - alpha)))
        quantile_level = min(1.0, k / n)

        q = float(np.quantile(self.calibration_scores, quantile_level, method="higher"))
        return q

    def predict(
        self,
        X: pd.DataFrame,
        confidence_level: float = 0.90
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates point estimates and lower/upper prediction limits.

        Returns
        -------
        point_pred : np.ndarray
            Central out-of-pocket prediction ($ USD).
        lower_bound : np.ndarray
            Lower conformal bound ($ USD), clipped at $0.00.
        upper_bound : np.ndarray
            Upper conformal bound ($ USD).
        interval_width : np.ndarray
            Width of the prediction interval ($ USD).
        """
        if not self.is_calibrated:
            raise ValueError("Predictor must be calibrated before generating prediction intervals.")

        # Ensure correct column ordering
        X_df = X[self.feature_names].copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names)

        pred_trans = self.point_model.predict(X_df)
        pred_scale = np.maximum(np.exp(self.scale_model.predict(X_df)), self.scale_floor)

        q = self.get_conformal_quantile(confidence_level)

        if self.target_transform == "log1p":
            lower_trans = np.maximum(0.0, pred_trans - q * pred_scale)
            upper_trans = pred_trans + q * pred_scale

            point_pred = np.maximum(0.0, np.expm1(pred_trans))
            lower_bound = np.maximum(0.0, np.expm1(lower_trans))
            upper_bound = np.maximum(0.0, np.expm1(upper_trans))
        else:
            point_pred = np.maximum(0.0, pred_trans)
            lower_bound = np.maximum(0.0, pred_trans - q * pred_scale)
            upper_bound = pred_trans + q * pred_scale

        interval_width = upper_bound - lower_bound
        return point_pred, lower_bound, upper_bound, interval_width

    def evaluate_test_set(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        confidence_level: float = 0.90
    ) -> Dict:
        """
        Evaluates empirical coverage, point metrics, and interval widths on untouched test set.
        """
        y_true = np.asarray(y_test, dtype=float)
        point_pred, lower_bound, upper_bound, interval_width = self.predict(
            X_test, confidence_level=confidence_level
        )

        covered = (y_true >= lower_bound) & (y_true <= upper_bound)
        empirical_coverage = float(np.mean(covered))

        mae = float(mean_absolute_error(y_true, point_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, point_pred)))
        r2 = float(r2_score(y_true, point_pred))
        med_ae = float(median_absolute_error(y_true, point_pred))

        zero_mask = (y_true == 0)
        coverage_zero = float(np.mean(covered[zero_mask])) if zero_mask.sum() > 0 else 0.0
        coverage_nonzero = float(np.mean(covered[~zero_mask])) if (~zero_mask).sum() > 0 else 0.0

        return {
            "nominal_coverage": float(confidence_level),
            "empirical_coverage": empirical_coverage,
            "conformal_quantile_q": self.get_conformal_quantile(confidence_level),
            "scale_floor": self.scale_floor,
            "test_sample_size": len(y_true),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "median_absolute_error": med_ae,
            "mean_interval_width": float(np.mean(interval_width)),
            "median_interval_width": float(np.median(interval_width)),
            "min_interval_width": float(np.min(interval_width)),
            "max_interval_width": float(np.max(interval_width)),
            "zero_spenders_coverage": coverage_zero,
            "nonzero_spenders_coverage": coverage_nonzero,
            "covered_array": covered,
            "point_pred": point_pred,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }
