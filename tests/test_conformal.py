"""
Tests for Normalized Split Conformal Prediction methodology and finite-sample guarantees.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge

from src.conformal_prediction import NormalizedSplitConformalPredictor


def test_finite_sample_quantile_calculation():
    """Verify exact formula for finite-sample conformal quantile: ceil((n + 1)(1 - alpha)) / n."""
    np.random.seed(42)
    scores = np.sort(np.random.exponential(scale=2.0, size=100))

    predictor = NormalizedSplitConformalPredictor(
        point_model=DummyRegressor(strategy="mean"),
        scale_model=DummyRegressor(strategy="mean")
    )
    predictor.calibration_scores = scores
    predictor.is_calibrated = True

    # For n=100, alpha=0.10:
    # k = ceil(101 * 0.90) = ceil(90.9) = 91
    # quantile probability = 91 / 100 = 0.91
    q_90 = predictor.get_conformal_quantile(0.90)
    expected_q = float(np.quantile(scores, 91 / 100, method="higher"))
    assert np.isclose(q_90, expected_q)


def test_conformal_coverage_monotonicity():
    """Verify that higher confidence level yields wider intervals."""
    # Build a small dummy conformal predictor
    X = pd.DataFrame({"feat1": np.linspace(1, 10, 50)})
    y = np.linspace(10, 100, 50) + np.random.normal(0, 5, 50)
    y = np.maximum(0.0, y)

    pred = NormalizedSplitConformalPredictor(
        point_model=Ridge(),
        scale_model=DummyRegressor(strategy="constant", constant=1.0),
        target_transform="none"
    )
    pred.fit(X, y)
    pred.calibrate(X, y)

    X_test = pd.DataFrame({"feat1": [5.0, 7.5]})
    _, lower_80, upper_80, width_80 = pred.predict(X_test, confidence_level=0.80)
    _, lower_90, upper_90, width_90 = pred.predict(X_test, confidence_level=0.90)
    _, lower_95, upper_95, width_95 = pred.predict(X_test, confidence_level=0.95)

    # 95% interval must be wider than 90%, which must be wider than 80%
    assert (width_95 >= width_90).all()
    assert (width_90 >= width_80).all()
    assert (lower_95 <= lower_90).all()
    assert (upper_95 >= upper_90).all()


def test_interval_nonnegativity_clipping():
    """Verify that lower prediction limit is strictly clamped at zero."""
    X = pd.DataFrame({"feat1": [1.0, 2.0, 3.0]})
    y = np.array([0.0, 1.0, 2.0])

    pred = NormalizedSplitConformalPredictor(
        point_model=DummyRegressor(strategy="constant", constant=0.5),
        scale_model=DummyRegressor(strategy="constant", constant=5.0),
        target_transform="log1p"
    )
    pred.fit(X, y)
    pred.calibrate(X, y)

    _, lower, upper, _ = pred.predict(X, confidence_level=0.90)
    assert (lower >= 0.0).all(), "Lower prediction limit must never be negative for healthcare costs"
