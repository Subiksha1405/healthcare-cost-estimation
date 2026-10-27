"""
Tests for regression model and prediction artifact loading.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.conformal_prediction import NormalizedSplitConformalPredictor
from src.data_preprocessing import ALL_PREDICTORS


@pytest.fixture(scope="module")
def trained_predictor():
    model_path = Path(__file__).resolve().parent.parent / "models" / "MEPS_NCP_webapp_model.joblib"
    assert model_path.exists(), f"Model artifact not found at {model_path}"
    predictor = joblib.load(model_path)
    return predictor


def test_model_artifact_attributes(trained_predictor):
    """Verify that saved model contains all necessary attributes."""
    assert isinstance(trained_predictor, NormalizedSplitConformalPredictor)
    assert trained_predictor.is_fitted
    assert trained_predictor.is_calibrated
    assert len(trained_predictor.calibration_scores) > 0
    assert trained_predictor.scale_floor > 0
    assert len(trained_predictor.feature_names) == len(ALL_PREDICTORS)


def test_model_point_prediction_properties(trained_predictor):
    """Verify point predictions are non-negative and finite."""
    # Create sample patient input
    sample_df = pd.DataFrame([{
        "AGE24X": 52.0,
        "FAMINC24": 75000.0,
        "POVLEV24": 350.0,
        "FAMSZE24": 2.0,
        "RUSIZE24": 2.0,
        "EDUCYR": 14.0,
        "SEX": 2,
        "RACETHX": 2,
        "HISPANX": 2,
        "MARRY24X": 1,
        "HIDEG": 4,
        "REGION24": 3,
        "RUCLAS24": 1,
        "POVCAT24": 4,
        "INSCOV24": 1,
        "INSURC24": 1,
        "UNINS24": 2,
        "PRVEV24": 1,
        "MCREV24": 2,
        "MCDEV24": 2,
        "TRIEV24": 2,
        "VAEV24": 2,
        "RTHLTH31": 2,
        "MNHLTH31": 2,
        "HIBPDX": 1,
        "CHDDX": 2,
        "DIABDX_M18": 2,
        "ASTHDX": 2,
        "ARTHDX": 2,
        "CANCERDX": 2,
        "STRKDX": 2,
        "CHBRON31": 2,
        "CHOLDX": 1,
        "EMPST31": 1
    }])[ALL_PREDICTORS]

    point_pred, lower, upper, width = trained_predictor.predict(sample_df, confidence_level=0.90)

    assert len(point_pred) == 1
    assert np.isfinite(point_pred[0])
    assert point_pred[0] >= 0.0
    assert np.isfinite(lower[0])
    assert np.isfinite(upper[0])
    assert lower[0] >= 0.0
    assert upper[0] >= lower[0]
    assert width[0] >= 0.0
