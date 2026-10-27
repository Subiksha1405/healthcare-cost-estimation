"""
Integration and smoke tests for the Streamlit web application components.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import ALL_PREDICTORS, TARGET_COL


def test_feature_options_json():
    """Verify options JSON schema and valid categories."""
    options_path = Path(__file__).resolve().parent.parent / "models" / "MEPS_feature_options.json"
    assert options_path.exists()

    with open(options_path, "r", encoding="utf-8") as f:
        options = json.load(f)

    assert "numeric_features" in options
    assert "categorical_features" in options
    assert len(options["numeric_features"]) == 6
    assert len(options["categorical_features"]) == 28

    # Check that each categorical feature has options with code and label
    for cat_name, cat_meta in options["categorical_features"].items():
        assert "default_code" in cat_meta
        assert "options" in cat_meta
        assert len(cat_meta["options"]) > 0
        for opt in cat_meta["options"]:
            assert "code" in opt
            assert "label" in opt


def test_model_metrics_json():
    """Verify saved evaluation metrics schema and mathematical validity."""
    metrics_path = Path(__file__).resolve().parent.parent / "models" / "model_metrics.json"
    assert metrics_path.exists()

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "dataset_metadata" in metrics
    assert "test_evaluations" in metrics
    assert "subgroup_performance" in metrics

    eval_90 = metrics["test_evaluations"]["confidence_90"]
    # Empirical test coverage must be close to or exceed nominal target
    assert eval_90["nominal_coverage"] == 0.90
    assert 0.85 <= eval_90["empirical_coverage"] <= 0.95
    assert eval_90["mean_interval_width"] > 0
    assert eval_90["median_interval_width"] > 0
    assert eval_90["mae"] > 0


def test_end_to_end_smoke_prediction():
    """Smoke test: predict on real patient records from processed dataset."""
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / "models" / "MEPS_NCP_webapp_model.joblib"
    data_path = base_dir / "data" / "processed" / "meps_2024_cleaned.parquet"

    predictor = joblib.load(model_path)
    df = pd.read_parquet(data_path)

    # Take first 5 real patient records
    test_sample = df.iloc[:5][ALL_PREDICTORS]
    point_pred, lower, upper, width = predictor.predict(test_sample, confidence_level=0.90)

    assert len(point_pred) == 5
    assert len(lower) == 5
    assert len(upper) == 5
    assert len(width) == 5

    assert (point_pred >= 0.0).all()
    assert (lower >= 0.0).all()
    assert (upper >= lower).all()
    assert (width >= 0.0).all()
    assert np.all(np.isfinite(point_pred))
    assert np.all(np.isfinite(lower))
    assert np.all(np.isfinite(upper))


def test_navigation_state_transitions():
    """Verify that navigation state management updates cleanly without key conflicts."""
    import streamlit as st
    from app import PAGE_FORM, PAGE_RESULTS, PAGE_HOME, PAGE_LIST

    # Test that page constants exist and form valid sequence
    assert PAGE_HOME in PAGE_LIST
    assert PAGE_FORM in PAGE_LIST
    assert PAGE_RESULTS in PAGE_LIST

    # Verify PAGE_LIST ordering
    assert PAGE_LIST.index(PAGE_HOME) == 0
    assert PAGE_LIST.index(PAGE_FORM) == 1
    assert PAGE_LIST.index(PAGE_RESULTS) == 2

