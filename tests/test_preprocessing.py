"""
Tests for MEPS data preprocessing pipeline.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import (
    ALL_PREDICTORS,
    CATEGORICAL_FEATURES,
    FEATURE_VALUE_LABELS,
    NUMERIC_FEATURES,
    TARGET_COL,
    clean_meps_data,
    find_raw_excel
)


def test_find_raw_excel():
    """Verify raw Excel file location detection."""
    excel_path = find_raw_excel()
    assert excel_path.exists()
    assert excel_path.name.lower().endswith((".xlsx", ".xls"))


def test_clean_meps_data_cohort_filtering():
    """Test in-scope filtering logic on mock MEPS data."""
    mock_data = pd.DataFrame({
        "DUPERSID": [1, 2, 3, 4],
        "PERWT24F": [1500.0, 0.0, 2000.0, 1800.0],
        "AGE24X": [45, 30, -1, 60],
        "TOTSLF24": [250.0, 100.0, 50.0, 0.0],
        "TOTEXP24": [1000.0, 500.0, 200.0, 0.0],
        "FAMINC24": [50000.0, 30000.0, 40000.0, -500.0],
        "POVLEV24": [250.0, 150.0, 200.0, -10.0],
        "FAMSZE24": [3, 2, 4, 1],
        "RUSIZE24": [3, 2, 4, 1],
        "EDUCYR": [12, -1, 16, -7],
        "SEX": [1, 2, 1, 2],
        "RACETHX": [2, 1, 3, 4],
        "HISPANX": [2, 1, 2, 2],
        "MARRY24X": [1, 5, 2, -7],
        "HIDEG": [3, 1, 4, -8],
        "REGION24": [2, 3, 1, 4],
        "RUCLAS24": [1, 1, 1, 2],
        "POVCAT24": [4, 3, 4, 1],
        "INSCOV24": [1, 2, 1, 3],
        "INSURC24": [1, 2, 1, 3],
        "UNINS24": [2, 2, 2, 1],
        "PRVEV24": [1, 2, 1, 2],
        "MCREV24": [2, 2, 2, 2],
        "MCDEV24": [2, 1, 2, 2],
        "TRIEV24": [2, 2, 2, 2],
        "VAEV24": [2, 2, 2, 2],
        "RTHLTH31": [2, 3, 1, -8],
        "MNHLTH31": [2, 2, 1, 4],
        "HIBPDX": [2, 2, 1, -1],
        "CHDDX": [2, 2, 2, -1],
        "DIABDX_M18": [2, 2, 2, -1],
        "ASTHDX": [2, 2, 1, -1],
        "ARTHDX": [2, 2, 2, -1],
        "CANCERDX": [2, 2, 2, -1],
        "STRKDX": [2, 2, 2, -1],
        "CHBRON31": [2, 2, 2, -1],
        "CHOLDX": [2, 2, 2, -1],
        "EMPST31": [1, 4, 1, -1]
    })

    cleaned_df, report = clean_meps_data(mock_data)

    # Row 2 has PERWT24F == 0 (excluded)
    # Row 3 has AGE24X == -1 (excluded)
    # Remaining should be exactly Row 1 and Row 4
    assert len(cleaned_df) == 2
    assert report["exclusions"]["non_positive_weight"] == 1
    assert report["exclusions"]["age_out_of_scope"] == 1
    assert (cleaned_df[TARGET_COL] >= 0).all()


def test_processed_dataset_schema():
    """Verify schema and completeness of the saved processed dataset if available."""
    processed_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "meps_2024_cleaned.parquet"
    if processed_path.exists():
        df = pd.read_parquet(processed_path)
        assert len(df) > 10000
        assert TARGET_COL in df.columns
        for feature in ALL_PREDICTORS:
            assert feature in df.columns, f"Missing feature: {feature}"
        assert not df[ALL_PREDICTORS].isna().any().any(), "Cleaned dataset contains unexpected NaNs"
        assert (df[TARGET_COL] >= 0).all(), "Target contains negative values"
