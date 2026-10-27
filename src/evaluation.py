"""
Evaluation and subgroup audit utilities for conformal healthcare cost prediction.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_preprocessing import FEATURE_VALUE_LABELS


def evaluate_subgroups(
    df_test: pd.DataFrame,
    y_test: np.ndarray,
    covered: np.ndarray,
    point_pred: np.ndarray,
    interval_width: np.ndarray
) -> Dict[str, pd.DataFrame]:
    """
    Computes empirical coverage, mean interval width, and MAE across key clinical and demographic subgroups.
    """
    subgroup_dfs = {}

    eval_df = df_test.copy()
    eval_df["y_true"] = y_test
    eval_df["covered"] = covered
    eval_df["point_pred"] = point_pred
    eval_df["interval_width"] = interval_width
    eval_df["error"] = np.abs(y_test - point_pred)

    # 1. Age Groups
    eval_df["Age_Group"] = pd.cut(
        eval_df["AGE24X"],
        bins=[-1, 17, 44, 64, 85],
        labels=["0-17 (Children)", "18-44 (Young Adults)", "45-64 (Middle-Aged)", "65-85 (Older Adults)"]
    )
    age_audit = eval_df.groupby("Age_Group", observed=False).agg(
        Count=("y_true", "count"),
        Empirical_Coverage=("covered", "mean"),
        Mean_Interval_Width=("interval_width", "mean"),
        Median_Interval_Width=("interval_width", "median"),
        MAE=("error", "mean")
    ).reset_index()
    subgroup_dfs["Age_Group"] = age_audit

    # 2. Insurance Coverage
    ins_map = {1: "Any Private", 2: "Public Only", 3: "Uninsured"}
    eval_df["Insurance"] = eval_df["INSCOV24"].map(ins_map).fillna("Unknown")
    ins_audit = eval_df.groupby("Insurance", observed=False).agg(
        Count=("y_true", "count"),
        Empirical_Coverage=("covered", "mean"),
        Mean_Interval_Width=("interval_width", "mean"),
        Median_Interval_Width=("interval_width", "median"),
        MAE=("error", "mean")
    ).reset_index()
    subgroup_dfs["Insurance"] = ins_audit

    # 3. Poverty Category
    pov_map = {
        1: "Poor / Negative (<100% FPL)",
        2: "Near Poor (100% - 124% FPL)",
        3: "Low Income (125% - 199% FPL)",
        4: "Middle Income (200% - 399% FPL)",
        5: "High Income (>=400% FPL)"
    }
    eval_df["Poverty_Category"] = eval_df["POVCAT24"].map(pov_map).fillna("Unknown")
    pov_audit = eval_df.groupby("Poverty_Category", observed=False).agg(
        Count=("y_true", "count"),
        Empirical_Coverage=("covered", "mean"),
        Mean_Interval_Width=("interval_width", "mean"),
        Median_Interval_Width=("interval_width", "median"),
        MAE=("error", "mean")
    ).reset_index()
    subgroup_dfs["Poverty_Category"] = pov_audit

    # 4. Self-rated Physical Health
    health_map = {1: "1 - Excellent", 2: "2 - Very Good", 3: "3 - Good", 4: "4 - Fair", 5: "5 - Poor"}
    eval_df["Health_Status"] = eval_df["RTHLTH31"].map(health_map).fillna("Unknown")
    health_audit = eval_df.groupby("Health_Status", observed=False).agg(
        Count=("y_true", "count"),
        Empirical_Coverage=("covered", "mean"),
        Mean_Interval_Width=("interval_width", "mean"),
        Median_Interval_Width=("interval_width", "median"),
        MAE=("error", "mean")
    ).reset_index()
    subgroup_dfs["Health_Status"] = health_audit

    # 5. Chronic Condition Comorbidities
    chronic_vars = [
        "HIBPDX", "CHDDX", "DIABDX_M18", "ASTHDX", "ARTHDX",
        "CANCERDX", "STRKDX", "CHBRON31", "CHOLDX"
    ]
    eval_df["Chronic_Count"] = eval_df[chronic_vars].apply(lambda row: sum(row == 1), axis=1)
    eval_df["Chronic_Category"] = eval_df["Chronic_Count"].apply(
        lambda c: "0 conditions" if c == 0 else ("1 condition" if c == 1 else ("2 conditions" if c == 2 else "3+ conditions"))
    )
    chronic_audit = eval_df.groupby("Chronic_Category", observed=False).agg(
        Count=("y_true", "count"),
        Empirical_Coverage=("covered", "mean"),
        Mean_Interval_Width=("interval_width", "mean"),
        Median_Interval_Width=("interval_width", "median"),
        MAE=("error", "mean")
    ).reset_index()
    subgroup_dfs["Chronic_Category"] = chronic_audit

    return subgroup_dfs
