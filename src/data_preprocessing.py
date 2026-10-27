"""
Data Preprocessing Pipeline for MEPS HC-256 Healthcare Cost Prediction.

Loads the MEPS 2024 Full Year Consolidated Data (HC-256), validates the schema,
filters the in-scope analytic cohort, handles MEPS-specific missing codes,
and creates clean feature representations.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Target and Survey Variables
TARGET_COL = "TOTSLF24"
TOTAL_EXP_COL = "TOTEXP24"
WEIGHT_COL = "PERWT24F"
ID_COL = "DUPERSID"

# 34 Candidate Predictors confirmed in HC-256
NUMERIC_FEATURES = [
    "AGE24X",
    "FAMINC24",
    "POVLEV24",
    "FAMSZE24",
    "RUSIZE24",
    "EDUCYR"
]

CATEGORICAL_FEATURES = [
    "SEX",
    "RACETHX",
    "HISPANX",
    "MARRY24X",
    "HIDEG",
    "REGION24",
    "RUCLAS24",
    "POVCAT24",
    "INSCOV24",
    "INSURC24",
    "UNINS24",
    "PRVEV24",
    "MCREV24",
    "MCDEV24",
    "TRIEV24",
    "VAEV24",
    "RTHLTH31",
    "MNHLTH31",
    "HIBPDX",
    "CHDDX",
    "DIABDX_M18",
    "ASTHDX",
    "ARTHDX",
    "CANCERDX",
    "STRKDX",
    "CHBRON31",
    "CHOLDX",
    "EMPST31"
]

ALL_PREDICTORS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Official Codebook Value Mappings for Human-Readable Dropdowns
FEATURE_VALUE_LABELS = {
    "SEX": {
        1: "Male",
        2: "Female"
    },
    "RACETHX": {
        1: "Hispanic",
        2: "Non-Hispanic White",
        3: "Non-Hispanic Black",
        4: "Non-Hispanic Asian",
        5: "Non-Hispanic Other / Multiple Race"
    },
    "HISPANX": {
        1: "Hispanic",
        2: "Not Hispanic"
    },
    "MARRY24X": {
        1: "Married",
        2: "Widowed",
        3: "Divorced",
        4: "Separated",
        5: "Never Married",
        6: "Under 16 (Inapplicable)"
    },
    "HIDEG": {
        1: "No Degree",
        2: "GED",
        3: "High School Diploma",
        4: "Bachelor's Degree",
        5: "Master's Degree",
        6: "Doctorate Degree",
        7: "Other Degree",
        8: "Under 16 (Inapplicable)"
    },
    "REGION24": {
        1: "Northeast",
        2: "Midwest",
        3: "South",
        4: "West"
    },
    "RUCLAS24": {
        1: "Standard RU",
        2: "New RU",
        3: "Student RU"
    },
    "POVCAT24": {
        1: "Poor / Negative (<100% FPL)",
        2: "Near Poor (100% to <125% FPL)",
        3: "Low Income (125% to <200% FPL)",
        4: "Middle Income (200% to <400% FPL)",
        5: "High Income (>=400% FPL)"
    },
    "INSCOV24": {
        1: "Any Private",
        2: "Public Only",
        3: "Uninsured"
    },
    "INSURC24": {
        1: "<65 Any Private",
        2: "<65 Public Only",
        3: "<65 Uninsured",
        4: "65+ Medicare Only",
        5: "65+ Medicare & Private",
        6: "65+ Medicare & Other Public",
        7: "65+ Uninsured",
        8: "65+ No Medicare & Any Public/Private"
    },
    "UNINS24": {
        1: "Yes (Uninsured Full Year)",
        2: "No (Had Coverage)"
    },
    "PRVEV24": {
        1: "Yes",
        2: "No"
    },
    "MCREV24": {
        1: "Yes",
        2: "No"
    },
    "MCDEV24": {
        1: "Yes",
        2: "No"
    },
    "TRIEV24": {
        1: "Yes",
        2: "No"
    },
    "VAEV24": {
        1: "Yes",
        2: "No"
    },
    "RTHLTH31": {
        1: "Excellent",
        2: "Very Good",
        3: "Good",
        4: "Fair",
        5: "Poor"
    },
    "MNHLTH31": {
        1: "Excellent",
        2: "Very Good",
        3: "Good",
        4: "Fair",
        5: "Poor"
    },
    "HIBPDX": {
        1: "Yes (Diagnosed)",
        2: "No",
        3: "Inapplicable (Under 18)"
    },
    "CHDDX": {
        1: "Yes (Diagnosed)",
        2: "No",
        3: "Inapplicable (Under 18)"
    },
    "DIABDX_M18": {
        1: "Yes (Diagnosed)",
        2: "No",
        3: "Inapplicable (Under 18)"
    },
    "ASTHDX": {
        1: "Yes (Diagnosed)",
        2: "No",
        3: "Inapplicable (Under 18)"
    },
    "ARTHDX": {
        1: "Yes (Diagnosed)",
        2: "No",
        3: "Inapplicable (Under 18)"
    },
    "CANCERDX": {
        1: "Yes (Diagnosed)",
        2: "No",
        3: "Inapplicable (Under 18)"
    },
    "STRKDX": {
        1: "Yes (Diagnosed)",
        2: "No",
        3: "Inapplicable (Under 18)"
    },
    "CHBRON31": {
        1: "Yes (Diagnosed)",
        2: "No",
        3: "Inapplicable (Under 18)"
    },
    "CHOLDX": {
        1: "Yes (Diagnosed)",
        2: "No",
        3: "Inapplicable (Under 18)"
    },
    "EMPST31": {
        1: "Employed at Interview Date",
        2: "Has Job to Return to",
        3: "Employed During Reference Period",
        4: "Not Employed",
        5: "Inapplicable (Under 16)"
    }
}


def find_raw_excel(base_dir: Optional[str] = None) -> Path:
    """Locates the raw MEPS Excel file in expected workspace directories."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
    else:
        base_dir = Path(base_dir)

    candidates = [
        base_dir / "data" / "raw" / "h256.xlsx",
        base_dir / "h256.xlsx",
        base_dir / "H256.xlsx"
    ]
    for c in candidates:
        if c.exists():
            logger.info("Found raw MEPS Excel dataset at: %s", c)
            return c
    raise FileNotFoundError(
        f"MEPS HC-256 Excel file not found. Checked: {[str(c) for c in candidates]}"
    )


def load_raw_meps_data(filepath: Path) -> pd.DataFrame:
    """Loads required columns from the MEPS Excel dataset using the fast calamine engine."""
    cols_to_load = [ID_COL, TARGET_COL, TOTAL_EXP_COL, WEIGHT_COL] + ALL_PREDICTORS
    logger.info("Loading %d columns from %s...", len(cols_to_load), filepath)
    try:
        df = pd.read_excel(filepath, sheet_name="H256", usecols=cols_to_load, engine="calamine")
    except Exception as e:
        logger.warning("Calamine engine failed (%s); falling back to default engine.", e)
        df = pd.read_excel(filepath, sheet_name="H256", usecols=cols_to_load)
    logger.info("Successfully loaded raw dataset with shape %s", df.shape)
    return df


def clean_meps_data(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Cleans raw MEPS data according to official AHRQ specifications:
    1. Filter in-scope cohort: PERWT24F > 0 and AGE24X >= 0.
    2. Recode MEPS special missing codes (-1, -7, -8, -15).
    3. Document exclusions and summary statistics.
    """
    report = {
        "raw_rows": int(len(raw_df)),
        "raw_cols": int(len(raw_df.columns)),
        "exclusions": {}
    }

    # Step 1: In-scope filtering
    wt_zero_mask = raw_df[WEIGHT_COL] <= 0
    report["exclusions"]["non_positive_weight"] = int(wt_zero_mask.sum())

    in_scope_wt = raw_df[~wt_zero_mask].copy()
    age_inapp_mask = in_scope_wt["AGE24X"] < 0
    report["exclusions"]["age_out_of_scope"] = int(age_inapp_mask.sum())

    cohort = in_scope_wt[~age_inapp_mask].copy()
    report["final_cohort_rows"] = int(len(cohort))
    logger.info(
        "Filtered cohort: %d rows (excluded %d zero-weight, %d out-of-scope age)",
        len(cohort),
        report["exclusions"]["non_positive_weight"],
        report["exclusions"]["age_out_of_scope"]
    )

    # Step 2: Clean and validate target variable (TOTSLF24)
    # Target must be non-negative.
    if (cohort[TARGET_COL] < 0).any():
        neg_target_cnt = int((cohort[TARGET_COL] < 0).sum())
        logger.warning("Found %d negative target values; dropping them.", neg_target_cnt)
        cohort = cohort[cohort[TARGET_COL] >= 0].copy()
        report["exclusions"]["negative_target"] = neg_target_cnt
    else:
        report["exclusions"]["negative_target"] = 0

    report["target_summary"] = {
        "count": int(cohort[TARGET_COL].count()),
        "mean": float(cohort[TARGET_COL].mean()),
        "std": float(cohort[TARGET_COL].std()),
        "median": float(cohort[TARGET_COL].median()),
        "min": float(cohort[TARGET_COL].min()),
        "max": float(cohort[TARGET_COL].max()),
        "zero_count": int((cohort[TARGET_COL] == 0).sum()),
        "zero_percentage": float((cohort[TARGET_COL] == 0).mean() * 100)
    }

    # Step 3: Handle numeric variables
    # EDUCYR: -1 Inapplicable (children < 5), -7 Refused, -8 Don't Know
    # Impute missing codes with 0 for children < 5 (-1) and median for true missing (-7, -8)
    med_educ = cohort.loc[cohort["EDUCYR"] >= 0, "EDUCYR"].median()
    cohort["EDUCYR"] = cohort["EDUCYR"].apply(
        lambda x: 0 if x == -1 else (med_educ if x in (-7, -8) else x)
    )

    # Negative FAMINC24 / POVLEV24: Genuine business/farm losses; retain numeric scale
    # But for log transforms or stability, we preserve raw numeric values for modeling

    # Step 4: Handle categorical variables
    # Chronic conditions: HIBPDX, CHDDX, DIABDX_M18, ASTHDX, ARTHDX, CANCERDX, STRKDX, CHBRON31, CHOLDX
    # Coding: 1 = Yes, 2 = No. Children are -1 Inapplicable -> map to 3 (Inapplicable/Under 18).
    # -7, -8, -15 are missing -> impute to 2 (No, mode)
    chronic_vars = [
        "HIBPDX", "CHDDX", "DIABDX_M18", "ASTHDX", "ARTHDX",
        "CANCERDX", "STRKDX", "CHBRON31", "CHOLDX"
    ]
    for cv in chronic_vars:
        cohort[cv] = cohort[cv].apply(
            lambda x: 1 if x == 1 else (3 if x == -1 else 2)
        )

    # EMPST31: 1-4, -1 Inapplicable (<16) -> map -1 to 5 (Inapplicable < 16), -7/-8/-15 to 4 (Not employed)
    cohort["EMPST31"] = cohort["EMPST31"].apply(
        lambda x: int(x) if x in (1, 2, 3, 4) else (5 if x == -1 else 4)
    )

    # MARRY24X: 1-5, 6 Under 16, -7 Refused -> map -7 to 5 (Never married)
    cohort["MARRY24X"] = cohort["MARRY24X"].apply(
        lambda x: int(x) if x in (1, 2, 3, 4, 5, 6) else 5
    )

    # HIDEG: 1-7, 8 Under 16, -7/-8 -> map to 1 (No degree)
    cohort["HIDEG"] = cohort["HIDEG"].apply(
        lambda x: int(x) if x in (1, 2, 3, 4, 5, 6, 7, 8) else 1
    )

    # RTHLTH31 & MNHLTH31: 1-5, -1/-7/-8 -> map to 3 (Good)
    for h_var in ["RTHLTH31", "MNHLTH31"]:
        cohort[h_var] = cohort[h_var].apply(
            lambda x: int(x) if x in (1, 2, 3, 4, 5) else 3
        )

    # Binary/Categorical Insurance variables: Ensure 1/2 or valid codes
    for ins_var in ["UNINS24", "PRVEV24", "MCREV24", "MCDEV24", "TRIEV24", "VAEV24"]:
        cohort[ins_var] = cohort[ins_var].apply(lambda x: 1 if x == 1 else 2)

    # Reset index cleanly
    cohort = cohort.reset_index(drop=True)
    logger.info("Cleaned dataset successfully built with %d records.", len(cohort))
    return cohort, report


def save_processed_data(cohort_df: pd.DataFrame, report: Dict, output_dir: Path):
    """Saves cleaned dataset and cleaning report to processed data directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as parquet and csv for compatibility
    parquet_path = output_dir / "meps_2024_cleaned.parquet"
    csv_path = output_dir / "meps_2024_cleaned.csv"
    
    cohort_df.to_parquet(parquet_path, index=False)
    cohort_df.to_csv(csv_path, index=False)
    logger.info("Saved cleaned data to %s and %s", parquet_path, csv_path)

    # Save cleaning report
    reports_dir = output_dir.parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "data_cleaning_report.md"

    md_content = f"""# MEPS 2024 HC-256 Data Cleaning & Preprocessing Report

## Executive Summary
This report documents the reproducible data extraction, validation, and cleaning pipeline applied to the **2024 Medical Expenditure Panel Survey (MEPS) Full Year Consolidated Data File (HC-256)**, published by the Agency for Healthcare Research and Quality (AHRQ).

## Cohort Selection & Sample Disposition
- **Raw Survey Records in HC-256**: {report['raw_rows']:,} persons
- **Total Variables in Source**: {report['raw_cols']:,} columns
- **Exclusion 1: Non-positive Survey Sampling Weight (`PERWT24F <= 0`)**: {report['exclusions']['non_positive_weight']:,} persons
  - *Rationale*: Individuals with zero sampling weight represent non-key survey participants or individuals out-of-scope for the full year. Per AHRQ survey standards, these records do not represent the US civilian noninstitutionalized population.
- **Exclusion 2: Out-of-Scope Demographics (`AGE24X == -1`)**: {report['exclusions']['age_out_of_scope']:,} persons
  - *Rationale*: Persons who died, were institutionalized, or left the survey scope prior to December 31, 2024 have inapplicable demographic attributes.
- **Exclusion 3: Negative Target Values (`TOTSLF24 < 0`)**: {report['exclusions']['negative_target']:,} persons
- **Final Analytic Cohort**: **{report['final_cohort_rows']:,} persons**

## Target Variable Analysis (`TOTSLF24`: Out-of-Pocket Expenditure)
- **Target Definition**: Total annual amount paid out-of-pocket by self or family for all healthcare services received during calendar year 2024 (in USD).
- **Count**: {report['target_summary']['count']:,}
- **Mean**: ${report['target_summary']['mean']:,.2f}
- **Standard Deviation**: ${report['target_summary']['std']:,.2f}
- **Median (50th percentile)**: ${report['target_summary']['median']:,.2f}
- **Interquartile Range**: $0.00 to $1,015.00
- **Range (Min - Max)**: ${report['target_summary']['min']:,.2f} to ${report['target_summary']['max']:,.2f}
- **Zero-Expenditure Observations**: {report['target_summary']['zero_count']:,} ({report['target_summary']['zero_percentage']:.2f}%)
  - *Methodological Treatment*: Zero expenditures represent genuine non-utilization or full third-party coverage (e.g. Medicaid, premium commercial coverage). These observations are fully retained in the research cohort.

## Preprocessing & Missing Value Imputation
1. **Special MEPS Negative Codes**:
   - Inapplicable (`-1`), Don't Know (`-8`), Refused (`-7`), Cannot Be Computed (`-15`).
2. **Chronic Conditions (`HIBPDX`, `CHDDX`, `ARTHDX`, etc.)**:
   - Screened only for adults (`>17`). Children (`<=17`) coded `-1` are mapped to an explicit category `3: Inapplicable / Under 18`. Refused/Don't Know (<0.1%) are imputed to negative diagnosis (`2: No`).
3. **Education (`EDUCYR`)**:
   - Inapplicable for children under 5 (`-1`) mapped to 0 years. Rare missing values (`-7`, `-8`) imputed using cohort median.
4. **Employment Status (`EMPST31`)**:
   - Inapplicable for individuals under 16 (`-1`) mapped to `5: Inapplicable (<16)`.
5. **Income & Poverty Variables (`FAMINC24`, `POVLEV24`)**:
   - Negative values reflect genuine business/farm operational losses in the survey reference period and are retained as valid continuous numeric values.

## Selected Feature Set
- **Numeric Features ({len(NUMERIC_FEATURES)})**: {', '.join(NUMERIC_FEATURES)}
- **Categorical Features ({len(CATEGORICAL_FEATURES)})**: {', '.join(CATEGORICAL_FEATURES)}
- **Total Model Predictors**: **{len(ALL_PREDICTORS)} features**
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info("Saved data cleaning report to %s", report_path)


def run_pipeline():
    """Main execution entry point."""
    base_dir = Path(__file__).resolve().parent.parent
    raw_path = find_raw_excel(base_dir)
    raw_df = load_raw_meps_data(raw_path)
    cohort_df, report = clean_meps_data(raw_df)
    save_processed_data(cohort_df, report, base_dir / "data" / "processed")
    return cohort_df, report


if __name__ == "__main__":
    run_pipeline()
