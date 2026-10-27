# Uncertainty-Aware Predictive Modeling Framework for Healthcare Cost Prediction Using Conformal Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 11 Passed](https://img.shields.io/badge/pytest-11%20passed-brightgreen.svg)](tests/)
[![Dataset: MEPS HC-256](https://img.shields.io/badge/MEPS-2024%20HC--256-orange.svg)](https://meps.ahrq.gov/)

An end-to-end statistical research framework and interactive web application implementing **Normalized Split Conformal Prediction** on the **Medical Expenditure Panel Survey (MEPS) 2024 Full Year Consolidated Data File (HC-256)**, published by the Agency for Healthcare Research and Quality (AHRQ).

---

## 1. Research Overview & Problem Statement

Annual out-of-pocket healthcare expenditure exhibits extreme right-skewness (kurtosis > 800) and zero-inflation (> 26% of Americans incur zero out-of-pocket costs). Traditional point-prediction regression models (e.g., standard linear models, random forests) produce single deterministic figures (such as *$1,200/year*) that fail to communicate severe uncertainty, tail risk, and heteroscedasticity across health conditions.

This research project introduces an uncertainty-aware framework using **Normalized Split Conformal Prediction**:
1. **Distribution-Free Finite-Sample Guarantees**: Under sample exchangeability, the constructed intervals achieve exact nominal coverage ($1 - \alpha$, e.g., 90%) without requiring normal errors or homoscedasticity.
2. **Adaptive Dispersion Scaling**: An auxiliary gradient-boosted residual dispersion model $\hat{s}(x)$ dynamically widens intervals for complex multimorbid patients while producing tight, actionable bounds for healthy individuals.
3. **Bound Constraints**: Enforces a strict non-negative expenditure lower bound ($y \ge 0$).

---

## 2. Dataset & Target Specification

- **Source**: Agency for Healthcare Research and Quality (AHRQ), U.S. Department of Health and Human Services.
- **Survey Release**: 2024 Full Year Consolidated Data File, Public Use File **HC-256**.
- **Target Variable**: `TOTSLF24` — Total annual amount paid out-of-pocket by self or family for all healthcare services received during calendar year 2024 (in USD).
- **Target Characteristics**:
  - Sample Mean: **$1,200.14** | Median: **$212.00**
  - Zero-Expenditure Proportion: **26.11%** ($n = 4,837$)
  - Maximum Observed Out-of-Pocket: **$317,942.00**
  - Skewness before transformation: **+19.46** (extreme right skew)
  - Skewness after $\log(1 + y)$ transformation: **-0.06** (near symmetric)

---

## 3. Cohort Selection & Feature Engineering

From the initial 19,140 survey participants, an analytic cohort of **18,526 individuals** was established under strict AHRQ survey standards:

| Filter Criterion | Excluded Rows | Methodological Rationale |
| :--- | :--- | :--- |
| **`PERWT24F <= 0`** | 457 | Non-positive person weight; out-of-scope/non-key respondents. |
| **`AGE24X == -1`** | 157 | Inapplicable age; deceased or institutionalized prior to 12/31/2024. |
| **`TOTSLF24 < 0`** | 0 | Negative expenditure check (no negative expenditures present). |
| **Final Usable Cohort** | **18,526** | **Represents 335.7 million U.S. civilian noninstitutionalized residents.** |

### Verified Predictor Features (34 Predictors)
- **Demographics (8)**: `AGE24X` (Age), `SEX` (Sex), `RACETHX` (Race/Ethnicity), `HISPANX` (Hispanic), `MARRY24X` (Marital Status), `HIDEG` (Highest Degree), `EDUCYR` (Years of Education), `REGION24` (Census Region).
- **Household & Financial (7)**: `FAMINC24` (Family Income USD), `FAMSZE24` (Family Size), `RUSIZE24` (RU Size), `RUCLAS24` (RU Class), `POVCAT24` (Poverty Category), `POVLEV24` (% of Poverty Line), `EMPST31` (Employment Status).
- **Health Insurance (8)**: `INSCOV24` (Coverage Indicator), `INSURC24` (Full-Year Insurance Status), `UNINS24` (Uninsured Flag), `PRVEV24` (Private), `MCREV24` (Medicare), `MCDEV24` (Medicaid), `TRIEV24` (TRICARE), `VAEV24` (VA).
- **Health Status & Chronic Diseases (11)**: `RTHLTH31` (Self-Rated Physical Health), `MNHLTH31` (Mental Health), `HIBPDX` (Hypertension), `CHDDX` (Coronary Heart Disease), `DIABDX_M18` (Diabetes), `ASTHDX` (Asthma), `ARTHDX` (Arthritis), `CANCERDX` (Cancer), `STRKDX` (Stroke), `CHBRON31` (Chronic Bronchitis), `CHOLDX` (High Cholesterol).

---

## 4. Normalized Split Conformal Prediction Methodology

The cohort was partitioned into three strictly disjoint subsets:
- **Training Partition (60% | $n = 11,115$)**: Fits the point predictor $f(x)$ on $\log(1 + y)$ and fits the dispersion model $s(x)$ on log-absolute training residuals $\log(|y - f(x)| + \epsilon)$.
- **Calibration Partition (20% | $n = 3,705$)**: Computes normalized nonconformity scores:
  $$\alpha_j = \frac{|\log(1 + y_j) - f(x_j)|}{\max(s(x_j), \sigma_{\text{floor}})}$$
  The finite-sample conformal quantile for target coverage $1 - \alpha$ is:
  $$q = \text{Quantile}_{\lceil (n_{\text{cal}} + 1)(1 - \alpha) \rceil / n_{\text{cal}}}\left(\{\alpha_j\}_{j=1}^{n_{\text{cal}}}\right)$$
- **Untouched Test Partition (20% | $n = 3,706$)**: Reserved strictly for final out-of-sample evaluation.

Inference produces instance-specific, non-negative intervals:
$$C(x) = \left[ \max\left(0, \exp(f(x) - q \cdot s(x)) - 1\right), \; \exp(f(x) + q \cdot s(x)) - 1 \right]$$

---

## 5. Empirical Results on Untouched Test Set

| Evaluation Metric | Nominal Target | Observed Test Value | Assessment |
| :--- | :--- | :--- | :--- |
| **Marginal Empirical Coverage** | **90.0%** | **90.91%** | **Valid (Exceeds Nominal Guarantee)** |
| **Conformal Threshold ($q$)** | N/A | **2.9427** | Calibrated on $n = 3,705$ |
| **Mean Absolute Error (MAE)** | Baseline: $1,178 | **$904.65** | $274/year error reduction |
| **Root Mean Squared Error (RMSE)** | Baseline: $4,289 | **$3,017.05** | Outlier-penalized accuracy |
| **Median Interval Width** | N/A | **$6,249.64** | Typical patient width |
| **Mean Interval Width** | N/A | **$8,272.77** | Expands for complex patients |
| **Zero-Spenders Coverage ($n=1,001$)** | 90.0% | **81.82%** | Bounded at $0.00 |
| **Non-Zero Spenders Coverage ($n=2,705$)**| 90.0% | **94.27%** | Strong positive coverage |

### Alternative Confidence Levels
- **80% Nominal Target**: Empirical Coverage = **81.54%** | Median Width = **$4,089.47**
- **95% Nominal Target**: Empirical Coverage = **95.66%** | Median Width = **$8,460.62**

---

## 6. Project Directory Structure

```text
healthcare-cost-estimation/
│
├── app.py                             # Complete Streamlit Web Application
├── requirements.txt                   # Production dependencies
├── README.md                          # Comprehensive research documentation
├── .gitignore                         # Git exclusion rules
│
├── data/
│   ├── raw/
│   │   └── h256.xlsx                  # Original MEPS HC-256 Excel dataset
│   └── processed/
│       ├── meps_2024_cleaned.parquet  # High-performance cleaned cohort
│       └── meps_2024_cleaned.csv      # Standard CSV cleaned cohort
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Exploratory analysis notebook
│   └── 02_modeling_and_conformal_prediction.ipynb # Conformal prediction notebook
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py          # Extraction, cleaning & schema validation
│   ├── conformal_prediction.py        # NormalizedSplitConformalPredictor implementation
│   ├── evaluation.py                  # Subgroup fairness & metric auditing
│   ├── visualization.py               # Interactive Plotly figures & interval chart
│   ├── train_model.py                 # Training, calibration & artifact saving
│   ├── generate_eda_report.py         # Empirical EDA report compilation
│   ├── create_eda_notebook.py         # EDA notebook generator
│   └── create_modeling_notebook.py    # Modeling notebook generator
│
├── models/
│   ├── MEPS_NCP_webapp_model.joblib   # Trained conformal predictor (690 KB)
│   ├── MEPS_feature_options.json      # Codebook value mappings & defaults
│   └── model_metrics.json             # Empirical test metrics & subgroup tables
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py          # Data cleaning & schema tests
│   ├── test_model.py                  # Predictor loading & finite prediction tests
│   ├── test_conformal.py              # Finite-sample quantile & coverage tests
│   └── test_app.py                    # Smoke test on real patient records & JSON
│
└── reports/
    ├── data_cleaning_report.md        # Cohort selection & exclusion report
    ├── eda_report.md                  # Statistical EDA & skewness findings
    ├── model_evaluation_report.md     # Test set metrics & subgroup audits
    └── research_summary.md            # Academic research executive summary
```

---

## 7. Installation & Reproducibility Guide

### Step 1: Create and Activate Virtual Environment (Windows)
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Step 2: Install Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 3: Run Automated Tests
```powershell
python -m pytest -v
```
*(Expected output: 11 passed in ~11 seconds)*

### Step 4: Run Training & Conformal Calibration (Optional - Artifacts Included)
```powershell
python -m src.train_model
```

### Step 5: Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 8. Streamlit Web Application Features

1. **Home**: Overview of research motivation, why single point predictions fail, dataset documentation, and research prototype disclaimers.
2. **Estimate Cost**: Multi-section clinical input form covering Demographics, Household Structure & Income, Health Insurance, and Diagnosed Chronic Conditions.
3. **Prediction Results**:
   - Three primary metric cards: **Central Point Estimate**, **Lower Conformal Bound**, and **Upper Conformal Bound**.
   - Interactive **Plotly Conformal Interval Chart** with diamond point marker and green interval bar.
   - Plain-language explanation of uncertainty.
   - **Download CSV Report** containing profile and interval metadata.
4. **Research & Model**: Interactive metrics dashboard reading directly from `model_metrics.json`, displaying empirical test coverage (90.91%), error distributions, and subgroup audit tabs (Age, Insurance, Comorbidities).
5. **How to Read My Estimate**: Educational guide explaining prediction intervals, heteroscedasticity, and clinical limitations.

---

## 9. Research Integrity & Limitations

1. **U.S. Healthcare System Specificity**: MEPS is a survey of the U.S. civilian noninstitutionalized population. Insurance cost-sharing, deductibles, and prices reflect U.S. healthcare structures. **This framework is not calibrated or valid for Indian or international healthcare costs.**
2. **Non-Causal Associations**: Observed relationships between chronic conditions and expenditures reflect observational statistical associations, not causal effects.
3. **Marginal vs. Conditional Coverage**: Split conformal prediction guarantees marginal coverage over the population distribution. While our normalized scale model $\hat{s}(x)$ adapts interval width, exact conditional coverage for every individual sub-population is an active area of statistical research.
4. **Not Clinical or Billing Advice**: This application is a scientific research prototype and does not constitute medical diagnosis, treatment planning, health insurance underwriting, or binding hospital billing estimates.
