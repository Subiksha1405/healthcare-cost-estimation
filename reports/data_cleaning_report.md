# MEPS 2024 HC-256 Data Cleaning & Preprocessing Report

## Executive Summary
This report documents the reproducible data extraction, validation, and cleaning pipeline applied to the **2024 Medical Expenditure Panel Survey (MEPS) Full Year Consolidated Data File (HC-256)**, published by the Agency for Healthcare Research and Quality (AHRQ).

## Cohort Selection & Sample Disposition
- **Raw Survey Records in HC-256**: 19,140 persons
- **Total Variables in Source**: 38 columns
- **Exclusion 1: Non-positive Survey Sampling Weight (`PERWT24F <= 0`)**: 457 persons
  - *Rationale*: Individuals with zero sampling weight represent non-key survey participants or individuals out-of-scope for the full year. Per AHRQ survey standards, these records do not represent the US civilian noninstitutionalized population.
- **Exclusion 2: Out-of-Scope Demographics (`AGE24X == -1`)**: 157 persons
  - *Rationale*: Persons who died, were institutionalized, or left the survey scope prior to December 31, 2024 have inapplicable demographic attributes.
- **Exclusion 3: Negative Target Values (`TOTSLF24 < 0`)**: 0 persons
- **Final Analytic Cohort**: **18,526 persons**

## Target Variable Analysis (`TOTSLF24`: Out-of-Pocket Expenditure)
- **Target Definition**: Total annual amount paid out-of-pocket by self or family for all healthcare services received during calendar year 2024 (in USD).
- **Count**: 18,526
- **Mean**: $1,200.14
- **Standard Deviation**: $4,800.81
- **Median (50th percentile)**: $212.00
- **Interquartile Range**: $0.00 to $1,015.00
- **Range (Min - Max)**: $0.00 to $317,942.00
- **Zero-Expenditure Observations**: 4,837 (26.11%)
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
- **Numeric Features (6)**: AGE24X, FAMINC24, POVLEV24, FAMSZE24, RUSIZE24, EDUCYR
- **Categorical Features (28)**: SEX, RACETHX, HISPANX, MARRY24X, HIDEG, REGION24, RUCLAS24, POVCAT24, INSCOV24, INSURC24, UNINS24, PRVEV24, MCREV24, MCDEV24, TRIEV24, VAEV24, RTHLTH31, MNHLTH31, HIBPDX, CHDDX, DIABDX_M18, ASTHDX, ARTHDX, CANCERDX, STRKDX, CHBRON31, CHOLDX, EMPST31
- **Total Model Predictors**: **34 features**
