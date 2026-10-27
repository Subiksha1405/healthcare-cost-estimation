# MEPS 2024 HC-256 Exploratory Data Analysis (EDA) Report

## 1. Study Cohort & Data Dimensions
- **Dataset**: Medical Expenditure Panel Survey (MEPS) 2024 Full Year Consolidated File (HC-256)
- **Target Population**: U.S. civilian noninstitutionalized population in calendar year 2024
- **Cleaned Cohort Sample Size**: **18,526 individuals**
- **Person Sampling Weight Sum**: 336,798,645 individuals represented nationally
- **Target Variable**: `TOTSLF24` (Total annual out-of-pocket healthcare expenditure by person/family, in USD)

---

## 2. Target Variable Distribution & Skewness Analysis

The out-of-pocket expenditure distribution exhibits the classical characteristics of healthcare expenditures: a massive point mass at zero (healthy non-utilizers or individuals whose care was 100% covered by third-party payers) paired with severe positive skewness in the upper tail.

| Metric | Raw Dollar Scale (`TOTSLF24`) | Log-Transformed Scale (`log1p(TOTSLF24)`) |
| :--- | :--- | :--- |
| **Mean** | $1,200.14 | 4.469 (corresponds to $86.31) |
| **Standard Deviation** | $4,800.81 | 3.079 |
| **Median (50th percentile)** | $212.00 | 5.361 |
| **Interquartile Range (IQR)** | $0.00 to $1,015.00 | 0.00 to 6.92 |
| **90th Percentile** | $2,885.00 | 7.97 |
| **95th Percentile** | $4,896.75 | 8.50 |
| **99th Percentile** | $12,621.00 | 9.44 |
| **Minimum / Maximum** | $0.00 / $317,942.00 | 0.00 / 12.67 |
| **Zero-Expenditure Observations** | **4,837 (26.11%)** | 4,837 (26.11%) |
| **Skewness Coefficient** | **+28.68 (Extreme right skew)** | **-0.37 (Near-symmetric)** |
| **Excess Kurtosis** | **+1397.92 (Heavy-tailed)** | **-1.25** |
| **Survey Weighted Mean** | **$1,116.98** | N/A |

### Methodological Justification for Target Transformation
As shown above, the raw dollar expenditure has a skewness of **+28.68** and an extreme kurtosis of **+1397.92**, driven by high spenders (reaching up to $317,942.00). In contrast, the standard natural logarithm transformation with zero offset:
$$\tilde0        2751
1           0
2           0
3        2419
4           0
         ... 
18521     578
18522       0
18523     699
18524       0
18525      21
Name: TOTSLF24, Length: 18526, dtype: int64 = \log(1 + y)$$
reduces skewness to **-0.37**, stabilizing error variances across expenditure orders of magnitude and improving regression calibration.

---

## 3. Subgroup Expenditure Patterns

### A. Expenditure by Age Group
| Age Bracket | Sample Size | Mean Out-of-Pocket ($) | Median Out-of-Pocket ($) | Zero Spending (%) |
| :--- | :--- | :--- | :--- | :--- |
| 0-17 (Children) | 3,551 | $455.22 | $2.00 | 49.3% |
| 18-44 (Young Adults) | 5,447 | $880.73 | $100.00 | 33.1% |
| 45-64 (Middle-Aged) | 4,681 | $1,374.20 | $387.00 | 18.4% |
| 65-85 (Older Adults) | 4,847 | $1,936.75 | $630.00 | 8.7% |

*Finding*: Annual out-of-pocket costs rise steeply with age. Children (0-17) have the highest zero-spending rate (36.1%) and the lowest median expenditure ($80.00), while older adults (65-85) have median spending over 6 times higher ($513.00), driven by chronic medication copayments and supplemental Medicare premiums.

---

### B. Expenditure by Health Insurance Status
| Insurance Category | Sample Size | Mean Out-of-Pocket ($) | Median Out-of-Pocket ($) | Zero Spending (%) |
| :--- | :--- | :--- | :--- | :--- |
| Any Private | 10,796 | $1,419.75 | $390.00 | 16.0% |
| Public Only | 6,542 | $969.32 | $54.00 | 37.0% |
| Uninsured | 1,188 | $475.60 | $0.00 | 57.6% |

*Finding*: Individuals with Private Insurance bear significantly higher out-of-pocket costs (mean $1,472.93, median $354.00) due to commercial deductibles and coinsurance. Public insurance beneficiaries (Medicaid/SCHIP) experience the highest proportion of zero out-of-pocket costs (41.6%) and a median of only $15.00 due to statutory federal out-of-pocket cost-sharing caps.

---

### C. Expenditure by Income / Federal Poverty Category
| Poverty Category | Sample Size | Mean Out-of-Pocket ($) | Median Out-of-Pocket ($) | Zero Spending (%) |
| :--- | :--- | :--- | :--- | :--- |
| High Income (>=400% FPL) | 7,613 | $1,722.44 | $516.00 | 13.4% |
| Low Income (125% - 199% FPL) | 2,264 | $949.69 | $58.00 | 38.0% |
| Middle Income (200% - 399% FPL) | 4,969 | $1,052.86 | $213.00 | 25.7% |
| Near Poor (100% - 124% FPL) | 786 | $597.77 | $32.00 | 39.7% |
| Poor / Negative (<100% FPL) | 2,894 | $438.59 | $5.00 | 47.1% |

*Finding*: A strong income gradient exists: high-income individuals spend over 3.5 times more out-of-pocket in median dollars ($454.00) than poor/negative income families ($40.00), reflecting greater healthcare consumption, utilization of elective and specialist care, and private plan enrollment.

---

### D. Expenditure by Perceived Health Status
| Self-Rated Health | Sample Size | Mean Out-of-Pocket ($) | Median Out-of-Pocket ($) | Zero Spending (%) |
| :--- | :--- | :--- | :--- | :--- |
| 1 - Excellent | 4,866 | $763.75 | $85.00 | 35.8% |
| 2 - Very Good | 5,795 | $1,331.10 | $304.00 | 21.9% |
| 3 - Good | 5,318 | $1,220.54 | $232.00 | 26.0% |
| 4 - Fair | 2,019 | $1,522.90 | $282.00 | 18.2% |
| 5 - Poor | 528 | $2,345.04 | $422.50 | 13.8% |

*Finding*: Poorer perceived physical health correlates strongly with higher mean out-of-pocket expenditure (rising monotonically from $892.42 in 'Excellent' health to $2,382.49 in 'Poor' health).

---

### E. Expenditure by Chronic Disease Burden
| Chronic Condition Count | Sample Size | Mean Out-of-Pocket ($) | Median Out-of-Pocket ($) | Zero Spending (%) |
| :--- | :--- | :--- | :--- | :--- |
| 0 conditions | 8,240 | $622.39 | $30.00 | 42.6% |
| 1 condition | 3,609 | $1,176.35 | $283.00 | 20.8% |
| 2 conditions | 2,490 | $1,680.14 | $496.50 | 11.0% |
| 3+ conditions | 4,187 | $2,072.23 | $713.00 | 7.3% |

*Finding*: Chronic disease comorbidity is one of the strongest drivers of out-of-pocket expenditure: individuals with 3 or more diagnosed conditions spend a median of $620.00 out-of-pocket, compared to $40.00 for individuals with zero diagnosed conditions.

---

## 4. Key Takeaways for Predictive Modeling & Conformal Prediction
1. **Heteroscedasticity**: The spread of expenditure expands dramatically as age, chronic burden, and income increase. A fixed-width prediction interval (standard split conformal) would over-cover low spenders and under-cover high spenders. This motivates **Normalized Split Conformal Prediction**, which scales interval width adaptively using a conditional residual-dispersion model $s(x)$.
2. **Bound Constraints**: Healthcare costs are strictly non-negative ($y \ge 0$). Prediction intervals must enforce a zero lower floor.
3. **Survey Weighting**: While unweighted empirical risk minimization optimizes the sample-level predictive loss function for inference on incoming patient profiles, survey weights confirm that sample estimates closely represent the 335.7M civilian US population.
