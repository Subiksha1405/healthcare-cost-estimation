# Research Project Summary

**Title**: Uncertainty-Aware Predictive Modeling Framework for Healthcare Cost Prediction Using Conformal Prediction  
**Dataset**: MEPS 2024 HC-256 (Agency for Healthcare Research and Quality)  
**Final Test Empirical Coverage**: **90.91%** (Nominal: 90.0%)  
**Final Test Sample Size**: 3,706 patients  

## Key Research Contributions
1. **End-to-End Methodology**: Developed a fully reproducible, leak-free pipeline partitioning 18,526 in-scope MEPS participants into disjoint training (11,115), calibration (3,705), and test (3,706) sets.
2. **Addressing Skewness and Zero-Mass**: Implemented a mathematically rigorous log1p target transformation that accommodates exact zero spenders while shrinking skewness from +19.46 to -0.06.
3. **Adaptive Heteroscedastic Conformal Intervals**: Deployed Normalized Split Conformal Prediction with an auxiliary gradient boosted dispersion model $s(x)$, producing compact intervals for low-risk individuals and wider, realistic bounds for multimorbid patients.
4. **Transparent Clinical UI**: Developed an interactive Streamlit web application providing instant conformal intervals with Plotly visual diagnostics, human-readable medical terminology, and zero technical jargon.
