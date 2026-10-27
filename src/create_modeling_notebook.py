"""
Script to create notebooks/02_modeling_and_conformal_prediction.ipynb using nbformat
"""

import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

cells = []

# Title & Context
cells.append(nbf.v4.new_markdown_cell("""# 02. Modeling & Normalized Split Conformal Prediction
## Uncertainty-Aware Predictive Modeling Framework for Healthcare Cost Prediction

**Author**: Senior Research Data Scientist  
**Dataset**: MEPS 2024 HC-256 (Agency for Healthcare Research and Quality)  
**Methodology**: Normalized Split Conformal Prediction (Papadopoulos et al., 2002; Angelopoulos & Bates, 2021)
"""))

# Imports
cells.append(nbf.v4.new_code_cell("""import sys
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

repo_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(repo_root))

from src.conformal_prediction import NormalizedSplitConformalPredictor
from src.visualization import create_conformal_interval_figure

print("Modules initialized successfully.")
"""))

# Load Model and Metrics
cells.append(nbf.v4.new_code_cell("""models_dir = repo_root / "models"
model_path = models_dir / "MEPS_NCP_webapp_model.joblib"
metrics_path = models_dir / "model_metrics.json"

predictor = joblib.load(model_path)
with open(metrics_path) as f:
    metrics = json.load(f)

print(f"Loaded trained conformal model: {predictor}")
print(f"Dataset records: {metrics['dataset_metadata']['total_cleaned_observations']:,}")
print(f"Train: {metrics['dataset_metadata']['train_observations']:,} | Cal: {metrics['dataset_metadata']['calibration_observations']:,} | Test: {metrics['dataset_metadata']['test_observations']:,}")
"""))

# Test Set Results
cells.append(nbf.v4.new_markdown_cell("""### 1. Test Set Empirical Coverage & Interval Metrics
We verify that the finite-sample calibrated prediction intervals satisfy the nominal 90% coverage on the untouched test partition.
"""))

cells.append(nbf.v4.new_code_cell("""eval_90 = metrics["test_evaluations"]["confidence_90"]

print("=== Conformal Test Set Metrics (Nominal 90% Target) ===")
print(f"Nominal Coverage:         {eval_90['nominal_coverage']:.1%}")
print(f"Empirical Test Coverage:  {eval_90['empirical_coverage']:.2%}")
print(f"Conformal Threshold (q):  {eval_90['conformal_quantile_q']:.4f}")
print(f"Mean Interval Width:      ${eval_90['mean_interval_width']:,.2f}")
print(f"Median Interval Width:    ${eval_90['median_interval_width']:,.2f}")
print(f"Min Interval Width:       ${eval_90['min_interval_width']:,.2f}")
print(f"Max Interval Width:       ${eval_90['max_interval_width']:,.2f}")
print(f"Test MAE:                 ${eval_90['mae']:,.2f}")
print(f"Test RMSE:                ${eval_90['rmse']:,.2f}")
print(f"Test R^2:                 {eval_90['r2']:.4f}")
print(f"Zero Spenders Coverage:   {eval_90['zero_spenders_coverage']:.2%}")
print(f"Non-Zero Spenders Cov:    {eval_90['nonzero_spenders_coverage']:.2%}")
"""))

# Subgroup Fairness
cells.append(nbf.v4.new_markdown_cell("""### 2. Subgroup Coverage & Interval Width Audit
Examining empirical performance across demographic and clinical strata.
"""))

cells.append(nbf.v4.new_code_cell("""# Age group performance
age_df = pd.DataFrame(metrics["subgroup_performance"]["Age_Group"])
age_df[["Age_Group", "Count", "Empirical_Coverage", "Mean_Interval_Width", "Median_Interval_Width", "MAE"]]
"""))

cells.append(nbf.v4.new_code_cell("""# Chronic condition comorbidity performance
chronic_df = pd.DataFrame(metrics["subgroup_performance"]["Chronic_Category"])
chronic_df[["Chronic_Category", "Count", "Empirical_Coverage", "Mean_Interval_Width", "Median_Interval_Width", "MAE"]]
"""))

# Interactive Interval Example
cells.append(nbf.v4.new_markdown_cell("""### 3. Patient Inference Demonstration
Demonstrating conformal prediction on a sample test patient record.
"""))

cells.append(nbf.v4.new_code_cell("""data_path = repo_root / "data" / "processed" / "meps_2024_cleaned.parquet"
df = pd.read_parquet(data_path)
sample_patient = df.iloc[[10]]

point_pred, lower, upper, width = predictor.predict(sample_patient, confidence_level=0.90)

print(f"Actual Out-of-Pocket:   ${sample_patient['TOTSLF24'].values[0]:,.2f}")
print(f"Point Estimate:         ${point_pred[0]:,.2f}")
print(f"90% Prediction Interval: [${lower[0]:,.2f}, ${upper[0]:,.2f}]")
print(f"Interval Width:         ${width[0]:,.2f}")
"""))

nb["cells"] = cells

notebooks_dir = Path("notebooks")
notebooks_dir.mkdir(parents=True, exist_ok=True)
nb_path = notebooks_dir / "02_modeling_and_conformal_prediction.ipynb"

with open(nb_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Created notebook at {nb_path}")
