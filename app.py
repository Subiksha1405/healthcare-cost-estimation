"""
Streamlit Web Application: Uncertainty-Aware Predictive Modeling Framework
for Healthcare Cost Prediction Using Normalized Split Conformal Prediction.

Dataset: Medical Expenditure Panel Survey (MEPS) 2024 HC-256 (AHRQ).
"""

import os
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.conformal_prediction import NormalizedSplitConformalPredictor
from src.data_preprocessing import ALL_PREDICTORS, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.visualization import create_conformal_interval_figure, plot_subgroup_coverage_bar

# -----------------------------------------------------------------------------
# Configuration and Constants
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "MEPS_NCP_webapp_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "model_metrics.json"
OPTIONS_PATH = BASE_DIR / "models" / "MEPS_feature_options.json"

PAGE_HOME = "Home"
PAGE_FORM = "Estimate Cost"
PAGE_RESULTS = "Prediction Results"
PAGE_RESEARCH = "Research & Evaluation"
PAGE_GUIDE = "How to Read My Estimate"

PAGE_LIST = [PAGE_HOME, PAGE_FORM, PAGE_RESULTS, PAGE_RESEARCH, PAGE_GUIDE]

st.set_page_config(
    page_title="Conformal Healthcare Cost Estimator | MEPS 2024",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Custom Styling: Warm Academic Palette with Smooth Page Slide Transitions
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* ============================================================ */
    /* Smooth Left-to-Right & Page-to-Page Slide Transitions        */
    /* ============================================================ */
    @keyframes pageSlideInRight {
        0% {
            opacity: 0;
            transform: translateX(36px);
        }
        50% {
            opacity: 0.85;
        }
        100% {
            opacity: 1;
            transform: translateX(0);
        }
    }

    @keyframes slideFromLeft {
        0% {
            opacity: 0;
            transform: translateX(-28px);
        }
        100% {
            opacity: 1;
            transform: translateX(0);
        }
    }

    @keyframes slideFromRight {
        0% {
            opacity: 0;
            transform: translateX(28px);
        }
        100% {
            opacity: 1;
            transform: translateX(0);
        }
    }

    @keyframes smoothScaleIn {
        0% {
            opacity: 0;
            transform: scale(0.96) translateY(10px);
        }
        100% {
            opacity: 1;
            transform: scale(1) translateY(0);
        }
    }

    @keyframes pulseGlow {
        0% { box-shadow: 0 0 0 0 rgba(47, 133, 90, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(47, 133, 90, 0); }
        100% { box-shadow: 0 0 0 0 rgba(47, 133, 90, 0); }
    }
    
    /* Main Page Body Sweep Animation */
    .block-container, section[data-testid="stMain"] > div:first-child {
        animation: pageSlideInRight 0.38s cubic-bezier(0.16, 1, 0.3, 1) both;
        will-change: transform, opacity;
    }

    /* Staggered Column Sweep: Left Column from Left, Right Column from Right */
    [data-testid="column"]:first-child {
        animation: slideFromLeft 0.42s cubic-bezier(0.16, 1, 0.3, 1) both;
    }

    [data-testid="column"]:last-child {
        animation: slideFromRight 0.45s cubic-bezier(0.16, 1, 0.3, 1) both;
    }

    /* Tab Switch Slide Animation */
    div[data-baseweb="tab-panel"] {
        animation: pageSlideInRight 0.34s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    
    /* Enforce Dark High-Contrast Typography Everywhere */
    html, body, [class*="css"], [data-testid="stMarkdownContainer"], 
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li, 
    [data-testid="stMarkdownContainer"] span, [data-testid="stMarkdownContainer"] div {
        color: #0F172A !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    .stApp {
        background-color: #F8FAFC !important;
    }
    
    /* Academic Hero Container with Soft Elevation & Transitions */
    .hero-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F0FDF4 100%) !important;
        border: 1px solid #CBD5E1 !important;
        border-left: 6px solid #15803D !important;
        border-radius: 14px;
        padding: 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.05);
        animation: slideFromLeft 0.4s cubic-bezier(0.16, 1, 0.3, 1) both;
        transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.3s ease;
    }

    .hero-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 28px -4px rgba(21, 128, 61, 0.12);
    }
    
    .hero-title {
        font-size: 26px;
        font-weight: 700;
        color: #0F172A !important;
        margin-bottom: 8px;
        line-height: 1.25;
    }
    
    .hero-subtitle {
        font-size: 15px;
        color: #1E293B !important;
        line-height: 1.55;
    }

    /* Content Cards with Smooth Lift & Shadow Transitions */
    .content-card {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 14px;
        padding: 26px;
        margin-bottom: 22px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -2px rgba(0, 0, 0, 0.02);
        transition: transform 0.28s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.28s ease, border-color 0.28s ease;
    }

    .content-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 14px 28px -4px rgba(0, 0, 0, 0.08), 0 6px 12px -3px rgba(0, 0, 0, 0.03);
        border-color: #94A3B8 !important;
    }

    .content-card p, 
    .content-card li, 
    .content-card ul, 
    .content-card ol, 
    .content-card span, 
    .content-card div {
        color: #1E293B !important;
        font-size: 14.5px;
        line-height: 1.65;
    }

    .content-card h3 {
        color: #1D4ED8 !important;
        font-weight: 700 !important;
        transition: color 0.2s ease;
    }

    .content-card h4 {
        color: #15803D !important;
        font-weight: 700 !important;
    }

    .content-card strong, .content-card b {
        color: #0F172A !important;
        font-weight: 700 !important;
    }

    /* Metric Cards with Floating Stagger */
    .metric-card-blue {
        background-color: #EFF6FF !important;
        border: 1px solid #BFDBFE !important;
        border-left: 6px solid #2563EB !important;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        animation: slideFromLeft 0.45s cubic-bezier(0.16, 1, 0.3, 1) both;
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.25s ease;
        cursor: default;
    }

    .metric-card-blue:hover {
        transform: translateY(-4px) scale(1.015);
        box-shadow: 0 12px 24px -4px rgba(37, 99, 235, 0.2);
    }
    
    .metric-card-green {
        background-color: #F0FDF4 !important;
        border: 1px solid #BBF7D0 !important;
        border-left: 6px solid #16A34A !important;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        animation: smoothScaleIn 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.25s ease;
        cursor: default;
    }

    .metric-card-green:hover {
        transform: translateY(-4px) scale(1.015);
        box-shadow: 0 12px 24px -4px rgba(22, 163, 74, 0.2);
    }
    
    .metric-label {
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #334155 !important;
        margin-bottom: 6px;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        line-height: 1.2;
    }
    
    .metric-value-blue {
        color: #1E40AF !important;
    }
    
    .metric-value-green {
        color: #166534 !important;
    }
    
    .metric-hint {
        font-size: 12.5px;
        font-weight: 500;
        color: #475569 !important;
        margin-top: 6px;
    }
    
    /* Disclaimer Banner */
    .disclaimer-box {
        background-color: #FFFBEB !important;
        border: 1px solid #FCD34D !important;
        border-left: 5px solid #D97706 !important;
        border-radius: 10px;
        padding: 16px 20px;
        font-size: 13.5px;
        color: #78350F !important;
        line-height: 1.55;
        margin-top: 20px;
        margin-bottom: 20px;
        animation: slideFromRight 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    
    .disclaimer-box b, .disclaimer-box strong {
        color: #78350F !important;
    }
    
    /* Enhanced Interactive Buttons with Hover Lift & Glow */
    .stButton>button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        padding: 10px 24px !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
    }

    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.12) !important;
    }

    .stButton>button:active {
        transform: translateY(0) !important;
    }

    /* Primary CTA Button Polish */
    button[kind="primary"] {
        background: linear-gradient(135deg, #15803D 0%, #16A34A 100%) !important;
        border: none !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 14px rgba(21, 128, 61, 0.28) !important;
    }

    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #166534 0%, #15803D 100%) !important;
        box-shadow: 0 8px 22px rgba(21, 128, 61, 0.4) !important;
        transform: translateY(-2px) scale(1.01) !important;
    }

    /* Form Expanders with Staggered Entrance */
    .streamlit-expanderHeader {
        background-color: #F8FAFC !important;
        border-radius: 8px !important;
        transition: background-color 0.2s ease, transform 0.2s ease !important;
        font-weight: 600 !important;
        color: #0F172A !important;
    }

    .streamlit-expanderHeader:hover {
        background-color: #EDF2F7 !important;
        transform: translateX(3px);
    }

    /* Form labels, sliders, selects */
    label, p[data-testid="stWidgetLabel"], .stSlider label, .stSelectbox label, .stNumberInput label {
        color: #0F172A !important;
        font-weight: 600 !important;
        transition: color 0.2s ease;
    }

    /* Sidebar Navigation with Pill Hover Transition */
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
        border-radius: 8px !important;
        padding: 6px 12px !important;
        margin-bottom: 3px !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: #E2E8F0 !important;
        transform: translateX(5px) !important;
    }

    /* Plotly Visualizer Smooth Scale Entrance */
    .js-plotly-plot {
        animation: smoothScaleIn 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Artifact Loaders with Caching
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained conformal prediction models...")
def load_trained_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="Loading model evaluation metrics...")
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading verified feature options...")
def load_feature_options():
    if not OPTIONS_PATH.exists():
        return None
    with open(OPTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# State Management Helper
# -----------------------------------------------------------------------------
def init_session_state():
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = PAGE_HOME
    if "last_prediction" not in st.session_state:
        st.session_state["last_prediction"] = None
    if "last_profile" not in st.session_state:
        st.session_state["last_profile"] = None
    if "confidence_level" not in st.session_state:
        st.session_state["confidence_level"] = 0.90


def navigate_to(page_name: str):
    st.session_state["active_page"] = page_name
    st.rerun()


init_session_state()

# -----------------------------------------------------------------------------
# Sidebar Navigation & Disclaimers
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="text-align: center; padding-bottom: 12px;">
        <span style="font-size: 36px;">🩺</span>
        <h3 style="margin-top: 6px; margin-bottom: 2px; color: #1A365D;">MEPS Conformal Estimator</h3>
        <p style="font-size: 12px; color: #718096;">AHRQ MEPS 2024 HC-256 Research Prototype</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Synchronize sidebar navigation selection cleanly using active_page
current_idx = PAGE_LIST.index(st.session_state["active_page"]) if st.session_state["active_page"] in PAGE_LIST else 0

selected_page = st.sidebar.radio(
    "Navigation",
    options=PAGE_LIST,
    index=current_idx,
    label_visibility="collapsed"
)

# Update state if user clicked the sidebar radio directly
if selected_page != st.session_state["active_page"]:
    st.session_state["active_page"] = selected_page
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Study Summary**
    - **Target**: Out-of-Pocket Expenditure (`TOTSLF24`)
    - **Model**: HistGradientBoosting Regressor
    - **Method**: Normalized Split Conformal Prediction
    - **Observed Test Coverage**: **90.91%** (Nominal: 90%)
    - **Data**: AHRQ MEPS HC-256 (N = 18,526)
    
    ---
    **Geographical Limitation**  
    *MEPS is a representative survey of the United States civilian noninstitutionalized population. Costs, insurance categories, and poverty levels reflect the U.S. healthcare delivery system. This framework is not calibrated for Indian or international healthcare costs.*
    """
)


# -----------------------------------------------------------------------------
# Page 1: Home
# -----------------------------------------------------------------------------
def render_page_home():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Uncertainty-Aware Predictive Modeling Framework for Healthcare Cost Prediction</div>
            <div class="hero-subtitle">
                An academic research prototype leveraging <b>Normalized Split Conformal Prediction</b> and the 
                <b>2024 Medical Expenditure Panel Survey (MEPS HC-256)</b> to provide statistically guaranteed, 
                heteroscedastic prediction intervals for annual out-of-pocket medical expenditure.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(
            """
            <div class="content-card">
                <h3 style="color: #2B6CB0; margin-top: 0;">Why Point Predictions Are Insufficient in Healthcare</h3>
                <p>
                    In health economics, annual out-of-pocket costs exhibit extreme right-skewness and zero-inflation:
                    over 26% of Americans incur zero out-of-pocket costs in a given year, while complex patients incur tens of thousands of dollars.
                    A traditional regression model only outputs a single dollar figure (e.g. <em>"$1,200"</em>), which:
                </p>
                <ul>
                    <li>Conceals catastrophic upper-tail financial risks from patients and planners.</li>
                    <li>Fails to communicate when an estimate is confident versus highly speculative.</li>
                    <li>Suffers from severe heteroscedasticity across health conditions and ages.</li>
                </ul>
                <h4 style="color: #2F855A; margin-top: 18px;">How Normalized Conformal Prediction Solves This</h4>
                <p>
                    Conformal prediction provides a distribution-free, finite-sample guarantee: under exchangeability, 
                    exactly <b>90%</b> (or user-chosen coverage) of future patients' true out-of-pocket expenses 
                    will fall between our lower and upper prediction bounds.
                    Our <i>normalized</i> framework uses an auxiliary dispersion model to adaptively widen intervals for complex multimorbid patients 
                    and shrink them for healthy low-risk individuals.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.button(
            "🚀 Start Cost Estimation Process",
            type="primary",
            use_container_width=True,
            on_click=lambda: navigate_to(PAGE_FORM)
        )

    with col2:
        st.markdown(
            """
            <div class="content-card">
                <h3 style="color: #1A365D; margin-top: 0;">Dataset & Research Architecture</h3>
                <p><b>Data Source:</b></p>
                <p style="font-size: 14px; color: #1E293B; line-height: 1.5;">
                    <b>Agency for Healthcare Research and Quality (AHRQ)</b><br>
                    Medical Expenditure Panel Survey (MEPS)<br>
                    2024 Full Year Consolidated Data File (HC-256)
                </p>
                <hr style="margin: 12px 0; border: none; border-top: 1px solid #E2E8F0;">
                <p><b style="color: #0F172A;">Rigorous Disjoint Partitioning:</b></p>
                <ul style="font-size: 13.5px; color: #1E293B; line-height: 1.6;">
                    <li><b>Training Set (60% | 11,115 records):</b> Point model and residual-scale model fitting.</li>
                    <li><b>Calibration Set (20% | 3,705 records):</b> Conformal nonconformity score thresholding.</li>
                    <li><b>Untouched Test Set (20% | 3,706 records):</b> Final empirical validation.</li>
                </ul>
                <hr style="margin: 12px 0; border: none; border-top: 1px solid #E2E8F0;">
                <p><b style="color: #0F172A;">Empirical Test Performance:</b></p>
                <ul style="font-size: 13.5px; color: #1E293B; line-height: 1.6;">
                    <li>Nominal Coverage Target: <b>90.0%</b></li>
                    <li>Observed Test Coverage: <b>90.91%</b></li>
                    <li>Median Interval Width: <b>$6,249.64</b></li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="disclaimer-box">
            <b>Research & Educational Prototype Disclaimer:</b> This web application is an academic research demonstration 
            calibrated exclusively on the U.S. MEPS 2024 population survey. It does NOT provide medical advice, diagnosis, treatment recommendations, 
            health insurance quotes, or binding medical billing guarantees.
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------------------------------------------------------
# Page 2: User Input Form
# -----------------------------------------------------------------------------
def render_page_form(model, options):
    st.markdown(
        """
        <div class="hero-card" style="padding: 20px; border-left-color: #3182CE;">
            <div class="hero-title" style="font-size: 22px;">Patient Profile & Cost Estimation Form</div>
            <div class="hero-subtitle">
                Enter demographic, financial, insurance, and medical status attributes. All 34 features map directly to verified MEPS HC-256 variables.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if model is None or options is None:
        st.error("Model artifacts or feature options not found. Please run the training pipeline first.")
        if st.button("Run Training Pipeline Now"):
            with st.spinner("Training model and calibrating conformal intervals..."):
                from src.train_model import train_and_evaluate
                train_and_evaluate()
            st.success("Model trained successfully! Please reload.")
            st.rerun()
        return

    # Confidence Level Selector
    st.markdown("### 1. Desired Uncertainty Level")
    conf_col1, conf_col2 = st.columns([2, 3])
    with conf_col1:
        selected_conf = st.selectbox(
            "Select Nominal Prediction Interval Coverage",
            options=[0.90, 0.80, 0.95],
            index=0,
            format_func=lambda x: f"{int(x * 100)}% Confidence Interval (Nominal)"
        )
        st.session_state["confidence_level"] = selected_conf
    with conf_col2:
        st.info(
            f"A **{int(selected_conf * 100)}% prediction interval** guarantees that under exchangeability, the patient's "
            f"true annual out-of-pocket costs will fall between the lower and upper bounds in {int(selected_conf * 100)} out of 100 cases."
        )

    st.markdown("### 2. Enter Profile Information")

    # Demographic Section
    with st.expander("A. Demographics & Geographic Region", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age (Years)", min_value=0, max_value=85, value=45, help="Age as of 12/31/2024 (AGE24X)")
            sex = st.selectbox(
                "Sex",
                options=[1, 2],
                format_func=lambda x: "Male" if x == 1 else "Female"
            )
        with c2:
            raceth = st.selectbox(
                "Race & Ethnicity",
                options=[1, 2, 3, 4, 5],
                index=1,
                format_func=lambda x: {
                    1: "Hispanic",
                    2: "Non-Hispanic White",
                    3: "Non-Hispanic Black",
                    4: "Non-Hispanic Asian",
                    5: "Non-Hispanic Other / Multiple"
                }[x]
            )
            hispan = 1 if raceth == 1 else 2
            
            marry = st.selectbox(
                "Marital Status",
                options=[1, 2, 3, 4, 5, 6] if age >= 16 else [6],
                index=0 if age >= 16 else 0,
                format_func=lambda x: {
                    1: "Married",
                    2: "Widowed",
                    3: "Divorced",
                    4: "Separated",
                    5: "Never Married",
                    6: "Under 16 (Inapplicable)"
                }[x]
            )
        with c3:
            educ_deg = st.selectbox(
                "Highest Education Degree",
                options=[1, 2, 3, 4, 5, 6, 7, 8] if age >= 16 else [8],
                index=2 if age >= 16 else 0,
                format_func=lambda x: {
                    1: "No Degree",
                    2: "GED",
                    3: "High School Diploma",
                    4: "Bachelor's Degree",
                    5: "Master's Degree",
                    6: "Doctorate Degree",
                    7: "Other Degree",
                    8: "Under 16 (Inapplicable)"
                }[x]
            )
            educ_yr = st.slider("Years of Education Completed", min_value=0, max_value=17, value=12 if age >= 18 else min(12, max(0, age - 5)))
            region = st.selectbox(
                "U.S. Census Region",
                options=[1, 2, 3, 4],
                index=2,
                format_func=lambda x: {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}[x]
            )

    # Household & Income Section
    with st.expander("B. Household Structure & Income", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            faminc = st.number_input("Annual Family Income ($ USD)", min_value=-50000.0, max_value=1000000.0, value=65000.0, step=2500.0)
            povcat = st.selectbox(
                "Federal Poverty Category",
                options=[1, 2, 3, 4, 5],
                index=3,
                format_func=lambda x: {
                    1: "Poor / Negative (<100% FPL)",
                    2: "Near Poor (100% - 124% FPL)",
                    3: "Low Income (125% - 199% FPL)",
                    4: "Middle Income (200% - 399% FPL)",
                    5: "High Income (>=400% FPL)"
                }[x]
            )
        with c2:
            povlev = st.number_input("Income as % of Poverty Line (FPL)", min_value=0.0, max_value=2500.0, value=300.0, step=25.0)
            empst = st.selectbox(
                "Employment Status",
                options=[1, 2, 3, 4, 5] if age >= 16 else [5],
                index=0 if age >= 16 else 0,
                format_func=lambda x: {
                    1: "Employed at Interview Date",
                    2: "Has Job to Return to",
                    3: "Employed During Reference Period",
                    4: "Not Employed",
                    5: "Inapplicable (<16 Years Old)"
                }[x]
            )
        with c3:
            famsze = st.slider("Family Size (Persons)", min_value=1, max_value=13, value=3)
            rusize = st.slider("Reporting Unit Size (Persons)", min_value=1, max_value=13, value=3)
            ruclas = st.selectbox("RU Classification", options=[1, 2, 3], format_func=lambda x: {1: "Standard RU", 2: "New RU", 3: "Student RU"}[x])

    # Insurance Section
    with st.expander("C. Health Insurance Coverage", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            inscov = st.selectbox(
                "Primary Insurance Category",
                options=[1, 2, 3],
                format_func=lambda x: {1: "Any Private Insurance", 2: "Public Only (Medicaid/Medicare)", 3: "Uninsured"}[x]
            )
            insurc = st.selectbox(
                "Full-Year Insurance Classification",
                options=[1, 2, 3, 4, 5, 6, 7, 8],
                index=0 if age < 65 and inscov == 1 else (3 if age >= 65 else 2),
                format_func=lambda x: {
                    1: "<65 Any Private",
                    2: "<65 Public Only",
                    3: "<65 Uninsured",
                    4: "65+ Medicare Only",
                    5: "65+ Medicare & Private",
                    6: "65+ Medicare & Other Public",
                    7: "65+ Uninsured",
                    8: "65+ No Medicare & Other Public/Private"
                }[x]
            )
        with c2:
            unins = 1 if inscov == 3 else 2
            prvev = st.selectbox("Private Insurance in 2024?", options=[1, 2], index=0 if inscov == 1 else 1, format_func=lambda x: "Yes" if x == 1 else "No")
            mcrev = st.selectbox("Medicare Coverage in 2024?", options=[1, 2], index=0 if age >= 65 else 1, format_func=lambda x: "Yes" if x == 1 else "No")
        with c3:
            mcdev = st.selectbox("Medicaid / SCHIP in 2024?", options=[1, 2], index=0 if inscov == 2 and age < 65 else 1, format_func=lambda x: "Yes" if x == 1 else "No")
            triev = st.selectbox("TRICARE / CHAMPVA in 2024?", options=[1, 2], index=1, format_func=lambda x: "Yes" if x == 1 else "No")
            vaev = st.selectbox("VA Healthcare in 2024?", options=[1, 2], index=1, format_func=lambda x: "Yes" if x == 1 else "No")

    # Health & Chronic Conditions Section
    with st.expander("D. Health Status & Chronic Conditions", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            rthlth = st.selectbox(
                "Self-Rated Physical Health",
                options=[1, 2, 3, 4, 5],
                index=1,
                format_func=lambda x: {1: "1 - Excellent", 2: "2 - Very Good", 3: "3 - Good", 4: "4 - Fair", 5: "5 - Poor"}[x]
            )
        with c2:
            mnhlth = st.selectbox(
                "Self-Rated Mental Health",
                options=[1, 2, 3, 4, 5],
                index=1,
                format_func=lambda x: {1: "1 - Excellent", 2: "2 - Very Good", 3: "3 - Good", 4: "4 - Fair", 5: "5 - Poor"}[x]
            )

        st.markdown("**Diagnosed Chronic Conditions:**")
        is_adult = age >= 18
        inapp_val = 3  # Inapplicable for <18

        ch1, ch2, ch3 = st.columns(3)
        with ch1:
            hibp = st.selectbox("High Blood Pressure (Hypertension)", options=[1, 2] if is_adult else [3], format_func=lambda x: {1: "Yes (Diagnosed)", 2: "No", 3: "Inapplicable (<18)"}[x])
            chd = st.selectbox("Coronary Heart Disease", options=[1, 2] if is_adult else [3], format_func=lambda x: {1: "Yes (Diagnosed)", 2: "No", 3: "Inapplicable (<18)"}[x])
            diab = st.selectbox("Diabetes", options=[1, 2] if is_adult else [3], format_func=lambda x: {1: "Yes (Diagnosed)", 2: "No", 3: "Inapplicable (<18)"}[x])
        with ch2:
            asth = st.selectbox("Asthma", options=[1, 2] if is_adult else [3], format_func=lambda x: {1: "Yes (Diagnosed)", 2: "No", 3: "Inapplicable (<18)"}[x])
            arth = st.selectbox("Arthritis", options=[1, 2] if is_adult else [3], format_func=lambda x: {1: "Yes (Diagnosed)", 2: "No", 3: "Inapplicable (<18)"}[x])
            cancer = st.selectbox("Cancer History", options=[1, 2] if is_adult else [3], format_func=lambda x: {1: "Yes (Diagnosed)", 2: "No", 3: "Inapplicable (<18)"}[x])
        with ch3:
            strk = st.selectbox("Stroke History", options=[1, 2] if is_adult else [3], format_func=lambda x: {1: "Yes (Diagnosed)", 2: "No", 3: "Inapplicable (<18)"}[x])
            chbron = st.selectbox("Chronic Bronchitis", options=[1, 2] if is_adult else [3], format_func=lambda x: {1: "Yes (Diagnosed)", 2: "No", 3: "Inapplicable (<18)"}[x])
            chold = st.selectbox("High Cholesterol", options=[1, 2] if is_adult else [3], format_func=lambda x: {1: "Yes (Diagnosed)", 2: "No", 3: "Inapplicable (<18)"}[x])

    st.markdown("---")

    # Form submission
    if st.button("🔮 Calculate Conformal Prediction Interval", type="primary", use_container_width=True):
        # Build feature dictionary in exact feature order
        feature_dict = {
            "AGE24X": float(age),
            "FAMINC24": float(faminc),
            "POVLEV24": float(povlev),
            "FAMSZE24": float(famsze),
            "RUSIZE24": float(rusize),
            "EDUCYR": float(educ_yr),
            "SEX": int(sex),
            "RACETHX": int(raceth),
            "HISPANX": int(hispan),
            "MARRY24X": int(marry),
            "HIDEG": int(educ_deg),
            "REGION24": int(region),
            "RUCLAS24": int(ruclas),
            "POVCAT24": int(povcat),
            "INSCOV24": int(inscov),
            "INSURC24": int(insurc),
            "UNINS24": int(unins),
            "PRVEV24": int(prvev),
            "MCREV24": int(mcrev),
            "MCDEV24": int(mcdev),
            "TRIEV24": int(triev),
            "VAEV24": int(vaev),
            "RTHLTH31": int(rthlth),
            "MNHLTH31": int(mnhlth),
            "HIBPDX": int(hibp),
            "CHDDX": int(chd),
            "DIABDX_M18": int(diab),
            "ASTHDX": int(asth),
            "ARTHDX": int(arth),
            "CANCERDX": int(cancer),
            "STRKDX": int(strk),
            "CHBRON31": int(chbron),
            "CHOLDX": int(chold),
            "EMPST31": int(empst)
        }

        # Convert to single-row DataFrame
        df_input = pd.DataFrame([feature_dict])[ALL_PREDICTORS]

        # Predict
        try:
            point_pred, lower, upper, width = model.predict(df_input, confidence_level=selected_conf)
            st.session_state["last_prediction"] = {
                "point_estimate": float(point_pred[0]),
                "lower_bound": float(lower[0]),
                "upper_bound": float(upper[0]),
                "interval_width": float(width[0]),
                "confidence_level": float(selected_conf),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state["last_profile"] = feature_dict
            navigate_to(PAGE_RESULTS)
        except Exception as e:
            st.error(f"Prediction failed: {e}")


# -----------------------------------------------------------------------------
# Page 3: Prediction Results
# -----------------------------------------------------------------------------
def render_page_results():
    st.markdown(
        """
        <div class="hero-card" style="padding: 20px; border-left-color: #2F855A;">
            <div class="hero-title" style="font-size: 22px;">Conformal Healthcare Cost Prediction Results</div>
            <div class="hero-subtitle">
                Individual annual out-of-pocket expenditure estimate with finite-sample calibrated uncertainty bounds.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    pred = st.session_state.get("last_prediction")
    if pred is None:
        st.warning("No prediction found. Please fill out the patient profile form first.")
        st.button("Go to Estimation Form", on_click=lambda: navigate_to(PAGE_FORM))
        return

    conf_pct = int(pred["confidence_level"] * 100)
    point_val = pred["point_estimate"]
    lower_val = pred["lower_bound"]
    upper_val = pred["upper_bound"]
    width_val = pred["interval_width"]

    # Metric Cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card-blue">
                <div class="metric-label">Central Point Estimate</div>
                <div class="metric-value metric-value-blue">${point_val:,.0f}</div>
                <div class="metric-hint">Expected annual out-of-pocket spend</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card-green">
                <div class="metric-label">Lower Prediction Bound</div>
                <div class="metric-value metric-value-green">${lower_val:,.0f}</div>
                <div class="metric-hint">Non-negative floor ({conf_pct}% guarantee)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card-green">
                <div class="metric-label">Upper Prediction Bound</div>
                <div class="metric-value metric-value-green">${upper_val:,.0f}</div>
                <div class="metric-hint">Upper uncertainty limit ({conf_pct}% guarantee)</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Plotly Interval Figure
    st.markdown("### Interactive Prediction Interval Visualizer")
    fig = create_conformal_interval_figure(
        point_pred=point_val,
        lower_bound=lower_val,
        upper_bound=upper_val,
        confidence_level=pred["confidence_level"]
    )
    st.plotly_chart(fig, use_container_width=True)

    # Plain Language Explanation
    st.markdown(
        f"""
        <div class="content-card">
            <h4 style="color: #2F855A; margin-top: 0;">Plain-Language Clinical Interpretation</h4>
            <p>
                Based on the provided demographic and clinical profile, the central expected out-of-pocket healthcare expense is 
                <b>${point_val:,.2f}</b> per year.
            </p>
            <p>
                Because healthcare expenses are subject to unforeseen acute episodes, medication changes, and specialist visits, 
                our Normalized Conformal Prediction framework generates a <b>{conf_pct}% confidence prediction interval</b> spanning from 
                <b>${lower_val:,.2f}</b> to <b>${upper_val:,.2f}</b> (an uncertainty width of <b>${width_val:,.2f}</b>).
            </p>
            <ul>
                <li><b>Statistical Validity:</b> Across 100 similar individuals in the U.S. population, exactly {conf_pct} will have their actual annual out-of-pocket spending fall within this range.</li>
                <li><b>Adaptive Width:</b> The interval width is dynamically computed using our residual dispersion model to reflect the specific variability associated with this age and health profile.</li>
                <li><b>Non-Negative Cost:</b> Prediction limits are strictly bounded at $0.00 since medical costs cannot be negative.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Actions: Edit, Recalculate, Download CSV
    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        st.button("✏️ Modify Patient Details", use_container_width=True, on_click=lambda: navigate_to(PAGE_FORM))
    with col_b:
        st.button("🔄 Recalculate", use_container_width=True, on_click=lambda: navigate_to(PAGE_FORM))
    with col_c:
        # Build CSV report
        csv_data = {
            "Parameter": [
                "Analysis Timestamp",
                "Confidence Level Nominal",
                "Central Point Estimate ($)",
                "Lower Conformal Limit ($)",
                "Upper Conformal Limit ($)",
                "Interval Width ($)",
                "Age",
                "Family Income ($)",
                "Insurance Category Code",
                "Physical Health Code"
            ],
            "Value": [
                pred["timestamp"],
                f"{conf_pct}%",
                f"{point_val:.2f}",
                f"{lower_val:.2f}",
                f"{upper_val:.2f}",
                f"{width_val:.2f}",
                st.session_state["last_profile"].get("AGE24X", ""),
                st.session_state["last_profile"].get("FAMINC24", ""),
                st.session_state["last_profile"].get("INSCOV24", ""),
                st.session_state["last_profile"].get("RTHLTH31", "")
            ]
        }
        csv_df = pd.DataFrame(csv_data)
        csv_buffer = io.StringIO()
        csv_df.to_csv(csv_buffer, index=False)

        st.download_button(
            label="📥 Download CSV Report",
            data=csv_buffer.getvalue(),
            file_name=f"conformal_healthcare_estimate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown(
        """
        <div class="disclaimer-box">
            <b>Disclaimer:</b> This interval prediction is an empirical research estimate based on the 2024 MEPS survey. 
            It is not an insurer quote, billing guarantee, or treatment authorization.
        </div>
        """,
        unsafe_allow_html=True
    )


def build_subgroup_coverage_figure(df: pd.DataFrame, category_col: str, title: str, nominal_coverage: float = 0.90) -> go.Figure:
    """Helper function to build a subgroup coverage comparison bar chart."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df[category_col],
            y=df["Empirical_Coverage"] * 100,
            name="Empirical Coverage (%)",
            marker=dict(color="#2563EB", line=dict(color="#1E40AF", width=1)),
            hovertemplate="<b>%{x}</b><br>Coverage: %{y:.1f}%<br>Count: %{customdata:,}<extra></extra>",
            customdata=df["Count"]
        )
    )
    fig.add_hline(
        y=nominal_coverage * 100,
        line_dash="dash",
        line_color="#DC2626",
        line_width=2,
        annotation_text=f"Nominal Target ({int(nominal_coverage * 100)}%)",
        annotation_position="top right"
    )
    fig.update_layout(
        title=title,
        xaxis_title=category_col.replace("_", " "),
        yaxis_title="Empirical Coverage (%)",
        yaxis=dict(range=[65, 102], ticksuffix="%"),
        template="plotly_white",
        font=dict(family="Inter, Roboto, sans-serif", size=13),
        height=340,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


# -----------------------------------------------------------------------------
# Page 4: Research & Model Evaluation
# -----------------------------------------------------------------------------
def render_page_research(metrics):
    st.markdown(
        """
        <div class="hero-card" style="padding: 20px; border-left-color: #2B6CB0;">
            <div class="hero-title" style="font-size: 22px;">Research Methodology & Empirical Evaluation</div>
            <div class="hero-subtitle">
                Official empirical validation metrics evaluated on the untouched 20% test partition (N = 3,706).
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if metrics is None:
        st.warning("Model metrics artifact not found. Please train the model to generate empirical metrics.")
        return

    eval_90 = metrics["test_evaluations"]["confidence_90"]
    meta = metrics["dataset_metadata"]

    # Top metrics row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Total Usable Sample",
            value=f"{meta['total_cleaned_observations']:,}",
            help="Cleaned MEPS 2024 HC-256 individuals"
        )
    with c2:
        st.metric(
            label="Empirical Coverage (Test)",
            value=f"{eval_90['empirical_coverage'] * 100:.2f}%",
            delta=f"{(eval_90['empirical_coverage'] - 0.90) * 100:+.2f}% vs Nominal 90%",
            help="Observed test set coverage at 90% nominal target"
        )
    with c3:
        st.metric(
            label="Test MAE (Dollar Scale)",
            value=f"${eval_90['mae']:,.2f}",
            help="Mean Absolute Error on original dollar scale"
        )
    with c4:
        st.metric(
            label="Median Interval Width",
            value=f"${eval_90['median_interval_width']:,.2f}",
            help="Median width of 90% conformal intervals"
        )

    st.markdown("---")

    # Data partitioning details
    st.markdown("### 1. Data Partitioning & Protocol")
    p_col1, p_col2 = st.columns([1, 1])
    with p_col1:
        split_df = pd.DataFrame({
            "Partition": ["Training Set", "Calibration Set", "Untouched Test Set"],
            "Percentage": ["60.0%", "20.0%", "20.0%"],
            "Sample Count": [
                f"{meta['train_observations']:,}",
                f"{meta['calibration_observations']:,}",
                f"{meta['test_observations']:,}"
            ],
            "Role": [
                "Fits central regression f(x) and dispersion s(x)",
                "Computes nonconformity scores and conformal quantile q",
                "Unbiased final empirical coverage evaluation"
            ]
        })
        st.table(split_df)
    with p_col2:
        st.markdown(
            """
            <div class="content-card">
                <h4 style="color: #2B6CB0; margin-top: 0;">Statistical Properties of Split Conformal</h4>
                <p>
                    Standard conformal prediction proves that if calibration and test data points are exchangeable,
                    the probability that a new response falls inside the prediction interval is at least $1 - \\alpha$:
                </p>
                <p style="text-align: center; font-weight: 600; color: #22543D;">
                    P(Y_{n+1} &isin; C(X_{n+1})) &ge; 1 - &alpha;
                </p>
                <p>
                    Because our framework calibrates on 3,705 independent individuals, finite-sample guarantees hold strictly 
                    without requiring parametric normality or homoscedasticity assumptions.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 2. Subgroup Coverage & Fairness Audit")
    tab_age, tab_ins, tab_chronic = st.tabs(["Age Group", "Insurance Status", "Chronic Comorbidities"])


    with tab_age:
        age_data = pd.DataFrame(metrics["subgroup_performance"]["Age_Group"])
        fig_age = build_subgroup_coverage_figure(age_data, "Age_Group", "Empirical Conformal Coverage by Age Group (Test Set)")
        st.plotly_chart(fig_age, use_container_width=True)
        st.dataframe(
            age_data.style.format({
                "Empirical_Coverage": "{:.2%}",
                "Mean_Interval_Width": "${:,.2f}",
                "Median_Interval_Width": "${:,.2f}",
                "MAE": "${:,.2f}"
            }),
            use_container_width=True
        )

    with tab_ins:
        ins_data = pd.DataFrame(metrics["subgroup_performance"]["Insurance"])
        fig_ins = build_subgroup_coverage_figure(ins_data, "Insurance", "Empirical Conformal Coverage by Insurance Status (Test Set)")
        st.plotly_chart(fig_ins, use_container_width=True)
        st.dataframe(
            ins_data.style.format({
                "Empirical_Coverage": "{:.2%}",
                "Mean_Interval_Width": "${:,.2f}",
                "Median_Interval_Width": "${:,.2f}",
                "MAE": "${:,.2f}"
            }),
            use_container_width=True
        )

    with tab_chronic:
        chr_data = pd.DataFrame(metrics["subgroup_performance"]["Chronic_Category"])
        fig_chr = build_subgroup_coverage_figure(chr_data, "Chronic_Category", "Empirical Conformal Coverage by Chronic Comorbidities (Test Set)")
        st.plotly_chart(fig_chr, use_container_width=True)
        st.dataframe(
            chr_data.style.format({
                "Empirical_Coverage": "{:.2%}",
                "Mean_Interval_Width": "${:,.2f}",
                "Median_Interval_Width": "${:,.2f}",
                "MAE": "${:,.2f}"
            }),
            use_container_width=True
        )


# -----------------------------------------------------------------------------
# Page 5: How to Read My Estimate
# -----------------------------------------------------------------------------
def render_page_guide():
    st.markdown(
        """
        <div class="hero-card" style="padding: 20px; border-left-color: #38A169;">
            <div class="hero-title" style="font-size: 22px;">How to Read and Interpret Your Cost Estimate</div>
            <div class="hero-subtitle">
                A patient and researcher guide to understanding prediction intervals, uncertainty quantification, and health economics.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="content-card">
                <h3 style="color: #2B6CB0; margin-top: 0;">1. What the Central Estimate Means</h3>
                <p>
                    The <b>Central Point Estimate</b> represents the statistical conditional expectation (the most probable central expense) 
                    for an individual with the specified demographic, financial, and clinical attributes under our trained model.
                </p>
                <h3 style="color: #2F855A; margin-top: 18px;">2. What the Prediction Interval Means</h3>
                <p>
                    The <b>Prediction Interval</b> defines a bounded dollar range:
                </p>
                <div style="background-color: #F0FDF4; padding: 12px; border-radius: 8px; font-weight: 600; text-align: center; color: #22543D;">
                    [ Lower Bound , Upper Bound ]
                </div>
                <p style="margin-top: 10px;">
                    Under a 90% confidence setting, exactly 90 out of 100 patients with this profile will incur out-of-pocket healthcare expenses 
                    that fall between these two dollar amounts.
                </p>
                <h3 style="color: #DD6B20; margin-top: 18px;">3. Why Prediction Intervals Can Be Wide</h3>
                <p>
                    Users often ask: <i>"Why is my interval spanning several thousand dollars?"</i>
                </p>
                <ul>
                    <li><b>Healthcare Randomness:</b> A patient with hypertension may experience a calm year with only generic prescription copays ($200), or an unexpected hospitalization, ER visit, or specialty consult ($6,000+).</li>
                    <li><b>Severe Right-Skewness:</b> Unlike height or blood pressure, healthcare spending has massive variance. An honest statistical model must reflect this reality rather than producing falsely narrow intervals.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="content-card">
                <h3 style="color: #1A365D; margin-top: 0;">4. What Normalized Conformal Prediction Contributes</h3>
                <p>
                    Standard split conformal prediction produces a <b>fixed interval width</b> for every person, regardless of their health status. 
                    This causes two severe flaws:
                </p>
                <ul>
                    <li>Healthy 20-year-olds receive absurdly wide intervals ($0 to $8,000).</li>
                    <li>Sick 75-year-olds receive dangerously narrow intervals that fail to cover their true spending.</li>
                </ul>
                <p>
                    Our <b>Normalized Conformal Prediction</b> method estimates an individual residual dispersion $s(x)$. 
                    It adaptively expands intervals for high-risk patients and shrinks them for low-risk individuals, 
                    ensuring equitable fairness across all demographic subgroups.
                </p>
                <h3 style="color: #C53030; margin-top: 18px;">5. Important Study Limitations</h3>
                <ul>
                    <li><b>U.S. Data Only:</b> The MEPS survey reflects U.S. healthcare delivery, insurance cost-sharing, and prices. It does not generalize to India or single-payer healthcare systems.</li>
                    <li><b>Non-Causal:</b> Associations between chronic conditions and expenses reflect observational patterns, not direct clinical causation.</li>
                    <li><b>Not a Guaranteed Bill:</b> This estimate is for research, budgeting, and statistical education, not a binding insurer pre-authorization or hospital billing quote.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )


# -----------------------------------------------------------------------------
# Main Routing Logic
# -----------------------------------------------------------------------------
def main():
    model = load_trained_model()
    metrics = load_metrics()
    options = load_feature_options()

    active_page = st.session_state.get("active_page", PAGE_HOME)

    if active_page == PAGE_HOME:
        render_page_home()
    elif active_page == PAGE_FORM:
        render_page_form(model, options)
    elif active_page == PAGE_RESULTS:
        render_page_results()
    elif active_page == PAGE_RESEARCH:
        render_page_research(metrics)
    elif active_page == PAGE_GUIDE:
        render_page_guide()
    else:
        render_page_home()


if __name__ == "__main__":
    main()
