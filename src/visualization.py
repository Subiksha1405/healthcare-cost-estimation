"""
Visualization utilities for healthcare cost estimation and conformal prediction.
Provides both Plotly (interactive for web app & notebooks) and Matplotlib figures.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str = "TOTSLF24",
    output_html: Optional[str] = None
) -> go.Figure:
    """
    Creates side-by-side interactive histograms: raw dollar scale vs log1p scale.
    """
    y_raw = df[target_col]
    y_log = np.log1p(y_raw)

    fig = go.Figure()

    # Raw scale histogram (filtered to 99th percentile for display clarity)
    p99 = np.percentile(y_raw, 99)
    fig.add_trace(
        go.Histogram(
            x=y_raw[y_raw <= p99],
            nbinsx=50,
            name="Raw Out-of-Pocket ($)",
            marker=dict(color="#2B6CB0", line=dict(color="#1A365D", width=1)),
            opacity=0.75,
            hovertemplate="Cost Range: $%{x:,.0f}<br>Count: %{y:,}<extra></extra>"
        )
    )

    fig.update_layout(
        title="Distribution of Annual Out-of-Pocket Healthcare Expenditure (up to 99th percentile)",
        xaxis_title="Out-of-Pocket Expenditure ($ USD)",
        yaxis_title="Survey Weighted / Unweighted Frequency",
        template="plotly_white",
        font=dict(family="Inter, Roboto, sans-serif", size=13),
        margin=dict(l=40, r=40, t=50, b=40)
    )

    if output_html:
        fig.write_html(output_html)
    return fig


def plot_expenditure_by_subgroup(
    df: pd.DataFrame,
    subgroup_col: str,
    subgroup_labels: Dict[int, str],
    target_col: str = "TOTSLF24",
    title: Optional[str] = None
) -> go.Figure:
    """
    Generates box/bar plot showing mean and median out-of-pocket spending across demographic/clinical subgroups.
    """
    temp_df = df[[subgroup_col, target_col]].copy()
    temp_df["Subgroup"] = temp_df[subgroup_col].map(subgroup_labels).fillna("Other")

    summary = temp_df.groupby("Subgroup")[target_col].agg(
        Mean="mean",
        Median="median",
        Q25=lambda x: np.percentile(x, 25),
        Q75=lambda x: np.percentile(x, 75),
        Count="count"
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=summary["Subgroup"],
            y=summary["Mean"],
            name="Mean Expenditure ($)",
            marker_color="#2B6CB0",
            hovertemplate="<b>%{x}</b><br>Mean: $%{y:,.2f}<extra></extra>"
        )
    )
    fig.add_trace(
        go.Bar(
            x=summary["Subgroup"],
            y=summary["Median"],
            name="Median Expenditure ($)",
            marker_color="#319795",
            hovertemplate="<b>%{x}</b><br>Median: $%{y:,.2f}<extra></extra>"
        )
    )

    fig.update_layout(
        title=title or f"Out-of-Pocket Expenditure by {subgroup_col}",
        xaxis_title="Subgroup",
        yaxis_title="Expenditure ($ USD)",
        barmode="group",
        template="plotly_white",
        font=dict(family="Inter, Roboto, sans-serif", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_conformal_interval_figure(
    point_pred: float,
    lower_bound: float,
    upper_bound: float,
    confidence_level: float = 0.90
) -> go.Figure:
    """
    Creates a clean, accessible Plotly interval chart for individual prediction:
    - Blue diamond for central point estimate.
    - Green horizontal interval line for uncertainty bounds.
    - Labels for point estimate, lower and upper bounds.
    """
    fig = go.Figure()

    # Horizontal prediction interval line
    fig.add_trace(
        go.Scatter(
            x=[lower_bound, upper_bound],
            y=[0, 0],
            mode="lines+markers",
            line=dict(color="#2F855A", width=5),
            marker=dict(size=12, symbol="line-ns-open", color="#2F855A", line=dict(width=3)),
            name=f"{int(confidence_level * 100)}% Prediction Interval",
            hoverinfo="skip"
        )
    )

    # Point estimate diamond marker
    fig.add_trace(
        go.Scatter(
            x=[point_pred],
            y=[0],
            mode="markers+text",
            marker=dict(symbol="diamond", size=16, color="#2B6CB0", line=dict(color="#1A365D", width=2)),
            text=[f"<b>${point_pred:,.0f}</b>"],
            textposition="top center",
            name="Point Estimate (Central)",
            hovertemplate="<b>Central Estimate:</b> $%{x:,.2f}<extra></extra>"
        )
    )

    # Annotations for Lower and Upper limits
    fig.add_annotation(
        x=lower_bound,
        y=0,
        text=f"<b>Lower: ${lower_bound:,.0f}</b>",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.5,
        arrowcolor="#2F855A",
        ax=0,
        ay=40,
        font=dict(color="#22543D", size=12, family="Inter, Roboto, sans-serif")
    )

    fig.add_annotation(
        x=upper_bound,
        y=0,
        text=f"<b>Upper: ${upper_bound:,.0f}</b>",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.5,
        arrowcolor="#2F855A",
        ax=0,
        ay=40,
        font=dict(color="#22543D", size=12, family="Inter, Roboto, sans-serif")
    )

    padding = max(50.0, (upper_bound - lower_bound) * 0.15)
    x_min = max(0.0, lower_bound - padding)
    x_max = upper_bound + padding

    fig.update_layout(
        xaxis=dict(
            title="<b>Annual Out-of-Pocket Healthcare Expenditure ($ USD)</b>",
            tickprefix="$",
            tickformat=",.0f",
            range=[x_min, x_max],
            showgrid=True,
            gridcolor="#E2E8F0"
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[-0.6, 0.6]
        ),
        template="plotly_white",
        height=260,
        margin=dict(l=40, r=40, t=30, b=40),
        showlegend=False,
        plot_bgcolor="#F7FAFC",
        paper_bgcolor="#FFFFFF"
    )

    return fig


def plot_subgroup_coverage_bar(
    subgroup_metrics_df: pd.DataFrame,
    subgroup_col: Optional[str] = None,
    nominal_coverage: float = 0.90
) -> go.Figure:
    """
    Plots test empirical coverage vs nominal target line across demographic & health subgroups.
    """
    if subgroup_col is None or subgroup_col not in subgroup_metrics_df.columns:
        # Detect the category column (the first non-metric column)
        metric_cols = {"Count", "Empirical_Coverage", "Mean_Interval_Width", "Median_Interval_Width", "MAE"}
        candidate_cols = [c for c in subgroup_metrics_df.columns if c not in metric_cols]
        subgroup_col = candidate_cols[0] if candidate_cols else subgroup_metrics_df.columns[0]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=subgroup_metrics_df[subgroup_col],
            y=subgroup_metrics_df["Empirical_Coverage"] * 100,
            name="Empirical Coverage (%)",
            marker=dict(color="#2563EB", line=dict(color="#1E40AF", width=1)),
            hovertemplate="<b>%{x}</b><br>Coverage: %{y:.1f}%<br>Count: %{customdata:,}<extra></extra>",
            customdata=subgroup_metrics_df["Count"]
        )
    )

    # Nominal coverage reference line
    fig.add_hline(
        y=nominal_coverage * 100,
        line_dash="dash",
        line_color="#DC2626",
        line_width=2,
        annotation_text=f"Nominal Target ({int(nominal_coverage * 100)}%)",
        annotation_position="top right"
    )

    fig.update_layout(
        title=f"Empirical Conformal Coverage by {subgroup_col.replace('_', ' ')} (Untouched Test Set)",
        xaxis_title=subgroup_col.replace("_", " "),
        yaxis_title="Empirical Coverage (%)",
        yaxis=dict(range=[65, 102], ticksuffix="%"),
        template="plotly_white",
        font=dict(family="Inter, Roboto, sans-serif", size=13),
        height=340,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig
