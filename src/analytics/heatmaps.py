"""
Heatmap Visualizations for Bloomberg-style Analytics

Creates professional heatmaps for:
- Monthly returns calendar
- Correlation matrices
- Risk matrices

Author: GARCH Algo Intelligence Platform
License: MIT
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional


def create_monthly_returns_heatmap(
    monthly_table: pd.DataFrame,
    title: str = "Monthly Returns Heatmap (%)"
) -> go.Figure:
    """
    Create Bloomberg-style monthly returns heatmap.

    Args:
        monthly_table: DataFrame with years as rows, months as columns
        title: Chart title

    Returns:
        Plotly figure
    """
    # Separate annual column if present
    has_annual = 'Annual' in monthly_table.columns
    if has_annual:
        annual_col = monthly_table['Annual']
        data_table = monthly_table.drop('Annual', axis=1)
    else:
        data_table = monthly_table
        annual_col = None

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=data_table.values,
        x=data_table.columns,
        y=data_table.index,
        colorscale='RdYlGn',
        zmid=0,
        text=data_table.values,
        texttemplate='%{text:.1f}%',
        textfont={"size": 10},
        colorbar=dict(title="Return %"),
        hovertemplate='%{y} %{x}: %{z:.2f}%<extra></extra>'
    ))

    # Add annual column as separate trace if present
    if has_annual:
        # Add some space and the annual column
        annual_x = ['', 'Annual']  # Empty space + Annual
        annual_z = [[None] * len(data_table.index), annual_col.values]

        fig.add_trace(go.Heatmap(
            z=np.array(annual_z).T,
            x=annual_x,
            y=data_table.index,
            colorscale='RdYlGn',
            zmid=0,
            showscale=False,
            text=np.array([[None] * len(data_table.index), annual_col.values]).T,
            texttemplate='%{text:.1f}%',
            textfont={"size": 10, "color": "white"},
            hovertemplate='%{y} Annual: %{z:.2f}%<extra></extra>'
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title="Year",
        template="plotly_dark",
        height=400,
        margin=dict(l=60, r=20, t=60, b=40)
    )

    return fig


def create_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    title: str = "Asset Correlation Matrix"
) -> go.Figure:
    """
    Create correlation matrix heatmap.

    Args:
        correlation_matrix: Correlation matrix DataFrame
        title: Chart title

    Returns:
        Plotly figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=correlation_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 12},
        colorbar=dict(title="Correlation"),
        hovertemplate='%{y} vs %{x}: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=500,
        margin=dict(l=80, r=20, t=60, b=80),
        xaxis={'side': 'bottom'},
        yaxis={'side': 'left'}
    )

    return fig


def create_drawdown_chart(
    prices: pd.Series,
    drawdown: pd.Series,
    title: str = "Portfolio Drawdown (Underwater Equity)"
) -> go.Figure:
    """
    Create drawdown visualization.

    Args:
        prices: Price series
        drawdown: Drawdown series (as decimals)
        title: Chart title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Drawdown area
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values * 100,  # Convert to percentage
        fill='tozeroy',
        fillcolor='rgba(255,0,0,0.3)',
        line=dict(color='red', width=2),
        name='Drawdown',
        hovertemplate='%{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
    ))

    # Zero line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        annotation_text="Peak",
        annotation_position="right"
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_dark",
        height=350,
        margin=dict(l=60, r=20, t=60, b=40),
        showlegend=False
    )

    fig.update_yaxes(ticksuffix="%")

    return fig


def create_rolling_metrics_chart(
    rolling_sharpe: pd.Series,
    title: str = "Rolling Sharpe Ratio (252-day)"
) -> go.Figure:
    """
    Create rolling metrics chart.

    Args:
        rolling_sharpe: Rolling Sharpe ratio series
        title: Chart title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index,
        y=rolling_sharpe.values,
        mode='lines',
        line=dict(color='cyan', width=2),
        name='Rolling Sharpe',
        hovertemplate='%{x}<br>Sharpe: %{y:.2f}<extra></extra>'
    ))

    # Add reference lines
    fig.add_hline(y=1.0, line_dash="dash", line_color="yellow",
                  annotation_text="Good (1.0)", annotation_position="right")
    fig.add_hline(y=2.0, line_dash="dash", line_color="green",
                  annotation_text="Excellent (2.0)", annotation_position="right")

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        template="plotly_dark",
        height=350,
        margin=dict(l=60, r=20, t=60, b=40),
        showlegend=False
    )

    return fig


def create_returns_distribution(
    returns: pd.Series,
    title: str = "Returns Distribution"
) -> go.Figure:
    """
    Create returns distribution histogram.

    Args:
        returns: Returns series
        title: Chart title

    Returns:
        Plotly figure
    """
    returns_pct = returns * 100

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=returns_pct,
        nbinsx=50,
        marker=dict(
            color=returns_pct,
            colorscale='RdYlGn',
            cmid=0,
            showscale=False
        ),
        hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
    ))

    # Add mean line
    mean_return = returns_pct.mean()
    fig.add_vline(
        x=mean_return,
        line_dash="dash",
        line_color="cyan",
        annotation_text=f"Mean: {mean_return:.2f}%",
        annotation_position="top"
    )

    fig.update_layout(
        title=title,
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
        template="plotly_dark",
        height=350,
        margin=dict(l=60, r=20, t=60, b=40),
        showlegend=False
    )

    return fig
