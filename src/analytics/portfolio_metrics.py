"""
Portfolio Analytics & Performance Metrics

Bloomberg-style analytics for professional investors:
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Maximum Drawdown analysis
- Rolling metrics
- Monthly returns heatmap data
- Risk-adjusted performance

Author: GARCH Algo Intelligence Platform
License: MIT
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for portfolio performance metrics."""
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_win: float
    avg_loss: float

    def __repr__(self) -> str:
        return (
            f"PerformanceMetrics("
            f"Return={self.annual_return:.2f}%, "
            f"Sharpe={self.sharpe_ratio:.2f}, "
            f"MaxDD={self.max_drawdown:.2f}%)"
        )


def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate simple returns from prices.

    Args:
        prices: Price series

    Returns:
        Returns series
    """
    returns = prices.pct_change().dropna()
    returns.name = 'returns'
    return returns


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate log returns from prices.

    Args:
        prices: Price series

    Returns:
        Log returns series
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    log_returns.name = 'log_returns'
    return log_returns


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe Ratio (return per unit of total risk).

    Formula: (Mean Return - Risk Free Rate) / Std Dev of Returns

    Args:
        returns: Returns series (decimal, e.g., 0.01 for 1%)
        risk_free_rate: Annual risk-free rate (default 2%)
        periods_per_year: Trading periods per year

    Returns:
        Sharpe ratio (annualized)
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)

    if excess_returns.std() == 0:
        return 0.0

    sharpe = excess_returns.mean() / excess_returns.std()
    sharpe_annual = sharpe * np.sqrt(periods_per_year)

    return float(sharpe_annual)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino Ratio (return per unit of downside risk).

    Like Sharpe but only penalizes downside volatility.

    Args:
        returns: Returns series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        Sortino ratio (annualized)
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)

    # Downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0

    downside_std = downside_returns.std()
    sortino = excess_returns.mean() / downside_std
    sortino_annual = sortino * np.sqrt(periods_per_year)

    return float(sortino_annual)


def calculate_drawdown(prices: pd.Series) -> pd.Series:
    """
    Calculate drawdown series (distance from peak).

    Args:
        prices: Price series

    Returns:
        Drawdown series (as decimals, e.g., -0.10 for -10%)
    """
    cumulative_max = prices.cummax()
    drawdown = (prices - cumulative_max) / cumulative_max
    return drawdown


def max_drawdown(prices: pd.Series) -> Tuple[float, int]:
    """
    Calculate maximum drawdown and its duration.

    Args:
        prices: Price series

    Returns:
        Tuple of (max_drawdown_pct, duration_in_periods)
    """
    dd = calculate_drawdown(prices)
    max_dd = dd.min()

    # Calculate duration
    in_drawdown = dd < 0
    drawdown_periods = in_drawdown.astype(int).groupby(
        (in_drawdown != in_drawdown.shift()).cumsum()
    ).sum()

    max_duration = int(drawdown_periods.max()) if len(drawdown_periods) > 0 else 0

    return float(max_dd), max_duration


def calmar_ratio(
    returns: pd.Series,
    prices: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar Ratio (annual return / max drawdown).

    Args:
        returns: Returns series
        prices: Price series
        periods_per_year: Trading periods per year

    Returns:
        Calmar ratio
    """
    if len(returns) < 2:
        return 0.0

    annual_ret = returns.mean() * periods_per_year
    max_dd, _ = max_drawdown(prices)

    if max_dd == 0:
        return 0.0

    calmar = annual_ret / abs(max_dd)
    return float(calmar)


def monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    """
    Create monthly returns table for heatmap.

    Args:
        returns: Returns series with datetime index

    Returns:
        DataFrame with years as rows, months as columns
    """
    # Resample to monthly
    monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

    # Create pivot table
    monthly_df = pd.DataFrame({
        'year': monthly.index.year,
        'month': monthly.index.month,
        'return': monthly.values * 100  # Convert to percentage
    })

    pivot = monthly_df.pivot(index='year', columns='month', values='return')

    # Rename columns to month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot.columns = [month_names[i-1] for i in pivot.columns]

    # Add annual total
    pivot['Annual'] = pivot.sum(axis=1)

    return pivot


def calculate_win_rate(returns: pd.Series) -> float:
    """
    Calculate win rate (% of positive return periods).

    Args:
        returns: Returns series

    Returns:
        Win rate as percentage
    """
    if len(returns) == 0:
        return 0.0

    wins = (returns > 0).sum()
    total = len(returns)

    return float(wins / total * 100)


def profit_factor(returns: pd.Series) -> float:
    """
    Calculate profit factor (total wins / total losses).

    Args:
        returns: Returns series

    Returns:
        Profit factor
    """
    wins = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())

    if losses == 0:
        return float('inf') if wins > 0 else 0.0

    return float(wins / losses)


def rolling_sharpe(
    returns: pd.Series,
    window: int = 252,
    periods_per_year: int = 252
) -> pd.Series:
    """
    Calculate rolling Sharpe ratio.

    Args:
        returns: Returns series
        window: Rolling window size
        periods_per_year: Trading periods per year

    Returns:
        Rolling Sharpe ratio series
    """
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()

    sharpe = (rolling_mean / rolling_std) * np.sqrt(periods_per_year)
    return sharpe


def calculate_all_metrics(
    prices: pd.Series,
    returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> PerformanceMetrics:
    """
    Calculate all performance metrics at once.

    Args:
        prices: Price series
        returns: Returns series (will be calculated if not provided)
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        PerformanceMetrics object
    """
    if returns is None:
        returns = calculate_returns(prices)

    # Total return
    total_ret = (prices.iloc[-1] / prices.iloc[0] - 1) * 100

    # Annualized return
    num_years = len(returns) / periods_per_year
    if num_years > 0:
        annual_ret = ((1 + total_ret/100) ** (1/num_years) - 1) * 100
    else:
        annual_ret = 0.0

    # Volatility
    annual_vol = returns.std() * np.sqrt(periods_per_year) * 100

    # Sharpe & Sortino
    sharpe = sharpe_ratio(returns, risk_free_rate, periods_per_year)
    sortino = sortino_ratio(returns, risk_free_rate, periods_per_year)

    # Drawdown
    max_dd, dd_duration = max_drawdown(prices)
    max_dd_pct = max_dd * 100

    # Calmar
    calmar = calmar_ratio(returns, prices, periods_per_year)

    # Win rate & profit factor
    win_rate_pct = calculate_win_rate(returns)
    pf = profit_factor(returns)

    # Average win/loss
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    avg_win = wins.mean() * 100 if len(wins) > 0 else 0.0
    avg_loss = losses.mean() * 100 if len(losses) > 0 else 0.0

    return PerformanceMetrics(
        total_return=total_ret,
        annual_return=annual_ret,
        annual_volatility=annual_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd_pct,
        max_drawdown_duration=dd_duration,
        win_rate=win_rate_pct,
        profit_factor=pf,
        num_trades=len(returns),
        avg_win=avg_win,
        avg_loss=avg_loss
    )


def benchmark_comparison(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series
) -> dict:
    """
    Compare portfolio to benchmark.

    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns

    Returns:
        Dictionary with comparison metrics
    """
    # Align returns
    aligned = pd.DataFrame({
        'portfolio': portfolio_returns,
        'benchmark': benchmark_returns
    }).dropna()

    # Beta (sensitivity to benchmark)
    covariance = aligned['portfolio'].cov(aligned['benchmark'])
    benchmark_var = aligned['benchmark'].var()
    beta = covariance / benchmark_var if benchmark_var != 0 else 0.0

    # Alpha (excess return)
    portfolio_mean = aligned['portfolio'].mean()
    benchmark_mean = aligned['benchmark'].mean()
    alpha = portfolio_mean - (beta * benchmark_mean)

    # Correlation
    correlation = aligned['portfolio'].corr(aligned['benchmark'])

    # Information ratio
    excess_returns = aligned['portfolio'] - aligned['benchmark']
    tracking_error = excess_returns.std()
    information_ratio = excess_returns.mean() / tracking_error if tracking_error != 0 else 0.0

    return {
        'beta': float(beta),
        'alpha': float(alpha * 252 * 100),  # Annualized %
        'correlation': float(correlation),
        'information_ratio': float(information_ratio * np.sqrt(252)),
        'tracking_error': float(tracking_error * np.sqrt(252) * 100)
    }
