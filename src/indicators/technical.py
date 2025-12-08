"""
Technical Indicators Module

Comprehensive library of technical analysis indicators.

Includes:
- Trend: EMA, SMA, MACD
- Momentum: RSI, Stochastic, CCI
- Volatility: ATR, Bollinger Bands
- Volume: OBV, VWAP
- Strength: ADX, Aroon

Author: GARCH Algo Intelligence Platform
License: MIT
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple


def ema(series: pd.Series, span: int) -> pd.Series:
    """
    Exponential Moving Average.

    Args:
        series: Price series
        span: EMA period

    Returns:
        EMA series
    """
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """
    Simple Moving Average.

    Args:
        series: Price series
        period: SMA period

    Returns:
        SMA series
    """
    return series.rolling(window=period).mean()


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD (Moving Average Convergence Divergence).

    Args:
        close: Close prices
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index.

    Args:
        close: Close prices
        period: RSI period

    Returns:
        RSI series (0-100)
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi_val = 100 - (100 / (1 + rs))

    return rsi_val


def atr_series(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Average True Range.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period

    Returns:
        ATR series
    """
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1/period, adjust=False).mean()


def adx_series(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Average Directional Index (ADX).

    Measures trend strength (0-100).

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period

    Returns:
        ADX series
    """
    plus_dm = high.diff().where(high.diff() > low.diff(), 0.0).clip(lower=0)
    minus_dm = -low.diff().where(low.diff() > high.diff(), 0.0).clip(lower=0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr

    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = dx.fillna(0).ewm(alpha=1/period, adjust=False).mean()

    return adx


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.

    Args:
        close: Close prices
        period: MA period
        std_dev: Standard deviation multiplier

    Returns:
        Tuple of (middle_band, upper_band, lower_band)
    """
    middle = sma(close, period)
    std = close.rolling(window=period).std()

    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    return middle, upper, lower


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_period: %K period
        d_period: %D period (SMA of %K)

    Returns:
        Tuple of (%K, %D)
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()

    return k, d


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume.

    Args:
        close: Close prices
        volume: Volume

    Returns:
        OBV series
    """
    direction = np.sign(close.diff())
    obv_series = (direction * volume).cumsum()
    return obv_series


def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series
) -> pd.Series:
    """
    Volume Weighted Average Price.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume

    Returns:
        VWAP series
    """
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum()


def cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20
) -> pd.Series:
    """
    Commodity Channel Index.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: CCI period

    Returns:
        CCI series
    """
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(
        lambda x: np.abs(x - x.mean()).mean()
    )

    cci_val = (typical_price - sma_tp) / (0.015 * mean_deviation)
    return cci_val


def williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Williams %R.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Period

    Returns:
        Williams %R series (-100 to 0)
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()

    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return wr


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to DataFrame.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with added indicator columns
    """
    df = df.copy()

    # Trend
    df['EMA_12'] = ema(df['Close'], 12)
    df['EMA_26'] = ema(df['Close'], 26)
    df['SMA_20'] = sma(df['Close'], 20)
    df['SMA_50'] = sma(df['Close'], 50)

    # MACD
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = macd(df['Close'])

    # Momentum
    df['RSI'] = rsi(df['Close'])
    df['Stoch_K'], df['Stoch_D'] = stochastic(df['High'], df['Low'], df['Close'])
    df['CCI'] = cci(df['High'], df['Low'], df['Close'])
    df['Williams_R'] = williams_r(df['High'], df['Low'], df['Close'])

    # Volatility
    df['ATR'] = atr_series(df['High'], df['Low'], df['Close'])
    df['BB_middle'], df['BB_upper'], df['BB_lower'] = bollinger_bands(df['Close'])

    # Strength
    df['ADX'] = adx_series(df['High'], df['Low'], df['Close'])

    # Volume
    if 'Volume' in df.columns and df['Volume'].notna().any():
        df['OBV'] = obv(df['Close'], df['Volume'])
        df['VWAP'] = vwap(df['High'], df['Low'], df['Close'], df['Volume'])

    return df
