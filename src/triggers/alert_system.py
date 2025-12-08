"""
Smart Alert & Trigger System

Detects trading opportunities and sends alerts:
- Dip buying opportunities (oversold conditions)
- Breakout detection (price/volume)
- Volatility spikes (GARCH-based)
- Support/resistance breaks
- MACD crossovers
- Mean reversion setups

Notification channels:
- Email (SMTP)
- Discord webhooks
- Twilio SMS (optional)
- In-app notifications

Author: GARCH Algo Intelligence Platform
License: MIT
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Callable
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Types of alerts."""
    DIP_BUY = "dip_buy"
    BREAKOUT_LONG = "breakout_long"
    BREAKOUT_SHORT = "breakout_short"
    VOLATILITY_SPIKE = "volatility_spike"
    SUPPORT_BREAK = "support_break"
    RESISTANCE_BREAK = "resistance_break"
    MACD_CROSS_BULL = "macd_cross_bull"
    MACD_CROSS_BEAR = "macd_cross_bear"
    MEAN_REVERSION = "mean_reversion"
    VOLUME_SURGE = "volume_surge"
    CUSTOM = "custom"


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Alert:
    """
    Alert object containing trigger information.

    Attributes:
        alert_type: Type of alert
        severity: Alert severity
        symbol: Trading symbol
        price: Current price
        message: Alert message
        timestamp: When alert was generated
        metadata: Additional data (dict)
    """
    alert_type: AlertType
    severity: AlertSeverity
    symbol: str
    price: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"[{self.severity.value.upper()}] {self.alert_type.value}: "
            f"{self.symbol} @ {self.price:.2f} - {self.message}"
        )


class AlertSystem:
    """
    Central alert management system.

    Example:
        >>> alerts = AlertSystem()
        >>> alerts.add_handler(email_handler)
        >>> alerts.check_dip_buy(df, threshold=0.05)
    """

    def __init__(self):
        """Initialize alert system."""
        self.handlers: List[Callable[[Alert], None]] = []
        self.alert_history: List[Alert] = []
        self.enabled = True

    def add_handler(self, handler: Callable[[Alert], None]):
        """
        Add alert handler (callback function).

        Args:
            handler: Function that takes Alert and sends notification
        """
        self.handlers.append(handler)
        logger.info(f"Added alert handler: {handler.__name__}")

    def send_alert(self, alert: Alert):
        """
        Send alert through all handlers.

        Args:
            alert: Alert object to send
        """
        if not self.enabled:
            logger.debug(f"Alerts disabled, skipping: {alert}")
            return

        logger.info(f"Sending alert: {alert}")
        self.alert_history.append(alert)

        for handler in self.handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Handler {handler.__name__} failed: {e}")

    def clear_history(self):
        """Clear alert history."""
        self.alert_history.clear()


# ============ TRIGGER DETECTION FUNCTIONS ============

def detect_dip_buy(
    df: pd.DataFrame,
    threshold: float = 0.05,
    lookback: int = 48,
    rsi_threshold: float = 30.0
) -> Optional[Alert]:
    """
    Detect dip buying opportunity.

    Triggers when:
    - Price dropped by threshold% from recent high
    - Optionally: RSI below threshold (oversold)

    Args:
        df: OHLCV DataFrame
        threshold: Drop percentage (0.05 = 5%)
        lookback: Bars to look back for high
        rsi_threshold: RSI oversold level

    Returns:
        Alert if dip detected, None otherwise
    """
    if len(df) < lookback:
        return None

    recent = df.tail(lookback)
    high = recent['High'].max()
    current = df['Close'].iloc[-1]
    drop_pct = (high - current) / high

    if drop_pct >= threshold:
        # Additional confirmation: check if RSI available
        has_rsi = 'RSI' in df.columns
        rsi_confirm = True

        if has_rsi:
            rsi = df['RSI'].iloc[-1]
            rsi_confirm = rsi < rsi_threshold

        if rsi_confirm:
            return Alert(
                alert_type=AlertType.DIP_BUY,
                severity=AlertSeverity.HIGH,
                symbol=df.get('symbol', 'Unknown'),
                price=float(current),
                message=f"Price dropped {drop_pct*100:.1f}% from recent high of {high:.2f}",
                metadata={
                    "drop_pct": drop_pct,
                    "recent_high": high,
                    "lookback": lookback
                }
            )

    return None


def detect_breakout(
    df: pd.DataFrame,
    resistance_lookback: int = 100,
    volume_mult: float = 1.5
) -> Optional[Alert]:
    """
    Detect price breakout above resistance.

    Triggers when:
    - Price breaks above recent high
    - Volume is elevated

    Args:
        df: OHLCV DataFrame
        resistance_lookback: Bars for resistance calculation
        volume_mult: Volume multiplier vs average

    Returns:
        Alert if breakout detected
    """
    if len(df) < resistance_lookback + 20:
        return None

    recent = df.tail(resistance_lookback)
    resistance = recent['High'].max()
    current_price = df['Close'].iloc[-1]

    # Check if breaking out
    if current_price > resistance * 1.001:  # 0.1% above resistance
        # Check volume
        if 'Volume' in df.columns:
            avg_volume = df['Volume'].tail(20).mean()
            current_volume = df['Volume'].iloc[-1]

            if current_volume > avg_volume * volume_mult:
                return Alert(
                    alert_type=AlertType.BREAKOUT_LONG,
                    severity=AlertSeverity.CRITICAL,
                    symbol=df.get('symbol', 'Unknown'),
                    price=float(current_price),
                    message=f"Breakout above {resistance:.2f} with {volume_mult}x volume",
                    metadata={
                        "resistance": resistance,
                        "volume_ratio": current_volume / avg_volume
                    }
                )

    return None


def detect_volatility_spike(
    df: pd.DataFrame,
    sigma_series: pd.Series,
    threshold_std: float = 2.0
) -> Optional[Alert]:
    """
    Detect volatility spike using GARCH conditional volatility.

    Triggers when:
    - Current volatility > mean + threshold_std * std

    Args:
        df: OHLCV DataFrame
        sigma_series: GARCH conditional volatility series
        threshold_std: Number of standard deviations

    Returns:
        Alert if spike detected
    """
    if len(sigma_series) < 100:
        return None

    recent_sigma = sigma_series.tail(100)
    mean_vol = recent_sigma.mean()
    std_vol = recent_sigma.std()
    current_vol = sigma_series.iloc[-1]

    threshold = mean_vol + threshold_std * std_vol

    if current_vol > threshold:
        return Alert(
            alert_type=AlertType.VOLATILITY_SPIKE,
            severity=AlertSeverity.HIGH,
            symbol=df.get('symbol', 'Unknown'),
            price=float(df['Close'].iloc[-1]),
            message=f"Volatility spike: {current_vol:.2f}% (threshold: {threshold:.2f}%)",
            metadata={
                "current_vol": current_vol,
                "mean_vol": mean_vol,
                "std_vol": std_vol,
                "z_score": (current_vol - mean_vol) / std_vol
            }
        )

    return None


def detect_macd_crossover(
    df: pd.DataFrame,
    adx_filter: float = 20.0
) -> Optional[Alert]:
    """
    Detect MACD signal line crossover.

    Args:
        df: DataFrame with MACD, MACD_signal, ADX columns
        adx_filter: Minimum ADX for trend confirmation

    Returns:
        Alert if crossover detected
    """
    if 'MACD' not in df.columns or 'MACD_signal' not in df.columns:
        return None

    if len(df) < 2:
        return None

    macd_curr = df['MACD'].iloc[-1]
    macd_prev = df['MACD'].iloc[-2]
    sig_curr = df['MACD_signal'].iloc[-1]
    sig_prev = df['MACD_signal'].iloc[-2]

    # Check ADX filter
    has_adx = 'ADX' in df.columns
    adx_pass = True
    if has_adx:
        adx_pass = df['ADX'].iloc[-1] >= adx_filter

    # Bullish cross
    if macd_prev <= sig_prev and macd_curr > sig_curr and adx_pass:
        return Alert(
            alert_type=AlertType.MACD_CROSS_BULL,
            severity=AlertSeverity.MEDIUM,
            symbol=df.get('symbol', 'Unknown'),
            price=float(df['Close'].iloc[-1]),
            message=f"MACD bullish crossover (ADX: {df.get('ADX', [0]).iloc[-1]:.1f})",
            metadata={"macd": macd_curr, "signal": sig_curr}
        )

    # Bearish cross
    if macd_prev >= sig_prev and macd_curr < sig_curr and adx_pass:
        return Alert(
            alert_type=AlertType.MACD_CROSS_BEAR,
            severity=AlertSeverity.MEDIUM,
            symbol=df.get('symbol', 'Unknown'),
            price=float(df['Close'].iloc[-1]),
            message=f"MACD bearish crossover (ADX: {df.get('ADX', [0]).iloc[-1]:.1f})",
            metadata={"macd": macd_curr, "signal": sig_curr}
        )

    return None


def detect_support_break(
    df: pd.DataFrame,
    support_level: float,
    tolerance: float = 0.01
) -> Optional[Alert]:
    """
    Detect support level break.

    Args:
        df: OHLCV DataFrame
        support_level: Support price level
        tolerance: Break tolerance (1% = 0.01)

    Returns:
        Alert if support broken
    """
    if len(df) < 2:
        return None

    current = df['Close'].iloc[-1]
    prev = df['Close'].iloc[-2]

    break_level = support_level * (1 - tolerance)

    # Was above support, now below
    if prev >= support_level and current < break_level:
        return Alert(
            alert_type=AlertType.SUPPORT_BREAK,
            severity=AlertSeverity.HIGH,
            symbol=df.get('symbol', 'Unknown'),
            price=float(current),
            message=f"Support broken at {support_level:.2f}",
            metadata={"support": support_level, "break_pct": (support_level - current) / support_level}
        )

    return None


def detect_mean_reversion(
    df: pd.DataFrame,
    sma_period: int = 20,
    std_threshold: float = 2.0
) -> Optional[Alert]:
    """
    Detect mean reversion setup (Bollinger Band bounce).

    Triggers when price touches lower band and starts reverting.

    Args:
        df: OHLCV DataFrame
        sma_period: SMA period for mean
        std_threshold: Standard deviation multiplier

    Returns:
        Alert if setup detected
    """
    if len(df) < sma_period + 5:
        return None

    closes = df['Close'].tail(sma_period + 5)
    sma = closes.rolling(sma_period).mean()
    std = closes.rolling(sma_period).std()

    lower_band = sma - std_threshold * std
    upper_band = sma + std_threshold * std

    current = df['Close'].iloc[-1]
    prev = df['Close'].iloc[-2]
    lower = lower_band.iloc[-1]

    # Price touched lower band and is reversing
    if prev <= lower and current > prev:
        return Alert(
            alert_type=AlertType.MEAN_REVERSION,
            severity=AlertSeverity.MEDIUM,
            symbol=df.get('symbol', 'Unknown'),
            price=float(current),
            message=f"Mean reversion setup at lower BB ({lower:.2f})",
            metadata={
                "sma": sma.iloc[-1],
                "lower_band": lower,
                "upper_band": upper_band.iloc[-1]
            }
        )

    return None


# ============ NOTIFICATION HANDLERS ============

def email_handler(
    alert: Alert,
    smtp_config: Dict[str, str]
) -> None:
    """
    Send alert via email.

    Args:
        alert: Alert object
        smtp_config: Dict with server, port, username, password, to_email
    """
    try:
        msg = MIMEMultipart()
        msg['From'] = smtp_config.get('username', '')
        msg['To'] = smtp_config.get('to_email', '')
        msg['Subject'] = f"Trading Alert: {alert.alert_type.value.upper()}"

        body = f"""
        Trading Alert Generated

        Type: {alert.alert_type.value}
        Severity: {alert.severity.value.upper()}
        Symbol: {alert.symbol}
        Price: ${alert.price:.2f}
        Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

        Message: {alert.message}

        Metadata:
        {chr(10).join(f'  {k}: {v}' for k, v in alert.metadata.items())}

        ---
        GARCH Algo Intelligence Platform
        """

        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(
            smtp_config.get('server', 'smtp.gmail.com'),
            int(smtp_config.get('port', 587))
        )
        server.starttls()
        server.login(
            smtp_config['username'],
            smtp_config['password']
        )
        server.send_message(msg)
        server.quit()

        logger.info(f"Email sent to {smtp_config.get('to_email')}")

    except Exception as e:
        logger.error(f"Email send failed: {e}")
        raise


def discord_webhook_handler(
    alert: Alert,
    webhook_url: str
) -> None:
    """
    Send alert via Discord webhook.

    Args:
        alert: Alert object
        webhook_url: Discord webhook URL
    """
    try:
        import requests

        # Color codes
        colors = {
            AlertSeverity.LOW: 3447003,      # Blue
            AlertSeverity.MEDIUM: 16776960,  # Yellow
            AlertSeverity.HIGH: 16744192,    # Orange
            AlertSeverity.CRITICAL: 16711680 # Red
        }

        embed = {
            "title": f"{alert.alert_type.value.replace('_', ' ').title()}",
            "description": alert.message,
            "color": colors.get(alert.severity, 3447003),
            "fields": [
                {"name": "Symbol", "value": alert.symbol, "inline": True},
                {"name": "Price", "value": f"${alert.price:.2f}", "inline": True},
                {"name": "Severity", "value": alert.severity.value.upper(), "inline": True},
            ],
            "timestamp": alert.timestamp.isoformat(),
            "footer": {"text": "GARCH Algo Intelligence Platform"}
        }

        # Add metadata fields
        for key, value in alert.metadata.items():
            embed["fields"].append({
                "name": key,
                "value": str(value),
                "inline": True
            })

        payload = {"embeds": [embed]}

        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()

        logger.info("Discord alert sent")

    except Exception as e:
        logger.error(f"Discord webhook failed: {e}")
        raise


def console_handler(alert: Alert) -> None:
    """
    Print alert to console (for testing).

    Args:
        alert: Alert object
    """
    print("\n" + "=" * 60)
    print(f"🚨 TRADING ALERT 🚨")
    print("=" * 60)
    print(f"Type:     {alert.alert_type.value.upper()}")
    print(f"Severity: {alert.severity.value.upper()}")
    print(f"Symbol:   {alert.symbol}")
    print(f"Price:    ${alert.price:.2f}")
    print(f"Time:     {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nMessage:  {alert.message}")
    if alert.metadata:
        print("\nMetadata:")
        for k, v in alert.metadata.items():
            print(f"  {k}: {v}")
    print("=" * 60 + "\n")


# ============ CONVENIENCE FUNCTIONS ============

def scan_all_triggers(
    df: pd.DataFrame,
    sigma_series: Optional[pd.Series] = None,
    support_levels: Optional[List[float]] = None,
    resistance_levels: Optional[List[float]] = None
) -> List[Alert]:
    """
    Scan for all trigger types.

    Args:
        df: OHLCV DataFrame with indicators
        sigma_series: GARCH volatility (optional)
        support_levels: List of support levels to monitor
        resistance_levels: List of resistance levels to monitor

    Returns:
        List of triggered alerts
    """
    alerts = []

    # Dip buy
    dip = detect_dip_buy(df)
    if dip:
        alerts.append(dip)

    # Breakout
    breakout = detect_breakout(df)
    if breakout:
        alerts.append(breakout)

    # Volatility spike
    if sigma_series is not None:
        vol_spike = detect_volatility_spike(df, sigma_series)
        if vol_spike:
            alerts.append(vol_spike)

    # MACD crossover
    macd_alert = detect_macd_crossover(df)
    if macd_alert:
        alerts.append(macd_alert)

    # Support breaks
    if support_levels:
        for level in support_levels:
            support_alert = detect_support_break(df, level)
            if support_alert:
                alerts.append(support_alert)

    # Mean reversion
    mr_alert = detect_mean_reversion(df)
    if mr_alert:
        alerts.append(mr_alert)

    return alerts
