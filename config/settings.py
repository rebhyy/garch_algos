"""
Configuration Management

Centralized configuration using environment variables and .env files.

Author: GARCH Algo Intelligence Platform
License: MIT
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env file
ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(ENV_PATH)


class Settings:
    """Application settings loaded from environment."""

    # API Keys
    TWELVEDATA_API_KEY: str = os.getenv("TWELVEDATA_API_KEY", "")
    ALPHAVANTAGE_API_KEY: str = os.getenv("ALPHAVANTAGE_API_KEY", "")

    # Application Settings
    TIMEZONE: str = os.getenv("TIMEZONE", "Africa/Tunis")
    DEFAULT_TIMEFRAME: str = os.getenv("DEFAULT_TIMEFRAME", "30m")
    DEFAULT_PERIOD: str = os.getenv("DEFAULT_PERIOD", "60d")
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "120"))

    # Alert Settings
    ALERT_EMAIL: str = os.getenv("ALERT_EMAIL", "")
    SMTP_SERVER: str = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME: str = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")

    # Risk Management
    DEFAULT_RISK_PER_TRADE: float = float(os.getenv("DEFAULT_RISK_PER_TRADE", "1.0"))
    DEFAULT_MAX_POSITION_SIZE: float = float(os.getenv("DEFAULT_MAX_POSITION_SIZE", "0.20"))
    DEFAULT_STOP_LOSS_ATR_MULT: float = float(os.getenv("DEFAULT_STOP_LOSS_ATR_MULT", "1.5"))

    # Model Settings
    GARCH_MIN_OBSERVATIONS: int = 100
    ARIMA_MIN_OBSERVATIONS: int = 30
    LSTM_LOOKBACK: int = 60

    # Asset Presets
    ASSET_PRESETS = {
        "BTC/USDT (OKX)": {
            "src": "ccxt",
            "market": "BTC/USDT",
            "exchange": "okx",
            "timeframe": "30m"
        },
        "ETH/USDT (OKX)": {
            "src": "ccxt",
            "market": "ETH/USDT",
            "exchange": "okx",
            "timeframe": "30m"
        },
        "EURUSD (Yahoo)": {
            "src": "yfinance",
            "symbol": "EURUSD=X",
            "period": "60d",
            "interval": "30m"
        },
        "Gold XAUUSD (Yahoo)": {
            "src": "yfinance",
            "symbol": "XAUUSD=X",
            "period": "60d",
            "interval": "30m",
            "fallback": "GC=F"
        },
        "Custom…": {"src": "custom"},
    }

    @classmethod
    def get_smtp_config(cls) -> dict:
        """Get SMTP configuration for email alerts."""
        return {
            "server": cls.SMTP_SERVER,
            "port": cls.SMTP_PORT,
            "username": cls.SMTP_USERNAME,
            "password": cls.SMTP_PASSWORD,
            "to_email": cls.ALERT_EMAIL
        }

    @classmethod
    def validate(cls) -> bool:
        """Validate critical settings."""
        warnings = []

        if not cls.TWELVEDATA_API_KEY:
            warnings.append("TWELVEDATA_API_KEY not set")

        if not cls.ALPHAVANTAGE_API_KEY:
            warnings.append("ALPHAVANTAGE_API_KEY not set")

        if cls.ALERT_EMAIL and not cls.SMTP_USERNAME:
            warnings.append("Email alerts configured but SMTP credentials missing")

        if warnings:
            print("⚠️  Configuration warnings:")
            for w in warnings:
                print(f"   - {w}")
            return False

        return True


# Global settings instance
settings = Settings()
