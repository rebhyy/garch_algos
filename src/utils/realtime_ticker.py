"""
Real-Time Price Ticker Module

Provides fast, smooth price updates like Binance/TradingView.
Uses optimized API calls and caching strategies.

Author: GARCH Algo Intelligence Platform
License: MIT
"""

from __future__ import annotations
import time
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

import requests
import streamlit as st

logger = logging.getLogger(__name__)


class RealtimeTicker:
    """
    Fast real-time price ticker with minimal latency.
    
    Optimized for speed - only fetches essential data.
    """
    
    def __init__(self, update_interval: float = 0.5):
        """
        Initialize ticker.
        
        Args:
            update_interval: Seconds between updates (default 0.5s = 500ms)
        """
        self.update_interval = update_interval
        self.last_update = None
        self.cache = {}
        self.cache_duration = timedelta(seconds=1)
    
    def get_okx_ticker_fast(self, inst_id: str = "BTC-USDT") -> Dict[str, Any]:
        """
        Get OKX ticker data with minimal latency.
        
        Args:
            inst_id: Instrument ID (e.g., "BTC-USDT")
        
        Returns:
            Dict with price, change, volume
        """
        # Check cache first
        cache_key = f"okx_{inst_id}"
        now = datetime.now()
        
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if now - cached_time < self.cache_duration:
                return cached_data
        
        try:
            # Fast API call - only ticker endpoint
            url = "https://www.okx.com/api/v5/market/ticker"
            response = requests.get(
                url,
                params={"instId": inst_id},
                timeout=2  # 2 second timeout
            )
            response.raise_for_status()
            
            data = response.json()
            if data.get("code") == "0" and data.get("data"):
                ticker = data["data"][0]
                
                result = {
                    "last": float(ticker.get("last", 0)),
                    "open24h": float(ticker.get("open24h", 0)),
                    "high24h": float(ticker.get("high24h", 0)),
                    "low24h": float(ticker.get("low24h", 0)),
                    "vol24h": float(ticker.get("volCcy24h", ticker.get("vol24h", 0))),
                    "change24h": float(ticker.get("changeUtc8", 0)),
                    "timestamp": now
                }
                
                # Cache result
                self.cache[cache_key] = (result, now)
                return result
            
            return self._get_fallback_data()
            
        except Exception as e:
            logger.warning(f"OKX ticker error: {e}")
            return self._get_fallback_data()
    
    def get_binance_ticker_fast(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """
        Get Binance ticker data with minimal latency.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
        
        Returns:
            Dict with price, change, volume
        """
        cache_key = f"binance_{symbol}"
        now = datetime.now()
        
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if now - cached_time < self.cache_duration:
                return cached_data
        
        try:
            url = "https://api.binance.com/api/v3/ticker/24hr"
            response = requests.get(
                url,
                params={"symbol": symbol},
                timeout=2
            )
            response.raise_for_status()
            
            data = response.json()
            
            result = {
                "last": float(data.get("lastPrice", 0)),
                "open24h": float(data.get("openPrice", 0)),
                "high24h": float(data.get("highPrice", 0)),
                "low24h": float(data.get("lowPrice", 0)),
                "vol24h": float(data.get("volume", 0)),
                "change24h": float(data.get("priceChangePercent", 0)),
                "timestamp": now
            }
            
            self.cache[cache_key] = (result, now)
            return result
            
        except Exception as e:
            logger.warning(f"Binance ticker error: {e}")
            return self._get_fallback_data()
    
    def _get_fallback_data(self) -> Dict[str, Any]:
        """Return fallback data when API fails."""
        return {
            "last": 0.0,
            "open24h": 0.0,
            "high24h": 0.0,
            "low24h": 0.0,
            "vol24h": 0.0,
            "change24h": 0.0,
            "timestamp": datetime.now()
        }
    
    def should_update(self) -> bool:
        """Check if enough time has passed for next update."""
        if self.last_update is None:
            return True
        
        elapsed = time.time() - self.last_update
        return elapsed >= self.update_interval
    
    def mark_updated(self):
        """Mark that an update occurred."""
        self.last_update = time.time()


def display_fast_ticker(
    ticker_data: Dict[str, Any],
    title: str = "BTC/USDT",
    container: Optional[Any] = None
):
    """
    Display fast-updating ticker in Streamlit.
    
    Args:
        ticker_data: Ticker data dict
        title: Display title
        container: Streamlit container (or None for default)
    """
    if container is None:
        container = st
    
    last_price = ticker_data.get("last", 0)
    change_24h = ticker_data.get("change24h", 0)
    vol_24h = ticker_data.get("vol24h", 0)
    high_24h = ticker_data.get("high24h", 0)
    low_24h = ticker_data.get("low24h", 0)
    
    # Calculate percentage change if not provided
    if change_24h == 0 and ticker_data.get("open24h", 0) > 0:
        open_24h = ticker_data["open24h"]
        change_24h = ((last_price - open_24h) / open_24h) * 100
    
    # Color for change
    change_color = "normal" if change_24h >= 0 else "inverse"
    
    # Display in columns
    col1, col2, col3, col4 = container.columns(4)
    
    with col1:
        container.metric(
            title,
            f"${last_price:,.2f}",
            delta=f"{change_24h:+.2f}%",
            delta_color=change_color
        )
    
    with col2:
        container.metric(
            "24h High",
            f"${high_24h:,.2f}"
        )
    
    with col3:
        container.metric(
            "24h Low",
            f"${low_24h:,.2f}"
        )
    
    with col4:
        container.metric(
            "24h Volume",
            f"${vol_24h:,.0f}" if vol_24h > 1000 else f"{vol_24h:,.2f}"
        )


def create_ticker_placeholder():
    """
    Create a placeholder for fast ticker updates.
    
    Returns:
        Streamlit empty container
    """
    return st.empty()
