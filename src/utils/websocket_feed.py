"""
WebSocket Real-Time Price Streaming

Ultra-fast price updates (10-50ms) using WebSocket connections.
This is how Binance, TradingView, and professional platforms work.

Author: GARCH Algo Intelligence Platform
License: MIT
"""

from __future__ import annotations
import json
import logging
import asyncio
from typing import Optional, Dict, Any, Callable
from datetime import datetime
import threading

try:
    import websocket
except ImportError:
    websocket = None

logger = logging.getLogger(__name__)


class WebSocketPriceFeed:
    """
    WebSocket-based real-time price feed.
    
    Provides 10-50ms latency updates directly from exchange.
    """
    
    def __init__(self):
        """Initialize WebSocket feed."""
        self.ws = None
        self.running = False
        self.latest_data = {}
        self.callbacks = []
        self.thread = None
    
    def add_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for price updates."""
        self.callbacks.append(callback)
    
    def start_okx_stream(self, inst_id: str = "BTC-USDT"):
        """
        Start OKX WebSocket stream.
        
        Args:
            inst_id: Instrument ID (e.g., "BTC-USDT")
        """
        if websocket is None:
            logger.error("websocket-client not installed. Run: pip install websocket-client")
            return
        
        def on_message(ws, message):
            """Handle incoming WebSocket message."""
            try:
                data = json.loads(message)
                
                # OKX sends data in 'data' field
                if 'data' in data and len(data['data']) > 0:
                    ticker = data['data'][0]
                    
                    self.latest_data = {
                        'last': float(ticker.get('last', 0)),
                        'open24h': float(ticker.get('open24h', 0)),
                        'high24h': float(ticker.get('high24h', 0)),
                        'low24h': float(ticker.get('low24h', 0)),
                        'vol24h': float(ticker.get('volCcy24h', ticker.get('vol24h', 0))),
                        'change24h': 0,  # Calculate below
                        'timestamp': datetime.now(),
                        'source': 'okx_websocket'
                    }
                    
                    # Calculate 24h change
                    if self.latest_data['open24h'] > 0:
                        change = ((self.latest_data['last'] - self.latest_data['open24h']) 
                                 / self.latest_data['open24h']) * 100
                        self.latest_data['change24h'] = change
                    
                    # Notify callbacks
                    for callback in self.callbacks:
                        try:
                            callback(self.latest_data)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")
                            
            except Exception as e:
                logger.error(f"WebSocket message error: {e}")
        
        def on_error(ws, error):
            """Handle WebSocket error."""
            logger.error(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            """Handle WebSocket close."""
            logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
            self.running = False
        
        def on_open(ws):
            """Handle WebSocket open."""
            logger.info("WebSocket connected")
            self.running = True
            
            # Subscribe to ticker channel
            subscribe_msg = {
                "op": "subscribe",
                "args": [{
                    "channel": "tickers",
                    "instId": inst_id
                }]
            }
            ws.send(json.dumps(subscribe_msg))
        
        def run_websocket():
            """Run WebSocket in thread."""
            websocket.enableTrace(False)
            self.ws = websocket.WebSocketApp(
                "wss://ws.okx.com:8443/ws/v5/public",
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            self.ws.run_forever()
        
        # Start WebSocket in background thread
        self.thread = threading.Thread(target=run_websocket, daemon=True)
        self.thread.start()
    
    def start_binance_stream(self, symbol: str = "btcusdt"):
        """
        Start Binance WebSocket stream.
        
        Args:
            symbol: Trading pair (lowercase, e.g., "btcusdt")
        """
        if websocket is None:
            logger.error("websocket-client not installed")
            return
        
        def on_message(ws, message):
            """Handle Binance WebSocket message."""
            try:
                data = json.loads(message)
                
                # Binance 24hr ticker format
                if 'e' in data and data['e'] == '24hrTicker':
                    self.latest_data = {
                        'last': float(data.get('c', 0)),
                        'open24h': float(data.get('o', 0)),
                        'high24h': float(data.get('h', 0)),
                        'low24h': float(data.get('l', 0)),
                        'vol24h': float(data.get('v', 0)),
                        'change24h': float(data.get('P', 0)),
                        'timestamp': datetime.now(),
                        'source': 'binance_websocket'
                    }
                    
                    # Notify callbacks
                    for callback in self.callbacks:
                        try:
                            callback(self.latest_data)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")
                            
            except Exception as e:
                logger.error(f"WebSocket message error: {e}")
        
        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket closed")
            self.running = False
        
        def on_open(ws):
            logger.info("Binance WebSocket connected")
            self.running = True
        
        def run_websocket():
            websocket.enableTrace(False)
            url = f"wss://stream.binance.com:9443/ws/{symbol}@ticker"
            self.ws = websocket.WebSocketApp(
                url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            self.ws.run_forever()
        
        self.thread = threading.Thread(target=run_websocket, daemon=True)
        self.thread.start()
    
    def get_latest(self) -> Dict[str, Any]:
        """Get latest price data."""
        return self.latest_data if self.latest_data else {
            'last': 0.0,
            'open24h': 0.0,
            'high24h': 0.0,
            'low24h': 0.0,
            'vol24h': 0.0,
            'change24h': 0.0,
            'timestamp': datetime.now(),
            'source': 'none'
        }
    
    def stop(self):
        """Stop WebSocket connection."""
        self.running = False
        if self.ws:
            self.ws.close()


# Fallback: Fast REST API ticker (if WebSocket not available)
def get_fast_ticker_rest(exchange: str = "okx", symbol: str = "BTC-USDT") -> Dict[str, Any]:
    """
    Fast REST API ticker (fallback if WebSocket unavailable).
    
    Args:
        exchange: "okx" or "binance"
        symbol: Trading pair
    
    Returns:
        Ticker data dict
    """
    import requests
    
    try:
        if exchange == "okx":
            url = "https://www.okx.com/api/v5/market/ticker"
            response = requests.get(url, params={"instId": symbol}, timeout=1)
            response.raise_for_status()
            
            data = response.json()
            if data.get("code") == "0" and data.get("data"):
                ticker = data["data"][0]
                return {
                    'last': float(ticker.get('last', 0)),
                    'open24h': float(ticker.get('open24h', 0)),
                    'high24h': float(ticker.get('high24h', 0)),
                    'low24h': float(ticker.get('low24h', 0)),
                    'vol24h': float(ticker.get('volCcy24h', 0)),
                    'change24h': 0,
                    'timestamp': datetime.now(),
                    'source': 'okx_rest'
                }
        
        elif exchange == "binance":
            url = "https://api.binance.com/api/v3/ticker/24hr"
            symbol_binance = symbol.replace("-", "").replace("/", "")
            response = requests.get(url, params={"symbol": symbol_binance}, timeout=1)
            response.raise_for_status()
            
            data = response.json()
            return {
                'last': float(data.get('lastPrice', 0)),
                'open24h': float(data.get('openPrice', 0)),
                'high24h': float(data.get('highPrice', 0)),
                'low24h': float(data.get('lowPrice', 0)),
                'vol24h': float(data.get('volume', 0)),
                'change24h': float(data.get('priceChangePercent', 0)),
                'timestamp': datetime.now(),
                'source': 'binance_rest'
            }
    
    except Exception as e:
        logger.error(f"REST ticker error: {e}")
    
    return {
        'last': 0.0,
        'open24h': 0.0,
        'high24h': 0.0,
        'low24h': 0.0,
        'vol24h': 0.0,
        'change24h': 0.0,
        'timestamp': datetime.now(),
        'source': 'error'
    }
