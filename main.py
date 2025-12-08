# app.py — Streamlit GARCH + ARIMA + Market(Pro)
# pip install streamlit ccxt yfinance arch statsmodels mplfinance pandas numpy matplotlib \
#             streamlit-autorefresh requests plotly

from __future__ import annotations
import io, logging, warnings
from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
import matplotlib
from streamlit.elements import json

matplotlib.use("Agg")  # set backend before importing pyplot
import matplotlib.pyplot as plt
plt.rcParams["figure.facecolor"] = "none"

import mplfinance as mpf
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import requests
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from math import log, sqrt, exp, erf

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Portfolio Analytics modules
from src.analytics.portfolio_metrics import (
    calculate_returns, calculate_all_metrics, monthly_returns_table,
    calculate_drawdown, rolling_sharpe
)
from src.analytics.heatmaps import (
    create_monthly_returns_heatmap, create_correlation_heatmap,
    create_drawdown_chart, create_rolling_metrics_chart,
    create_returns_distribution
)

# Real-time WebSocket feed for ultra-fast updates (10-50ms)
from src.utils.websocket_feed import WebSocketPriceFeed, get_fast_ticker_rest

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib").setLevel(logging.WARNING)

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    def st_autorefresh(*args, **kwargs): return None

# ========= CONFIG =========
TZ = "Africa/Tunis"
DEFAULT_TIMEFRAME = "30m"
DEFAULT_PERIOD_YF = "60d"

# Hard-coded API keys (override in UI/secrets for production).
TWELVEDATA_API_KEY_DEFAULT   = "290cd640d4b9450b80e4fc11c525c23a"
ALPHAVANTAGE_API_KEY_DEFAULT = "CDIQ9XC57KV7Y9KR"

ASSET_PRESETS = {
    "BTC/USDT (OKX)": {"src": "ccxt", "market": "BTC/USDT", "exchange": "okx", "timeframe": "30m"},
    "ETH/USDT (OKX)": {"src": "ccxt", "market": "ETH/USDT", "exchange": "okx", "timeframe": "30m"},
    "EURUSD (Yahoo)": {"src": "yfinance", "symbol": "EURUSD=X", "period": "60d", "interval": "30m"},
    "Gold XAUUSD (Yahoo)": {"src": "yfinance", "symbol": "XAUUSD=X", "period": "60d", "interval": "30m", "fallback": "GC=F"},
    "Custom…": {"src": "custom"},
}

OKX_BASE = "https://www.okx.com"

# ========= UTILS =========
def ensure_tz(idx: pd.DatetimeIndex, tz=TZ) -> pd.DatetimeIndex:
    if idx.tz is None: idx = idx.tz_localize("UTC")
    return idx.tz_convert(tz)

def fib_levels(df: pd.DataFrame, lookback=200):
    recent = df.tail(lookback)
    hi = float(recent["High"].max()); lo = float(recent["Low"].min())
    d = hi - lo
    lvls = {"0.0%":hi, "23.6%":hi-0.236*d, "38.2%":hi-0.382*d,
            "50.0%":hi-0.5*d, "61.8%":hi-0.618*d, "78.6%":hi-0.786*d, "100%":lo}
    return lo, hi, lvls

def log_returns(df: pd.DataFrame) -> pd.Series:
    r = np.log(df["Close"]).diff().dropna(); r.name = "log_ret"; return r

def infer_bars_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 3: return 365*24*2
    deltas = np.diff(index.view("i8"))
    sec = np.median(deltas) / 1e9
    return (365 * 24 * 3600) / sec if sec > 0 else 365*24*2

def simple_pnl(df: pd.DataFrame, sigs: pd.DataFrame, tp=0.008, sl=0.006):
    pnl, wins, n = 0.0, 0, 0
    for ts, row in sigs.iterrows():
        side = row["side"]; entry = float(row["Close"])
        after = df.loc[df.index > ts].head(96)  # up to ~2 days on 30m bars
        if after.empty: break
        if side == "BUY":
            up_hit = after["High"] >= entry*(1+tp)
            dn_hit = after["Low"]  <= entry*(1-sl)
            if dn_hit.any() and (not up_hit.any() or after.index[dn_hit.argmax()] <= after.index[up_hit.argmax()]):
                pnl -= sl; n += 1
            elif up_hit.any():
                pnl += tp; wins += 1; n += 1
        else:  # SELL
            dn_hit = after["Low"]  <= entry*(1-tp)
            up_hit = after["High"] >= entry*(1+sl)
            if up_hit.any() and (not dn_hit.any() or after.index[up_hit.argmax()] <= after.index[dn_hit.argmax()]):
                pnl -= sl; n += 1
            elif dn_hit.any():
                pnl += tp; wins += 1; n += 1
    wr = (wins/n*100.0) if n else 0.0
    return {"trades": n, "win_rate": wr, "pnl_pct": pnl*100.0}

# ========= INDICATORS =========
def ema(s: pd.Series, span: int) -> pd.Series: return s.ewm(span=span, adjust=False).mean()

def macd(close: pd.Series, fast=12, slow=26, signal=9):
    m = ema(close, fast) - ema(close, slow)
    sig = ema(m, signal)
    hist = m - sig
    return m, sig, hist

def adx_series(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    plus_dm  = high.diff().where(high.diff() >  low.diff(), 0.0).clip(lower=0)
    minus_dm = -low.diff().where(low.diff()  > high.diff(), 0.0).clip(lower=0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.fillna(0).ewm(alpha=1/period, adjust=False).mean()

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def expected_move(S: float, ann_sigma: float, horizon_days: float) -> float:
    return S * ann_sigma/100.0 * sqrt(horizon_days/365.0)

def hist_var(df: pd.DataFrame, conf=0.95) -> float:
    r = log_returns(df).dropna()
    return float(-np.percentile(r, (1.0-conf)*100))

def macd_adx_signals(df: pd.DataFrame, adx_min=18.0) -> pd.DataFrame:
    z = df.copy()
    z["fast"] = ema(z["Close"], 12)
    z["slow"] = ema(z["Close"], 26)
    z["macd"] = z["fast"] - z["slow"]
    z["sig"]  = ema(z["macd"], 9)
    z["adx"]  = adx_series(z["High"], z["Low"], z["Close"])
    z["cross_up"]   = (z["macd"] > z["sig"]) & (z["macd"].shift(1) <= z["sig"].shift(1))
    z["cross_down"] = (z["macd"] < z["sig"]) & (z["macd"].shift(1) >= z["sig"].shift(1))
    z["enter_long"]  = z["cross_up"] & (z["adx"] >= adx_min)
    z["enter_short"] = z["cross_down"] & (z["adx"] >= adx_min)
    sigs = z.loc[z["enter_long"] | z["enter_short"], ["Close","enter_long","enter_short"]].copy()
    sigs["side"] = np.where(sigs["enter_long"], "BUY", "SELL")
    return sigs[["side","Close"]]

def bs_price(S, K, T, r, sigma, kind="call"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return 0.0
    d1 = (log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    if kind.lower().startswith("c"):
        return S*norm_cdf(d1) - K*exp(-r*T)*norm_cdf(d2)
    return K*exp(-r*T)*norm_cdf(-d2) - S*norm_cdf(-d1)

def bs_greeks(S, K, T, r, sigma, kind="call"):
    if T <= 0 or sigma <= 0: return {"delta":0,"gamma":0,"vega":0,"theta":0,"rho":0}
    d1 = (log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    pdf = lambda x: np.exp(-0.5*x*x)/sqrt(2*np.pi)
    delta = norm_cdf(d1) if kind[0].lower()=="c" else (norm_cdf(d1)-1)
    gamma = pdf(d1)/(S*sigma*sqrt(T))
    vega  = S*pdf(d1)*sqrt(T)/100.0
    theta = (-(S*pdf(d1)*sigma)/(2*sqrt(T)) - (kind[0].lower()=="c")*r*K*exp(-r*T)*norm_cdf(d2)
             + (kind[0].lower()!="c")*r*K*exp(-r*T)*norm_cdf(-d2))/365.0
    rho   = (K*T*exp(-r*T)*norm_cdf(d2 if kind[0].lower()=="c" else -d2))/100.0
    return {"delta":delta, "gamma":gamma, "vega":vega, "theta":theta, "rho":rho}

def implied_vol_bisect(S, K, T, r, price, kind="call", lo=1e-4, hi=5.0, tol=1e-5, iters=100):
    if price <= 0: return np.nan
    for _ in range(iters):
        mid = 0.5*(lo+hi)
        pm  = bs_price(S,K,T,r,mid,kind)
        if abs(pm - price) < tol: return mid
        if pm > price: hi = mid
        else: lo = mid
    return mid

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Calculate MACD
    m, s, h = macd(df["Close"])
    df["MACD"] = m
    df["MACD_signal"] = s
    df["MACD_hist"] = h

    # Calculate ADX - ensure it's a Series
    adx_result = adx_series(df["High"], df["Low"], df["Close"])
    if isinstance(adx_result, pd.DataFrame):
        df["ADX"] = adx_result.iloc[:, 0]  # Take first column if DataFrame
    else:
        df["ADX"] = adx_result

    return df

# ========= DATA PROVIDERS =========
CACHE_TTL = 1  # seconds - Real-time updates like trading websites
@st.cache_data(show_spinner=False, ttl=CACHE_TTL)
def load_yfinance(symbol: str, period=DEFAULT_PERIOD_YF, interval=DEFAULT_TIMEFRAME) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty: raise ValueError(f"yfinance returned no data for {symbol}")

    # Handle multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    # Rename columns to standard format
    df = df.rename(columns=str.title)
    df.index = ensure_tz(df.index)
    return df

@st.cache_data(show_spinner=False, ttl=CACHE_TTL)
def load_ccxt(market: str, exchange: Literal["okx","binance"]="okx",
              timeframe=DEFAULT_TIMEFRAME, limit=2000) -> pd.DataFrame:
    import ccxt
    try:
        ex = getattr(ccxt, exchange)(); ex.load_markets()
        ohlcv = ex.fetch_ohlcv(market, timeframe=timeframe, limit=limit)
    except Exception:
        ex = ccxt.binance(); ex.load_markets()
        ohlcv = ex.fetch_ohlcv(market, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["Datetime","Open","High","Low","Close","Volume"])
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="ms", utc=True).dt.tz_convert(TZ)
    return df.set_index("Datetime")

@st.cache_data(show_spinner=False, ttl=CACHE_TTL)
def load_twelvedata(symbol: str, interval="30min", outputsize=5000, tz=TZ) -> pd.DataFrame:
    key = st.session_state.get("TD_KEY") or st.secrets.get("TWELVEDATA_API_KEY") or TWELVEDATA_API_KEY_DEFAULT
    url = (
        "https://api.twelvedata.com/time_series"
        f"?symbol={symbol}&interval={interval}&outputsize={outputsize}&dp=8&format=JSON&apikey={key}"
    )
    js = requests.get(url, timeout=30).json()
    if "values" not in js:
        msg = js.get("message") or js.get("status") or "Twelve Data error"
        raise ValueError(f"Twelve Data: {msg}")
    df = pd.DataFrame(js["values"])
    df.rename(columns={"datetime":"Datetime","open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"}, inplace=True)
    df[["Open","High","Low","Close","Volume"]] = df[["Open","High","Low","Close","Volume"]].astype(float)
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True).dt.tz_convert(tz)
    return df.sort_values("Datetime").set_index("Datetime")

@st.cache_data(show_spinner=False, ttl=CACHE_TTL)
def load_alpha_fx(pair="EUR/USD", interval="30min", tz=TZ) -> pd.DataFrame:
    key = st.session_state.get("AV_KEY") or st.secrets.get("ALPHAVANTAGE_API_KEY") or ALPHAVANTAGE_API_KEY_DEFAULT
    frm, to = [x.strip().upper() for x in pair.split("/")]
    url = (
        "https://www.alphavantage.co/query"
        f"?function=FX_INTRADAY&from_symbol={frm}&to_symbol={to}&interval={interval}&outputsize=full&apikey={key}"
    )
    js = requests.get(url, timeout=30).json()
    fld = f"Time Series FX ({interval})"
    if fld not in js:
        msg = js.get("Note") or js.get("Error Message") or js.get("Information") or "Alpha Vantage error (check key or rate limits)"
        raise ValueError(f"Alpha Vantage: {msg}")
    df = pd.DataFrame(js[fld]).T.rename(columns={"1. open":"Open","2. high":"High","3. low":"Low","4. close":"Close"})
    df = df.astype(float); df["Volume"] = np.nan
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(tz)
    return df.sort_index()[["Open","High","Low","Close","Volume"]]

# ========= OKX MARKET (pro) HELPERS =========
def okx_ticker(inst_id="BTC-USDT"):
    r = requests.get(f"{OKX_BASE}/api/v5/market/ticker", params={"instId": inst_id}, timeout=10)
    r.raise_for_status(); js = r.json()
    return (js.get("data") or [])[0] if js.get("data") else {}

def okx_orderbook(inst_id="BTC-USDT", sz=25):
    r = requests.get(f"{OKX_BASE}/api/v5/market/books", params={"instId": inst_id, "sz": sz}, timeout=10)
    r.raise_for_status(); js = r.json()
    if not js.get("data"): return pd.DataFrame(), pd.DataFrame()
    d = js["data"][0]
    bids = pd.DataFrame(d["bids"], columns=["price","size","liqPx","cnt"])
    asks = pd.DataFrame(d["asks"], columns=["price","size","liqPx","cnt"])
    for df in (bids, asks):
        df["price"] = df["price"].astype(float)
        df["size"]  = df["size"].astype(float)
        df["cum"]   = df["size"].cumsum()
    return bids, asks

def okx_trades(inst_id="BTC-USDT", limit=50):
    r = requests.get(f"{OKX_BASE}/api/v5/market/trades", params={"instId": inst_id, "limit": limit}, timeout=10)
    r.raise_for_status(); js = r.json()
    trades = pd.DataFrame(js.get("data", []))
    if trades.empty: return trades
    trades["px"]  = trades["px"].astype(float)
    trades["sz"]  = trades["sz"].astype(float)
    trades["side"]= trades["side"].astype(str)
    trades["ts"]  = pd.to_datetime(trades["ts"].astype(int), unit="ms", utc=True).dt.tz_convert(TZ)
    return trades[["ts","side","px","sz"]].sort_values("ts", ascending=False)

# === ATR (for stops/sizing) ===
def atr_series(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

# === Watchlist for correlation ===
ASSET_SHORTLIST = ["BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","DOGE/USDT"]
FX_COMMOD = {"XAUUSD=X":"Gold", "EURUSD=X":"EURUSD"}

@st.cache_data(ttl=1, show_spinner=False)  # Real-time updates
def load_close_for_corr() -> pd.DataFrame:
    frames = []
    # crypto via OKX
    for m in ASSET_SHORTLIST:
        try:
            d = load_ccxt(m, exchange="okx", timeframe="1h", limit=600)[["Close"]]
            d.columns = [m]
            frames.append(d)
        except Exception:
            pass
    # Gold / EURUSD via Yahoo
    for sym in FX_COMMOD.keys():
        try:
            d = load_yfinance(sym, period="90d", interval="60m")[["Close"]]
            d.columns = [sym]
            frames.append(d)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    dfc = pd.concat(frames, axis=1).dropna(how="all")
    return dfc.pct_change().dropna()

def tradingview_widget(symbol_tv="OKX:BTCUSDT", height=600, theme="dark"):
    components.html(f"""
    <div id="tvwrap" style="width:100%;height:{height}px;min-height:{height}px;overflow:hidden;"></div>
    <script src="https://s3.tradingview.com/tv.js"></script>
    <script>
      new TradingView.widget({{
        width: "100%",
        height: {height},
        symbol: "{symbol_tv}",
        interval: "1",
        timezone: "Etc/UTC",
        theme: "{theme}",
        style: "1",
        locale: "en",
        toolbar_bg: "rgba(0,0,0,0)",
        enable_publishing: false,
        hide_side_toolbar: false,
        allow_symbol_change: true,
        studies: ["RSI@tv-basicstudies","MACD@tv-basicstudies"],
        container_id: "tvwrap"
      }});
    </script>
    """, height=height)

# ========= GARCH =========
@dataclass
class GarchFit:
    name: str
    res: any
    cond_vol: pd.Series
    sigma1: float

def _fit_model(name: str, am, idx: pd.DatetimeIndex) -> GarchFit:
    res = am.fit(disp="off")
    vol = pd.Series(res.conditional_volatility, index=idx, name=f"{name}_sigma")
    sigma1 = float(np.sqrt(res.forecast(horizon=1).variance.iloc[-1,0]))
    return GarchFit(name, res, vol, sigma1)

def fit_garch_family(returns: pd.Series) -> Dict[str, GarchFit]:
    y = (returns * 100.0).dropna()
    fits: Dict[str,GarchFit] = {}
    fits["GARCH11"] = _fit_model("GARCH11", arch_model(y, vol="GARCH", p=1, q=1, dist="t", mean="constant"), y.index)
    fits["EGARCH"]  = _fit_model("EGARCH",  arch_model(y, vol="EGARCH", p=1, q=1, dist="t", mean="constant"), y.index)
    fits["GJR"]     = _fit_model("GJR",     arch_model(y, vol="GARCH", p=1, o=1, q=1, power=2.0, dist="t", mean="constant"), y.index)
    fits["APARCH"]  = _fit_model("APARCH",  arch_model(y, vol="APARCH", p=1, q=1, dist="t", mean="constant"), y.index)
    return fits

def best_by_aic(fits: Dict[str,GarchFit]) -> GarchFit:
    return min(fits.values(), key=lambda f: f.res.aic)

# ========= Plotting =========
def tv_figure(df: pd.DataFrame, garch: Optional[GarchFit], title: str,
              fib_lookback: int = 200, figscale: float = 1.05):
    rc = {
        "figure.facecolor": "#0e1117",
        "axes.facecolor":   "#0e1117",
        "savefig.facecolor":"#0e1117",
        "axes.edgecolor":   "#dddddd",
        "xtick.color":      "#dddddd",
        "ytick.color":      "#dddddd",
        "text.color":       "#dddddd",
    }
    style = mpf.make_mpf_style(base_mpf_style="yahoo", gridstyle=":", y_on_right=True, rc=rc)
    add = [
        mpf.make_addplot(df["MACD"], panel=1, ylabel="MACD"),
        mpf.make_addplot(df["MACD_signal"], panel=1),
        mpf.make_addplot(df["MACD_hist"], type="bar", panel=1, alpha=0.35),
        mpf.make_addplot(df["ADX"], panel=2, ylabel="ADX"),
    ]
    if garch is not None:
        vol = garch.cond_vol.reindex(df.index)
        add.append(mpf.make_addplot(vol, panel=3, ylabel=f"{garch.name} σ(t) [%]"))

    fig, axes = mpf.plot(
        df, type="candle", addplot=add, volume=True,
        style=style, title=title,
        figratio=(21, 9), figscale=figscale,
        tight_layout=True, returnfig=True, closefig=False
    )

    price_ax = axes[0]
    _, _, levels = fib_levels(df, lookback=fib_lookback)
    for lbl, y in levels.items():
        price_ax.axhline(y, linestyle="--", linewidth=1, alpha=0.5)
        price_ax.text(df.index[-1], y, f"  {lbl}", va="center", fontsize=8)

    fig.canvas.draw()
    return fig

def compare_sigmas_chart(fits: Dict[str,GarchFit], index: pd.DatetimeIndex, title: str):
    plt.figure(figsize=(14,6))
    for name in ["GARCH11","EGARCH","GJR","APARCH"]:
        s = fits[name].cond_vol.reindex(index)
        plt.plot(s.index, s.values, label=name)
    plt.title(title); plt.legend(); plt.grid(True, linestyle=":"); plt.tight_layout()
    return plt.gcf()

def show_mpf(fig, fallback_max_px=(3600, 2000), dpi=120):
    try:
        st.pyplot(fig, use_container_width=True, clear_figure=False)
    except Exception:
        import PIL.Image as PILImage
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        im = PILImage.open(buf)
        im.thumbnail(fallback_max_px)
        out = io.BytesIO()
        im.save(out, format="PNG")
        out.seek(0)
        st.image(out.getvalue(), use_container_width=True)
    finally:
        plt.close(fig)

# ========= ARIMA → TV-lite =========
def arima_forecast_prices(df: pd.DataFrame, steps=16, order=(1,0,1)) -> pd.DataFrame:
    y = log_returns(df).dropna()
    if len(y) < 30: raise ValueError("Not enough data for ARIMA.")
    m = ARIMA(y, order=order).fit()
    fc = m.get_forecast(steps=steps)
    mu = fc.predicted_mean
    band = fc.conf_int(alpha=0.2)
    last = df["Close"].iloc[-1]
    mean_path = last * np.exp(np.cumsum(mu.values))
    lo_path   = last * np.exp(np.cumsum(band.iloc[:,0].values))
    hi_path   = last * np.exp(np.cumsum(band.iloc[:,1].values))
    base = df.index[-1]; freq = pd.infer_freq(df.index) or "30T"
    idx = pd.date_range(base, periods=steps+1, freq=freq, tz=base.tz)[1:]
    return pd.DataFrame({"mean": mean_path, "lo": lo_path, "hi": hi_path}, index=idx)

def render_lightweight_chart(df: pd.DataFrame, forecast: Optional[pd.DataFrame]=None, height=480):
    candles = [{"time": int(ts.timestamp()), "open": float(r.Open), "high": float(r.High),
                "low": float(r.Low), "close": float(r.Close)}
               for ts, r in df[["Open","High","Low","Close"]].tail(400).iterrows()]
    f_mean = [{"time": int(ts.timestamp()), "value": float(v)} for ts, v in (forecast["mean"].items() if forecast is not None else [])]
    f_lo   = [{"time": int(ts.timestamp()), "value": float(v)} for ts, v in (forecast["lo"].items() if forecast is not None else [])]
    f_hi   = [{"time": int(ts.timestamp()), "value": float(v)} for ts, v in (forecast["hi"].items() if forecast is not None else [])]

    html = f"""
    <div id="lw" style="height:{height}px;"></div>
    <script src="https://unpkg.com/lightweight-charts@4.2.1/dist/lightweight-charts.standalone.production.js"></script>
    <script>
      const chart = LightweightCharts.createChart(document.getElementById('lw'), {{
        layout: {{ background: {{ type:'solid', color: 'transparent' }}, textColor: '#ddd' }},
        grid: {{ vertLines: {{ color:'rgba(70,70,70,0.3)' }}, horzLines: {{ color:'rgba(70,70,70,0.3)' }} }},
        timeScale: {{ timeVisible: true, secondsVisible: false }},
        crosshair: {{ mode: 0 }}
      }});
      const cs = chart.addCandlestickSeries();
      cs.setData({json.dumps(candles)});
      {"const m = chart.addLineSeries({ color:'#3BA7FF', priceLineVisible:false, lineWidth:2 }); m.setData(" + json.dumps(f_mean) + ");" if f_mean else ""}
      {"const lo = chart.addLineSeries({ color:'#65C466', lineStyle: 1, priceLineVisible:false }); lo.setData(" + json.dumps(f_lo) + ");" if f_lo else ""}
      {"const hi = chart.addLineSeries({ color:'#E57373', lineStyle: 1, priceLineVisible:false }); hi.setData(" + json.dumps(f_hi) + ");" if f_hi else ""}
    </script>
    """
    components.html(html, height=height)

def plotly_mpf(df: pd.DataFrame, garch: Optional[GarchFit], title: str, fib_lookback: int = 200):
    view = df.tail(600).copy()
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.58, 0.16, 0.14, 0.12])
    fig.add_trace(go.Candlestick(x=view.index, open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"],
                                 name="Price", increasing_line_width=1, decreasing_line_width=1), row=1, col=1)
    fig.add_trace(go.Bar(x=view.index, y=view["Volume"].fillna(0), name="Volume", opacity=0.3), row=1, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["MACD"], name="MACD", mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["MACD_signal"], name="Signal", mode="lines"), row=2, col=1)
    fig.add_trace(go.Bar(x=view.index, y=view["MACD_hist"], name="Hist", opacity=0.4), row=2, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["ADX"], name="ADX", mode="lines"), row=3, col=1)
    if garch is not None and isinstance(garch.cond_vol, pd.Series):
        vol = garch.cond_vol.reindex(view.index)
        fig.add_trace(go.Scatter(x=vol.index, y=vol.values, name=f"{garch.name} σ(t) [%]", mode="lines"), row=4, col=1)
    _, _, levels = fib_levels(view, lookback=fib_lookback)
    for lbl, y in levels.items():
        fig.add_shape(type="line", x0=view.index[0], x1=view.index[-1], y0=y, y1=y,
                      line=dict(width=1, dash="dot", color="rgba(180,180,180,0.7)"),
                      xref="x", yref="y")
        fig.add_annotation(x=view.index[-1], y=y, text=f" {lbl}", showarrow=False,
                           yanchor="middle", font=dict(size=10, color="#ddd"))
    fig.update_layout(template="plotly_dark", title=title, margin=dict(l=16, r=12, t=48, b=8),
                      xaxis_rangeslider_visible=False, height=720,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="Price/Vol", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="ADX", row=3, col=1)
    fig.update_yaxes(title_text="σ(t) [%]", row=4, col=1)
    return fig

def plotly_arima_chart(df: pd.DataFrame, forecast: Optional[pd.DataFrame] = None,
                       height: int = 540, lookback: int = 400):
    view = df.tail(lookback).copy()
    x_candles = pd.DatetimeIndex(view.index).tz_convert(None)
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(x=x_candles, open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"],
                                 name="Price", increasing_line_width=1, decreasing_line_width=1), row=1, col=1)
    fig.add_trace(go.Bar(x=x_candles, y=view["Volume"].fillna(0), name="Volume", opacity=0.25), row=1, col=1)
    if forecast is not None and not forecast.empty:
        f = forecast.copy()
        x_fc = pd.DatetimeIndex(f.index).tz_convert(None)
        fig.add_trace(go.Scatter(x=x_fc, y=f["hi"], name="ARIMA band (hi)", mode="lines",
                                 line=dict(width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_fc, y=f["lo"], name="ARIMA ±band", mode="lines",
                                 line=dict(width=0), fill="tonexty", fillcolor="rgba(59,167,255,0.18)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_fc, y=f["mean"], name="ARIMA mean", mode="lines", line=dict(width=2)), row=1, col=1)
    fig.update_layout(template="plotly_dark", title="Candles + ARIMA mean & band overlay",
                      xaxis_rangeslider_visible=False, height=height,
                      margin=dict(l=16, r=12, t=40, b=8),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="Price / Volume")
    return fig

# --- Symbol mappers & fallbacks ---
def td_default_symbol_for_preset(cfg: dict) -> str:
    """
    Map preset to a Twelve Data symbol. Twelve Data likes: BTC/USDT, EUR/USD, XAU/USD.
    """
    if cfg.get("src") == "ccxt":
        return cfg.get("market", "BTC/USDT").replace("-", "/")
    sym = (cfg.get("symbol") or "EURUSD=X").replace("=X", "")
    return f"{sym[:3]}/{sym[3:]}" if len(sym) == 6 else sym

def pair_to_yf_fx(pair: str) -> str:
    """'EUR/USD' -> 'EURUSD=X' (Yahoo FX/metals convention)"""
    p = pair.replace(" ", "").upper()
    if "/" in p and len(p.replace("/", "")) == 6:
        return p.replace("/", "") + "=X"
    return p

def is_fx_or_metal(pair: str) -> bool:
    p = pair.replace(" ", "").upper()
    return "/" in p and len(p.replace("/", "")) == 6

# ========= UI =========
st.set_page_config(page_title="GARCH Lab", layout="wide")
st.title("📈 GARCH Lab — Volatility, Forecast & Market View")

# ========= TRUE REAL-TIME TICKER (No Branding!) =========
st.markdown("### ⚡ Live Market — Real-Time")

with st.sidebar:
    st.header("Quick picks")
    preset = st.radio("Asset", list(ASSET_PRESETS.keys()), index=0)

    st.header("Provider (for GARCH/ARIMA)")
    provider = st.selectbox("Data source", [
        "Exchange (OKX/Binance via CCXT)", "Yahoo Finance (free)",
        "Twelve Data (key)", "Alpha Vantage FX (key)"], index=0)

# Get current asset symbol for real-time display
cfg = ASSET_PRESETS[preset]
if preset == "BTC/USDT (OKX)":
    tv_symbol = "OKX:BTCUSDT"
    display_name = "BTC/USDT"
elif preset == "ETH/USDT (OKX)":
    tv_symbol = "OKX:ETHUSDT"
    display_name = "ETH/USDT"
elif preset == "EURUSD (Yahoo)":
    tv_symbol = "FX:EURUSD"
    display_name = "EUR/USD"
elif preset == "Gold XAUUSD (Yahoo)":
    tv_symbol = "TVC:GOLD"
    display_name = "Gold (XAU/USD)"
else:
    tv_symbol = "OKX:BTCUSDT"
    display_name = cfg.get("market", "BTC/USDT")

# Dynamic ticker tape based on selection (minimal branding)
components.html(f"""
<style>
  .ticker-wrap {{
    width: 100%;
    overflow: hidden;
    background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 8px;
    padding: 12px 0;
  }}
  .ticker {{
    display: flex;
    animation: scroll 30s linear infinite;
  }}
  .ticker-item {{
    display: flex;
    align-items: center;
    padding: 0 30px;
    white-space: nowrap;
  }}
  .ticker-item .symbol {{
    font-weight: bold;
    color: #fff;
    font-size: 14px;
    margin-right: 8px;
  }}
  .ticker-item .price {{
    color: #00ff88;
    font-size: 16px;
    font-weight: bold;
  }}
  .ticker-item .change.up {{
    color: #00ff88;
    font-size: 12px;
    margin-left: 8px;
  }}
  .ticker-item .change.down {{
    color: #ff4444;
    font-size: 12px;
    margin-left: 8px;
  }}
  .live-dot {{
    width: 8px;
    height: 8px;
    background: #00ff88;
    border-radius: 50%;
    margin-right: 8px;
    animation: pulse 1s infinite;
  }}
  @keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.5; }}
  }}
  @keyframes scroll {{
    0% {{ transform: translateX(0); }}
    100% {{ transform: translateX(-50%); }}
  }}
</style>
<div class="ticker-wrap">
  <div class="ticker">
    <div class="ticker-item">
      <span class="live-dot"></span>
      <span class="symbol">🔴 LIVE</span>
    </div>
    <div class="ticker-item">
      <span class="symbol">{display_name}</span>
      <span class="price" id="price-display">Loading...</span>
    </div>
    <div class="ticker-item">
      <span class="symbol">Selected Asset</span>
      <span class="price">✓ {preset}</span>
    </div>
    <div class="ticker-item">
      <span class="symbol">Updates</span>
      <span class="price">Real-Time</span>
    </div>
  </div>
</div>
""", height=50)

# Dynamic chart - changes with asset selection
col_chart, col_info = st.columns([3, 1])

with col_chart:
    # TradingView chart with minimal branding (symbol changes based on selection!)
    components.html(f"""
    <div id="tv_chart" style="height:280px;width:100%;background:#0e1117;border-radius:8px;overflow:hidden;">
    </div>
    <script src="https://s3.tradingview.com/tv.js"></script>
    <script>
      new TradingView.widget({{
        "autosize": true,
        "symbol": "{tv_symbol}",
        "interval": "1",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#0e1117",
        "enable_publishing": false,
        "hide_top_toolbar": false,
        "hide_legend": true,
        "hide_side_toolbar": true,
        "allow_symbol_change": false,
        "save_image": false,
        "container_id": "tv_chart",
        "hide_volume": true,
        "backgroundColor": "#0e1117"
      }});
    </script>
    """, height=290)

with col_info:
    # Custom price display using our WebSocket data (no branding!)
    try:
        # Get fresh ticker data
        ticker_data = get_fast_ticker_rest("okx", display_name.replace("/", "-"))
        
        last_price = ticker_data.get('last', 0)
        change_24h = ticker_data.get('change24h', 0)
        high_24h = ticker_data.get('high24h', 0)
        low_24h = ticker_data.get('low24h', 0)
        
        if change_24h == 0 and ticker_data.get('open24h', 0) > 0:
            change_24h = ((last_price - ticker_data['open24h']) / ticker_data['open24h']) * 100
        
        # Display with custom styling
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);padding:20px;border-radius:12px;text-align:center;">
            <div style="font-size:14px;color:#888;margin-bottom:8px;">{display_name}</div>
            <div style="font-size:28px;font-weight:bold;color:#fff;">${last_price:,.2f}</div>
            <div style="font-size:16px;color:{'#00ff88' if change_24h >= 0 else '#ff4444'};margin-top:8px;">
                {'+' if change_24h >= 0 else ''}{change_24h:.2f}%
            </div>
            <div style="margin-top:16px;display:flex;justify-content:space-between;">
                <div>
                    <div style="font-size:10px;color:#666;">24h High</div>
                    <div style="font-size:12px;color:#fff;">${high_24h:,.2f}</div>
                </div>
                <div>
                    <div style="font-size:10px;color:#666;">24h Low</div>
                    <div style="font-size:12px;color:#fff;">${low_24h:,.2f}</div>
                </div>
            </div>
            <div style="margin-top:12px;font-size:10px;color:#00ff88;">
                🟢 LIVE • Real-Time Data
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.metric(display_name, "Loading...", help=str(e))

st.markdown("---")

with st.sidebar:
    # cfg already defined above
    if provider == "Twelve Data (key)":
        st.session_state["TD_KEY"] = st.text_input("Twelve Data API Key", value=TWELVEDATA_API_KEY_DEFAULT, type="password")
        td_default = td_default_symbol_for_preset(cfg)
        st.session_state["TD_SYMBOL"] = st.text_input(
            "Twelve Data symbol", value=td_default,
            help="Examples: BTC/USDT, ETH/USDT, EUR/USD, XAU/USD"
        )

    if provider == "Alpha Vantage FX (key)":
        st.session_state["AV_KEY"] = st.text_input("Alpha Vantage API Key", value=ALPHAVANTAGE_API_KEY_DEFAULT, type="password")
        st.session_state["AV_PAIR"] = st.text_input(
            "FX pair (AV only)", value="EUR/USD",
            help="Examples: EUR/USD, GBP/USD, USD/JPY (no crypto/metals on AV)"
        )


    st.header("⚡ Refresh Settings")
    st.caption("🔴 **WebSocket LIVE** • Prices update in real-time (10-50ms)")
    auto = st.toggle("Auto-refresh Analytics", value=True)
    every_sec = st.number_input("Analytics refresh (sec)", 5, 60, 10, step=5, disabled=not auto, 
                                help="WebSocket streams prices continuously. This controls GARCH/charts refresh.")
    if auto: st_autorefresh(interval=int(every_sec*1000), key="refresh_main")

    st.header("Display")
    use_plotly = st.toggle("Use Plotly renderer (no PIL)", value=True)
    fib_n = st.slider("Fibonacci lookback", 100, 500, 200, 50)
    show_grid = st.checkbox("Show 4-up (one chart per model)", value=True)
    arima_steps = st.slider("ARIMA steps (future)", 8, 48, 16, 4)

# Load data for GARCH/ARIMA
try:
    with st.spinner("Loading data..."):
        cfg = ASSET_PRESETS[preset]

        if provider.startswith("Exchange"):
            if cfg["src"] == "ccxt":
                df = load_ccxt(cfg["market"], exchange=cfg["exchange"], timeframe=cfg["timeframe"])
                title_base = f"{cfg['market']} • {cfg['timeframe']} ({cfg['exchange'].upper()})"
            else:
                df = load_ccxt("BTC/USDT", exchange="okx", timeframe="30m")
                title_base = "BTC/USDT • 30m (OKX)"

        elif provider.startswith("Yahoo"):
            if cfg["src"] == "yfinance":
                try:
                    df = load_yfinance(cfg["symbol"], period=cfg["period"], interval=cfg["interval"])
                    title_base = f"{cfg['symbol']} • {cfg['interval']} (Yahoo)"
                except Exception:
                    fb = cfg.get("fallback")
                    if fb:
                        df = load_yfinance(fb, period=cfg["period"], interval=cfg["interval"])
                        title_base = f"{fb} • {cfg['interval']} (Yahoo)"
                    else:
                        raise
            else:
                df = load_yfinance("EURUSD=X", period="60d", interval="30m")
                title_base = "EURUSD=X • 30m (Yahoo)"

        elif provider.startswith("Twelve"):
            sym = st.session_state.get("TD_SYMBOL") or td_default_symbol_for_preset(cfg)
            # Convert USDT to USD for Twelve Data compatibility
            sym_original = sym
            if "USDT" in sym:
                sym = sym.replace("USDT", "USD")
            if "BTC/USD" in sym or "ETH/USD" in sym:
                st.info(f"Using {sym} for Twelve Data" + (f" (converted from {sym_original})" if sym != sym_original else ""))
            try:
                df = load_twelvedata(symbol=sym, interval="30min")
                title_base = f"{sym} • 30min (Twelve Data)"
            except Exception as e:
                # Try fallback to Yahoo Finance
                if "/" in sym:
                    if is_fx_or_metal(sym):
                        yfs = pair_to_yf_fx(sym)
                    else:
                        # Crypto - use Yahoo's format
                        yfs = sym.replace("/", "-")
                    st.warning(f"Twelve Data failed for '{sym}' → trying Yahoo ({yfs}). Reason: {str(e)[:80]}")
                    df = load_yfinance(yfs, period="60d", interval="30m")
                    title_base = f"{yfs} • 30m (Yahoo fallback)"
                else:
                    raise

        else:  # Alpha Vantage FX
            pair = st.session_state.get("AV_PAIR") or "EUR/USD"
            try:
                df = load_alpha_fx(pair=pair, interval="30min")
                title_base = f"{pair} • 30min (Alpha Vantage)"
            except Exception as e:
                if is_fx_or_metal(pair):
                    yfs = pair_to_yf_fx(pair)
                    df = load_yfinance(yfs, period="60d", interval="30m")
                    title_base = f"{yfs} • 30m (Yahoo fallback)"
                    st.warning(f"Alpha Vantage failed for '{pair}' → using Yahoo ({yfs}). Reason: {e}")
                else:
                    raise

        df = df.dropna()
        df = add_indicators(df)

except Exception as e:
    st.error(f"Data error: {e}")
    st.stop()

# Tabs
tab1, tab2, tab_sig, tab_opt, tab_portfolio, tab3 = st.tabs([
    "Volatility (GARCH)", "ARIMA forecast → TV-lite",
    "Signals & Risk", "Options (Black–Scholes)", "📊 Portfolio Analytics", "Market (pro)"])

# ===== Tab 1: GARCH =====
with tab1:
    with st.spinner("Fitting GARCH family..."):
        returns = log_returns(df)
        fits = fit_garch_family(returns)
        best = best_by_aic(fits)

    st.subheader("Best model by AIC")
    if use_plotly:
        fig_best = plotly_mpf(df, best, f"{title_base} — Best: {best.name}", fib_lookback=fib_n)
        st.plotly_chart(fig_best, use_container_width=True, theme="streamlit")
    else:
        fig_best = tv_figure(df, best, f"{title_base} — Best: {best.name}", fib_lookback=fib_n)
        show_mpf(fig_best)

    st.subheader("Model σ(t) comparison")
    cmp_fig = compare_sigmas_chart(fits, df.index, f"{title_base} — GARCH σ(t)")
    st.pyplot(cmp_fig, use_container_width=True); plt.close(cmp_fig)

    if show_grid:
        st.subheader("All models — full charts")
        gcols = st.columns(2)
        for i, name in enumerate(["GARCH11", "EGARCH", "GJR", "APARCH"]):
            with gcols[i % 2]:
                if use_plotly:
                    f = plotly_mpf(df, fits[name], f"{title_base} — {name}", fib_lookback=fib_n)
                    st.plotly_chart(f, use_container_width=True, theme="streamlit")
                else:
                    f = tv_figure(df, fits[name], f"{title_base} — {name}", fib_lookback=fib_n)
                    show_mpf(f)

# ===== Tab 2: ARIMA + Enhanced Forecasting =====
with tab2:
    st.subheader("📈 ARIMA Price Forecast")
    st.caption("Professional time series forecasting with confidence bands")
    
    # Clean controls in columns
    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 1])
    with ctrl1:
        forecast_steps = st.slider("Forecast Steps", 8, 48, 16, 4, help="How many periods to forecast ahead")
    with ctrl2:
        auto_order = st.toggle("Auto-Select Model", value=True, help="Automatically find best ARIMA order")
    with ctrl3:
        show_advanced = st.toggle("Show Advanced", value=False, help="Show detailed diagnostics")
    
    try:
        # Import enhanced functions
        from src.models.arima import (
            enhanced_arima_analysis, create_fan_chart_data, 
            get_forecast_summary_for_display, compare_models,
            log_returns as arima_log_returns
        )
        
        # Run enhanced analysis
        analysis = enhanced_arima_analysis(
            df, 
            steps=forecast_steps, 
            auto_order=auto_order,
            run_validation=show_advanced
        )
        
        if analysis['success']:
            fc = analysis['forecast']
            diag = analysis['diagnostics']
            
            # ===== MAIN FORECAST DISPLAY (Always visible) =====
            st.markdown("---")
            
            # Key metrics row
            m1, m2, m3, m4 = st.columns(4)
            
            with m1:
                st.metric(
                    "Model", 
                    f"ARIMA{analysis['best_order']}",
                    help="Auto-selected based on AIC"
                )
            with m2:
                expected_price = float(fc.mean_prices.iloc[-1])
                current_price = float(df["Close"].iloc[-1])
                change_pct = ((expected_price / current_price) - 1) * 100
                st.metric(
                    f"Forecast ({forecast_steps} steps)",
                    f"${expected_price:,.2f}",
                    delta=f"{change_pct:+.2f}%"
                )
            with m3:
                st.metric("AIC Score", f"{fc.aic:.1f}", help="Lower is better")
            with m4:
                status = "✅ Valid" if diag.is_residuals_uncorrelated else "⚠️ Check"
                st.metric("Model Status", status, help="Based on residual diagnostics")
            
            # ===== FAN CHART (Main visualization) =====
            st.markdown("### 📊 Forecast with Confidence Bands")
            
            # Create fan chart with Plotly
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            # Historical prices (last 100 bars)
            hist_data = df["Close"].tail(100)
            fig.add_trace(go.Scatter(
                x=hist_data.index,
                y=hist_data.values,
                mode='lines',
                name='Historical',
                line=dict(color='#3BA7FF', width=2)
            ))
            
            # Forecast bands (95% - widest, lightest)
            if fc.bands_95:
                fig.add_trace(go.Scatter(
                    x=fc.forecast_index.tolist() + fc.forecast_index.tolist()[::-1],
                    y=fc.bands_95[1].tolist() + fc.bands_95[0].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255, 165, 0, 0.15)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name='95% CI',
                    showlegend=True
                ))
            
            # Forecast bands (80%)
            if fc.bands_80:
                fig.add_trace(go.Scatter(
                    x=fc.forecast_index.tolist() + fc.forecast_index.tolist()[::-1],
                    y=fc.bands_80[1].tolist() + fc.bands_80[0].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255, 165, 0, 0.25)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name='80% CI',
                    showlegend=True
                ))
            
            # Forecast bands (50% - narrowest, darkest)
            if fc.bands_50:
                fig.add_trace(go.Scatter(
                    x=fc.forecast_index.tolist() + fc.forecast_index.tolist()[::-1],
                    y=fc.bands_50[1].tolist() + fc.bands_50[0].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255, 165, 0, 0.4)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name='50% CI',
                    showlegend=True
                ))
            
            # Forecast mean line
            fig.add_trace(go.Scatter(
                x=fc.forecast_index,
                y=fc.mean_prices,
                mode='lines',
                name='Forecast',
                line=dict(color='#FFA500', width=3, dash='solid')
            ))
            
            # Current price line
            fig.add_hline(
                y=current_price,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"Current: ${current_price:,.2f}"
            )
            
            fig.update_layout(
                template="plotly_dark",
                height=450,
                margin=dict(l=10, r=10, t=40, b=10),
                title=f"{title_base} - ARIMA Forecast",
                xaxis_title="",
                yaxis_title="Price ($)",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
            
            # ===== FORECAST TABLE (Clean) =====
            st.markdown("### 📋 Forecast Values")
            
            # Create clean summary table
            fc_summary = pd.DataFrame({
                'Time': fc.forecast_index.strftime('%Y-%m-%d %H:%M'),
                'Mean ($)': fc.mean_prices.round(2),
                '50% Low': fc.bands_50[0].round(2) if fc.bands_50 else None,
                '50% High': fc.bands_50[1].round(2) if fc.bands_50 else None,
                '80% Low': fc.bands_80[0].round(2) if fc.bands_80 else None,
                '80% High': fc.bands_80[1].round(2) if fc.bands_80 else None,
            })
            fc_summary.index = range(1, len(fc_summary) + 1)
            fc_summary.index.name = 'Step'
            
            st.dataframe(fc_summary, use_container_width=True, height=200)
            
            # ===== ADVANCED SECTION (Hidden by default) =====
            if show_advanced:
                st.markdown("---")
                st.markdown("### 🔬 Advanced Analytics")
                
                # Model Comparison
                with st.expander("📊 Model Comparison", expanded=True):
                    if analysis['model_comparison']:
                        comp = analysis['model_comparison']
                        
                        # Create comparison table
                        comp_df = pd.DataFrame(comp.models)
                        if not comp_df.empty:
                            comp_df = comp_df[['name', 'aic', 'bic', 'residual_std', 'is_valid']]
                            comp_df.columns = ['Model', 'AIC', 'BIC', 'Residual σ', 'Valid']
                            comp_df['AIC'] = comp_df['AIC'].round(2)
                            comp_df['BIC'] = comp_df['BIC'].round(2)
                            comp_df['Residual σ'] = comp_df['Residual σ'].round(4)
                            comp_df['Valid'] = comp_df['Valid'].apply(lambda x: '✅' if x else '⚠️')
                            
                            st.dataframe(comp_df, use_container_width=True)
                            st.caption(f"🏆 Best Model: **{comp.best_model}** (lowest AIC)")
                
                # Diagnostics
                with st.expander("🔍 Model Diagnostics"):
                    d1, d2 = st.columns(2)
                    
                    with d1:
                        st.markdown("**Information Criteria**")
                        st.write(f"- AIC: {diag.aic:.2f}")
                        st.write(f"- BIC: {diag.bic:.2f}")
                        st.write(f"- Log-Likelihood: {diag.log_likelihood:.2f}")
                        
                        st.markdown("**Residual Statistics**")
                        st.write(f"- Mean: {diag.residual_mean:.6f}")
                        st.write(f"- Std Dev: {diag.residual_std:.6f}")
                        st.write(f"- Skewness: {diag.residual_skew:.4f}")
                        st.write(f"- Kurtosis: {diag.residual_kurtosis:.4f}")
                    
                    with d2:
                        st.markdown("**Statistical Tests**")
                        
                        # Ljung-Box test
                        lb_status = "✅ Pass" if diag.is_residuals_uncorrelated else "❌ Fail"
                        st.write(f"- Ljung-Box Test: {lb_status}")
                        st.write(f"  - Statistic: {diag.ljung_box_stat:.2f}")
                        st.write(f"  - p-value: {diag.ljung_box_pvalue:.4f}")
                        
                        # Jarque-Bera test
                        jb_status = "✅ Pass" if diag.is_residuals_normal else "⚠️ Non-normal"
                        st.write(f"- Jarque-Bera Test: {jb_status}")
                        st.write(f"  - Statistic: {diag.jarque_bera_stat:.2f}")
                        st.write(f"  - p-value: {diag.jarque_bera_pvalue:.4f}")
                
                # Walk-Forward Validation
                if analysis['validation']:
                    with st.expander("📈 Backtest Results (Walk-Forward Validation)"):
                        val = analysis['validation']
                        
                        if 'error' not in val:
                            v1, v2, v3 = st.columns(3)
                            v1.metric("Avg RMSE", f"{val['avg_rmse']:.4f}")
                            v2.metric("Avg MAE", f"{val['avg_mae']:.4f}")
                            v3.metric("Direction Accuracy", f"{val['avg_direction_accuracy']:.1f}%")
                            
                            st.caption(f"Based on {val['num_folds']} validation folds")
                            
                            # Show fold details
                            if val['folds']:
                                folds_df = pd.DataFrame(val['folds'])
                                st.dataframe(folds_df, use_container_width=True)
                        else:
                            st.warning(f"Validation error: {val['error']}")
        
        else:
            st.error(f"Forecast error: {analysis['error']}")
    
    except Exception as e:
        st.warning(f"ARIMA forecast unavailable: {e}")
        import traceback
        if show_advanced:
            st.code(traceback.format_exc())

# ===== Tab 3: Market (pro) =====
with tab3:
    st.subheader("Exchange-style view (OKX REST + TradingView)")

    preset_market = (ASSET_PRESETS[preset].get("market") or "BTC/USDT").replace("/", "-")
    inst_id = st.selectbox("Instrument", [preset_market, "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT"], index=0)
    enable_auto = st.toggle("Auto-refresh market panel", value=False)

    tv_symbol = f"OKX:{inst_id.replace('-','')}"
    depth_sz = st.slider("Orderbook depth (levels)", 10, 50, 25, 5)
    refresh = st.number_input("Refresh every (sec)", 10, 60, 20, step=5, disabled=not enable_auto)

    k1,k2,k3,k4 = st.columns(4)
    try:
        tk = okx_ticker(inst_id)
        last = float(tk.get("last", "nan"))
        open24h = float(tk.get("open24h", "nan"))
        pct = (last/open24h - 1) * 100 if (last==last and open24h==open24h) else np.nan
        vol = float(tk.get("volCcy24h", tk.get("vol24h", "nan")))
    except Exception:
        last = pct = vol = np.nan

    k1.metric("Last", f"{last:,.2f}")
    k2.metric("24h %", f"{pct:+.2f}%")
    k3.metric("24h Volume", f"{vol:,.0f}")
    k4.metric("Instrument", inst_id)

    left, right = st.columns([3,2])

    with left:
        tradingview_widget(tv_symbol, height=720, theme="dark")
        st.caption("Switch intervals & indicators from the chart toolbar.")

    with right:
        try:
            bids, asks = okx_orderbook(inst_id, sz=depth_sz)
            cA, cB = st.columns(2)
            with cA:
                st.markdown("**Bids**")
                st.dataframe(bids[["price","size","cum"]].round(6), use_container_width=True, height=260)
            with cB:
                st.markdown("**Asks**")
                st.dataframe(asks[["price","size","cum"]].round(6), use_container_width=True, height=260)
        except Exception as e:
            st.warning(f"Orderbook unavailable: {e}")

        try:
            tr = okx_trades(inst_id, limit=40)
            st.markdown("**Last trades**")
            st.dataframe(tr.rename(columns={"ts":"time","px":"price","sz":"size"}), use_container_width=True, height=260)
        except Exception as e:
            st.warning(f"Trades unavailable: {e}")

        try:
            if not bids.empty and not asks.empty:
                bsz = bids["size"].sum(); asz = asks["size"].sum()
                imbalance = (bsz - asz) / (bsz + asz) if (bsz + asz) > 0 else 0.0
                best_bid = float(bids["price"].iloc[0]); best_ask = float(asks["price"].iloc[0])
                micro_price = (best_bid*asz + best_ask*bsz) / (asz + bsz) if (asz+bsz)>0 else np.nan
                m1, m2, m3 = st.columns(3)
                m1.metric("OB Imbalance", f"{imbalance*100:.1f}%")
                m2.metric("Best Bid/Ask", f"{best_bid:,.2f} / {best_ask:,.2f}")
                m3.metric("Micro-price", f"{micro_price:,.2f}")
        except Exception:
            pass

    if enable_auto:
        st_autorefresh(interval=int(refresh*1000), key=f"pro_{inst_id}_{depth_sz}")

# ===== Tab 4: Signals & Risk =====
with tab_sig:
    st.subheader("MACD×ADX Signals + Risk")

    adx_min = st.slider("ADX filter (trend strength)", 10, 35, 18, 1)
    sigs = macd_adx_signals(df, adx_min=adx_min)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("**Recent signals**")
        st.dataframe(sigs.tail(12).rename(columns={"Close":"Price"}), use_container_width=True, height=280)
    with c2:
        mstats = simple_pnl(df, sigs)
        st.markdown("**Quick backtest (toy)**")
        st.metric("Trades", mstats["trades"])
        st.metric("Win rate", f"{mstats['win_rate']:.1f}%")
        st.metric("P&L (sum, %)", f"{mstats['pnl_pct']:.2f}%")

    st.markdown("---")
    st.markdown("### Risk — Expected Move & VaR")
    best_local = best_by_aic(fit_garch_family(log_returns(df)))
    bars_per_year = infer_bars_per_year(df.index)
    sigma_ann = best_local.sigma1 * np.sqrt(bars_per_year)
    horizon = st.slider("Horizon (days)", 1, 14, 3, 1)
    S = float(df["Close"].iloc[-1])
    em = expected_move(S, sigma_ann, horizon)
    st.metric("Annualized σ (from GARCH best)", f"{sigma_ann:.2f}%")
    st.metric(f"~Expected move in {horizon}d", f"±{em:,.2f}")

    var95 = hist_var(df, conf=0.95)
    st.metric("1-day VaR 95% (hist)", f"{(S*var95):,.2f}")

    st.markdown("---")
    st.markdown("### Cross-asset correlation (last 90d, 1h bars)")
    corr_ret = load_close_for_corr()
    if not corr_ret.empty:
        corr = corr_ret.corr().round(2)
        st.dataframe(corr, use_container_width=True, height=300)
    else:
        st.info("No correlation data available yet.")

    st.markdown("---")
    st.markdown("### Position sizing (ATR stop) + Capped Kelly")
    atr_now = float(atr_series(df["High"], df["Low"], df["Close"]).iloc[-1])
    atr_mult = st.slider("ATR multiple for stop", 0.5, 3.0, 1.5, 0.1)
    stop_dist = atr_mult * atr_now
    st.metric("ATR (14)", f"{atr_now:,.4f}")
    st.metric("Stop distance", f"{stop_dist:,.4f}")

    acc = st.number_input("Account size (USD)", 100.0, 10_000_000.0, 5000.0, step=100.0)
    risk_pct = st.slider("Risk per trade (%)", 0.1, 5.0, 1.0, 0.1)
    risk_amt = acc * (risk_pct/100.0)
    qty = (risk_amt / stop_dist) if stop_dist > 0 else 0.0
    st.metric("Suggested size (spot units)", f"{qty:,.6f}")

    tp, sl = 0.008, 0.006
    R = tp / sl
    p = (mstats["win_rate"] / 100.0) if mstats["trades"] else 0.5
    q = 1 - p
    kelly = p - q / R
    kelly_cap = min(max(kelly, 0.0), 0.20)
    st.metric("Kelly fraction (cap 20%)", f"{kelly_cap*100:.1f}%")
    st.caption("Sizing guide: risk ≤ Kelly-cap; ATR stop keeps risk in price terms.")

    st.markdown("**Export recent signals**")
    sig_csv = sigs.tail(200).to_csv().encode()
    st.download_button("Download signals (CSV)", sig_csv, file_name="signals_recent.csv", mime="text/csv")

# ===== Tab 5: Options =====
with tab_opt:
    st.subheader("Black–Scholes — quick pricer & Greeks")

    S = float(df["Close"].iloc[-1])
    c1,c2,c3,c4 = st.columns(4)
    K = c1.number_input("Strike (K)", value=round(S, 2))
    d_to_exp = c2.number_input("Days to expiry", 1, 365, 30)
    r = c3.number_input("Risk-free r (annual, %)", 0.0, 20.0, 4.0)/100.0
    side = c4.selectbox("Type", ["Call","Put"])

    best_local = best_by_aic(fit_garch_family(log_returns(df)))
    bars_per_year = infer_bars_per_year(df.index)
    sigma_ann = best_local.sigma1 * np.sqrt(bars_per_year)/100.0
    sigma = st.slider("Volatility σ (annual, %)", 5.0, 200.0, float(sigma_ann*100.0), 0.5)/100.0

    T = d_to_exp/365.0
    fair = bs_price(S,K,T,r,sigma, side)
    gks = bs_greeks(S,K,T,r,sigma, side)

    c5,c6 = st.columns(2)
    with c5:
        st.metric("Underlying (S)", f"{S:,.2f}")
        st.metric("Fair value", f"{fair:,.2f}")
    with c6:
        st.json({k: (float(v) if isinstance(v,(int,float,np.floating)) else v) for k,v in gks.items()})

    st.markdown("**Implied vol from market price (optional)**")
    q = st.number_input("Quoted option price", value=0.0)
    if q > 0:
        iv = implied_vol_bisect(S,K,T,r,q,side)
        st.metric("Implied σ (annual, %)", f"{iv*100.0:.2f}%")

# ===== Tab 6: Portfolio Analytics (Bloomberg-style) =====
with tab_portfolio:
    st.subheader("📊 Portfolio Analytics Dashboard")
    st.markdown("*Bloomberg Terminal-style performance analytics and risk metrics*")
    
    try:
        # Calculate returns and metrics
        prices = df["Close"].copy()
        returns = calculate_returns(prices)
        
        # Calculate all performance metrics
        metrics = calculate_all_metrics(prices, returns)
        
        # Display key metrics in cards (Bloomberg-style)
        st.markdown("### 📈 Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Return",
                f"{metrics.total_return:.2f}%",
                delta=f"Annual: {metrics.annual_return:.2f}%"
            )
        
        with col2:
            st.metric(
                "Sharpe Ratio",
                f"{metrics.sharpe_ratio:.2f}",
                delta="Risk-adjusted return",
                delta_color="off"
            )
        
        with col3:
            st.metric(
                "Max Drawdown",
                f"{metrics.max_drawdown:.2f}%",
                delta=f"{metrics.max_drawdown_duration} periods",
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                "Volatility (Annual)",
                f"{metrics.annual_volatility:.2f}%",
                delta=f"Sortino: {metrics.sortino_ratio:.2f}",
                delta_color="off"
            )
        
        # Second row of metrics
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.metric("Win Rate", f"{metrics.win_rate:.1f}%")
        
        with col6:
            st.metric("Profit Factor", f"{metrics.profit_factor:.2f}")
        
        with col7:
            st.metric("Calmar Ratio", f"{metrics.calmar_ratio:.2f}")
        
        with col8:
            st.metric("Num Trades", f"{metrics.num_trades}")
        
        st.markdown("---")
        
        # Monthly Returns Heatmap
        st.markdown("### 📅 Monthly Returns Heatmap")
        try:
            monthly_table = monthly_returns_table(returns)
            if not monthly_table.empty:
                fig_monthly = create_monthly_returns_heatmap(
                    monthly_table,
                    title=f"{title_base} - Monthly Returns (%)"
                )
                st.plotly_chart(fig_monthly, use_container_width=True, theme="streamlit")
            else:
                st.info("Not enough data for monthly returns heatmap (need at least 1 month)")
        except Exception as e:
            st.warning(f"Monthly heatmap unavailable: {e}")
        
        st.markdown("---")
        
        # Drawdown Chart
        st.markdown("### 📉 Drawdown Analysis (Underwater Equity)")
        try:
            drawdown = calculate_drawdown(prices)
            fig_dd = create_drawdown_chart(
                prices,
                drawdown,
                title=f"{title_base} - Portfolio Drawdown"
            )
            st.plotly_chart(fig_dd, use_container_width=True, theme="streamlit")
            
            # Drawdown statistics
            dd_col1, dd_col2, dd_col3 = st.columns(3)
            with dd_col1:
                st.metric("Current Drawdown", f"{drawdown.iloc[-1]*100:.2f}%")
            with dd_col2:
                st.metric("Max Drawdown", f"{metrics.max_drawdown:.2f}%")
            with dd_col3:
                st.metric("Max DD Duration", f"{metrics.max_drawdown_duration} periods")
        except Exception as e:
            st.warning(f"Drawdown chart unavailable: {e}")
        
        st.markdown("---")
        
        # Two-column layout for additional charts
        left_col, right_col = st.columns(2)
        
        with left_col:
            # Rolling Sharpe Ratio
            st.markdown("### 📊 Rolling Sharpe Ratio (252-period)")
            try:
                roll_sharpe = rolling_sharpe(returns, window=min(252, len(returns)//2))
                if len(roll_sharpe.dropna()) > 0:
                    fig_rolling = create_rolling_metrics_chart(
                        roll_sharpe,
                        title="Rolling Sharpe Ratio"
                    )
                    st.plotly_chart(fig_rolling, use_container_width=True, theme="streamlit")
                    
                    # Current rolling Sharpe
                    current_sharpe = roll_sharpe.iloc[-1] if not pd.isna(roll_sharpe.iloc[-1]) else 0.0
                    st.metric("Current Rolling Sharpe", f"{current_sharpe:.2f}")
                else:
                    st.info("Not enough data for rolling Sharpe (need at least 126 periods)")
            except Exception as e:
                st.warning(f"Rolling Sharpe unavailable: {e}")
        
        with right_col:
            # Returns Distribution
            st.markdown("### 📊 Returns Distribution")
            try:
                fig_dist = create_returns_distribution(
                    returns,
                    title="Daily Returns Distribution"
                )
                st.plotly_chart(fig_dist, use_container_width=True, theme="streamlit")
                
                # Distribution statistics
                dist_col1, dist_col2 = st.columns(2)
                with dist_col1:
                    st.metric("Mean Return", f"{returns.mean()*100:.3f}%")
                    st.metric("Median Return", f"{returns.median()*100:.3f}%")
                with dist_col2:
                    st.metric("Std Dev", f"{returns.std()*100:.3f}%")
                    st.metric("Skewness", f"{returns.skew():.2f}")
            except Exception as e:
                st.warning(f"Returns distribution unavailable: {e}")
        
        st.markdown("---")
        
        # Correlation Matrix (if we have multi-asset data)
        st.markdown("### 🔗 Cross-Asset Correlation Matrix")
        try:
            corr_data = load_close_for_corr()
            if not corr_data.empty and len(corr_data.columns) > 1:
                corr_matrix = corr_data.corr()
                fig_corr = create_correlation_heatmap(
                    corr_matrix,
                    title="Asset Correlation Matrix (90d, 1h returns)"
                )
                st.plotly_chart(fig_corr, use_container_width=True, theme="streamlit")
            else:
                st.info("Correlation matrix requires multiple assets. Showing single-asset analytics only.")
        except Exception as e:
            st.warning(f"Correlation matrix unavailable: {e}")
        
        st.markdown("---")
        
        # Detailed Metrics Table
        st.markdown("### 📋 Detailed Performance Metrics")
        metrics_df = pd.DataFrame({
            "Metric": [
                "Total Return",
                "Annual Return",
                "Annual Volatility",
                "Sharpe Ratio",
                "Sortino Ratio",
                "Calmar Ratio",
                "Max Drawdown",
                "Max DD Duration",
                "Win Rate",
                "Profit Factor",
                "Number of Trades",
                "Average Win",
                "Average Loss"
            ],
            "Value": [
                f"{metrics.total_return:.2f}%",
                f"{metrics.annual_return:.2f}%",
                f"{metrics.annual_volatility:.2f}%",
                f"{metrics.sharpe_ratio:.2f}",
                f"{metrics.sortino_ratio:.2f}",
                f"{metrics.calmar_ratio:.2f}",
                f"{metrics.max_drawdown:.2f}%",
                f"{metrics.max_drawdown_duration} periods",
                f"{metrics.win_rate:.1f}%",
                f"{metrics.profit_factor:.2f}",
                f"{metrics.num_trades}",
                f"{metrics.avg_win:.3f}%",
                f"{metrics.avg_loss:.3f}%"
            ]
        })
        st.dataframe(metrics_df, use_container_width=True, height=500)
        
        # Export functionality
        st.markdown("### 💾 Export Analytics")
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            # Export metrics as CSV
            metrics_csv = metrics_df.to_csv(index=False).encode()
            st.download_button(
                "📥 Download Metrics (CSV)",
                metrics_csv,
                file_name=f"portfolio_metrics_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with export_col2:
            # Export returns data
            returns_df = pd.DataFrame({
                "Date": returns.index,
                "Return": returns.values * 100
            })
            returns_csv = returns_df.to_csv(index=False).encode()
            st.download_button(
                "📥 Download Returns (CSV)",
                returns_csv,
                file_name=f"returns_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"Portfolio analytics error: {e}")
        import traceback
        st.code(traceback.format_exc())

st.caption("Notes: For reliability use OKX/Binance via CCXT, Twelve Data, or Alpha Vantage (FX). Yahoo works well as a fallback for FX/metals.")
