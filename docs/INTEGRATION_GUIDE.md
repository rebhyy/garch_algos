# 🔗 Integration Guide

**How to Use the New Modular Architecture**

This guide shows you how to integrate all the new modules into your application.

---

## 📦 Quick Import Reference

```python
# Configuration
from config.settings import settings

# Logging
from src.utils.logging_config import setup_logging, get_logger

# Models
from src.models.garch import fit_garch_family, best_by_aic, annualize_volatility
from src.models.arima import arima_forecast_prices, auto_select_order
from src.models.ml_predictor import quick_ml_forecast, LSTMPredictor, ProphetPredictor

# Indicators
from src.indicators.technical import add_all_indicators, macd, rsi, adx_series, atr_series

# Alerts
from src.triggers.alert_system import (
    AlertSystem,
    detect_dip_buy,
    detect_breakout,
    detect_volatility_spike,
    detect_macd_crossover,
    scan_all_triggers,
    console_handler,
    email_handler,
    discord_webhook_handler
)
```

---

## 🚀 Complete Example: End-to-End Analysis

```python
import pandas as pd
import numpy as np
from config.settings import settings
from src.utils.logging_config import setup_logging, get_logger

# Initialize
setup_logging(level="INFO")
logger = get_logger(__name__)

# Load data (pseudo-code - use your data provider)
df = load_data_somehow()  # OHLCV DataFrame

# ============================================
# 1. Add Technical Indicators
# ============================================
from src.indicators.technical import add_all_indicators

df = add_all_indicators(df)
logger.info(f"Added indicators: {df.columns.tolist()}")

# ============================================
# 2. GARCH Volatility Analysis
# ============================================
from src.models.garch import fit_garch_family, best_by_aic, annualize_volatility

# Calculate log returns
returns = np.log(df['Close']).diff().dropna()

# Fit all GARCH models
fits = fit_garch_family(returns)

# Get best model
best = best_by_aic(fits)
logger.info(f"Best GARCH model: {best.name}, AIC={best.aic:.2f}")

# Get annualized volatility
bars_per_year = 252 * 24 * 2  # 30-min bars
sigma_annual = annualize_volatility(best.one_step_forecast, bars_per_year)
logger.info(f"Annualized volatility: {sigma_annual:.2f}%")

# ============================================
# 3. ARIMA Price Forecasting
# ============================================
from src.models.arima import arima_forecast_prices

forecast = arima_forecast_prices(df, steps=24, order=(1,0,1))
logger.info(f"24-step forecast:\n{forecast.mean_prices}")

# ============================================
# 4. AI/ML Predictions
# ============================================
from src.models.ml_predictor import quick_ml_forecast

# Prophet forecast
ml_forecast = quick_ml_forecast(df, steps=24, method='prophet')
logger.info(f"Prophet forecast: {ml_forecast.predictions.tail()}")

# Optional: LSTM (if TensorFlow available and enough data)
if len(df) >= 500:
    try:
        lstm_forecast = quick_ml_forecast(df, steps=24, method='lstm')
        logger.info(f"LSTM forecast: {lstm_forecast.predictions.tail()}")
    except Exception as e:
        logger.warning(f"LSTM failed: {e}")

# ============================================
# 5. Smart Alert System
# ============================================
from src.triggers.alert_system import (
    AlertSystem,
    scan_all_triggers,
    console_handler,
    email_handler
)

# Setup alert system
alert_system = AlertSystem()
alert_system.add_handler(console_handler)

# Optional: Email alerts
if settings.ALERT_EMAIL and settings.SMTP_USERNAME:
    smtp_config = settings.get_smtp_config()
    alert_system.add_handler(
        lambda alert: email_handler(alert, smtp_config)
    )

# Scan for triggers
alerts = scan_all_triggers(
    df,
    sigma_series=best.conditional_volatility,
    support_levels=[df['Close'].iloc[-100:].min()],
    resistance_levels=[df['Close'].iloc[-100:].max()]
)

# Send alerts
for alert in alerts:
    alert_system.send_alert(alert)

logger.info(f"Triggered {len(alerts)} alerts")

# ============================================
# 6. Risk Analysis
# ============================================
from src.indicators.technical import atr_series

# Calculate ATR for position sizing
atr = atr_series(df['High'], df['Low'], df['Close']).iloc[-1]
logger.info(f"ATR (14): {atr:.4f}")

# Position sizing
account_size = 10000  # USD
risk_per_trade = 0.01  # 1%
atr_multiplier = 1.5

stop_distance = atr * atr_multiplier
risk_amount = account_size * risk_per_trade
position_size = risk_amount / stop_distance

logger.info(f"Suggested position size: {position_size:.4f} units")

# ============================================
# 7. Display Results
# ============================================
print("\n" + "="*60)
print("ANALYSIS SUMMARY")
print("="*60)
print(f"Symbol: {df.get('symbol', 'Unknown')}")
print(f"Last Price: ${df['Close'].iloc[-1]:.2f}")
print(f"\nVolatility (GARCH {best.name}):")
print(f"  - 1-step forecast: {best.one_step_forecast:.4f}%")
print(f"  - Annualized: {sigma_annual:.2f}%")
print(f"\nForecasts (24 steps):")
print(f"  - ARIMA mean: ${forecast.mean_prices.iloc[-1]:.2f}")
print(f"  - ML (Prophet): ${ml_forecast.predictions.iloc[-1]:.2f}")
print(f"\nAlerts: {len(alerts)} triggered")
for alert in alerts:
    print(f"  - {alert.alert_type.value}: {alert.message}")
print("="*60)
```

---

## 🎨 Streamlit Integration Example

```python
import streamlit as st
from config.settings import settings
from src.models.garch import fit_garch_family, best_by_aic
from src.models.ml_predictor import quick_ml_forecast
from src.triggers.alert_system import AlertSystem, scan_all_triggers, console_handler

st.set_page_config(page_title="GARCH Platform", layout="wide")
st.title("📈 GARCH Algo Intelligence Platform")

# Sidebar
with st.sidebar:
    st.header("Settings")
    symbol = st.selectbox("Asset", ["BTC/USDT", "ETH/USDT", "EURUSD"])
    enable_alerts = st.checkbox("Enable Alerts", value=True)

# Load data
df = load_your_data(symbol)  # Implement this

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "GARCH Volatility",
    "AI Predictions",
    "Smart Alerts",
    "Risk Management"
])

# ============================================
# Tab 1: GARCH Volatility
# ============================================
with tab1:
    st.subheader("GARCH Family Volatility Models")

    with st.spinner("Fitting GARCH models..."):
        returns = np.log(df['Close']).diff().dropna()
        fits = fit_garch_family(returns)
        best = best_by_aic(fits)

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Model", best.name)
    col2.metric("AIC", f"{best.aic:.2f}")
    col3.metric("σ (1-step)", f"{best.one_step_forecast:.4f}%")

    # Plot
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=best.conditional_volatility.index,
        y=best.conditional_volatility.values,
        name=best.name
    ))
    fig.update_layout(title="Conditional Volatility", xaxis_title="Date", yaxis_title="σ (%)")
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# Tab 2: AI Predictions
# ============================================
with tab2:
    st.subheader("AI/ML Price Predictions")

    forecast_steps = st.slider("Forecast Horizon", 8, 48, 24)
    method = st.selectbox("Method", ["prophet", "lstm", "ensemble"])

    if st.button("Generate Forecast"):
        with st.spinner(f"Running {method.upper()} forecast..."):
            try:
                ml_fc = quick_ml_forecast(df, steps=forecast_steps, method=method)

                st.success(f"Forecast generated using {ml_fc.model_type}")

                # Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df.index.tail(100),
                    y=df['Close'].tail(100),
                    name="Historical"
                ))
                fig.add_trace(go.Scatter(
                    x=ml_fc.forecast_index,
                    y=ml_fc.predictions,
                    name="Forecast",
                    line=dict(dash='dash')
                ))

                if ml_fc.lower_bound is not None:
                    fig.add_trace(go.Scatter(
                        x=ml_fc.forecast_index,
                        y=ml_fc.upper_bound,
                        fill=None,
                        mode='lines',
                        line_color='lightgray',
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=ml_fc.forecast_index,
                        y=ml_fc.lower_bound,
                        fill='tonexty',
                        mode='lines',
                        line_color='lightgray',
                        name='Confidence'
                    ))

                st.plotly_chart(fig, use_container_width=True)

                # Show predictions table
                st.dataframe(ml_fc.predictions.tail(10))

            except Exception as e:
                st.error(f"Forecast failed: {e}")

# ============================================
# Tab 3: Smart Alerts
# ============================================
with tab3:
    st.subheader("Smart Alert System")

    if enable_alerts:
        # Setup alert system
        alert_system = AlertSystem()
        alert_system.add_handler(console_handler)

        # Scan for triggers
        alerts = scan_all_triggers(df)

        if alerts:
            st.warning(f"🚨 {len(alerts)} alerts triggered!")

            for alert in alerts:
                severity_colors = {
                    "low": "🟢",
                    "medium": "🟡",
                    "high": "🟠",
                    "critical": "🔴"
                }
                icon = severity_colors.get(alert.severity.value, "⚪")

                with st.expander(f"{icon} {alert.alert_type.value.upper()} @ ${alert.price:.2f}"):
                    st.write(f"**Message**: {alert.message}")
                    st.write(f"**Severity**: {alert.severity.value.upper()}")
                    st.write(f"**Time**: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

                    if alert.metadata:
                        st.json(alert.metadata)

                # Send alert
                alert_system.send_alert(alert)
        else:
            st.info("No alerts triggered. Market conditions normal.")

        # Alert history
        if alert_system.alert_history:
            st.subheader("Alert History")
            history_df = pd.DataFrame([
                {
                    "Time": a.timestamp,
                    "Type": a.alert_type.value,
                    "Severity": a.severity.value,
                    "Price": a.price,
                    "Message": a.message
                }
                for a in alert_system.alert_history[-10:]
            ])
            st.dataframe(history_df)
    else:
        st.info("Alerts disabled. Enable in sidebar.")

# ============================================
# Tab 4: Risk Management
# ============================================
with tab4:
    st.subheader("Risk Management & Position Sizing")

    # ATR calculation
    from src.indicators.technical import atr_series

    atr = atr_series(df['High'], df['Low'], df['Close']).iloc[-1]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("ATR (14)", f"{atr:.4f}")

        account = st.number_input("Account Size ($)", 100, 1000000, 10000)
        risk_pct = st.slider("Risk per Trade (%)", 0.1, 5.0, 1.0, 0.1)
        atr_mult = st.slider("ATR Multiplier (Stop)", 0.5, 3.0, 1.5, 0.1)

    with col2:
        stop_dist = atr * atr_mult
        risk_amt = account * (risk_pct / 100)
        position = risk_amt / stop_dist

        st.metric("Stop Distance", f"{stop_dist:.4f}")
        st.metric("Risk Amount", f"${risk_amt:.2f}")
        st.metric("Position Size", f"{position:.4f} units")

    st.info(f"💡 With ${account:,.0f} account risking {risk_pct}% per trade, "
            f"trade {position:.4f} units with stop {atr_mult}x ATR = {stop_dist:.4f} away")
```

---

## 🔔 Alert Notification Setup

### **Email Alerts**

```python
from src.triggers.alert_system import AlertSystem, email_handler
from config.settings import settings

# Setup
alert_system = AlertSystem()
smtp_config = settings.get_smtp_config()

# Add email handler
alert_system.add_handler(lambda alert: email_handler(alert, smtp_config))

# Send test alert
from src.triggers.alert_system import Alert, AlertType, AlertSeverity
from datetime import datetime

test_alert = Alert(
    alert_type=AlertType.DIP_BUY,
    severity=AlertSeverity.HIGH,
    symbol="BTC/USDT",
    price=65000.00,
    message="Test alert - 5% dip detected",
    timestamp=datetime.now()
)

alert_system.send_alert(test_alert)
```

### **Discord Webhook**

```python
from src.triggers.alert_system import discord_webhook_handler

webhook_url = "https://discord.com/api/webhooks/YOUR_WEBHOOK_URL"

alert_system.add_handler(lambda alert: discord_webhook_handler(alert, webhook_url))
```

---

## 🧪 Testing Your Integration

```python
# test_integration.py
import pandas as pd
import numpy as np

def test_full_pipeline():
    """Test complete analysis pipeline."""

    # Create sample data
    dates = pd.date_range('2024-01-01', periods=500, freq='30min')
    prices = 50000 + np.cumsum(np.random.randn(500) * 100)

    df = pd.DataFrame({
        'Open': prices + np.random.randn(500) * 50,
        'High': prices + np.abs(np.random.randn(500) * 100),
        'Low': prices - np.abs(np.random.randn(500) * 100),
        'Close': prices,
        'Volume': np.random.randint(100, 1000, 500)
    }, index=dates)

    # Add indicators
    from src.indicators.technical import add_all_indicators
    df = add_all_indicators(df)

    assert 'MACD' in df.columns
    assert 'RSI' in df.columns
    print("✓ Indicators added")

    # GARCH
    from src.models.garch import fit_garch_family, best_by_aic
    returns = np.log(df['Close']).diff().dropna()
    fits = fit_garch_family(returns, min_obs=50)
    best = best_by_aic(fits)

    assert best is not None
    assert best.aic > 0
    print(f"✓ GARCH fitted: {best.name}, AIC={best.aic:.2f}")

    # ARIMA
    from src.models.arima import arima_forecast_prices
    forecast = arima_forecast_prices(df, steps=10, min_obs=50)

    assert len(forecast.mean_prices) == 10
    print("✓ ARIMA forecast generated")

    # Alerts
    from src.triggers.alert_system import scan_all_triggers
    alerts = scan_all_triggers(df, sigma_series=best.conditional_volatility)

    print(f"✓ Alert scan complete: {len(alerts)} alerts")

    print("\n🎉 All tests passed!")

if __name__ == "__main__":
    test_full_pipeline()
```

Run with:
```bash
python test_integration.py
```

---

## 📝 Common Patterns

### **Pattern 1: Continuous Monitoring Loop**

```python
import time
from datetime import datetime

while True:
    print(f"\n[{datetime.now()}] Running analysis...")

    # Fetch latest data
    df = fetch_latest_data()

    # Run GARCH
    returns = np.log(df['Close']).diff().dropna()
    fits = fit_garch_family(returns)
    best = best_by_aic(fits)

    # Scan for alerts
    alerts = scan_all_triggers(df, sigma_series=best.conditional_volatility)

    # Send alerts
    for alert in alerts:
        alert_system.send_alert(alert)

    # Wait
    time.sleep(120)  # 2 minutes
```

### **Pattern 2: Batch Processing Multiple Symbols**

```python
symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

results = {}
for symbol in symbols:
    df = load_data(symbol)

    # GARCH
    returns = np.log(df['Close']).diff().dropna()
    fits = fit_garch_family(returns)
    best = best_by_aic(fits)

    # Store results
    results[symbol] = {
        "model": best.name,
        "aic": best.aic,
        "volatility": best.one_step_forecast
    }

# Compare
import pandas as pd
comparison = pd.DataFrame(results).T
print(comparison.sort_values('volatility', ascending=False))
```

---

<div align="center">

**You're now ready to integrate all modules!**

Next: Refactor `main.py` to use these imports

</div>
