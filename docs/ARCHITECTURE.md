# 🏗️ System Architecture

## Overview

The GARCH Algo Intelligence Platform follows a **modular, layered architecture** designed for scalability, maintainability, and extensibility.

---

## 🎯 Design Principles

1. **Separation of Concerns**: Each module has a single responsibility
2. **Dependency Injection**: Configuration via environment variables
3. **Error Resilience**: Comprehensive exception handling at every layer
4. **Testability**: Pure functions with minimal side effects
5. **Extensibility**: Plugin architecture for models, indicators, alerts

---

## 📊 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                           │
│                    (Streamlit Web App)                           │
│  ┌────────┬────────┬────────┬────────┬────────┐                 │
│  │ GARCH  │ ARIMA  │Signals │Options │ Market │                 │
│  │  Tab   │  Tab   │  Tab   │  Tab   │  Tab   │                 │
│  └────────┴────────┴────────┴────────┴────────┘                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                     APPLICATION LAYER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Models     │  │  Indicators  │  │   Triggers   │          │
│  │              │  │              │  │              │          │
│  │ • GARCH      │  │ • MACD       │  │ • Dip Buy    │          │
│  │ • ARIMA      │  │ • RSI        │  │ • Breakout   │          │
│  │ • LSTM       │  │ • ADX        │  │ • Vol Spike  │          │
│  │ • Prophet    │  │ • ATR        │  │ • MACD Cross │          │
│  │              │  │ • Bollinger  │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Options    │  │ Backtesting  │  │Visualization │          │
│  │              │  │              │  │              │          │
│  │ • BS Pricing │  │ • P&L Engine │  │ • Charts     │          │
│  │ • Greeks     │  │ • Risk Calc  │  │ • Reports    │          │
│  │ • Impl. Vol  │  │ • Kelly      │  │ • PDF Export │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                     DATA ACCESS LAYER                            │
│  ┌────────────┬────────────┬────────────┬────────────┐          │
│  │   CCXT     │   Yahoo    │  Twelve    │   Alpha    │          │
│  │ (Crypto)   │ (Stocks/FX)│   Data     │  Vantage   │          │
│  │            │            │            │            │          │
│  │ • OKX      │ • Free     │ • Premium  │ • FX Only  │          │
│  │ • Binance  │ • Global   │ • Global   │ • Free     │          │
│  │ • 100+ Ex  │ • Reliable │ • API Key  │ • API Key  │          │
│  └────────────┴────────────┴────────────┴────────────┘          │
│                                                                  │
│  ┌──────────────────────────────────────────────────┐          │
│  │          Data Validation & Caching               │          │
│  │  • Schema validation                             │          │
│  │  • OHLCV integrity checks                        │          │
│  │  • Streamlit @cache_data (TTL: 120s)            │          │
│  │  • Fallback mechanism                            │          │
│  └──────────────────────────────────────────────────┘          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                  INFRASTRUCTURE LAYER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Config     │  │   Logging    │  │    Alerts    │          │
│  │              │  │              │  │              │          │
│  │ • .env       │  │ • File logs  │  │ • Email      │          │
│  │ • Settings   │  │ • Console    │  │ • Discord    │          │
│  │ • Secrets    │  │ • Rotating   │  │ • SMS        │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow

### **1. Market Data Ingestion**

```
User Selection
    ↓
Data Provider Selection (CCXT/Yahoo/TwelveData/AlphaVantage)
    ↓
API Call with Retry Logic
    ↓
Data Validation (OHLCV schema, integrity checks)
    ↓
Cache (120s TTL)
    ↓
Technical Indicators Added (MACD, ADX, ATR, etc.)
    ↓
DataFrame Ready for Analysis
```

### **2. GARCH Volatility Modeling**

```
OHLCV DataFrame
    ↓
Calculate Log Returns
    ↓
Validate Returns (min observations, no NaN, non-constant)
    ↓
Fit 4 GARCH Models in Parallel:
    • GARCH(1,1)
    • EGARCH
    • GJR
    • APARCH
    ↓
Check Convergence & Extract:
    • Conditional Volatility σ(t)
    • 1-step Forecast
    • AIC/BIC
    ↓
Select Best Model (min AIC)
    ↓
Annualize Volatility (bars_per_year * sqrt(σ))
    ↓
Return GarchFit Objects
```

### **3. AI/ML Prediction Pipeline**

```
Price Series
    ↓
Method Selection (LSTM / Prophet / Ensemble)
    ↓
─────────────────────────────────────────────────
│ LSTM Branch                │ Prophet Branch   │
├────────────────────────────┼──────────────────┤
│ • Scale Data (MinMax)      │ • Format ds/y    │
│ • Create Sequences         │ • Fit model      │
│ • Train NN (early stop)    │ • Forecast       │
│ • Iterative Forecast       │ • Extract bounds │
└────────────────────────────┴──────────────────┘
    ↓
Ensemble (weighted average if both available)
    ↓
Return MLForecast with Confidence Intervals
```

### **4. Alert Trigger System**

```
New Market Data Arrives
    ↓
Run Trigger Scanners in Parallel:
    • detect_dip_buy()
    • detect_breakout()
    • detect_volatility_spike()
    • detect_macd_crossover()
    • detect_support_break()
    • detect_mean_reversion()
    ↓
Alerts Generated → Alert Objects
    ↓
AlertSystem.send_alert()
    ↓
Dispatch to Handlers:
    • Email (SMTP)
    • Discord (Webhook)
    • Console (Debug)
    • SMS (Twilio - optional)
    ↓
Store in alert_history[]
```

---

## 🧩 Module Breakdown

### **Core Models**

#### **`src/models/garch.py`**
- **Purpose**: Volatility modeling
- **Key Classes**: `GarchFit`, `GARCHModelError`
- **Key Functions**: `fit_garch_family()`, `best_by_aic()`, `annualize_volatility()`
- **Dependencies**: arch, numpy, pandas
- **Error Handling**: Custom exceptions for convergence, insufficient data

#### **`src/models/arima.py`**
- **Purpose**: Time series forecasting
- **Key Classes**: `ARIMAForecast`
- **Key Functions**: `arima_forecast_prices()`, `auto_select_order()`
- **Dependencies**: statsmodels
- **Features**: Auto-ARIMA, confidence intervals, returns→prices conversion

#### **`src/models/ml_predictor.py`**
- **Purpose**: AI/ML predictions
- **Key Classes**: `LSTMPredictor`, `ProphetPredictor`, `MLForecast`
- **Key Functions**: `quick_ml_forecast()`, `ensemble_forecast()`
- **Dependencies**: tensorflow, prophet, sklearn
- **Features**: LSTM training, Prophet seasonality, ensemble combining

### **Data Layer**

#### **`src/data/providers.py`**
- **Purpose**: Multi-source data acquisition
- **Providers**: CCXT, Yahoo Finance, Twelve Data, Alpha Vantage
- **Features**: Automatic fallbacks, retry logic, rate limiting
- **Caching**: Streamlit `@cache_data` with 120s TTL

### **Indicators**

#### **`src/indicators/technical.py`**
- **Purpose**: Technical analysis
- **Indicators**: MACD, RSI, ADX, ATR, Bollinger Bands, Stochastic, CCI, Williams %R, OBV, VWAP
- **Function**: `add_all_indicators()` adds all at once

### **Risk & Trading**

#### **`src/triggers/alert_system.py`**
- **Purpose**: Smart alerts and triggers
- **Alert Types**: Dip buy, breakout, volatility spike, MACD cross, support/resistance break
- **Notification**: Email, Discord, Console
- **Severity Levels**: LOW, MEDIUM, HIGH, CRITICAL

#### **`src/options/black_scholes.py`**
- **Purpose**: Options pricing
- **Features**: Call/Put pricing, Greeks (Δ, Γ, ν, Θ, ρ), implied volatility solver

#### **`src/backtesting/engine.py`**
- **Purpose**: Strategy testing
- **Features**: P&L tracking, win rate, risk metrics, position sizing

### **Utilities**

#### **`config/settings.py`**
- **Purpose**: Configuration management
- **Source**: Environment variables (.env)
- **Validation**: Settings.validate() checks critical config

#### **`src/utils/logging_config.py`**
- **Purpose**: Logging infrastructure
- **Features**: Colored console output, file rotation, module-level loggers

---

## 🔐 Security Architecture

### **API Key Management**
```
.env (git-ignored)
    ↓
os.getenv()
    ↓
config/settings.py
    ↓
Application Code (never hardcoded)
```

### **Data Validation**
- Schema validation (pydantic/cerberus)
- OHLCV integrity checks (High ≥ Low, etc.)
- Type checking
- Range validation

### **Error Handling Strategy**
```
Try-Except at Every Layer
    ↓
Custom Exceptions (GARCHModelError, ARIMAModelError, etc.)
    ↓
Logging (ERROR level)
    ↓
Graceful Degradation (fallback providers, default values)
    ↓
User Notification (Streamlit warnings/errors)
```

---

## ⚡ Performance Optimizations

1. **Caching**
   - Streamlit `@cache_data` for expensive computations
   - TTL-based invalidation (120s default)
   - Market data cached per symbol/timeframe

2. **Lazy Loading**
   - Indicators computed only when needed
   - ML models trained on-demand

3. **Parallel Processing**
   - GARCH models fitted concurrently
   - Multiple data providers queried in parallel (fallback chain)

4. **Efficient Data Structures**
   - NumPy vectorization
   - Pandas optimized operations
   - Avoid loops where possible

---

## 🧪 Testing Strategy

### **Unit Tests**
- Each module has corresponding test file
- Mock external API calls
- Test edge cases (empty data, convergence failure, etc.)

### **Integration Tests**
- End-to-end data flow
- Multi-provider fallback
- Alert system end-to-end

### **Performance Tests**
- GARCH fitting speed
- LSTM training time
- API response times

---

## 🚀 Deployment Architecture

### **Development**
```
Local Machine
    ↓
Streamlit Dev Server (port 8501)
    ↓
Local .env file
```

### **Production (Streamlit Cloud)**
```
GitHub Repository
    ↓
Streamlit Cloud (auto-deploy on push)
    ↓
Secrets Management (Streamlit Secrets)
    ↓
HTTPS (automatic)
```

### **Production (Docker)**
```
Dockerfile
    ↓
Docker Image
    ↓
Container Orchestration (Kubernetes/ECS)
    ↓
Load Balancer
    ↓
Auto-scaling
```

---

## 📊 Scalability Considerations

### **Current Capacity**
- Handles 1-10 concurrent users
- 2000 bars (60+ days of 30m data)
- Real-time updates via auto-refresh

### **Scaling to 100+ Users**
1. **Backend API**: FastAPI/Flask REST API
2. **Database**: PostgreSQL for historical data, Redis for caching
3. **Message Queue**: Celery for async tasks
4. **WebSockets**: Real-time data streaming
5. **CDN**: Static asset delivery

### **Scaling to 10,000+ Users**
1. **Microservices**: Separate services for GARCH, ML, Alerts
2. **Kubernetes**: Container orchestration
3. **Load Balancing**: NGINX/AWS ALB
4. **Distributed Caching**: Redis Cluster
5. **Time-Series DB**: InfluxDB/TimescaleDB

---

## 🔮 Future Enhancements

### **Short-Term (1-3 months)**
- WebSocket real-time data feeds
- PDF report generation (ReportLab)
- Enhanced backtesting with slippage/commissions
- Model persistence (save/load trained models)

### **Medium-Term (3-6 months)**
- Portfolio optimization (MPT, Black-Litterman)
- Multi-asset correlation trading
- Automated order execution (paper trading)
- Mobile app (React Native)

### **Long-Term (6-12 months)**
- Transformer models for price prediction
- Reinforcement learning for strategy optimization
- Social trading features
- Cloud-native serverless architecture

---

## 📚 Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | Streamlit, Plotly, TradingView Widget |
| **Backend** | Python 3.10+, FastAPI (future) |
| **ML/AI** | TensorFlow/Keras, Prophet, scikit-learn |
| **Econometrics** | arch (GARCH), statsmodels (ARIMA) |
| **Data** | ccxt, yfinance, pandas, numpy |
| **Alerts** | smtplib, Discord webhooks, Twilio |
| **Testing** | pytest, pytest-cov |
| **Deployment** | Docker, Streamlit Cloud, AWS |
| **Monitoring** | Loguru, Sentry |

---

## 🤝 Contributing Guidelines

When adding new features:

1. **Follow Module Pattern**: Create new module in appropriate `src/` subdirectory
2. **Add Tests**: Minimum 80% code coverage
3. **Document**: Comprehensive docstrings (Google style)
4. **Type Hints**: Use type annotations
5. **Error Handling**: Custom exceptions with logging
6. **Config**: All settings via environment variables

---

<div align="center">

**Architecture designed for resilience, performance, and growth**

</div>
