# 📊 Portfolio Analytics Dashboard

## Bloomberg Terminal-Style Analytics for Professional Investors

The Portfolio Analytics module provides institutional-grade performance metrics and risk analytics, designed to look and feel like a professional Bloomberg Terminal.

---

## 🎯 Features Overview

### 1. **Performance Metrics Dashboard**

Eight key metrics displayed in Bloomberg-style cards:

#### Primary Metrics (Row 1):
- **Total Return**: Cumulative return over the entire period
- **Sharpe Ratio**: Risk-adjusted return (return per unit of total risk)
- **Max Drawdown**: Largest peak-to-trough decline
- **Annual Volatility**: Annualized standard deviation of returns

#### Secondary Metrics (Row 2):
- **Win Rate**: Percentage of positive return periods
- **Profit Factor**: Ratio of total wins to total losses
- **Calmar Ratio**: Annual return divided by max drawdown
- **Number of Trades**: Total number of return periods analyzed

---

### 2. **Monthly Returns Heatmap** 📅

Bloomberg-style calendar heatmap showing monthly returns:

- **Color Scheme**: Red-Yellow-Green gradient
  - 🟢 Green: Positive returns
  - 🟡 Yellow: Near-zero returns
  - 🔴 Red: Negative returns
- **Annual Column**: Shows total annual return
- **Interactive**: Hover to see exact monthly return percentages

**Use Case**: Quickly identify seasonal patterns, strong/weak months, and year-over-year performance.

---

### 3. **Drawdown Analysis** 📉

Underwater equity chart showing portfolio drawdowns:

- **Visualization**: Red area chart showing distance from peak
- **Current Drawdown**: How far the portfolio is from its all-time high
- **Max Drawdown**: Largest historical decline
- **Max DD Duration**: Longest time spent in drawdown

**Use Case**: Understand risk exposure and recovery patterns. Critical for risk management.

---

### 4. **Rolling Sharpe Ratio** 📊

252-period (1-year) rolling Sharpe ratio:

- **Reference Lines**:
  - Yellow dashed line at 1.0 (Good performance)
  - Green dashed line at 2.0 (Excellent performance)
- **Current Rolling Sharpe**: Latest value displayed as metric

**Use Case**: Track consistency of risk-adjusted returns over time. Identify periods of strong vs. weak performance.

---

### 5. **Returns Distribution** 📊

Histogram showing the distribution of daily returns:

- **Color Gradient**: Returns colored by magnitude (red-yellow-green)
- **Mean Line**: Cyan dashed line showing average return
- **Statistics**:
  - Mean Return
  - Median Return
  - Standard Deviation
  - Skewness (asymmetry of distribution)

**Use Case**: Understand return characteristics. Negative skew indicates more frequent small gains with occasional large losses.

---

### 6. **Cross-Asset Correlation Matrix** 🔗

Heatmap showing correlations between multiple assets:

- **Assets Included**:
  - Crypto: BTC/USDT, ETH/USDT, SOL/USDT, XRP/USDT, DOGE/USDT
  - Traditional: Gold (XAUUSD), EUR/USD
- **Color Scheme**: Red-Blue gradient
  - 🔴 Red: Negative correlation (-1.0)
  - ⚪ White: No correlation (0.0)
  - 🔵 Blue: Positive correlation (+1.0)

**Use Case**: Portfolio diversification. Look for low/negative correlations to reduce overall portfolio risk.

---

### 7. **Detailed Metrics Table** 📋

Comprehensive table with all calculated metrics:

| Metric | Description |
|--------|-------------|
| Total Return | Cumulative return from start to end |
| Annual Return | Annualized return (CAGR) |
| Annual Volatility | Annualized standard deviation |
| Sharpe Ratio | (Return - Risk-Free Rate) / Volatility |
| Sortino Ratio | Like Sharpe, but only penalizes downside volatility |
| Calmar Ratio | Annual Return / Max Drawdown |
| Max Drawdown | Largest peak-to-trough decline |
| Max DD Duration | Longest drawdown period (in bars) |
| Win Rate | % of positive return periods |
| Profit Factor | Sum of wins / Sum of losses |
| Number of Trades | Total return periods |
| Average Win | Mean of positive returns |
| Average Loss | Mean of negative returns |

---

### 8. **Export Functionality** 💾

Two export options:

1. **Download Metrics (CSV)**: All performance metrics in tabular format
2. **Download Returns (CSV)**: Complete returns time series

**File Naming**: Automatically includes current date (e.g., `portfolio_metrics_20251208.csv`)

---

## 📐 Metric Calculations

### Sharpe Ratio
```
Sharpe = (Mean Return - Risk-Free Rate) / Std Dev of Returns
```
- Annualized using √252 (trading days per year)
- Risk-free rate default: 2% annual

### Sortino Ratio
```
Sortino = (Mean Return - Risk-Free Rate) / Downside Deviation
```
- Only penalizes negative returns (downside risk)
- Better for asymmetric return distributions

### Calmar Ratio
```
Calmar = Annual Return / |Max Drawdown|
```
- Measures return per unit of maximum loss
- Higher is better (more return for less drawdown)

### Max Drawdown
```
Drawdown(t) = (Price(t) - Peak(t)) / Peak(t)
Max DD = min(Drawdown(t))
```
- Always negative or zero
- Expressed as percentage

### Win Rate
```
Win Rate = (Number of Positive Returns / Total Returns) × 100
```

### Profit Factor
```
Profit Factor = Sum(Positive Returns) / |Sum(Negative Returns)|
```
- PF > 1.0: Profitable overall
- PF < 1.0: Losing overall

---

## 🎨 Design Philosophy

The Portfolio Analytics dashboard follows Bloomberg Terminal design principles:

1. **Dark Theme**: Professional dark background with high-contrast text
2. **Information Density**: Maximum information in minimal space
3. **Color Coding**: Intuitive red/green for losses/gains
4. **Interactive Charts**: Hover tooltips for detailed information
5. **Metric Cards**: Key metrics prominently displayed at the top
6. **Hierarchical Layout**: Most important info first, details below

---

## 🚀 Usage

### Accessing the Dashboard

1. Run the Streamlit app: `streamlit run main.py`
2. Navigate to the **"📊 Portfolio Analytics"** tab
3. The dashboard will automatically calculate all metrics from the loaded data

### Interpreting Results

#### Good Performance Indicators:
- ✅ Sharpe Ratio > 1.0 (excellent if > 2.0)
- ✅ Sortino Ratio > Sharpe Ratio (limited downside risk)
- ✅ Max Drawdown < 20%
- ✅ Win Rate > 50%
- ✅ Profit Factor > 1.5
- ✅ Positive skewness in returns distribution

#### Warning Signs:
- ⚠️ Sharpe Ratio < 0.5
- ⚠️ Max Drawdown > 30%
- ⚠️ Long drawdown durations (> 100 periods)
- ⚠️ Negative skewness (tail risk)
- ⚠️ Profit Factor < 1.0

---

## 🔧 Technical Implementation

### Module Structure

```
src/analytics/
├── __init__.py
├── portfolio_metrics.py    # Metric calculations
└── heatmaps.py            # Plotly visualizations
```

### Key Functions

**portfolio_metrics.py**:
- `calculate_returns()`: Convert prices to returns
- `sharpe_ratio()`: Calculate Sharpe ratio
- `sortino_ratio()`: Calculate Sortino ratio
- `calculate_drawdown()`: Compute drawdown series
- `max_drawdown()`: Find maximum drawdown
- `monthly_returns_table()`: Prepare data for heatmap
- `calculate_all_metrics()`: Comprehensive metrics calculation

**heatmaps.py**:
- `create_monthly_returns_heatmap()`: Bloomberg-style calendar
- `create_correlation_heatmap()`: Asset correlation matrix
- `create_drawdown_chart()`: Underwater equity visualization
- `create_rolling_metrics_chart()`: Rolling Sharpe ratio
- `create_returns_distribution()`: Returns histogram

---

## 📊 Example Interpretation

### Sample Output:

```
Total Return: +45.23%
Annual Return: +18.50%
Sharpe Ratio: 1.85
Max Drawdown: -12.34%
Win Rate: 58.3%
Profit Factor: 1.92
```

**Interpretation**:
- **Strong Performance**: 18.5% annual return is excellent
- **Good Risk-Adjusted Returns**: Sharpe of 1.85 is very good
- **Controlled Risk**: Max drawdown of 12.34% is acceptable
- **Consistent**: 58.3% win rate shows reliability
- **Profitable**: Profit factor of 1.92 means wins are nearly 2x losses

---

## 🎯 Use Cases

### For Investors:
1. **Performance Review**: Monthly heatmap shows seasonal patterns
2. **Risk Assessment**: Drawdown chart reveals maximum pain points
3. **Diversification**: Correlation matrix guides asset allocation
4. **Due Diligence**: Comprehensive metrics for investment decisions

### For Traders:
1. **Strategy Validation**: Sharpe/Sortino ratios measure effectiveness
2. **Risk Management**: Max drawdown sets stop-loss levels
3. **Consistency Check**: Rolling Sharpe shows stability over time
4. **Return Analysis**: Distribution reveals profit/loss patterns

### For Portfolio Managers:
1. **Client Reporting**: Professional-grade analytics for presentations
2. **Benchmark Comparison**: Compare against indices or peers
3. **Risk Monitoring**: Track drawdowns and volatility in real-time
4. **Performance Attribution**: Identify strong/weak periods

---

## 🔮 Future Enhancements

Potential additions for future versions:

1. **Benchmark Comparison**: Alpha, Beta, Information Ratio vs. S&P 500
2. **Multi-Portfolio View**: Compare multiple strategies side-by-side
3. **Risk Metrics**: VaR, CVaR, Expected Shortfall
4. **PDF Reports**: Automated tear sheets for client distribution
5. **Custom Date Ranges**: Filter analytics by specific time periods
6. **Monte Carlo Simulation**: Project future returns and drawdowns
7. **Factor Analysis**: Fama-French factor exposures
8. **Transaction Costs**: Include slippage and fees in calculations

---

## 📚 References

### Academic Papers:
- Sharpe, W. F. (1966). "Mutual Fund Performance"
- Sortino, F. A., & Price, L. N. (1994). "Performance Measurement in a Downside Risk Framework"

### Industry Standards:
- CFA Institute: Global Investment Performance Standards (GIPS)
- Bloomberg Terminal: Professional analytics interface

### Libraries Used:
- **Pandas**: Time series calculations
- **NumPy**: Numerical computations
- **Plotly**: Interactive visualizations
- **Streamlit**: Web application framework

---

## 💡 Tips & Best Practices

1. **Data Quality**: Ensure clean, complete price data for accurate metrics
2. **Time Period**: Use at least 1 year of data for meaningful statistics
3. **Frequency**: Daily or hourly data works best; avoid tick data
4. **Risk-Free Rate**: Adjust based on your currency and time period
5. **Benchmark**: Always compare to a relevant benchmark (e.g., BTC for crypto)
6. **Context**: Consider market conditions when interpreting metrics
7. **Multiple Metrics**: Don't rely on a single metric; use comprehensive analysis

---

## 🐛 Troubleshooting

### Common Issues:

**"Not enough data for monthly heatmap"**
- Need at least 1 full month of data
- Solution: Load more historical data

**"Rolling Sharpe unavailable"**
- Need at least 126 periods for 252-period rolling window
- Solution: Use shorter window or load more data

**"Correlation matrix unavailable"**
- Requires multiple assets
- Solution: This is expected for single-asset analysis

**Metrics show as 0.0 or NaN**
- Insufficient data or constant prices
- Solution: Check data quality and ensure price variation

---

## 📞 Support

For questions or issues:
1. Check the main `README.md` for general setup
2. Review `ARCHITECTURE.md` for system design
3. See `INTEGRATION_GUIDE.md` for usage examples
4. Check `TEST_RESULTS.md` for validation

---

**Built with ❤️ for professional investors and traders**

*Last Updated: December 8, 2025*
