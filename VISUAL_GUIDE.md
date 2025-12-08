# 📊 Portfolio Analytics Dashboard - Visual Guide

## What You'll See in the Dashboard

### 🎯 Tab Location
Navigate to: **📊 Portfolio Analytics** (6th tab in the main interface)

---

## 📈 Section 1: Performance Metrics Cards

### Row 1 (Primary Metrics):
```
┌──────────────────┬──────────────────┬──────────────────┬──────────────────┐
│  Total Return    │  Sharpe Ratio    │  Max Drawdown    │  Volatility      │
│  +45.23%         │  1.85            │  -12.34%         │  35.67%          │
│  Annual: 18.50%  │  Risk-adjusted   │  120 periods     │  Sortino: 2.34   │
└──────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

### Row 2 (Secondary Metrics):
```
┌──────────────────┬──────────────────┬──────────────────┬──────────────────┐
│  Win Rate        │  Profit Factor   │  Calmar Ratio    │  Num Trades      │
│  58.3%           │  1.92            │  1.50            │  2,000           │
└──────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

**Color Coding**:
- 🟢 Green delta: Positive/good metrics
- 🔴 Red delta: Negative/warning metrics
- ⚪ Gray delta: Informational only

---

## 📅 Section 2: Monthly Returns Heatmap

### Visual Example:
```
BTC/USDT - Monthly Returns (%)

Year │ Jan  │ Feb  │ Mar  │ Apr  │ May  │ Jun  │ Jul  │ Aug  │ Sep  │ Oct  │ Nov  │ Dec  │ Annual
─────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼────────
2024 │ +5.2 │ +8.1 │ -2.3 │ +12.4│ -5.6 │ +3.8 │ +15.2│ -8.9 │ +6.7 │ +9.3 │ +4.5 │ +7.8 │ +45.2
2023 │ +3.4 │ -4.2 │ +11.5│ +6.8 │ +2.1 │ -7.3 │ +9.4 │ +5.6 │ -3.8 │ +8.2 │ +12.1│ +4.9 │ +38.7
```

**Color Legend**:
- 🟢 **Green**: Positive returns (darker = higher %)
- 🟡 **Yellow**: Near-zero returns
- 🔴 **Red**: Negative returns (darker = larger loss)

**Interactive Features**:
- Hover over any cell to see exact percentage
- Annual column shows yearly total
- Cells are clickable for drill-down (future feature)

---

## 📉 Section 3: Drawdown Analysis

### Underwater Equity Chart:
```
  0% ┤──────────────────────────────────────────────────────────
     │                    ╱╲
 -5% ┤                   ╱  ╲
     │                  ╱    ╲
-10% ┤                 ╱      ╲        ╱╲
     │                ╱        ╲      ╱  ╲
-15% ┤               ╱          ╲    ╱    ╲
     │              ╱            ╲  ╱      ╲
-20% ┤─────────────╱──────────────╲╱────────╲──────────────────
     └────────────────────────────────────────────────────────▶
                              Time
```

**Red shaded area** = Distance from peak (underwater equity)

**Metrics Below Chart**:
```
┌──────────────────────┬──────────────────────┬──────────────────────┐
│  Current Drawdown    │  Max Drawdown        │  Max DD Duration     │
│  -5.23%              │  -12.34%             │  120 periods         │
└──────────────────────┴──────────────────────┴──────────────────────┘
```

---

## 📊 Section 4: Two-Column Charts

### Left Column: Rolling Sharpe Ratio
```
 3.0 ┤                                    ╱╲
     │                                   ╱  ╲
 2.0 ┤─────────────────────────────────────────── Excellent (2.0)
     │              ╱╲                  ╱    ╲
 1.0 ┤─────────────╱──╲────────────────────────── Good (1.0)
     │            ╱    ╲    ╱╲        ╱      ╲
 0.0 ┤───────────╱──────╲──╱──╲──────╱────────╲─────────────
     └────────────────────────────────────────────────────────▶
                         Time
```

**Reference Lines**:
- Yellow dashed at 1.0 (Good)
- Green dashed at 2.0 (Excellent)

**Current Metric**: Shows latest rolling Sharpe value

---

### Right Column: Returns Distribution
```
Frequency
    │
200 ┤        ╭───╮
    │        │   │
150 ┤      ╭─┤   ├─╮
    │      │ │   │ │
100 ┤    ╭─┤ │   │ ├─╮
    │    │ │ │   │ │ │
 50 ┤  ╭─┤ │ │   │ │ ├─╮
    │  │ │ │ │   │ │ │ │
  0 ┤──┴─┴─┴─┴───┴─┴─┴─┴──
    └──────────────────────▶
      -10  -5   0   5  10
         Return (%)
```

**Cyan dashed line** = Mean return

**Statistics Below**:
```
┌──────────────┬──────────────┐
│ Mean Return  │ Median Return│
│ +0.123%      │ +0.098%      │
├──────────────┼──────────────┤
│ Std Dev      │ Skewness     │
│ 2.345%       │ -0.15        │
└──────────────┴──────────────┘
```

---

## 🔗 Section 5: Correlation Matrix

### Multi-Asset Heatmap:
```
           BTC/USDT  ETH/USDT  SOL/USDT  XRP/USDT  DOGE/USDT  XAUUSD  EURUSD
BTC/USDT      1.00      0.85      0.72      0.65       0.58    0.12   -0.05
ETH/USDT      0.85      1.00      0.78      0.71       0.63    0.08   -0.03
SOL/USDT      0.72      0.78      1.00      0.68       0.61    0.05   -0.02
XRP/USDT      0.65      0.71      0.68      1.00       0.74    0.03   -0.01
DOGE/USDT     0.58      0.63      0.61      0.74       1.00    0.02    0.00
XAUUSD        0.12      0.08      0.05      0.03       0.02    1.00    0.45
EURUSD       -0.05     -0.03     -0.02     -0.01       0.00    0.45    1.00
```

**Color Scheme**:
- 🔵 **Blue**: Positive correlation (1.0 = perfect)
- ⚪ **White**: No correlation (0.0)
- 🔴 **Red**: Negative correlation (-1.0 = perfect inverse)

**Interpretation**:
- High correlation (>0.7): Assets move together
- Low correlation (<0.3): Good for diversification
- Negative correlation (<0): Hedge potential

---

## 📋 Section 6: Detailed Metrics Table

### Full Metrics Table:
```
┌─────────────────────┬──────────────┐
│ Metric              │ Value        │
├─────────────────────┼──────────────┤
│ Total Return        │ +45.23%      │
│ Annual Return       │ +18.50%      │
│ Annual Volatility   │ 35.67%       │
│ Sharpe Ratio        │ 1.85         │
│ Sortino Ratio       │ 2.34         │
│ Calmar Ratio        │ 1.50         │
│ Max Drawdown        │ -12.34%      │
│ Max DD Duration     │ 120 periods  │
│ Win Rate            │ 58.3%        │
│ Profit Factor       │ 1.92         │
│ Number of Trades    │ 2,000        │
│ Average Win         │ +1.234%      │
│ Average Loss        │ -0.987%      │
└─────────────────────┴──────────────┘
```

**Scrollable**: Can view all 13 metrics
**Sortable**: Click column headers to sort (future feature)

---

## 💾 Section 7: Export Functionality

### Download Buttons:
```
┌─────────────────────────────────┬─────────────────────────────────┐
│  📥 Download Metrics (CSV)      │  📥 Download Returns (CSV)      │
│  portfolio_metrics_20251208.csv │  returns_20251208.csv           │
└─────────────────────────────────┴─────────────────────────────────┘
```

**Metrics CSV Contains**:
- All 13 performance metrics
- Formatted as table (Metric, Value)
- Ready for Excel/Google Sheets

**Returns CSV Contains**:
- Date column
- Return column (%)
- Complete time series
- Ready for further analysis

---

## 🎨 Design Highlights

### Color Palette:
- **Background**: Dark (#0e1117) - Professional Bloomberg look
- **Text**: Light gray (#dddddd) - High contrast
- **Positive**: Green (#65C466) - Gains/good metrics
- **Negative**: Red (#E57373) - Losses/warnings
- **Neutral**: Cyan (#3BA7FF) - Information
- **Heatmap**: RdYlGn (Red-Yellow-Green) - Intuitive

### Typography:
- **Headers**: Bold, larger font
- **Metrics**: Clear, readable numbers
- **Labels**: Descriptive, concise

### Layout:
- **Top**: Most important metrics (cards)
- **Middle**: Key visualizations (heatmap, drawdown)
- **Bottom**: Detailed analysis (table, exports)

---

## 🖱️ Interactive Features

### Hover Tooltips:
- **Heatmap cells**: Exact monthly return percentage
- **Charts**: Date, value, metric name
- **Correlation matrix**: Asset pair, correlation value

### Responsive Design:
- **Desktop**: Full-width charts
- **Tablet**: Stacked layout
- **Mobile**: Vertical scrolling (future optimization)

---

## 📊 Data Flow

```
Price Data (df["Close"])
    ↓
calculate_returns()
    ↓
calculate_all_metrics()
    ↓
┌─────────────────────────────────────────┐
│  PerformanceMetrics (dataclass)         │
│  - total_return                         │
│  - sharpe_ratio                         │
│  - max_drawdown                         │
│  - ... (13 metrics total)               │
└─────────────────────────────────────────┘
    ↓
Display in Streamlit
    ↓
┌──────────────┬──────────────┬──────────────┐
│  Metric Cards│  Heatmaps    │  Charts      │
└──────────────┴──────────────┴──────────────┘
```

---

## 🎯 Quick Interpretation Guide

### Good Performance:
✅ Sharpe Ratio > 1.0 (excellent if > 2.0)
✅ Max Drawdown < 20%
✅ Win Rate > 50%
✅ Profit Factor > 1.5
✅ Positive annual return
✅ Low correlation with other assets (diversification)

### Warning Signs:
⚠️ Sharpe Ratio < 0.5
⚠️ Max Drawdown > 30%
⚠️ Win Rate < 45%
⚠️ Profit Factor < 1.0
⚠️ Negative skewness (tail risk)
⚠️ Long drawdown durations

---

## 🚀 Usage Tips

1. **First Time**:
   - Load data (select asset and provider)
   - Navigate to Portfolio Analytics tab
   - Wait for calculations (usually < 5 seconds)

2. **Regular Use**:
   - Check monthly heatmap for patterns
   - Monitor current drawdown
   - Track rolling Sharpe for consistency
   - Export metrics for reporting

3. **Client Presentations**:
   - Start with performance metrics cards
   - Show monthly heatmap (visual impact)
   - Explain drawdown chart (risk)
   - Highlight Sharpe ratio (risk-adjusted returns)
   - Export to CSV for detailed analysis

---

## 🎬 Demo Flow (Recommended)

### For Client Presentation:

1. **Open App** → Navigate to Portfolio Analytics
2. **Show Metrics Cards** → "Here's our performance summary"
3. **Monthly Heatmap** → "Green months are profits, red are losses"
4. **Drawdown Chart** → "This shows our maximum risk exposure"
5. **Rolling Sharpe** → "Consistent risk-adjusted returns over time"
6. **Correlation Matrix** → "Low correlation = good diversification"
7. **Export** → "All metrics available for your records"

**Total Demo Time**: ~5 minutes
**Impact**: High (professional, Bloomberg-style)

---

## 📱 Accessibility

### Keyboard Navigation:
- Tab through sections
- Arrow keys in tables
- Enter to download

### Screen Readers:
- All charts have alt text
- Metrics have descriptive labels
- Tables are properly structured

---

## 🔧 Troubleshooting

### "Not enough data for monthly heatmap"
**Solution**: Load more historical data (need at least 1 month)

### "Rolling Sharpe unavailable"
**Solution**: Need at least 126 periods for 252-period window

### Charts not rendering
**Solution**: Check browser console, refresh page

### Slow performance
**Solution**: Reduce data size or use faster data provider

---

## 📞 Support

For visual issues or questions:
1. Check `PORTFOLIO_ANALYTICS.md` for detailed documentation
2. Review `BLOOMBERG_TERMINAL_COMPLETE.md` for implementation details
3. See inline code comments in `main.py` (lines 914-1154)

---

**Visual Guide Complete!** 🎨

*Your Bloomberg Terminal-style dashboard is ready to impress!*

Last Updated: December 8, 2025
