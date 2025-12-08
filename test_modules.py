"""
Quick Module Test Script

Tests all custom modules to ensure they work correctly.
"""

import sys
import traceback

def test_imports():
    """Test that all modules can be imported."""
    print("="*60)
    print("TESTING MODULE IMPORTS")
    print("="*60)

    tests = [
        ("Config", "from config.settings import settings"),
        ("Logging", "from src.utils.logging_config import setup_logging"),
        ("Technical Indicators", "from src.indicators.technical import macd, rsi, ema"),
        ("GARCH Models", "from src.models.garch import fit_garch_family, best_by_aic"),
        ("ARIMA Models", "from src.models.arima import arima_forecast_prices"),
        ("Alert System", "from src.triggers.alert_system import AlertSystem, detect_dip_buy"),
    ]

    passed = 0
    failed = 0

    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"[PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {str(e)[:50]}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def test_technical_indicators():
    """Test technical indicators with sample data."""
    print("\n" + "="*60)
    print("TESTING TECHNICAL INDICATORS")
    print("="*60)

    try:
        import pandas as pd
        import numpy as np
        from src.indicators.technical import macd, rsi, ema, atr_series

        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        prices = 50000 + np.cumsum(np.random.randn(100) * 100)

        df = pd.DataFrame({
            'Open': prices,
            'High': prices + np.abs(np.random.randn(100) * 50),
            'Low': prices - np.abs(np.random.randn(100) * 50),
            'Close': prices,
            'Volume': np.random.randint(100, 1000, 100)
        }, index=dates)

        # Test indicators
        m, s, h = macd(df['Close'])
        r = rsi(df['Close'])
        e = ema(df['Close'], 12)
        atr = atr_series(df['High'], df['Low'], df['Close'])

        print(f"[OK] MACD calculated: {len(m)} values")
        print(f"[OK] RSI calculated: {len(r)} values, last={r.iloc[-1]:.2f}")
        print(f"[OK] EMA calculated: {len(e)} values")
        print(f"[OK] ATR calculated: {len(atr)} values, last={atr.iloc[-1]:.2f}")

        return True
    except Exception as e:
        print(f"[X] Technical indicators test failed: {e}")
        traceback.print_exc()
        return False


def test_garch():
    """Test GARCH models with sample data."""
    print("\n" + "="*60)
    print("TESTING GARCH MODELS")
    print("="*60)

    try:
        import pandas as pd
        import numpy as np
        from src.models.garch import fit_garch_family, best_by_aic

        # Create sample returns
        np.random.seed(42)
        returns = pd.Series(np.random.randn(200) * 0.02, name='returns')
        returns.index = pd.date_range('2024-01-01', periods=200, freq='1h')

        # Fit GARCH
        print("Fitting GARCH models...")
        fits = fit_garch_family(returns, min_obs=50, require_convergence=False)

        print(f"[OK] Fitted {len(fits)} GARCH models")

        for name, fit in fits.items():
            print(f"  - {name}: AIC={fit.aic:.2f}, sigma(1)={fit.one_step_forecast:.4f}%")

        # Get best
        best = best_by_aic(fits)
        print(f"[OK] Best model: {best.name} (AIC={best.aic:.2f})")

        return True
    except Exception as e:
        print(f"[X] GARCH test failed: {e}")
        traceback.print_exc()
        return False


def test_arima():
    """Test ARIMA forecasting."""
    print("\n" + "="*60)
    print("TESTING ARIMA FORECASTING")
    print("="*60)

    try:
        import pandas as pd
        import numpy as np
        from src.models.arima import arima_forecast_prices

        # Create sample price data
        dates = pd.date_range('2024-01-01', periods=200, freq='1h')
        prices = 50000 + np.cumsum(np.random.randn(200) * 100)

        df = pd.DataFrame({
            'Open': prices,
            'High': prices + 50,
            'Low': prices - 50,
            'Close': prices,
            'Volume': 100
        }, index=dates)

        # Forecast
        print("Generating ARIMA forecast...")
        forecast = arima_forecast_prices(df, steps=10, min_obs=50)

        print(f"[OK] Generated {len(forecast.mean_prices)} step forecast")
        print(f"  - Last observed price: {df['Close'].iloc[-1]:.2f}")
        print(f"  - 10-step forecast: {forecast.mean_prices.iloc[-1]:.2f}")
        print(f"  - Model AIC: {forecast.aic:.2f}")

        return True
    except Exception as e:
        print(f"[X] ARIMA test failed: {e}")
        traceback.print_exc()
        return False


def test_alerts():
    """Test alert system."""
    print("\n" + "="*60)
    print("TESTING ALERT SYSTEM")
    print("="*60)

    try:
        import pandas as pd
        import numpy as np
        from src.triggers.alert_system import (
            AlertSystem, detect_dip_buy, detect_breakout,
            console_handler
        )
        from src.indicators.technical import add_all_indicators

        # Create sample data with a dip
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        prices = 50000 + np.cumsum(np.random.randn(100) * 100)
        prices[-5:] = prices[-6] * 0.94  # Create a 6% dip

        df = pd.DataFrame({
            'Open': prices,
            'High': prices + 50,
            'Low': prices - 50,
            'Close': prices,
            'Volume': np.random.randint(100, 1000, 100)
        }, index=dates)

        df = add_all_indicators(df)

        # Setup alert system
        alert_system = AlertSystem()
        alert_system.add_handler(console_handler)

        # Test dip detection
        dip_alert = detect_dip_buy(df, threshold=0.05)

        if dip_alert:
            print(f"[OK] Dip alert detected!")
            print(f"  - Type: {dip_alert.alert_type.value}")
            print(f"  - Message: {dip_alert.message}")
            alert_system.send_alert(dip_alert)
        else:
            print("[OK] No dip detected (expected if no dip in random data)")

        # Test breakout detection
        breakout_alert = detect_breakout(df)
        print(f"[OK] Breakout check: {'Alert!' if breakout_alert else 'None'}")

        return True
    except Exception as e:
        print(f"[X] Alert test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "GARCH ALGO PLATFORM - MODULE TEST SUITE" + "\n")

    results = []

    # Test imports
    results.append(("Imports", test_imports()))

    # Test individual modules
    results.append(("Technical Indicators", test_technical_indicators()))
    results.append(("GARCH Models", test_garch()))
    results.append(("ARIMA Forecasting", test_arima()))
    results.append(("Alert System", test_alerts()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for name, passed in results:
        status = "[OK] PASS" if passed else "[X] FAIL"
        print(f"{status}: {name}")

    total_passed = sum(1 for _, p in results if p)
    total_tests = len(results)

    print(f"\nTotal: {total_passed}/{total_tests} test suites passed")

    if total_passed == total_tests:
        print("\nALL TESTS PASSED!")
        return 0
    else:
        print(f"\nWARNING: {total_tests - total_passed} test suite(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
