"""
Machine Learning Prediction Module

Implements AI/ML-based price prediction using:
- LSTM (Long Short-Term Memory) neural networks
- Facebook Prophet for time series forecasting
- Ensemble methods combining multiple models

Features:
- Multi-step ahead forecasting
- Feature engineering (technical indicators, lag features)
- Model persistence and caching
- Confidence intervals

Author: GARCH Algo Intelligence Platform
License: MIT
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# TensorFlow/Keras for LSTM
try:
    import tensorflow as tf
    from tensorflow import keras
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    from keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available - LSTM predictions disabled")

# Prophet for time series
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet not available - Prophet predictions disabled")

logger = logging.getLogger(__name__)


@dataclass
class MLForecast:
    """
    Container for ML forecast results.

    Attributes:
        forecast_index: DatetimeIndex for predictions
        predictions: Predicted price values
        lower_bound: Lower confidence bound (if available)
        upper_bound: Upper confidence bound (if available)
        model_type: Type of model ('LSTM', 'Prophet', 'Ensemble')
        mae: Mean Absolute Error on validation set
        rmse: Root Mean Squared Error on validation set
        confidence: Confidence level (0-1)
    """
    forecast_index: pd.DatetimeIndex
    predictions: pd.Series
    lower_bound: Optional[pd.Series] = None
    upper_bound: Optional[pd.Series] = None
    model_type: str = "Unknown"
    mae: Optional[float] = None
    rmse: Optional[float] = None
    confidence: float = 0.80


class MLPredictorError(Exception):
    """Base exception for ML predictor errors."""
    pass


def create_sequences(
    data: np.ndarray,
    lookback: int = 60,
    forecast_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training.

    Args:
        data: 1D array of values (e.g., prices)
        lookback: Number of past steps to use as features
        forecast_horizon: Number of steps ahead to predict

    Returns:
        Tuple of (X, y) where X has shape (samples, lookback, 1)
        and y has shape (samples, forecast_horizon)
    """
    X, y = [], []

    for i in range(lookback, len(data) - forecast_horizon + 1):
        X.append(data[i - lookback:i])
        y.append(data[i:i + forecast_horizon])

    return np.array(X), np.array(y)


def build_lstm_model(
    lookback: int,
    forecast_horizon: int = 1,
    lstm_units: List[int] = [64, 32],
    dropout: float = 0.2
) -> Sequential:
    """
    Build LSTM model architecture.

    Args:
        lookback: Number of timesteps to look back
        forecast_horizon: Number of steps to predict
        lstm_units: List of units for each LSTM layer
        dropout: Dropout rate for regularization

    Returns:
        Compiled Keras Sequential model
    """
    if not TF_AVAILABLE:
        raise MLPredictorError("TensorFlow not available")

    model = Sequential()

    # First LSTM layer
    model.add(LSTM(
        lstm_units[0],
        return_sequences=len(lstm_units) > 1,
        input_shape=(lookback, 1)
    ))
    model.add(Dropout(dropout))

    # Additional LSTM layers
    for i in range(1, len(lstm_units)):
        return_seq = i < len(lstm_units) - 1
        model.add(LSTM(lstm_units[i], return_sequences=return_seq))
        model.add(Dropout(dropout))

    # Output layer
    model.add(Dense(forecast_horizon))

    # Compile
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    logger.info(f"Built LSTM model: {lstm_units} units, lookback={lookback}")
    return model


class LSTMPredictor:
    """
    LSTM-based price predictor.

    Example:
        >>> predictor = LSTMPredictor(lookback=60)
        >>> predictor.train(df['Close'])
        >>> forecast = predictor.predict(steps=24)
    """

    def __init__(
        self,
        lookback: int = 60,
        forecast_horizon: int = 1,
        lstm_units: List[int] = [64, 32],
        dropout: float = 0.2,
        val_split: float = 0.2
    ):
        """
        Initialize LSTM predictor.

        Args:
            lookback: Number of past periods to use
            forecast_horizon: Steps ahead to predict
            lstm_units: LSTM layer sizes
            dropout: Dropout rate
            val_split: Validation set fraction
        """
        if not TF_AVAILABLE:
            raise MLPredictorError("TensorFlow not installed")

        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.val_split = val_split

        self.model: Optional[Sequential] = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
        self.history = None

    def train(
        self,
        prices: pd.Series,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 0
    ) -> Dict[str, float]:
        """
        Train LSTM model on price data.

        Args:
            prices: Price series
            epochs: Training epochs
            batch_size: Batch size
            verbose: Keras verbosity

        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training LSTM on {len(prices)} observations")

        # Scale data
        prices_array = prices.values.reshape(-1, 1)
        scaled_prices = self.scaler.fit_transform(prices_array)

        # Create sequences
        X, y = create_sequences(scaled_prices, self.lookback, self.forecast_horizon)

        if len(X) == 0:
            raise MLPredictorError(
                f"Insufficient data for lookback={self.lookback}. "
                f"Need at least {self.lookback + self.forecast_horizon} points."
            )

        # Reshape for LSTM [samples, timesteps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Build model
        self.model = build_lstm_model(
            self.lookback,
            self.forecast_horizon,
            self.lstm_units,
            self.dropout
        )

        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # Train
        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=self.val_split,
            callbacks=[early_stop],
            verbose=verbose
        )

        self.is_trained = True

        # Calculate metrics
        val_loss = min(self.history.history['val_loss'])
        val_mae = min(self.history.history['val_mae'])

        logger.info(f"LSTM trained - Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}")

        return {
            "val_loss": val_loss,
            "val_mae": val_mae,
            "epochs_trained": len(self.history.history['loss'])
        }

    def predict(
        self,
        prices: pd.Series,
        steps: int = 24
    ) -> pd.Series:
        """
        Generate multi-step forecast.

        Args:
            prices: Historical prices
            steps: Number of steps to forecast

        Returns:
            Series with predictions
        """
        if not self.is_trained:
            raise MLPredictorError("Model not trained. Call train() first.")

        # Get last lookback window
        last_window = prices.values[-self.lookback:]
        scaled_window = self.scaler.transform(last_window.reshape(-1, 1))

        predictions = []
        current_window = scaled_window.copy()

        # Iterative forecasting
        for _ in range(steps):
            # Reshape for prediction
            X = current_window.reshape(1, self.lookback, 1)

            # Predict next step
            pred_scaled = self.model.predict(X, verbose=0)
            pred_value = pred_scaled[0, 0]  # Get first element of forecast

            predictions.append(pred_value)

            # Update window (rolling forecast)
            current_window = np.roll(current_window, -1)
            current_window[-1] = pred_value

        # Inverse transform
        predictions_array = np.array(predictions).reshape(-1, 1)
        predictions_unscaled = self.scaler.inverse_transform(predictions_array)

        # Create series with index
        last_idx = prices.index[-1]
        freq = pd.infer_freq(prices.index) or '30T'
        forecast_idx = pd.date_range(
            start=last_idx,
            periods=steps + 1,
            freq=freq,
            tz=last_idx.tz
        )[1:]

        return pd.Series(
            predictions_unscaled.flatten(),
            index=forecast_idx,
            name='lstm_forecast'
        )


class ProphetPredictor:
    """
    Facebook Prophet-based price predictor.

    Example:
        >>> predictor = ProphetPredictor()
        >>> forecast = predictor.predict(df, steps=24)
    """

    def __init__(
        self,
        changepoint_prior_scale: float = 0.05,
        seasonality_mode: str = 'multiplicative',
        daily_seasonality: bool = True,
        weekly_seasonality: bool = True
    ):
        """
        Initialize Prophet predictor.

        Args:
            changepoint_prior_scale: Flexibility of trend changes
            seasonality_mode: 'additive' or 'multiplicative'
            daily_seasonality: Enable daily seasonality
            weekly_seasonality: Enable weekly seasonality
        """
        if not PROPHET_AVAILABLE:
            raise MLPredictorError("Prophet not installed")

        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_mode = seasonality_mode
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality

    def predict(
        self,
        df: pd.DataFrame,
        steps: int = 24,
        price_col: str = 'Close'
    ) -> MLForecast:
        """
        Generate Prophet forecast.

        Args:
            df: DataFrame with datetime index and prices
            steps: Number of periods to forecast
            price_col: Column with prices

        Returns:
            MLForecast object
        """
        logger.info(f"Running Prophet forecast for {steps} steps")

        # Prepare data for Prophet (needs 'ds' and 'y' columns)
        prophet_df = pd.DataFrame({
            'ds': df.index,
            'y': df[price_col].values
        })

        # Initialize and fit Prophet
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = Prophet(
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_mode=self.seasonality_mode,
                daily_seasonality=self.daily_seasonality,
                weekly_seasonality=self.weekly_seasonality
            )
            model.fit(prophet_df)

        # Create future dataframe
        freq = pd.infer_freq(df.index) or '30T'
        future = model.make_future_dataframe(periods=steps, freq=freq)

        # Forecast
        forecast = model.predict(future)

        # Extract forecast portion
        forecast_only = forecast.tail(steps).copy()

        # Create result
        predictions = pd.Series(
            forecast_only['yhat'].values,
            index=pd.DatetimeIndex(forecast_only['ds']),
            name='prophet_forecast'
        )

        lower = pd.Series(
            forecast_only['yhat_lower'].values,
            index=pd.DatetimeIndex(forecast_only['ds']),
            name='prophet_lower'
        )

        upper = pd.Series(
            forecast_only['yhat_upper'].values,
            index=pd.DatetimeIndex(forecast_only['ds']),
            name='prophet_upper'
        )

        logger.info("Prophet forecast completed")

        return MLForecast(
            forecast_index=predictions.index,
            predictions=predictions,
            lower_bound=lower,
            upper_bound=upper,
            model_type="Prophet",
            confidence=0.80
        )


def ensemble_forecast(
    forecasts: List[MLForecast],
    weights: Optional[List[float]] = None
) -> MLForecast:
    """
    Combine multiple forecasts using weighted average.

    Args:
        forecasts: List of MLForecast objects
        weights: Optional weights (must sum to 1)

    Returns:
        Ensemble MLForecast
    """
    if not forecasts:
        raise MLPredictorError("No forecasts provided")

    if weights is None:
        weights = [1.0 / len(forecasts)] * len(forecasts)

    if len(weights) != len(forecasts):
        raise MLPredictorError("Weights must match number of forecasts")

    if not np.isclose(sum(weights), 1.0):
        raise MLPredictorError("Weights must sum to 1.0")

    # Combine predictions
    ensemble_pred = sum(
        fc.predictions * w for fc, w in zip(forecasts, weights)
    )

    # Combine bounds if available
    has_bounds = all(fc.lower_bound is not None for fc in forecasts)
    if has_bounds:
        ensemble_lower = sum(
            fc.lower_bound * w for fc, w in zip(forecasts, weights)
        )
        ensemble_upper = sum(
            fc.upper_bound * w for fc, w in zip(forecasts, weights)
        )
    else:
        ensemble_lower = None
        ensemble_upper = None

    model_types = "+".join(fc.model_type for fc in forecasts)

    return MLForecast(
        forecast_index=ensemble_pred.index,
        predictions=ensemble_pred,
        lower_bound=ensemble_lower,
        upper_bound=ensemble_upper,
        model_type=f"Ensemble({model_types})",
        confidence=0.80
    )


def quick_ml_forecast(
    df: pd.DataFrame,
    steps: int = 24,
    method: str = 'prophet',
    price_col: str = 'Close'
) -> MLForecast:
    """
    Quick ML forecast with sensible defaults.

    Args:
        df: DataFrame with price data
        steps: Forecast horizon
        method: 'prophet', 'lstm', or 'ensemble'
        price_col: Price column name

    Returns:
        MLForecast object

    Example:
        >>> forecast = quick_ml_forecast(df, steps=24, method='prophet')
        >>> print(forecast.predictions)
    """
    method = method.lower()

    if method == 'prophet':
        if not PROPHET_AVAILABLE:
            raise MLPredictorError("Prophet not available")
        predictor = ProphetPredictor()
        return predictor.predict(df, steps, price_col)

    elif method == 'lstm':
        if not TF_AVAILABLE:
            raise MLPredictorError("TensorFlow not available")
        predictor = LSTMPredictor(lookback=60)
        predictor.train(df[price_col], epochs=30, verbose=0)
        predictions = predictor.predict(df[price_col], steps)

        return MLForecast(
            forecast_index=predictions.index,
            predictions=predictions,
            model_type="LSTM",
            confidence=0.80
        )

    elif method == 'ensemble':
        forecasts = []

        if PROPHET_AVAILABLE:
            try:
                prophet_fc = quick_ml_forecast(df, steps, 'prophet', price_col)
                forecasts.append(prophet_fc)
            except Exception as e:
                logger.warning(f"Prophet failed: {e}")

        if TF_AVAILABLE and len(df) >= 100:
            try:
                lstm_fc = quick_ml_forecast(df, steps, 'lstm', price_col)
                forecasts.append(lstm_fc)
            except Exception as e:
                logger.warning(f"LSTM failed: {e}")

        if not forecasts:
            raise MLPredictorError("All ensemble methods failed")

        return ensemble_forecast(forecasts)

    else:
        raise MLPredictorError(f"Unknown method: {method}")
