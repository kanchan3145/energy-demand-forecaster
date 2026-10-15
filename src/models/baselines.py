"""Classical and Naive baseline models for energy demand forecasting.

Implements:
1. Naive persistence (t - 1)
2. Seasonal Naive 24h (t - 24)
3. Seasonal Naive 168h (t - 168)
4. Linear Regression on autoregressive lags
5. ARIMA / SARIMA benchmark
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from statsmodels.tsa.arima.model import ARIMA

from src.evaluation.metrics import evaluate_forecast

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NaiveForecaster:
    """Naive persistence forecaster: predicts y_t = y_{t - lag}."""

    def __init__(self, lag: int = 1):
        self.lag = lag

    def predict(self, series: pd.Series) -> np.ndarray:
        """Generate forecasts based on the given lag."""
        pred = series.shift(self.lag)
        return pred.values


class LinearLagForecaster:
    """Linear regression / Ridge on autoregressive lags."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=self.alpha)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> LinearLagForecaster:
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


def run_sarima_baseline(
    train_series: pd.Series,
    test_series: pd.Series,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 0, 0, 24),
    sample_size: int = 500,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit ARIMA/SARIMA model on a representative recent train slice and forecast test slice.

    Full SARIMA on 25,000+ hourly steps is computationally prohibitive on CPU;
    evaluating on a representative test slice ensures exact reproducibility in seconds.
    """
    logger.info(f"Fitting SARIMA{order}x{seasonal_order} on recent {sample_size} training hours...")
    recent_train = train_series.iloc[-sample_size:].values
    model = ARIMA(recent_train, order=order, seasonal_order=seasonal_order)
    fitted = model.fit()

    test_slice = test_series.iloc[:sample_size].values
    forecasts = fitted.forecast(steps=len(test_slice))

    metrics = evaluate_forecast(test_slice, forecasts)
    return forecasts, metrics


def evaluate_all_baselines(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str = "total_load_actual",
    lag_cols: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute benchmark performance for Naive, Seasonal Naive (24h, 168h), Linear Regression, and SARIMA."""
    # Concatenate train and test to allow seamless lagged lookback into end of train
    combined = pd.concat([df_train, df_test]).sort_index()
    test_idx = df_test.index
    y_test = df_test[target_col].values

    results = {}

    # 1. Naive (1h persistence)
    y_pred_naive = combined[target_col].shift(1).loc[test_idx].values
    results["Naive_1h"] = evaluate_forecast(y_test, y_pred_naive)

    # 2. Seasonal Naive 24h (yesterday same hour)
    y_pred_snaive_24 = combined[target_col].shift(24).loc[test_idx].values
    results["Seasonal_Naive_24h"] = evaluate_forecast(y_test, y_pred_snaive_24)

    # 3. Seasonal Naive 168h (last week same hour)
    y_pred_snaive_168 = combined[target_col].shift(168).loc[test_idx].values
    results["Seasonal_Naive_168h"] = evaluate_forecast(y_test, y_pred_snaive_168)

    # 4. Linear Regression on Lag Features
    if lag_cols is None:
        lag_cols = [c for c in df_train.columns if "lag" in c or "roll" in c]

    if lag_cols:
        X_train_lags = df_train[lag_cols].values
        y_train = df_train[target_col].values
        X_test_lags = df_test[lag_cols].values

        lr_model = LinearLagForecaster(alpha=10.0)
        lr_model.fit(X_train_lags, y_train)
        y_pred_lr = lr_model.predict(X_test_lags)
        results["Linear_Regression_Lags"] = evaluate_forecast(y_test, y_pred_lr)

    # 5. Fast SARIMA benchmark on sample slice
    try:
        _, sarima_metrics = run_sarima_baseline(
            df_train[target_col],
            df_test[target_col],
            order=(1, 1, 1),
            seasonal_order=(1, 0, 0, 24),
            sample_size=336, # 2-week representative evaluation
        )
        results["SARIMA_Benchmark"] = sarima_metrics
    except Exception as e:
        logger.warning(f"SARIMA baseline failed: {e}")

    return results
