"""Regression and forecasting evaluation metrics.

Computes MSE, MAE, RMSE, R2, MAPE, directional accuracy, and pinball loss
for uncertainty evaluation.
"""

from __future__ import annotations

import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-5,
) -> float:
    """Compute Mean Absolute Percentage Error (percentage format, e.g., 5.2%)."""
    denominator = np.where(np.abs(y_true) < epsilon, epsilon, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute percentage of signs correctly predicted (for returns or price changes)."""
    true_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)
    return float(np.mean(true_sign == pred_sign) * 100.0)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Compute pinball (quantile) loss for a given target quantile tau in (0, 1)."""
    error = y_true - y_pred
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def evaluate_forecast(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
) -> dict[str, float]:
    """Evaluate comprehensive standard regression metrics.

    Returns:
        Dict containing mse, mae, rmse, r2, and mape.
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()

    return {
        "mse": mean_squared_error(y_t, y_p),
        "mae": mean_absolute_error(y_t, y_p),
        "rmse": root_mean_squared_error(y_t, y_p),
        "r2": r2_score(y_t, y_p),
        "mape": mean_absolute_percentage_error(y_t, y_p),
    }
