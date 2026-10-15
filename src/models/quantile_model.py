"""Uncertainty-aware Quantile Regression Forecaster.

Implements multi-quantile pinball loss at 10%, 50%, and 90% quantiles to produce
calibrated prediction interval bands (P10 to P90).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def multi_quantile_pinball_loss(quantiles: list[float] = (0.1, 0.5, 0.9)):
    """Return a composite Keras loss function computing average pinball loss across quantiles.

    Args:
        quantiles: Tuple/List of float quantiles (e.g. [0.1, 0.5, 0.9]).
    """
    taus = tf.reshape(tf.constant(quantiles, dtype=tf.float32), [1, -1])

    def loss(y_true, y_pred):
        y_true_col = tf.reshape(y_true, [-1, 1])
        error = y_true_col - y_pred
        q_loss = tf.maximum(taus * error, (taus - 1.0) * error)
        return tf.reduce_mean(q_loss)

    return loss


def build_quantile_forecaster(
    cell_type: str = "GRU",
    input_shape: tuple[int, int] = (24, 1),
    layer_units: list[int] = (64, 32),
    dropout_rate: float = 0.2,
    quantiles: list[float] = (0.1, 0.5, 0.9),
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """Build a quantile forecasting model outputting multiple quantile estimates."""
    inputs = layers.Input(shape=input_shape, name="sequence_input")
    x = inputs

    if cell_type.upper() == "GRU":
        layer_cls = layers.GRU
    else:
        layer_cls = layers.LSTM

    x = layer_cls(layer_units[0], return_sequences=True)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layer_cls(layer_units[1], return_sequences=False)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Output dimension matches number of quantiles
    outputs = layers.Dense(len(quantiles), activation="linear", name="quantile_outputs")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"Quantile_{cell_type}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=multi_quantile_pinball_loss(quantiles),
    )
    return model


def evaluate_prediction_intervals(
    y_true: np.ndarray,
    q10: np.ndarray,
    q90: np.ndarray,
) -> dict[str, float]:
    """Compute empirical coverage percentage and average interval width.

    Nominal coverage for [q10, q90] is 80.0%.
    """
    y = np.asarray(y_true).ravel()
    low = np.asarray(q10).ravel()
    high = np.asarray(q90).ravel()

    inside = (y >= low) & (y <= high)
    coverage = float(np.mean(inside) * 100.0)
    mean_width = float(np.mean(high - low))

    return {
        "coverage_percentage": coverage,
        "nominal_target": 80.0,
        "mean_interval_width": mean_width,
    }
