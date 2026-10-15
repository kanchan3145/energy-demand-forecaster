"""Recurrent Neural Network Architectures: SimpleRNN, LSTM, GRU, BiLSTM, and CNN-LSTM.

All models share identical parameterizations to isolate cell-type effects.
"""

from __future__ import annotations

import logging

import tensorflow as tf
from tensorflow.keras import layers, regularizers

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_rnn_model(
    cell_type: str = "LSTM",
    input_shape: tuple[int, int] = (24, 1),
    layer_units: list[int] = (64, 32),
    dropout_rate: float = 0.2,
    output_dim: int = 1,
    learning_rate: float = 0.001,
    l2_reg: float = 0.0,
    optimizer_name: str = "adam",
    momentum: float = 0.0,
) -> tf.keras.Model:
    """Build a standard 2-layer stacked Recurrent Neural Network.

    Args:
        cell_type: One of 'SimpleRNN', 'LSTM', 'GRU', 'BiLSTM', 'CNN_LSTM'.
        input_shape: (window_size, num_features).
        layer_units: Units in layer 1 and layer 2. Default [64, 32].
        dropout_rate: Dropout fraction after each recurrent layer.
        output_dim: Dimensionality of output (1 for single-step, 24 for direct multi-step).
        learning_rate: Optimizer learning rate.
        l2_reg: L2 kernel regularization coefficient.
        optimizer_name: 'adam' or 'sgd'.
        momentum: Momentum if optimizer is 'sgd'.

    Returns:
        Compiled tf.keras.Model.
    """
    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None
    cell = cell_type.upper()

    inputs = layers.Input(shape=input_shape, name="sequence_input")
    x = inputs

    if cell == "CNN_LSTM":
        # 1D Convolution feature extraction followed by LSTM
        x = layers.Conv1D(filters=32, kernel_size=3, padding="causal", activation="relu")(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.LSTM(layer_units[0], return_sequences=True, kernel_regularizer=reg)(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.LSTM(layer_units[1], return_sequences=False, kernel_regularizer=reg)(x)
        x = layers.Dropout(dropout_rate)(x)

    elif cell == "BILSTM":
        # Bidirectional LSTM
        x = layers.Bidirectional(
            layers.LSTM(layer_units[0], return_sequences=True, kernel_regularizer=reg)
        )(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Bidirectional(
            layers.LSTM(layer_units[1], return_sequences=False, kernel_regularizer=reg)
        )(x)
        x = layers.Dropout(dropout_rate)(x)

    else:
        # Standard SimpleRNN, LSTM, or GRU
        if cell == "SIMPLERNN":
            layer_cls = layers.SimpleRNN
        elif cell == "LSTM":
            layer_cls = layers.LSTM
        elif cell == "GRU":
            layer_cls = layers.GRU
        else:
            raise ValueError(f"Unsupported cell_type: {cell_type}")

        # Layer 1 (returns full sequence to feed layer 2)
        x = layer_cls(layer_units[0], return_sequences=True, kernel_regularizer=reg, name="rnn_layer_1")(x)
        x = layers.Dropout(dropout_rate, name="dropout_1")(x)

        # Layer 2 (returns last state)
        x = layer_cls(layer_units[1], return_sequences=False, kernel_regularizer=reg, name="rnn_layer_2")(x)
        x = layers.Dropout(dropout_rate, name="dropout_2")(x)

    # Dense Output Layer
    outputs = layers.Dense(output_dim, activation="linear", name="dense_output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{cell}_Forecaster")

    # Select Optimizer
    if optimizer_name.lower() == "sgd":
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    elif optimizer_name.lower() == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        opt = tf.keras.optimizers.get(optimizer_name)

    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model
