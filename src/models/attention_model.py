"""Transformer Time Series Forecaster with Multi-Head Self-Attention.

Implements Positional Encoding and MultiHeadAttention encoder blocks for sequential electricity demand.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class PositionalEncoding(layers.Layer):
    """Sinusoidal positional encoding layer for sequence inputs."""

    def __init__(self, sequence_length: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model

        # Compute sinusoidal positional encodings
        pos = np.arange(sequence_length)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1.0 / np.power(10000.0, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates

        # Apply sin to even indices, cos to odd
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.zeros((sequence_length, d_model))
        pos_encoding[:, 0::2] = sines
        pos_encoding[:, 1::2] = cosines

        self.pos_encoding = tf.constant(pos_encoding[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "d_model": self.d_model,
        })
        return config


def build_transformer_forecaster(
    input_shape: tuple[int, int] = (24, 1),
    d_model: int = 64,
    num_heads: int = 4,
    ff_dim: int = 128,
    dropout_rate: float = 0.2,
    output_dim: int = 1,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """Build a Transformer Encoder model for time series forecasting.

    Includes multi-head self-attention, layer normalization, residual connections,
    and a feed-forward network.
    """
    seq_len, num_features = input_shape
    inputs = layers.Input(shape=input_shape, name="time_series_input")

    # Project input features to d_model space
    x = layers.Dense(d_model, name="input_projection")(inputs)

    # Add Positional Encoding
    x = PositionalEncoding(sequence_length=seq_len, d_model=d_model, name="pos_encoding")(x)
    x = layers.Dropout(dropout_rate)(x)

    # Multi-Head Self-Attention Block
    attn_output, attn_weights = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        name="multihead_attention",
    )(x, x, return_attention_scores=True)
    attn_output = layers.Dropout(dropout_rate)(attn_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

    # Feed-Forward Network
    ffn = tf.keras.Sequential([
        layers.Dense(ff_dim, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(d_model),
    ], name="ffn_block")
    ffn_output = ffn(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

    # Sequence Pooling and Output
    pooled = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    outputs = layers.Dense(output_dim, activation="linear", name="forecast_output")(pooled)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Transformer_Forecaster")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model
