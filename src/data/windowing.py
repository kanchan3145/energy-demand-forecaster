"""Sliding window sequence generator for time series forecasting.

Transforms 2D tabular feature matrices into 3D sequence tensors (samples, window_size, features)
and corresponding target arrays (samples, horizon) without future lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_sliding_windows(
    data: np.ndarray | pd.DataFrame,
    target_idx: int = 0,
    window_size: int = 24,
    horizon: int = 1,
    step: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create input-target sliding sequence pairs.

    Args:
        data: 2D array or DataFrame of shape (num_timesteps, num_features).
        target_idx: Column index of the target feature within data.
        window_size: Number of past time steps to look back (input window).
        horizon: Number of future time steps to forecast (1 for one-step, >1 for direct multi-step).
        step: Stride between successive sliding windows (default 1).

    Returns:
        X: 3D numpy array of shape (num_samples, window_size, num_features).
        y: 1D or 2D numpy array of targets.
           If horizon == 1: shape (num_samples,)
           If horizon > 1: shape (num_samples, horizon)
    """
    if isinstance(data, pd.DataFrame):
        arr = data.values
    else:
        arr = np.asarray(data)

    if arr.ndim != 2:
        raise ValueError(f"Input data must be 2D array (timesteps, features), got ndim={arr.ndim}")

    total_len = len(arr)
    required_len = window_size + horizon
    if total_len < required_len:
        raise ValueError(
            f"Data length ({total_len}) is smaller than window_size ({window_size}) + horizon ({horizon})"
        )

    X_list = []
    y_list = []

    for i in range(0, total_len - required_len + 1, step):
        # Past window: i to i + window_size
        x_window = arr[i : i + window_size, :]
        # Future targets: i + window_size to i + window_size + horizon
        if horizon == 1:
            y_target = arr[i + window_size, target_idx]
        else:
            y_target = arr[i + window_size : i + window_size + horizon, target_idx]

        X_list.append(x_window)
        y_list.append(y_target)

    return np.asarray(X_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32)


def get_window_timestamps(
    datetime_series: pd.Series,
    window_size: int = 24,
    horizon: int = 1,
    step: int = 1,
) -> tuple[list, list]:
    """Track timestamps associated with input windows and forecast targets.

    Used for auditing, leak verification, and plotting.

    Returns:
        X_times: List of timestamp ranges/tuples [(t_start, t_end), ...]
        y_times: List of target timestamp tuples [(t_tgt_start, t_tgt_end), ...]
    """
    dt_vals = pd.to_datetime(datetime_series).tolist()
    total_len = len(dt_vals)
    required_len = window_size + horizon

    X_times = []
    y_times = []
    for i in range(0, total_len - required_len + 1, step):
        x_span = (dt_vals[i], dt_vals[i + window_size - 1])
        if horizon == 1:
            y_span = (dt_vals[i + window_size], dt_vals[i + window_size])
        else:
            y_span = (dt_vals[i + window_size], dt_vals[i + window_size + horizon - 1])
        X_times.append(x_span)
        y_times.append(y_span)

    return X_times, y_times
