"""Lag and rolling statistics feature engineering module.

Constructs strictly historical autoregressive lags and rolling window aggregates
without forward-looking leakage.
"""

from __future__ import annotations

import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: list[int] = (24, 48, 168),
) -> pd.DataFrame:
    """Create backward lag features for target column.

    Args:
        df: Input DataFrame with chronological index or ordering.
        target_col: Target column to lag.
        lags: List of positive integer shift steps (e.g. 24 for 1 day prior, 168 for 1 week prior).

    Returns:
        DataFrame with added lag columns named f'{target_col}_lag_{k}'.
    """
    df_out = df.copy()
    for lag in lags:
        if lag <= 0:
            raise ValueError(f"Lag must be positive integer to prevent lookahead, got {lag}")
        col_name = f"{target_col}_lag_{lag}"
        df_out[col_name] = df_out[target_col].shift(lag)
    return df_out


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    windows: list[int] = (24, 168),
    min_shift: int = 1,
) -> pd.DataFrame:
    """Create rolling mean and std features shifted by min_shift to prevent leakage.

    Args:
        df: Input DataFrame.
        target_col: Target column to compute rolling stats over.
        windows: List of window lengths in steps.
        min_shift: Minimum shift step to ensure t does not include current observation t.
                   Default is 1 (shifted by 1 hour).

    Returns:
        DataFrame with added rolling statistic columns.
    """
    df_out = df.copy()
    shifted_series = df_out[target_col].shift(min_shift)

    for w in windows:
        mean_col = f"{target_col}_roll_mean_{w}"
        std_col = f"{target_col}_roll_std_{w}"
        df_out[mean_col] = shifted_series.rolling(window=w, min_periods=max(1, w // 4)).mean()
        df_out[std_col] = shifted_series.rolling(window=w, min_periods=max(1, w // 4)).std()

    return df_out


def extract_all_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: list[int] = (24, 48, 168),
    windows: list[int] = (24, 168),
) -> pd.DataFrame:
    """Convenience pipeline to create both lags and rolling statistics."""
    df_lagged = add_lag_features(df, target_col=target_col, lags=lags)
    return add_rolling_features(df_lagged, target_col=target_col, windows=windows)
