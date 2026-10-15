"""Time feature engineering module.

Extracts cyclical trigonometric features (sin/cos for hour, day of week, month),
binary flags for weekend and national/regional holidays (Spain and generic).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

try:
    import holidays
except ImportError:
    holidays = None


def add_cyclical_time_features(df: pd.DataFrame, datetime_col: str | None = None) -> pd.DataFrame:
    """Add sin and cos cyclical encodings for hour, day of week, and month.

    Args:
        df: Input DataFrame containing datetime index or datetime column.
        datetime_col: Column name containing timestamps. If None, uses df.index.

    Returns:
        DataFrame copy with added cyclical columns:
        - hour_sin, hour_cos
        - dayofweek_sin, dayofweek_cos
        - month_sin, month_cos
    """
    df_out = df.copy()
    if datetime_col is not None:
        dt_series = pd.to_datetime(df_out[datetime_col])
    else:
        dt_series = pd.Series(pd.to_datetime(df_out.index), index=df_out.index)

    # Hour of day: 0 to 23 -> period = 24
    hours = dt_series.dt.hour
    df_out["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df_out["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)

    # Day of week: 0 (Monday) to 6 (Sunday) -> period = 7
    dows = dt_series.dt.dayofweek
    df_out["dayofweek_sin"] = np.sin(2 * np.pi * dows / 7.0)
    df_out["dayofweek_cos"] = np.cos(2 * np.pi * dows / 7.0)

    # Month: 1 to 12 -> period = 12
    months = dt_series.dt.month
    df_out["month_sin"] = np.sin(2 * np.pi * (months - 1) / 12.0)
    df_out["month_cos"] = np.cos(2 * np.pi * (months - 1) / 12.0)

    # Weekend flag (Saturday=5, Sunday=6)
    df_out["is_weekend"] = (dows >= 5).astype(int)

    return df_out


# Fixed Spanish national holidays (month, day) for fallback
SPAIN_FIXED_HOLIDAYS = {
    (1, 1),   # New Year's Day
    (1, 6),   # Epiphany
    (5, 1),   # Labour Day
    (8, 15),  # Assumption of Mary
    (10, 12), # National Day
    (11, 1),  # All Saints' Day
    (12, 6),  # Constitution Day
    (12, 8),  # Immaculate Conception
    (12, 25), # Christmas Day
}


def add_holiday_features(
    df: pd.DataFrame,
    country: str = "ES",
    years: Iterable[int] | None = None,
    datetime_col: str | None = None,
) -> pd.DataFrame:
    """Add binary indicator for public holidays in the target country."""
    df_out = df.copy()
    if datetime_col is not None:
        dt_series = pd.to_datetime(df_out[datetime_col])
    else:
        dt_series = pd.Series(pd.to_datetime(df_out.index), index=df_out.index)

    dates = dt_series.dt.date

    if holidays is not None:
        if years is None:
            years = dt_series.dt.year.unique()
        try:
            country_holidays = holidays.country_holidays(country, years=list(years))
        except Exception:
            country_holidays = holidays.Spain(years=list(years))
        df_out["is_holiday"] = dates.apply(lambda d: 1 if d in country_holidays else 0).astype(int)
    else:
        # Fallback to fixed national holidays
        df_out["is_holiday"] = dt_series.apply(
            lambda dt: 1 if (dt.month, dt.day) in SPAIN_FIXED_HOLIDAYS else 0
        ).astype(int)

    return df_out


def extract_all_time_features(
    df: pd.DataFrame,
    country: str = "ES",
    datetime_col: str | None = None,
) -> pd.DataFrame:
    """Convenience pipeline to extract all cyclical, weekend, and holiday features."""
    df_cyclical = add_cyclical_time_features(df, datetime_col=datetime_col)
    return add_holiday_features(df_cyclical, country=country, datetime_col=datetime_col)
