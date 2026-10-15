"""Data cleaning, alignment, preprocessing, and leak-free scaling module.

Processes Spain hourly energy and weather data:
1. Deduplication and time alignment to 1h frequency.
2. Time-based interpolation of missing load values with audit logging.
3. Aggregates multi-city weather features into national means.
4. Feature enrichment: cyclical calendar, holidays, autoregressive lags, and rolling stats.
5. Strict chronological train/val/test splitting.
6. Leak-free scaler fitting (fitted on TRAIN ONLY).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

from src.features.lag_features import extract_all_lag_features
from src.features.time_features import extract_all_time_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    """Load yaml configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_energy_data(csv_path: Path) -> pd.DataFrame:
    """Clean and standardize energy dataset.

    Args:
        csv_path: Path to energy_dataset.csv

    Returns:
        Clean DataFrame indexed by hourly UTC timestamp with target 'total_load_actual'.
    """
    logger.info(f"Loading energy data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Standardize time column
    time_col = "time" if "time" in df.columns else df.columns[0]
    df["time"] = pd.to_datetime(df[time_col], utc=True)
    df = df.sort_values("time").drop_duplicates(subset=["time"])
    df = df.set_index("time")

    # Rename target column if needed
    target_col = "total load actual"
    if target_col in df.columns:
        df["total_load_actual"] = df[target_col]
    elif "total_load_actual" not in df.columns:
        raise KeyError("Could not find 'total load actual' column in energy data")

    # Verify hourly frequency and fill gaps
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="1h")
    missing_intervals = len(full_idx) - len(df)
    if missing_intervals > 0:
        logger.info(f"Reindexing to continuous 1h frequency. Found {missing_intervals} missing time steps.")
        df = df.reindex(full_idx)

    # Count and interpolate missing values
    missing_load = df["total_load_actual"].isna().sum()
    logger.info(f"Missing target values in 'total_load_actual': {missing_load} ({missing_load / len(df) * 100:.2f}%)")
    df["total_load_actual"] = df["total_load_actual"].interpolate(method="time")
    # Backfill/forward fill any remaining border NaNs
    df["total_load_actual"] = df["total_load_actual"].bfill().ffill()

    return df[["total_load_actual"]]


def clean_weather_data(csv_path: Path) -> pd.DataFrame:
    """Clean and aggregate 5-city weather features to mean hourly weather.

    Args:
        csv_path: Path to weather_features.csv

    Returns:
        Hourly DataFrame with averaged weather metrics:
        temp_mean, humidity_mean, wind_speed_mean, pressure_mean, clouds_all_mean
    """
    logger.info(f"Loading weather data from {csv_path}...")
    df = pd.read_csv(csv_path)

    dt_col = "dt_iso" if "dt_iso" in df.columns else df.columns[0]
    # Remove timezone offset if present and parse as UTC
    df["time"] = pd.to_datetime(df[dt_col].astype(str).str.slice(0, 19), utc=True)

    # Dedup per city and timestamp
    city_col = "city_name" if "city_name" in df.columns else "city"
    df = df.drop_duplicates(subset=["time", city_col])

    # Convert kelvin to celsius if temp > 150
    if "temp" in df.columns and df["temp"].mean() > 150:
        df["temp"] = df["temp"] - 273.15

    weather_cols = {
        "temp": "temp_mean",
        "humidity": "humidity_mean",
        "wind_speed": "wind_speed_mean",
        "pressure": "pressure_mean",
        "clouds_all": "clouds_all_mean",
    }

    avail_cols = [c for c in weather_cols if c in df.columns]
    agg_df = df.groupby("time")[avail_cols].mean()
    agg_df = agg_df.rename(columns=weather_cols)

    # Continuous hourly reindex & time interpolation
    full_idx = pd.date_range(start=agg_df.index.min(), end=agg_df.index.max(), freq="1h")
    agg_df = agg_df.reindex(full_idx).interpolate(method="time").bfill().ffill()

    return agg_df


def build_full_processed_dataset(
    config_path: str = "configs/config.yaml",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean, merge, feature-engineer, and return full dataset and split subsets."""
    cfg = load_config(config_path)
    energy_csv = Path(cfg["paths"]["spain_energy_raw"])
    weather_csv = Path(cfg["paths"]["spain_weather_raw"])
    proc_path = Path(cfg["paths"]["spain_processed"])
    proc_path.parent.mkdir(parents=True, exist_ok=True)

    if not energy_csv.exists() or not weather_csv.exists():
        raise FileNotFoundError(
            f"Required raw files missing: {energy_csv.name} or {weather_csv.name}. "
            "Please run src/data/download.py first."
        )

    energy_df = clean_energy_data(energy_csv)
    weather_df = clean_weather_data(weather_csv)

    # Inner join on timestamps
    df = energy_df.join(weather_df, how="inner")

    # Add calendar and cyclical features
    logger.info("Adding cyclical and holiday features...")
    df = extract_all_time_features(df, country="ES")

    # Add autoregressive lags and rolling features
    logger.info("Adding autoregressive lag and rolling window features...")
    target_col = cfg["dataset"]["target_col"]
    lags = cfg["dataset"]["features"]["lag_hours"]
    rolling_windows = cfg["dataset"]["features"]["rolling_windows"]
    df = extract_all_lag_features(df, target_col=target_col, lags=lags, windows=rolling_windows)

    # Drop early rows containing NaN from initial lag shift (max lag: 168h = 7 days)
    initial_len = len(df)
    df = df.dropna()
    logger.info(f"Dropped {initial_len - len(df)} warm-up rows due to 168h lag. Clean dataset: {len(df)} rows.")

    # Save processed dataset
    df.to_parquet(proc_path)
    logger.info(f"Saved processed dataset to {proc_path}")

    return df, cfg


def split_and_scale_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "total_load_actual",
    config_path: str = "configs/config.yaml",
) -> dict[str, Any]:
    """Perform chronological train/val/test split and fit scalers strictly on TRAIN.

    NO DATA LEAKAGE: Scalers are fitted on train only and applied to val/test.

    Returns:
        Dictionary containing scaled arrays, original dataframes, and fitted scalers.
    """
    cfg = load_config(config_path)
    split_dates = cfg["dataset"]["split_dates"]

    # Convert timestamps to UTC-aware pandas datetime for comparison
    train_mask = (df.index >= pd.to_datetime(split_dates["train_start"], utc=True)) & (
        df.index <= pd.to_datetime(split_dates["train_end"], utc=True)
    )
    val_mask = (df.index >= pd.to_datetime(split_dates["val_start"], utc=True)) & (
        df.index <= pd.to_datetime(split_dates["val_end"], utc=True)
    )
    test_mask = (df.index >= pd.to_datetime(split_dates["test_start"], utc=True)) & (
        df.index <= pd.to_datetime(split_dates["test_end"], utc=True)
    )

    df_train = df.loc[train_mask]
    df_val = df.loc[val_mask]
    df_test = df.loc[test_mask]

    logger.info(f"Chronological split sizes -> Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # Fit Scaler on TRAIN features only
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(df_train[feature_cols].values)
    X_val_scaled = feature_scaler.transform(df_val[feature_cols].values)
    X_test_scaled = feature_scaler.transform(df_test[feature_cols].values)

    # Fit Target Scaler on TRAIN target only
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(df_train[[target_col]].values)
    y_val_scaled = target_scaler.transform(df_val[[target_col]].values)
    y_test_scaled = target_scaler.transform(df_test[[target_col]].values)

    # Save scalers for serving/inference
    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_scaler, models_dir / "feature_scaler.joblib")
    joblib.dump(target_scaler, models_dir / "target_scaler.joblib")

    return {
        "df_train": df_train,
        "df_val": df_val,
        "df_test": df_test,
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train_scaled": y_train_scaled,
        "y_val_scaled": y_val_scaled,
        "y_test_scaled": y_test_scaled,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "feature_cols": feature_cols,
        "target_col": target_col,
    }


if __name__ == "__main__":
    build_full_processed_dataset()
