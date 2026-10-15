"""Comprehensive Plotting and Visualization Module.

Generates high-resolution figures for:
- EDA (Full series, zoomed week/day, diurnal/weekly/monthly profiles, STL, ACF/PACF)
- Training dynamics (loss curves, gradient norms for vanishing gradients)
- Model forecasts (representative week, extreme weather week, prediction intervals)
- Residual diagnostics (residuals over time, error by hour of day, error by month)
- Feature importance and Transformer attention maps.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.seasonal import seasonal_decompose
except ImportError:
    plot_acf, plot_pacf, seasonal_decompose = None, None, None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Style defaults
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["figure.titlesize"] = 12


def plot_eda_series(df: pd.DataFrame, target_col: str, save_dir: Path) -> list[str]:
    """Generate Phase 1 exploratory data analysis time series figures."""
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    # 1. Full series
    fig, ax = plt.subplots(figsize=(14, 4), dpi=150)
    ax.plot(df.index, df[target_col], color="#1f77b4", linewidth=0.6, label="Hourly Electricity Demand (MW)")
    ax.set_title("Spain Total Hourly Electricity Demand (2015 - 2018)", fontweight="bold")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Demand (MW)")
    ax.legend(loc="upper right")
    fpath = save_dir / "01_full_series.png"
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()
    saved_files.append(str(fpath))

    # 2. Zoomed representative week (7 consecutive days)
    sample_week = df.loc["2017-06-05":"2017-06-11"]
    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    ax.plot(sample_week.index, sample_week[target_col], marker="o", markersize=3, color="#2ca02c", label="Demand (MW)")
    ax.set_title("Representative Zoomed Week (June 5 - June 11, 2017) Showing Diurnal Peaks", fontweight="bold")
    ax.set_xlabel("Time (Hourly)")
    ax.set_ylabel("Demand (MW)")
    ax.legend()
    fpath = save_dir / "02_zoomed_week.png"
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()
    saved_files.append(str(fpath))

    # 3. Zoomed single day (24 hours)
    sample_day = df.loc["2017-06-07"]
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.plot(sample_day.index.hour, sample_day[target_col], marker="s", color="#ff7f0e", label="Single Day Demand")
    ax.set_title("Single Day Profile (June 7, 2017): Afternoon & Evening Peaks", fontweight="bold")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Demand (MW)")
    ax.set_xticks(range(24))
    ax.legend()
    fpath = save_dir / "03_zoomed_day.png"
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()
    saved_files.append(str(fpath))

    # 4. Average demand by Hour of Day, Day of Week, Month
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), dpi=150)

    # Hour of day
    hourly_avg = df.groupby(df.index.hour)[target_col].mean()
    axes[0].plot(hourly_avg.index, hourly_avg.values, marker="o", color="#d62728")
    axes[0].set_title("Average Demand by Hour of Day")
    axes[0].set_xlabel("Hour (0-23)")
    axes[0].set_ylabel("Mean Demand (MW)")
    axes[0].set_xticks(range(0, 24, 2))

    # Day of week
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_avg = df.groupby(df.index.dayofweek)[target_col].mean()
    axes[1].bar(dow_names, dow_avg.values, color="#9467bd", alpha=0.85)
    axes[1].set_title("Average Demand by Day of Week")
    axes[1].set_ylabel("Mean Demand (MW)")

    # Month
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_avg = df.groupby(df.index.month)[target_col].mean()
    axes[2].bar(month_names, month_avg.values, color="#8c564b", alpha=0.85)
    axes[2].set_title("Average Demand by Month")
    axes[2].set_ylabel("Mean Demand (MW)")

    fpath = save_dir / "04_diurnal_weekly_monthly_profiles.png"
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()
    saved_files.append(str(fpath))

    # 5. Seasonal Decomposition (STL / Classic)
    try:
        # Sample 2000 hours for fast, clear decomposition
        decomp = seasonal_decompose(df[target_col].iloc[:2000], model="additive", period=24)
        fig = decomp.plot()
        fig.set_size_inches(12, 8)
        fig.suptitle("Additive Seasonal Decomposition (Trend, 24h Seasonality, Residuals)", fontweight="bold", y=1.01)
        fpath = save_dir / "05_seasonal_decomposition.png"
        plt.tight_layout()
        plt.savefig(fpath)
        plt.close()
        saved_files.append(str(fpath))
    except Exception as e:
        logger.warning(f"Seasonal decomposition plot failed: {e}")

    # 6. Autocorrelation & Partial Autocorrelation (ACF / PACF)
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4), dpi=150)
        plot_acf(df[target_col].iloc[:1000], lags=48, ax=axes[0], title="Autocorrelation Function (ACF - 48 Lags)")
        plot_pacf(df[target_col].iloc[:1000], lags=48, ax=axes[1], title="Partial Autocorrelation Function (PACF - 48 Lags)")
        fpath = save_dir / "06_acf_pacf.png"
        plt.tight_layout()
        plt.savefig(fpath)
        plt.close()
        saved_files.append(str(fpath))
    except Exception as e:
        logger.warning(f"ACF/PACF plot failed: {e}")

    return saved_files


def plot_training_history(history: dict[str, list], model_name: str, save_path: Path) -> None:
    """Plot training and validation loss curves across epochs."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    epochs = range(1, len(history["loss"]) + 1)
    ax.plot(epochs, history["loss"], label="Training Loss (MSE)", color="#1f77b4", linewidth=1.5)
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], label="Validation Loss (MSE)", color="#ff7f0e", linewidth=1.5, linestyle="--")
    ax.set_title(f"Learning Curves: {model_name}", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_forecast_vs_actual(
    timestamps: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """Plot time-aligned actual vs forecast curves."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 4), dpi=150)
    ax.plot(timestamps, y_true, label="Actual Demand", color="black", linewidth=1.2)
    ax.plot(timestamps, y_pred, label="Predicted Demand", color="#1f77b4", linestyle="--", linewidth=1.2)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Demand (MW)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_residuals_diagnostics(
    timestamps: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: Path,
    prefix: str = "core_lstm",
) -> list[str]:
    """Plot residual distribution, residuals over time, and errors by hour and month."""
    save_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    residuals = y_true - y_pred
    abs_errors = np.abs(residuals)

    df_err = pd.DataFrame({
        "time": timestamps,
        "residual": residuals,
        "mae": abs_errors,
        "hour": timestamps.hour,
        "month": timestamps.month,
    })

    # 1. Residual distribution and time trace
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), dpi=150)
    sns.histplot(residuals, kde=True, ax=axes[0], color="#2b5c8f")
    axes[0].set_title("Residual Error Distribution (Actual - Pred)")
    axes[0].set_xlabel("Residual (MW)")

    axes[1].scatter(timestamps, residuals, alpha=0.3, s=4, color="#d62728")
    axes[1].axhline(0, color="black", linestyle="--")
    axes[1].set_title("Residuals Over Test Horizon")
    axes[1].set_xlabel("Timestamp")
    axes[1].set_ylabel("Residual (MW)")
    fpath1 = save_dir / f"{prefix}_residuals_overview.png"
    plt.tight_layout()
    plt.savefig(fpath1)
    plt.close()
    saved.append(str(fpath1))

    # 2. MAE by hour of day and by month
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), dpi=150)
    hour_err = df_err.groupby("hour")["mae"].mean()
    axes[0].plot(hour_err.index, hour_err.values, marker="o", color="#e377c2")
    axes[0].set_title("Mean Absolute Error by Hour of Day")
    axes[0].set_xlabel("Hour (0-23)")
    axes[0].set_ylabel("MAE (MW)")
    axes[0].set_xticks(range(0, 24, 2))

    month_err = df_err.groupby("month")["mae"].mean()
    axes[1].bar(month_err.index, month_err.values, color="#17becf")
    axes[1].set_title("Mean Absolute Error by Month")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("MAE (MW)")
    fpath2 = save_dir / f"{prefix}_error_by_hour_month.png"
    plt.tight_layout()
    plt.savefig(fpath2)
    plt.close()
    saved.append(str(fpath2))

    return saved


def plot_prediction_intervals(
    timestamps: pd.DatetimeIndex,
    y_true: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    save_path: Path,
) -> None:
    """Plot uncertainty prediction intervals band (P10 - P90) with median P50 and actuals."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    ax.fill_between(timestamps, q10, q90, color="#aec7e8", alpha=0.5, label="80% Prediction Interval (P10 - P90)")
    ax.plot(timestamps, q50, color="#1f77b4", linewidth=1.5, label="Median Forecast (P50)")
    ax.plot(timestamps, y_true, color="black", linewidth=1.2, label="Actual Demand")
    ax.set_title("Uncertainty-Aware Demand Forecast: 80% Prediction Band (Pinball Loss)", fontweight="bold")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Demand (MW)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_vanishing_gradients(
    time_steps: list[int],
    rnn_grads: list[float],
    lstm_grads: list[float],
    gru_grads: list[float],
    save_path: Path,
) -> None:
    """Plot gradient norms backpropagated across time steps showing vanishing gradients."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    ax.plot(time_steps, rnn_grads, marker="o", label="SimpleRNN (Severe Vanishing)", color="#d62728", linewidth=1.8)
    ax.plot(time_steps, lstm_grads, marker="s", label="LSTM (Constant Error Carousel)", color="#2ca02c", linewidth=1.8)
    ax.plot(time_steps, gru_grads, marker="^", label="GRU (Gated Reset/Update)", color="#1f77b4", linewidth=1.8)
    ax.set_title("Backpropagation Through Time: Gradient Norm vs Sequence Lag", fontweight="bold")
    ax.set_xlabel("Time Steps Back from Target (t - k)")
    ax.set_ylabel("Average Gradient Norm ||dL/dh_t|| (Log Scale)")
    ax.set_yscale("log")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_feature_importance(
    feature_names: list[str],
    importances: list[float],
    save_path: Path,
) -> None:
    """Plot permutation feature importance horizontal bar chart."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    sorted_idx = np.argsort(importances)
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, np.array(importances)[sorted_idx], color="#4c72b0", align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(np.array(feature_names)[sorted_idx])
    ax.set_title("Permutation Feature Importance (Increase in Test RMSE)", fontweight="bold")
    ax.set_xlabel("Delta RMSE (MW)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
