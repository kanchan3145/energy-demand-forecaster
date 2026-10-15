"""Phase 1 EDA and Exploratory Analysis Execution Script.

Generates:
1. Continuous hourly aligned dataset with time-interpolated load.
2. Exploratory time series figures saved to reports/figures/:
   - 01_full_series.png
   - 02_zoomed_week.png
   - 03_zoomed_day.png
   - 04_diurnal_weekly_monthly_profiles.png
   - 05_seasonal_decomposition.png
   - 06_acf_pacf.png
3. EDA quantitative summary saved to reports/results/eda_summary.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from src.data.preprocess import clean_energy_data, clean_weather_data
from src.evaluation.plots import plot_eda_series

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_phase1_eda(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_energy = Path(cfg["paths"]["spain_energy_raw"])
    raw_weather = Path(cfg["paths"]["spain_weather_raw"])
    figures_dir = Path(cfg["paths"]["reports_figures_dir"])
    results_dir = Path(cfg["paths"]["reports_results_dir"])

    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Phase 1 EDA & Data Understanding...")
    energy_df = clean_energy_data(raw_energy)
    weather_df = clean_weather_data(raw_weather)

    target_col = cfg["dataset"]["target_col"]
    df = energy_df.join(weather_df, how="inner")

    # Generate figures
    logger.info(f"Generating EDA plots into {figures_dir}...")
    saved_plots = plot_eda_series(df, target_col=target_col, save_dir=figures_dir)

    # Compute key quantitative metrics
    summary = {
        "start_time": str(df.index.min()),
        "end_time": str(df.index.max()),
        "total_hourly_records": len(df),
        "mean_demand_mw": float(df[target_col].mean()),
        "std_demand_mw": float(df[target_col].std()),
        "min_demand_mw": float(df[target_col].min()),
        "max_demand_mw": float(df[target_col].max()),
        "weekday_mean_demand_mw": float(df[df.index.dayofweek < 5][target_col].mean()),
        "weekend_mean_demand_mw": float(df[df.index.dayofweek >= 5][target_col].mean()),
        "weekend_drop_percentage": float(
            (1.0 - df[df.index.dayofweek >= 5][target_col].mean() / df[df.index.dayofweek < 5][target_col].mean()) * 100.0
        ),
        "peak_hour": int(df.groupby(df.index.hour)[target_col].mean().idxmax()),
        "peak_hour_mean_mw": float(df.groupby(df.index.hour)[target_col].mean().max()),
        "trough_hour": int(df.groupby(df.index.hour)[target_col].mean().idxmin()),
        "trough_hour_mean_mw": float(df.groupby(df.index.hour)[target_col].mean().min()),
        "mean_temperature_celsius": float(df["temp_mean"].mean()) if "temp_mean" in df.columns else None,
        "saved_figure_files": saved_plots,
    }

    summary_file = results_dir / "eda_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Phase 1 Complete. Summary written to {summary_file}")
    print("\n" + "=" * 50)
    print("PHASE 1 SUMMARY STATISTICS:")
    print(f"  Total Records: {summary['total_hourly_records']} hours ({summary['start_time'][:10]} to {summary['end_time'][:10]})")
    print(f"  Mean Demand:   {summary['mean_demand_mw']:.1f} MW (Std: {summary['std_demand_mw']:.1f} MW)")
    print(f"  Min / Max:     {summary['min_demand_mw']:.1f} MW / {summary['max_demand_mw']:.1f} MW")
    print(f"  Weekday Mean:  {summary['weekday_mean_demand_mw']:.1f} MW")
    print(f"  Weekend Mean:  {summary['weekend_mean_demand_mw']:.1f} MW (Drop: -{summary['weekend_drop_percentage']:.1f}%)")
    print(f"  Diurnal Peak:  Hour {summary['peak_hour']:02d}:00 ({summary['peak_hour_mean_mw']:.1f} MW)")
    print(f"  Diurnal Min:   Hour {summary['trough_hour']:02d}:00 ({summary['trough_hour_mean_mw']:.1f} MW)")
    print("=" * 50 + "\n")

    return summary


if __name__ == "__main__":
    run_phase1_eda()
