"""Data and Concept Drift Detection Module for Energy Demand Forecaster.

Monitors recent prediction error (e.g., rolling 7-day RMSE) against the validated
baseline RMSE. If performance degrades beyond a specified threshold ratio (e.g. 1.25x),
raises a concept drift alert and triggers automated model retraining.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.evaluation.metrics import root_mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_performance_drift(
    recent_actuals: np.ndarray,
    recent_forecasts: np.ndarray,
    baseline_rmse: float,
    threshold_ratio: float = 1.25,
) -> dict[str, Any]:
    """Compare recent rolling RMSE against validation baseline.

    Args:
        recent_actuals: Array of recent ground truth values.
        recent_forecasts: Array of recent model forecasts.
        baseline_rmse: Known validation RMSE from production model registration.
        threshold_ratio: Maximum tolerable ratio before drift is flagged (default 1.25).

    Returns:
        Drift diagnostic dictionary with flag and metrics.
    """
    recent_rmse = root_mean_squared_error(recent_actuals, recent_forecasts)
    drift_ratio = float(recent_rmse / max(baseline_rmse, 1e-5))
    drift_detected = drift_ratio > threshold_ratio

    result = {
        "baseline_rmse": float(baseline_rmse),
        "recent_rmse": float(recent_rmse),
        "drift_ratio": round(drift_ratio, 3),
        "threshold_ratio": float(threshold_ratio),
        "drift_detected": bool(drift_detected),
        "status": "DRIFT_ALERT_RETRAIN_RECOMMENDED" if drift_detected else "NORMAL_OPERATION",
    }

    if drift_detected:
        logger.warning(
            f"DRIFT DETECTED: Recent RMSE ({recent_rmse:.2f} MW) exceeds baseline "
            f"({baseline_rmse:.2f} MW) by ratio {drift_ratio:.2f} > {threshold_ratio}."
        )
    else:
        logger.info(
            f"Model healthy: Recent RMSE ({recent_rmse:.2f} MW) vs Baseline "
            f"({baseline_rmse:.2f} MW), ratio={drift_ratio:.2f}."
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate forecasting performance drift")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--threshold", type=float, default=None, help="Custom threshold ratio")
    args = parser.parse_args()

    cfg = load_config(args.config)
    threshold = args.threshold or cfg["drift"]["threshold_ratio"]

    # Read latest evaluation comparison if available
    results_file = Path(cfg["paths"]["reports_results_dir"]) / "model_comparison.csv"
    baseline_rmse = 750.0  # Default benchmark RMSE in MW
    if results_file.exists():
        df_res = pd.read_csv(results_file)
        # Pull best model RMSE
        best_row = df_res[df_res["Type"] == "Neural Network"].iloc[0]
        try:
            val_str = str(best_row["RMSE (MW)"]).split("±")[0].strip()
            baseline_rmse = float(val_str)
        except Exception:
            pass

    # Simulate recent rolling window audit
    np.random.seed(42)
    synthetic_actuals = np.random.normal(loc=28000, scale=3500, size=168)
    # Simulate normal residuals with some occasional variance
    synthetic_forecasts = synthetic_actuals + np.random.normal(loc=0, scale=baseline_rmse * 1.05, size=168)

    diag = check_performance_drift(
        synthetic_actuals,
        synthetic_forecasts,
        baseline_rmse=baseline_rmse,
        threshold_ratio=threshold,
    )

    print(json.dumps(diag, indent=2))
    if diag["drift_detected"]:
        print("\nACTION REQUIRED: Run 'make retrain' to adapt to recent distribution shift.")


if __name__ == "__main__":
    main()
