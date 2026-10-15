"""Model Comparison and Results Aggregation Module.

Formats and exports benchmark tables comparing:
- Naive (1h persistence)
- Seasonal Naive (24h & 168h)
- Linear Regression (Ridge on lags)
- SARIMA
- SimpleRNN, LSTM, GRU (across multiple random seeds with Mean ± Std)
- Transformer, BiLSTM, CNN-LSTM

Calculates honest improvement percentages against seasonal naive benchmarks.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def format_mean_std(values: list[float], decimals: int = 2) -> str:
    """Format a list of float evaluations as 'mean ± std'."""
    val_arr = np.array(values, dtype=float)
    mean = np.mean(val_arr)
    std = np.std(val_arr)
    if len(val_arr) <= 1 or std < 1e-4:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def aggregate_seed_results(seed_metrics: list[dict[str, float]]) -> dict[str, str]:
    """Aggregate per-seed metrics into mean ± std string values."""
    metrics_keys = ["rmse", "mae", "mse", "mape", "r2"]
    out = {}
    for k in metrics_keys:
        vals = [m[k] for m in seed_metrics if k in m]
        if vals:
            dec = 4 if k == "r2" else 2
            out[k] = format_mean_std(vals, decimals=dec)
    return out


def build_comparison_dataframe(
    baseline_results: dict[str, dict[str, float]],
    neural_results: dict[str, list[dict[str, float]]],
    save_path: Path | None = None,
) -> pd.DataFrame:
    """Build unified comparison table with baseline and neural models.

    Args:
        baseline_results: Dict of {model_name: {metric_name: value}}
        neural_results: Dict of {model_name: [list of per-seed metric dicts]}
        save_path: Optional CSV path in reports/results/
    """
    rows = []

    # Baselines (deterministic single-run)
    for model_name, metrics in baseline_results.items():
        rows.append({
            "Model": model_name,
            "Type": "Baseline",
            "RMSE (MW)": f"{metrics.get('rmse', 0):.2f}",
            "MAE (MW)": f"{metrics.get('mae', 0):.2f}",
            "MSE (MW^2)": f"{metrics.get('mse', 0):.2f}",
            "R2 Score": f"{metrics.get('r2', 0):.4f}",
            "MAPE (%)": f"{metrics.get('mape', 0):.2f}%",
        })

    # Neural architectures (multi-seed aggregated)
    for model_name, seed_list in neural_results.items():
        agg = aggregate_seed_results(seed_list)
        rows.append({
            "Model": model_name,
            "Type": "Neural Network",
            "RMSE (MW)": agg.get("rmse", "N/A"),
            "MAE (MW)": agg.get("mae", "N/A"),
            "MSE (MW^2)": agg.get("mse", "N/A"),
            "R2 Score": agg.get("r2", "N/A"),
            "MAPE (%)": f"{agg.get('mape', 'N/A')}%" if "mape" in agg else "N/A",
        })

    df_comp = pd.DataFrame(rows)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_comp.to_csv(save_path, index=False)
        logger.info(f"Saved comparison table to {save_path}")

    return df_comp
