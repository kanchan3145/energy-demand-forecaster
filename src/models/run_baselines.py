"""Run and export baseline benchmark results."""

import json
from pathlib import Path

import pandas as pd
import yaml

from src.models.baselines import evaluate_all_baselines


def main():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    proc_path = cfg["paths"]["spain_processed"]
    df = pd.read_parquet(proc_path)
    split_dates = cfg["dataset"]["split_dates"]

    df_train = df.loc[df.index <= pd.to_datetime(split_dates["train_end"], utc=True)]
    df_test = df.loc[df.index >= pd.to_datetime(split_dates["test_start"], utc=True)]

    print(f"Train size: {len(df_train)} hours, Test size: {len(df_test)} hours")

    results = evaluate_all_baselines(df_train, df_test)

    print("\n" + "=" * 70)
    print("BASELINE MODEL EVALUATION ON UNSEEN TEST SET:")
    print(f"{'Model':<26} {'RMSE (MW)':<12} {'MAE (MW)':<12} {'MAPE (%)':<10} {'R2 Score':<10}")
    print("-" * 70)
    for model_name, m in results.items():
        print(f"{model_name:<26} {m['rmse']:<12.2f} {m['mae']:<12.2f} {m['mape']:<10.2f} {m['r2']:<10.4f}")
    print("=" * 70 + "\n")

    res_dir = Path(cfg["paths"]["reports_results_dir"])
    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / "baselines_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
