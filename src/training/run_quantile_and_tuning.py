"""Execute remaining progressive levels: Quantile Uncertainty, GA vs Optuna, and Explainability."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from src.data.preprocess import build_full_processed_dataset, split_and_scale_data
from src.data.windowing import create_sliding_windows
from src.evaluation.metrics import root_mean_squared_error
from src.evaluation.plots import plot_feature_importance, plot_prediction_intervals
from src.models.quantile_model import (
    build_quantile_forecaster,
    evaluate_prediction_intervals,
)
from src.training.train import train_single_model
from src.training.tune_genetic import GeneticAlgorithmTuner
from src.training.tune_optuna import run_optuna_study

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    fig_dir = Path(cfg["paths"]["reports_figures_dir"])
    res_dir = Path(cfg["paths"]["reports_results_dir"])

    df, _ = build_full_processed_dataset()
    target_col = cfg["dataset"]["target_col"]
    all_feature_cols = [c for c in df.columns if c != target_col]
    feature_cols = [target_col] + all_feature_cols

    split_res = split_and_scale_data(df, feature_cols=feature_cols, target_col=target_col)
    df_test = split_res["df_test"]
    target_scaler = split_res["target_scaler"]

    w_size = cfg["windowing"]["default_input_window"]
    X_tr, y_tr = create_sliding_windows(split_res["X_train_scaled"], target_idx=0, window_size=w_size, horizon=1)
    X_va, y_va = create_sliding_windows(split_res["X_val_scaled"], target_idx=0, window_size=w_size, horizon=1)
    X_te, y_te = create_sliding_windows(split_res["X_test_scaled"], target_idx=0, window_size=w_size, horizon=1)

    y_test_actuals = target_scaler.inverse_transform(y_te.reshape(-1, 1)).ravel()
    test_timestamps = df_test.index[w_size:]

    # 1. Quantile Forecaster (P10, P50, P90)
    logger.info("=== Running Quantile Forecaster (P10, P50, P90) ===")
    q_model = build_quantile_forecaster(cell_type="GRU", input_shape=(w_size, X_tr.shape[2]), quantiles=[0.1, 0.5, 0.9])
    q_model, _, _ = train_single_model(q_model, X_tr, y_tr, X_va, y_va, epochs=8, batch_size=128)
    q_preds_scaled = q_model.predict(X_te, verbose=0)
    q10 = target_scaler.inverse_transform(q_preds_scaled[:, 0:1]).ravel()
    q50 = target_scaler.inverse_transform(q_preds_scaled[:, 1:2]).ravel()
    q90 = target_scaler.inverse_transform(q_preds_scaled[:, 2:3]).ravel()

    interval_metrics = evaluate_prediction_intervals(y_test_actuals, q10, q90)
    logger.info(f"Quantile Intervals: Coverage={interval_metrics['coverage_percentage']:.1f}%, Mean Width={interval_metrics['mean_interval_width']:.1f} MW")

    # Plot Prediction Intervals for a representative test week
    week_slice = slice(24 * 7, 24 * 14)
    plot_prediction_intervals(
        test_timestamps[week_slice],
        y_test_actuals[week_slice],
        q10[week_slice],
        q50[week_slice],
        q90[week_slice],
        fig_dir / "10_prediction_intervals.png",
    )

    # 2. Hyperparameter Optimization: Optuna vs Genetic Algorithm
    logger.info("=== Running Optuna Hyperparameter Optimization ===")
    study, best_optuna_params = run_optuna_study(
        split_res["X_train_scaled"],
        split_res["X_val_scaled"],
        target_scaler,
        cell_type="GRU",
        n_trials=5,
        epochs_per_trial=4,
    )

    logger.info("=== Running From-Scratch Genetic Algorithm Tuner ===")
    ga_tuner = GeneticAlgorithmTuner(
        population_size=4,
        generations=2,
        cell_type="GRU",
        epochs_per_eval=4,
    )
    best_ga_ind, ga_history = ga_tuner.optimize(
        split_res["X_train_scaled"],
        split_res["X_val_scaled"],
        target_scaler,
    )

    # Save Tuning Comparison
    tuning_df = pd.DataFrame([
        {
            "Method": "Optuna (Bayesian TPE)",
            "Best Val RMSE (MW)": f"{study.best_value:.2f}",
            "Key Hyperparameters": str(best_optuna_params),
            "Search Paradigm": "Sequential density estimation via Parzen estimators",
        },
        {
            "Method": "Genetic Algorithm (Evolutionary)",
            "Best Val RMSE (MW)": f"{best_ga_ind.fitness:.2f}",
            "Key Hyperparameters": str(best_ga_ind.genes),
            "Search Paradigm": "Stochastic tournament selection, crossover & mutation",
        },
    ])
    tuning_df.to_csv(res_dir / "tuning_comparison.csv", index=False)
    logger.info("\n" + tuning_df.to_string(index=False))

    # 3. Permutation Feature Importance (Level 6 Explainability)
    logger.info("=== Computing Permutation Feature Importance ===")
    best_model_path = Path("models/best_model.keras")
    best_model = tf.keras.models.load_model(best_model_path)
    base_preds = target_scaler.inverse_transform(best_model.predict(X_te, verbose=0).reshape(-1, 1)).ravel()
    base_rmse = root_mean_squared_error(y_test_actuals, base_preds)

    feat_importances = []
    top_features = feature_cols[:10]  # Top 10 key features
    for idx, col in enumerate(top_features):
        X_te_perm = X_te.copy()
        # Permute feature values across samples
        np.random.seed(42)
        perm_idx = np.random.permutation(len(X_te_perm))
        X_te_perm[:, :, idx] = X_te_perm[perm_idx, :, idx]
        p_preds = target_scaler.inverse_transform(best_model.predict(X_te_perm, verbose=0).reshape(-1, 1)).ravel()
        p_rmse = root_mean_squared_error(y_test_actuals, p_preds)
        delta_rmse = max(0.0, p_rmse - base_rmse)
        feat_importances.append(delta_rmse)
        logger.info(f"Feature: {col:<25} Delta RMSE: +{delta_rmse:.2f} MW")

    plot_feature_importance(top_features, feat_importances, fig_dir / "11_feature_importance.png")

    # Export final progressive summary
    prog_summary = {
        "quantile_coverage_pct": interval_metrics["coverage_percentage"],
        "quantile_mean_width_mw": interval_metrics["mean_interval_width"],
        "optuna_best_val_rmse": study.best_value,
        "ga_best_val_rmse": best_ga_ind.fitness,
        "champion_test_rmse": base_rmse,
    }
    with open(res_dir / "progressive_summary.json", "w", encoding="utf-8") as f:
        json.dump(prog_summary, f, indent=2)

    logger.info("Quantile, Tuning, and Explainability runs completed successfully!")


if __name__ == "__main__":
    main()
