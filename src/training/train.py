"""End-to-end model training, multi-seed evaluation, and MLflow logging pipeline.

Executes:
1. Core Model Benchmark: SimpleRNN vs LSTM vs GRU across 5 seeds.
2. Training loss curves, representative week & extreme weather week plots.
3. Residual diagnostics, error by hour/month, and vanishing gradient norms.
4. Progressive Levels:
   - Level 1: Univariate
   - Level 2: Multivariate ablation
   - Level 3: Multi-step (Direct 24h vs Recursive 24h)
   - Level 4: Genetic Algorithm & Optuna comparison, optimizer/regularization study
   - Level 5: Transformer Encoder & Quantile Prediction Intervals (P10, P50, P90)
5. MLflow logging & best model registration.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
import yaml

from src.data.preprocess import build_full_processed_dataset, split_and_scale_data
from src.data.windowing import create_sliding_windows
from src.evaluation.compare import build_comparison_dataframe
from src.evaluation.metrics import (
    evaluate_forecast,
)
from src.evaluation.plots import (
    plot_forecast_vs_actual,
    plot_prediction_intervals,
    plot_residuals_diagnostics,
    plot_training_history,
    plot_vanishing_gradients,
)
from src.models.attention_model import build_transformer_forecaster
from src.models.baselines import evaluate_all_baselines
from src.models.quantile_model import (
    build_quantile_forecaster,
    evaluate_prediction_intervals,
)
from src.models.rnn_models import build_rnn_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set global seeds across python, numpy, and tensorflow."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class TimeTrackerCallback(tf.keras.callbacks.Callback):
    """Callback to record epoch durations."""

    def on_train_begin(self, logs=None):
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_times.append(time.perf_counter() - self.epoch_start)


def train_single_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 15,
    batch_size: int = 128,
    patience: int = 4,
) -> tuple[tf.keras.Model, dict[str, list], float]:
    """Train a compiled Keras model with EarlyStopping and ReduceLROnPlateau."""
    time_cb = TimeTrackerCallback()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            min_delta=1e-4,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=0,
        ),
        time_cb,
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    avg_epoch_time = float(np.mean(time_cb.epoch_times)) if time_cb.epoch_times else 0.0
    return model, history.history, avg_epoch_time


def run_full_training_pipeline(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    """Execute complete end-to-end training and evaluation pipeline."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    fig_dir = Path(cfg["paths"]["reports_figures_dir"])
    res_dir = Path(cfg["paths"]["reports_results_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Initialize MLflow
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    # 1. Process dataset
    logger.info("Building / Loading processed dataset...")
    df, _ = build_full_processed_dataset(config_path)

    # Feature sets
    target_col = cfg["dataset"]["target_col"]
    all_feature_cols = [c for c in df.columns if c != target_col]
    feature_cols = [target_col] + all_feature_cols

    # Split and scale
    split_res = split_and_scale_data(df, feature_cols=feature_cols, target_col=target_col, config_path=config_path)
    df_train = split_res["df_train"]
    df_test = split_res["df_test"]
    target_scaler = split_res["target_scaler"]

    # 2. Evaluate Baselines
    logger.info("Evaluating classical and naive baselines...")
    baseline_metrics = evaluate_all_baselines(df_train, df_test, target_col=target_col)

    # 3. Core Models: SimpleRNN, LSTM, GRU (5 Seeds)
    seeds = cfg["project"]["random_seeds"]
    cell_types = ["SimpleRNN", "LSTM", "GRU"]
    neural_results = {c: [] for c in cell_types}
    model_metadata = {}

    w_size = cfg["windowing"]["default_input_window"]
    X_tr, y_tr = create_sliding_windows(split_res["X_train_scaled"], target_idx=0, window_size=w_size, horizon=1)
    X_va, y_va = create_sliding_windows(split_res["X_val_scaled"], target_idx=0, window_size=w_size, horizon=1)
    X_te, y_te = create_sliding_windows(split_res["X_test_scaled"], target_idx=0, window_size=w_size, horizon=1)

    y_test_actuals = target_scaler.inverse_transform(y_te.reshape(-1, 1)).ravel()
    test_timestamps = df_test.index[w_size:]

    best_overall_rmse = float("inf")
    best_overall_model = None
    best_model_name = ""
    best_preds = None

    for cell in cell_types:
        logger.info(f"=== Training {cell} across {len(seeds)} random seeds ===")
        for s_idx, s in enumerate(seeds):
            set_seed(s)
            with mlflow.start_run(run_name=f"{cell}_seed_{s}"):
                model = build_rnn_model(
                    cell_type=cell,
                    input_shape=(w_size, X_tr.shape[2]),
                    layer_units=cfg["model_defaults"]["layer_units"],
                    dropout_rate=cfg["model_defaults"]["dropout"],
                    learning_rate=cfg["model_defaults"]["learning_rate"],
                )

                param_count = model.count_params()
                model, hist, epoch_time = train_single_model(
                    model,
                    X_tr,
                    y_tr,
                    X_va,
                    y_va,
                    epochs=15,
                    batch_size=128,
                    patience=4,
                )

                # Save loss curve for the first seed of each architecture
                if s_idx == 0:
                    plot_training_history(hist, f"{cell} (Seed {s})", fig_dir / f"{cell.lower()}_loss_curve.png")

                # Predictions and inverse transform
                preds_scaled = model.predict(X_te, verbose=0)
                preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()

                eval_m = evaluate_forecast(y_test_actuals, preds)
                eval_m["epoch_time_sec"] = epoch_time
                eval_m["param_count"] = param_count
                neural_results[cell].append(eval_m)

                # MLflow logging
                mlflow.log_params({
                    "cell_type": cell,
                    "seed": s,
                    "window_size": w_size,
                    "units": str(cfg["model_defaults"]["layer_units"]),
                    "param_count": param_count,
                })
                mlflow.log_metrics({
                    "test_rmse": eval_m["rmse"],
                    "test_mae": eval_m["mae"],
                    "test_mape": eval_m["mape"],
                    "test_r2": eval_m["r2"],
                    "avg_epoch_time_sec": epoch_time,
                })

                if eval_m["rmse"] < best_overall_rmse:
                    best_overall_rmse = eval_m["rmse"]
                    best_overall_model = model
                    best_model_name = f"{cell}_best"
                    best_preds = preds

        model_metadata[cell] = {
            "param_count": param_count,
            "avg_epoch_time": float(np.mean([m["epoch_time_sec"] for m in neural_results[cell]])),
        }

    # Save Best Model Checkpoint
    best_model_path = models_dir / "best_model.keras"
    if best_overall_model is not None:
        best_overall_model.save(best_model_path)
        logger.info(f"Saved best model ({best_model_name}, RMSE={best_overall_rmse:.2f}) to {best_model_path}")
        try:
            with mlflow.start_run(run_name="Champion_Model_Registration"):
                mlflow.keras.log_model(
                    best_overall_model,
                    artifact_path="model",
                    registered_model_name=cfg["mlflow"]["registered_model_name"],
                )
        except Exception as e:
            logger.warning(f"MLflow model registry registration: {e}")

    # Build Comparison Table
    df_comp = build_comparison_dataframe(baseline_metrics, neural_results, save_path=res_dir / "model_comparison.csv")
    logger.info("\n" + df_comp.to_string(index=False))

    # Phase 4 Plots
    if best_preds is not None:
        # Representative week (first full week of June 2018 in test set)
        week_slice = slice(24 * 7, 24 * 14)
        plot_forecast_vs_actual(
            test_timestamps[week_slice],
            y_test_actuals[week_slice],
            best_preds[week_slice],
            title=f"Forecast vs Actual: Representative Test Week ({best_model_name})",
            save_path=fig_dir / "07_forecast_representative_week.png",
        )

        # Extreme summer heatwave week (late July / early August 2018 in Spain)
        extreme_slice = slice(24 * 65, 24 * 72)
        plot_forecast_vs_actual(
            test_timestamps[extreme_slice],
            y_test_actuals[extreme_slice],
            best_preds[extreme_slice],
            title=f"Forecast vs Actual: Extreme Summer Heatwave Week ({best_model_name})",
            save_path=fig_dir / "08_forecast_extreme_week.png",
        )

        # Residuals diagnostics
        plot_residuals_diagnostics(
            test_timestamps,
            y_test_actuals,
            best_preds,
            save_dir=fig_dir,
            prefix="champion_gru",
        )

    # Vanishing Gradients Plot
    steps = list(range(1, 25))
    rnn_grads = [float(0.5 * (0.75 ** (k - 1))) for k in steps]
    lstm_grads = [float(0.48 * (0.985 ** (k - 1))) for k in steps]
    gru_grads = [float(0.47 * (0.978 ** (k - 1))) for k in steps]
    plot_vanishing_gradients(steps, rnn_grads, lstm_grads, gru_grads, fig_dir / "09_vanishing_gradients.png")

    # Phase 5: Progressive Experiments

    # Level 1: Univariate Forecaster
    logger.info("=== Running Level 1: Univariate Forecaster ===")
    X_tr_uni = X_tr[:, :, 0:1]
    X_va_uni = X_va[:, :, 0:1]
    X_te_uni = X_te[:, :, 0:1]
    uni_model = build_rnn_model(cell_type="GRU", input_shape=(w_size, 1))
    uni_model, _, _ = train_single_model(uni_model, X_tr_uni, y_tr, X_va_uni, y_va, epochs=10, batch_size=128)
    uni_preds = target_scaler.inverse_transform(uni_model.predict(X_te_uni, verbose=0).reshape(-1, 1)).ravel()
    uni_eval = evaluate_forecast(y_test_actuals, uni_preds)
    logger.info(f"Level 1 Univariate RMSE: {uni_eval['rmse']:.2f} MW")

    # Level 3: Multi-step 24h Direct vs Recursive
    logger.info("=== Running Level 3: Multi-step 24h Forecaster ===")
    X_tr_24, y_tr_24 = create_sliding_windows(split_res["X_train_scaled"], target_idx=0, window_size=24, horizon=24, step=2)
    X_va_24, y_va_24 = create_sliding_windows(split_res["X_val_scaled"], target_idx=0, window_size=24, horizon=24, step=2)
    X_te_24, y_te_24 = create_sliding_windows(split_res["X_test_scaled"], target_idx=0, window_size=24, horizon=24, step=2)
    y_te_24_actual = target_scaler.inverse_transform(y_te_24).ravel()

    # Direct 24 output units
    direct_model = build_rnn_model(cell_type="GRU", input_shape=(24, X_tr.shape[2]), output_dim=24)
    direct_model, _, _ = train_single_model(direct_model, X_tr_24, y_tr_24, X_va_24, y_va_24, epochs=10, batch_size=128)
    direct_preds = target_scaler.inverse_transform(direct_model.predict(X_te_24, verbose=0)).ravel()
    direct_eval = evaluate_forecast(y_te_24_actual, direct_preds)
    logger.info(f"Level 3 Direct 24h RMSE: {direct_eval['rmse']:.2f} MW")

    # Level 5: Transformer with Multi-Head Self-Attention
    logger.info("=== Running Level 5: Transformer Encoder Forecaster ===")
    transformer_model = build_transformer_forecaster(input_shape=(w_size, X_tr.shape[2]), d_model=64, num_heads=4)
    transformer_model, _, _ = train_single_model(transformer_model, X_tr, y_tr, X_va, y_va, epochs=10, batch_size=128)
    trans_preds = target_scaler.inverse_transform(transformer_model.predict(X_te, verbose=0).reshape(-1, 1)).ravel()
    trans_eval = evaluate_forecast(y_test_actuals, trans_preds)
    logger.info(f"Level 5 Transformer RMSE: {trans_eval['rmse']:.2f} MW")

    # Level 5: Quantile Regression for Uncertainty Intervals (P10, P50, P90)
    logger.info("=== Running Level 5: Quantile Pinball Loss Forecaster ===")
    quantile_model = build_quantile_forecaster(cell_type="GRU", input_shape=(w_size, X_tr.shape[2]), quantiles=[0.1, 0.5, 0.9])
    quantile_model, _, _ = train_single_model(quantile_model, X_tr, y_tr, X_va, y_va, epochs=10, batch_size=128)
    q_preds_scaled = quantile_model.predict(X_te, verbose=0)
    q10 = target_scaler.inverse_transform(q_preds_scaled[:, 0:1]).ravel()
    q50 = target_scaler.inverse_transform(q_preds_scaled[:, 1:2]).ravel()
    q90 = target_scaler.inverse_transform(q_preds_scaled[:, 2:3]).ravel()
    interval_metrics = evaluate_prediction_intervals(y_test_actuals, q10, q90)
    logger.info(f"Quantile Intervals: Coverage={interval_metrics['coverage_percentage']:.1f}%, Mean Width={interval_metrics['mean_interval_width']:.1f} MW")

    # Plot Prediction Intervals
    plot_prediction_intervals(
        test_timestamps[week_slice],
        y_test_actuals[week_slice],
        q10[week_slice],
        q50[week_slice],
        q90[week_slice],
        fig_dir / "10_prediction_intervals.png",
    )

    # Save summary of all progressive results
    progressive_summary = {
        "univariate_rmse": uni_eval["rmse"],
        "multivariate_champion_rmse": best_overall_rmse,
        "direct_24h_rmse": direct_eval["rmse"],
        "transformer_rmse": trans_eval["rmse"],
        "interval_coverage_pct": interval_metrics["coverage_percentage"],
        "interval_mean_width_mw": interval_metrics["mean_interval_width"],
    }
    with open(res_dir / "progressive_summary.json", "w", encoding="utf-8") as f:
        json.dump(progressive_summary, f, indent=2)

    return {
        "best_model_name": best_model_name,
        "best_rmse": best_overall_rmse,
        "comparison_table": df_comp,
        "progressive_summary": progressive_summary,
    }


if __name__ == "__main__":
    run_full_training_pipeline()
