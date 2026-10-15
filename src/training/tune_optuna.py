"""Optuna Hyperparameter Tuning for RNN/LSTM/GRU Architectures.

Searches window size, layer units, dropout, learning rate, and batch size
using Tree-structured Parzen Estimator (TPE).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
import tensorflow as tf

from src.data.windowing import create_sliding_windows
from src.evaluation.metrics import root_mean_squared_error
from src.models.rnn_models import build_rnn_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress overly verbose optuna output
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_optuna_study(
    train_data: np.ndarray,
    val_data: np.ndarray,
    target_scaler,
    cell_type: str = "LSTM",
    n_trials: int = 15,
    epochs_per_trial: int = 8,
    seed: int = 42,
) -> tuple[optuna.Study, dict[str, Any]]:
    """Run an Optuna Bayesian hyperparameter optimization study."""
    np.random.seed(seed)
    tf.random.set_seed(seed)

    def objective(trial: optuna.Trial) -> float:
        # Hyperparameter search space
        window_size = trial.suggest_categorical("window_size", [12, 24, 48])
        units_l1 = trial.suggest_categorical("units_l1", [32, 64, 96])
        units_l2 = trial.suggest_categorical("units_l2", [16, 32, 64])
        dropout = trial.suggest_float("dropout", 0.1, 0.4, step=0.1)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64])

        X_tr, y_tr = create_sliding_windows(train_data, target_idx=0, window_size=window_size, horizon=1)
        X_va, y_va = create_sliding_windows(val_data, target_idx=0, window_size=window_size, horizon=1)

        sample_limit = 10000
        if len(X_tr) > sample_limit:
            X_tr, y_tr = X_tr[-sample_limit:], y_tr[-sample_limit:]

        model = build_rnn_model(
            cell_type=cell_type,
            input_shape=(window_size, train_data.shape[1]),
            layer_units=[units_l1, units_l2],
            dropout_rate=dropout,
            learning_rate=learning_rate,
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ]

        model.fit(
            X_tr,
            y_tr,
            validation_data=(X_va, y_va),
            epochs=epochs_per_trial,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        preds_scaled = model.predict(X_va, verbose=0)
        preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
        actuals = target_scaler.inverse_transform(y_va.reshape(-1, 1)).ravel()

        rmse = root_mean_squared_error(actuals, preds)
        return rmse

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    logger.info(f"Starting Optuna search ({n_trials} trials)...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"Optuna Best Trial: Value={study.best_value:.2f} MW, Params={study.best_params}")
    return study, study.best_params
