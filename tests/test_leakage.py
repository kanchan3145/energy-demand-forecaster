"""Unit tests verifying strict absence of data leakage."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.windowing import get_window_timestamps
from src.features.lag_features import add_lag_features, add_rolling_features
from src.features.time_features import add_cyclical_time_features


def test_scaler_no_leakage():
    """Verify scalers are strictly fitted on train data statistics only."""
    np.random.seed(42)
    train_data = np.random.normal(loc=25000.0, scale=3000.0, size=(1000, 3))
    test_data = np.random.normal(loc=35000.0, scale=4000.0, size=(300, 3))

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    assert train_scaled.shape == (1000, 3)

    # Scaler mean and variance must exactly match train statistics
    np.testing.assert_allclose(scaler.mean_, train_data.mean(axis=0), rtol=1e-5)
    np.testing.assert_allclose(scaler.var_, train_data.var(axis=0), rtol=1e-5)

    # Scaler must transform test data without adapting to test distribution
    test_scaled = scaler.transform(test_data)
    expected_test_scaled = (test_data - scaler.mean_) / np.sqrt(scaler.var_)
    np.testing.assert_allclose(test_scaled, expected_test_scaled, rtol=1e-5)


def test_no_test_timestamps_in_train_windows():
    """Verify that no test timestamp appears in train input windows or target horizons."""
    dates = pd.date_range("2017-01-01", periods=1000, freq="1h")
    train_dates = dates[:700]
    test_dates = dates[700:]

    window_size = 24
    horizon = 24

    train_X_times, train_y_times = get_window_timestamps(
        train_dates.to_series(), window_size=window_size, horizon=horizon
    )
    test_X_times, test_y_times = get_window_timestamps(
        test_dates.to_series(), window_size=window_size, horizon=horizon
    )

    # Gather all train timestamps (both inputs and targets)
    all_train_timestamps = set()
    for (start, end) in train_X_times:
        all_train_timestamps.update(pd.date_range(start, end, freq="1h"))
    for (start, end) in train_y_times:
        all_train_timestamps.update(pd.date_range(start, end, freq="1h"))

    # Gather all test timestamps
    all_test_timestamps = set()
    for (start, end) in test_X_times:
        all_test_timestamps.update(pd.date_range(start, end, freq="1h"))
    for (start, end) in test_y_times:
        all_test_timestamps.update(pd.date_range(start, end, freq="1h"))

    # Strict disjointness: No test timestamp can exist in train
    overlap = all_train_timestamps.intersection(all_test_timestamps)
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping timestamps between train and test: {overlap}"


def test_lag_features_no_lookahead():
    """Verify that lag features strictly pull from the past."""
    df = pd.DataFrame({"total_load_actual": [10.0 * i for i in range(100)]})
    df_lagged = add_lag_features(df, target_col="total_load_actual", lags=[1, 24])

    # Lag 1 at index 10 should equal target at index 9
    assert df_lagged.loc[10, "total_load_actual_lag_1"] == df.loc[9, "total_load_actual"]
    # Lag 24 at index 30 should equal target at index 6
    assert df_lagged.loc[30, "total_load_actual_lag_24"] == df.loc[6, "total_load_actual"]
    # Lag 24 at index 0..23 should be NaN (warmup)
    assert df_lagged.loc[23, "total_load_actual_lag_24"] is np.nan or pd.isna(df_lagged.loc[23, "total_load_actual_lag_24"])


def test_rolling_features_no_lookahead():
    """Verify rolling statistics do not include current time step t."""
    df = pd.DataFrame({"total_load_actual": [float(i) for i in range(50)]})
    df_roll = add_rolling_features(df, target_col="total_load_actual", windows=[5], min_shift=1)

    # At row 10, rolling mean over 5 elements with min_shift=1 must average rows [5, 6, 7, 8, 9]
    expected_mean = np.mean([5.0, 6.0, 7.0, 8.0, 9.0])
    np.testing.assert_allclose(df_roll.loc[10, "total_load_actual_roll_mean_5"], expected_mean)


def test_cyclical_encoding_bounds():
    """Verify trigonometric sin and cos encodings lie within [-1.0, 1.0]."""
    dates = pd.date_range("2018-01-01", periods=8760, freq="1h")
    df = pd.DataFrame(index=dates)
    df_enc = add_cyclical_time_features(df)

    for col in ["hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos", "month_sin", "month_cos"]:
        assert df_enc[col].min() >= -1.0 - 1e-6
        assert df_enc[col].max() <= 1.0 + 1e-6
