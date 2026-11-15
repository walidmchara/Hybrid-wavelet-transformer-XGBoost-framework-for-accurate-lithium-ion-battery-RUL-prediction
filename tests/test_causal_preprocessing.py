"""Regression tests for past-only wavelet and train-fitted preprocessing."""
import numpy as np
import pandas as pd

from wtx_cbo.core import (dwt_multiscale_same_length, clean_numeric_frame,
                          fit_iqr_bounds)


def test_dwt_output_shape_and_finite_values():
    x = np.linspace(0, 1, 42)[:, None]
    z = dwt_multiscale_same_length(x, wavelet="db4", level=4)
    assert z.shape == (42, 5)
    assert np.isfinite(z).all()


def test_wavelet_is_causal_to_future_measurements():
    x = np.arange(48, dtype=float).reshape(-1, 1)
    y = x.copy()
    y[30:] = 9999.0
    a = dwt_multiscale_same_length(x, "db4", 4)
    b = dwt_multiscale_same_length(y, "db4", 4)
    np.testing.assert_allclose(a[:30], b[:30], atol=1e-10)


def test_preprocessing_does_not_use_future():
    x = pd.DataFrame({"voltage": [4., np.nan, 3.8, 3.7, 3.6, 3.5]})
    y = x.copy()
    y.loc[4:, "voltage"] = 1e6
    bounds = {"voltage": (0., 5.)}
    a = clean_numeric_frame(x, ["voltage"], bounds=bounds)
    b = clean_numeric_frame(y, ["voltage"], bounds=bounds)
    np.testing.assert_allclose(a["voltage"].iloc[:4], b["voltage"].iloc[:4])


def test_iqr_fitted_to_prefix_only():
    train = pd.DataFrame({"v": [1., 1.1, 1.2, 1.3, 1.4]})
    bounds = fit_iqr_bounds(train, ["v"])
    altered_test = pd.DataFrame({"v": [1000.]})
    assert bounds == fit_iqr_bounds(train, ["v"])
    assert clean_numeric_frame(altered_test, ["v"], bounds=bounds)["v"].iloc[0] < 10
