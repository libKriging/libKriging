"""Tests for gradient-enhanced kriging (fit(..., dydX=...))."""

import numpy as np
import pytest

import pylibkriging as lk


def f(x):
    return np.sin(3.0 * x[0]) + np.cos(5.0 * x[1])


def df(x):
    return np.array([3.0 * np.cos(3.0 * x[0]), -5.0 * np.sin(5.0 * x[1])])


def make_design(n, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 2))
    y = np.array([f(x) for x in X])
    dy = np.array([df(x) for x in X])
    return X, y, dy


def test_dydX_interpolates_values_and_gradients():
    X, y, dy = make_design(20)
    k = lk.Kriging(y, X, "gauss", dydX=dy)

    assert k.dy().shape == (20, 2)

    mean, stdev, cov, mean_deriv, stdev_deriv = k.predict(X, True, False, True)
    assert np.abs(mean.flatten() - y).max() < 1e-4
    assert np.abs(mean_deriv - dy).max() < 1e-3


def test_dydX_none_is_a_value_only_fit():
    X, y, _dy = make_design(20)
    k = lk.Kriging(y, X, "gauss")
    assert k.dy().size == 0


def test_dydX_beats_value_only_out_of_sample():
    X, y, dy = make_design(15, seed=72)
    X_test, y_test, _ = make_design(200, seed=720)

    k_plain = lk.Kriging(y, X, "gauss")
    k_grad = lk.Kriging(y, X, "gauss", dydX=dy)

    mean_plain, *_ = k_plain.predict(X_test)
    mean_grad, *_ = k_grad.predict(X_test)

    rmse_plain = np.sqrt(np.mean((mean_plain.flatten() - y_test) ** 2))
    rmse_grad = np.sqrt(np.mean((mean_grad.flatten() - y_test) ** 2))
    assert rmse_grad < rmse_plain


def test_dydX_via_fit_method():
    X, y, dy = make_design(20)
    k = lk.Kriging("gauss")
    k.fit(y, X, dydX=dy)
    assert not k.dy().size == 0

    # A later fit() without dydX clears the gradient observations.
    k.fit(y, X)
    assert k.dy().size == 0


def test_dydX_rejects_non_differentiable_kernel():
    X, y, dy = make_design(10)
    with pytest.raises(RuntimeError):
        lk.Kriging(y, X, "exp", dydX=dy)


def test_dydX_rejects_wrong_shape():
    X, y, dy = make_design(10)
    with pytest.raises(RuntimeError):
        lk.Kriging(y, X, "gauss", dydX=dy[:, :1])
