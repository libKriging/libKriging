"""Tests for WarpKriging Python binding.

Mirrors bindings/R/rlibkriging/tests/testthat/test-WarpKriging.R.
"""

import numpy as np
import pytest

import pylibkriging as lk


def f1d(x):
    x = np.asarray(x).ravel()
    return 1 - 0.5 * (np.sin(12 * x) / (1 + x) + 2 * np.cos(7 * x) * x ** 5 + 0.7)


def branin(x1, x2):
    a, b, cc = 1.0, 5.1 / (4 * np.pi ** 2), 5.0 / np.pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * np.pi)
    return a * (x2 - b * x1 ** 2 + cc * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


def _as_col(x):
    return np.asarray(x, dtype=np.float64).reshape(-1, 1)


def test_kumaraswamy_1d():
    X = _as_col(np.linspace(0.01, 0.99, 8))
    y = f1d(X)

    k = lk.WarpKriging(y, X, ["kumaraswamy"], "gauss",
                       parameters={"max_iter_adam": "200"})
    assert isinstance(k.summary(), str) and len(k.summary()) > 0

    mean, stdev, _, _, _ = k.predict(X, True, False, False)
    assert mean.shape[0] == 8
    assert np.all(np.isfinite(mean))

    x_pred = _as_col(np.linspace(0.01, 0.99, 50))
    mean2, _, _, _, _ = k.predict(x_pred, True, False, False)
    rmse = float(np.sqrt(np.mean((mean2.ravel() - f1d(x_pred)) ** 2)))
    assert np.isfinite(rmse)


def test_categorical_embedding():
    mu = np.array([1.0, 5.0, 3.0])
    rng = np.random.default_rng(42)
    n = 15
    levels = np.tile([0, 1, 2], int(np.ceil(n / 3)))[:n]
    X = _as_col(levels.astype(float))
    y = mu[levels] + rng.normal(scale=0.1, size=n)

    k = lk.WarpKriging(y, X, ["categorical(3,2)"], "gauss",
                       parameters={"max_iter_adam": "200"})

    X_test = _as_col([0.0, 1.0, 2.0])
    mean, _, _, _, _ = k.predict(X_test, True, False, False)
    assert mean.shape[0] == 3
    assert np.all(np.isfinite(mean))


def test_mixed_continuous_categorical():
    offset = np.array([1.0, 2.0, 0.5])
    rng = np.random.default_rng(99)
    n = 30
    x1 = rng.uniform(0, 1, n)
    x2 = np.tile([0, 1, 2], int(np.ceil(n / 3)))[:n]
    X = np.column_stack([x1, x2.astype(float)])
    y = np.sin(2 * np.pi * x1) * offset[x2]

    k = lk.WarpKriging(y, X, ["kumaraswamy", "categorical(3,2)"], "matern5_2",
                       parameters={"max_iter_adam": "300"})

    xc = np.linspace(0.01, 0.99, 20)
    for cat_idx in range(3):
        X_test = np.column_stack([xc, np.full(20, float(cat_idx))])
        mean, _, _, _, _ = k.predict(X_test, True, False, False)
        assert mean.shape[0] == 20
        assert np.all(np.isfinite(mean))


def test_ordinal():
    L = 5
    rng = np.random.default_rng(7)
    n = 20
    levels = np.tile(np.arange(L), int(np.ceil(n / L)))[:n]
    X = _as_col(levels.astype(float))
    y = levels ** 2 + rng.normal(scale=0.1, size=n)

    k = lk.WarpKriging(y, X, ["ordinal(5)"], "gauss",
                       parameters={"max_iter_adam": "200"})

    X_test = _as_col(np.arange(L, dtype=float))
    mean, _, _, _, _ = k.predict(X_test, True, False, False)
    assert mean.shape[0] == L
    assert np.all(np.isfinite(mean))


def test_neural_mono():
    X = _as_col(np.linspace(0.01, 0.99, 10))
    y = f1d(X)

    k = lk.WarpKriging(y, X, ["neural_mono(8)"], "gauss",
                       parameters={"max_iter_adam": "200"})

    mean, _, _, _, _ = k.predict(X, True, False, False)
    assert np.all(np.isfinite(mean))


def test_mlp_warping_1d():
    X = _as_col(np.linspace(0.01, 0.99, 10))
    y = f1d(X)

    k = lk.WarpKriging(y, X, ["mlp(16:8,3,selu)"], "gauss",
                       parameters={"max_iter_adam": "300"})

    x_pred = _as_col(np.linspace(0.01, 0.99, 50))
    mean, _, _, _, _ = k.predict(x_pred, True, False, False)
    rmse = float(np.sqrt(np.mean((mean.ravel() - f1d(x_pred)) ** 2)))
    assert np.isfinite(rmse)


def test_mlp_plus_categorical_mixed():
    offset = np.array([1.0, 2.0, 0.5])
    rng = np.random.default_rng(99)
    n = 30
    x1 = rng.uniform(0, 1, n)
    x2 = np.tile([0, 1, 2], int(np.ceil(n / 3)))[:n]
    X = np.column_stack([x1, x2.astype(float)])
    y = np.sin(2 * np.pi * x1) * offset[x2]

    k = lk.WarpKriging(y, X,
                       ["mlp(16:8,2,tanh)", "categorical(3,2)"],
                       "matern5_2",
                       parameters={"max_iter_adam": "300"})

    xc = np.linspace(0.01, 0.99, 20)
    for cat_idx in range(3):
        X_test = np.column_stack([xc, np.full(20, float(cat_idx))])
        mean, _, _, _, _ = k.predict(X_test, True, False, False)
        assert mean.shape[0] == 20
        assert np.all(np.isfinite(mean))


def test_simulate_mixed():
    rng = np.random.default_rng(42)
    n = 20
    x1 = rng.uniform(0, 1, n)
    x2 = np.tile([0, 1], int(np.ceil(n / 2)))[:n]
    X = np.column_stack([x1, x2.astype(float)])
    y = np.sin(2 * np.pi * x1) * np.array([1.0, 3.0])[x2]

    k = lk.WarpKriging(y, X, ["affine", "categorical(2,2)"], "gauss",
                       parameters={"max_iter_adam": "200"})

    X_sim = np.column_stack([np.linspace(0.1, 0.9, 10), np.zeros(10)])
    sims = k.simulate(30, 123, X_sim)
    assert sims.shape == (10, 30)
    assert np.all(np.isfinite(sims))


def test_update():
    X0 = np.array([[0.1, 0.0], [0.5, 1.0], [0.9, 0.0]])
    y0 = np.array([1.0, 3.0, 0.5])

    k = lk.WarpKriging(y0, X0, ["none", "categorical(2,1)"], "gauss",
                       parameters={"max_iter_adam": "100"})

    X_new = np.array([[0.3, 1.0], [0.7, 0.0]])
    k.update(np.array([2.0, 1.5]), X_new)

    mean, _, _, _, _ = k.predict(X0, True, False, False)
    assert np.all(np.isfinite(mean))


def test_branin_2d_mlp():
    rng = np.random.default_rng(77)
    n = 25
    X = rng.uniform(0, 1, (n, 2))
    y = np.array([branin(X[i, 0] * 15 - 5, X[i, 1] * 15) for i in range(n)])

    k = lk.WarpKriging(y, X,
                       ["mlp(16:8,2,selu)", "mlp(16:8,2,selu)"],
                       "matern5_2",
                       normalize=True,
                       parameters={"max_iter_adam": "300"})

    rng2 = np.random.default_rng(88)
    X_test = rng2.uniform(0, 1, (15, 2))
    mean, stdev, _, _, _ = k.predict(X_test, True, False, False)
    assert mean.shape[0] == 15
    assert np.all(np.isfinite(stdev))

    sims = k.simulate(20, 42, X_test)
    assert sims.shape == (15, 20)


def test_none_vs_kumaraswamy_vs_mlp():
    X = _as_col(np.linspace(0.01, 0.99, 12))
    y = f1d(X)
    xp = _as_col(np.linspace(0.01, 0.99, 50))
    ytrue = f1d(xp)

    k_none = lk.WarpKriging(y, X, ["none"], "gauss",
                            parameters={"max_iter_adam": "200"})
    k_kuma = lk.WarpKriging(y, X, ["kumaraswamy"], "gauss",
                            parameters={"max_iter_adam": "200"})
    k_mlp = lk.WarpKriging(y, X, ["mlp(16:8,2,selu)"], "gauss",
                           parameters={"max_iter_adam": "300"})

    def rmse(model):
        mean, _, _, _, _ = model.predict(xp, True, False, False)
        return float(np.sqrt(np.mean((mean.ravel() - ytrue) ** 2)))

    for m in (k_none, k_kuma, k_mlp):
        assert np.isfinite(rmse(m))
        assert np.isfinite(m.logLikelihood())


def test_loglikelihood_fun_gradient():
    X = _as_col(np.linspace(0.01, 0.99, 8))
    y = f1d(X)

    k = lk.WarpKriging(y, X, ["affine"], "gauss",
                       parameters={"max_iter_adam": "100"})

    th = k.theta()
    ll, grad, _ = k.logLikelihoodFun(th, return_grad=True)
    assert np.isfinite(ll)
    assert np.all(np.isfinite(grad))

    h = 1e-5
    grad_num = np.zeros_like(th)
    for i in range(th.shape[0]):
        tp = th.copy()
        tm = th.copy()
        tp[i] += h
        tm[i] -= h
        ll_p, _, _ = k.logLikelihoodFun(tp, return_grad=False)
        ll_m, _, _ = k.logLikelihoodFun(tm, return_grad=False)
        grad_num[i] = (ll_p - ll_m) / (2 * h)

    rel = float(np.linalg.norm(grad.ravel() - grad_num.ravel())
                / (np.linalg.norm(grad_num) + 1e-12))
    assert rel < 1e-2


def test_accessors_summary():
    X = _as_col(np.linspace(0.01, 0.99, 6))
    y = f1d(X)

    k = lk.WarpKriging(y, X, ["boxcox"], "matern3_2",
                       parameters={"max_iter_adam": "100"})

    s = k.summary()
    assert isinstance(s, str) and len(s) > 0

    th = k.theta()
    assert th.size > 0
    assert np.all(th > 0)

    assert k.sigma2() > 0
    assert k.kernel() == "matern3_2"
    assert np.isfinite(k.logLikelihood())
    assert k.warping() == ["boxcox"]
    assert k.is_fitted()


def test_predict_with_deriv():
    X = _as_col(np.linspace(0.01, 0.99, 8))
    y = f1d(X)

    k = lk.WarpKriging(y, X, ["affine"], "gauss",
                       parameters={"max_iter_adam": "100"})

    X_new = _as_col(np.linspace(0.1, 0.9, 5))
    mean, stdev, _, mean_deriv, stdev_deriv = k.predict(X_new, True, False, True)
    assert mean_deriv.shape == (5, 1)
    assert stdev_deriv.shape == (5, 1)
    assert np.all(np.isfinite(mean_deriv))
    assert np.all(np.isfinite(stdev_deriv))


def test_getters():
    X = _as_col(np.linspace(0.01, 0.99, 8))
    y = f1d(X)

    k = lk.WarpKriging(y, X, ["affine"], "gauss",
                       parameters={"max_iter_adam": "100"})

    assert k.X().shape == (8, 1)
    assert k.y().shape[0] == 8
    th = k.theta()
    assert th.size > 0 and np.all(th > 0)
    assert k.sigma2() > 0
    assert k.feature_dim() > 0


def test_copy():
    X = _as_col(np.linspace(0.01, 0.99, 8))
    y = f1d(X)

    k = lk.WarpKriging(y, X, ["affine"], "gauss",
                       parameters={"max_iter_adam": "100"})

    k2 = k.copy()
    assert k2.is_fitted()
    assert k2.kernel() == k.kernel()
    assert k2.warping() == k.warping()

    X_test = _as_col(np.linspace(0.1, 0.9, 5))
    mean1, _, _, _, _ = k.predict(X_test, True, False, False)
    mean2, _, _, _, _ = k2.predict(X_test, True, False, False)
    assert np.allclose(mean1, mean2)
