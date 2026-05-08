"""Tests for MLPKriging Python binding.

Mirrors bindings/R/rlibkriging/tests/testthat/test-MLPKriging.R.
"""

import numpy as np

import pylibkriging as lk


def f1d(x):
    x = np.asarray(x).ravel()
    return 1 - 0.5 * (np.sin(12 * x) / (1 + x) + 2 * np.cos(7 * x) * x ** 5 + 0.7)


def branin(x1, x2):
    a, b, cc = 1.0, 5.1 / (4 * np.pi ** 2), 5.0 / np.pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * np.pi)
    return a * (x2 - b * x1 ** 2 + cc * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


def test_mlpkriging_1d_gauss():
    X = np.linspace(0.01, 0.99, 10).reshape(-1, 1)
    y = f1d(X)

    k = lk.MLPKriging(y, X,
                      hidden_dims=[16, 8], d_out=2,
                      activation="selu", kernel="gauss",
                      normalize=True,
                      parameters={"max_iter_adam": "300"})

    assert isinstance(k.summary(), str) and len(k.summary()) > 0

    mean, stdev, _, _, _ = k.predict(X, True, False, False)
    assert mean.shape[0] == 10
    assert np.all(np.isfinite(mean))

    x_pred = np.linspace(0.01, 0.99, 50).reshape(-1, 1)
    mean2, _, _, _, _ = k.predict(x_pred, True, False, False)
    rmse = float(np.sqrt(np.mean((mean2.ravel() - f1d(x_pred)) ** 2)))
    assert np.isfinite(rmse)

    assert k.kernel() == "gauss"
    assert k.activation() == "selu"
    assert list(k.hidden_dims()) == [16, 8]
    assert k.feature_dim() == 2
    assert k.sigma2() > 0
    assert np.isfinite(k.logLikelihood())


def test_mlpkriging_branin_matern():
    rng = np.random.default_rng(77)
    n = 20
    X = rng.uniform(0, 1, (n, 2))
    y = np.array([branin(X[i, 0] * 15 - 5, X[i, 1] * 15) for i in range(n)])

    k = lk.MLPKriging(y, X,
                      hidden_dims=[16, 8], d_out=2,
                      activation="selu", kernel="gauss",
                      normalize=True,
                      parameters={"max_iter_adam": "300"})

    X_test = rng.uniform(0, 1, (10, 2))
    mean, stdev, _, _, _ = k.predict(X_test, True, False, False)
    assert mean.shape[0] == 10
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(stdev))

    sims = k.simulate(20, 42, X_test)
    assert sims.shape == (10, 20)
    assert np.all(np.isfinite(sims))

    rng2 = np.random.default_rng(1234)
    X_new = rng2.uniform(0, 1, (3, 2))
    y_new = np.array([branin(X_new[i, 0] * 15 - 5, X_new[i, 1] * 15) for i in range(3)])
    k.update(y_new, X_new, refit=False)

    mean2, _, _, _, _ = k.predict(X_test, True, False, False)
    assert mean2.shape[0] == 10


def test_mlpkriging_predict_deriv():
    rng = np.random.default_rng(88)
    n = 20
    X = rng.uniform(0, 1, (n, 2))
    def f2(x1, x2):
        return np.sin(3 * x1) + np.cos(4 * x2) + 0.5 * x1 * x2
    y = np.array([f2(X[i, 0], X[i, 1]) for i in range(n)])

    k = lk.MLPKriging(y, X,
                      hidden_dims=[16, 8], d_out=2,
                      activation="selu", kernel="gauss",
                      parameters={"max_iter_adam": "100"})

    X_new = rng.uniform(0, 1, (5, 2))
    mean, stdev, cov, mean_deriv, stdev_deriv = k.predict(X_new, True, False, True)
    assert mean_deriv.shape == (5, 2)
    assert stdev_deriv.shape == (5, 2)
    assert np.all(np.isfinite(mean_deriv))
    assert np.all(np.isfinite(stdev_deriv))


def test_mlpkriging_loglikelihood_fun():
    X = np.linspace(0.01, 0.99, 10).reshape(-1, 1)
    y = f1d(X)

    k = lk.MLPKriging(y, X,
                      hidden_dims=[16, 8], d_out=2,
                      activation="selu", kernel="gauss",
                      parameters={"max_iter_adam": "200"})

    th = k.theta()
    ll_val = k.logLikelihood()
    assert np.isfinite(ll_val)

    ll, grad, _ = k.logLikelihoodFun(th, return_grad=True)
    assert np.isfinite(ll)
    assert np.all(np.isfinite(grad))

    ll2, _, _ = k.logLikelihoodFun(th, return_grad=False)
    assert np.isfinite(ll2)


def test_mlpkriging_is_fitted():
    X = np.linspace(0.01, 0.99, 8).reshape(-1, 1)
    y = f1d(X)

    k = lk.MLPKriging(y, X,
                      hidden_dims=[16, 8], d_out=2,
                      activation="selu", kernel="gauss",
                      parameters={"max_iter_adam": "100"})

    assert k.is_fitted()


def test_mlpkriging_getters():
    X = np.linspace(0.01, 0.99, 8).reshape(-1, 1)
    y = f1d(X)

    k = lk.MLPKriging(y, X,
                      hidden_dims=[16, 8], d_out=2,
                      activation="selu", kernel="gauss",
                      parameters={"max_iter_adam": "100"})

    assert k.X().shape == (8, 1)
    assert k.y().shape[0] == 8
    th = k.theta()
    assert th.size > 0 and np.all(th > 0)
    assert k.sigma2() > 0
