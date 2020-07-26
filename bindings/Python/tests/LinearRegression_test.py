import pylibkriging as lk
import numpy as np
import pytest


@pytest.mark.parametrize("n", [40, 100, 1000])
@pytest.mark.parametrize("m", [3, 6])
def test_linear_regression_exact(n, m):
    sol = np.asarray(np.random.normal(size=(m, 1)), dtype=np.float64, order='F')
    X = np.asarray(np.random.normal(size=(n, m)), dtype=np.float64, order='F')
    X[:, 0] = 1
    y = X.dot(sol)  # or X @ sol

    rl = lk.LinearRegression()
    rl.fit(y, X)

    y2, _stderr = rl.predict(X)

    eps = 1e-5
    assert np.linalg.norm(y - y2, ord=np.inf) <= eps


@pytest.mark.parametrize("n", [40, 100, 1000])
@pytest.mark.parametrize("m", [3, 6])
def test_linear_regression_noisy(n, m):
    sol = np.asarray(np.random.normal(size=(m, 1)), dtype=np.float64, order='F')
    X = np.asarray(np.random.normal(size=(n, m)), dtype=np.float64, order='F')
    X[:, 0] = 1
    y = X.dot(sol)  # or X @ sol

    e = 1e-8
    noiser = lambda x: x * np.random.normal(1, e)
    y = np.vectorize(noiser)(y)
    # y = np.array([noiser(v) for v in y])
    # y = np.array(list(map(noiser, y)))

    rl = lk.LinearRegression()
    rl.fit(y, X)

    y2, _stderr = rl.predict(X)

    eps = 1e-5
    assert np.linalg.norm(y - y2, ord=np.inf) <= eps
