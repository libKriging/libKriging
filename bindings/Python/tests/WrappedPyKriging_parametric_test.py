import pylibkriging as lk
import numpy as np
import pytest


def f(X):
    return np.apply_along_axis(lambda x: np.prod(np.sin((x - 0.5) ** 2)), axis=1, arr=X)


# @pytest.mark.skip(reason="Needs to be stabilized; got runtime error: chol(): decomposition failed")
@pytest.mark.parametrize("n", [40, 100])
@pytest.mark.parametrize("m", [3, 6])
def test_kriging_f_order(n, m):
    g = lk.RandomGenerator(123)
    X = np.asarray(g.uniform(n, m), dtype=np.float64, order='F')
    y = f(X)

    rl = lk.WrappedPyKriging(y, X, "gauss")

    y_pred, _stderr, _cov = rl.predict(X, True, True)


# @pytest.mark.skip(reason="Needs to be stabilized; got runtime error: chol(): decomposition failed")
@pytest.mark.parametrize("n", [40, 100])
@pytest.mark.parametrize("m", [3, 6])
def test_kriging_c_order(n, m):
    g = lk.RandomGenerator(123)
    # X = np.asarray(g.uniform(n, m), dtype=np.float64, order='C')
    X = g.uniform(n, m)  # this is the default
    y = f(X)

    rl = lk.WrappedPyKriging(y, X, "gauss")

    y_pred, _stderr, _cov = rl.predict(X, True, True)
