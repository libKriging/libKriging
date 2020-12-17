import pylibkriging as lk
import numpy as np
import pytest


def f(X):
    return np.apply_along_axis(lambda x: np.prod(np.sin((x - 0.5) ** 2)), axis=1, arr=X)


@pytest.mark.parametrize("n", [40, 100])
@pytest.mark.parametrize("m", [3, 6])
def test_ordinary_kriging(n, m):
    X = np.asarray(np.random.uniform(size=(n, m)), dtype=np.float64, order='F')
    y = f(X)

    rl = lk.OrdinaryKriging("gauss")
    rl.fit(y, X)

    y_pred, _stderr, _cov = rl.predict(X, True, True)
