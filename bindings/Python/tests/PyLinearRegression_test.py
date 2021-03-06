import pylibkriging as lk
import numpy as np


def test_basic():
    """Test of manual binding"""
    n = 40
    m = 3

    g = lk.RandomGenerator(123)
    sol = np.asarray(g.uniform(m, 1), dtype=np.float64, order='F')
    X = np.asarray(g.uniform(n, m), dtype=np.float64, order='F')
    X[:, 0] = 1
    y = X.dot(sol)  # or X @ sol

    rl = lk.PyLinearRegression()
    rl.fit(y, X)

    y2, _stderr = rl.predict(X)

    eps = 1e-5
    assert np.linalg.norm(y - y2, ord=2) <= eps
