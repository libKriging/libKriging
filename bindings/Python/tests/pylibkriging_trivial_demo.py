import pylibkriging as lk
import numpy as np


def f(X):
    return np.apply_along_axis(lambda x: np.prod(np.sin((x - 0.5) ** 2)), axis=1, arr=X)


def test_kriging():
    n = 40
    m = 3
    g = lk.RandomGenerator(123)
    X = g.uniform(n, m)  # this is the default
    y = f(X)
    rl = lk.Kriging(y, X, "gauss")
    y_pred, _stderr, _cov = rl.predict(X, True, True)
    print(rl.describeModel())


if __name__ == '__main__':
    test_kriging()
