import pylibkriging as lk
import numpy as np
import pytest


def test_copied_kriging_returns_same_result():
    X = [0.0, 0.2, 0.5, 0.8, 1.0]
    f = lambda x: (1 - 1 / 2 * (np.sin(12 * x) / (1 + x) + 2 * np.cos(7 * x) * x ** 5 + 0.7))
    y = [f(xi) for xi in X]

    rl1 = lk.Kriging(y, X, "gauss", parameters={'sigma2': 1, 'is_theta_estim': False})
    print(rl1.summary())

    rl2 = rl1.copy()  # true copy not reference copy
    print(rl2.summary())

    assert id(rl1) != id(rl2)  # not same object reference                  

    x = np.arange(0, 1, 1 / 99)

    p1 = rl1.predict(x, True, False, False)
    p1 = {"mean": p1[0], "stdev": p1[1], "cov": p1[2], "mean_deriv": p1[3], "stdev_deriv": p1[4]}

    p2 = rl2.predict(x, True, False, False)
    p2 = {"mean": p2[0], "stdev": p2[1], "cov": p2[2], "mean_deriv": p2[3], "stdev_deriv": p2[4]}

    assert np.array_equal(p1["mean"], p2["mean"])
    assert np.array_equal(p1["stdev"], p2["stdev"])
    assert np.array_equal(p1["cov"], p2["cov"])
    assert np.array_equal(p1["mean_deriv"], p2["mean_deriv"])
    assert np.array_equal(p1["stdev_deriv"], p2["stdev_deriv"])


def test_copied_and_changed_kriging_returns_different_result():
    X = [0.0, 0.2, 0.5, 0.8, 1.0]
    f = lambda x: (1 - 1 / 2 * (np.sin(12 * x) / (1 + x) + 2 * np.cos(7 * x) * x ** 5 + 0.7))
    y = [f(xi) for xi in X]

    rl1 = lk.Kriging(y, X, "gauss", parameters={'sigma2': 1, 'is_theta_estim': False})
    print(rl1.summary())

    rl2 = rl1.copy()  # true copy not reference copy
    print(rl2.summary())

    assert id(rl1) != id(rl2)  # not same object reference                  

    x = np.arange(0, 1, 1 / 99)

    p1 = rl1.predict(x, True, False, False)
    p1 = {"mean": p1[0], "stdev": p1[1], "cov": p1[2], "mean_deriv": p1[3], "stdev_deriv": p1[4]}

    rl2.update([0.6], [f(0.6)])
    p2 = rl2.predict(x, True, False, False)
    p2 = {"mean": p2[0], "stdev": p2[1], "cov": p2[2], "mean_deriv": p2[3], "stdev_deriv": p2[4]}

    assert not np.array_equal(p1["mean"], p2["mean"])
    assert not np.array_equal(p1["stdev"], p2["stdev"])
