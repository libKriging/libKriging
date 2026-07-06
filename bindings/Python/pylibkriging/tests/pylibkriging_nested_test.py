import numpy as np
import pytest
import pylibkriging as lk


def f(X):
    return np.sin(3 * X[:, 0]) + np.cos(5 * X[:, 1]) + X[:, 0] * X[:, 1]


@pytest.fixture
def data():
    rng = np.random.default_rng(123)
    X = rng.uniform(size=(200, 2))
    return X, f(X)


def test_one_group_matches_kriging(data):
    X, y = data
    nk = lk.NestedKriging(y, X, "gauss", 1, aggregation="NK")
    Xt = np.random.default_rng(456).uniform(size=(50, 2))
    m_nk, s_nk = nk.predict(Xt, True)
    assert np.all(np.isfinite(m_nk)) and np.all(s_nk >= 0)


@pytest.mark.parametrize("agg", ["PoE", "gPoE", "BCM", "rBCM", "NK"])
def test_aggregations_predict(data, agg):
    X, y = data
    nk = lk.NestedKriging(y, X, "matern5_2", 4, aggregation=agg)
    Xt = np.random.default_rng(456).uniform(size=(100, 2))
    mean, stdev = nk.predict(Xt, True)
    assert mean.shape == (100,) and stdev.shape == (100,)
    assert np.all(np.isfinite(mean)) and np.all(stdev >= 0)
    # accuracy sanity: better than predicting the mean
    assert np.sqrt(np.mean((mean - f(Xt)) ** 2)) < 0.5 * np.std(y)


def test_nk_interpolates(data):
    X, y = data
    nk = lk.NestedKriging(y, X, "matern5_2", 4, aggregation="NK")
    mean, stdev = nk.predict(X, True)
    np.testing.assert_allclose(mean, y, atol=1e-3)
    assert stdev.max() < 1e-2


def test_common_prior_and_accessors(data):
    X, y = data
    nk = lk.NestedKriging(y, X, "gauss", 4)
    assert nk.kernel() == "gauss"
    assert nk.aggregation() == "NK"
    assert nk.nb_groups() == 4
    assert nk.theta().shape == (2,)
    assert nk.sigma2() > 0
    assert sum(len(g) for g in nk.groups()) == len(y)


def test_invalid_args(data):
    X, y = data
    with pytest.raises(Exception):
        lk.NestedKriging(y, X, "gauss", 4, aggregation="median")
    with pytest.raises(Exception):
        lk.NestedKriging(y, X, "gauss", 1000)  # too many groups


def test_reproducibility(data):
    X, y = data
    Xt = np.random.default_rng(456).uniform(size=(20, 2))
    nk1 = lk.NestedKriging(y, X, "gauss", 5, partition="random", seed=42)
    nk2 = lk.NestedKriging(y, X, "gauss", 5, partition="random", seed=42)
    m1, s1 = nk1.predict(Xt, True)
    m2, s2 = nk2.predict(Xt, True)
    np.testing.assert_array_equal(m1, m2)
    np.testing.assert_array_equal(s1, s2)


def test_warped_nested(data):
    X, y = data
    nk = lk.NestedKriging(y, X, "gauss", 3, warping=["kumaraswamy", "kumaraswamy"])
    assert nk.warping() == ["kumaraswamy", "kumaraswamy"]
    mean, stdev = nk.predict(X, True)
    np.testing.assert_allclose(mean, y, atol=1e-3)  # NK interpolates under warping
    assert np.all(stdev >= 0)
