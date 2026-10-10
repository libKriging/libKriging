"""Tests for MultiOutputKrigingRegressor in pylibkriging.sklearn. Skips
entirely if scikit-learn isn't installed (optional 'sklearn' extra).
"""
import pickle

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")

import pylibkriging as lk  # noqa: E402
from pylibkriging.sklearn import KrigingRegressor, MultiOutputKrigingRegressor  # noqa: E402

t_out = np.linspace(0.5, 10, 20)


def _make_data(n=30, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(size=(n, 2))
    Y = np.exp(-(0.1 + 0.5 * X[:, [1]]) * t_out) * np.cos(2 * np.pi * (0.5 + 1.5 * X[:, [0]]) * t_out / 5)
    return X, Y


@pytest.mark.parametrize("output_model", ["pca", "shared", "separable(matern5_2)"])
def test_shapes_and_delegation(output_model):
    X, Y = _make_data()
    reg = MultiOutputKrigingRegressor(output_model=output_model, normalize=True,
                                      output_coordinates=t_out).fit(X, Y)
    assert reg.n_targets_ == 20 and reg.n_features_in_ == 2
    Xn = X[:4] + 0.01
    mean, std = reg.predict(Xn, return_std=True)
    assert mean.shape == (4, 20) and std.shape == (4, 20)
    m2, s2, c2, _ = reg.model_.predict(Xn, True, True, False)
    np.testing.assert_allclose(mean, m2)
    np.testing.assert_allclose(std, s2)
    _, cov = reg.predict(Xn, return_cov=True)
    assert cov.shape == (4, 4, 20)
    for j in (0, 7, 19):  # per-target blocks of the joint covariance over vec(Y)
        np.testing.assert_allclose(cov[:, :, j], c2[4 * j:4 * (j + 1), 4 * j:4 * (j + 1)])
        np.testing.assert_allclose(np.sqrt(np.diag(cov[:, :, j])), std[:, j], rtol=1e-6, atol=1e-10)
    sims = reg.sample_y(Xn, n_samples=5, random_state=3)
    assert sims.shape == (4, 20, 5)
    np.testing.assert_array_equal(sims, reg.model_.simulate(5, 3, Xn))
    assert reg.score(X, Y) > 0.99  # interpolation, up to the "pca" truncation


def test_one_dimensional_y_matches_kriging():
    X, Y = _make_data()
    y = Y[:, 3]
    reg = MultiOutputKrigingRegressor(output_model="shared").fit(X, y)
    ref = KrigingRegressor().fit(X, y)
    mean, std = reg.predict(X[:5] + 0.02, return_std=True)
    assert mean.shape == (5,) and std.shape == (5,)
    m_ref, s_ref = ref.predict(X[:5] + 0.02, return_std=True)
    np.testing.assert_allclose(mean, m_ref, rtol=1e-6, atol=1e-6 * np.std(y))
    np.testing.assert_allclose(std, s_ref, rtol=1e-4, atol=1e-6 * np.std(y))
    _, cov = reg.predict(X[:5], return_cov=True)
    assert cov.shape == (5, 5)
    assert reg.sample_y(X[:5], n_samples=3).shape == (5, 3)


def test_parameters_accept_vectors():
    X, Y = _make_data()
    reg = MultiOutputKrigingRegressor(output_model="separable(matern5_2)", output_coordinates=t_out,
                                      parameters={"theta": [0.3, 0.4], "output_theta": [2.0],
                                                  "is_theta_estim": False}).fit(X, Y)
    np.testing.assert_allclose(reg.model_.theta().ravel(), [0.3, 0.4])
    np.testing.assert_allclose(reg.model_.output_theta().ravel(), [2.0])


def test_return_std_and_cov_exclusive():
    X, Y = _make_data()
    reg = MultiOutputKrigingRegressor().fit(X, Y)
    with pytest.raises(RuntimeError):
        reg.predict(X[:2], return_std=True, return_cov=True)


def test_pickle_round_trip():
    X, Y = _make_data()
    for output_model in ("pca(0.999)", "separable"):
        reg = MultiOutputKrigingRegressor(output_model=output_model).fit(X, Y[:, :4])
        back = pickle.loads(pickle.dumps(reg))
        assert isinstance(back.model_, lk.MultiOutputKriging)
        np.testing.assert_allclose(back.predict(X[:3] + 0.01), reg.predict(X[:3] + 0.01), atol=1e-12)
        assert reg.get_params() == back.get_params()


def test_grid_search_over_output_model():
    from sklearn.model_selection import GridSearchCV
    X, Y = _make_data(n=40)
    gscv = GridSearchCV(MultiOutputKrigingRegressor(), {"output_model": ["pca(0.999)", "shared"]}, cv=3)
    gscv.fit(X, Y)
    assert gscv.best_params_["output_model"] in ("pca(0.999)", "shared")
    assert gscv.predict(X[:2]).shape == (2, 20)


@pytest.mark.parametrize("output_model", ["pca", "shared", "separable"])
def test_check_estimator(output_model):
    from sklearn.utils.estimator_checks import check_estimator
    check_estimator(MultiOutputKrigingRegressor(output_model=output_model))
