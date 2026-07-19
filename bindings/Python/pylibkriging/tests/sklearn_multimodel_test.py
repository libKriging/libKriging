"""Tests for the WarpKriging/MLPKriging/NestedKriging sklearn wrappers in
pylibkriging.sklearn (KrigingRegressor itself is covered by
sklearn_estimator_test.py). Skips entirely if scikit-learn isn't
installed (optional 'sklearn' extra, see setup.py).
"""
import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")

from pylibkriging.sklearn import (  # noqa: E402
    MLPKrigingRegressor,
    NestedKrigingRegressor,
    WarpKrigingRegressor,
)


def _make_data(n=30, d=3, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(size=(n, d))
    y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + (X[:, 2] if d > 2 else 0.0)
    return X, y


# --- WarpKrigingRegressor -----------------------------------------------

def test_warp_default_warping_is_none_for_every_column():
    X, y = _make_data()
    reg = WarpKrigingRegressor().fit(X, y)
    assert reg.warping_ == ["none"] * X.shape[1]
    mean = reg.predict(X[:3])
    assert mean.shape == (3,)


def test_warp_explicit_warping():
    X, y = _make_data()
    reg = WarpKrigingRegressor(warping=["affine", "none", "kumaraswamy"]).fit(X, y)
    mean, std = reg.predict(X[:3], return_std=True)
    assert mean.shape == (3,) and std.shape == (3,)


def test_warp_wrong_length_raises():
    X, y = _make_data()
    with pytest.raises(ValueError, match="warping"):
        WarpKrigingRegressor(warping=["affine"]).fit(X, y)  # only 1, X has 3 cols


def test_warp_check_estimator():
    from sklearn.utils.estimator_checks import check_estimator
    check_estimator(WarpKrigingRegressor())


# --- MLPKrigingRegressor -------------------------------------------------

def test_mlp_fit_predict():
    X, y = _make_data()
    reg = MLPKrigingRegressor(hidden_dims=(4, 4), d_out=2).fit(X, y)
    mean = reg.predict(X[:3])
    assert mean.shape == (3,)


def test_mlp_return_cov():
    X, y = _make_data()
    reg = MLPKrigingRegressor().fit(X, y)
    mean, cov = reg.predict(X[:3], return_cov=True)
    assert mean.shape == (3,) and cov.shape == (3, 3)


def test_mlp_check_estimator():
    from sklearn.utils.estimator_checks import check_estimator
    check_estimator(MLPKrigingRegressor())


# --- NestedKrigingRegressor -----------------------------------------------

def test_nested_fit_predict():
    X, y = _make_data(n=40)
    reg = NestedKrigingRegressor(nb_groups=4).fit(X, y)
    mean, std = reg.predict(X[:3], return_std=True)
    assert mean.shape == (3,) and std.shape == (3,)


def test_nested_nk_requires_constant_regmodel():
    X, y = _make_data(n=40)
    with pytest.raises(ValueError, match="constant"):
        NestedKrigingRegressor(aggregation="NK", regmodel="linear").fit(X, y)


def test_nested_poe_allows_non_constant_regmodel():
    X, y = _make_data(n=40)
    # Should NOT raise: the "NK requires constant" restriction is specific
    # to the NK aggregation.
    NestedKrigingRegressor(aggregation="PoE", regmodel="linear", nb_groups=4).fit(X, y)


def test_nested_predict_has_no_return_cov():
    X, y = _make_data(n=40)
    reg = NestedKrigingRegressor(nb_groups=4).fit(X, y)
    with pytest.raises(TypeError):
        reg.predict(X[:3], return_cov=True)


def test_nested_has_no_sample_y():
    assert not hasattr(NestedKrigingRegressor(), "sample_y")


def test_nested_check_estimator():
    from sklearn.utils.estimator_checks import check_estimator
    check_estimator(NestedKrigingRegressor())


# --- cross-cutting: all three behave inside sklearn machinery -------------

@pytest.mark.parametrize("estimator,param_grid", [
    (WarpKrigingRegressor(warping=["none", "none", "none"]), {"kernel": ["gauss"]}),
    (MLPKrigingRegressor(hidden_dims=(4,)), {"d_out": [1, 2]}),
    (NestedKrigingRegressor(nb_groups=4), {"nb_groups": [3, 4]}),
])
def test_pipeline_and_grid_search(estimator, param_grid):
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = _make_data(n=40)

    cloned = clone(estimator)
    assert cloned.get_params() == estimator.get_params()

    pipe = Pipeline([("scale", StandardScaler()), ("est", estimator)]).fit(X, y)
    pred = pipe.predict(X[:3])
    assert pred.shape == (3,)

    gscv = GridSearchCV(estimator, param_grid, cv=3).fit(X, y)
    assert set(gscv.best_params_) == set(param_grid)
