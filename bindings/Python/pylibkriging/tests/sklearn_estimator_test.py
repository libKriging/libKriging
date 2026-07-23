"""Tests for pylibkriging.sklearn.KrigingRegressor.

Skips entirely if scikit-learn isn't installed, since it's an optional
extra (see setup.py's 'sklearn' extras_require) -- not a hard dependency
of pylibkriging.
"""
import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")

from pylibkriging.sklearn import KrigingRegressor  # noqa: E402


def _make_data(n=20, d=2, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(size=(n, d))
    y = np.sin(X[:, 0]) + np.cos(X[:, 1] if d > 1 else X[:, 0])
    return X, y


def test_fit_predict_basic():
    X, y = _make_data()
    reg = KrigingRegressor().fit(X, y)
    mean = reg.predict(X[:3])
    assert mean.shape == (3,)
    assert reg.n_features_in_ == 2


def test_predict_return_std():
    X, y = _make_data()
    reg = KrigingRegressor().fit(X, y)
    mean, std = reg.predict(X[:3], return_std=True)
    assert mean.shape == (3,)
    assert std.shape == (3,)


def test_predict_return_cov():
    X, y = _make_data()
    reg = KrigingRegressor().fit(X, y)
    mean, cov = reg.predict(X[:3], return_cov=True)
    assert mean.shape == (3,)
    assert cov.shape == (3, 3)


def test_predict_return_std_and_cov_raises():
    X, y = _make_data()
    reg = KrigingRegressor().fit(X, y)
    with pytest.raises(RuntimeError):
        reg.predict(X[:3], return_std=True, return_cov=True)


def test_predict_feature_mismatch_raises():
    X, y = _make_data()
    reg = KrigingRegressor().fit(X, y)
    with pytest.raises(ValueError):
        reg.predict(X[:, :1])


def test_sample_y():
    X, y = _make_data()
    reg = KrigingRegressor().fit(X, y)
    samples = reg.sample_y(X[:3], n_samples=5, random_state=42)
    assert samples.shape == (3, 5)


def test_get_params_set_params_clone():
    from sklearn.base import clone
    reg = KrigingRegressor(kernel="gauss", objective="LOO")
    params = reg.get_params()
    assert params["kernel"] == "gauss"
    assert params["objective"] == "LOO"
    reg2 = clone(reg)
    assert reg2.get_params() == params
    # clone must not carry over fitted state
    X, y = _make_data()
    reg.fit(X, y)
    assert not hasattr(reg2, "model_")


def test_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = _make_data()
    pipe = Pipeline([("scale", StandardScaler()), ("krig", KrigingRegressor())])
    pipe.fit(X, y)
    pred = pipe.predict(X[:3])
    assert pred.shape == (3,)


def test_cross_val_score_and_grid_search():
    from sklearn.model_selection import GridSearchCV, cross_val_score

    X, y = _make_data(n=30)
    scores = cross_val_score(KrigingRegressor(), X, y, cv=3)
    assert scores.shape == (3,)

    gscv = GridSearchCV(KrigingRegressor(), {"kernel": ["matern5_2", "gauss"]}, cv=3)
    gscv.fit(X, y)
    assert gscv.best_params_["kernel"] in ("matern5_2", "gauss")


def test_check_estimator():
    """Run scikit-learn's own estimator conformance checks."""
    from sklearn.utils.estimator_checks import check_estimator

    check_estimator(KrigingRegressor())


def test_sklearn_missing_gives_actionable_error(monkeypatch):
    """Importing pylibkriging.sklearn without scikit-learn should fail
    with a clear message, not a bare ImportError/ModuleNotFoundError."""
    import builtins
    import importlib
    import sys

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sklearn" or name.startswith("sklearn."):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    original_module = sys.modules.get("pylibkriging.sklearn")
    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("pylibkriging.sklearn", None)
    try:
        with pytest.raises(ImportError, match="pip install pylibkriging\\[sklearn\\]"):
            importlib.import_module("pylibkriging.sklearn")
    finally:
        # Restore the *original* module object (not just re-import it):
        # other already-imported test modules hold references to classes
        # from that original object, and leaving a freshly-reimported
        # (distinct) module in sys.modules breaks identity checks
        # elsewhere in the same test session (e.g. pickling in
        # check_estimator, which compares class objects by identity).
        if original_module is not None:
            sys.modules["pylibkriging.sklearn"] = original_module
        else:
            sys.modules.pop("pylibkriging.sklearn", None)
