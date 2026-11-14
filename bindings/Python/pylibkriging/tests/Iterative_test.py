import os
import subprocess
import sys

import numpy as np
import pylibkriging as lk
import pytest


def f2d(x1, x2):
    return np.sin(3 * x1) + np.cos(5 * x2) + x1 * x2


def make_data(n, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, 2))
    y = f2d(X[:, 0], X[:, 1])
    return X, y


def make_fixed_theta_model(y, X, method="exact", theta_val=0.3):
    # fixed, moderate theta (optim="none"): a free fit on this noise-free
    # function drifts toward a near-singular correlation matrix, unrelated to CG
    parameters = {"theta": np.full((1, X.shape[1]), theta_val), "is_theta_estim": False}
    return lk.Kriging(y, X, "matern5_2", regmodel="constant", optim="none",
                      objective="LL", parameters=parameters, method=method)


def test_predict_iterative_matches_exact_predict_on_an_exact_fit():
    # theta = 0.1: well conditioned, the default CG tolerance (1e-8) suffices
    X, y = make_data(60)
    k = make_fixed_theta_model(y, X, theta_val=0.1)
    Xt, _ = make_data(20, seed=456)
    m_ex, s_ex, _, _, _ = k.predict(Xt, True, False, False)
    m_cg, s_cg, _, _, _ = k.predict(Xt, True, False, False, method="iterative")
    assert m_cg.shape == (20,)  # 1-D outputs, as the other methods
    sdy = np.std(y)
    assert np.max(np.abs(m_ex - m_cg)) < 1e-5 * sdy
    assert np.max(np.abs(s_ex - s_cg)) < 1e-4 * sdy


def test_predict_iterative_nystrom_preconditioner_spec():
    X, y = make_data(60)
    k = make_fixed_theta_model(y, X, theta_val=0.1)
    Xt, _ = make_data(20, seed=456)
    m_ex, s_ex, _, _, _ = k.predict(Xt, True, False, False)
    m_pc, s_pc, _, _, _ = k.predict(Xt, True, False, False, method="iterative(30,20)")
    sdy = np.std(y)
    assert np.max(np.abs(m_ex - m_pc)) < 1e-5 * sdy
    assert np.max(np.abs(s_ex - s_pc)) < 1e-4 * sdy


def test_iterative_fit_light_and_explicit():
    X, y = make_data(40)
    k = make_fixed_theta_model(y, X, method="iterative(8)")
    assert k.method() == "iterative(8)"
    Xt, _ = make_data(5, seed=789)
    with pytest.raises(Exception):  # no exact factorization: no implicit switch
        k.predict(Xt, True, False, False)
    mean, stdev, _, _, _ = k.predict(Xt, True, False, False, method="iterative")
    assert np.all(np.isfinite(mean)) and np.all(np.isfinite(stdev))
    with pytest.raises(Exception):  # return_cov: exact only
        k.predict(Xt, True, True, False, method="iterative")
    with pytest.raises(Exception):
        k.simulate(3, 123, Xt)
    assert np.isfinite(k.logLikelihood("iterative"))


def test_iterative_restrictions_raise():
    X, y = make_data(20)
    with pytest.raises(Exception):
        lk.Kriging(y, X, "matern5_2", objective="LOO", method="iterative")
    with pytest.raises(Exception):
        lk.Kriging(y, X, "matern5_2", noise="nugget", method="iterative")
    with pytest.raises(Exception):
        make_fixed_theta_model(y, X, method="iterative(0)")


def test_iterative_save_load_round_trip(tmp_path):
    X, y = make_data(30)
    k = make_fixed_theta_model(y, X, method="iterative(8,4)")
    f = str(tmp_path / "iterative.json")
    k.save(f)
    kl = lk.Kriging.load(f)
    assert kl.method() == k.method()
    assert kl.logLikelihood("iterative") == pytest.approx(k.logLikelihood("iterative"), rel=1e-10)


def test_iterative_update():
    X, y = make_data(30)
    k = make_fixed_theta_model(y, X, method="iterative(8)")
    Xu, yu = make_data(5, seed=11)
    k.update(yu, Xu, False)
    assert k.X().shape[0] == 35
    mean, _, _, _, _ = k.predict(Xu, False, False, False, method="iterative")
    assert np.all(np.isfinite(mean))


def test_warp_mlp_predict_iterative_matches_exact():
    X = np.linspace(0.01, 0.99, 8).reshape(-1, 1)
    y = np.sin(6 * X[:, 0])
    xp = np.linspace(0.01, 0.99, 20).reshape(-1, 1)
    w = lk.WarpKriging(["kumaraswamy"], "gauss")
    w.fit(y, X, "constant", False, "none", "LL", {"theta": np.array([0.3])})
    m_ex, s_ex, _, _, _ = w.predict(xp, True, False, False)
    m_cg, s_cg, _, _, _ = w.predict(xp, True, False, False, method="iterative(30,0,20,250,1e-12)")
    assert np.max(np.abs(m_ex - m_cg)) < 1e-4 * np.std(y)
    assert np.max(np.abs(s_ex - s_cg)) < 1e-4 * np.std(y)


def test_linearalgebra_iterative_settings():
    LA = lk.LinearAlgebra
    orig = LA.get_iterative_dense_max_mb()
    try:
        LA.set_iterative_dense_max_mb(0)
        assert LA.get_iterative_dense_max_mb() == 0
        with pytest.raises(Exception):
            LA.set_iterative_dense_max_mb(-1)
    finally:
        LA.set_iterative_dense_max_mb(orig)
    LA.set_cg_warning(True)
    LA.set_cg_auto_precond(True)


def test_gpu_settings_always_defined():
    G = lk.Gpu
    assert isinstance(G.compiled_backends(), str)
    assert isinstance(G.available(), bool)
    assert G.active_backend() in ("none", "cuda", "hip", "sycl", "metal")
    orig = (G.get_enabled(), G.get_dense_max_mb(), G.get_mixed_precision(), G.get_verbose())
    try:
        G.set_enabled(False)
        assert not G.get_enabled()
        assert G.active_backend() == "none"
        G.set_enabled(True)
        if not G.available():  # never turns on a backend without a device
            assert G.active_backend() == "none"
        G.set_dense_max_mb(0)
        assert G.get_dense_max_mb() == 0
        with pytest.raises(Exception):
            G.set_dense_max_mb(-1)
        G.set_mixed_precision(True)
        assert G.get_mixed_precision()
        G.set_verbose(True)
        assert G.get_verbose()
    finally:
        G.set_enabled(orig[0])
        G.set_dense_max_mb(orig[1])
        G.set_mixed_precision(orig[2])
        G.set_verbose(orig[3])


def test_gpu_disabled_by_environment():
    code = "import pylibkriging as lk; print(lk.Gpu.get_enabled(), lk.Gpu.active_backend())"
    env = dict(os.environ, LK_ITERATIVE_GPU="0")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.split()[-2:] == ["False", "none"]
