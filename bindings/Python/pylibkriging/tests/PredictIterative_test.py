import sys
import faulthandler

import numpy as np
import pylibkriging as lk
import pytest

# TEMP DEBUG (issue #351): dump every thread's Python-level stack after 60s
# without needing a hang to finish first -- ctest's --timeout kills the
# process at 120s (see tools/windows/test.sh), so this should fire well
# before that and land in the captured stdout. Revert before merge.
faulthandler.enable()
faulthandler.dump_traceback_later(60, exit=False)


def _breadcrumb(msg):
    print(f"[DEBUG-351] {msg}", flush=True)
    sys.stdout.flush()


def f2d(x1, x2):
    return np.sin(3 * x1) + np.cos(5 * x2) + x1 * x2


def make_data(n, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, 2))
    y = f2d(X[:, 0], X[:, 1])
    return X, y


def make_fixed_theta_model(y, X, theta_val=0.3):
    # Fixed, moderate theta (optim="none"): on this noise-free deterministic
    # test function, a free BFGS fit is known to drift theta toward a
    # near-singular correlation matrix (unrelated to predictIterative itself --
    # see docs/math/PredictIterative.md), which would make predict/predictIterative's
    # agreement noisy rather than a clean correctness signal.
    parameters = {"theta": np.full((1, X.shape[1]), theta_val), "sigma2": 1.0}
    _breadcrumb("make_fixed_theta_model: before lk.Kriging(...) constructor")
    k = lk.Kriging(y, X, "matern5_2", regmodel="constant", optim="none",
                  objective="LL", parameters=parameters)
    _breadcrumb("make_fixed_theta_model: after lk.Kriging(...) constructor")
    return k


def test_prediterative_mean_stdev_match_exact_predict_at_a_moderate_theta():
    _breadcrumb("test start")
    X, y = make_data(60)
    _breadcrumb("make_data(60) done")
    k = make_fixed_theta_model(y, X)
    _breadcrumb("make_fixed_theta_model done")
    Xt, _ = make_data(20, seed=456)
    _breadcrumb("make_data(20) done")

    _breadcrumb("before k.predict(...)")
    m_ex, s_ex, _, _, _ = k.predict(Xt, True, False, False)
    _breadcrumb("after k.predict(...), before k.predictIterative(...)")
    m_cg, s_cg = k.predictIterative(Xt, True)
    _breadcrumb("after k.predictIterative(...)")

    sdy = np.std(y)
    assert np.max(np.abs(m_ex - m_cg)) < 0.05 * sdy
    assert np.max(np.abs(s_ex - s_cg)) < 0.05 * sdy
    _breadcrumb("test end")


def test_prediterative_defaults_to_mean_only():
    X, y = make_data(40)
    k = make_fixed_theta_model(y, X)
    Xt, _ = make_data(5, seed=789)

    mean, stdev = k.predictIterative(Xt)
    assert len(mean) == 5
    assert len(stdev) == 0


def test_prediterative_interpolates_the_training_data():
    X, y = make_data(30)
    k = make_fixed_theta_model(y, X)

    mean, stdev = k.predictIterative(X, True)
    sdy = np.std(y)
    assert np.max(np.abs(mean.flatten() - y)) < 0.05 * sdy
    assert np.max(stdev) < 0.05 * sdy


def test_prediterative_rejects_a_negative_max_iter():
    X, y = make_data(20)
    k = make_fixed_theta_model(y, X)
    Xt, _ = make_data(5, seed=789)

    with pytest.raises(Exception):
        k.predictIterative(Xt, False, -1)


def test_prediterative_nystrom_precond_matches_exact_predict():
    X, y = make_data(60)
    k = make_fixed_theta_model(y, X)
    Xt, _ = make_data(20, seed=456)

    m_ex, s_ex, _, _, _ = k.predict(Xt, True, False, False)
    m_pc, s_pc = k.predictIterative(Xt, True, 0, 1e-8, True, 20)

    sdy = np.std(y)
    assert np.max(np.abs(m_ex - m_pc)) < 0.05 * sdy
    assert np.max(np.abs(s_ex - s_pc)) < 0.05 * sdy


def test_prediterative_rejects_a_negative_precond_rank():
    X, y = make_data(20)
    k = make_fixed_theta_model(y, X)
    Xt, _ = make_data(5, seed=789)

    with pytest.raises(Exception):
        k.predictIterative(Xt, False, 0, 1e-8, True, -1)


def test_subset_of_data_returns_n_max_sorted_0based_indices():
    X, _ = make_data(200)

    idx = lk.Kriging.subsetOfData(X, 20)
    idx = np.asarray(idx).flatten()

    assert len(idx) == 20
    assert np.array_equal(idx, np.sort(idx))
    assert idx.min() >= 0
    assert idx.max() < X.shape[0]


def test_subset_of_data_is_a_noop_when_n_max_covers_all_rows():
    X, _ = make_data(15)

    idx = np.asarray(lk.Kriging.subsetOfData(X, 15)).flatten()

    assert np.array_equal(idx, np.arange(15))
