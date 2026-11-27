"""Settings of docs/Settings.md (#468): set / get round trip of every user
setting, and the precedence rule (LK_* environment read once at import, a
setter call overrides it)."""

import os
import subprocess
import sys

import pytest
import pylibkriging as lk

# (class, setter, getter, value)
SETTINGS = [
    (lk.Optim, "set_max_restart", "get_max_restart", 3),
    (lk.Optim, "set_max_iteration", "get_max_iteration", 57),
    (lk.Optim, "set_gradient_tolerance", "get_gradient_tolerance", 0.0125),
    (lk.Optim, "set_objective_rel_tolerance", "get_objective_rel_tolerance", 0.0025),
    (lk.Optim, "set_theta_lower_factor", "get_theta_lower_factor", 0.25),
    (lk.Optim, "set_theta_upper_factor", "get_theta_upper_factor", 4.5),
    (lk.Optim, "use_variogram_bounds_heuristic", "variogram_bounds_heuristic_used", False),
    (lk.Optim, "use_reparametrize", "is_reparametrized", False),
    (lk.Optim, "set_log_level", "get_log_level", 2),
    (lk.Parallel, "set_num_threads_percent", "get_num_threads_percent", 0.5),
    (lk.Parallel, "set_thread_pool_size", "get_thread_pool_size", 2),
    (lk.Parallel, "set_thread_start_delay_ms", "get_thread_start_delay_ms", 7),
    (lk.Parallel, "set_blas_threads_min_n", "get_blas_threads_min_n", 250),
    (lk.Parallel, "set_omp_min_rows", "get_omp_min_rows", 300),
    (lk.LinearAlgebra, "set_chol_jitter", "get_chol_jitter", 1e-7),
]


@pytest.mark.parametrize("cls,setter,getter,value", SETTINGS,
                         ids=[s[1] for s in SETTINGS])
def test_round_trip(cls, setter, getter, value):
    orig = getattr(cls, getter)()
    try:
        getattr(cls, setter)(value)
        assert getattr(cls, getter)() == value
    finally:
        getattr(cls, setter)(orig)
    assert getattr(cls, getter)() == orig


def test_max_restart_rejects_negative():
    orig = lk.Optim.get_max_restart()
    with pytest.raises(Exception):
        lk.Optim.set_max_restart(-1)
    assert lk.Optim.get_max_restart() == orig


def test_max_restart_fit():
    # a fit runs with restarts disabled, and the setting is kept afterwards
    import numpy as np
    orig = lk.Optim.get_max_restart()
    try:
        lk.Optim.set_max_restart(0)
        X = np.linspace(0, 1, 12).reshape(-1, 1)
        y = np.sin(6 * X[:, 0])
        k = lk.Kriging(y, X, "matern5_2", optim="BFGS")
        assert np.isfinite(k.logLikelihood())
        assert lk.Optim.get_max_restart() == 0
    finally:
        lk.Optim.set_max_restart(orig)


def test_environment_read_at_import_setter_wins():
    code = (
        "import os, pylibkriging as lk\n"
        "assert lk.Optim.get_max_restart() == 4, lk.Optim.get_max_restart()\n"
        "assert lk.Optim.get_max_iteration() == 33\n"
        "assert lk.Parallel.get_omp_min_rows() == 456\n"
        "assert lk.LinearAlgebra.get_chol_jitter() == 1e-8\n"
        "assert not lk.Optim.is_reparametrized()\n"
        # changing the environment after import has no effect
        "os.environ['LK_MAX_RESTART'] = '9'\n"
        "assert lk.Optim.get_max_restart() == 4\n"
        # the setter wins
        "lk.Optim.set_max_restart(1)\n"
        "assert lk.Optim.get_max_restart() == 1\n"
    )
    env = dict(os.environ, LK_MAX_RESTART="4", LK_MAX_ITERATION="33", LK_OMP_MIN_ROWS="456",
               LK_CHOL_JITTER="1e-8", LK_REPARAMETRIZE="0")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_thread_settings_moved_to_parallel():
    # 2.0 (#468): the thread settings are in Parallel only
    for name in ("set_num_threads_percent", "get_num_threads_percent", "get_num_threads",
                 "set_thread_pool_size", "get_thread_pool_size",
                 "set_thread_start_delay_ms", "get_thread_start_delay_ms"):
        assert not hasattr(lk.Optim, name), name
        assert hasattr(lk.Parallel, name), name


def test_invalid_environment_value_warns_and_uses_default():
    code = (
        "import pylibkriging as lk\n"
        "assert lk.Optim.get_max_restart() == 10, lk.Optim.get_max_restart()\n"
        "assert lk.Parallel.get_thread_pool_size() == 0\n"
        "assert lk.Optim.is_reparametrized()\n"
    )
    env = dict(os.environ, LK_MAX_RESTART="abc", LK_THREAD_POOL_SIZE="-2", LK_REPARAMETRIZE="maybe")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout + r.stderr
    for name in ("LK_MAX_RESTART", "LK_THREAD_POOL_SIZE", "LK_REPARAMETRIZE"):
        assert "[WARNING] " + name + "=" in out, out
