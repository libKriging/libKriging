"""libKriging allocates from worker threads (OpenMP regions, multistart
std::threads) while the calling Python thread holds the GIL. Those
allocations must not go through numpy/tracemalloc (which takes the GIL):
it used to deadlock. Each case runs in a subprocess with a timeout so a
regression fails instead of hanging the test suite."""
import os
import subprocess
import sys
import textwrap

import pytest

PRELUDE = textwrap.dedent("""
    import numpy as np
    import pylibkriging as lk
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(400, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1]
""")

CASES = {
    # OpenMP pair loop of the NK aggregation
    "nested_nk_predict": "lk.NestedKriging(y, X, 'gauss', 4).predict(rng.uniform(size=(50, 2)), True)",
    # parallel multistart (std::thread workers)
    "warp_multistart_fit": "lk.WarpKriging(y[:100], X[:100], ['kumaraswamy', 'kumaraswamy'], 'gauss',"
                           " 'constant', False, 'BFGS4', 'LL')",
}


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("pyopts", [[], ["-X", "tracemalloc"]])
def test_no_deadlock_in_threaded_paths(name, pyopts):
    env = dict(os.environ)
    env.pop("OMP_NUM_THREADS", None)  # keep the threaded code paths enabled
    code = PRELUDE + CASES[name] + "\nprint('done')\n"
    try:
        out = subprocess.run([sys.executable, *pyopts, "-c", code], env=env, capture_output=True, text=True,
                             timeout=300)
    except subprocess.TimeoutExpired:
        pytest.fail(f"{name}: deadlock (no completion within 300 s)")
    assert out.returncode == 0, out.stderr
    assert "done" in out.stdout
