"""fit / predict / simulate / update run the libKriging call without the GIL
(call_guard.hpp): other Python threads keep running, calls on different models
run at the same time, calls on the same model are serialized. Each case runs in
a subprocess with a timeout so that a deadlock fails instead of hanging."""
import os
import subprocess
import sys
import textwrap

import pytest

PRELUDE = textwrap.dedent("""
    import threading, time, warnings
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    import pylibkriging as lk

    def data(seed, n=150, d=3):
        rng = np.random.default_rng(seed)
        X = rng.uniform(size=(n, d))
        return np.sin(3 * X[:, 0]) + X[:, 1] * X[:, -1], X
""")

CASES = {
    # The Python main thread keeps running while a fit runs in another thread
    # (with the GIL held by the call, it would not get any time slice).
    "gil_released_during_fit": """
        y, X = data(1, n=500, d=4)
        done = threading.Event()
        def work():
            lk.Kriging(y, X, "matern5_2", "constant", False, "BFGS10", "LL")
            done.set()
        t = threading.Thread(target=work)
        t.start()
        time.sleep(0.05)  # let the fit start
        ticks = 0
        while not done.is_set():
            ticks += 1
            time.sleep(0.001)
        t.join()
        print("ticks", ticks)
        assert ticks >= 20, ticks
    """,
    # Different models from a thread pool: same results as one after the other
    # (up to the BLAS rounding of the budget split, docs/dev/parallel.md 3.1).
    "concurrent_models_match_sequential": """
        sets = [data(s) for s in range(4)]
        def fit(i):
            y, X = sets[i]
            k = lk.Kriging(y, X, "gauss", "constant", False, "BFGS", "LL")
            return k.theta().ravel(), k.predict(X[:10], True, False, False)[0], k.simulate(3, 7, X[:10])
        seq = [fit(i) for i in range(4)]
        with ThreadPoolExecutor(4) as ex:
            par = list(ex.map(fit, range(4)))
        for (t1, p1, s1), (t2, p2, s2) in zip(seq, par):
            np.testing.assert_allclose(t2, t1, rtol=1e-3)
            np.testing.assert_allclose(p2, p1, rtol=1e-3, atol=1e-6)
            assert s2.shape == s1.shape and np.all(np.isfinite(s2))
    """,
    # The same model from several threads: serialized, every result equal to
    # the single-thread one.
    "same_model_serialized": """
        y, X = data(2)
        k = lk.Kriging(y, X, "gauss")
        Xn = data(3, n=200)[1]
        ref = k.predict(Xn, True, True, False)
        def pred(_):
            return k.predict(Xn, True, True, False)
        with ThreadPoolExecutor(8) as ex:
            res = list(ex.map(pred, range(32)))
        for r in res:
            for a, b in zip(r, ref):
                np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-15)
        def upd(i):
            k2 = k.copy()
            k2.update(np.array([0.1 * i]), np.array([[0.5, 0.5, 0.5]]), False)
            return k2.predict(Xn[:5], False, False, False)[0]
        with ThreadPoolExecutor(4) as ex:
            assert all(np.all(np.isfinite(r)) for r in ex.map(upd, range(8)))
    """,
    # Every class goes through the guard: Warp / MLP / Nested fit and predict
    # from a thread pool.
    "other_classes": """
        y, X = data(4, n=120, d=2)
        def run(kind):
            if kind == "warp":
                m = lk.WarpKriging(y, X, ["kumaraswamy", "none"], "gauss", "constant", False, "BFGS2", "LL")
            elif kind == "mlp":
                m = lk.MLPKriging(y, X, [4])
            else:
                m = lk.NestedKriging(y, X, "gauss", 3)
            p = m.predict(X[:5], True)
            return np.all(np.isfinite(p[0]))
        with ThreadPoolExecutor(3) as ex:
            assert all(ex.map(run, ["warp", "mlp", "nested"]))
    """,
    # The deprecation handler re-acquires the GIL inside the call (warning, and
    # exception under an "error" filter).
    "deprecation_handler_without_gil": """
        y, X = data(5, n=40, d=2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lk.Kriging(y, X, "matern5_2", objective="LLVecchia(7)")
        assert any(issubclass(x.category, DeprecationWarning) for x in w)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            try:
                lk.Kriging(y, X, "matern5_2", objective="LLVecchia(7)")
                raise AssertionError("no DeprecationWarning raised")
            except DeprecationWarning:
                pass
    """,
}


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("pyopts", [[], ["-X", "tracemalloc"]])
def test_gil_release(name, pyopts):
    env = dict(os.environ)
    env.pop("OMP_NUM_THREADS", None)  # keep the threaded code paths enabled
    code = PRELUDE + textwrap.dedent(CASES[name]) + "\nprint('done')\n"
    try:
        out = subprocess.run([sys.executable, *pyopts, "-c", code], env=env, capture_output=True, text=True,
                             timeout=300)
    except subprocess.TimeoutExpired:
        pytest.fail(f"{name}: deadlock (no completion within 300 s)")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "done" in out.stdout
