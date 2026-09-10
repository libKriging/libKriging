"""Response surfaces for the GPU comparison benchmark.

`branin`, `hartmann6` and `borehole` are re-used verbatim from
`../comparison/functions.py` (same references: Surjanovic & Bingham,
https://www.sfu.ca/~ssurjano/); `sine_sum` is the d-dimensional
`sum_i sin(2*pi*x_i)` toy used by
`docs/comparisons/libKriging_vs_GPyTorch.ipynb` §4, kept here so this
benchmark's per-evaluation cost can be sanity-checked against that
notebook and `bench/bench-iterative-cuda.cpp` (same design).

All functions take X of shape (n, d) in their native domain and return
y of shape (n,). Domains are `DOMAINS[name]`, an (d, 2) array.
"""
import numpy as np

_COMP = None


def _load_comparison():
    global _COMP
    if _COMP is None:
        import importlib.util
        import os
        p = os.path.join(os.path.dirname(__file__), "..", "comparison", "functions.py")
        spec = importlib.util.spec_from_file_location("_comparison_functions", p)
        _COMP = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_COMP)
    return _COMP


def branin(X):
    return _load_comparison().branin(X)


def hartmann3(X):
    return _load_comparison().hartmann3(X)


def hartmann6(X):
    return _load_comparison().hartmann6(X)


def borehole(X):
    return _load_comparison().borehole(X)


def sine_sum(X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return np.sin(2 * np.pi * X).sum(axis=1)


# sine_sum has no intrinsic dimension; SINE_SUM_DIM sets the one used here
# (d=4, matching libKriging_vs_GPyTorch.ipynb and bench-iterative-cuda.cpp).
SINE_SUM_DIM = 4

FUNCTIONS = {
    "sine_sum": sine_sum,
    "hartmann3": hartmann3,
    "hartmann6": hartmann6,
    "borehole": borehole,
    "branin": branin,
}

DOMAINS = {
    "sine_sum": np.array([[0.0, 1.0]] * SINE_SUM_DIM),
    "hartmann3": np.array([[0.0, 1.0]] * 3),
    "hartmann6": np.array([[0.0, 1.0]] * 6),
    "borehole": np.array([[0.05, 0.15], [100.0, 50000.0], [63070.0, 115600.0],
                          [990.0, 1110.0], [63.1, 116.0], [700.0, 820.0],
                          [1120.0, 1680.0], [9855.0, 12045.0]]),
    "branin": np.array([[-5.0, 10.0], [0.0, 15.0]]),
}
