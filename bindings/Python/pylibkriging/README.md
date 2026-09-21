# pylibkriging — Python binding for libKriging

[libKriging](https://github.com/libKriging/libKriging) is a C++ library for Kriging / Gaussian process
regression (fit, predict, simulate, update, input warpings, and scalable variants for large designs).
`pylibkriging` exposes it to Python through [pybind11](https://github.com/pybind/pybind11).

## Installation

```shell
pip3 install pylibkriging
```

Requires Python ≥ 3.7 and NumPy (≥ 1.18, NumPy 2 is supported). Pre-built wheels are published for the usual
Linux / macOS / Windows targets; see the
[releases page](https://github.com/libKriging/libKriging/releases) for the other packages.

## Example

`X` is an `n × d` `float64` array (one row per observation) and `y` a length-`n` vector.

```python
import numpy as np
import pylibkriging as lk

X = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
f = lambda x: 1 - 0.5 * (np.sin(12 * x) / (1 + x) + 2 * np.cos(7 * x) * x ** 5 + 0.7)
y = f(X[:, 0])

model = lk.Kriging(y, X, "matern5_2")
print(model.summary())

x = np.linspace(0, 1, 100).reshape(-1, 1)
mean, stdev, cov, mean_deriv, stdev_deriv = model.predict(x, return_stdev=True)

sims = model.simulate(nsim=10, seed=123, X=x)  # shape (100, 10)
```

Other classes: `WarpKriging` (input warpings, categorical / ordinal inputs), `MLPKriging`, `NestedKriging` (large
designs), and scikit-learn compatible estimators in `pylibkriging.sklearn`. See the
[method reference](../../README.md) for all bindings and the
[main README](../../../README.md) for more examples.

## Building from source

Requires a C++17 compiler, CMake ≥ 3.13 and a BLAS/LAPACK implementation. From the repository root (cloned with
`--recurse-submodules`):

```shell
ENABLE_PYTHON_BINDING=on tools/linux-macos/install.sh
ENABLE_PYTHON_BINDING=on tools/linux-macos/build.sh
ENABLE_PYTHON_BINDING=on tools/linux-macos/test.sh
```

or `python3 -m pip install .` from the repository root to build and install the wheel directly. See
[bindings/Python/README.md](../README.md) for the Windows commands and the other options.

To use a rebuilt module in a running interpreter:

```python
import importlib
import pylibkriging as lk
importlib.reload(lk)
```

## Releasing

Wheels are built and uploaded to PyPI by the `release-python` GitHub workflow
(`.github/workflows/release-python.yml`, which calls `tools/release/python-release.sh`) when a release tag is pushed.
