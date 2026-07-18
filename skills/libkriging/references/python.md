# libKriging — Python (pylibkriging)

```python
import pylibkriging as lk
import numpy as np
```

See `SKILL.md` in this directory for *which* class/options to pick; this
file gives the exact call syntax. `X` is `n × d` (rows = observations,
`float64`), `y` is a length-`n` vector.

## Kriging (noise-free or noisy)

```python
model = lk.Kriging(
    y, X, "matern5_2",
    regmodel="constant",       # "none" | "constant" | "linear" | "interactive" | "quadratic"
    normalize=False,
    optim="BFGS",              # "BFGS", "BFGS10" (10 restarts), "none"
    objective="LL",            # "LL" | "LOO" | "LMP" | "VLL" | "VLL(m)"
    parameters={},
    noise=None,                 # None | "nugget" | per-observation variance vector
)

mean, stdev, cov, mean_deriv, stdev_deriv = model.predict(
    Xnew, return_stdev=True, return_cov=False, return_deriv=False)
# predict() always returns this fixed 5-tuple; the boolean flags only
# control whether cov/mean_deriv/stdev_deriv are actually computed
# (empty arrays otherwise), not how many values come back.
sims = model.simulate(nsim=10, seed=123, X=Xnew)
model.update(y_u, X_u, refit=True)

model.logLikelihood()
model.leaveOneOut()
model.logMargPost()
ll, grad = model.logLikelihoodFun(theta, return_grad=True, want_hess=False)
```

Do **not** instantiate `NuggetKriging`/`NoiseKriging` — pass `noise=` to
`Kriging` instead (see `SKILL.md` §1.2). Those class names, if they still
appear, are for `pylibkriging.load(filename)` reading models saved by older
libKriging versions.

Introspection: `model.kernel()`, `.optim()`, `.objective()`, `.theta()`,
`.sigma2()`, `.beta()`, `.nugget()`, `.noise()`, `.is_theta_estim()`, etc.
mirror every constructor argument and fitted hyperparameter 1:1.

## WarpKriging

```python
model = lk.WarpKriging(
    y, X, ["kumaraswamy", "categorical(5,2)", "none"],  # one spec per column of X
    kernel="gauss",
    regmodel="constant",
    normalize=False,
    optim="BFGS",
    objective="LL",
    parameters={},
    noise=None,   # same semantics as Kriging: None | "nugget" | variance vector
)
mean, stdev, cov, mean_deriv, stdev_deriv = model.predict(Xnew, return_stdev=True)

```python
model = lk.MLPKriging(
    y, X,
    hidden_dims=[16, 8],   # MLP layer widths
    d_out=2,               # output (feature-map) dimension fed to the GP kernel
    activation="selu",
    kernel="gauss",
    regmodel="constant",
    normalize=False,
    optim="BFGS",
    objective="LL",
    parameters={},
)
```

## NestedKriging

```python
model = lk.NestedKriging(
    y, X, "matern5_2", nb_groups=20,
    aggregation="NK",       # "PoE" | "gPoE" | "BCM" | "rBCM" | "NK" (default)
    partition="kmeans",     # "kmeans" | "random"
    seed=123,
    regmodel="constant",
    optim="BFGS",
    objective="LL",         # "VLL(m)" to fit the common prior via one global Vecchia fit
    parameters={},
    warping=[],             # non-empty -> submodels are WarpKriging instead of Kriging
)
mean, stdev = model.predict(Xnew, return_stdev=True)
```
`aggregation="NK"` requires `regmodel="constant"`. No `noise=`, no
`normalize=`, no `save()`/`load()` yet on `NestedKriging` (v1.1).

## Loading a saved model

```python
model = lk.load("model.h5")  # auto-detects class, incl. legacy Nugget/NoiseKriging saves
```

## Common pitfalls to flag in review

- `NuggetKriging(...)` / `NoiseKriging(...)` constructor calls for new
  models — use `Kriging(..., noise=...)`.
- `X`/`y` as Python lists instead of NumPy `float64` arrays — pybind11
  bindings expect `numpy.ndarray`.
- `aggregation="NK"` combined with `regmodel != "constant"` on `NestedKriging`.
