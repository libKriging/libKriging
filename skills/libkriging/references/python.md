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
    objective="LL",            # "LL" | "LOO" | "LMP" | "LLVecchia" | "LLVecchia(m)" | "LLNystrom" | "LLNystrom(k)"
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
    objective="LL",       # only "LL": any other value is ignored
    parameters={},
    noise=None,   # None | per-observation variance vector (no "nugget" mode, unlike Kriging)
)
mean, stdev, cov, mean_deriv, stdev_deriv = model.predict(Xnew, return_stdev=True)
```
One spec string per column of `X` (see `SKILL.md` §4). If `X` has string
columns, `WarpKriging` can auto-encode them — but being explicit about the
`warping` spec per column is safer for review.

## MLPKriging

```python
model = lk.MLPKriging(
    y, X,
    hidden_dims=[16, 8],   # MLP layer widths
    d_out=2,               # output (feature-map) dimension fed to the GP kernel
    activation="selu",     # "selu" | "relu" | "tanh" | "sigmoid" | "elu" (default "selu")
    kernel="gauss",
    regmodel="constant",
    normalize=False,
    optim="BFGS",
    objective="LL",
    parameters={},
)
```
Prefer `activation="tanh"` over the default `"selu"` when fitting with a
single-start gradient-based optimizer (`optim="BFGS"`, no restarts): SELU
has a non-smooth kink at `z=0` that can make the likelihood surface locally
jagged and destabilize a single BFGS run (the analytic gradient is still
correct — this is an optimization-landscape issue, not a code bug). `tanh`
is smooth everywhere and a safer default recommendation unless the user
specifically wants SELU's less-saturating behavior for deep/wide MLPs.

## Large designs

```python
# Pre-fit reduction: keep n_max representative rows (k-means centroids snapped
# to real observations). Returns 0-based row indices, shape (n_max, 1).
idx = lk.Kriging.subsetOfData(X, n_max=2000, method="kmeans", seed=123).ravel()
model = lk.Kriging(y[idx], X[idx], "matern5_2")

# Or keep every point and approximate the objective (noise-free Kriging only)
model = lk.Kriging(y, X, "matern5_2", objective="LLVecchia(30)")   # d <~ 5
model = lk.Kriging(y, X, "matern5_2", objective="LLNystrom(50)")   # higher d
model.nystrom_rank()   # 50 (0 if the model was not fitted with LLNystrom)
```
`predict` is the only prediction entry point from Python: `predictVecchia`,
`predictNystrom`, `simulateNystrom` and `set_vecchia_exact_commit` (the "light"
Vecchia mode) exist in C++ only.

## NestedKriging

```python
model = lk.NestedKriging(
    y, X, "matern5_2", nb_groups=20,
    aggregation="NK",       # "PoE" | "gPoE" | "BCM" | "rBCM" | "NK" (default)
    partition="kmeans",     # "kmeans" | "random"
    seed=123,
    regmodel="constant",
    optim="BFGS",
    objective="LL",         # "LLVecchia(m)" to fit the common prior via one global Vecchia fit
    parameters={},
    warping=[],             # non-empty -> submodels are WarpKriging instead of Kriging
)
mean, stdev = model.predict(Xnew, return_stdev=True)
```
`aggregation="NK"` requires `regmodel="constant"`. No `noise=`, no
`normalize=`, no `save()`/`load()` yet on `NestedKriging`.

## scikit-learn estimators

```python
# pip install pylibkriging[sklearn]
from pylibkriging.sklearn import KrigingRegressor   # also WarpKrigingRegressor,
                                                    # MLPKrigingRegressor, NestedKrigingRegressor
est = KrigingRegressor(kernel="matern5_2").fit(X, y)   # sklearn order: (X, y)
mean, std = est.predict(Xnew, return_std=True)         # return_std and return_cov are exclusive
```
Constructor parameters mirror the `Kriging` ones (`regmodel`, `normalize`,
`optim`, `objective`, `noise`, `parameters`), so `get_params` / `set_params` /
`clone`, `Pipeline` and `GridSearchCV` work as for any scikit-learn regressor.

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

## See also

Worked, end-to-end examples fitting the *same* Branin 2D function with
`pylibkriging` and mimicking a specific competitor package are in
`docs/comparisons/`:
`libKriging_vs_SMT.ipynb` (KPLS dimension reduction),
`libKriging_vs_OpenTURNS.ipynb` (joint conditional simulation),
`libKriging_vs_GPy.ipynb` (sparse/inducing-point scaling),
`libKriging_vs_sklearn.ipynb` (`WhiteKernel` noise estimation vs
`noise="nugget"`), and `libKriging_vs_GPflow.ipynb` (HMC full-Bayesian
hyperparameters vs `objective="LMP"`). Each notebook also ends with an
argument-correspondence table between `pylibkriging` and the competitor's
API.
