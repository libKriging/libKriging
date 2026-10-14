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
    objective="LL",            # "LL" | "LOO" | "LMP"
    parameters={},
    noise=None,                 # None | "nugget" | per-observation variance vector
    method="exact",            # "exact" | "vecchia" | "vecchia(m)" | "nystrom" | "nystrom(k)" (noise models: LL only)
)

mean, stdev, cov, mean_deriv, stdev_deriv = model.predict(
    Xnew, return_stdev=True, return_cov=False, return_deriv=False)
# predict() always returns this fixed 5-tuple; the boolean flags only
# control whether cov/mean_deriv/stdev_deriv are actually computed
# (empty arrays otherwise), not how many values come back.
# mean / stdev are 1-D, shape (q,); cov is (q, q).
# predict / simulate / logLikelihood / leaveOneOut / leaveOneOutVec / logMargPost
# also take method="exact" (default) | "vecchia" | "vecchia(m)" | "nystrom"
# -- see "Large designs" below.
sims = model.simulate(nsim=10, seed=123, X=Xnew, will_update=True)   # shape (q, nsim)
sims_u = model.update_simulate(y_u, X_u)   # condition those paths on new observations
model.update(y_u, X_u, refit=True)
# heterogeneous noise model (noise=<variance vector>): with_noise= / noise_u= are
# noise VARIANCES, like the fit's noise; update without noise_u raises ValueError
# sims = model.simulate(10, 123, Xnew, with_noise=noise_var_new)
# model.update(y_u, X_u, refit=True, noise_u=noise_var_u)

model.logLikelihood()
model.leaveOneOut()
model.logMargPost()
ll, grad, _hess = model.logLikelihoodFun(theta, return_grad=True, want_hess=False)  # (ll, grad, hess: always empty)
```

Do **not** instantiate `NuggetKriging`/`NoiseKriging` — pass `noise=` to
`Kriging` instead (see `SKILL.md` §1.2). Those class names, if they still
appear, are for `pylibkriging.load(filename)` reading models saved by older
libKriging versions.

Introspection: `model.kernel()`, `.optim()`, `.objective()`, `.method()`, `.theta()`,
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
    method="exact",       # only "exact" is implemented (same for MLPKriging)
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
# to real observations). Returns 0-based row indices, a 1-D array of length n_max.
idx = lk.Kriging.subsetOfData(X, n_max=2000, method="kmeans", seed=123).ravel()
model = lk.Kriging(y[idx], X[idx], "matern5_2")

# Or keep every point and approximate the criterion (any objective, noise-free Kriging only)
model = lk.Kriging(y, X, "matern5_2", method="vecchia(30)")   # d <~ 5
model = lk.Kriging(y, X, "matern5_2", method="nystrom(50)")   # higher d
model.method()         # "nystrom(50)"; model.objective() is "LL"
model.nystrom_rank()   # 50 (0 if the model was not fitted with method="nystrom")
model.vecchia_neighbors()   # 30 for a method="vecchia(30)" fit (0 otherwise)
```
An approximate fit keeps no exact factorization, and the method is explicit at every step
(`method="exact"` by default, never inferred from the fit):
```python
kn = lk.Kriging(y, X, "matern5_2", method="nystrom(50)")
mean, stdev, *_ = kn.predict(Xnew, method="nystrom")            # "exact" would raise
sims = kn.simulate(10, 123, Xnew, method="nystrom")
kn.logLikelihood(method="nystrom")                               # the fitted objective
# noisy data: Nystrom (not Vecchia) supports both noise models
knn = lk.Kriging(y, X, "matern5_2", noise="nugget", method="nystrom(50)")
knh = lk.Kriging(y, X, "matern5_2", noise=noise_var, method="nystrom(50)")
knh.update(y_new, X_new, noise_u=noise_var_new)                  # heterogeneous: new noise required
kv = lk.Kriging(y, X, "matern5_2", method="vecchia(30)")
mean, stdev, *_ = kv.predict(Xnew, method="vecchia")             # or method=kv.method()
kv.update(y_u, X_u)                                              # Vecchia refit, still approximate
# the exact model at the Vecchia estimate: a separate exact fit, theta fixed
ke = lk.Kriging(y, X, "matern5_2", optim="none",
                parameters={"theta": np.atleast_2d(kv.theta()), "is_theta_estim": False})
ke.predict(Xnew, True, True, False)                              # exact, with covariance
```
The Vecchia predictor (`method="vecchia"`) is also usable after an exact fit. Approximate
methods return mean / stdev only (`return_cov`, `return_deriv`, `will_update` raise).
`objective="LLVecchia(m)"` / `"LLNystrom(k)"` are **deprecated** aliases of
`objective="LL", method="vecchia(m)"` / `"nystrom(k)"`.
`predictVecchia`, `predictNystrom`, `simulateVecchia` and `simulateNystrom` (behind `method`) exist in C++ only.

## NestedKriging

```python
model = lk.NestedKriging(
    y, X, "matern5_2", nb_groups=20,
    aggregation="NK",       # "PoE" | "gPoE" | "BCM" | "rBCM" | "NK" (default)
    partition="kmeans",     # "kmeans" | "random"
    seed=123,
    regmodel="constant",
    optim="BFGS",
    objective="LL",
    parameters={},
    warping=[],             # non-empty -> submodels are WarpKriging instead of Kriging
    method="exact",         # "vecchia(m)" / "nystrom(k)": common prior from one global approximate fit
)
mean, stdev = model.predict(Xnew, return_stdev=True)
model.method(), model.objective()
```
`aggregation="NK"` requires `regmodel="constant"`. No `noise=`, no
`normalize=`. `nk.save("nk.json")` / `lk.NestedKriging.load("nk.json")` (or
the generic `lk.load`) work as for the other classes.

## scikit-learn estimators

```python
# pip install pylibkriging[sklearn]
from pylibkriging.sklearn import KrigingRegressor   # also WarpKrigingRegressor,
                                                    # MLPKrigingRegressor, NestedKrigingRegressor
est = KrigingRegressor(kernel="matern5_2").fit(X, y)   # sklearn order: (X, y)
mean, std = est.predict(Xnew, return_std=True)         # return_std and return_cov are exclusive
```
Constructor parameters mirror the `Kriging` ones (`regmodel`, `normalize`,
`optim`, `objective`, `noise`, `parameters`, `method` — which applies to `fit`, `predict`
and `sample_y`), so `get_params` / `set_params` /
`clone`, `Pipeline` and `GridSearchCV` work as for any scikit-learn regressor.

## Loading a saved model

```python
model.save("model.json")     # JSON, any class (Kriging, WarpKriging, MLPKriging, NestedKriging)
model = lk.load("model.json")  # auto-detects class, incl. legacy Nugget/NoiseKriging saves
# models are also picklable (pickle / joblib / multiprocessing)
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
