# libKriging — R (rlibkriging)

```r
library(rlibkriging)
```

See `SKILL.md` in this directory for *which* class/options to pick; this
file gives the exact call syntax. `X` is `n × d` (rows = observations,
a numeric matrix), `y` a numeric vector.

## Kriging (noise-free or noisy)

```r
k <- Kriging(y, X, kernel = "matern5_2",
             regmodel = c("constant", "linear", "interactive", "quadratic", "none"),
             normalize = FALSE,
             optim = "BFGS",              # "BFGS", "BFGS10" (10 restarts), "none"
             objective = "LL",            # "LL" | "LOO" | "LMP"
             parameters = NULL,
             noise = NULL,                # NULL | "nugget" | numeric vector
             method = "exact")            # "exact" | "vecchia" | "vecchia(m)" | "nystrom" | "nystrom(k)" (noise models: LL only)

p <- predict(k, x = Xnew, return_stdev = TRUE, return_cov = FALSE)
#   predict / simulate / logLikelihood also take method = "exact" (default) |
#   "vecchia" | "vecchia(m)" | "nystrom" -- see "Large designs" below
s <- simulate(k, nsim = 10, seed = 123, x = Xnew, will_update = TRUE)   # length(Xnew) x nsim
s_u <- update_simulate(k, y_u, X_u)   # condition those paths on new observations
update(k, y_u, X_u, refit = TRUE)
# heterogeneous noise model (noise = <variance vector>): with_noise / noise_u are
# noise VARIANCES, like the fit's noise; update without noise_u raises an error
# s <- simulate(k, nsim = 10, seed = 123, x = Xnew, with_noise = noise_var_new)
# update(k, y_u, noise_var_u, X_u, refit = TRUE)

save(k, "k.json")      # JSON, any libKriging class
k <- load("k.json")    # auto-detects the class

logLikelihood(k)
leaveOneOut(k)                 # method = "exact" (default) | "vecchia" | "nystrom"
logMargPost(k)                 # idem
```

Do **not** call `NuggetKriging(...)`/`NoiseKriging(...)` for new models —
pass `noise=` to `Kriging()` instead (see `SKILL.md` §1.2); those class
names only remain relevant for loading legacy saved models.

`k` also exposes every constructor argument and fitted hyperparameter as a
method: `k$kernel()`, `k$optim()`, `k$objective()`, `k$method()`, `k$theta()`,
`k$sigma2()`, `k$beta()`, `k$nugget()`, `k$noise()`, etc.

## WarpKriging

```r
wk <- WarpKriging(y, X, warping = c("kumaraswamy", "categorical(5,2)", "none"),
                  kernel = "gauss",
                  regmodel = "constant",
                  normalize = FALSE,
                  optim = "BFGS+Adam",   # different default from Kriging: warp params need Adam-style steps
                  objective = "LL",     # only "LL": any other value is ignored
                  parameters = NULL,
                  noise = NULL,        # NULL | numeric variance vector (no "nugget" mode)
                  method = "exact")    # only "exact" is implemented (same for MLPKriging)
predict(wk, x = Xnew, return_stdev = TRUE)
s <- simulate(wk, nsim = 10, seed = 123, x = Xnew, will_update = FALSE, with_noise = NULL)
# with_noise: NULL / FALSE (latent process) | numeric noise VARIANCES, a scalar or
# one per row of x (same for MLPKriging; negative/NA raise; TRUE raises: no
# "fitted nugget" shortcut as in simulate.Kriging)
```
One spec string per column of `X` (see `SKILL.md` §4). If `X` has string
columns, `WarpKriging` can auto-encode them — but being explicit about the
`warping` spec per column is safer for review.

## MLPKriging

```r
mk <- MLPKriging(y, X, hidden_dims = c(16, 8),
                 d_out = 2,
                 activation = "selu",   # "selu" | "relu" | "tanh" | "sigmoid" | "elu"
                 kernel = "gauss",
                 regmodel = "constant",
                 normalize = FALSE,
                 optim = "BFGS+Adam",
                 objective = "LL",
                 parameters = NULL)
```
Prefer `activation = "tanh"` over the default `"selu"` if the user reports
an unstable/stuck fit with a single-start optimizer: SELU's kink at `z = 0`
can make the likelihood surface locally jagged (the gradient itself is
still correct — this is an optimization-landscape issue, not a bug).

## Large designs

```r
# Pre-fit reduction: keep n_max representative rows (k-means centroids snapped
# to real observations). Returns 1-based row indices.
idx <- subsetOfData(X, n_max = 2000, method = "kmeans", seed = 123)
k <- Kriging(y[idx], X[idx, , drop = FALSE], "matern5_2")

# Or keep every point and approximate the criterion (any objective, noise-free Kriging only)
k <- Kriging(y, X, "matern5_2", method = "vecchia(30)")   # d <~ 5
k <- Kriging(y, X, "matern5_2", method = "nystrom(50)")   # higher d
k$method()         # "nystrom(50)"; k$objective() is "LL"
k$nystrom_rank()   # 50 (0 if the model was not fitted with method = "nystrom")
k$vecchia_neighbors()   # 30 for a method = "vecchia(30)" fit (0 otherwise)
```
An approximate fit keeps no exact factorization, and the method is explicit at every step
(`method = "exact"` by default, never inferred from the fit):
```r
kn <- Kriging(y, X, "matern5_2", method = "nystrom(50)")
p <- kn$predict(Xnew, method = "nystrom")                        # "exact" would raise
s <- kn$simulate(nsim = 10, seed = 123, x = Xnew, method = "nystrom")
kn$logLikelihood(method = "nystrom")                             # the fitted objective
# noisy data: Nystrom (not Vecchia) supports both noise models
knn <- Kriging(y, X, "matern5_2", noise = "nugget", method = "nystrom(50)")
knh <- Kriging(y, X, "matern5_2", noise = noise_var, method = "nystrom(50)")
kv <- Kriging(y, X, "matern5_2", method = "vecchia(30)")
p <- kv$predict(Xnew, method = "vecchia")                        # or method = kv$method()
kv$update(y_u, X_u)                                              # Vecchia refit, still approximate
# the exact model at the Vecchia estimate: a separate exact fit, theta fixed
ke <- Kriging(y, X, "matern5_2", optim = "none",
              parameters = list(theta = matrix(kv$theta(), nrow = 1), is_theta_estim = FALSE))
ke$predict(Xnew, return_cov = TRUE)                              # exact, with covariance
```
The Vecchia predictor (`method = "vecchia"`) is also usable after an exact fit. Approximate
methods return mean / stdev only (`return_cov`, `return_deriv`, `will_update` raise).
`objective = "LLVecchia(m)"` / `"LLNystrom(k)"` are **deprecated** aliases of
`objective = "LL", method = "vecchia(m)"` / `"nystrom(k)"`.
`predictVecchia`, `predictNystrom`, `simulateVecchia` and `simulateNystrom` (behind `method`) exist in C++ only.

## NestedKriging

```r
nk <- NestedKriging(y, X, kernel = "matern5_2", nb_groups = 20,
                     aggregation = "NK",     # "PoE" | "gPoE" | "BCM" | "rBCM" | "NK" (default)
                     partition = "kmeans",   # "kmeans" | "random"
                     seed = 123,
                     regmodel = "constant",
                     optim = "BFGS",
                     objective = "LL",
                     parameters = NULL,
                     warping = NULL,         # non-NULL -> WarpKriging submodels
                     method = "exact")       # "vecchia(m)" / "nystrom(k)": common prior from one global approximate fit
predict(nk, x = Xnew, return_stdev = TRUE)
nk$method(); nk$objective()
```
`aggregation = "NK"` requires `regmodel = "constant"`. No `noise=`, no
`normalize=`. `save(nk, "nk.json")` / `load("nk.json")` work as for the
other classes.

## Common pitfalls to flag in review

- `NuggetKriging()`/`NoiseKriging()` calls for new fits.
- Forgetting `X <- as.matrix(X)` when `X` comes from a `data.frame` — the
  bindings expect a plain numeric matrix.
- `aggregation = "NK"` with `regmodel` other than `"constant"`.
- Assuming `WarpKriging`/`MLPKriging` default to `optim = "BFGS"` like plain
  `Kriging` — their default is `"BFGS+Adam"`.
- Forgetting `y <- as.numeric(y)` when `y` comes from a single-column
  `data.frame`/`matrix` (e.g. from a DOE package) — the bindings expect a
  plain numeric vector, not an `n × 1` object.

## Installation

`rlibkriging` is published on CRAN — `install.packages("rlibkriging")` is
the normal path for a user who only wants to *use* the package. Building
from this repository's sources (`tools/r-linux-macos/build.sh`) is only
needed when developing libKriging itself or testing an unreleased fix.
When writing an example/notebook, prefer a simple availability check
(`requireNamespace("rlibkriging", quietly = TRUE)`, with a `warning()`
suggesting `install.packages(...)` if missing) over embedding a
build-from-source step — see `docs/comparisons/libKriging_vs_DiceKriging.ipynb`
and `libKriging_vs_RobustGaSP.ipynb` for the pattern used across this
repository's comparison notebooks.

## See also

`docs/comparisons/libKriging_vs_DiceKriging.ipynb` (mimicking `km()`'s
default MLE fit, plus the `knots` non-stationary warp vs
`WarpKriging(..., warping = "knots(K)")`) and
`libKriging_vs_RobustGaSP.ipynb` (mimicking `rgasp()`'s default fit, plus
`rgasp(method = "post_mode")`'s robust jointly-robust-prior estimation vs
`Kriging(objective = "LMP")`) are complete, executable, worked examples in
R — including an argument-correspondence table with the competitor
package.
