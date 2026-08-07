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
             objective = "LL",            # "LL" | "LOO" | "LMP" | "LLVecchia" | "LLVecchia(m)" | "LLNystrom" | "LLNystrom(k)"
             parameters = NULL,
             noise = NULL)                # NULL | "nugget" | numeric vector

p <- predict(k, x = Xnew, return_stdev = TRUE, return_cov = FALSE)
s <- simulate(k, nsim = 10, seed = 123, x = Xnew)
update(k, y_u, X_u, refit = TRUE)

# Matrix-free CG prediction (predict-only, any exact "LL"/"LOO"/"LMP" fit):
p_cg <- predictCG(k, x = Xnew, return_stdev = TRUE)  # return_stdev default FALSE
# max_iter = 0L (default) means 2n; see docs/math/PredictCG.md

logLikelihood(k)
leaveOneOut(k)
logMargPost(k)
```

Do **not** call `NuggetKriging(...)`/`NoiseKriging(...)` for new models —
pass `noise=` to `Kriging()` instead (see `SKILL.md` §1.2); those class
names only remain relevant for loading legacy saved models.

`k` also exposes every constructor argument and fitted hyperparameter as a
method: `k$kernel()`, `k$optim()`, `k$objective()`, `k$theta()`,
`k$sigma2()`, `k$beta()`, `k$nugget()`, `k$noise()`, etc.

## WarpKriging

```r
wk <- WarpKriging(y, X, warping = c("kumaraswamy", "categorical(5,2)", "none"),
                  kernel = "gauss",
                  regmodel = "constant",
                  normalize = FALSE,
                  optim = "BFGS+Adam",   # different default from Kriging: warp params need Adam-style steps
                  objective = "LL",
                  parameters = NULL,
                  noise = NULL)
predict(wk, x = Xnew, return_stdev = TRUE)
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

## NestedKriging

```r
nk <- NestedKriging(y, X, kernel = "matern5_2", nb_groups = 20,
                     aggregation = "NK",     # "PoE" | "gPoE" | "BCM" | "rBCM" | "NK" (default)
                     partition = "kmeans",   # "kmeans" | "random"
                     seed = 123,
                     regmodel = "constant",
                     optim = "BFGS",
                     objective = "LL",       # "LLVecchia(m)" for large-n common-prior fit
                     parameters = NULL,
                     warping = NULL)         # non-NULL -> WarpKriging submodels
predict(nk, x = Xnew, return_stdev = TRUE)
```
`aggregation = "NK"` requires `regmodel = "constant"`. No `noise=`, no
`normalize=`, no save/load yet on `NestedKriging` (v1.1).

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
