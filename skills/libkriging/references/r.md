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
             objective = "LL",            # "LL" | "LOO" | "LMP" | "VLL" | "VLL(m)"
             parameters = NULL,
             noise = NULL)                # NULL | "nugget" | numeric vector

p <- predict(k, x = Xnew, return_stdev = TRUE, return_cov = FALSE)
s <- simulate(k, nsim = 10, seed = 123, x = Xnew)
update(k, y_u, X_u, refit = TRUE)

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
                 activation = "selu",
                 kernel = "gauss",
                 regmodel = "constant",
                 normalize = FALSE,
                 optim = "BFGS+Adam",
                 objective = "LL",
                 parameters = NULL)
```

## NestedKriging

```r
nk <- NestedKriging(y, X, kernel = "matern5_2", nb_groups = 20,
                     aggregation = "NK",     # "PoE" | "gPoE" | "BCM" | "rBCM" | "NK" (default)
                     partition = "kmeans",   # "kmeans" | "random"
                     seed = 123,
                     regmodel = "constant",
                     optim = "BFGS",
                     objective = "LL",       # "VLL(m)" for large-n common-prior fit
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
