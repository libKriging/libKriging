# libKriging — Julia (jlibkriging)

```julia
using jlibkriging
```

Installed through [JLibKriging.jl](https://github.com/libKriging/JLibKriging.jl) (the packaged form of this binding),
the module is `JLibKriging` with the same API: `using JLibKriging`.

See `SKILL.md` in this directory for *which* class/options to pick; this
file gives the exact call syntax. `X::Matrix{Float64}` is `n × d` (rows =
observations), `y::Vector{Float64}` length `n`.

## Kriging (noise-free or noisy)

```julia
k = Kriging(y, X, "matern5_2";
            regmodel="constant",   # "constant" | "linear" | "interactive" | "quadratic" | "none"
            normalize=false,
            optim="BFGS",          # "BFGS", "BFGS10" (10 restarts), "none"
            objective="LL",        # "LL" | "LOO" | "LMP"
            noise=nothing,         # nothing | "nugget" | Float64 | Vector{Float64}
            method="exact")        # "exact" | "vecchia" | "vecchia(m)" | "nystrom" | "nystrom(k)" (noise models: LL only)

p = predict(k, Xnew; return_stdev=true, return_cov=false, return_deriv=false)
#   predict / simulate / log_likelihood also take method="exact" (default) |
#   "vecchia" | "vecchia(m)" | "nystrom" -- see "Large designs" below
# p.mean, p.stdev, p.cov, p.mean_deriv, p.stdev_deriv

s = simulate(k, 10, 123, Xnew; will_update=true)   # (k, nsim, seed, X) -> size(Xnew,1) x nsim
s_u = update_simulate(k, y_u, X_u)   # condition those paths on new observations
update!(k, y_u, X_u; refit=true)
# heterogeneous noise model (noise=<variance vector>): with_noise= / noise_u= are
# noise VARIANCES, like the fit's noise; update! / update_simulate without
# noise_u raise an error
# s = simulate(k, 10, 123, Xnew; with_noise=noise_var_new, will_update=true)
# s_u = update_simulate(k, y_u, X_u; noise_u=noise_var_u)
# update!(k, y_u, X_u; noise_u=noise_var_u, refit=true)
# fit! / update! drop the last simulate: simulate(...; will_update=true) again

save(k, "k.json")              # JSON, any libKriging class
k = jlibkriging.load("k.json")  # auto-detects the class

log_likelihood(k)
leave_one_out(k)
log_marg_post(k)
```

Initial/fixed hyperparameters: either individual keywords
(`sigma2=`, `theta=`, `beta=`, `nugget=`, plus matching `is_*_estim=`
booleans) or a single `parameters=Dict(...)` with the same keys — both
forms exist for API consistency with `WarpKriging`/`MLPKriging`, the dict
keys take precedence if both are given.

Do **not** construct a `NuggetKriging`/`NoiseKriging` type for new models —
use `Kriging(...; noise=...)` (see `SKILL.md` §1.2); those names, where
still recognized (e.g. by `load()`), are for legacy saved files only.

## WarpKriging

```julia
wk = WarpKriging(y, X, ["kumaraswamy", "categorical(5,2)", "none"], "gauss";
                 regmodel="constant",
                 normalize=false,
                 optim="BFGS+Adam",   # different default from Kriging
                 objective="LL",     # only "LL": any other value is ignored
                 noise=nothing,      # nothing | Vector{Float64} of variances (no "nugget" mode)
                 method="exact")     # only "exact" is implemented (same for MLPKriging)
predict(wk, Xnew; return_stdev=true)
s = simulate(wk, 10, 123, Xnew; will_update=false, with_noise=nothing)
# with_noise: nothing (latent process) | noise VARIANCES, a scalar or a
# Vector{Float64} with one per row of Xnew (same for MLPKriging; negative/NaN raise)
```

## MLPKriging

```julia
mk = MLPKriging(y, X, [16, 8], 2;    # hidden_dims, d_out
                activation="selu",   # "selu" | "relu" | "tanh" | "sigmoid" | "elu"
                kernel="gauss",
                regmodel="constant",
                normalize=false,
                optim="BFGS",        # default: joint L-BFGS-B on MLP weights + θ
                objective="LL")
```
Prefer `activation="tanh"` over the default `"selu"` if a single-start fit
looks unstable: SELU has a non-smooth kink at `z=0` that can make the
likelihood surface locally jagged for a gradient-based optimizer (the
analytic gradient itself is correct — this is an optimization-landscape
issue, not a bug).

## Large designs

```julia
# Pre-fit reduction: keep n_max representative rows (k-means centroids snapped
# to real observations). Returns 0-based row indices (Vector{Int}): add 1.
idx = subsetOfData(X, 2000; method="kmeans", seed=123) .+ 1
k = Kriging(y[idx], X[idx, :], "matern5_2")

# Or keep every point and approximate the criterion (any objective, noise-free Kriging only)
k = Kriging(y, X, "matern5_2"; method="vecchia(30)")   # d <~ 5
k = Kriging(y, X, "matern5_2"; method="nystrom(50)")   # higher d
jlibkriging.method(k)   # "nystrom(50)" (exported as `method`); objective(k) is "LL"
nystrom_rank(k)   # 50 (0 if the model was not fitted with method="nystrom")
vecchia_neighbors(k)   # 30 for a method="vecchia(30)" fit (0 otherwise)
```
An approximate fit keeps no exact factorization, and the method is explicit at every step
(`method="exact"` by default, never inferred from the fit):
```julia
kn = Kriging(y, X, "matern5_2"; method="nystrom(50)")
p = predict(kn, Xnew; method="nystrom")                       # "exact" would raise
s = simulate(kn, 10, 123, Xnew; method="nystrom")
log_likelihood(kn; method="nystrom")                          # the fitted objective
# noisy data: Nystrom (not Vecchia) supports both noise models
knn = Kriging(y, X, "matern5_2"; noise="nugget", method="nystrom(50)")
knh = Kriging(y, X, "matern5_2"; noise=noise_var, method="nystrom(50)")
kv = Kriging(y, X, "matern5_2"; method="vecchia(30)")
p = predict(kv, Xnew; method="vecchia")                       # or method=jlibkriging.method(kv)
update!(kv, y_u, X_u)                                         # Vecchia refit, still approximate
# the exact model at the Vecchia estimate: a separate exact fit, theta fixed
ke = Kriging(y, X, "matern5_2"; optim="none", theta=theta(kv), is_theta_estim=false)
predict(ke, Xnew; return_cov=true)                            # exact, with covariance
```
The Vecchia predictor (`method="vecchia"`) is also usable after an exact fit. Approximate
methods return mean / stdev only (`return_cov`, `return_deriv`, `will_update` raise).
`objective="LLVecchia(m)"` / `"LLNystrom(k)"` are **deprecated** aliases of
`objective="LL", method="vecchia(m)"` / `"nystrom(k)"`.
`predictVecchia`, `predictNystrom`, `simulateVecchia` and `simulateNystrom` (behind `method`) exist in C++ only.

## NestedKriging

```julia
nk = NestedKriging(y, X, "matern5_2", 20;   # nb_groups
                   aggregation="NK",         # "PoE" | "gPoE" | "BCM" | "rBCM" | "NK" (default)
                   partition="kmeans",       # "kmeans" | "random"
                   seed=123,
                   warping=String[],         # non-empty -> WarpKriging submodels
                   regmodel="constant",
                   optim="BFGS",
                   objective="LL",
                   method="exact")           # "vecchia(m)" / "nystrom(k)": common prior from one global approximate fit
predict(nk, Xnew; return_stdev=true)
jlibkriging.method(nk), objective(nk)
```
`aggregation="NK"` requires `regmodel="constant"`. No `noise=`, no
`normalize=`. `save(nk, "nk.json")` / `load("nk.json")` (or
`load_nested_kriging`) work as for the other classes.

## Common pitfalls to flag in review

- Using a `NuggetKriging`/`NoiseKriging` constructor for a new fit.
- Passing `X`/`y` as `Vector{Vector{Float64}}` or a non-`Float64` matrix —
  the ccall FFI layer expects `Matrix{Float64}`/`Vector{Float64}` exactly.
- Assuming `WarpKriging` defaults to `optim="BFGS"` — its default is
  `"BFGS+Adam"` (`MLPKriging` defaults to `"BFGS"`).
- `aggregation="NK"` with `regmodel != "constant"` on `NestedKriging`.
- **Name clashes with other loaded packages**: `jlibkriging` exports common
  names (`predict`, `update`, `simulate`, ...) that many other Julia
  packages also export (e.g. `GaussianProcesses.jl`, `StatsAPI`,
  `StatsBase`, `ScikitLearnBase`). If more than one such package is
  `using`'d in the same session, Julia raises an `UndefVarError`/ambiguity
  error at the call site rather than silently picking one — qualify the
  call explicitly (`jlibkriging.predict(k, Xnew; ...)`) whenever this can
  happen, e.g. in any notebook that also loads a competitor GP package.
- `jlibkriging` is **not** registered in Julia's General registry (as of
  this writing) — it must be added via
  `Pkg.develop(path="bindings/Julia/jlibkriging")` after building the C++
  core (see `bindings/Julia/README.md`), not `Pkg.add("jlibkriging")`.

## See also

`docs/comparisons/libKriging_vs_GaussianProcessesJL.ipynb` mimics
`GaussianProcesses.jl`'s default `GP`/`Mat52Iso` MLE fit with `Kriging`,
then contrasts `jlibkriging`'s fixed named-kernel catalogue with
`GaussianProcesses.jl`'s composable kernel API (`k1 + k2`, `k1 * k2`),
including a full argument-correspondence table.
