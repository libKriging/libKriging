# libKriging — Julia (jlibkriging)

```julia
using jlibkriging
```

See `SKILL.md` in this directory for *which* class/options to pick; this
file gives the exact call syntax. `X::Matrix{Float64}` is `n × d` (rows =
observations), `y::Vector{Float64}` length `n`.

## Kriging (noise-free or noisy)

```julia
k = Kriging(y, X, "matern5_2";
            regmodel="constant",   # "constant" | "linear" | "interactive" | "quadratic" | "none"
            normalize=false,
            optim="BFGS",          # "BFGS", "BFGS10" (10 restarts), "none"
            objective="LL",        # "LL" | "LOO" | "LMP" | "LLVecchia" | "LLVecchia(m)" | "LLNystrom" | "LLNystrom(k)"
            noise=nothing)         # nothing | "nugget" | Float64 | Vector{Float64}

p = predict(k, Xnew; return_stdev=true, return_cov=false, return_deriv=false)
# p.mean, p.stdev, p.cov, p.mean_deriv, p.stdev_deriv

s = simulate(k, 10, 123, Xnew; will_update=false)   # (k, nsim, seed, X)
update!(k, y_u, X_u; refit=true)

# Matrix-free CG prediction (predict-only, any exact "LL"/"LOO"/"LMP" fit):
p_cg = predictCG(k, Xnew; return_stdev=true)  # return_stdev default false
# max_iter=0 (default) means 2n; see docs/math/PredictCG.md

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
                 objective="LL",
                 noise=nothing)
predict(wk, Xnew; return_stdev=true)
```

## MLPKriging

```julia
mk = MLPKriging(y, X, [16, 8], 2;    # hidden_dims, d_out
                activation="selu",   # "selu" | "relu" | "tanh" | "sigmoid" | "elu"
                kernel="gauss",
                regmodel="constant",
                normalize=false,
                optim="BFGS+Adam",
                objective="LL")
```
Prefer `activation="tanh"` over the default `"selu"` if a single-start fit
looks unstable: SELU has a non-smooth kink at `z=0` that can make the
likelihood surface locally jagged for a gradient-based optimizer (the
analytic gradient itself is correct — this is an optimization-landscape
issue, not a bug).

## NestedKriging

```julia
nk = NestedKriging(y, X, "matern5_2", 20;   # nb_groups
                   aggregation="NK",         # "PoE" | "gPoE" | "BCM" | "rBCM" | "NK" (default)
                   partition="kmeans",       # "kmeans" | "random"
                   seed=123,
                   warping=String[],         # non-empty -> WarpKriging submodels
                   regmodel="constant",
                   optim="BFGS",
                   objective="LL")
predict(nk, Xnew; return_stdev=true)
```
`aggregation="NK"` requires `regmodel="constant"`. No `noise=`, no
`normalize=`, no save/load yet on `NestedKriging` (v1.1).

## Common pitfalls to flag in review

- Using a `NuggetKriging`/`NoiseKriging` constructor for a new fit.
- Passing `X`/`y` as `Vector{Vector{Float64}}` or a non-`Float64` matrix —
  the ccall FFI layer expects `Matrix{Float64}`/`Vector{Float64}` exactly.
- Assuming `WarpKriging`/`MLPKriging` default to `optim="BFGS"` — their
  default is `"BFGS+Adam"`.
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
