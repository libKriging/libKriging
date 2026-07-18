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
            objective="LL",        # "LL" | "LOO" | "LMP" | "VLL" | "VLL(m)"
            noise=nothing)         # nothing | "nugget" | Float64 | Vector{Float64}

p = predict(k, Xnew; return_stdev=true, return_cov=false, return_deriv=false)
# p.mean, p.stdev, p.cov, p.mean_deriv, p.stdev_deriv

s = simulate(k, 10, 123, Xnew; will_update=false)   # (k, nsim, seed, X)
update!(k, y_u, X_u; refit=true)

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
                activation="selu",
                kernel="gauss",
                regmodel="constant",
                normalize=false,
                optim="BFGS+Adam",
                objective="LL")
```

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
