# Diagonal jitter: what each class and method adds to the correlation matrix

State of `perf/thread-budget` after 9efff162 (2026-10-21). Background and decisions:
[CholJitter-TODO.md](CholJitter-TODO.md) (4e, 4g), [parallel.md](parallel.md) 8.6.

## The three mechanisms

| name | value | where it is set | nature |
|---|---|---|---|
| search jitter `J` | `kernel_chol_jitter(kernel)`: gauss 1e-9, matern5_2 1e-10, others 0; `set_chol_jitter(x)` / `LK_CHOL_JITTER` | `Kriging::m_chol_jitter`, `WarpKriging::m_chol_jitter`, saved in the JSON | fixed, the same at every theta |
| `num_nugget` `N` | 1e-10 (`set_num_nugget`) | `LinearAlgebra::num_nugget` | fixed constant |
| retry | `N`·10^k, k = 0..10, added by `safe_chol_lower` when the Cholesky fails or rcond < 1e-18 | `LinearAlgebra::safe_chol_lower_retry` | stepwise: depends on whether a Cholesky fails, so on rounding (the discontinuity the search jitter removes) |

Everything below adds the retry on top when the Cholesky still fails, except where `arma::chol` is
called directly (noted "no retry").

## Kriging, noise-free (`noise = none`)

| method | factorization | diagonal |
|---|---|---|
| fit, `LL` / `LOO` / `LMP` objective (exact method) | `populate_Model` | 1 + `J` |
| `logLikelihood()`, `logLikelihoodFun`, `leaveOneOut()`, `leaveOneOutFun/Vec`, `logMargPost(Fun)` | `make_Model` (objective) | 1 + `J` |
| committed model `m_T` (after BFGS, `optim = "none"`, warm restart) | `commit_model_chol_jitter` / `make_committed_Model` | 1 (exact) |
| `predict` (mean, stdev, cov, deriv) | `m_T` | 1 (exact): interpolates the data |
| `simulate` | `m_T`, then `safe_chol_lower` of the conditional covariance at `X_n` | 1 (exact); conditional covariance: 0 + retry |
| `update(refit = false)` | `update_cholCov` extends `m_T` | 1 (exact) |
| `update(refit = true)` (warm restart) | search on 1 + `J`, then commit | 1 + `J`, then 1 |
| `update_simulate` | `R_uu` exact, eigen-decomposition clamped at 0 | 1 (exact), consistent with `update` |
| `save` / `load` | `chol_jitter` (`J`) written when > 0; `m_T` saved as is | absent → 0 |

## Kriging, `method = "vecchia(m)"` / `"nystrom(k)"` (noise-free)

| method | diagonal |
|---|---|
| Vecchia LL (fit, `logLikelihood`), `predictVecchia`, `simulateVecchia` | neighbour blocks 1 + `J` + `N` |
| Vecchia `LOO` / `LMP` objectives (`build_vecchia_precision`) | neighbour blocks 1 + `N` (**no `J`**) |
| `update` (Vecchia, Nystrom) | the model's objective again (warm restart if `refit`), then the light commit (`commit_approx`): as the rows above |
| Nystrom LL, `predictNystrom`, `simulate` (Nystrom) | landmark block + `N` (no retry); residual `D0 + J`, `D0` floored at `N` |
| Nystrom `LOO` / `LMP` objectives (`build_nystrom_precision`) | landmark block + `N`; residual `D0` floored at `N` (**no `J`**) |
| `simulate` (Nystrom) conditional covariance | eigenvalues floored at max(`N`, sqrt(eps)·max eigenvalue), then `safe_chol_lower` |

The approximate methods keep the search diagonal in their committed state (it comes from the objective):
they do not interpolate exactly, which is in their nature anyway.

## Kriging with a noise model (`nugget`, `noise = vector`)

`J` = 0. The diagonal is that of the model: `alpha`·R + (1 - `alpha`)·I (nugget) or R + diag(noise / sigma2),
for the objective, the committed model, `predict`, `simulate`, `update` and `update_simulate` alike.

## WarpKriging, MLPKriging

`MLPKriging` is a facade over `WarpKriging(mlp_joint)`: same rules. One value, `m_chol_jitter` = `J` if > 0,
else `N`, used by `populate_Model` for everything: objective, committed model, `predict`, `simulate`,
`update` (both forms), with or without observation noise (the noise adds to it). Exceptions:

| method | diagonal |
|---|---|
| `update_simulate` | `R_uu` exact (1, or 1 + noise / sigma2), **no `J`**, while `update` adds it |
| `simulate` conditional covariance | 0 + retry (`simulate_impl`, as Kriging) |
| `load` of a file without `chol_jitter` | `N` (its former constant diagonal) |

So a noise-free WarpKriging does not interpolate exactly: on the python-sdist data (n = 10, gauss,
warping `"none"`) the error at the design points is 5.6e-5 (`J` = 1e-9), 1.0e-5 with `set_chol_jitter(0)`
(`N`); Kriging: 1.4e-11.

## NestedKriging

| step | diagonal |
|---|---|
| fit of the common prior | `Kriging` (or `WarpKriging` when warped) on the data: rules above |
| submodels (`optim = "none"`) | Kriging: committed, 1 (exact); WarpKriging: its `m_chol_jitter` |
| `precompute_nk` group factors | the submodel's: Kriging exact (`safe_chol_lower`), WarpKriging its jitter, else `N` |
| `predict`, PoE family (PoE, gPoE, BCM, rBCM) | each submodel's `predict` |
| `predict`, NK aggregation | group factors as above; aggregation matrix `K_M` + `N` (`solve`, `pinv` fallback, no retry) |

No `simulate` / `update`.

## Inconsistencies left

1. **WarpKriging / MLPKriging**: the final model keeps the search jitter (it does not interpolate:
   5.6e-5 above), unlike Kriging since 4g; and `update_simulate` is exact while `update` is not (the
   situation Kriging had before 4g). Aligning on Kriging would mean a committed exact factorization in
   WarpKriging (a second value, as `m_model_chol_jitter`), keeping `N` for the warpings that collapse rows
   (embedding / ordinal), for which the preemptive nugget was introduced.
2. **Vecchia / Nystrom LOO and LMP** use `N` but not `J`, while their LL uses `J`: for gauss the
   objective diagonal is 1e-10 for LOO / LMP and 1e-9 + 1e-10 for LL.
3. **`simulate` conditional covariance** (Kriging and WarpKriging, exact method) still relies on the
   stepwise retry: at points close to the design (or close to each other) the covariance is singular
   and the added nugget, `N`·10^k, depends on which Cholesky fails, so on rounding. Not measured across
   thread budgets yet; a fixed floor (as `simulateNystrom` does with its eigenvalue floor) would remove
   the step.
4. **NestedKriging NK aggregation** adds `N` to `K_M` regardless of the kernel (no `J`, no retry, `pinv`
   fallback): fine for stability, noted for completeness.
