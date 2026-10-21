# Diagonal jitter: what each class and method adds to the correlation matrix

State of `perf/thread-budget` after 4h (2026-10-22). Background and decisions:
[CholJitter-TODO.md](CholJitter-TODO.md) (4e, 4g, 4h), [parallel.md](parallel.md) 8.6.

Rule: the search adds `J`; the committed model is exact for Kriging, at most `N` for WarpKriging;
an approximate method keeps the diagonal of its objective.

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
| Vecchia `LOO` / `LMP` objectives (`build_vecchia_precision`) | neighbour blocks 1 + `J` + `N` |
| `update` (Vecchia, Nystrom) | the model's objective again (warm restart if `refit`), then the light commit (`commit_approx`): as the rows above |
| Nystrom LL, `predictNystrom`, `simulate` (Nystrom) | landmark block + `N` (no retry); residual `D0 + J`, `D0` floored at `N` |
| Nystrom `LOO` / `LMP` objectives (`build_nystrom_precision`) | landmark block + `N` (no retry); residual `D0 + J`, `D0` floored at `N` |
| `simulate` (Nystrom) conditional covariance | eigenvalues floored at max(`N`, sqrt(eps)·max eigenvalue), then `safe_chol_lower` |

The approximate methods keep the search diagonal in their committed state (it comes from the objective):
they do not interpolate exactly, which is in their nature anyway.

## Kriging with a noise model (`nugget`, `noise = vector`)

`J` = 0. The diagonal is that of the model: `alpha`·R + (1 - `alpha`)·I (nugget) or R + diag(noise / sigma2),
for the objective, the committed model, `predict`, `simulate`, `update` and `update_simulate` alike.

## WarpKriging, MLPKriging

`MLPKriging` is a facade over `WarpKriging(mlp_joint)`: same rules. Two values, as Kriging:
`m_chol_jitter` (search: `J` if > 0, else `N`) and `m_model_chol_jitter` (committed model:
min(`m_chol_jitter`, `N`)); `m_cache_jitter` is the one of the current cache (the search's while
optimizing). Observation noise adds to either.

| method | diagonal |
|---|---|
| fit / `update(refit = true)` objective, every start (the parallel clones too), `logLikelihoodFun` | 1 + `m_chol_jitter` |
| `logLikelihood()` | the optimized objective (saved as `objective_ll`); after `update(refit = false)`, the committed model's own |
| committed model (`commit_model_chol_jitter`), `predict`, `simulate`, `update(refit = false)` | 1 + min(`m_chol_jitter`, `N`): gauss 1e-10 |
| `update_simulate` | `R_uu` exact (1, or 1 + noise / sigma2) |
| `simulate` conditional covariance | 0 + retry (`simulate_impl`, as Kriging) |
| `load` of a file without `model_chol_jitter` | `m_chol_jitter` (one jitter for both, as saved) |

Not exact, unlike Kriging: an exact factorization lost the predictive variance where warps bring
training rows together in the feature space (Kumaraswamy, saturated tanh MLP: stdev 0, NaN stdev
derivative and NLPD in WarpKrigingPerWarpTest / WarpKrigingCollapseTest). Interpolation error on the
python-sdist data: warping `"none"` 1.0e-5 (5.6e-5 with `J` = 1e-9), Kumaraswamy 1.1e-6.

## NestedKriging

| step | diagonal |
|---|---|
| fit of the common prior | `Kriging` (or `WarpKriging` when warped) on the data: rules above |
| submodels (`optim = "none"`) | their committed model: Kriging 1 (exact), WarpKriging 1 + min(`J`, `N`) |
| `precompute_nk` group factors | the submodel's committed diagonal (`safe_chol_lower`) |
| `predict`, PoE family (PoE, gPoE, BCM, rBCM) | each submodel's `predict` |
| `predict`, NK aggregation | group factors as above; aggregation matrix `K_M` + `N` (`solve`, `pinv` fallback, no retry) |

No `simulate` / `update`.

## Inconsistencies left

1. **WarpKriging / MLPKriging** commit with `N` instead of exactly (see above; a deliberate
   difference), and their `update_simulate` is exact while `update` keeps `N` (a 1e-10 relative
   difference).
2. **`simulate` conditional covariance** (Kriging and WarpKriging, exact method) still relies on the
   stepwise retry: at points close to the design (or close to each other) the covariance is singular
   and the added nugget, `N`·10^k, could depend on which Cholesky fails, so on rounding. Measured
   (2026-10-22, `bench/thread-budget/sim_retry_probe.cpp`, 5 configurations x 10 data sets, budgets
   10 / 100 / 200 %, 50 simulations at 200 random points, optionally + design points / a 1e-7
   cluster): the retry fires in nearly every gauss case (the conditional covariance of 200 smooth
   points is numerically singular) and in matern5_2 only with design points or clusters, always
   at the first level (`N`), never at a different level across budgets; simulations differ by at
   most 4e-11 relative across budgets. In practice the retry acts as a fixed `N` here. A fixed floor
   (as `simulateNystrom`'s eigenvalue floor) would remove the residual risk: rcond values of 2e-18,
   next to the 1e-18 threshold of the rcond check, were seen (gauss n = 10). Not changed.
3. **NestedKriging NK aggregation** adds `N` to `K_M` regardless of the kernel (no `J`, no retry, `pinv`
   fallback): fine for stability, noted for completeness.
