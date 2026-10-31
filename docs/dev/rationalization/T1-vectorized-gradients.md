# T1. Vectorized LL / LMP theta gradients — work log

Task specification: `docs/dev/Rationalization.md`, section T1.

## Status

Done. All acceptance criteria met (table below); x2 on the end-to-end LL
gradient is only reached thanks to a change outside the pair loop
(`LinearAlgebra::inv_sympd`), see decision 4.

| Criterion | Result |
|---|---|
| Gradients equal to the former ones to 1e-10 relative, well-conditioned R, all kernels, d 1-10 | 2.5e-12 worst (cond(R) < 1e8) |
| FD gradient tests, full ctest | pass (226/226) |
| Single-thread gradient time x2 lower, n = 500 and 1500, d = 4 | LL x2.2-3.7, LMP x10-15 (pair loop alone: LL x2.5-3.7) |
| One summation order whatever threads / OS | yes for the pair loops (bitwise, 1/3/4 threads); BLAS calls excluded |
| Memory stated | section Memory |

## Changes

- `src/lib/SeparableDln.hpp` (new, private): one-dimension factors of the
  kernels' d ln k / d theta (`gauss`, `exp`, `matern3_2`, `matern5_2`). The
  kernels are separable, component k of `Covariance::DlnCovDtheta_*(dX, theta)`
  only depends on `(dX[k], theta[k])`; the theta-only constants (1/theta,
  1/theta^3, sqrt(5)/theta, ...) are computed once per dimension and per
  evaluation, so a pair costs at most one division per dimension (one to three
  in `Covariance.cpp`).
- `KrigingImpl.cpp`: one pair loop `sum_pairs_dln<NW>` shared by both
  gradients: `out[l*d + k] = sum_{j<i} W(i,j)[l] * D_k(i,j)`, the kernel
  dispatched once per call (template, inlined), no `std::function` and no
  `arma::vec` per pair, contiguous accesses (column i of the symmetric R,
  R^-1, S instead of row i).
  - LL: `W = (2 x_i x_j R_ij, -2 Rinv_ij R_ij)` (term1_k, term2_k).
  - LMP: algebraic reduction of the former `Wb_k = (L^-T L^-1 G_k)' - G_k P`
    (`G_k = dR/dtheta_k`, `P = Rinv_X_Xt_Rinv_X_inv_Xt_Rinv`) to `G_k M`,
    `M = R^-1 - P`, hence
    `ans_k = -tr(G_k M)/2 + (M y)' G_k Q / (2 sigma2) = sum_{j<i} G_k(i,j) [-(M_ij + M_ji)/2 + (u_i q_j + u_j q_i)/(2 sigma2)]`,
    `u = M y`: one R^-1 (O(n^3), reused from `m.Rinv` when cached) and one
    O(d n^2) loop, instead of d pairs of triangular solves with n right-hand
    sides (O(d n^3)) and d * n^2/2 `std::function` calls each returning d
    values of which one was used (O(d^2 n^2)).
- `LinearAlgebra::inv_sympd(L)`: LAPACK `*potri` on the Cholesky factor
  (2n^3/3 flops) instead of two triangular solves on the identity (2n^3 flops);
  the former code is kept as fallback (`info != 0`). Reads the lower triangle
  of L only, as before.

## Summation order

The pair sum is taken row by row (j ascending), then over the rows (i
ascending), for every n, thread count and OS. OpenMP only spreads the rows
over the threads from n = `kOmpMinRows` on, which does not change the result:
the `kOmpMinRows` order dependence of the gradient (200 on Linux, 800
elsewhere) is removed. Checked: gradients bitwise identical with
`OMP_NUM_THREADS` = 1, 3, 4 at n = 150 (< kOmpMinRows), 300, 1000 (gauss,
matern5_2, d = 4, `OPENBLAS_NUM_THREADS=1`). Not covered by T1: the BLAS / LAPACK
calls around the loop (R^-1, triangular solves) still depend on the BLAS
thread count and vendor.

## Measurements

Machine: 4 vCPU Intel Xeon @ 2.80 GHz (cloud VM, shared), Ubuntu 24.04,
GCC 13, `-O3`, OpenBLAS 0.3.26 pthreads, `OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1`. Harness: `Kriging(y, X, kernel, "linear", false,
"none", "LL", theta fixed)`, X ~ U[0,1]^d (seed 123), y = sum sin(3 X),
theta_k = (0.05 + 0.1 k/(d-1)) sqrt(d); median over 7 runs. "Before" is the
parent commit of this branch (`13cace6`), "after" this branch.

Raw data: `T1/timing-1t.csv`; harness and scripts: `T1/` (`gradcheck.cpp`,
`mk.sh` builds it against a build tree, `timing.py`, `cmp.py`). The VM is
shared: the same configuration varies by up to ~30 % between runs (e.g. the
no-gradient times, unchanged by this task, differ by that much between the
two columns); ratios below x1.3 are not significant.

Gradient time = time of the call with gradient minus time without (ms):

| n | kernel | LL before | LL after | ratio | LMP before | LMP after | ratio |
|---|---|---|---|---|---|---|---|
| 500 | matern5_2 | 25.2 | 11.6 | x2.2 | 103.0 | 10.1 | x10 |
| 500 | gauss | 25.3 | 11.7 | x2.2 | 121.1 | 9.4 | x13 |
| 500 | exp | 22.9 | 6.2 | x3.7 | 102.7 | 9.5 | x11 |
| 500 | matern3_2 | 22.7 | 9.2 | x2.5 | 103.4 | 9.5 | x11 |
| 1500 | matern5_2 | 311.6 | 132.9 | x2.3 | 1625.6 | 122.3 | x13 |
| 1500 | gauss | 265.1 | 110.0 | x2.4 | 1585.4 | 114.0 | x14 |
| 1500 | exp | 352.9 | 110.2 | x3.2 | 1724.9 | 124.9 | x14 |
| 1500 | matern3_2 | 330.6 | 97.5 | x3.4 | 1638.3 | 107.0 | x15 |

Pair loops alone (`bench=true` timings: `gradR computation` for LL; for LMP
before `gradR_k` + `Wb_k` + `ans[k]`, after `u = M y, S` + `ans[k]`, R^-1 apart):

| n | kernel | LL loop before | after | ratio | LMP before | after (+ R^-1) |
|---|---|---|---|---|---|---|
| 500 | matern5_2 | 5.74 | 2.12 | x2.7 | 118.5 | 2.5 (+ 7.0) |
| 500 | gauss | 4.95 | 2.01 | x2.5 | 114.7 | 2.2 (+ 6.9) |
| 500 | exp | 4.72 | 1.57 | x3.0 | 112.3 | 1.9 (+ 6.6) |
| 500 | matern3_2 | 4.99 | 1.78 | x2.8 | 107.5 | 2.1 (+ 6.9) |
| 1500 | matern5_2 | 49.5 | 14.9 | x3.3 | 1650.7 | 22.3 (+ 80.2) |
| 1500 | gauss | 41.2 | 14.6 | x2.8 | 1605.6 | 21.6 (+ 82.8) |
| 1500 | exp | 46.7 | 12.7 | x3.7 | 1670.5 | 20.6 (+ 82.6) |
| 1500 | matern3_2 | 47.6 | 13.4 | x3.6 | 1640.4 | 23.4 (+ 83.0) |

Intermediate step (loop change without `*potri`, n = 1500, matern5_2): LL
gradient 297 → 283 ms (x1.05), R^-1 235 ms; with `*potri` R^-1 is 80 ms.
Multi-thread times were not measured (the loop is parallelized as before).

## Accuracy

`cmp.py` compares the gradients of the two builds (`logLikelihoodFun` and
`logMargPostFun`, `"linear"` trend) for n = 60, 300, 1000, the 5 kernels and
d = 1..10 (300 gradients of each), with cond(R) (2-norm, SVD) computed
alongside:

- cond(R) < 1e8 (276 cases, max cond 2.9e7): worst component-relative
  difference 2.6e-14 (LL) and 2.5e-12 (LMP). Criterion 1e-10 met.
- Larger differences only occur for cond(R) >= 1.4e7 and grow with it, up to
  9.6e-6 (LMP, matern5_2, d = 1, n = 1000, cond 3e18). These are 1-2 D designs
  with n up to 1000 where R is numerically singular: both versions are
  accurate to about cond(R)·eps there, neither is a reference. On such R the
  gradient (hence the optimum reached) depends on the rounding order, as it
  already depended on `kOmpMinRows` before.
- Nugget model (no FD test of its LMP gradient in the suite): `gradcheck
  <n> <d> <kernel> nugget`, n = 80, d = 3, gauss / matern5_2 / exp, gamma =
  (theta, alpha = 0.8): LMP and LL gradients equal to the former ones to
  ~1e-15 relative and to a central finite difference to 1e-8.
- Finite-difference gradient tests (`KrigingLogLikTest`, `NuggetKrigingLogLikTest`,
  `NoiseKrigingLogLikTest`; LMP gradients in `KrigingFitTest`, `NuggetKrigingFitTest`) and the full ctest suite pass (226 tests,
  Release, OpenBLAS). One run of `ctest -j3` failed `KrigingWarmRestartTest`
  on its wall-time check (`warm_ms < 0.8 cold_ms`) under CPU contention; it
  passes alone (3/3) and in the `-j2` run: timing flakiness, unrelated.

## Memory

- LL loop: NW·d·n = 2·d·n doubles of row partial sums (the former parallel
  path already used them) and d doubles per row. No D_k matrix: the
  "d·n² doubles" option of the specification is not needed, since D_k is
  evaluated on the fly from `m_dX` (already d·n² doubles, unchanged).
- LMP: R^-1 (n², only when `m.Rinv` is not cached) and S = M + M' (n², plus
  one n² temporary for P' while it is built), u (n); R^-1 is released before
  the loop. Before: per k, gradR_k (n²) and the solve temporaries (2–3 n²).
  Peak about the same (≈ 3 n² beside P), at n = 1500 ≈ 54 MB.
- `inv_sympd`: copy of L (n²) and the symmetrized result (n²), against the
  identity, the intermediate solve and the result (3 n²) before.

## Decisions

1. **No D_k matrices.** Building d matrices of n² doubles (or blocks) to call
   BLAS-like element-wise operations brings nothing here: the per-pair work is
   a few flops, the cost is the kernel evaluation itself (divisions) and memory
   traffic. Evaluating D_k on the fly in the pair loop keeps the extra memory
   at O(d n).
2. **Where the time was.** Measured before the change, the `std::function`
   call and the per-pair `arma::vec` were not the main cost (an `arma::vec` of
   d <= 16 uses Armadillo's local storage, no heap allocation): replacing them
   by inlined kernels with the same formulas gave only x1.5 on the LL loop
   (47 → 32 ms, n = 1500, matern5_2). The remaining x2.3 comes from the
   precomputed theta constants (divisions) and contiguous accesses. Hence the
   new per-dimension formulas: they round differently (a few ulps per value)
   from `Covariance.cpp`, which is left unchanged (also used by Vecchia,
   Nystrom, WarpKriging, the X-gradients).
3. **LMP algebraic reduction** rather than vectorizing the former loop: the
   d n^3 solves were 90 % of the LMP gradient; vectorizing the pair loop only
   would have saved ~10 %.
4. **`inv_sympd` via `*potri`.** After the loop change the end-to-end LL
   gradient (`logLikelihoodFun(theta, true)` minus `logLikelihoodFun(theta,
   false)`) was only x1.05–1.16 faster: R^-1 (235 ms of ~300 ms at n = 1500)
   dominates it and is not part of the pair loop. The x2 criterion could thus
   only be met on the pair loop itself. `*potri` divides the R^-1 time by ~3
   (235 → 80 ms), a change in `LinearAlgebra.cpp` that also benefits the other
   users of `inv_sympd` (WarpKriging, Vecchia / Nystrom inner solves). It
   changes the rounding of R^-1 (κ(R)·eps level). Kept in this task because
   the criterion is about the gradient time; can be split out if preferred.
5. **Generic fallback removed.** The kernel set is closed
   (`Covariance::resolve` throws for any other name, and `make_Cov` sets
   `m_covType` and `_DlnCovDtheta` together): an unknown `m_covType` throws
   instead of falling back to `_DlnCovDtheta`.
6. **Not done.** The Nugget dimension of the LMP gradient (`Kriging.cpp`,
   `gradR_d = R / alpha`) still does two n^3 triangular solves; the same
   reduction (`tr(G M)`, `(M y)' G Q`) applies and would remove them (follow-up,
   NuggetKriging LMP only). `WarpKriging.cpp` and the Vecchia / Nystrom
   gradients keep their own per-pair `_DlnCovDtheta` calls.
