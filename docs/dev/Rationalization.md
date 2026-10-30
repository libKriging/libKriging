# Rationalization program (performance / convergence commits, Sept.–Oct. 2026)

Critical review of the commits that aimed at performance or optimizer robustness
(`eea5f9a` #415, `4482b49`, `7258992` #410, `680e7b6`, `7217956` #382, `8feb1be` / `5f74f26` #407,
`bf53898` #406, `77a5ec2` #428, `7f793ee`), and the program of work it leads to. Each task is
carried by its own pull request, based on the branch of this document.

## 1. Findings

### 1.1 Complexity and code volume

| Indicator | Before (`7f793ee^`) | Now |
|---|---|---|
| `src/lib/Kriging.cpp` | 2873 lines | 5008 lines (part of it: Vecchia / Nystrom, #375) |
| `src/lib/WarpKriging.cpp` | 3216 lines | 3685 lines |
| `Kriging::fit_impl` | | 874 lines in one function |
| `LK_*` environment variables read | | 16, plus compile-time macros (`LK_OMP_MIN_ROWS`, `LK_OMP_LOOPS`, ...) |

- Threading layer: `ThreadBudget`, `TaskPool`, `BlasThreads`, `OmpLoops` (~470 lines, 3 levels:
  pool / OpenMP / BLAS). BLAS control only works with OpenBLAS: with MKL, Accelerate or the
  reference BLAS, `blas_threads_for()` returns 0 and every pool worker keeps all BLAS threads
  (oversubscription in the multistart).
- Restart logic triplicated and already diverging: `Kriging.cpp` (fit_impl loop),
  `WarpKriging.cpp` (joint BFGS), `AdamBFGS.hpp` (inner theta BFGS).
- `compute_ll_grad_theta_vecs` calls a `std::function` returning an `arma::vec` for each of the
  n²/2 pairs; the OpenMP loop parallelizes this cost instead of removing it. The LMP gradient is
  O(d·n³) with the same per-pair calls.
- Multistart memory: `preallocated_models(multistart)` allocates R and L (n×n) for every start,
  and `OptimizationResult` deep-copies L, R, Rinv per start (> 8 GB at n = 5000, BFGS20), while
  the pool bounds the number of concurrent starts.

### 1.2 Convergence stability

- Chain of compensating heuristics: parallelism → thread-count dependent theta → fixed jitter →
  flat long-range LL → log-uniform starts → long-range restart.
- Unfounded constants: `long_range_restart = 2`, `max_long_range_restart = 2`, jitter 1e-9 / 1e-10 /
  0 by kernel, `kOmpMinRows` 200 / 800 by OS, Adam lr 3e-2 / patience 20 / 2 decays, collapse
  guard "10x flatter", Kumaraswamy bounds [1/3, 3], 5 calibration folds, z² clamp >= 1, variance
  floor 4 eps.
- Long-range restart: calibrated on 6–20 data sets per configuration; fires on inactive inputs
  (x 1.8–3.2 time, 50–83 % misses left on `sinhalf` d 8 vs 0 % for BFGS10); active only for
  `multistart == 1`, noise-free, LL. `docs/dev/CholJitter.md` shows BFGS10 (parallel starts)
  faster than the restarted single start at n = 200 and always at the optimum.
- `worse` criterion: a worse restart triggers another one, up to `max_restart` = 10.
- MLPKriging iteration budget changed twice in two days (#382, #407).

### 1.3 Reproducibility and threads

- The Kriging multistart was sequential to keep BFGS20 == best of 20 BFGS1 under MKL; it is now
  parallel and the test is relaxed to 1e-3.
- Parallel and sequential gradients sum in different orders, and the switch (`kOmpMinRows`) is
  200 on Linux and 800 elsewhere: for 200 <= n < 800 the result depends on the OS.
- `ScopedBlasThreads` saves / restores a process-global OpenBLAS count without a lock; `Optim::*`
  are unsynchronized mutable statics; concurrent user threads each get the whole budget.
- `physical_cores()` reads cgroup v2 only.
- Measurements from two machines; macOS (#416) / Windows (#417) not measured; the 190 reference
  CSVs were removed.

### 1.4 Genericity

- Jitter handled differently by Kriging (search with J, exact committed model), WarpKriging
  (committed min(J, num_nugget), `logLikelihood()` = optimized objective) and NestedKriging.
- Per-kernel jitter hard-coded (0 for kernels not listed).
- Fixes validated on one mode only (LL noise-free, joint MLP excluded from log-uniform starts...).
- Settings unevenly exposed in the bindings (long-range restart: environment only).

### 1.5 Hidden cost of the MLPKriging calibration

- 5-fold calibration on by default: ~6x the CPU work, hidden in wall time by the pool on 12 cores,
  x5–10 on 1–2 cores. Factor clamped >= 1: it can only widen the variance.

## 2. Tasks

Each task has its own branch and pull request, based on this branch. Acceptance criteria are
mandatory; measurements go in the task's log `docs/dev/rationalization/T<n>-*.md`.

| # | Task | Branch suffix | Depends on |
|---|---|---|---|
| T1 | Vectorized LL / LMP theta gradients | `-t1-vectorized-gradients` | |
| T2 | Single restart policy (Kriging, WarpKriging, AdamBFGS) | `-t2-restart-policy` | |
| T3 | Default multistart vs single start + long-range restart | `-t3-default-multistart` | T7 (benchmarks), T2 |
| T4 | KModel buffers per worker, not per start | `-t4-kmodel-per-worker` | |
| T5 | OS-independent OpenMP threshold, non-OpenBLAS backends | `-t5-omp-threshold-blas` | T1 (may remove the need) |
| T6 | Thread safety of thread-count scopes and `Optim` settings | `-t6-thread-safety` | |
| T7 | Statistical benchmark base for the thresholds | `-t7-benchmark-base` | |
| T8 | MLPKriging calibration cost | `-t8-mlp-calibration-cost` | |

### T1. Vectorized LL / LMP theta gradients

- Replace the per-pair `std::function` / `arma::vec` calls of `compute_ll_grad_theta_vecs` and
  `compute_lmp_theta_ans` by per-dimension matrices (D_k = dlnCov_k over the pairs, built once per
  evaluation or with a kernel-level vectorized derivative): term2_k = −Σ Rinv ∘ R ∘ D_k,
  term1_k = xᵀ (R ∘ D_k) x.
- One summation order whatever the thread count and the OS.
- Acceptance: gradients equal to the current ones to 1e-10 relative on well-conditioned R (all
  kernels, d 1–10); finite-difference checks pass; single-thread time per gradient at least x2
  lower at n = 500 and n = 1500, d = 4; memory stated (d·n² doubles or a blocked variant).

### T2. Single restart policy

- One private component (e.g. `src/lib/RestartPolicy.hpp`) holding the criteria (abnormal stop,
  `num_iters <= 2`, at lower bound, no improvement within `factr*eps`, optional long-range) and
  the restart point, used by `Kriging::fit_impl`, `WarpKriging` joint BFGS and `AdamBFGS`.
- Bound the cascade of the `worse` criterion (a worse restart does not trigger another one by
  itself).
- Acceptance: no behaviour change other than the documented `worse` bound; all tests pass; the
  three copies removed.

### T3. Default multistart vs single start

- Using T7's protocol, compare `BFGS` (with and without long-range restart) and `BFGS4` / `BFGS10`
  with parallel starts: miss rate, RMSE, wall and CPU time, on 1, 4 and all cores.
- Decide: change the default optimizer and drop the long-range restart, or keep it with
  measured justification (including matern5_2, nugget, LOO / LMP).
- Acceptance: decision documented with confidence intervals; bindings defaults and docs aligned.

### T4. KModel buffers per worker

- Allocate the n×n buffers per pool worker (min(multistart, pool size)), keep only the best
  start's matrices (or refactorize at the optimum).
- Acceptance: identical results; peak memory measured before / after (n = 2000, BFGS20);
  no deep copy of L, R, Rinv for non-best starts.

### T5. OpenMP threshold and BLAS backends

- Make the parallel / sequential paths give the same result (after T1) or use one threshold on
  every OS; document or implement the thread control for MKL (`mkl_set_num_threads_local`) and
  Accelerate; read cgroup v1 quotas.
- Acceptance: same fitted theta on Linux and macOS CI for n in [200, 800); no oversubscription
  under MKL in the multistart (measured or documented limitation).

### T6. Thread safety

- Make `ScopedBlasThreads` / `BudgetScope` safe under concurrent calls from user threads
  (reference counting or a mutex-protected global state); make `Optim::*` settings atomic or
  document them as configuration-before-use.
- Acceptance: stress test with concurrent `fit` / `predict` from several threads (no crash, final
  thread count restored, results equal to the sequential ones); TSan clean.

### T7. Statistical benchmark base

- Reusable driver (in `bench/`) running a fit protocol over >= 30 data sets per configuration:
  gauss and matern5_2, noise-free / nugget, LL / LOO / LMP, active and inactive inputs,
  n 100–1000; reports miss rate with Wilson intervals, RMSE, wall and CPU time.
- Reference results committed (compressed CSV or CI artifact), with the machine description.
- Acceptance: one command reproduces the tables; used by T3.

### T8. MLPKriging calibration cost

- Make the calibration opt-in or cheaper by default (K = 3, reduced per-fold budget, warm start
  from the fitted parameters); review the clamp >= 1 (allow < 1 or justify).
- Acceptance: CPU time of the default MLPKriging fit measured on 1 and 4 cores before / after;
  calibration quality (coverage of 95 % intervals) measured on the `MLPKrigingCalibrationTest`
  cases and >= 10 more data sets.
