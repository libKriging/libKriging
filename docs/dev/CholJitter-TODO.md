# Fixed Cholesky jitter — TODO (hand-off 2026-10-19)

Branch `perf/thread-budget` (PR #415, base `master`, includes #414). Work in
progress, **option (a) decided**: a fixed jitter on the diagonal of the training
correlation matrix, **on by default**, per kernel, with
`LinearAlgebra::set_chol_jitter` to disable it. Background, measurements and the
thread-budget work it came from: [parallel.md](parallel.md) (section 8.6),
[ThreadBudget-TODO.md](ThreadBudget-TODO.md),
[bench/thread-budget/RESULTS.md](../../bench/thread-budget/RESULTS.md).

## Constraints (from the user)

- Work locally: **no `git push`, no GitHub Actions** (no `workflow_dispatch`) until
  told otherwise. All commits since `bf8cb8d9` are local only (`git log
  origin/perf/thread-budget..HEAD`).
- Never push to `master`/`main` (AGENTS.md).
- Machine: i9-10900K, 10 cores / 20 threads, Ubuntu 24.04, GCC 13.3, OpenBLAS
  0.3.26 pthreads (`bench/thread-budget/machine.txt`). Stop syncthing while
  benchmarking (the repo is in `~/Sync`).

## 1. Why

A single-start fit of a smooth kernel (gauss, matern5_2) on noise-free data
depends on the rounding (BLAS thread count, BLAS build, CPU): 7–75 % of the data
sets end at a different optimum with 1, 10 or 20 BLAS threads, up to hundreds of
log-likelihood units apart; matern3_2 and exp are bit-identical. Cause: the
optimizer enters the near-singular region of the correlation matrix, where
`LinearAlgebra::safe_chol_lower` adds a stepwise numerical nugget
(`num_nugget` × 10^k when the Cholesky fails or rcond < `min_rcond` = 1e-18):
the objective is noisy and discontinuous there, pass / fail flips on the last
bits, and the L-BFGS-B paths diverge (47 of 52 differing cases already in the
first run; 5 only through the restart triggered by
`ABNORMAL_TERMINATION_IN_LNSRCH`). Not caused by the thread budget, which only
changed the default BLAS count (20 → 10), hence which outcome a user gets. The
open item of #414 (Octave Linux job: wrong gradients and failed Cholesky on an
Intel Xeon 6973P runner only) is probably the same phenomenon.

Experiment (temporary hook, `fit-repro-*.csv`): a fixed jitter makes the
objective continuous, the stepwise nugget never fires, results become
bit-identical across thread counts from **1e-10 (matern5_2)** and **1e-8 / 1e-6
(gauss)**; prediction RMSE on 1000 test points: gauss n = 100, d = 2: 0.10 →
4.9e-4 (most fits without jitter ended at a poor optimum); gauss n = 200, d = 4:
0.147 → 0.154 (1e-6); matern5_2: ×2 worse (3e-4 → 6e-4, 1.5e-3 → 3e-3) because the
model no longer interpolates exactly; matern5_2 n = 100 fit 100 → 8 ms. Full
table: parallel.md 8.6.

## 2. State of the code (uncommitted at hand-off → committed as WIP, see 5)

Implemented (`git show HEAD` after the WIP commit):

- `LinearAlgebra::chol_jitter` (static, default from `LK_CHOL_JITTER`, else -1),
  `set_chol_jitter(x)` / `get_chol_jitter()`: x < 0 automatic, 0 disabled, > 0
  this value for every kernel; `kernel_chol_jitter(kernel)`: automatic values
  gauss 1e-8, matern5_2 1e-10, others 0 (`src/lib/LinearAlgebra.cpp`,
  `src/lib/include/libKriging/LinearAlgebra.hpp`).
- Applied in `Kriging::populate_Model` (`src/lib/Kriging.cpp`) for the
  noise-free mode only: `diag = 1 + jitter` (constant, so theta / alpha gradients
  are unchanged); the Nugget / Heterogeneous modes keep their own diagonal.
  `populate_Model` is the single place where the training R is built for the
  fit, the final model and `update` (`cholCov` / `update_cholCov` in
  `KrigingImpl::populate_Model`).

ctest with the jitter on: **214 / 218** at the first hand-off, 4 failures, all
understood (3 and 4 fixed since, see 3.1; **217 / 219** now with the save/load test, tests 1 and 2 left):

| # | Test | Cause | Planned fix |
|---|---|---|---|
| 1 | `KrigingPredictTest - Check gradient vs finite differences` (tests/KrigingPredictTest.cpp:66, :102) | analytical derivative is right; the fit now reaches a longer range (theta 0.86, 0.53 → 6.56, 1.13), R is conditioned ~1/jitter and the central difference with h = 1e-6 is rounding-noise dominated (FD converges to the analytical value at h = 1e-3: `bench/thread-budget/predict_fd_probe.cpp`) | use h = 1e-4 (or 1e-3) in the test |
| 2 | `Fit: unstable LL (long range with 1D Gauss kernel)` (tests/unstableLLTest.cpp:192, :199) | real statistical change: BFGS now stops at theta = 1.37, LL = -14.01, BFGS10 finds theta = 0.15, LL = +2.29 (identical before: the stepwise nugget acted as a barrier that guided the single start) | decide: change the expectation to "BFGS10 LL >= BFGS LL" and document, or treat it as a blocker (study more cases, see 3.4) |
| 3 | `NestedKriging with one group reduces to plain Kriging` (tests/NestedKrigingTest.cpp:38, :56) | NestedKriging builds some covariance matrices itself, without the jitter: gap 0.026 | apply the same jitter wherever NestedKriging builds a training covariance (`src/lib/NestedKriging.cpp`) |
| 4 | `Nystrom is not sensitive to rounding at the landmark rows (#410)` (tests/KrigingNystromTest.cpp:944, :945) | Nystrom with n landmarks must equal the exact fit to 1e-9; the Nystrom path has no jitter | add the jitter to the Nystrom landmark block (consistency with the exact path) |

All 4 pass with `LK_CHOL_JITTER=0`.

## 3. To do (option a)

1. **Consistency of every path building a training covariance** (otherwise exact,
   Nystrom, Vecchia, Nested, Warp disagree):
   - [x] NestedKriging (test 3): `precompute_nk` factorizes each group with
         `kernel_chol_jitter` when > 0 (else its own 1e-10, also for warped
         submodels); the PoE-family mean check of the test relaxed 1e-8 → 1e-6
         (the expert and the aggregate factorize R separately and R is now
         conditioned ~1e8 for gauss: ~1e-8 gaps), same tolerance as the NK check.
   - [x] Nystrom (test 4): `D = D0 + kernel_chol_jitter` in the noise-free branch
         of `_logLikelihoodNystrom` (the stored `m_nystrom_D` used by
         predictNystrom / simulateNystrom comes from there); k = n reproduces the
         exact model again.
   - [x] Vecchia: observation diagonals `1 + jitter` in the noise-free mode of
         `_logLikelihoodVecchia` (`diag_of`), `predictVecchia` (`RN`) and
         `simulateVecchia` (`var_of`, observations only, not the simulated
         points). 28 / 28 Vecchia tests pass.
   - [x] WarpKriging (decided: align): `populate_Model` adds
         `kernel_chol_jitter` when > 0, else `LinearAlgebra::num_nugget`
         (1e-10) as before (gauss moves 1e-10 → 1e-8, matern5_2 unchanged);
         NestedKriging uses the same rule for warped submodels. The interpolation
         check of `NestedKriging warped hyperparameters come from a subsample
         fit` relaxed 1e-3 → 2e-3 (1.03e-3 measured with 1e-8, passes with 0
         or 1e-10).
   - [x] predict / simulate / update reuse `m_T` / `m_R` from `populate_Model`
         (jittered); the other `covMat_sym_X` calls build R_nn of new points or
         the Nystrom landmark block (latent, no jitter, correct).
   - [x] update_simulate: **no** jitter on the new observations (R_uu), on
         purpose: with it, an update point on the simulation design is no longer
         reproduced exactly (the jitter acts as an observation noise of sd
         sqrt(J)·sigma: 1.7e-5 for matern5_2, `KrigingUpdateSimulateTest`),
         while update() differs from it by J only (1e-10..1e-8 relative).
   - Consequence to document: with the jitter, conditional simulations (and
     predictions) at an observed point are no longer exact but within
     ~sqrt(J)·sigma (1e-5·sigma matern5_2, 1e-4·sigma gauss) and ~J·|y|.
   - [x] Save / load (decided: store it): `KrigingImpl::m_chol_jitter` is set
         by the fit (Kriging: `kernel_chol_jitter` if noise-free, else 0;
         WarpKriging: rule above) and used by every later factorization
         (populate_Model, Nystrom D, Vecchia diagonals), so changing
         `set_chol_jitter` after a fit no longer changes the model; public
         getter `chol_jitter()`. JSON key `chol_jitter`: Kriging writes it only
         when > 0 (files without jitter stay byte-identical; absent → 0, the
         former behaviour); WarpKriging always writes it (absent → current
         `num_nugget`, the former behaviour). Test `fixed Cholesky jitter is
         kept with the model (save & reload)` (tests/KrigingTest.cpp).
2. **Tests**:
   - [ ] Test 1: larger FD step.
   - [ ] Test 2: decide (see 1 above) and adapt.
   - [ ] New test: same single-start fit (gauss, matern5_2, noise-free) with
         `Optim::set_num_threads_percent` 10 / 100 / 200 → bit-identical theta and
         LL (the property the jitter is for).
   - [ ] New test: `set_chol_jitter(0)` restores the former results (regression
         guard) and `LK_CHOL_JITTER` is read.
   - [ ] Full `ctest`, then the Python / R / Octave binding tests locally
         (reference values against DiceKriging / RobustGaSP may move for gauss and
         matern5_2: list every changed expectation and why).
3. **API in the bindings**: expose `set_chol_jitter` / `get_chol_jitter` next to
   `set_num_nugget` (Python `pylibkriging.cpp`, R `bindings/R/rlibkriging/src`,
   Octave `bindings/Octave/mlibkriging`), with tests.
4. **Wider study before release** (`bench/thread-budget/fit_repro_study.cpp`, add
   the jitter hook back as a call to `set_chol_jitter` instead of the removed
   `TMP_JITTER`):
   - [ ] noisy data (Nugget / Heterogeneous unaffected by design, check the
         noise-free fit of slightly noisy data), d ∈ {1, 2, 4, 8, 16}, n up to
         2000, other test functions (Branin, Hartmann, ...), multistart (BFGS10).
   - [ ] how often a single start ends worse than before (test 2 pattern) vs
         better (study: gauss d = 2) — the statistical cost of the default.
   - [ ] calibration (predictive sd / RMSE): gauss is overconfident (0.25–0.43)
         with or without jitter; matern5_2 ≈ 1.
5. **Docs**: parallel.md 8.6 (decision a, values, how to disable), FAQ
   reproducibility paragraph, CHANGELOG (behaviour change for gauss / matern5_2
   noise-free fits, `set_chol_jitter(0)` / `LK_CHOL_JITTER=0` to get the former
   behaviour).
6. **Then**: re-check the #414 Octave item (ThreadBudget-TODO.md, hand-off).

## 4. Remaining thread-budget work (ThreadBudget-TODO.md)

- Step 3: bindings for `Optim::set_num_threads_percent` / `get_num_threads_percent`
  / `get_num_threads` (Python, R, Octave; Julia: none or all thread setters).
- Step 5: tests (pool × share ≤ budget, explicit `OMP_NUM_THREADS` respected, host
  ICV unchanged, results identical for budgets 1 / 2 / 4 / 10, TSan, R fork).
- Step 6: FAQ, bindings READMEs, CHANGELOG.
- Other BLAS (MKL `mkl_set_num_threads_local`, FlexiBLAS, BLIS): parallel.md 8.5.
- macOS #416, Windows #417 (measure `kOmpMinRows` there; it stays 800 off Linux).

## 5. Practical

- Build dir: `build-tb` (Release, current code; `ninja -C build-tb`, `cd build-tb
  && ctest -j8 --timeout 900`; `KrigingWarmRestartTest` is timing-flaky under
  `-j8`, re-run it alone). `build-tb200` / `build-tbnc` are stale (old code,
  compile-time overrides `LK_OMP_MIN_ROWS=200` / `LK_OMP_THREAD_CAP`, the latter
  no longer exists): delete or reconfigure.
- Study / probe drivers are not built by CMake: compile them with the flags of
  `bench-fit-profile` from `build-tb/compile_commands.json`, then
  `g++ -O2 x.o -o x -Lbuild-tb/src/lib -lKriging -Wl,-rpath,$PWD/build-tb/src/lib -fopenmp`
  (see the previous session: replace `-o`/`-c` in that compile command).
- Benchmarks busy-wait 300 ms before measuring (`LK_BENCH_WARMUP_MS`): OpenBLAS
  pthreads threads spin ~100 ms after loading (parallel.md 6.2). Always compare
  configurations after that warm-up, and compare fits by work done (iterations,
  restarts), not only by time: a different BLAS count can change the optimizer
  path.
- A "before" baseline: `git worktree add <dir> bf8cb8d9` (branch start, #414
  head), copy the current `bench/bench-fit-profile.cpp` into it, build
  `bench-fit-profile` there.
