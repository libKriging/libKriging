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
understood (1, 3 and 4 fixed since, see 3.1 / 3.2 / 4c; **218 / 219** now with the save/load test, test 2 left):

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
   - [x] Test 1: FD step h = 1e-6 → 1e-3 (both sections, see 4c), passes.
   - [x] Test 2 (decided 2026-10-20: option 1): the "rcond failover" section
         runs with `set_chol_jitter(0)` (the jitter bypasses the failover); a
         new section checks the default: BFGS10 finds theta = 0.1506 and
         LL(BFGS10) >= LL(BFGS). Same in R (`test-unstableLL.R`). The
         single-start robustness (option 2: starting-point strategy) is
         issue #418. ctest 219 / 219.
   - [ ] New test: same single-start fit (gauss, matern5_2, noise-free) with
         `Optim::set_num_threads_percent` 10 / 100 / 200 → bit-identical theta and
         LL (the property the jitter is for).
   - [ ] New test: `set_chol_jitter(0)` restores the former results (regression
         guard) and `LK_CHOL_JITTER` is read.
   - [ ] Full `ctest`, then the Python / R / Octave binding tests locally
         (reference values against DiceKriging / RobustGaSP may move for gauss and
         matern5_2: list every changed expectation and why).
3. **API in the bindings** — done (2026-10-20):
   - Python: class `LinearAlgebra` (static `get/set_chol_jitter`,
     `get/set_num_nugget`) in `pylibkriging.cpp`;
   - R: internal `rlibkriging:::linalg_get/set_chol_jitter` in
     `src/linalg.cpp` (as `linalg_set_num_nugget`);
   - Octave: class `LinearAlgebra.m` (same four methods), routes
     `LinearAlgebra::*` in `mLibKriging.cpp`, `LinearAlgebraBinding` in
     `Optim_binding.cpp`;
   - Julia: `get/set_chol_jitter`, `get/set_num_nugget` (exported functions;
     C API `lk_linalg_*` in `csrc/libkriging_c.{h,cpp}`), test
     `tests/chol_jitter_test.jl`; Julia ctest 12 / 12;
   - not exposed: `Kriging.chol_jitter()` / `model_chol_jitter()` getters.
   - The three `binding_consistency` tests (references from DiceKriging, exact
     LL / LOO at theta = 0.3, 1e-12) now run with `set_chol_jitter(0)` and
     restore it; they also check the setter.
   - Local runs: Python (build-py, `-DPYTHON_EXECUTABLE=$(which python3)`,
     else CMake picks /usr/bin/python3.7 without pytest) 20 / 20; Octave
     (build-oct, static) 16 / 17 — the Flatpak Octave cannot load the mex
     (libblas.so.3 / liblapack.so.3 not in the sandbox): local workaround
     `build-oct/flatpak-shim` (links to /app/lib/libopenblas.so.0 and an
     octave wrapper with `--env=LD_LIBRARY_PATH`, set as OCTAVE_EXECUTABLE);
     the remaining failure, `mLibKriging-Demo`, aborts in the plotting after
     simulate (lines 55-60 plot even under GITHUB_ACTION), no display in the
     sandbox, unrelated; R (libKriging installed in build-tb/installed,
     package in the scratchpad, `LD_LIBRARY_PATH` needed at install):
     binding-consistency 49 / 49; full suite (32 files, 695 tests) only
     `test-unstableLL.R` "unstable LL fixed using rcond failover" fails (the
     R twin of test 2, same decision) — run it with
     `test_dir(".", package = "rlibkriging", load_package = "installed")`:
     without the package environment `load.NestedKriging` (S3, not exported)
     is not found.
4. **Wider study before release** (`bench/thread-budget/fit_repro_study.cpp`, add
   the jitter hook back as a call to `set_chol_jitter` instead of the removed
   `TMP_JITTER`):
   - [ ] noisy data (Nugget / Heterogeneous unaffected by design, check the
         noise-free fit of slightly noisy data), d ∈ {1, 2, 4, 8, 16}, n up to
         2000, other test functions (Branin, Hartmann, ...), multistart (BFGS10).
   - [x] how often a single start ends worse than before (test 2 pattern) vs
         better — see 4b.
   - [ ] **4b. Test 2 study (2026-10-19).**
     - *The test case* (`bench/thread-budget/unstable_ll_profile.cpp`): the LL
       profile has a single maximum, theta = 0.1506, LL = 2.288, identical with
       and without jitter (BFGS10 finds it in every setting). Beyond
       theta ≈ 0.5 the two profiles differ: without jitter the LL keeps falling
       (-41 at theta = 2) until the Cholesky fails, then becomes erratic (the
       stepwise nugget: -19.7 / -46 / -23 between 2.5 and 4); with jitter it
       flattens into a bumpy plateau (-12 to -19 for theta in 0.6..10: the
       jitter acts as a nugget, the long-range limit is "trend + tiny noise").
       The single BFGS start is random in the theta bounds; it lands in the tail
       and stops on a plateau bump (theta 1.37, LL -14.0). It passed before only
       with the test's own settings (`num_nugget = 1e-15` + rcond check): with
       the *default* settings and no jitter, the same single start already
       ended at theta = 9.06, LL = -22.2.
     - *800 data sets per kernel* (`jitter_multistart_study.cpp`,
       `jitter-ms-*.csv`, `summarize_jitter_ms.py`; 100 sets × {unstable n 7 /
       15 d 1, sin n 10 d 1 / 20 d 2 / 50 d 2 / 50 d 4, branin n 15 / 30};
       default settings; "miss" = BFGS LL < BFGS10 LL - 0.01):

       | kernel | miss, jitter off | miss, jitter on | sets newly missed / newly found with jitter |
       |---|---|---|---|
       | gauss | 353 / 800 (44 %) | 322 / 800 (40 %) | 132 / 163 |
       | matern5_2 | 96 / 800 (12 %) | 83 / 800 (10 %) | 11 / 24 |
       | matern3_2 (jitter 0) | identical | identical | 0 / 0 |

       The jitter does not make a single start miss more often; it changes
       *which* data sets miss (with jitter, misses sit on the long-range
       plateau, median theta 2–5). A single start of gauss misses the optimum
       in ~40 % of the cases either way. Per-family rates: off → on 27 → 47 %
       (branin 15), 27 → 40 % (sin 20 d 2), but 39 → 16 % (sin 10 d 1),
       40 → 25 % (unstable 7), 28 → 11 % (unstable 15).
     - *Accuracy* (median ratio of normalized test RMSE, on / off, BFGS10):
       gauss 1.0–1.9 (1.81 branin 30, 1.64 sin 10 d 1, 1.88 unstable 15),
       matern5_2 1.0–1.6 (1.58 sin 50 d 2). Consistent with 8.6: the jitter
       smooths the interpolation (point 3 of 4 above, calibration / accuracy,
       still to weigh).
     - *Reading of test 2*: its section "unstable LL fixed using rcond
       failover" exercises the rcond/num_nugget failover, which the jitter
       bypasses (the Cholesky no longer fails). Options: (i) run that section
       with `set_chol_jitter(0)` (it tests the failover) and add a jitter-on
       check "BFGS10 finds theta ≈ 0.15 and LL(BFGS10) ≥ LL(BFGS)";
       (ii) keep it as a blocker until a single start is made robust to the
       plateau (e.g. a deterministic first start instead of a random one).
   - [ ] **4c. Other jitter strategies (2026-10-20, question of the user).**
     `jitter_split_study.cpp`, `jitter-split-*.csv`, `summarize_jitter_split.py`
     (60 sets per row, pct 10 / 100 / 200; "div" = relative spread of theta
     > 1e-3 across the thread budgets). Variants: `off` (jitter 0), `on`
     (current), `on>0` / `on>10` (theta of `on`, final model rebuilt at that
     theta with jitter 0 / 1e-10), `on~0` (continuation: a last BFGS with
     jitter 0 started from the theta of `on`). Median normalized test RMSE:

     | kernel | data | optim | off | on | on>0 | on>10 | on~0 | div off | div on, on>0, on>10 | div on~0 |
     |---|---|---|---|---|---|---|---|---|---|---|
     | gauss | branin 30 | BFGS10 | 0.0217 | 0.059 | 0.0374 | 0.0484 | 0.0324 | 15 % | 0 % | 2 % |
     | gauss | sin 10 d 1 | BFGS10 | 0.0222 | 0.0365 | 0.0324 | 0.0302 | 0.0219 | 0 % | 0 % | 0 % |
     | gauss | sin 50 d 2 | BFGS | 0.193 | 0.405 | 0.401 | 0.401 | 0.131 | 5 % | 0 % | 8 % |
     | gauss | sin 50 d 2 | BFGS10 | 0.0653 | 0.0959 | 0.0831 | 0.0834 | 0.0395 | 3 % | 0 % | 3 % |
     | gauss | sin1 100 d 2 | BFGS | 0.101 | 5.8e-4 | 2.4e-4 | 2.4e-4 | 1.1e-4 | 55 % | 0 % | 0 % |
     | gauss | unstable 15 | BFGS10 | 0.0057 | 0.0098 | 0.0067 | 0.0081 | 0.0068 | 0 % | 0 % | 15 % |
     | matern5_2 | sin 50 d 2 | BFGS / BFGS10 | 0.0106 | 0.0173 | 0.0153 | 0.0173 | 0.0106 | 13 / 0 % | 0 % | 2 / 5 % |
     | matern5_2 | sin1 100 d 2 | BFGS | 2.7e-4 | 5.7e-4 | 4.8e-4 | 5.7e-4 | 2.7e-4 | 98 % | 0 % | 93 % |
     | matern5_2 | branin 30, sin 10, unstable 15 | BFGS10 | ≈ | ≈ | ≈ | ≈ | ≈ | 0 % | 0 % | 0 % |

     Reading:
     - the accuracy cost of `on` has two parts: the smoothing of the final
       predictor (recovered by `on>0`: about half for gauss, e.g. branin
       0.059 → 0.037, unstable 0.0098 → 0.0067) and the theta it selects
       (the jitter caps the conditioning where the optimum of the exact LL
       sits: matern5_2 sin1 off LL is erratic, 357–444 for theta 5–10);
     - `on>0` keeps the reproducibility (theta is that of `on`; prediction
       spread across thread budgets of the same order as `on`: median
       1e-10..1e-6, max ~5e-4 on branin, scale ~50);
     - `on~0` gets the best accuracy (≈ off, better than off when the
       jittered search found a better basin: gauss sin 50 0.0395 vs 0.0653)
       but loses the reproducibility (up to 93 %): its last stage is the
       former, rounding-sensitive objective;
     - not measured: a theta-dependent continuous jitter (J(theta) from an
       a-priori bound of lambda_min, zero where R is well conditioned): at the
       optimum R is usually ill-conditioned, so it would select the same
       theta as `on`; it would only recover the smoothing part, which `on>0`
       recovers already. A jitter growing with n (conditioning grows with n:
       gauss 1e-8 still left 2 % divergence at n = 200, d = 4) is still open.
     - **Decided (2026-10-20): `on>10`, implemented.** Two jitters in Kriging:
       - `m_chol_jitter` (`chol_jitter()`): the objective's,
         `kernel_chol_jitter` (gauss 1e-8, matern5_2 1e-10): optimization,
         `logLikelihood()` / `logLikelihoodFun()` / LOO / LMP (so
         `logLikelihood() == logLikelihoodFun(theta())` is still the optimized
         value), Vecchia and Nystrom (their committed state comes from the
         objective);
       - `m_model_chol_jitter` (`model_chol_jitter()`): the committed exact
         factorization (m_T: predict / simulate / update), 
         `LinearAlgebra::model_chol_jitter` = min(kernel jitter, num_nugget)
         (gauss 1e-10, matern5_2 1e-10, 0 when the kernel jitter is 0).
         `Kriging::commit_model_chol_jitter` rebuilds the model at the optimal
         theta after the BFGS commit and after the warm-restart update;
         `optim = "none"` and update(refit = false) build it directly
         (`make_committed_Model`).
       - JSON: `model_chol_jitter` written when > 0 (absent → 0).
       - NestedKriging: non-warped groups use `model_chol_jitter` (as its
         Kriging submodels, fitted with `optim = "none"`); WarpKriging still
         uses its one jitter for both (kernel jitter, else num_nugget).
       - Check (`jitter-impl-*.csv`, same driver after the change): the
         default (`on` column) now equals the former `on>10` in every row
         (RMSE gauss branin 0.059 → 0.048, sin 10 0.037 → 0.030, unstable
         0.0098 → 0.0081, sin1 100 5.8e-4 → 2.4e-4; matern5_2 unchanged) with
         0 % divergence across thread budgets.
       - Tests: `KrigingFitTest` (BFGS LL >= grid LL) needed
         logLikelihood(Fun) to stay the objective (it failed with a single
         jitter at 1e-10 after the search); `KrigingPredictTest` FD step
         1e-4 → 1e-3 (the committed R is conditioned ~1e10 again);
         ctest 218 / 219 (test 2 left). Python binding (build-py): 19 / 20,
         `binding_consistency` fails: LL / LOO at theta = 0.3 against the
         DiceKriging references (tests/references, gauss) at 1e-12, off by
         1e-10..1.4e-5 relative because of the objective jitter 1e-8 (already
         the case since option a). Fix: expose set_chol_jitter in the bindings
         (3.3) and run the reference-consistency tests with jitter 0 (they
         check the formula); the R reference tests probably need the same.
   - [x] **4d. Wider grid (2026-10-20)** (`jitter_scale_study.cpp`,
     `jitter-scale-*.csv`, `summarize_jitter_scale.py`; gauss / matern5_2;
     sin d 1–16, n 50–2000, Branin, Hartmann6, BFGS10, noisy data fitted
     noise-free; jitter off vs on (current default), pct 10 / 100 / 200;
     "div" = theta differs by > 1e-3 across the thread budgets). Findings:
     1. **The jitter plateau traps the starts** (blocker for the default):
        gauss sin n 200 d 2, 12 / 30 single starts and **6 / 20 BFGS10** end
        at theta ≈ 7–10 (upper bound 10) with RMSE 0.6–0.9 vs ~1e-4 without
        jitter. Optimizer miss, not the objective: the jittered LL at the
        no-jitter theta is +490..+1120 vs -170..-310 on the plateau
        (scratch probe). Cause: the uniform draw in [0.02, 10] x maxdX puts a
        start in the useful region (all theta_j < 1 x maxdX) with probability
        0.1^d (1 % in d = 2); without jitter the erratic tail made the
        restart criteria (`num_iters <= 2`, ABNORMAL) send the start towards
        short ranges; the smooth plateau does not trigger them (issue #418,
        comment: restart diagnosis).
     2. **Log-uniform draw fixes it** (temporary `LK_TMP_LOG_START` hook in
        `fit_impl`, reverted; `jitter-logstart-*.csv`): with the jitter,
        gauss sin 200 d 2 BFGS RMSE 0.63 → 2.0e-4, BFGS10 2.0e-4 → 1.5e-4,
        sin 200 d 4 0.74 → 0.54, sin 50 d 2 0.17 → 0.095, Hartmann6 60
        0.96 → 0.86; divergence with jitter 0–3 % (without jitter matern5_2
        stays at 73–87 %).
     3. **Accuracy cost of the gauss objective jitter on very smooth
        functions**, revealed once the starts are good: Branin n 50 with log
        start, RMSE 4.9e-4 without jitter vs 0.126 with (×24): the exact
        optimum needs a conditioning that 1e-8 forbids (theta-choice part of
        the cost, see 4c).
     4. **Residual divergence with the jitter** (uniform start): up to 20 %
        (gauss Hartmann6 300), 40 % (matern5_2 Hartmann6 300), 60 % (matern5_2
        sin 2000 d 8), some severe (dLL 100s, RMSE ×1000). Causes (scratch
        probes with the BFGS log): (a) restart decisions on rounding-level
        differences: `f_opt > best_f_opt` compares two equal optima
        (63.8263 vs 63.8263) exactly, and the line-search outcome
        (ABNORMAL vs CONVERGENCE) flips; (b) `max_iteration` (20) reached on a
        flat objective (matern5_2 d 8): the path itself is rounding-chaotic;
        (c) the rcond check uses `rcond(L)^2`, pessimistic by ~n^2, against
        1e-18: at large n it may still trigger the stepwise nugget despite
        the jitter (warnings at 2.2e-16 for n 500, below n x 1e-18; nugget
        not added there).
     5. Fit time with the jitter: median ratio on / off 0.12–1.35, mostly
        < 1 (fewer restarts).
     **Decided (2026-10-20): A, B (ii), C.**
     - A. Log-uniform draw of the BFGS starting points in `Kriging::fit_impl`
       (`theta0_rand`; uniform for a dimension with a non-positive lower
       bound). WarpKriging unchanged (its first start is the current theta).
       Effect on the tests: `KrigingPredictTest` back to its original FD step
       (the fit now reaches the short-range optimum, theta 0.85 / 0.53, as
       without jitter before); NestedKriging one-group rBCM mean tolerance
       1e-6 → 1e-5 (committed R conditioned ~1e13, 1.8e-6 measured).
     - C. Restart "no improvement" criterion with the optimizer's own relative
       tolerance (`factr * epsmch`) instead of an exact comparison.
     - B (ii). gauss objective jitter 1e-8 / 1e-9 / 1e-10 measured with A and C
       (`gaussjitter-<J>-*.csv`): see 4e.
     - ctest 220 / 220 (KrigingWarmRestartTest timing-flaky under -j8, passes
       alone). Binding tests to rerun (reference values may move with A).
   - [ ] **4e. gauss objective jitter with A + C (2026-10-20)**
     (`gaussjitter-<J>-gauss-*.csv`, `summarize_jitter_scale.py gaussjitter-<J>`;
     240 data sets: Branin 50, sin n 50 / 100 / 200 d 2, 200 / 500 d 4, 200 d 8,
     Hartmann6 60 / 300, BFGS10 sin 200 d 2):

     | J (objective) | div (theta differs across thread budgets) | median RMSE ratio on / off | Branin 50 RMSE quartiles |
     |---|---|---|---|
     | 0 (off) | 43 / 240 (18 %) | 1 | 2.5e-4 / 5.2e-4 / 4.5e-3 |
     | 1e-8 (current) | 5 / 240 (2.1 %) | 1.0–1.24, Branin 246 | 0.13 / 0.14 / 0.16 |
     | 1e-9 | 4 / 240 (1.7 %) | 1.0–1.06, Branin 3.8 | 1.5e-3 / 0.058 / 0.15 |
     | 1e-10 | 15 / 240 (6.2 %) | 1.00, Branin 3.8 | 1.1e-3 / 3.5e-3 / 0.14 |

     1e-9 keeps the reproducibility of 1e-8 with almost no accuracy cost but
     on Branin (very smooth, polynomial-like: the exact optimum needs an even
     higher conditioning); 1e-10 loses reproducibility (sin d 4: 20–50 %).
     Remaining divergence with 1e-8 / 1e-9: sin d 4 n 200 / 500 (10–20 %),
     the iteration-limit / path chaos of 4d point 4(b). Decision pending.
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
