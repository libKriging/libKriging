# macOS / Accelerate: open questions and TODO for a Claude Code session on a Mac

Context: rationalization program (#440), tasks T5 (#445), T6 (#446), T9 (#449), T10 (#451),
T8 (#448). Every macOS / Accelerate statement below comes from GitHub-hosted `macos-15`
(3 cores) runs or from inference; none was measured on a many-core Mac with the current
branches. This file lists what to measure, how, and what to report. Related: #416 (OpenMP
threshold on macOS).

## Open questions

| # | Question | Current state | Source |
|---|---|---|---|
| Q1 | Is the intermittent failure of the exact equality `leaveOneOut("nystrom(80)") == leaveOneOut("nystrom")` caused by Accelerate splitting reductions differently from call to call? | Inferred, not measured. Library code shown deterministic on Linux (2000 calls, 0 bitwise difference). Test relaxed to 1e-5 relative in #451 | T10 log, sections 2–5 |
| Q2 | Are the 6 other bitwise checks of BLAS-dependent values exposed the same way on Accelerate? | Not run on macOS | T10 log, section 5 |
| Q3 | Why does `MLPKriging predictive variance is calibrated (#407)` fail in macOS **Release** (single start: coverage 0.72 < 0.8, Q2 −0.68) while it passes in macOS Debug and on Linux? | Attributed to AppleClang Release rounding + libc++ `uniform_real_distribution`; not reproduced on a Mac | T9 log, section 5; T8 log |
| Q4 | Which `kOmpMinRows` is right on macOS (200 like Linux, or the current 800)? Is OpenMP even enabled in the macOS build? | Never measured (#416) | `src/lib/OmpLoops.hpp`, T5 log |
| Q5 | Does a multistart oversubscribe the cores with Accelerate on a many-core Mac (no per-thread thread control)? Does `BLASSetThreading(single)` or `VECLIB_MAXIMUM_THREADS=1` help? | No gain measured on 3 cores (CI) and on one 10-core Mac (self-hosted, 2026-11-01), within ±15 % noise | T9 log, section 5 |
| Q6 | Is the `ConcurrentCallsTest` full-budget tolerance 1e-4 (decision D2) safe on a many-core Mac? | Margin 1.4x measured on a 10-core Linux machine; not measured on macOS with the final test | T6 log, T9 log 2.6 |
| Q7 | Linux / macOS shared reference of `BlasBackendTest` (fitted theta, LL, gradient for 200 <= n < 800): still within tolerance with the final branches? | Passed on `macos-15` (3 cores) before T5's gradient revert and T1 | T9 log, section 4 |

## Setup (once)

```sh
# record the machine
sw_vers; sysctl -n machdep.cpu.brand_string hw.physicalcpu hw.logicalcpu hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu
clang --version; cmake --version
# dependencies (see tools/linux-macos/install.sh and AGENTS.md); OpenMP for AppleClang:
brew install cmake ninja libomp
git clone --recurse-submodules https://github.com/libKriging/libKriging.git && cd libKriging
```

Read `AGENTS.md` and use the `libkriging:build` skill (C++ only, no binding needed except for
Q3 if the Octave job is to be reproduced). Build each branch in its own build directory
(`build-<branch>-<Release|Debug>`). Check in the CMake output whether OpenMP is found
(`OpenMP_CXX_FOUND`) and whether Armadillo uses Accelerate (`ARMA_USE_ACCELERATE` / the
`LinearAlgebra::check` test output names `Accelerate.framework`). If OpenMP is not found,
report it: it changes Q4 entirely.

Results go to a new branch `ccr-54a50343-4jupn9-macos-accelerate` created from
`origin/ccr-54a50343-4jupn9-t9-blas-backends-ci`, in `docs/dev/rationalization/macOS-Accelerate-results.md`
(tables with the raw numbers, machine description, commands). Do not change library code or
test tolerances on that branch without a decision of the maintainer: report and propose.
Never push to `master`. Summarize the results in a comment on this issue.

## TODO

### 1. Q1 / Q2 — Nystrom (and other) bitwise checks on Accelerate

Branch: `origin/ccr-54a50343-4jupn9-t10-nystrom-determinism`, Release.

- [ ] Restore the original exact check locally (revert the change of
      `tests/KrigingApproxObjectivesTest.cpp` from #451: `git diff origin/ccr-54a50343-4jupn9 -- tests/KrigingApproxObjectivesTest.cpp`, apply in reverse), build.
- [ ] `ctest -R "consistent with the exact fits" --repeat until-fail:500` with default threading;
      report the number of failures and, for each failure, the two values to 17 digits
      (add a temporary `printf("%.17g")` if needed).
- [ ] Same with `VECLIB_MAXIMUM_THREADS=1` (environment, at load), then with
      `BLASSetThreading(BLAS_THREADING_SINGLE_THREADED)` called at the start of the test
      (macOS 15+, `<vecLib/thread_api.h>`; temporary patch). Expected if the T10 attribution is
      right: failures with default threading, none single-threaded.
- [ ] Build and run `repro.cpp` from the appendix of `docs/dev/rationalization/T10-nystrom-determinism.md`
      (2000 alternated calls, with and without heap shifts): count bitwise differences.
- [ ] Q2: `--repeat until-fail:500` on the 6 tests listed in T10 log section 5
      (`KrigingNystromTest.cpp:172`, `:816`, `KrigingVecchiaTest.cpp:356`, `:390`,
      `KrigingApproxObjectivesTest.cpp:301-302`, `KrigingNormalizeScaleTest.cpp:812`);
      report failures and gaps.
- [ ] Conclusion: confirmed / refuted attribution; whether 1e-5 is right (largest observed gap
      vs 1e-5); which of the 6 need the same treatment.

### 2. Q3 — MLPKriging calibration test in Release

Branches: `origin/master` and `origin/ccr-54a50343-4jupn9-t8-mlp-calibration-cost` (K = 3), Release and Debug.

- [ ] Run `MLPKrigingCalibrationTest` 10 times in each of the 4 builds; the test prints its
      `[calibration]` scores on T8's branch: report coverage, NLPD, Q2, `var_scale` for the
      `BFGS` and `BFGS4` cases.
- [ ] If Release fails and Debug passes: find the first divergence (print the MLP init weights
      and the first objective values in both builds; check `-ffp-contract` / FMA:
      rebuild Release with `-ffp-contract=off` and see if it passes).
- [ ] Compare the data and MLP init with Linux: print the first 5 rows of X and the first
      weights, to confirm or refute the libc++ vs libstdc++ `uniform_real_distribution` explanation.
- [ ] Proposal for the test (do not apply): design on which the single start converges on every
      platform, or single-start case checking only what calibration can do (NLPD gain, `var_scale > 1`).

### 3. Q4 — OpenMP threshold `kOmpMinRows` on macOS (#416)

Branch: `origin/ccr-54a50343-4jupn9-t9-blas-backends-ci` (contains T5 / T6; the gradient
result no longer depends on the threshold after T1, only the speed does).

- [ ] Builds with `-DCMAKE_CXX_FLAGS=-DLK_OMP_MIN_ROWS=200` and the default (800), Release.
- [ ] `bench/bench-threads` (see `bench/thread-budget/README.md`) for n in {100, 200, 400, 800, 1500},
      d = 4: objective + gradient time per evaluation, medians of >= 5, after the 300 ms warm-up.
- [ ] Full single-start and BFGS4 fits (`bench/bench-fit-profile`) at n = 200, 500, 1000.
- [ ] Report the break-even n; propose the macOS value of `LK_OMP_MIN_ROWS`.

### 4. Q5 — multistart oversubscription with Accelerate

Same branch, Release, default threshold.

- [ ] `BlasBackendTest` hidden timing case (`[.timing]`, see T9 log section 5) with budget
      100 % and with `LK_NUM_THREADS_PERCENT` 50: `BFGS` vs `BFGS4` vs `BFGS10`,
      n = 200, 400, 800, 1500; default threading, `VECLIB_MAXIMUM_THREADS=1`,
      `BLASSetThreading(single)`; medians of 5; wall and CPU time (`/usr/bin/time -l`).
- [ ] Report whether any setting gives a gain above the noise (state the noise from repeats);
      if `VECLIB_MAXIMUM_THREADS=1` never helps, propose removing it from the FAQ / `parallel.md`.

### 5. Q6 — `ConcurrentCallsTest` tolerance 1e-4

Same branch, Release.

- [ ] `ctest -R ConcurrentCallsTest --repeat until-fail:20`, default budget (all cores), then
      `LK_NUM_THREADS_PERCENT=50`; report the largest gap per quantity printed by the test
      (theta, sigma2, objective, mean, stdev, simulations) and any failure.

### 6. Q7 — shared reference and full suite

Same branch.

- [ ] `BlasBackendTest` (`LK_EXPECT_BLAS=accelerate`): report the gaps to the shared reference.
- [ ] Full `ctest` in Release and Debug; list failures with their output.

## Report format

For each question: the measurements (table), the conclusion (confirmed / refuted / inconclusive
with why), and a proposal if a change is needed (library, test, docs), with its expected effect.
