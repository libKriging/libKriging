# macOS Apple M4 — session 2 results (issue #472, follow-up of #452)

Machine: see `macOS-M4-session-2-machine.txt`. **MacBook Air M4** (`Mac16,12`, fanless, 16 GB), 4 P + 6 E
cores, macOS 15.7.4, Apple clang 17.0.0, cmake 4.4.0, Homebrew libomp 22.1.8, Accelerate.
Branch `macos-m4-session-2` from `origin/master` 8e90e42d (#480). No PR for these results.

> **Timing conditions.** Sections 2, 3 and 5 are timings; they must be taken on mains power with nothing else
> running. The machine was on **battery** (72-78 %) at the start of this session. See "Timing conditions" below
> for what was actually measured where.

## 0. Builds (Release, C++ only, bindings off; `build-*` in the worktree)

| Dir | Options | OpenMP in the CMake output | BLAS |
|---|---|---|---|
| `build-seq` | default | `OpenMP not found - multistart optimization will run sequentially` | `-framework Accelerate` (`ARMA_LIBS`; `BlasBackendTest`: accelerate) |
| `build-omp` | `-DCMAKE_PREFIX_PATH=$(brew --prefix libomp) -DOpenMP_ROOT=...` | `Found OpenMP_CXX: -Xclang -fopenmp (found version "5.1")` | Accelerate |
| `build-omp200` | as `build-omp` + `-DCMAKE_CXX_FLAGS=-DLK_OMP_MIN_ROWS=200` | same | Accelerate |
| `build-omp-debug` | as `build-omp`, Debug | same | Accelerate |
| `build-gcd`, `build-gcd200` | branch `feat/parallel-for-gcd` (section 2), default options / + `-DLK_OMP_MIN_ROWS=200` | OpenMP not found | Accelerate |

`-DOpenMP_ROOT=$(brew --prefix libomp)` alone (without `CMAKE_PREFIX_PATH`) is enough to find OpenMP.

## 1. Master on the M4: full suites

| Build | Result |
|---|---|
| `build-seq` Release | **243 / 243** (25 s) |
| `build-omp` Release | **245 / 245** (25 s) |
| `build-omp200` Release | **245 / 245** (23 s) |
| `build-omp` Debug | **245 / 245** (66 s) |

All pass, including the `fork()` tests, `cmake_package` and the MLP calibration test. Files:
`macOS-M4-session-2-ctest-*.txt`.

`ctest -R fork --repeat until-fail:30`: `build-seq` and `build-omp`, **30 / 30** (the two `fork()` tests of master;
three with #487).

R binding (built with R's compiler against `build-R-omp`, a Release libomp build of master), `test-ForkSafety.R`
10 times (testthat is not installed: the file is run unchanged with a 20-line stand-in for `test_that` /
`expect_*`, in the session scratch): **10 / 10, 27 checks each, 0 failures** (`macOS-M4-session-2-forksafety-R.txt`).

### Finding: the `#459` fix does not cover the threads of the multistart pool (fixed in #487)

The scenario of #457 / #459 (`macOS-M4-session-2-mclapply.R`: warmed-up parent, then children doing a `BFGS4` fit,
predict with stdev / cov, `simulate` at 300 points and `update`) **crashes on master**: 0 / 40 children ok,
`caught segfault, address 0x110`, with libomp and **also in the default build without OpenMP**. In pure C++
(`fork` after a warmed-up `BFGS4` fit, the child fits `BFGS4`): killed by signal 11, 4 / 4 children, 6 runs out of 6.
`test-ForkSafety.R` (n = 60) and the C++ `fork()` tests do not see it.

| Child workload | master |
|---|---|
| `BFGS`, n = 100 / 200 / 400 (cold or warm parent) | ok |
| `BFGS4`, n = 400 (cold parent) | **crash** |
| `BFGS4`, n = 100 (warm parent) | **crash** |
| any of the crashing cases with `VECLIB_MAXIMUM_THREADS=1` in the parent | ok (4 / 4) |

Instrumenting the child handler (temporary, not kept) shows it runs in every child: `accelerate_set_threading`
non-null, `BLASSetThreading(single)` returns 0, `BLASGetThreading` returns 1. **`BLASSetThreading` is per thread**
(`thr_scope.cpp` in the session scratch: after `BLASSetThreading(single)` in the main thread, a thread created
afterwards reports mode 0 = multithreaded; a thread that sets its own mode does not change the others; the main
thread stays at 1). The handler makes the forking thread single-threaded; the threads that `run_task_pool`
creates afterwards in the child (the `BFGS4` pool) keep Accelerate's default mode, whose large BLAS / LAPACK
calls use libdispatch, which does not survive `fork()` without exec.

Fix: **PR #487** (`fix/fork-child-pool-accelerate`): the handler records the fact, each pool worker applies
`BLASSetThreading(single)` to its own thread first. New C++ test (3 children, 3 of 3 killed before, 3 of 3
exit 0 after). rlibkriging scenario with the change: **40 / 40** (two runs), without OpenMP and with libomp
(master: 0 / 40). Suite 244 / 244, `fork` tests 30 / 30.

(A first comparison against my old `#457` stack passed 40 / 40: that stack had no budget-1-in-the-child
logic of #460 and a different pool path; the cause on master is the one above, checked directly.)

## 2. #456 part (b): GCD backend — in progress

Branch `feat/parallel-for-gcd` (local, commit "wip: GCD backend", not a PR yet): `dispatch_apply_f` path in
master's `parallel_for` when `__APPLE__` and OpenMP is not compiled in; `-DLK_NO_GCD_LOOPS` for measurement; no
`LK_APPLE_OPENMP`, no threshold change; a fork child has a budget of 1 (#460) so libdispatch is not used there
(no separate flag needed). `clang-format` 14 clean. `build-gcd` full suite: **243 / 243**.

**Determinism** (`macOS-M4-session-2-determinism.cpp`, results `macOS-M4-session-2-determinism-results.txt`): fixed
Weyl design, d = 4; `BFGS4` fit; LL, LMP, LOO with gradients at a fixed theta; predict with stdev and cov (300
points); `simulate` (seed 123); Vecchia predict (m = 100: parallel branch) and simulate; Nystrom(60) fit and
predict; Vecchia(30) fit; NestedKriging (4 groups, NK). 94 832 numbers per run printed with `%.17g`:

| n | `build-omp` vs `build-seq` | `build-omp200` vs `build-seq` | `build-gcd` vs `build-seq` |
|---|---|---|---|
| 500 | identical | identical | identical |
| 1500 | identical | identical | identical |

**Bitwise identical** in all six comparisons (cmp on the full outputs). At n = 500 only `build-omp200` runs the
O(n²) loops in parallel (threshold 800 elsewhere); `build-gcd200` (threshold 200) is compared in the timing part.

## 4. CI: PR #488

`ci/macos-libomp-one-job` → master: "macOS Debug" installs libomp and passes `-DOpenMP_ROOT=$(brew --prefix libomp)`;
the other macOS jobs stay sequential. `OpenMP_ROOT` documented in `docs/dev/AllCMakeOptions.md`; consumers of an
installed libomp build need it too (`docs/dev/DevTips.md`). Checked on the M4: `OpenMP_ROOT` alone finds OpenMP;
Debug suite 245 / 245. Risk noted in the PR: the Python tests of that job could load a second `libomp`
(scikit-learn wheels bundle one, `OMP: Error #15`); the job to switch to is "Julia macOS".

## Sections 2 (timings), 3, 5: pending

Waiting for mains power and an idle machine (see top).
