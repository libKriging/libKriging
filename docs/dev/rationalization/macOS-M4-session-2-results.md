# macOS Apple M4 — session 2 results (issue #472, follow-up of #452)

Machine: see `macOS-M4-session-2-machine.txt`. **MacBook Air M4** (`Mac16,12`, fanless, 16 GB), 4 P + 6 E
cores, macOS 15.7.4, Apple clang 17.0.0, cmake 4.4.0, Homebrew libomp 22.1.8, Accelerate.
Branch `macos-m4-session-2` from `origin/master` 8e90e42d (#480). No PR for these results.

> **Timing conditions: read this first.** The task file asks for mains power and an idle machine. The Mac could not be
> plugged in; the timings were taken **twice, both times on battery** (the CPU otherwise idle, `caffeinate -i`, fanless
> MacBook Air, no thermal warning recorded; snapshots in `macOS-M4-session-2-conditions.txt` of each series):
>
> * **series 1**: battery 65 -> 60 %, macOS **Low Power Mode ON**; files `macOS-M4-session-2-*.csv`;
> * **series 2 (reference)**: battery 59 -> 49 %, Low Power Mode **OFF** (`lowpowermode 0` in every snapshot; the script
>   refuses to start otherwise); files in `macOS-M4-session-2-rerun-lpm-off/`. Absolute times match the first session
>   of #452 (LL at n = 1500, 10 threads: 30 ms), so this series is representative of full power.
>
> Builds were interleaved inside each of the 5 rounds, so ratios between builds are the reliable part. Series 2 is the
> one the conclusions rely on; series 1 is kept as a cross-check (Low Power Mode: parallel speed-ups per evaluation
> ×2.1 instead of ×1.6-1.8, slower absolute times, same decisions). What was not done: mains power. Power state is
> the only remaining difference from the task file; the CPU was at full power in series 2.

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

## 2. #456 part (b): GCD backend — decision

Branch `feat/parallel-for-gcd` (local, commit "wip: GCD backend", **not opened as a PR**): `dispatch_apply_f` path in
master's `parallel_for` when `__APPLE__` and OpenMP is not compiled in; `-DLK_NO_GCD_LOOPS` for measurement; no
`LK_APPLE_OPENMP`, no threshold change; a fork child has a budget of 1 (#460), so libdispatch is not used there (no
separate flag needed). `clang-format` 14 clean. `build-gcd` full suite: **243 / 243**. Builds: `seq` (default macOS
build), `gcd` (threshold 800, as master), `gcd200` (threshold 200), `omp` (libomp, 800), `omp200`, `omp50`.

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

### Series 2 (Low Power Mode OFF, the reference) — per evaluation, LL + gradient (ms), median (min-max) of 5 rounds

### eval LL + gradient (ms): median (min-max) of 5 rounds
| n | seq | gcd | gcd200 | omp | omp200 | omp50 |
|---|---|---|---|---|---|---|
| 100 | 0.42 (0.41-0.44) | 0.42 (0.41-0.42) | 0.41 (0.41-0.43) | 0.41 (0.40-0.46) | 0.42 (0.42-0.43) | 0.36 (0.34-0.38) |
| 150 | 0.76 (0.71-0.78) | 0.74 (0.73-0.76) | 0.76 (0.74-0.76) | 0.74 (0.71-0.77) | 0.74 (0.72-0.76) | 0.51 (0.49-0.53) |
| 200 | 1.31 (1.28-1.39) | 1.31 (1.26-1.37) | 0.81 (0.79-0.82) | 1.30 (1.28-1.32) | 0.79 (0.78-0.81) | 0.77 (0.75-0.81) |
| 400 | 5.66 (5.61-5.83) | 5.69 (5.58-5.73) | 3.15 (3.07-3.21) | 5.65 (5.63-5.71) | 3.16 (3.10-3.21) | 3.21 (3.08-3.30) |
| 800 | 26.11 (25.27-30.86) | 15.96 (15.73-16.30) | 15.80 (14.48-16.09) | 14.54 (14.39-16.06) | 14.49 (14.43-15.65) | 14.54 (14.50-14.70) |
| 1500 | 93.69 (93.39-94.27) | 52.36 (51.92-52.85) | 52.49 (52.12-52.67) | 52.69 (52.52-54.29) | 52.92 (52.35-53.31) | 52.65 (52.30-53.05) |

### eval speed-up vs seq (LL+grad, medians)
| n | gcd | gcd200 | omp | omp200 | omp50 |
|---|---|---|---|---|---|
| 100 | x1.01 | x1.03 | x1.04 | x1.01 | x1.19 |
| 150 | x1.02 | x1.00 | x1.03 | x1.03 | x1.48 |
| 200 | x1.00 | x1.62 | x1.01 | x1.67 | x1.70 |
| 400 | x0.99 | x1.80 | x1.00 | x1.79 | x1.77 |
| 800 | x1.64 | x1.65 | x1.80 | x1.80 | x1.80 |
| 1500 | x1.79 | x1.78 | x1.78 | x1.77 | x1.78 |

GCD and libomp give the **same times** (within 3 %) at both thresholds. Speed-up of an evaluation at threshold 200:
×1.6-1.8 at n = 200-400, ×1.8 at n = 800-1500 (series 1: ×2.1 and ×1.9).

### Series 2 — full fits (fit wall time in ms, median (min-max) of 5 rounds)

**`BFGS4` (the default since T3):**

### fits BFGS4: fit wall time (ms), median (min-max)
| n | seq | gcd | gcd200 | omp | omp200 |
|---|---|---|---|---|---|
| 200 | 30 (30-34) | 31 (29-32) | 28 (27-34) | 31 (30-35) | 28 (28-32) |
| 500 | 399 (382-413) | 390 (364-426) | 395 (326-407) | 395 (385-437) | 382 (327-449) |
| 1000 | 1976 (1906-2080) | 1933 (1875-2039) | 1992 (1855-2118) | 1942 (1864-1997) | 1924 (1877-1994) |
| 1500 | 4572 (4394-4748) | 4402 (4353-4889) | 4390 (4189-5472) | 4433 (4286-5584) | 4334 (4217-5621) |

### fits BFGS4: speed-up of the fit wall time vs seq
| n | gcd | gcd200 | omp | omp200 |
|---|---|---|---|---|
| 200 | x0.99 | x1.09 | x0.99 | x1.08 |
| 500 | x1.03 | x1.01 | x1.01 | x1.04 |
| 1000 | x1.02 | x0.99 | x1.02 | x1.03 |
| 1500 | x1.04 | x1.04 | x1.03 | x1.05 |

CPU time of the process minus the n = 10 baseline, s:

### fits BFGS4: CPU seconds of the process minus the n=10 baseline, median
| n | seq | gcd | gcd200 | omp | omp200 |
|---|---|---|---|---|---|
| 200 | 0.13 | 0.14 | 0.14 | 0.14 | 0.14 |
| 500 | 3.34 | 3.23 | 3.39 | 3.37 | 3.41 |
| 1000 | 16.20 | 15.79 | 16.15 | 15.86 | 16.22 |
| 1500 | 38.59 | 38.81 | 39.63 | 39.83 | 38.66 |

**`BFGS` (single start):**

### fits BFGS: fit wall time (ms), median (min-max)
| n | seq | gcd | gcd200 | omp | omp200 |
|---|---|---|---|---|---|
| 200 | 59 (54-74) | 55 (49-63) | 44 (40-44) | 55 (51-63) | 42 (40-50) |
| 500 | 155 (149-173) | 151 (148-153) | 115 (112-119) | 160 (148-189) | 119 (115-123) |
| 1000 | 622 (614-686) | 477 (473-480) | 454 (440-456) | 448 (444-487) | 458 (449-477) |
| 1500 | 1579 (1574-1590) | 1155 (1139-1166) | 1193 (1124-1226) | 1150 (1122-1219) | 1133 (1124-1140) |

### fits BFGS: speed-up of the fit wall time vs seq
| n | gcd | gcd200 | omp | omp200 |
|---|---|---|---|---|
| 200 | x1.08 | x1.34 | x1.08 | x1.42 |
| 500 | x1.02 | x1.34 | x0.97 | x1.30 |
| 1000 | x1.31 | x1.37 | x1.39 | x1.36 |
| 1500 | x1.37 | x1.32 | x1.37 | x1.39 |

### Series 2 — decision against the rule of the task file

Rule: keep GCD if it is ≥ ×1.3 faster than `build-seq` on `BFGS4` fits for n ≥ 500 **and** never slower for n < 500.

| n | `BFGS4`: `gcd` (threshold 800) | `gcd200` | libomp `omp200` | (series 1: `gcd200`) |
|---|---|---|---|---|
| 500 | ×1.03 | ×1.01 | ×1.04 | ×1.32 |
| 1000 | ×1.02 | ×0.99 | ×1.03 | ×1.09 |
| 1500 | ×1.04 | ×1.04 | ×1.05 | ×1.06 |

**The rule is not met, with no ambiguity: at full power the loops add nothing to a `BFGS4` fit at any n ≥ 500
(×0.99-1.05).** The multistart pool already keeps the cores busy (process CPU 16 s for 2 s of wall at n = 1000);
the loops inside a start have no free core to use. Single-start `BFGS` fits do gain (×1.3-1.4 at n = 200-1500 with
threshold 200, ×1.3-1.4 from n = 1000 at threshold 800), and so do individual evaluations. libomp is no better than
GCD. **Proposal confirmed: do not take the GCD backend; close #456.**

### Series 1 (Low Power Mode ON) — per evaluation (`bench/bench-threads`, d = 4, matern5_2; median (min-max) of 5 rounds, 9 reps each, ms)

### eval LL (ms): median (min-max) of 5 rounds
| n | seq | gcd | gcd200 | omp | omp200 | omp50 |
|---|---|---|---|---|---|---|
| 100 | 0.45 (0.45-0.45) | 0.45 (0.45-0.45) | 0.45 (0.45-0.45) | 0.45 (0.45-0.45) | 0.45 (0.45-0.45) | 0.27 (0.27-0.28) |
| 150 | 0.97 (0.97-1.40) | 0.97 (0.97-1.34) | 0.97 (0.97-0.97) | 0.98 (0.97-0.98) | 0.98 (0.97-0.98) | 0.43 (0.42-0.44) |
| 200 | 1.79 (1.78-2.46) | 1.70 (1.70-1.79) | 0.67 (0.64-0.71) | 1.72 (1.70-1.79) | 0.70 (0.68-0.71) | 0.70 (0.67-0.74) |
| 400 | 7.86 (7.78-7.88) | 7.82 (7.78-7.84) | 3.23 (3.21-3.26) | 7.85 (7.76-7.93) | 3.24 (3.18-3.28) | 3.18 (3.16-3.30) |
| 800 | 33.28 (33.04-33.48) | 13.79 (13.73-13.96) | 13.94 (13.67-13.99) | 13.76 (13.67-13.95) | 13.92 (13.51-14.47) | 13.74 (13.59-13.91) |
| 1500 | 115.84 (115.49-117.68) | 50.09 (50.05-50.51) | 50.09 (50.03-52.82) | 50.41 (50.15-50.58) | 50.33 (50.09-50.63) | 50.36 (50.06-50.59) |

### eval LL + gradient (ms): median (min-max) of 5 rounds
| n | seq | gcd | gcd200 | omp | omp200 | omp50 |
|---|---|---|---|---|---|---|
| 100 | 0.71 (0.71-0.72) | 0.73 (0.70-0.75) | 0.72 (0.70-0.73) | 0.72 (0.70-0.74) | 0.72 (0.70-0.75) | 0.52 (0.51-0.52) |
| 150 | 1.47 (1.43-1.84) | 1.49 (1.40-1.78) | 1.48 (1.43-1.51) | 1.48 (1.42-1.49) | 1.47 (1.41-1.51) | 0.79 (0.77-0.83) |
| 200 | 2.61 (2.56-3.28) | 2.54 (2.50-2.57) | 1.22 (1.18-1.24) | 2.53 (2.51-2.55) | 1.23 (1.22-1.27) | 1.23 (1.21-1.28) |
| 400 | 10.77 (10.61-10.97) | 10.87 (10.79-10.98) | 5.16 (5.14-5.31) | 10.92 (10.81-11.37) | 5.21 (5.11-5.27) | 5.27 (5.22-5.30) |
| 800 | 44.37 (44.34-45.01) | 22.86 (22.52-22.97) | 22.89 (22.66-23.16) | 22.81 (22.53-22.98) | 22.63 (22.48-31.11) | 22.69 (22.45-23.08) |
| 1500 | 159.91 (159.74-161.06) | 83.60 (83.40-84.87) | 83.71 (81.38-84.34) | 83.48 (83.27-83.82) | 83.88 (82.65-93.55) | 83.65 (82.50-84.31) |

### eval speed-up vs seq (LL+grad, medians)
| n | gcd | gcd200 | omp | omp200 | omp50 |
|---|---|---|---|---|---|
| 100 | x0.97 | x0.98 | x0.98 | x0.99 | x1.37 |
| 150 | x0.99 | x1.00 | x1.00 | x1.00 | x1.87 |
| 200 | x1.03 | x2.14 | x1.03 | x2.12 | x2.13 |
| 400 | x0.99 | x2.09 | x0.99 | x2.07 | x2.04 |
| 800 | x1.94 | x1.94 | x1.95 | x1.96 | x1.96 |
| 1500 | x1.91 | x1.91 | x1.92 | x1.91 | x1.91 |

GCD and libomp give the **same times** (within 2 %) at both thresholds: the backend does not matter, only whether
the loops run in parallel (threshold) and with how many threads. Speed-up of an evaluation: ×2.1 at n = 200-400
(threshold 200), ×1.9 at n = 800-1500.

### Series 1 (Low Power Mode ON) — full fits (`bench/bench-fit-profile`, d = 4; fit wall time in ms, median (min-max) of 5 rounds)

**`BFGS4` (the default since T3):**

### fits BFGS4: fit wall time (ms), median (min-max)
| n | seq | gcd | gcd200 | omp | omp200 |
|---|---|---|---|---|---|
| 200 | 52 (48-53) | 51 (48-58) | 41 (41-43) | 49 (49-51) | 43 (42-46) |
| 500 | 672 (611-720) | 557 (535-584) | 508 (489-516) | 550 (503-661) | 489 (483-509) |
| 1000 | 3379 (3086-3442) | 3123 (3025-3238) | 3095 (3038-3154) | 3163 (3037-3308) | 3147 (3056-3297) |
| 1500 | 8308 (7940-10801) | 7907 (7785-10179) | 7834 (7789-7994) | 7843 (7789-7875) | 7834 (7739-8001) |

### fits BFGS4: speed-up of the fit wall time vs seq
| n | gcd | gcd200 | omp | omp200 |
|---|---|---|---|---|
| 200 | x1.02 | x1.26 | x1.05 | x1.22 |
| 500 | x1.21 | x1.32 | x1.22 | x1.37 |
| 1000 | x1.08 | x1.09 | x1.07 | x1.07 |
| 1500 | x1.05 | x1.06 | x1.06 | x1.06 |

CPU time of the process minus the n = 10 baseline (data, 300 ms warm-up busy-wait, start-up; 0.30 s of CPU), s:

### fits BFGS4: CPU seconds of the process minus the n=10 baseline, median
| n | seq | gcd | gcd200 | omp | omp200 |
|---|---|---|---|---|---|
| 200 | 0.23 | 0.23 | 0.21 | 0.21 | 0.22 |
| 500 | 5.17 | 4.17 | 4.21 | 4.38 | 4.12 |
| 1000 | 23.45 | 23.43 | 22.98 | 23.66 | 23.76 |
| 1500 | 61.39 | 61.61 | 61.02 | 60.19 | 60.88 |

**`BFGS` (single start):**

### fits BFGS: fit wall time (ms), median (min-max)
| n | seq | gcd | gcd200 | omp | omp200 |
|---|---|---|---|---|---|
| 200 | 97 (93-104) | 103 (97-113) | 78 (66-82) | 102 (94-105) | 78 (73-92) |
| 500 | 272 (270-274) | 286 (268-365) | 185 (184-191) | 273 (270-275) | 187 (186-192) |
| 1000 | 992 (983-1068) | 685 (677-700) | 682 (678-688) | 688 (682-764) | 688 (678-693) |
| 1500 | 2535 (2510-2541) | 1766 (1755-1771) | 1766 (1754-1806) | 1766 (1755-1823) | 1775 (1743-1839) |

### fits BFGS: speed-up of the fit wall time vs seq
| n | gcd | gcd200 | omp | omp200 |
|---|---|---|---|---|
| 200 | x0.94 | x1.24 | x0.95 | x1.24 |
| 500 | x0.95 | x1.47 | x1.00 | x1.45 |
| 1000 | x1.45 | x1.46 | x1.44 | x1.44 |
| 1500 | x1.44 | x1.44 | x1.44 | x1.43 |

### fits BFGS: CPU seconds of the process minus the n=10 baseline, median
| n | seq | gcd | gcd200 | omp | omp200 |
|---|---|---|---|---|---|
| 200 | 0.15 | 0.17 | 0.21 | 0.17 | 0.21 |
| 500 | 0.42 | 0.54 | 0.45 | 0.43 | 0.45 |
| 1000 | 1.73 | 1.83 | 1.82 | 1.86 | 1.83 |
| 1500 | 4.96 | 5.22 | 5.24 | 5.22 | 5.26 |

### Series 1 — decision against the rule of the task file

Rule: keep GCD if it is ≥ ×1.3 faster than `build-seq` on `BFGS4` fits for n ≥ 500 **and** never slower for n < 500.

| n | `BFGS4`: `gcd` (threshold 800) | `gcd200` | libomp `omp200` |
|---|---|---|---|
| 500 | ×1.21 | ×1.32 | ×1.37 |
| 1000 | ×1.08 | ×1.09 | ×1.07 |
| 1500 | ×1.05 | ×1.06 | ×1.06 |

**The rule is not met** (n = 1000 and 1500: ×1.05-1.09; only n = 500 with threshold 200 reaches ×1.3), and libomp
is not better than GCD. Why: the default multistart already runs its 4 starts on a pool of threads that uses the
cores (process CPU 23 s for 3.1-3.4 s of wall at n = 1000: ~7 cores busy); the loops inside a start get a share of
~2 threads and add little. Where the loops do matter is the single-start `BFGS` fit (×1.44-1.47 at n ≥ 500 with 10
threads, ×1.24 at n = 200), and the per-evaluation speed-up. The CPU time is the same in every build (the loops
add no work).

**Proposal: do not take the GCD backend; close #456** (a note on the PR carries these numbers). The default macOS
build (sequential loops) loses ×1.0-1.4 on `BFGS4` fits for n = 200-500 and ≤ ×1.1 beyond, and ×1.4-1.5 on single
start fits at n ≥ 500; whoever needs it can build with libomp (`OpenMP_ROOT`, #488 documents it). Caveat: measured
in Low Power Mode on battery; the multistart pool saturating the cores should hold at full power, which makes the
conclusion for `BFGS4` unlikely to flip, but the single-start numbers could move.

## 4. CI: PR #488

`ci/macos-libomp-one-job` → master: "macOS Debug" installs libomp and passes `-DOpenMP_ROOT=$(brew --prefix libomp)`;
the other macOS jobs stay sequential. `OpenMP_ROOT` documented in `docs/dev/AllCMakeOptions.md`; consumers of an
installed libomp build need it too (`docs/dev/DevTips.md`). Checked on the M4: `OpenMP_ROOT` alone finds OpenMP;
Debug suite 245 / 245. Risk noted in the PR: the Python tests of that job could load a second `libomp`
(scikit-learn wheels bundle one, `OMP: Error #15`); the job to switch to is "Julia macOS".

## 3. Thread budget on Apple Silicon

### Q9, P-cores (4 threads, `LK_NUM_THREADS_PERCENT=40`) vs all cores (10, 100 %)

**Series 2 (Low Power Mode OFF, reference):**

### Q9: fit wall (ms) median (min-max), LK_NUM_THREADS_PERCENT 40 (4 threads) vs 100 (10); and 40/100 time ratio
| build | n | optim | 40 % | 100 % | t40 / t100 | CPU s (40 / 100) |
|---|---|---|---|---|---|---|
| omp | 500 | BFGS | 156 (150-165) | 158 (153-161) | 0.99 | 0.6 / 0.6 |
| omp | 500 | BFGS4 | 392 (377-458) | 396 (373-447) | 0.99 | 3.7 / 3.6 |
| omp | 1500 | BFGS | 1171 (1152-1194) | 1165 (1154-1173) | 1.01 | 3.8 / 4.1 |
| omp | 1500 | BFGS4 | 4751 (4581-4842) | 4732 (4326-5138) | 1.00 | 41.3 / 41.0 |
| gcd200 | 500 | BFGS | 125 (120-134) | 120 (114-124) | 1.04 | 0.6 / 0.7 |
| gcd200 | 500 | BFGS4 | 383 (357-419) | 387 (365-402) | 0.99 | 3.5 / 3.8 |
| gcd200 | 1500 | BFGS | 1169 (1158-1199) | 1137 (1114-1214) | 1.03 | 3.8 / 4.3 |
| gcd200 | 1500 | BFGS4 | 5050 (4548-5526) | 4771 (4341-5047) | 1.06 | 41.4 / 41.2 |
| seq | 500 | BFGS | 156 (149-171) | 154 (152-174) | 1.01 | 0.6 / 0.6 |
| seq | 500 | BFGS4 | 407 (387-420) | 421 (379-424) | 0.97 | 3.8 / 3.9 |
| seq | 1500 | BFGS | 1587 (1564-1612) | 1570 (1567-1617) | 1.01 | 3.9 / 3.7 |
| seq | 1500 | BFGS4 | 4980 (4459-5300) | 5014 (4447-5252) | 0.99 | 40.1 / 41.3 |

`t40 / t100` = 0.97-1.06: **4 threads and 10 threads give the same time** (CPU unchanged). Refuted: keep `hw.physicalcpu` as the 100 % (no change).

**Series 1 (Low Power Mode ON):**

### Q9: fit wall (ms) median (min-max), LK_NUM_THREADS_PERCENT 40 (4 threads) vs 100 (10); and 40/100 time ratio
| build | n | optim | 40 % | 100 % | t40 / t100 | CPU s (40 / 100) |
|---|---|---|---|---|---|---|
| omp | 500 | BFGS | 274 (270-278) | 270 (269-369) | 1.02 | 0.7 / 0.7 |
| omp | 500 | BFGS4 | 619 (532-644) | 527 (497-569) | 1.18 | 5.1 / 4.4 |
| omp | 1500 | BFGS | 1861 (1839-1911) | 1759 (1753-1803) | 1.06 | 5.2 / 5.5 |
| omp | 1500 | BFGS4 | 8320 (8004-8558) | 7781 (7722-7841) | 1.07 | 61.2 / 60.0 |
| gcd200 | 500 | BFGS | 200 (194-203) | 188 (187-200) | 1.06 | 0.8 / 0.8 |
| gcd200 | 500 | BFGS4 | 541 (519-580) | 516 (463-535) | 1.05 | 4.6 / 4.7 |
| gcd200 | 1500 | BFGS | 1875 (1834-1898) | 1770 (1761-1839) | 1.06 | 5.4 / 5.6 |
| gcd200 | 1500 | BFGS4 | 8031 (7930-8400) | 7731 (7689-7854) | 1.04 | 60.6 / 60.2 |
| seq | 500 | BFGS | 269 (265-388) | 269 (268-275) | 1.00 | 0.7 / 0.7 |
| seq | 500 | BFGS4 | 567 (498-632) | 524 (507-572) | 1.08 | 4.7 / 4.4 |
| seq | 1500 | BFGS | 2509 (2507-2524) | 2519 (2510-2589) | 1.00 | 5.2 / 5.2 |
| seq | 1500 | BFGS4 | 8109 (8044-8371) | 8220 (8114-8341) | 0.99 | 60.2 / 61.8 |

Four threads are never faster: `t40 / t100` = 1.00-1.18 (1.18 at `BFGS4`, n = 500, with libomp), 1.04-1.07 elsewhere; CPU
time unchanged. **Refuted: keep `hw.physicalcpu` (10) as the 100 %** (no change proposed; Low Power Mode caveat).

### Q10, libomp spin (`KMP_BLOCKTIME=0`, `build-omp200`)

**Series 2 (reference):**

### Q10: omp200, per evaluation (ms) median (min-max), default vs KMP_BLOCKTIME=0
| n | LL default | LL KMP_BLOCKTIME=0 | LL+grad default | LL+grad KMP_BLOCKTIME=0 | ratio (LL+grad, 0 / default) |
|---|---|---|---|---|---|
| 100 | 0.24 (0.23-0.24) | 0.24 (0.24-0.24) | 0.43 (0.40-0.45) | 0.41 (0.41-0.44) | 0.95 |
| 150 | 0.51 (0.51-0.52) | 0.52 (0.51-0.52) | 0.77 (0.73-0.79) | 0.78 (0.76-0.80) | 1.01 |
| 200 | 0.44 (0.43-0.46) | 0.43 (0.42-0.47) | 0.80 (0.78-0.80) | 0.79 (0.78-0.81) | 0.99 |
| 400 | 1.92 (1.91-1.97) | 1.93 (1.90-1.96) | 3.24 (3.21-3.39) | 3.23 (3.17-3.31) | 1.00 |
| 800 | 9.43 (9.06-9.49) | 8.54 (8.43-9.52) | 14.84 (14.47-15.97) | 14.59 (14.49-14.82) | 0.98 |

Ratio 0.95-1.01: no effect. **Series 1:**

### Q10: omp200, per evaluation (ms) median (min-max), default vs KMP_BLOCKTIME=0
| n | LL default | LL KMP_BLOCKTIME=0 | LL+grad default | LL+grad KMP_BLOCKTIME=0 | ratio (LL+grad, 0 / default) |
|---|---|---|---|---|---|
| 100 | 0.45 (0.45-0.45) | 0.45 (0.45-0.45) | 0.71 (0.71-0.76) | 0.71 (0.69-0.72) | 0.99 |
| 150 | 0.97 (0.97-0.98) | 0.97 (0.97-0.97) | 1.45 (1.39-1.50) | 1.43 (1.42-1.47) | 0.98 |
| 200 | 0.71 (0.70-0.75) | 0.69 (0.62-0.74) | 1.21 (1.19-1.28) | 1.25 (1.19-1.27) | 1.03 |
| 400 | 3.19 (3.16-3.22) | 3.21 (3.10-3.26) | 5.17 (5.05-5.28) | 5.25 (5.16-5.27) | 1.02 |
| 800 | 13.82 (13.25-14.26) | 13.99 (13.56-14.02) | 22.70 (22.39-22.92) | 22.62 (22.50-23.12) | 1.00 |

**No effect** (ratio 0.98-1.03, inside the noise) at n = 100-800. Refuted; nothing to document.

### Q4 follow-up, `kOmpMinRows` break-even

`build-omp50` (threshold 50) against `seq` per evaluation, LL + gradient: n = 100 **×1.37** (LL ×1.67), n = 150 **×1.87**,
n = 200 ×2.13, n = 400 ×2.04. So the loops already pay off at n = 100; the break-even is below 100 (not located:
n < 100 not measured). Threshold 200 is therefore safe and conservative for libomp builds: ×2.1 per evaluation at
n = 200-400 (the first session measured ×2.1 too, with full power), ×1.24-1.47 on `BFGS` fits and ×1.2-1.4 on
`BFGS4` fits at n = 200-500. Lowering to 100 would add ×1.4-1.9 per evaluation at n = 100-150 (0.2-0.7 ms: negligible
in a fit). **Confirmed twice: `LK_OMP_MIN_ROWS` 200 on macOS** (drop the `__linux__` condition of
`src/lib/OmpLoops.hpp`); the results do not depend on it (bitwise identical outputs at n = 500 between `seq`, `omp`
and `omp200`, section 2). It affects libomp builds only (the default macOS build has no OpenMP loops). Small PR:
see the end of this file.

## 5. Optional

### Q5 complement, `VECLIB_MAXIMUM_THREADS=1` (fit wall time, process CPU; `bench-fit-profile`, d = 4)

**Series 2 (Low Power Mode OFF, reference):**

### Q5: fit wall (ms) median (min-max) and process CPU (s), default vs VECLIB_MAXIMUM_THREADS=1
| build | n | optim | wall default | wall VECLIB=1 | VECLIB=1 / default | CPU s default | CPU s VECLIB=1 |
|---|---|---|---|---|---|---|---|
| seq | 800 | BFGS | 428 (421-432) | 410 (406-411) | 0.96 | 1.1 | 0.7 |
| seq | 800 | BFGS4 | 1109 (1084-1169) | 622 (598-712) | 0.56 | 9.3 | 2.3 |
| seq | 800 | BFGS10 | 2791 (2393-3087) | 1492 (1360-1804) | 0.53 | 23.4 | 8.6 |
| seq | 1500 | BFGS | 1598 (1552-1746) | 1618 (1578-1625) | 1.01 | 4.0 | 1.9 |
| seq | 1500 | BFGS4 | 4663 (4626-5335) | 3234 (2880-3940) | 0.69 | 39.7 | 12.1 |
| seq | 1500 | BFGS10 | 8991 (8025-10395) | 5992 (5397-7049) | 0.67 | 79.1 | 40.3 |
| omp | 800 | BFGS | 329 (326-340) | 310 (307-319) | 0.94 | 1.2 | 0.8 |
| omp | 800 | BFGS4 | 1071 (1012-1199) | 568 (539-639) | 0.53 | 9.4 | 2.5 |
| omp | 800 | BFGS10 | 2739 (2418-3185) | 1494 (1361-1855) | 0.55 | 23.5 | 8.6 |
| omp | 1500 | BFGS | 1157 (1120-1223) | 1212 (1144-1218) | 1.05 | 4.1 | 2.3 |
| omp | 1500 | BFGS4 | 5041 (4335-5320) | 2925 (2731-3386) | 0.58 | 44.6 | 12.8 |
| omp | 1500 | BFGS10 | 9443 (8172-10664) | 5919 (5385-7059) | 0.63 | 82.9 | 39.6 |

**Confirmed at full power**: with `VECLIB_MAXIMUM_THREADS=1`, multistart fits take 0.53-0.58 of the time at n = 800
(`BFGS4` 1109 -> 622 ms without libomp, 1071 -> 568 ms with it; `BFGS10` 2791 -> 1492 ms) and 0.58-0.69 at n = 1500
(`BFGS4` 4663 -> 3234 ms, 5041 -> 2925 ms), with **3.5-4.1x less CPU** at n = 800 (9.3 -> 2.3 s); single-start `BFGS` is
unchanged (0.94-1.05). The effect is the same with and without Low Power Mode, so it is not an artefact of the
power state. (The first session's `[.timing]` case did not show it: it compared the `BFGS4` / `BFGS` ratio, and its
`BLASSetThreading` variant only switched the calling thread, see section 1.)

**Series 1 (Low Power Mode ON):**

### Q5: fit wall (ms) median (min-max) and process CPU (s), default vs VECLIB_MAXIMUM_THREADS=1
| build | n | optim | wall default | wall VECLIB=1 | VECLIB=1 / default | CPU s default | CPU s VECLIB=1 |
|---|---|---|---|---|---|---|---|
| seq | 800 | BFGS | 689 (683-697) | 698 (695-704) | 1.01 | 1.5 | 1.0 |
| seq | 800 | BFGS4 | 1920 (1806-2017) | 1067 (1001-1094) | 0.56 | 13.7 | 3.6 |
| seq | 800 | BFGS10 | 4251 (4201-4336) | 2537 (2490-2634) | 0.60 | 31.9 | 13.7 |
| seq | 1500 | BFGS | 2522 (2501-2527) | 2757 (2745-2774) | 1.09 | 5.2 | 3.1 |
| seq | 1500 | BFGS4 | 8469 (8331-8503) | 5333 (5188-5445) | 0.63 | 61.4 | 19.9 |
| seq | 1500 | BFGS10 | 15296 (15077-15470) | 11266 (11140-11349) | 0.74 | 112.5 | 66.4 |
| omp | 800 | BFGS | 471 (470-481) | 497 (494-502) | 1.06 | 1.5 | 1.1 |
| omp | 800 | BFGS4 | 1700 (1622-1735) | 859 (851-881) | 0.51 | 13.3 | 3.6 |
| omp | 800 | BFGS10 | 4350 (4199-4466) | 2510 (2456-2571) | 0.58 | 32.9 | 13.7 |
| omp | 1500 | BFGS | 1812 (1750-1841) | 2031 (2023-2039) | 1.12 | 6.0 | 3.4 |
| omp | 1500 | BFGS4 | 7894 (7740-7978) | 4902 (4824-4980) | 0.62 | 61.0 | 21.2 |
| omp | 1500 | BFGS10 | 15194 (15148-15259) | 11138 (10999-11162) | 0.73 | 112.3 | 65.9 |

**This contradicts the first session** (which found no gain and proposed to drop the advice) **and the advice stays,
with a stronger case**: with `VECLIB_MAXIMUM_THREADS=1`, multistart fits are ×1.6-1.8 faster at n = 800 (`BFGS4`
1920 -> 1067 ms, `BFGS10` 4251 -> 2537 ms) and ×1.35-1.6 at n = 1500 (`BFGS4` 8469 -> 5333 ms), with **3-4x less CPU**
at n = 800 (13.7 -> 3.6 s): the pool workers and Accelerate's own threads oversubscribe the cores. Single-start `BFGS`
is unchanged or 6-12 % slower at n = 1500 (it has no pool: Accelerate's threads help it). Same with and without libomp.
Why the first session missed it: its `[.timing]` case compared the *`BFGS4` / `BFGS` ratio* (0.54-0.61 in both cases)
and its "`BLASSetThreading(single)`" variant only switched the **calling** thread (the setting is per thread, see
section 1), not the pool workers. Proposal (not applied, no PR asked): (1) keep the `VECLIB_MAXIMUM_THREADS=1`
advice in the FAQ / `parallel.md` for multistart fits on macOS; (2) make it unnecessary: let the pool workers call
`detail::accelerate_single_thread_here()` (PR #487) **in the parent too** whenever the pool has more than one
worker (the budget logic gives each worker a share; Accelerate has no per-thread count, only single / multi). To be
confirmed by the re-run on mains power with Low Power Mode off, since the size of the effect may depend on it.

### Q8 follow-up, cost of the `potri` check (#458) inside a full fit (`seq` vs the check compiled out)

**Series 2 (reference):**

### Q8: fit wall (ms) median (min-max), with the potri check (seq) vs compiled out (nocheck)
| n | optim | with check | without | with / without |
|---|---|---|---|---|
| 800 | BFGS | 428 (420-524) | 420 (416-422) | 1.019 |
| 800 | BFGS4 | 1163 (1118-1302) | 1263 (1180-1389) | 0.921 |
| 1500 | BFGS | 1771 (1697-1793) | 1611 (1568-1658) | 1.099 |
| 1500 | BFGS4 | 5570 (5438-5702) | 5990 (5773-6206) | 0.930 |

Mixed signs and ±8-10 % between two different binaries (`BFGS` 1.02 / 1.10, `BFGS4` 0.92 / 0.93: the `seq` build was also
slower than itself between sections B and F, 1771 vs 1579 ms at `BFGS` n = 1500): **the cost of the check is below
what a full fit can resolve here** (series 1: 1.3 % / 0 %, ratios inside the spread). Bound from the micro-benchmark
(5-10 % of `potri` with the `trtrs` check, and `potri` is a fraction of the gradient evaluation): well under 1-2 % of a fit.

**Series 1:**

### Q8: fit wall (ms) median (min-max), with the potri check (seq) vs compiled out (nocheck)
| n | optim | with check | without | with / without |
|---|---|---|---|---|
| 800 | BFGS | 688 (685-793) | 679 (675-727) | 1.013 |
| 800 | BFGS4 | 1857 (1495-1951) | 1758 (1694-1904) | 1.056 |
| 1500 | BFGS | 2518 (2493-2593) | 2519 (2471-2567) | 1.000 |
| 1500 | BFGS4 | 7927 (7796-8536) | 8174 (7995-8200) | 0.970 |

The check costs **1.3 % (n = 800) and 0 % (n = 1500) of a single-start fit**; the `BFGS4` ratios (1.056, 0.970) are
inside the spread of that workload (min-max ±5-10 %), so inconclusive but not above ~5 %. Input for the #458
decision "check everywhere (a) / Accelerate only (b) / remove (c)": **(a) is affordable** (≲ 1-2 % of a fit).

## Summary of the proposals

| Item | Result | Proposal |
|---|---|---|
| Fork crash of `BFGS4` children (section 1) | found on master | **PR #487** |
| CI without OpenMP on macOS (section 4) | confirmed | **PR #488** |
| GCD backend (#456 part b) | rule not met: ×0.99-1.05 on `BFGS4` at n ≥ 500 (full power) | **close #456** |
| `LK_OMP_MIN_ROWS` on macOS | 200 confirmed (×1.6-1.8 per evaluation at 200-400, ×1.3-1.4 per `BFGS` fit) | small PR #490 (libomp builds) |
| P-cores as the 100 % (Q9) | 4 and 10 threads equal (ratio 0.97-1.06) | keep 10 |
| `KMP_BLOCKTIME=0` (Q10) | no effect | nothing |
| `VECLIB_MAXIMUM_THREADS=1` advice (Q5) | `BFGS4`/`BFGS10` fits ×1.5-1.9 faster, 3.5-4x less CPU, with and without Low Power Mode | **keep it**; pool workers single-threaded in the parent too (proposal) |
| `potri` check cost (Q8) | below what a fit resolves (≲ 1-2 %) | (a) check everywhere is affordable |

Everything in sections 2, 3 and 5 was measured twice on battery (series 2: Low Power Mode off, the reference); only mains power was not available.
