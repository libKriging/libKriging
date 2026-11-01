# T9. CI of the BLAS backends (MKL, Accelerate) — work log

Task specification: `docs/dev/Rationalization.md`, section T9.

This branch merges T5 (#445) and T6 (#446) into the program branch (PR #449). The T5 / T6
merge conflicts (`BlasThreads.hpp`, `ThreadBudget.hpp`, `parallel.md`, CHANGELOG) had been
resolved by keeping both sides, unbuilt.

## Status

Done on Linux (OpenBLAS, oneMKL preloaded and linked, GNU and INTEL threading layers: run
locally and in CI). macOS arm64 (Accelerate): see section 5 (CI only, no Mac here).

Local machine: Intel Xeon @ 2.80 GHz, 4 vCPUs (`physical_cores()` = 4), Ubuntu 24.04, GCC 13,
OpenBLAS 0.3.26 pthreads (`libopenblas-pthread-dev`), oneMKL 2024.2.2 (pip wheel, with
`intel-openmp` 2024.2.1 and `tbb` 2021.13.1), Release builds. Executing the MKL wheel was
possible in this session (it was not in T5's).

## 1. Merged branch, OpenBLAS

`git submodule update --init --recursive`, Release, C++ only: builds without change;
`ctest -j2`: **231 / 231** passed (66 s). No T5 / T6 integration issue shows with OpenBLAS.

## 2. Merged branch, oneMKL preloaded: four failures, analysis and fixes

`LD_PRELOAD=libmkl_rt.so.2 MKL_THREADING_LAYER=GNU ctest -j2` on the OpenBLAS build: 4 / 231
failed. Each is listed with its cause and what was changed; "port to" says where the change
belongs if T5 / T6 are merged separately.

### 2.1 `fork() during concurrent fits` (ConcurrentCallsTest): child deadlocks — T5 x T6, fixed

1 child of 5 killed by its 60 s alarm. Backtrace of the stuck child (gdb on the child):

```
#1 __cxa_guard_acquire () from libstdc++.so.6
#2 libKriging::detail::BudgetScope::BudgetScope()
#3 Kriging::fit_impl(...)
```

`BudgetScope` reads function-local statics: `initial_mkl_threads()` (T5),
`initial_blas_threads()`, `physical_cores()`, `blas_state()`. The first
`MKL_Get_Max_Threads()` loads and initializes MKL's core / threading libraries (milliseconds):
a `fork()` made by the test while another thread was inside that first initialization copied
the C++ guard in its "in progress" state, and the child's first call waits for it forever. With
OpenBLAS alone the window is a few instructions (no MKL: a null weak symbol), which is why T6's
stress test did not see it; the hazard is the same for any lazy static.

Fix (`ThreadBudget.cpp`, `blas_count_fork_prepare`): the prepare handler initializes these
statics (and the dlsym / GetProcAddress MKL entry points, `touch_mkl_entry_points`) before
locking the count state. If another thread is initializing one, prepare waits for it to finish,
so `fork()` never copies a guard in progress. None of them takes a library lock: no new lock
order. Port to: T6 (fork handlers; `initial_mkl_threads` comes from T5).

After the fix, a second, different deadlock remained (2 of 20 runs, GNU layer; 1 of 10, INTEL):

```
#0 mm_cleanup_thread_cbk () from libmkl_core.so.2      (GNU layer, at worker thread exit)
#0 mm_account_ptr_by_tid () from libmkl_core.so.2       (INTEL layer)
#1 mkl_serv_allocate () from libmkl_core.so.2
```

MKL's fast memory manager lock, held by a parent thread inside MKL at `fork()` time: MKL's own
state, which the library cannot reach. With `MKL_DISABLE_FAST_MM=1` (MKL's documented switch:
plain `malloc` / `free`, fork-safe through glibc): 25 / 25 runs pass. **Known limitation**:
with MKL, `fork()` while other threads run MKL calls (libKriging's or anyone's, e.g. numpy) needs
`MKL_DISABLE_FAST_MM=1`, or a fork from a thread that runs no concurrent MKL work (R's
`mclapply` forks from the single R thread: not affected). The CI runs the fork case with
`MKL_DISABLE_FAST_MM=1` and every other test with MKL's defaults.

### 2.2 MKL count in the fork child — T5 x T6, fixed

The child handler sets OpenBLAS and OpenMP to 1, not MKL. Measured with a small C program
(MKL preloaded, `fork()`, count read in the child):

| child handler | GNU layer | INTEL layer |
|---|---|---|
| `omp_set_num_threads(1)` only (program linked to libgomp) | 1 | 1 |
| nothing (program without OpenMP) | 4 | 4 |
| `MKL_Set_Num_Threads(1)` | 1 | 1 |

MKL follows the OpenMP ICV when no MKL count is set, so the OpenMP builds were covered by
accident; a build without OpenMP (`CMAKE_DISABLE_FIND_PACKAGE_OpenMP`, the R / Linux jobs
without OpenMP) left MKL at 4 threads in the child. Fix: the child handler also calls
`MKL_Set_Num_Threads(1)` (global count; C entry point, weak symbol / dlsym / GetProcAddress as
the others). The parent's count is not changed. Test: `BlasThreadsTest` fork case checks the
child's MKL count and the parent's after. Port to: T6 (fork handlers) with T5's MKL entry points.

### 2.3 `Objective, gradient and update do not depend on the OpenMP thread count` — test premise

`omp_set_num_threads(1)` vs `4` lowers the budget, hence the BLAS count the library sets
(`blas_threads_for` / `mkl_threads_for`), not only the OpenMP loops. OpenBLAS's results did not
change with its count on this test; MKL's do: n = 900, relative gaps ll 1.5e-11, gradient
1.1e-9, updated ll 3.7e-10. With `MKL_NUM_THREADS=1` (count cannot change) both layers are
bit-identical. Change: exact equality kept whenever the BLAS count cannot change; with an MKL
loaded with more than 1 thread, 1e-8 relative (9x the largest gap), with a warning; new ctest
`BlasThreadsTest/BLAS_NUM_THREADS=1` (`OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1`) requires
bit-identity on every backend. The library's documented dependence on the BLAS count
(`parallel.md` 4) is unchanged. Port to: T5 (test).

### 2.4 `KrigingFitTest - y in the span of the trend, noise models` — library, fixed

`CHECK(knug.sigma2() == 0.0)` / `nugget() == 0.0` failed: the degenerate path (y in the span
of the trend) is documented to commit sigma2 = 0 (nugget = 0) but computed SSE / n from the
rounding residual. Constant y, gauss, n = 10:

| | OpenBLAS | MKL (GNU, INTEL, SEQUENTIAL) |
|---|---|---|
| nugget model: sigma2, nugget | 0, 0 | 2.42e-32, 2.43e-35 |
| no noise: sigma2 | 2.42e-32 | 2.42e-32 |

Fix (`Kriging.cpp`, degenerate exact path): estimated sigma2 and nugget set to exactly 0. Not
T5 / T6: port to master.

### 2.5 `KrigingWarmRestartTest` (warm update < 0.8 x cold fit) — timing noise, test fixed

Warm 55–62 ms vs cold 16–27 ms (GNU), reproducible. One fit / update sequence repeated in one
process (n = 33, ms; OpenBLAS: update 0.7–1.4 ms every time):

```
MKL GNU, first update of the main thread per process (6 processes): 33.9 48.7 1.7 61.7 51.9 45.7
then:                                                                 0.8 – 5.6
```

The first threaded MKL call of a thread costs 30–60 ms here (MKL per-thread initialization /
OpenMP team start), whoever sets its count; under `ctest -j2` the INTEL layer added contention
(libiomp5's 200 ms spin, `KMP_BLOCKTIME`, against the other test's threads: warm 108 ms best of
3). Not T5 / T6 (the counts are unchanged in this test: pool of 1, share = budget). Changes:
the test takes the best of 3 runs (each from a new fit) for both times, the assertion is
unchanged; the workflow runs `ctest` serially, as `tools/linux-macos/test.sh` does. After:
8 / 8 isolated runs pass (both layers), full serial suites pass.

### 2.6 After the fixes

| Build / BLAS | ctest | fork case |
|---|---|---|
| OpenBLAS build, OpenBLAS (`ctest -j2`) | 238 / 238 | included |
| OpenBLAS build, MKL preloaded, GNU (serial) | 237 / 237 | 1 / 1 with `MKL_DISABLE_FAST_MM=1` |
| OpenBLAS build, MKL preloaded, INTEL (serial) | 237 / 237 | 1 / 1 with `MKL_DISABLE_FAST_MM=1` |
| linked to MKL (`-Dmkl_rt_LIBRARY`), GNU (serial) | 237 / 237 | 1 / 1 with `MKL_DISABLE_FAST_MM=1` |
| linked to MKL, INTEL (serial) | 237 / 237 | 1 / 1 with `MKL_DISABLE_FAST_MM=1` |

## 3. CI workflow `.github/workflows/blas-backends.yml`

Triggers: `pull_request` touching `src/**`, `tests/**` or the workflow; `workflow_dispatch`.
Release, C++ only, `OPENBLAS_CORETYPE=Haswell` as the other Linux jobs (#429).

- **Linux, OpenBLAS build, oneMKL preloaded** (ubuntu-24.04, `libopenblas-pthread-dev`):
  BlasBackendTest + full `ctest` with OpenBLAS (`LK_EXPECT_BLAS=openblas`), then with the
  oneMKL wheel preloaded (`LD_PRELOAD=libmkl_rt.so.2`), GNU then INTEL threading layer
  (`LK_EXPECT_BLAS=mkl`); `MKL_VERBOSE` histogram of a BFGS4 fit; timing case (informative).
- **Linux, linked to oneMKL**: how Armadillo picks the BLAS
  (`dependencies/armadillo-code/CMakeLists.txt`): FlexiBLAS first (`ALLOW_FLEXIBLAS_LINUX`),
  then MKL (`ARMA_FindMKL`: `find_library(mkl_rt)` under `$MKLROOT/lib/intel64`, `/opt/intel/...`,
  not `BLA_VENDOR`), then OpenBLAS, BLAS / LAPACK. The wheel has no `libmkl_rt.so` link and no
  MKLROOT layout: the job passes `-Dmkl_rt_LIBRARY=<wheel>/libmkl_rt.so.2` (a preset cache
  variable skips the search); the configure step fails if `MKL_FOUND` is not `YES`. Then the
  same GNU / INTEL runs.
- **macOS arm64** (macos-15): Accelerate (Armadillo's default there), BlasBackendTest
  (`LK_EXPECT_BLAS=accelerate`) + full `ctest`; timing case with default threading,
  `BLASSetThreading(single)` (macOS 15+) and `VECLIB_MAXIMUM_THREADS=1`.

`LK_EXPECT_BLAS` makes a backend that silently failed to load an error instead of a skip of
every backend case. The fork case runs with `MKL_DISABLE_FAST_MM=1` under MKL (2.1).

## 4. Tests `[backend]` (`tests/BlasBackendTest.cpp`)

Each case checks the BLAS that serves `dgemm_` in the process (`dladdr`), and warns / returns
for another one. On Linux, `tests/backend_probe.c` (C, compiled into the test only, outside
Armadillo's prototypes) defines `dpotrf_` in the executable: ELF's lookup order sends the
library's Cholesky calls there first; it records the calling thread, `MKL_Get_Max_Threads()`
and `openblas_get_num_threads()` in that thread, then forwards to the next definition
(`RTLD_NEXT`: MKL or OpenBLAS). This reads the count **in the thread of each actual LAPACK call**
of the fit, not a count read beside it.

| Case | Checks | Result (4 cores) |
|---|---|---|
| BLAS backend in use | prints the backend; equals `LK_EXPECT_BLAS` when set | — |
| MKL: each multistart worker runs its BLAS at its share | BFGS4: `pool` distinct worker threads, all their calls at `share` (MKL loaded with more), the caller's at the budget, caller's count unchanged | MKL loaded with 4, pool 4 x 1: workers NThr 1 (130 calls), caller NThr 4 (2 calls); GNU, INTEL, preloaded, linked |
| OpenBLAS: each multistart worker runs its BLAS at its share | same with the process-global OpenBLAS count (T6: smallest live request) | pass |
| MKL: the caller's count is restored | fit / predict / simulate / update, count and thread-local count after each: no local count at budget 100 % and 1 thread, caller's own local 3 at budget 50 % | pass (GNU, INTEL) |
| BFGS4 with 1 or 4 workers reaches the same optimum | `thread_pool_size` 1 vs 4, matern5_2 and gauss, n = 300 | below |
| Fitted theta for 200 <= n < 800 matches the shared reference | n = 250, 500, 750, both kernels, single start: LL and gradient at a fixed theta, fitted theta and LL | below |
| Multistart timing per BLAS threading (hidden, `[.timing]`) | BFGS vs BFGS4 wall time, Accelerate `BLASSetThreading` | sections 5–6 |

`MKL_VERBOSE=1` on the BFGS4 case (GNU, preloaded): 1447 calls `NThr:1`, 21 calls `NThr:4`
(the calling thread's setup and final model). Before T5 every call ran at MKL's default count
with 4 workers (16 threads on 4 cores; T5 log, not re-measured here).

### BFGS4, pool of 1 vs 4 (relative gaps)

| BLAS | matern5_2 theta / LL | gauss theta / LL |
|---|---|---|
| OpenBLAS, 4 threads | 8.1e-6 / 7.7e-12 | 2.1e-5 / 3.8e-9 |
| OpenBLAS, 1 thread | 0 / 0 | 0 / 0 |
| MKL GNU or INTEL, 4 threads | 8.8e-5 / 4.7e-9 | 5.7e-9 / 1.4e-9 |
| MKL, 1 thread | 0 / 0 | 0 / 0 |

Same starts, same best start: the only difference is the BLAS count (budget 4 vs share 1),
whose rounding moves the BFGS stopping point in the flat LL. Tolerances: LL 1e-7, theta 1e-3.

### Shared reference, 200 <= n < 800 (relative gaps to OpenBLAS 1 thread)

Data: Weyl sequence in [0, 1)^3 (`fmod`, no RNG distribution: libstdc++ and libc++ differ),
y = sin(6 x1) + cos(4 x2) x3 + 0.5 sin(5 x3). Largest gaps over n = 250, 500, 750:

| BLAS | matern5_2: LL / gradient at theta = (0.3, 0.5, 0.7) | matern5_2: fitted theta / LL | gauss: LL / gradient at fixed theta | gauss: fitted theta / LL |
|---|---|---|---|---|
| OpenBLAS, 4 threads | 3.2e-13 / 2.3e-12 | 3.5e-5 / 1.6e-8 | 2.2e-9 / 3.2e-8 | 3.2e-8 / 2.4e-9 |
| MKL, 4 threads (GNU = INTEL) | 1.5e-13 / 4.0e-12 | 1.5e-4 / 2.2e-8 | 1.4e-9 / 2.9e-8 | 5.3e-8 / 1.5e-9 |
| MKL, 1 thread | (same order) | 2.2e-5 / 1.2e-8 | | 1.0e-7 / 2.2e-9 |
| Accelerate (macOS CI) | section 5 | | | |

Tolerances (about 10–50 x these): fixed-theta LL 1e-11 (matern5_2) / 1e-8 (gauss), gradient
1e-10 / 1e-6; fitted LL 1e-7, theta 1e-3 / 1e-5.

**Limitation of this check.** T5's threshold effect (before T5, Linux vs macOS / Windows,
same BLAS) was 1.9e-4–3.7e-4 on matern5_2 fitted theta and 3.4e-8 on the gauss gradient: the
same order as the gaps between BLAS backends and thread counts above. A cross-platform
comparison of fitted values therefore cannot distinguish a regression of T5 from a BLAS
difference; T5's own check (byte-identical outputs across thresholds, one BLAS: T5 log 1) is the
discriminating one. The shared reference catches gross OS-dependent differences (a different
optimum, a kernel / libm / summation bug), not last-digit ones.

## 5. macOS arm64, Accelerate

(CI results: see the update at the end of this log.)

## 6. Timings (local, informative; `[.timing]`, median of 3, matern5_2, budget 4)

| BLAS | n = 200 BFGS / BFGS4 | n = 400 | n = 800 |
|---|---|---|---|
| OpenBLAS (4 threads at load) | 0.85 / 0.19 s | 1.00 / 0.57 s | 4.11 / 2.31 s |
| OpenBLAS, `OPENBLAS_NUM_THREADS=1` | 0.21 / 0.16 s | 1.09 / 0.54 s | 6.91 / 2.38 s |
| MKL GNU | 0.15 / 0.12 s | 0.50 / 0.79 s | 2.04 / 1.62 s |

Single-run medians on a noisy 4-vCPU VM: orders of magnitude, not a benchmark. The BFGS
(single start) value includes the long-range restart.

## Decisions

1. The fork prepare handler initializes the lazy statics of the call path (2.1); MKL's global
   count is set to 1 in the child (2.2).
2. MKL fast-memory-manager deadlock after `fork()` with concurrent MKL work: known MKL
   limitation, documented (`parallel.md` 7), `MKL_DISABLE_FAST_MM=1` as workaround; the library
   does not disable MKL's memory manager for the whole process.
3. Tests keep bit-identity where the BLAS count cannot change and compare at measured
   rounding level where it does; tolerances are 10–50 x the measured gaps.
4. CI MKL runs: preloaded on OpenBLAS and linked, both threading layers, serial `ctest`.
