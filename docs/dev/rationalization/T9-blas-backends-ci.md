# T9. CI of the BLAS backends (MKL, Accelerate) — work log

Task specification: `docs/dev/Rationalization.md`, section T9.

This branch merges T5 (#445) and T6 (#446) into the program branch (PR #449). The T5 / T6
merge conflicts (`BlasThreads.hpp`, `ThreadBudget.hpp`, `parallel.md`, CHANGELOG) had been
resolved by keeping both sides, unbuilt.

## Status

Done; the fixes found here have since been ported to T5 / T6 (section 0). Last CI runs before
the port: self-hosted run 41332870292 (commit 31f81f69) green on all three jobs; GitHub-hosted
run 41304855128 (commit e30acf5) green apart from the MLPKriging limitation (section 5).
Since 2026-11-02 16:42 UTC (first on this branch: commit 7e7fe984, run 41484067950), every
workflow run ends in `startup_failure`, for every workflow of the commit (`main.yml`,
`binding-api`, `notebooks`, `compat-r` as well as `blas-backends.yml`, including those the
commit does not touch) and on other branches: repository / account level, not this branch.
The merged branch (commit 6bce4ab6, run 41632134231: `startup_failure`) is checked locally only
(section 0); no retry or empty commit was pushed.

Two known limitations: MKL's fast memory manager after `fork()` with concurrent MKL work
(2.1), the single-start MLPKriging calibration case on macOS Release (5).

## 0. Ported to T5 / T6, what stays here (2026-11-03)

T5 (#445, 9fd2049) and T6 (#446, 26434ab) carry the fixes of this log that need only one of
them; this branch merges both again and keeps only what needs both, plus the T9 work.

| Section | Change | Now in |
|---|---|---|
| 2.1 | fork prepare handler initializes `physical_cores` / `initial_blas_threads` | T6 |
| 2.1 | ... and the MKL statics (`initial_mkl_threads`, `touch_mkl_entry_points`) | **T9** (needs T5's MKL count and T6's handlers) |
| 2.2 | MKL entry points `set_mkl_threads_global`, `touch_mkl_entry_points` | T5 |
| 2.2 | fork child handler calls `set_mkl_threads_global(1)`; fork test checks the child's MKL count | **T9** (needs both) |
| 2.3 | OpenMP-independence test premise under MKL, ctest `BlasThreadsTest/BLAS_NUM_THREADS=1` | T5 |
| 2.4 | degenerate fit: sigma2 = nugget = 0 exactly | master through PR #450 (decision D7); the copy here stays until this branch is rebased after #450 (the MKL runs need it) |
| 2.5 | KrigingWarmRestartTest best of 3 | **T9** |
| 2.6 | ConcurrentCallsTest budget-1 bit-identical case; full-budget tolerance 1e-4 relative per quantity (decision D2); fork case tagged `[fork]`, excluded under ASan / LSan only (D3) | T6 (4784784, merged here 2026-11-04) |
| — | T6 additions: fork verdict through a pipe (valgrind), no allocation in the fork child under ASan / LSan | T6 |
| 3–6 | workflow `blas-backends.yml`, `BlasBackendTest` / `backend_probe.c`, docs | **T9** |

T5 also reverted its LL gradient reduction (kept in T1, #441). Consequence for the shared
reference of section 4: on Linux (threshold 200) n = 250 / 500 / 750 take the parallel path,
whose order is the one of the reference: identical to the bit with OpenBLAS at 1 thread
(checked after the merge). On macOS / Windows (threshold 800) they take the sequential path
again until T1: gaps of the order T5 measured before its change (fitted theta 3.7e-4,
gradient 3.4e-8), within the tolerances (1e-3, 1e-6) but not run in CI since (startup_failure).

The ASan / LSan change of T6 (the fork child does not fit) applies to sanitizer builds only:
the MKL jobs (Release) still fit in the child, so the fork case still exercises MKL there.

2026-11-05: T6 2037a10 merged (ConcurrentCallsTest reads the OpenBLAS count with
GetProcAddress under MSVC, as the library: with weak symbols only it read 0 on the Julia
Windows job and required bit-identity, gaps 2.3e-9–7.0e-7). Same defect for MKL, fixed here
(test only, to port to T6): `mkl_count()` read MKL through ELF weak symbols only, while the
library drives MKL also through `dlsym` (macOS / other POSIX) and `GetProcAddress` on
`mkl_rt*.dll`; it now uses the library's three mechanisms. Check: MKL preloaded with
`OPENBLAS_NUM_THREADS=1`, the test reads "OpenBLAS 1, MKL 4" and applies 1e-4 instead of
requiring bit-identity. ctest after the merge: OpenBLAS 239 / 239, MKL GNU and INTEL
237 / 237 + 2 / 2.

2026-11-06: master merged (a866c44, #436 "small-problem BLAS rule": OpenBLAS at 1 thread
for a fit / predict / update on fewer than `Optim::blas_threads_min_n` observations; automatic
threshold 500 when the BLAS threads and the OpenMP team outnumber the logical CPUs). Conflicts
in `ThreadBudget.cpp` / `.hpp`, resolved as follows (to carry into the program branch when it
merges master):
- `BudgetScope(n)` keeps T6's `CallCount` first (the rule's `budget_total()` counts the call)
  and T5's per-thread MKL scope; the rule only changes the count the scope's
  `ScopedBlasThreads` requests (1 instead of `blas_threads_for(budget)`). It therefore goes
  through T6's shared count state: smallest live request in effect, restored by the last
  scope, never a direct `openblas_set_num_threads`. Consequence (OpenBLAS's count is
  process-global): while a small-problem call runs, concurrent calls run their OpenBLAS at 1
  thread too; results depend on that by rounding only (`ConcurrentCallsTest`, D2).
- The automatic condition read the current OpenBLAS count when the call leaves it as is; with
  T6 that count may be another thread's transient request. It now reads the count OpenBLAS was
  loaded with (`initial_blas_threads()`, on which `blas_threads_for` and the restore already
  rely), so a call's decision does not depend on the other calls running.
- Concurrent calls: the condition uses `budget_total()`, already divided by the number of
  running calls, so it holds less often when several calls share the budget (their
  oversubscription is already removed by the split).
- MKL: not concerned (#436 measured OpenBLAS only); the calling thread's MKL count stays
  `mkl_threads_for(budget_total())`. With MKL preloaded on an OpenBLAS build, the rule lowers
  an OpenBLAS count that the BLAS calls do not use: harmless.
- Pool workers: unchanged (the pool sets their share; the rule applies to the calling thread's
  scope only).
- On the local 4-core VM the automatic rule is active (budget 4, OpenBLAS loaded with 4, 4
  logical CPUs): the T5 / T6 / T9 thread tests now mostly run OpenBLAS at 1 thread for n < 500.
  They were also run with the rule off (`LK_BLAS_THREADS_MIN_N=0`): 29 / 29.

Checks after the merge: OpenBLAS ctest 240 / 240 (with #436's new test: 1 thread at n = 60,
count restored, 4 threads at n = 120); oneMKL preloaded, GNU and INTEL: 238 / 238 + 2 / 2.

Checks of the merged branch (local, Xeon 4 cores, Release):

| BLAS | ctest |
|---|---|
| OpenBLAS | 239 / 239 (also after the merge of T6 4784784, 2026-11-04) |
| oneMKL preloaded, GNU | 237 / 237 + 2 / 2, also after the merge of T6 4784784 (fork case and `ConcurrentCallsTest/OPENBLAS_NUM_THREADS=1` with `MKL_DISABLE_FAST_MM=1`) |
| oneMKL preloaded, INTEL | 237 / 237 + 2 / 2 (also after the merge of T6 4784784) |

`[backend]` on MKL: workers NThr 1 (130 calls), caller NThr 4, as before.

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
T5 / T6: goes to master through PR #450 (decision D7); this branch keeps its copy until it is
rebased after #450, since the MKL runs need it.

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

### 2.6 `Concurrent fit / predict / simulate from user threads` (ConcurrentCallsTest) on the CI runner — T6 test, fixed

Not seen locally (Intel, 4 cores: 6e-14; restricted to 2 cores with `taskset`: 0). On the CI
runner (AMD EPYC 9V74, 2 cores / 4 threads, budget 2, OpenBLAS with `OPENBLAS_CORETYPE=Haswell`)
every concurrent result of job 0 (gauss, single-start `BFGS`, LL) differed from the sequential
one by 6.6e-7 relative (identical in every thread and round), other jobs up to 9.5e-9; with MKL
preloaded 3.0e-7. Tolerance: 1e-10 (T6, "4e-14 measured"); and 0 in the
`OPENBLAS_NUM_THREADS=1` variant, which under MKL failed too.

Cause: 6 concurrent calls on a budget of 2 run their BLAS at 1 thread, the sequential reference
at 2; the OpenBLAS `OPENBLAS_NUM_THREADS=1` variant passed bit for bit on the same runner, so the
gap is the BLAS count's rounding, amplified by the optimizer's stopping point of a single-start
gauss fit (same mechanism as the BFGS4 pool 1 vs 4 gaps of section 4). T6's 1e-10 was the
rounding of one machine; the variant's premise ("the count cannot change") ignored MKL.

Change (`tests/ConcurrentCallsTest.cpp`):
- new case "Concurrent calls at a budget of 1 thread give the sequential results": sequential
  and 6 x 3 concurrent runs all at 1 BLAS / OpenMP thread, **bit-identical required on every
  backend** — what the concurrency itself can break (RNG, shared state, count save / restore)
  is checked at full strength, independently of the BLAS;
- full-budget case: bit-identical when neither OpenBLAS nor MKL can change count, else 1e-3.
  A first value, 1e-5, was set from the gauss observation alone (6.6e-7) and failed on the
  self-hosted i9-10900K (10 cores, OpenBLAS): matern5_2 BFGS4 at 7.2e-5, identical in every
  thread and round (BLAS-count rounding, not a race; the `OPENBLAS_NUM_THREADS=1` variant is
  bit-identical on the same machine). That scale had already been measured (BFGS4 pool 1 vs 4:
  8.8e-5, section 4); 1e-3 is the tolerance of the fitted theta between BLAS counts used in
  `BlasBackendTest`. A race or a shared RNG gives differences of order 1, and the budget-1 case
  checks the concurrency bit for bit.

Port to: T6.

**Superseded (2026-11-04, decisions D2 / D3, T6 4784784).** The full-budget tolerance is now
1e-4 relative, checked separately for theta, sigma2, objective, predicted mean, stdev and
simulations; the fork case fits in the child everywhere again and is tagged `[fork]`,
excluded only under ASan / LSan (`TEST_SPEC "~[fork]"`). Margin of 1e-4 against the gaps
measured: largest 7.2e-5 (i9-10900K, 10 cores, OpenBLAS, matern5_2 BFGS4: a ratio of 1.4),
6.6e-7 (AMD EPYC 2 cores), 3.0e-7 (AMD, MKL); BFGS4 pool 1 vs 4 (another data set, n = 300)
reached 8.8e-5 on theta under MKL. After the merge, 20 runs per configuration on the local
Xeon (4 cores and `taskset` 2 cores; OpenBLAS, MKL GNU, MKL INTEL; 120 runs): largest gaps
theta / sigma2 / objective / sims 0, mean 1.8e-15, stdev 4.1e-14 — this machine does not reach
the regime of the i9 (its budget is capped at its 4 logical CPUs), so it does not test the
margin; the i9 runner is no longer in CI (D1).

### 2.7 After the fixes

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
every backend case.

**Runners: GitHub-hosted only** (`ubuntu-24.04`, `macos-15`; user decision D1, 2026-11-04,
`docs/dev/Rationalization.md` section 3). From 2026-11-01 to 2026-11-03 the workflow ran on the
repository's self-hosted runners for branches of the repository (an i9-10900K Linux
workstation and an Apple silicon Mac); their results stay in this log as measurements
(section 3.2, sections 4–5), not as CI. Reasons for leaving them:
- a pull request of this public repository runs its own workflow files, so a fork can target
  any runner registered to the repository; routing forks to hosted runners in the workflow
  was not a security boundary (only the approval of outside-contributor runs is);
- the Linux runner was a personal workstation (personal account and home directory,
  persistent workspace), which the measures that would make it safe (dedicated user,
  ephemeral VM, network isolation) did not fit;
- the acceptance of T9 had been reached on hosted runners (run 41304855128).
What is lost: more than 4 cores in CI (the `ConcurrentCallsTest` tolerance defect showed only
at 10 cores) and short queues. Removing the runners from the repository (Settings → Actions →
Runners), not only from this workflow, is what removes their exposure.

The logic written for the self-hosted machines (tool discovery with `sudo -n`, OpenBLAS
runtime library as `openblas_LIBRARY`, installed oneAPI first, workspace cleaning, runner
variables) was removed with D1; kept: the configure check that Armadillo found OpenBLAS / MKL
(it falls back to the reference BLAS silently), `-DALLOW_FLEXIBLAS_LINUX=OFF`,
`permissions: contents: read`, `persist-credentials: false`, test steps gated on the build.


### 3.1 CI results (run 41300138594, commit e1fbae9)

Linux runners: AMD EPYC 9V74, 4 vCPUs = 2 cores x 2 threads: budget 2, MKL loaded with 2.

| Job / step | `[backend]` | ctest |
|---|---|---|
| OpenBLAS build, OpenBLAS | pass (34 assertions) | 250 / 251: ConcurrentCallsTest (2.6) |
| OpenBLAS build, MKL preloaded, GNU / INTEL | pass (53); workers NThr 1 (130 calls), caller NThr 2 (2 calls) | 248 / 250: ConcurrentCallsTest and its `OPENBLAS_NUM_THREADS=1` variant (2.6); fork case with `MKL_DISABLE_FAST_MM=1`: pass |
| MKL linked, GNU / INTEL | pass | same as preloaded |
| macOS 15 arm64, Accelerate | pass (29) | 233 / 234: MLPKriging calibration (section 5) |

`MKL_VERBOSE` (BFGS4, 2 workers x 1): 1447 calls `NThr:1`, 21 calls `NThr:2`.

Timing on the Linux runner (`[.timing]`, budget 2, informative): OpenBLAS BFGS / BFGS4 0.095 /
0.100 s (n = 200), 0.43 / 0.40 s (400), 1.72 / 4.62 s (800); MKL GNU 0.086 / 0.104, 0.39 / 0.42,
1.75 / 2.26 s. With 2 cores, BFGS4 runs 4 starts on 2 workers: twice a single start's work per
worker.

### 3.2 Self-hosted runners, 2026-11-01 (measurements; no longer CI since D1) (run 41331286043, commit 2d9932dd; macOS: run 41329668504)

| Runner | Machine | Result |
|---|---|---|
| `alien-runner` (Linux) | Intel i9-10900K, 10 cores / 20 threads, Ubuntu 24.04; OpenBLAS runtime only (`libopenblas.so.0`), sudo with a password; oneMKL wheel | `[backend]`: all pass with OpenBLAS (budget 10, OpenBLAS loaded with 20) and MKL (loaded with 10: BFGS4 pool 4 x 2, **workers NThr 2** in all 130 calls, caller NThr 10; `MKL_VERBOSE` 1447 calls `NThr:2`, 21 `NThr:10`); ctest: INTEL layer 251 / 251 in both builds; two failures, fixed in the next commit: `ConcurrentCallsTest` full-budget tolerance (2.6), and the `OPENBLAS_NUM_THREADS=1` variant, which also runs the fork case, run without `MKL_DISABLE_FAST_MM=1` under the GNU layer (children killed by their alarm: the MKL limitation of 2.1); the workflow now treats the variant as the fork case |
| `MacBook-Air-de-violette` (macOS) | Apple silicon, budget 10 | 234 / 234 except the MLPKriging calibration case (coverage 0.707, the known limitation of section 5) |

Shared reference on the i9 (OpenBLAS / MKL): largest gaps theta 1.0e-4, LL 1.5e-8, gradient at
fixed theta 3.1e-8: within the tolerances, same order as on the other machines.

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
| Accelerate (macOS 15 arm64, CI) | 4.2e-14 / 1.8e-12 | 1.0e-4 / 1.4e-8 | 8.6e-10 / 2.5e-8 | 5.4e-8 / 2.5e-9 |

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

CI run 41300138594, job "macOS arm64 Accelerate" (macos-15, 3 cores: budget 3, AppleClang,
Release). `dgemm_` from `Accelerate.framework/.../vecLib.framework/.../libBLAS.dylib`.

- `BlasBackendTest`: all `[backend]` cases pass (29 assertions; the MKL / OpenBLAS cases skip).
  BFGS4 with a pool of 1 or 4: **bit-identical** (the library does not change Accelerate's count,
  so nothing differs between the two runs but the order of the starts). Shared reference
  (Linux OpenBLAS): every gap within the tolerances, of the same order as between Linux
  backends (table in section 4: theta 1.0e-4 at most, matern5_2 n = 250).
- `ConcurrentCallsTest` (T6 stress test, including `fork()` during concurrent fits) and
  `BlasThreadsTest`: pass.
- Full `ctest`: 233 / 234 (run 41300138594) and 234 / 235 (run 41304855128, with the new
  budget-1 case of `ConcurrentCallsTest`: passes, bit-identical);
  `MLPKriging predictive variance is calibrated (#407)` fails, see below.
- Second timing sample (run 41304855128): BFGS4 multi / single 0.97, 1.02, 0.95 (n = 200,
  400, 800); with `VECLIB_MAXIMUM_THREADS=1` at load 1.05, 1.15, 1.03: run-to-run noise of
  about ±15 %, no consistent gain from single-threaded Accelerate; the decision below stands.

### BFGS4 vs BFGS, `BLASSetThreading` (macOS 15 `vecLib/thread_api.h`), `VECLIB_MAXIMUM_THREADS`

`[.timing]`, medians of 3, matern5_2, d = 3, budget 3 (pool of 3 workers x 1 for BFGS4):

| n | BFGS | BFGS4 | BFGS, `BLASSetThreading(single)` | BFGS4, single | BFGS4 multi / single |
|---|---|---|---|---|---|
| 200 | 0.098 s | 0.069 s | 0.104 s | 0.069 s | 0.99 |
| 400 | 0.435 s | 0.329 s | 0.524 s | 0.317 s | 1.04 |
| 800 | 1.981 s | 1.491 s | 2.698 s | 1.487 s | 1.00 |
| `VECLIB_MAXIMUM_THREADS=1` at load: 800 | 2.621 s | 1.667 s | | | |

- Single-threaded Accelerate during a multistart: no measurable gain (0.99–1.04, within
  the run-to-run noise); `VECLIB_MAXIMUM_THREADS=1` is slower for BFGS4 at n = 800 (1.67 vs
  1.49 s). On a single start it costs 20–36 % at n = 400–800.
- Self-hosted Mac, budget 10 (more cores than the 4 starts, the case left open on 3 cores):
  BFGS4 multi / single 1.00, 0.99, 0.97 (n = 200, 400, 800); with `VECLIB_MAXIMUM_THREADS=1`
  at load 1.11, 0.99, 1.01; BFGS4 / BFGS 0.54, 0.43, 0.55. No gain either.
- **Decision**: the library does not call `BLASSetThreading` around multistarts. It is
  process-wide (it would also slow every concurrent call of other threads), and it gave nothing
  here; the oversubscription that T5 documented for multithreaded BLAS in pool workers does not
  show with Accelerate on this 3-core runner. The FAQ / `parallel.md` advice
  (`VECLIB_MAXIMUM_THREADS=1` for multistart fits) is not supported by these numbers on 3 cores;
  it is kept as an option, not a recommendation: on 10 cores it brings nothing either.

### `MLPKriging predictive variance is calibrated (#407)` on macOS Release — known limitation

`optim=BFGS` case: coverage 0.72 (required > 0.8); uncalibrated Q2 -0.68, NLPD 47.3 (the
single start stopped at a poor mean, which no variance calibration compensates; the test's own
comment allows it for the NLPD check, #407 point 4). The `BFGS4` case passes.

Not caused by this branch: on Linux (Release, OpenBLAS) the branch and master give the same
values to every printed digit (single start: coverage 0.944, calibrated NLPD 5.3089 vs
uncalibrated 8.5564; BFGS4: coverage 1.0, NLPD 4.3149 vs 47.3406). On macOS the data and the
MLP initialization differ from Linux anyway (`arma::randu` goes through
`std::uniform_real_distribution`, whose libc++ and libstdc++ implementations differ), so the
single start meets another landscape. `main.yml`'s macOS job (Debug) passes this test on
master **and on this branch** (run 41300138490, commit e1fbae9, test step green): the failure is
specific to macOS Release (AppleClang optimizations change the rounding, hence the single
start's path), not to this branch. No master job runs the C++ tests in Release on macOS, so
master's behaviour there is not measured.

Decision: the test is left unchanged (making its coverage check depend on Q2 would hide this
case). The macOS job of `blas-backends.yml` runs it in a separate, reported, non-blocking step;
`main.yml`'s macOS job keeps it blocking. Follow-up for #407: a single-start MLPKriging fit can
end with Q2 < 0 depending on the data / platform; either the calibration test uses a design
where the single start is known to converge on every platform, or the single-start case only
checks what calibration can do (NLPD gain, `var_scale > 1`).

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
