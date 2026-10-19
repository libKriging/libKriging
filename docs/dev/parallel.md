# Parallelism in libKriging

How the library uses threads, how to control it, and what was measured. Plan
and history: [ThreadBudget-TODO.md](ThreadBudget-TODO.md); measurements:
[bench/thread-budget/RESULTS.md](../../bench/thread-budget/RESULTS.md).
Status: branch `perf/thread-budget` (PR #415), Linux measurements only; macOS in
#416, Windows in #417.

## 1. Three levels of parallelism

| Level | What | Code | Runs on |
|---|---|---|---|
| 1. Task pool | the starts of a multistart fit (`BFGS10`, ...) of `Kriging` (all noise models), `WarpKriging`, `MLPKriging`; WarpKriging calibration folds | `detail::run_task_pool`, `src/lib/TaskPool.cpp` | `std::thread` workers (not OpenMP) |
| 2. OpenMP loops | covariance matrices (`cholCov`, `update_cholCov`, `covMat_*`), `compute_dX`, log-likelihood gradient, predict / simulate cross-covariances `R_on`, Vecchia neighbour searches, NestedKriging pair loop | `LinearAlgebra.cpp`, `KrigingImpl.cpp`, `Kriging.cpp`, `NestedKriging.cpp` | OpenMP teams of the calling thread |
| 3. BLAS / LAPACK | Cholesky, triangular solves, products | linked BLAS (see section 5) | the BLAS's own threads |

BLAS calls and OpenMP loops of one thread run **one after the other**, never at
the same time: each may use the whole share of its thread. They do not split it.

## 2. The thread budget

One number, `budget`, bounds the threads of all three levels:

```
budget = round(share × physical cores)        at least 1
share  = Optim::set_num_threads_percent(x) / LK_NUM_THREADS_PERCENT=x
         x <= 1: ratio (0.5), x > 1: percentage (50), 0 or unset: 100 %
```

- **Physical cores**, not logical CPUs: on a hyperthreaded machine, BLAS / OpenMP
  teams on the logical CPUs are 1.3–2× slower (section 6). Counted from the cores
  of the CPUs in the affinity mask, capped by the cgroup CPU quota (Linux sysfs);
  `hw.physicalcpu` (macOS); `GetLogicalProcessorInformationEx` (Windows);
  `std::thread::hardware_concurrency()` if unreadable.
- Never more than the calling thread's OpenMP limit (`OMP_NUM_THREADS`,
  `omp_set_num_threads`).
- The OpenBLAS count it was loaded with (`OPENBLAS_NUM_THREADS`) bounds the BLAS
  only: below the budget it is kept as it is, and the pool and the OpenMP loops
  still get the whole budget.
- `Optim::get_num_threads()` returns the resolved budget.

API (C++; bindings to come):

```cpp
Optim::set_num_threads_percent(50);   // or 0.5: half of the physical cores
Optim::get_num_threads_percent();     // the setting
Optim::get_num_threads();             // resolved budget, e.g. 5 on 10 cores
Optim::set_thread_pool_size(n);       // optional cap on the multistart workers
```

## 3. Sharing the budget: multistart first

The starts of a multistart are independent: they get the budget first, the BLAS
and OpenMP of each start the rest.

```
workers = min(number of starts, thread_pool_size or budget)
share   = max(1, budget / workers)         per worker, for its BLAS and its OpenMP loops
```

| Budget | Starts | Workers × share |
|---|---|---|
| 10 | 1 (`BFGS`) | inline, 1 × 10 |
| 10 | 4 | 4 × 2 |
| 10 | 20 | 10 × 1 |
| 5 | 4 | 4 × 1 (the remainder is not redistributed) |

- OpenMP: every loop is `num_threads(detail::omp_threads())`, the worker's share
  (or the budget outside the pool). The library never changes the OpenMP ICV of a
  thread (except around `fork()`, see section 7).
- BLAS: the OpenBLAS count is process-global. Each public fit / predict /
  simulate / update sets it to the budget for the call, the pool to the share
  for its lifetime, and both restore it; an OpenBLAS loaded with fewer threads
  (`OPENBLAS_NUM_THREADS`) is left as it is.
- No nested pools: a pool started from a worker runs its tasks inline.
- No start stagger for the Kriging multistart (it cost 2× at n = 100 with 20
  workers); kept for WarpKriging (`Optim::thread_start_delay_ms`).

Thresholds below which a loop stays serial: `kOmpMinRows` (covariance, gradient)
= 200 on Linux, 800 elsewhere until measured (`src/lib/OmpLoops.hpp`); `n >= 200`
(`compute_dX`, `covMat_sym_*`); `n_o × n_n >= 40000` (predict / simulate `R_on`,
`covMat_rect`, Vecchia, NestedKriging).

## 4. Reproducibility

- Results do not depend on the OpenMP thread count: each loop writes disjoint
  entries or sums per-row partial results in a fixed order (`BlasThreadsTest`).
- They may depend on the BLAS thread count and build (FAQ).
- `BFGS20` matches the best of 20 `BFGS1` runs up to rounding: with MKL, BLAS
  calls from secondary threads may round differently (checked to 1e-3 relative in
  `KrigingTest`). The Kriging multistart was sequential before for that reason.

## 5. BLAS backends

The library drives **OpenBLAS only** (`openblas_set_num_threads` /
`openblas_get_num_threads`, weak symbols; `GetProcAddress` on Windows). With any
other BLAS only the OpenMP and pool parts of the budget apply.

| BLAS | Where | Threads | Run-time control | Driven by libKriging |
|---|---|---|---|---|
| OpenBLAS pthreads | Linux distributions, R Linux binary (bundled), Windows (conda / bundled `openblas.dll`), macOS Intel (brew) | own pool, spins after each call and for ~100 ms after loading (6.2) | `openblas_set_num_threads` (global) | yes |
| OpenBLAS OpenMP | Linux (`libopenblas0-openmp`) | libgomp team of the caller | follows `OMP_NUM_THREADS` | yes |
| Intel MKL / oneMKL | Linux / Windows x86, MATLAB | Intel OpenMP / GNU OpenMP / TBB / sequential | `mkl_set_num_threads`, **`mkl_set_num_threads_local` (per thread)** | no (candidate) |
| Apple Accelerate | macOS arm64 | GCD | `VECLIB_MAXIMUM_THREADS` at load; `BLASSetThreading` (macOS 15+, single / multi only) | no (#416) |
| FlexiBLAS | Fedora / RHEL / openSUSE; found first by Armadillo on Linux | the selected backend's | `flexiblas_set_num_threads` | no (`openblas_*` probably not visible through it) |
| BLIS / AMD AOCL | Linux, Windows | pthreads / OpenMP | `bli_thread_set_num_threads` | no |
| Reference (Netlib, R's Rblas) | fallback | sequential | — | nothing to drive |

Bindings: numpy / scipy and Julia load their own OpenBLAS (renamed / ILP64
symbols): a second OpenBLAS in the process, not driven by libKriging; MATLAB uses
its own MKL (`maxNumCompThreads`).

## 6. Measurements (Linux, i9-10900K 10 cores / 20 threads, OpenBLAS 0.3.26)

### 6.1 What the budget gives

Full fits (Kriging gauss, d = 4), no environment variable, 300 ms warm-up,
"before" = branch start (#414 head: sequential multistart, threshold 800, cap of
2 threads):

| n | optim | before | budget | gain |
|---|---|---|---|---|
| 100 | BFGS10 | 167 ms | 40 ms | 4.2 |
| 100 | BFGS20 | 445 ms | 103 ms | 4.3 |
| 500 | BFGS | 265 ms | 127 ms | 2.1 |
| 500 | BFGS10 | 10.0 s | 2.84 s | 3.5 |
| 500 | BFGS20 | 27.7 s | 3.38 s | 8.2 |
| 2000 | BFGS | 6.22 s | 3.64 s | 1.7 |

n = 100 with a single start (8 vs 58 ms) is not comparable: the BLAS thread count
changes the last bits of theta, which flips the restart test of the fit; the
slower run restarts 3 times and reaches a much better optimum (RESULTS.md 10).

predict / simulate / update: ×1.5–3.8 (single-point predict and small updates
unchanged, not thread-bound). Multistart in the pool instead of sequential:
×2.4–7.3.

Why: with both BLAS and OpenMP at the physical cores, the OpenMP loops pay off
from n = 200 (×3 per evaluation at n = 200–500); with either at the logical CPUs,
they collapse (up to 27–38× slower) because the spinning OpenBLAS pthreads
threads and the OpenMP team share the hyperthreads. The lower threshold and the
removal of the former cap of 2 threads must ship **with** the budget.

### 6.2 OpenBLAS pthreads: start-up transient only

OpenBLAS pthreads creates its threads when it is loaded (`OPENBLAS_NUM_THREADS`,
else the logical CPUs: 20 here); they spin for `OPENBLAS_THREAD_TIMEOUT`
(~50–100 ms here) before sleeping. Work done in that window competes with them:
the first single-start fit of a fresh process at n = 500 takes ~215 ms instead
of ~125 ms. Afterwards, lowering the count at run time (what the library does) is
as fast as setting `OPENBLAS_NUM_THREADS` before loading (RESULTS.md, section 9):

| n | per evaluation, loaded with 20, set to 10 | set to 1 | `OPENBLAS_NUM_THREADS=10` |
|---|---|---|---|
| 200 | 1.2 ms | 1.6 ms | 1.2 ms |
| 500 | 8.2–9.2 ms | 11.5–13.7 ms | 8.2 ms |
| 1000 | 33 ms | 62 ms | 37 ms |

- Matters only for short programs (a benchmark, a script that fits right after
  loading the library); not for an R / Python session that loads the package
  first. `OPENBLAS_NUM_THREADS` = physical cores or `OPENBLAS_THREAD_TIMEOUT=4`
  removes it.
- The benchmarks busy-wait 300 ms before measuring (`LK_BENCH_WARMUP_MS`): runs
  made before this was found overstate the cost of BLAS at 20 threads for
  short measurements (n = 200–500), the steady-state conclusions of 6.1 hold.

## 7. Fork safety

`pthread_atfork` handlers (`Kriging.cpp`) set BLAS and OpenMP to 1 thread around
`fork()` and restore the parent's counts after (R `parallel::mclapply`, #325 /
#387). The R Linux binary bundles OpenBLAS pthreads for that reason (the OpenMP
build deadlocks in forked workers). With `mclapply`, give each worker one thread
(`Optim::set_num_threads_percent` small, or `OPENBLAS_NUM_THREADS=1`).

## 8. Open decisions

1. ~~OpenBLAS loaded with more threads than the budget: leave it, run OpenMP
   and the multistart on 1 thread.~~ Dropped: the cost it addressed is the
   start-up transient of 6.2, and the rule was 2–4.7× slower elsewhere. A switch
   of OpenBLAS to 1 thread for small problems was also measured and dropped
   (slower at every size in steady state).
2. ~~OpenBLAS loaded with fewer threads than the budget capped the whole
   budget.~~ Fixed: it bounds the BLAS only (section 2).
3. **Pool workers on the logical CPUs** when each worker has one thread:
   20 workers are 0–10 % faster than 10 for `BFGS20` (n = 100, 500), within 1 %
   for `BFGS10` (after the warm-up; the 6–15 % measured before included the
   start-up transient).
6. **Restart test of the fit** (`sol_to_lb < eps` in `Kriging::fit_impl`):
   flips on the last bits of theta, so a single-start fit can end at a much
   better or worse optimum depending on the BLAS thread count (n = 100:
   f = 46.2 vs 107.2). Independent of the budget; to be made robust (tolerance
   relative to the bounds) in its own issue.
4. **Bindings setting `OPENBLAS_NUM_THREADS`** to the physical cores before
   loading their OpenBLAS (R binary), or documentation only.
5. **Other BLAS**: MKL (`mkl_set_num_threads_local` gives a real per-worker
   share), FlexiBLAS, BLIS — not driven yet.

## 9. Environment variables

| Variable | Read by | Effect |
|---|---|---|
| `LK_NUM_THREADS_PERCENT` | libKriging | budget share (x <= 1 ratio, x > 1 percentage) |
| `LK_THREAD_POOL_SIZE` | libKriging | cap on the multistart workers |
| `LK_THREAD_START_DELAY_MS` | libKriging | WarpKriging start stagger (10 ms) |
| `LK_BENCH_WARMUP_MS` | benchmarks | busy-wait before measuring (300 ms, see 6.2) |
| `OMP_NUM_THREADS` | OpenMP runtime | caps the budget |
| `OPENBLAS_NUM_THREADS` | OpenBLAS at load | size of its thread pool; below the budget, kept by libKriging |
| `OPENBLAS_THREAD_TIMEOUT` | OpenBLAS at load | spin time of its idle threads |
| `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `BLIS_NUM_THREADS` | other BLAS | their own thread counts |
| `GOMP_SPINCOUNT`, `KMP_BLOCKTIME` | libgomp / libomp | spin of idle OpenMP threads |
