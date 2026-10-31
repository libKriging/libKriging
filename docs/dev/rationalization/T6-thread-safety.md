# T6. Thread safety — work log

Task specification: `docs/dev/Rationalization.md`, section T6.

## Status

Done (PR #446, draft). Measurements: Linux x86_64, 4 CPUs, GCC 13, OpenBLAS 0.3.26
pthreads (Ubuntu 24.04 `libopenblas0-pthread`).

## Findings (beyond section 1.3)

| # | Shared state | Before | Effect of concurrent calls |
|---|---|---|---|
| 1 | OpenBLAS count (`ScopedBlasThreads`) | each scope saved the count it found and restored it on exit, no lock | interleaved save / restore: the count left after the calls is any of the intermediate values (stress test: 4 instead of the caller's 3) |
| 2 | Thread budget | every top-level call took the whole budget for its pool, its OpenMP loops and its BLAS | k concurrent calls run k × budget threads |
| 3 | `Random::engine` / `Random::seed` (`std::mt19937`, global) | reseeded by every multistart fit (`Random::init`) and every `simulate` (`reset_seed`), then drawn from | data race (UB); the starts of a fit and the draws of a simulation depend on the interleaving |
| 4 | `s_rcond_mutex` (`LinearAlgebra::rcond_chol`) | global mutex, not handled around `fork()` | a `fork()` made while another thread is in `rcond_chol` leaves it locked in the child, which deadlocks at its first fit (stack trace: child blocked in `rcond_chol` ← `safe_chol_lower` ← `cholCov`) |
| 5 | `Optim::*`, `LinearAlgebra::*` settings | plain mutable statics | read concurrently: safe; written while calls run: data race |

Not found: other mutable statics in `src/lib` (function-local statics are
`const`, atomic or initialized once). Armadillo's RNG (WarpKriging warp
initialization, subset-of-data selection: each reseeds it, then draws) is
`thread_local` except where Armadillo disables it (MinGW, Cygwin / MSYS, GCC on
macOS): there concurrent WarpKriging fits or subset selections still share it
(open item).

## Decisions

1. **One mutex-protected state for the OpenBLAS count** (`ThreadBudget.cpp`,
   `BlasCountState`): every `ScopedBlasThreads` (the `BudgetScope` of each public
   call, the multistart pool, `LinearAlgebra::check`) registers its request; the
   count in effect is the **smallest live request**; the first scope saves the
   count it finds, the last one restores it, whatever the closing order. The
   minimum, not the latest request: the count is process-global, so the most
   constrained call bounds every call's BLAS (a pool worker share of 1 must not
   be raised to the budget by another thread's predict). The state is never
   destroyed (scopes may close during static destruction).
2. **Budget shared between concurrent top-level calls**: `budget_total()` =
   `budget_full()` / number of public calls running in user threads (counted by
   `BudgetScope`, nested calls of one thread once, pool workers not), at least
   1. It is read when a call sizes its pool and at each OpenMP loop, so a call
   started alone keeps its pool and shrinks its later loops when others start;
   the BLAS count follows through decision 1 (the later call requests the
   smaller share). Results do not depend on the pool size nor on the OpenMP
   count (parallel.md, section 4); they depend on the BLAS count by rounding
   only. `Optim::get_num_threads()` returns `budget_full()` (the setting's
   resolution, not a value that changes with other threads' calls).
   Alternatives not kept: a fixed share at the entry of each call (the first
   call keeps everything); a global semaphore of the budget (calls would wait
   on each other).
3. **`Random` per thread**: `engine` and `seed` are `thread_local`. A
   single-threaded program behaves as before. Behaviour change for
   multithreaded C++ callers: `reset_seed()` applies to the calling thread
   (a new thread starts from 123). `reset_seed` is exposed only in the R
   binding (single R thread). `std::mt19937` is trivially destructible, which
   avoids the MinGW `thread_local` destructor issues that made Armadillo disable
   its own `thread_local` there.
4. **Fork handlers hold the library's mutexes**: prepare locks the count state,
   then the rcond mutex, then sets 1 thread as before; parent restores and
   unlocks; child unlocks, drops the requests and call counts of the parent's
   other threads (their scopes never close in the child: a generation counter
   makes the forking thread's own open scopes no-ops on exit) and sets 1
   thread. Lock order (count state, then rcond) is never reversed elsewhere:
   no scope is opened while the rcond mutex is held.
5. **Settings stay plain statics, documented as configuration-before-use**
   (`Optim.hpp`, `LinearAlgebra.hpp`, FAQ, parallel.md 3.1). `std::atomic`
   members were considered and not kept: they change the public header
   (`auto x = Optim::max_iteration;` no longer compiles, the `std::function`
   members cannot be atomic), and per-variable atomicity does not make a set
   during a fit meaningful (a fit reads `max_iteration`, tolerances,
   `reparametrize` at different times and could mix two configurations). The
   C++ API listing of `tools/checks/binding_api/dump_cpp.py` is unchanged.

## Stress test

`tests/ConcurrentCallsTest.cpp` (ctest `ConcurrentCallsTest` and
`ConcurrentCallsTest/OPENBLAS_NUM_THREADS=1`):

- 5 jobs (gauss / matern5_2 / matern3_2; noise-free and nugget; `BFGS`,
  `BFGS2`, `BFGS4`; LL / LOO / LMP; n = 50, d = 2), each fit + predict (200
  points, stdev) + simulate (3 paths, 20 points); sequential reference, then 6
  user threads × 3 rounds running every job (3 × 1 under TSan); a monitor thread
  records the largest OpenBLAS count; the caller's count is set to 3 first.
- `Optim::get_num_threads()` is not divided while another thread fits.
- `fork()` 5 times while 3 threads fit `BFGS4`; each child checks its count is
  1, fits, exits (alarm 60 s).

| | before (`4611ff6`) | after |
|---|---|---|
| OpenBLAS count after the calls | 4 (caller's 3 not restored) | 3 |
| largest count seen | 3 | 3 |
| results vs sequential, default OpenBLAS | 4.1e-14 relative | 4.1e-14 relative |
| results vs sequential, `OPENBLAS_NUM_THREADS=1` | | 0 (bit-identical) |
| fork children | 4 of 5 deadlocked (SIGALRM) in run 1; runs 2–3 did not finish in 300 s | 5 / 5 exit 0, 3 runs |

The 4e-14 difference is the rounding of the BLAS sums with the smaller BLAS
share of concurrent calls (identical in every thread and round): the test
requires bit-identical results when the count cannot change (no OpenBLAS, or
loaded with 1 thread), 1e-10 relative otherwise. Before the fix, a job's result
also depended on the job run before it in the same process: `simulate(seed)`
reseeds the engine that draws the next fit's starts (unchanged, now per
thread); the test resets the seed before each job.

## ThreadSanitizer

GCC 13, `-DCMAKE_BUILD_TYPE=Debug -DSANITIZE=thread` (as `analysis.yml`), no
suppression file unless stated:

| Binary | Environment | before (`4611ff6`) | after |
|---|---|---|---|
| `ConcurrentCallsTest` | none (6 → 3 threads, pool and BLAS shares active) | 18, 4, 8 reports in 3 runs, all on `Random::engine` / `Random::seed` (`Random::init` in `fit_impl`, `reset_seed` in `simulate_impl`) | 0 in 5 runs |
| `ConcurrentCallsTest` | `OPENBLAS_NUM_THREADS=1` | | 0 |
| `ConcurrentCallsTest` | CI (`OMP_NUM_THREADS=1`, `.tsan-suppressions`) | | 0 |
| `BlasThreadsTest` | none | 20 reports | 20 reports (same) |
| `BlasThreadsTest` | CI | | 0 |

- The fork case is compiled out under TSan (TSan and `fork()` from a
  multithreaded process do not mix); it runs in the normal build.
- TSan does not see the OpenBLAS count race of finding 1 (`openblas_set_num_threads`
  is not instrumented): the stress test checks the count left after the calls
  instead, and the mutex makes it race-free by construction.
- `BlasThreadsTest` without the CI environment: reports inside OpenBLAS's own
  worker threads (`dscal_k_SKYLAKEX` writing a buffer later read by
  `chol_block` / `syrk`): OpenBLAS's thread pool is not instrumented, as
  libgomp's (`tools/linux-macos/test.sh`); identical before and after, and absent
  with the CI environment, where OpenBLAS also runs on 1 thread
  (`OMP_NUM_THREADS=1`). Under the CI environment the budget is 1: no pool, no
  BLAS share change, so `ConcurrentCallsTest` exercises the concurrent
  scopes fully only without it (rows 1–2).

## Full test suite

Release build, OpenBLAS 0.3.26 pthreads: `ctest -j2`: **230 / 230 passed**
(80 s), including `ConcurrentCallsTest` (3 cases) and
`ConcurrentCallsTest/OPENBLAS_NUM_THREADS=1`. Bindings not built here (no
signature change: `dump_cpp.py` listing identical before / after).

## Open items

- MKL / Accelerate: their thread counts are not driven (T5); with MKL,
  `mkl_set_num_threads_local` would give a per-thread count and remove the
  "smallest request" coupling between concurrent calls.
- Armadillo's RNG is global on MinGW / macOS GCC (Armadillo disables its
  `thread_local` there): concurrent WarpKriging fits or subset-of-data
  selections race on it on these platforms (R Windows binding).
- Two threads on the same model object: not supported (documented).
- `simulate(seed)` changes the seed of the next fit's starts in the same
  thread (pre-existing coupling, kept).
