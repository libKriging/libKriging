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
share of concurrent calls (identical in every thread and round). The first
version required bit-identical results when the count cannot change (no
OpenBLAS, or loaded with 1 thread) and 1e-10 relative otherwise; the CI showed
that 1e-10 was the rounding of one machine (section "CI of #446" below). Before the fix, a job's result
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

## CI of #446 (head `7b74d6e`) and fixes

Failed jobs, the same in the push and pull_request runs:

| Job | Failing test | Cause |
|---|---|---|
| Julia Windows, Octave Linux (`main.yml`), Coverage mode (`analysis.yml`) | `Concurrent fit / predict / simulate from user threads` | job 0 (gauss, single-start `BFGS`, LL) differs from the sequential run by 7.0e-7 (Windows), 4.6e-7 (Coverage) relative, identical in every thread and round; tolerance 1e-10 |
| Address Sanitizer | `fork() during concurrent fits` (and the `OPENBLAS_NUM_THREADS=1` variant, which runs it too) | one child killed by its 60 s alarm |
| Leak Sanitizer | same | all 5 children killed by the alarm |
| Memcheck mode | same | children exit 1; child 0 killed by the alarm |

### Tolerance (T9 log, section 2.6, ported)

The concurrent calls run their BLAS at a smaller share than the sequential
reference (6 calls on a budget of 2: 1 thread vs 2); the rounding change is
amplified by the stopping point of a single-start gauss fit. Not a race: the
variant `ConcurrentCallsTest/OPENBLAS_NUM_THREADS=1` passed bit for bit on the
same runners. Ported from T9 (`e30acf5`, `31f81f6`):

- new case "Concurrent calls at a budget of 1 thread give the sequential
  results": sequential and concurrent calls all at 1 BLAS / OpenMP thread,
  **bit-identical required on every backend**; it checks the concurrency itself
  (RNG, shared state, count save / restore) at full strength, independently of
  the BLAS rounding;
- full-budget case: bit-identical when neither OpenBLAS nor MKL can change
  count (MKL detected through `MKL_Get_Max_Threads`, weak symbol), else
  **1e-3 relative — provisional, decision pending with the user**. T9's
  measurements: 6.6e-7 (AMD EPYC 2 cores), 7.2e-5 (i9-10900K, matern5_2 BFGS4),
  BFGS4 with 1 vs 4 workers 8.8e-5; 1e-3 is the tolerance on theta between BLAS
  counts in T9's `BlasBackendTest`. A race or a shared RNG gives differences of
  order 1 (other starts), and is caught bit for bit by the budget-1 case.

### Fork prepare handler (T9 log, section 2.1, ported)

`blas_count_fork_prepare` now initializes the function-local statics read on
entry to every public call (`physical_cores()`, `initial_blas_threads()`) before
locking: a `fork()` made while another thread is initializing one would copy
its guard "in progress" into the child, whose first call then waits forever
(seen by T9 with MKL, whose first count query loads MKL for milliseconds). With
OpenBLAS only, the window is short, but the test forks 20 ms after starting 3
fitting threads, i.e. during their first calls; under valgrind / ASan, which
slow these first calls down, it is the case of the CI's child 0. T9's
`initial_mkl_threads()` and `touch_mkl_entry_points()` belong to T5's MKL code
and are not on this branch: they come with T5 / T9.

Not ported: T9 section 2.2 (the child handler sets MKL's global count to 1): it
needs T5's MKL entry points (`MKL_Set_Num_Threads` via weak symbol / dlsym /
GetProcAddress, next to T5's `MKL_Get_Max_Threads` / `MKL_Set_Num_Threads_Local`);
left to T9.

### Fork test under valgrind and the sanitizers

- **Memcheck** (`--error-exitcode=1`): the child's exit status is valgrind's.
  Reproduced locally (Release, the CI's valgrind options): the parent has 0
  errors; every child reports 140–380 kB "definitely lost", all of them objects
  of the parent's fitting threads (the pool's `std::vector<std::thread>`,
  `run_task_pool` buffers): those threads do not exist in the child, so their
  memory is unreachable there. Not a leak of the library. Fix in the test: the
  child sends its verdict through a pipe (`write`, async-signal-safe) before
  `_exit`; the parent reads it (no byte: the child died or was killed), the
  exit status is only reported. After: the fork case and `[concurrent]` pass
  under valgrind with the CI options.
- **ASan** (GCC 13 locally, 3 runs: one random child per run) and **LSan**
  (GCC 11 on the CI, all children): gdb on the stuck ASan child:
  `__sanitizer::Mutex::Lock` ← `SizeClassAllocator64::GetFromAllocator` ←
  `asan_malloc` ← `lkalloc::malloc` ← `arma::Mat::init_cold` (the child's fit).
  The sanitizer's allocator lock was held by another parent thread at
  `fork()`: the runtimes of these versions do not lock their allocator around
  `fork()` (glibc's malloc does), and POSIX only guarantees async-signal-safe
  functions in the child of a multithreaded process. Outside libKriging; no
  lock of the library is involved. Change: with ASan / LSan
  (`LK_TEST_FORK_NO_CHILD_ALLOC`, set by `tests/CMakeLists.txt` for
  `SANITIZE=ADDRESS / LEAK`, and by `__SANITIZE_ADDRESS__` /
  `__has_feature`), the child checks its BLAS count and reports, without
  allocating (no fit); the parent side is unchanged (fork handlers run while 3
  threads fit, count restored, no deadlock in the parent). The child's fit is
  still run in every other build (Release / Debug CI jobs, Memcheck,
  Coverage). After: ASan 5 / 5 runs, LSan (GCC 13) pass.

### Local checks after the fixes

Release, OpenBLAS 0.3.26: `ctest -j2` 231 / 231; `ConcurrentCallsTest` 3 / 3
runs (budget 1: 0 difference; full budget: 4e-14). Under valgrind (CI options):
fork case and `[concurrent]` pass. ASan (Debug, GCC 13): 5 / 5 runs; LSan:
pass. TSan: 0 report in 3 runs, 0 with the CI environment.

## Open items

- MKL / Accelerate: their thread counts are not driven (T5); with MKL,
  `mkl_set_num_threads_local` would give a per-thread count and remove the
  "smallest request" coupling between concurrent calls.
- Armadillo's RNG is global on MinGW / macOS GCC (Armadillo disables its
  `thread_local` there): concurrent WarpKriging fits or subset-of-data
  selections race on it on these platforms (R Windows binding).
- Full-budget tolerance of `ConcurrentCallsTest`: 1e-3 provisional (T9's
  value), decision pending with the user.
- MKL count in the fork child (T9 section 2.2): with T5 / T9.
- Two threads on the same model object: not supported (documented).
- `simulate(seed)` changes the seed of the next fit's starts in the same
  thread (pre-existing coupling, kept).
