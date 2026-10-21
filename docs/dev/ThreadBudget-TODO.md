# Shared BLAS / OpenMP thread budget — TODO

Branch `perf/thread-budget`, based on PR #414 (`ccr-d554b384-m5sb5t`). Meant to be
run on a dedicated machine with ~20 cores. Each step lists its commands, what to
record and when it is done. Tick the boxes in this file as work progresses and
commit the results (CSV + summary) with it.

## Hand-off state (2026-10-19, second hand-off)

Read first: `docs/dev/parallel.md` (what is implemented and why, measurements,
decisions) and `docs/dev/CholJitter-TODO.md` (work in progress: fixed Cholesky
jitter, option (a)). Steps 0–2 done, step 4 dropped, Kriging multistart moved to
the thread pool; remaining here: steps 3, 5, 6 and the #414 Octave item.
Local commits only (no push: no GitHub Actions run until told otherwise);
PR #415 now targets `master` and includes #414 (closed).

## Hand-off state (2026-10-18)

- PR #414 head `fb8315a`; its CI was still running at hand-off. Open item on it:
  `build (Octave Linux, ubuntu-22.04, ...)` failed on d7d5ffc (wrong gradients,
  failed Cholesky, also for n < 200) on an Intel Xeon 6973P runner, and passed on
  52a7288 on an AMD EPYC 7763. That job links Ubuntu 22.04's OpenBLAS 0.3.20
  (pthreads, `DYNAMIC_ARCH`). Not reproduced locally (GCC 11, OpenBLAS 0.3.20
  `TARGET=COOPERLAKE` and 0.3.26, AVX-512 Xeon). Working hypothesis: OpenBLAS
  0.3.20 kernel selection on that CPU. See the comment on PR #414; check the
  Octave Linux job of the current head and its `lscpu` lines first.
  Local only (see Decisions): try OpenBLAS 0.3.20 `DYNAMIC_ARCH` here.
- [ ] Resolve the item above before merging #415 (#414 is now part of it). Likely
      the same phenomenon as docs/dev/parallel.md 8.6 (fits of smooth kernels
      sensitive to rounding in the near-singular region): check it once the
      fixed jitter (CholJitter-TODO.md) is in.

## Context (state after PR #414)

- Parallel levels in the library:
  1. multistart thread pool (`detail::run_task_pool`, `src/lib/TaskPool.cpp`):
     WarpKriging / MLPKriging multistart and calibration folds, and since
     2026-10-18 the Kriging multistart too (all noise models; it was sequential,
     see "Decisions" 4); no nested pools (a pool started from a worker runs inline);
  2. OpenMP loops (`src/lib/LinearAlgebra.cpp`: `compute_dX`, `cholCov`,
     `update_cholCov`, `covMat_*`; `src/lib/KrigingImpl.cpp`: predict `R_on`,
     `compute_ll_grad_theta_vecs`; `Kriging.cpp`: Vecchia neighbour searches;
     `NestedKriging.cpp`: pair loop);
  3. the BLAS (OpenBLAS / MKL / Accelerate / reference).
- Thread control today is scattered: `get_optimal_threads(2)` (copied in
  `LinearAlgebra.cpp`, `Kriging.cpp`, `KrigingImpl.cpp`), `Optim::thread_pool_size`,
  `detail::BlasThreadGuard` (`src/lib/BlasThreads.hpp`, restores counts on exit),
  `omp_set_num_threads` in the pool workers, `pthread_atfork` handlers (`Kriging.cpp`).
- `src/lib/OmpLoops.hpp`: `LK_OMP_LOOPS` (OpenMP loops compiled out under TSan) and
  `detail::kOmpMinRows = 800`, the threshold of the covariance / gradient loops.
- Problem measured on 4 cores (Xeon 2.1 GHz, OpenBLAS 0.3 pthreads, d = 4): with
  OpenBLAS and OpenMP both at 4 threads, an OpenMP loop is slower than one thread
  for n < ~700, because the OpenBLAS threads keep spinning after each BLAS call and
  compete with the OpenMP team (log-likelihood + gradient at n = 200: 3.9 ms serial,
  17 ms with the loops parallel). `compute_dX` (pre-existing, threshold n = 200)
  shows the same: 0.4 ms with OpenBLAS at 1 thread, ~8 ms at 4 threads.
- Constraints:
  - `openblas_set_num_threads` is process-global: no per-thread BLAS budget.
  - The R Linux binary bundles OpenBLAS **pthreads** on purpose (fork deadlock with
    the OpenMP build, #325 / #387): the problematic variant is the deployed one.
  - Results must stay independent of the OpenMP thread count (test
    `BlasThreadsTest`: "Objective, gradient and update do not depend on the OpenMP
    thread count").
  - Never change the host's settings permanently (OpenMP ICV of the calling thread,
    BLAS count): restore what was changed.

## Step 0 — machine setup

- [x] Record the machine: `lscpu`, `nproc`, `free -g`, OS, compiler, OpenBLAS version
      (`ls -l /usr/lib/x86_64-linux-gnu/libopenblas*`), into `bench/thread-budget/machine.txt`.
- [x] Install the BLAS variants (Debian/Ubuntu):
      `sudo apt install libopenblas0-pthread libopenblas0-openmp libopenblas-dev liblapack-dev libblas-dev`
- [x] Build (Release, tests + benches):
      ```
      git submodule update --init --recursive
      cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
            -DENABLE_OCTAVE_BINDING=OFF -DENABLE_MATLAB_BINDING=OFF \
            -DENABLE_PYTHON_BINDING=OFF -DENABLE_R_BINDING=OFF
      ninja -C build && (cd build && ctest -j8 --timeout 900)
      ```
      Done when: all tests pass (`KrigingWarmRestartTest` is a timing test, known to be
      flaky under load; re-run it alone before reporting it).
      2026-10-18, i9-10900K (10 cores / 20 threads, no AVX-512), build dir `build-tb`:
      216/218 with `-j8`; `KrigingWarmRestartTest` and "NuggetKriging multistart and
      parallel tests" failed under load and pass alone (the latter sets
      `OPENBLAS_NUM_THREADS` / `OMP_NUM_THREADS`: keep an eye on it in step 5).
- [x] Check the BLAS in use: `ldd build/src/lib/libKriging.so | grep -i blas`.
      Switch variant with `sudo update-alternatives --config libblas.so.3-x86_64-linux-gnu`
      (and `liblapack.so.3-...`); `bench-threads` prints `blas_kind`
      (0 sequential, 1 pthreads, 2 OpenMP, -1 not OpenBLAS) in every row.
      Here: OpenBLAS 0.3.26 pthreads (default) and reference BLAS installed;
      OpenMP OpenBLAS (`libopenblas0-openmp`) still to install.

## Step 1 — measure (go / no-go for steps 4–5)

Tool: `bench/bench-threads` (one CSV row: phase timings of one exact log-likelihood
evaluation under the environment's thread counts) and `bench/bench-threads.sh`
(sweep). Columns: `t_dX_ms` (`compute_dX`), `t_cholCov_ms` (covariance + Cholesky),
`t_chol_ms` (Cholesky alone), `t_cov_ms` (difference = covariance loop),
`t_ll_ms`, `t_llgrad_ms` (`logLikelihoodFun` without / with gradient).

- [x] Baseline sweep, OpenBLAS pthreads, `kOmpMinRows` as is:
      `T_LIST="1 2 4 10 20" bench/bench-threads.sh build bench/thread-budget/pthreads.csv`
- [x] Same with the loops forced on from n = 200: build with
      `-DCMAKE_CXX_FLAGS=-DLK_OMP_MIN_ROWS=200` (build dir `build-tb200`; the default
      stays 800), sweep `N_LIST="200 500 1000"` into `bench/thread-budget/pthreads-min200.csv`.
      2026-10-18, i9-10900K, `t_llgrad_ms`, d = 10 (d = 2 similar):

      | n | default 20/20, min 800 | best, min 800 | 10/10, min 200 | 20/20, min 200 |
      |---|---|---|---|---|
      | 200 | 6.4 | 4.9 | 1.5 | 41 |
      | 500 | 44.6 | 32.4 | 10.3 | 28 |
      | 1000 | 74 | 49.3 (10/10) | 48.6 | 76 |

      With BLAS and OpenMP both at the physical-core count (10), the loops are
      worth it from n = 200 on (×3 at n = 200 and 500); with either at 20 (logical
      CPUs) they collapse (spinning OpenBLAS threads + hyperthreads).
      BLAS variant switched without root: `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/openblas-openmp`
      (OpenMP OpenBLAS) or `.../blas:.../lapack` (reference).
- [x] Same two sweeps with OpenBLAS OpenMP (`openmp.csv`, `openmp-min200.csv`) and the
      reference BLAS (`reference.csv`; BLAS thread count irrelevant, use `T_LIST` for OpenMP only).
      State 2026-10-18 (bench stopped on request): `openmp.csv` and `openmp-min200.csv`
      complete (not analysed yet); `reference.csv` 47 / 50 rows (missing n = 3000,
      d = 10, OpenMP 4 / 10 / 20); `reference-min200.csv` not started. To resume:
      ```
      R=/usr/lib/x86_64-linux-gnu/blas:/usr/lib/x86_64-linux-gnu/lapack
      for to in 4 10 20; do LD_LIBRARY_PATH=$R OMP_NUM_THREADS=$to \
        build-tb/bench/bench-threads 3000 10 7 matern5_2 >> bench/thread-budget/reference.csv; done
      out=bench/thread-budget/reference-min200.csv
      LD_LIBRARY_PATH=$R build-tb200/bench/bench-threads 10 2 1 matern5_2 --header | head -1 > $out
      for n in 200 500 1000; do for d in 2 10; do for to in 1 2 4 10 20; do
        LD_LIBRARY_PATH=$R OMP_NUM_THREADS=$to build-tb200/bench/bench-threads $n $d 7 matern5_2 >> $out
      done; done; done
      ```
- [x] Variant (a) "BLAS to 1 thread around OpenMP loops": add an opt-in
      (`LK_BENCH_BLAS_HANDOFF=1` read once) in `cholCov` / `compute_ll_grad_theta_vecs`
      calling `detail::set_blas_threads(1)` before the loop and `restore_blas_threads`
      after; sweep with `kOmpMinRows = 200` into `pthreads-handoff.csv`. Measure
      whether the spinning stops immediately (compare with `OPENBLAS_NUM_THREADS=1`).
- [x] Variant (b): `OPENBLAS_THREAD_TIMEOUT=4` (minimum spin, read at OpenBLAS init),
      `kOmpMinRows = 200`, into `pthreads-timeout.csv`.
- [x] Variant (c): OpenMP team reduced to `ncpu - blas_threads` (or `ncpu / 2`) with
      `kOmpMinRows = 200`, into `pthreads-split.csv`.
- [x] Full fits: `bench-fit-profile` / `bench-kriging` with `BFGS` and `BFGS10`,
      n ∈ {500, 2000}, default environment, before / after each retained variant.
- [x] Write `bench/thread-budget/RESULTS.md`: for each BLAS variant, the best
      (BLAS, OpenMP) split per n, the break-even n of the OpenMP loops, and the
      speed-up of each variant vs baseline (median of ≥ 7 reps; note the spread).

Step 1 done 2026-10-18 (`bench/thread-budget/RESULTS.md`). Linux only: macOS
(Accelerate, libomp, P/E cores) in #416, Windows (OpenBLAS DLL lookup, libomp,
OpenMP hangs) in #417; `kOmpMinRows` stays 800 on both until measured. Outcome: step 4
dropped; the gain (full fits ×1.4–2.1) comes from capping BLAS and OpenMP at the
physical cores, which lets `kOmpMinRows` go down to 200 — the two must ship
together (threshold 200 without the budget: up to ×13.6 slower).

Decision gate (write it in RESULTS.md):
- If no variant (a)–(c) gains > 10 % over the baseline for 200 ≤ n ≤ 1000 with
  OpenBLAS pthreads: drop step 4, keep steps 2, 3, 5, 6 (API unification only) and
  re-tune `kOmpMinRows` from the measured break-even on 20 cores.
- Otherwise: implement the best variant in step 4.

## Step 2 — single source of truth: `detail::ThreadBudget` (private)

Done 2026-10-18 (differs from the plan below where noted):
- `src/lib/ThreadBudget.hpp/.cpp` (new file; `BlasThreads.hpp` kept for the
  fork handlers and the BLAS get/set): `physical_cores()` (Linux sysfs topology
  of the affinity mask + cgroup v2 quota; macOS `hw.physicalcpu`; Windows
  `GetLogicalProcessorInformationEx` — the last two not compiled here),
  `budget_total()`, `omp_threads()`, `ScopedShare`, `ScopedBlasThreads`,
  `BudgetScope` (OpenBLAS at the budget for each public fit / predict /
  simulate / update / update_simulate of Kriging, WarpKriging, NestedKriging;
  no-op in pool workers).
- `Optim::set_num_threads_percent` / `get_num_threads_percent` /
  `get_num_threads`, `LK_NUM_THREADS_PERCENT` (C++ only; bindings: step 3).
- Every OpenMP loop: `num_threads(omp_threads())`; `get_optimal_threads(2)` ×3
  and the cap of 2 removed; the pool workers no longer call
  `omp_set_num_threads` (no ICV of any thread is changed, except by the fork
  handlers).
- `kOmpMinRows`: 200 on Linux, 800 elsewhere (#416, #417).
- Pool (`TaskPool.cpp`): pool = min(starts, `thread_pool_size` or budget),
  share = budget / pool for BLAS (process-global, restored) and OpenMP (per
  worker). The remainder of budget / pool is not distributed (budget 5, 4 starts:
  4 × 1).
- ctest 218/218.

- [x] Rename `src/lib/BlasThreads.hpp` → `src/lib/ThreadBudget.hpp`; keep
      `BlasThreadGuard` (restore-on-exit) and the `pthread_atfork` handlers unchanged.
- [x] Add:
      ```cpp
      namespace libKriging::detail {
      struct ThreadBudget {
        static int total();         // see resolution order below
        static int omp_threads();   // share of the current thread (thread_local, default total())
        static int blas_threads();  // process-global BLAS count
        static int blas_kind();     // openblas_get_parallel(): 0/1/2, -1 unknown
      };
      struct ScopedShare {          // set by the multistart pool in each worker
        ScopedShare(int omp, int blas); ~ScopedShare();
      };
      }
      ```
      `total()` resolution: API value (step 3) > `LK_NUM_THREADS_PERCENT` > explicit
      `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` (never exceeded) >
      number of physical cores (see Decisions), `hardware_concurrency()` as fallback.
- [x] All OpenMP loops use `num_threads(detail::ThreadBudget::omp_threads())`;
      remove the three copies of `get_optimal_threads(2)`; remove the
      `omp_set_num_threads` call of the pool workers (`ScopedShare` instead).
- [x] Replace `kOmpMinRows` by a threshold from step 1 (keep the constant if the
      decision gate says so).
- [x] `run_task_pool`: `ScopedShare(total / pool, total / pool)` per worker; the BLAS
      count stays process-global (set once by the guard, as today).
- Done when: `ctest` passes; `grep -rn "get_optimal_threads\|omp_set_num_threads" src/lib`
  only finds `ThreadBudget.hpp` / the guard and the atfork handlers.

## Step 3 — public API and bindings

- [x] `Optim::set_num_threads_percent(double x)` (x <= 1: ratio, x > 1: percentage;
      0 = automatic = 100 %), `Optim::get_num_threads_percent()` and
      `Optim::get_num_threads()` (resolved `total`, read only) in
      `src/lib/include/libKriging/Optim.hpp` + `Optim.cpp`; `set_thread_pool_size`
      clamped by `total`.
- [x] Expose in the bindings next to `set_thread_pool_size` (2026-10-21; Julia:
      neither, as before; R unexported like the other `optim_*`):
      `bindings/Python/pylibkriging/src/_pylibkriging/pylibkriging.cpp`,
      `bindings/R/rlibkriging/src/optim.cpp` (+ R wrapper / NAMESPACE / Rd),
      `bindings/Octave/mlibkriging/Optim_binding.cpp` + `Optim.m` (Julia does not expose
      `set_thread_pool_size` today: add both or neither).
- [x] Binding tests next to the existing `thread_pool_size` ones
      (`test_new_features.py`, `test-new-features.R`, `test_new_features.m`).
      `test_new_features.m` was not registered in CMake (never ran) and failed on
      `Optim.set_log_level(2)`: the int setters of `Optim.m` now pass `int32`;
      registered as `Octave/NewFeatures`. Local: Python NewFeatures ok, Octave 17 / 18
      (`mLibKriging-Demo`: no display), R 697 tests 0 failed.

## Step 4 — BLAS / OpenMP arbitration — DROPPED (RESULTS.md: decision, sections 8–9)

- [ ] Implement the retained variant, keyed on `ThreadBudget::blas_kind()`:
      0 or 2 (sequential / OpenMP OpenBLAS) → nothing; 1 (pthreads) → the variant.
- [ ] If variant (a): make the handoff safe when several host threads call the library
      at once (the BLAS count is global): document it, or only hand off when the
      current thread is the only active caller.
- [ ] MKL / Accelerate: no code without a measurement; document the behaviour.

## Step 5 — tests

- [ ] Extend `tests/BlasThreadsTest.cpp`:
  - [ ] `pool × omp share ≤ total` and `pool × BLAS ≤ total` during a WarpKriging
        parallel multistart (record the shares seen by the workers);
  - [ ] an explicit `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` is never exceeded;
  - [ ] the calling thread's OpenMP ICV is unchanged after `fit` / `predict`;
  - [ ] objective, gradient and `update` bit-identical for `total` ∈ {1, 2, 4, 20}.
- [ ] TSan job: OpenMP loops stay compiled out (`LK_OMP_LOOPS`); fork-safety R test.
- [ ] Optional CI job running `bench-threads.sh` with a small grid, no threshold
      (runners are too noisy for a pass/fail on timings).

## Step 6 — documentation

- [ ] `docs/dev/FAQ.md`: thread settings (`Optim::set_num_threads_percent`, `LK_NUM_THREADS_PERCENT`,
      BLAS variants, nested-parallelism rule), reproducibility paragraph updated.
- [ ] Bindings READMEs (R: `parallel::mclapply` → one thread per worker advice).
- [ ] `CHANGELOG.md` (Unreleased).

## Delivery

- [ ] One PR per step 1 (results only), 2, 3, 4 (if any), 5–6 may ride with 2–3.
      Each PR: commands run, CSV excerpts, `ctest` result, and what was not verified
      (MSVC, MKL, macOS).

## Decisions (2026-10-18)

1. Default `total` = all the cores (100 %, current behaviour). Expose a way to set
   a **percentage** of the resources (only that, no absolute count), so that a
   host running several processes (R `parallel`, Python `multiprocessing`) can
   give each one a share.
   **100 % = the physical cores**, not the logical CPUs: on the i9-10900K
   (10 cores / 20 threads) OpenBLAS at 20 threads is slower than at 10 in every
   configuration measured (preliminary run, syncthing active; n = 1000, d = 10: 181 vs 143 ms with OpenMP at 1;
   default 20 / 20 = 92.6 ms vs 48.6 ms at 10 / 10). `std::thread::hardware_concurrency()`
   (used by `run_task_pool` today) counts logical CPUs: step 2 needs a
   physical-core count (Linux: `/sys/devices/system/cpu/cpu*/topology/core_cpus_list`
   or `sched_getaffinity` + topology, honouring the affinity mask / cgroup quota;
   macOS: `sysctl hw.physicalcpu`; Windows: `GetLogicalProcessorInformationEx`),
   with `hardware_concurrency()` as the fallback.
2. No new variable before an inventory: see "Thread settings inventory" below and
   the rationalisation proposal there.
3. The pthreads OpenBLAS of the R binary (#325 / #387) stays as is for now.
4. **The multistart goes first**: the starts are intrinsically parallel, so the
   budget is given to them before OpenMP and BLAS: pool = min(starts, pool size or
   budget), each worker gets budget / pool (≥ 1) for its BLAS and OpenMP calls.
   The Kriging multistart (all noise models) now uses the same pool as
   WarpKriging. It was sequential so that BFGS20 matched the best of 20 BFGS1
   bit for bit with MKL (secondary threads round differently); `KrigingTest`
   checks that invariant to 1e-3 relative, which still holds.
- Work locally only: no GitHub Actions run (no push of this branch, no
  `workflow_dispatch`) until told otherwise. The open item on #414 is to be
  reproduced locally, not through CI.

5. (2026-10-19) Pool workers stay on the physical cores; bindings do not set
   `OPENBLAS_NUM_THREADS` (start-up transient only, documented); #414 merged into
   #415 (base `master`); the fit restart robustness (parallel.md 8.6) is worked
   on in this branch. Reference: `docs/dev/parallel.md`.

## Thread settings inventory (state after #414)

Read by the library:

| Setting | Where | Default | Effect |
|---|---|---|---|
| `LK_THREAD_POOL_SIZE` / `Optim::set_thread_pool_size` | `Optim.cpp`, `run_task_pool` | 0 = `hardware_concurrency()` | multistart workers (Kriging, WarpKriging, MLPKriging), clamped to the number of starts |
| `LK_THREAD_START_DELAY_MS` / `Optim::set_thread_start_delay_ms` | `Optim.cpp`, `run_task_pool` | 10 | stagger of the first task of each worker (not a thread count) |

Implicit (hard-coded) in the library:

| Setting | Where | Effect |
|---|---|---|
| `get_optimal_threads(2)` (3 copies) | `LinearAlgebra.cpp` (`compute_dX`, `covMat_*`), `KrigingImpl.cpp` (predict `R_on` ×2), `Kriging.cpp` | caps those loops at **2** OpenMP threads whatever the machine |
| no `num_threads` | `LinearAlgebra.cpp` `cholCov` / `update_cholCov`, `KrigingImpl.cpp` gradient loop, `Kriging.cpp` Vecchia ×2, `NestedKriging.cpp` pair loop | use `omp_get_max_threads()` of the calling thread (all cores by default) |
| thresholds | `kOmpMinRows = 800` (cov / gradient), `n >= 200` (`compute_dX`, `covMat_*`), `work >= 40000` (predict, Vecchia, Nested) | serial below |
| pool share | `run_task_pool` | BLAS = `ncpu / pool` (process-global, via `BlasThreadGuard`), OpenMP = same value per worker (`omp_set_num_threads` in each worker) |
| `set_blas_threads(n)` | `Kriging.cpp` | sets **both** OpenBLAS and the calling thread's OpenMP ICV (restored by the guard) |
| `pthread_atfork` handlers | `Kriging.cpp` | BLAS / OpenMP to 1 thread around `fork()` |

Read by the runtimes, not by the library (only documented):
`OMP_NUM_THREADS`, `OMP_THREAD_LIMIT`, `GOMP_SPINCOUNT` (libgomp);
`OPENBLAS_NUM_THREADS` (falls back on `OMP_NUM_THREADS`), `OPENBLAS_THREAD_TIMEOUT`
(OpenBLAS); `MKL_NUM_THREADS`; `VECLIB_MAXIMUM_THREADS` (Accelerate). In-tree
uses: `docs/dev/FAQ.md` (reproducibility), `bindings/R/README.md`
(`OMP_THREAD_LIMIT=1 GOMP_SPINCOUNT=0 OPENBLAS_NUM_THREADS=1`), CI scripts
(`OMP_NUM_THREADS=1` on Windows and the notebooks), tests
(`NoiseKrigingTest`, `NuggetKrigingTest`, `threading_test.py`).

Rationalisation proposal (to validate before step 3):

- **One** new budget setting (decided 2026-10-18), a share of the physical cores:
  `Optim::set_num_threads_percent(double x)` and `LK_NUM_THREADS_PERCENT=x`, with
  x <= 1 read as a ratio (0.5) and x > 1 as a percentage (50); 0 or unset = 100 %.
  `total = max(1, round(share × physical cores))`. No absolute-count setter.
  `Optim::get_num_threads_percent()` returns the setting, `Optim::get_num_threads()`
  the resolved `total`.
- `total` = the setting above, never more than the counts in effect when the
  library is first used (`omp_get_max_threads()`, `openblas_get_num_threads()`),
  so that an explicit `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` is respected
  without the library reading them itself.
- `LK_THREAD_POOL_SIZE` kept (pool size inside `total`, clamped by it).
- `LK_THREAD_START_DELAY_MS` kept as a tuning knob, out of the budget.
- Removed: the three `get_optimal_threads(2)` (the cap of 2 becomes the budget
  share), the per-worker `omp_set_num_threads`, and the OpenMP part of
  `set_blas_threads` (OpenMP counts go through `num_threads(...)` only, so the
  host ICV is never touched).
- Thresholds (`kOmpMinRows`, `n >= 200`, `>= 40000`): one place (`OmpLoops.hpp`),
  values from step 1.
- Runtime variables stay the user's tools; documented in the FAQ, not read.
