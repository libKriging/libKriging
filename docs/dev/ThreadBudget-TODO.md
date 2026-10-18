# Shared BLAS / OpenMP thread budget — TODO

Branch `perf/thread-budget`, based on PR #414 (`ccr-d554b384-m5sb5t`). Meant to be
run on a dedicated machine with ~20 cores. Each step lists its commands, what to
record and when it is done. Tick the boxes in this file as work progresses and
commit the results (CSV + summary) with it.

## Context (state after PR #414)

- Parallel levels in the library:
  1. multistart thread pool (`run_task_pool` in `src/lib/WarpKriging.cpp`;
     WarpKriging / MLPKriging only — the Kriging multistart is sequential on purpose,
     see `Kriging::fit_impl`);
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

- [ ] Record the machine: `lscpu`, `nproc`, `free -g`, OS, compiler, OpenBLAS version
      (`ls -l /usr/lib/x86_64-linux-gnu/libopenblas*`), into `bench/thread-budget/machine.txt`.
- [ ] Install the BLAS variants (Debian/Ubuntu):
      `sudo apt install libopenblas0-pthread libopenblas0-openmp libopenblas-dev liblapack-dev libblas-dev`
- [ ] Build (Release, tests + benches):
      ```
      git submodule update --init --recursive
      cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
            -DENABLE_OCTAVE_BINDING=OFF -DENABLE_MATLAB_BINDING=OFF \
            -DENABLE_PYTHON_BINDING=OFF -DENABLE_R_BINDING=OFF
      ninja -C build && (cd build && ctest -j8 --timeout 900)
      ```
      Done when: all tests pass (`KrigingWarmRestartTest` is a timing test, known to be
      flaky under load; re-run it alone before reporting it).
- [ ] Check the BLAS in use: `ldd build/src/lib/libKriging.so | grep -i blas`.
      Switch variant with `sudo update-alternatives --config libblas.so.3-x86_64-linux-gnu`
      (and `liblapack.so.3-...`); `bench-threads` prints `blas_kind`
      (0 sequential, 1 pthreads, 2 OpenMP, -1 not OpenBLAS) in every row.

## Step 1 — measure (go / no-go for steps 4–5)

Tool: `bench/bench-threads` (one CSV row: phase timings of one exact log-likelihood
evaluation under the environment's thread counts) and `bench/bench-threads.sh`
(sweep). Columns: `t_dX_ms` (`compute_dX`), `t_cholCov_ms` (covariance + Cholesky),
`t_chol_ms` (Cholesky alone), `t_cov_ms` (difference = covariance loop),
`t_ll_ms`, `t_llgrad_ms` (`logLikelihoodFun` without / with gradient).

- [ ] Baseline sweep, OpenBLAS pthreads, `kOmpMinRows` as is:
      `T_LIST="1 2 4 10 20" bench/bench-threads.sh build bench/thread-budget/pthreads.csv`
- [ ] Same with the loops forced on from n = 200: temporarily set
      `kOmpMinRows = 200` in `src/lib/OmpLoops.hpp`, rebuild, sweep into
      `bench/thread-budget/pthreads-min200.csv`, then restore 800.
- [ ] Same two sweeps with OpenBLAS OpenMP (`openmp.csv`, `openmp-min200.csv`) and the
      reference BLAS (`reference.csv`; BLAS thread count irrelevant, use `T_LIST` for OpenMP only).
- [ ] Variant (a) "BLAS to 1 thread around OpenMP loops": add an opt-in
      (`LK_BENCH_BLAS_HANDOFF=1` read once) in `cholCov` / `compute_ll_grad_theta_vecs`
      calling `detail::set_blas_threads(1)` before the loop and `restore_blas_threads`
      after; sweep with `kOmpMinRows = 200` into `pthreads-handoff.csv`. Measure
      whether the spinning stops immediately (compare with `OPENBLAS_NUM_THREADS=1`).
- [ ] Variant (b): `OPENBLAS_THREAD_TIMEOUT=4` (minimum spin, read at OpenBLAS init),
      `kOmpMinRows = 200`, into `pthreads-timeout.csv`.
- [ ] Variant (c): OpenMP team reduced to `ncpu - blas_threads` (or `ncpu / 2`) with
      `kOmpMinRows = 200`, into `pthreads-split.csv`.
- [ ] Full fits: `bench-fit-profile` / `bench-kriging` with `BFGS` and `BFGS10`,
      n ∈ {500, 2000}, default environment, before / after each retained variant.
- [ ] Write `bench/thread-budget/RESULTS.md`: for each BLAS variant, the best
      (BLAS, OpenMP) split per n, the break-even n of the OpenMP loops, and the
      speed-up of each variant vs baseline (median of ≥ 7 reps; note the spread).

Decision gate (write it in RESULTS.md):
- If no variant (a)–(c) gains > 10 % over the baseline for 200 ≤ n ≤ 1000 with
  OpenBLAS pthreads: drop step 4, keep steps 2, 3, 5, 6 (API unification only) and
  re-tune `kOmpMinRows` from the measured break-even on 20 cores.
- Otherwise: implement the best variant in step 4.

## Step 2 — single source of truth: `detail::ThreadBudget` (private)

- [ ] Rename `src/lib/BlasThreads.hpp` → `src/lib/ThreadBudget.hpp`; keep
      `BlasThreadGuard` (restore-on-exit) and the `pthread_atfork` handlers unchanged.
- [ ] Add:
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
      `total()` resolution: API value (step 3) > `LK_NUM_THREADS` > explicit
      `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` (never exceeded) >
      `std::thread::hardware_concurrency()`.
- [ ] All OpenMP loops use `num_threads(detail::ThreadBudget::omp_threads())`;
      remove the three copies of `get_optimal_threads(2)`; remove the
      `omp_set_num_threads` call of the pool workers (`ScopedShare` instead).
- [ ] Replace `kOmpMinRows` by a threshold from step 1 (keep the constant if the
      decision gate says so).
- [ ] `run_task_pool`: `ScopedShare(total / pool, total / pool)` per worker; the BLAS
      count stays process-global (set once by the guard, as today).
- Done when: `ctest` passes; `grep -rn "get_optimal_threads\|omp_set_num_threads" src/lib`
  only finds `ThreadBudget.hpp` / the guard and the atfork handlers.

## Step 3 — public API and bindings

- [ ] `Optim::set_num_threads(int)` / `Optim::get_num_threads()` (0 = automatic) in
      `src/lib/include/libKriging/Optim.hpp` + `Optim.cpp`; `set_thread_pool_size`
      clamped by `total`.
- [ ] Expose in the bindings next to `set_thread_pool_size`:
      `bindings/Python/pylibkriging/src/_pylibkriging/pylibkriging.cpp`,
      `bindings/R/rlibkriging/src/optim.cpp` (+ R wrapper / NAMESPACE / Rd),
      `bindings/Octave/mlibkriging/Optim_binding.cpp` + `Optim.m` (Julia does not expose
      `set_thread_pool_size` today: add both or neither).
- [ ] Binding tests next to the existing `thread_pool_size` ones
      (`test_new_features.py`, `test-new-features.R`, `test_new_features.m`).

## Step 4 — BLAS / OpenMP arbitration (only if the step 1 gate says so)

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

- [ ] `docs/dev/FAQ.md`: thread settings (`Optim::set_num_threads`, `LK_NUM_THREADS`,
      BLAS variants, nested-parallelism rule), reproducibility paragraph updated.
- [ ] Bindings READMEs (R: `parallel::mclapply` → one thread per worker advice).
- [ ] `CHANGELOG.md` (Unreleased).

## Delivery

- [ ] One PR per step 1 (results only), 2, 3, 4 (if any), 5–6 may ride with 2–3.
      Each PR: commands run, CSV excerpts, `ctest` result, and what was not verified
      (MSVC, MKL, macOS).

## Open questions (to decide before step 3)

1. Default `total`: all cores (current behaviour) or fewer? With R `parallel` /
   Python `multiprocessing` every process takes all cores; the library cannot detect it.
2. Add `LK_NUM_THREADS`, or rely on the existing environment variables only?
3. Revisit the pthreads OpenBLAS choice of the R binary (#325 / #387)? Out of scope,
   but the only change that removes the contention at the root.
