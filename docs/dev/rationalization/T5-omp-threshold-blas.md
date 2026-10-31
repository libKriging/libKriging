# T5. OpenMP threshold and BLAS backends — work log

Task specification: `docs/dev/Rationalization.md`, section T5.

## Status

Done, apart from the MKL run-time check: implemented but not executed against a real MKL
(see "MKL" below). PR #445.

Machine for every measurement below: Intel Xeon @ 2.80 GHz, 4 cores / 4 threads (VM, cgroup v1
hybrid), Ubuntu 24.04, GCC 13, OpenBLAS 0.3.26 pthreads (`libopenblas-pthread-dev`), libgomp,
Release build. Timings on this VM are noisy (bimodal at n = 200: 6–16 ms for the same binary).

## 1. Threshold-dependent results (finding 1.3)

### Cause

The only OpenMP reduction whose order depended on `kOmpMinRows` was the LL theta gradient
(`KrigingImpl::compute_ll_grad_theta_vecs`): from `kOmpMinRows` on it summed per-row partial
sums then the rows in order; below it (and without OpenMP / under TSan) it added every pair to
the total directly. The other loops gated by `kOmpMinRows` (`cholCov`, `update_cholCov`) and the
`n >= 200` / `n_o × n_n >= 40000` loops (`compute_dX`, `covMat_*`, `R_on`) write independent
entries: no order dependence.

Measured before the change (`base`: Linux, threshold 200; `base800`: the same sources built with
`-DLK_OMP_MIN_ROWS=800`, i.e. what macOS / Windows run), d = 3, `y = Σ sin(6 x)`, gradient at
theta = (0.3, 0.5, 0.7) and single-start `BFGS` fit:

| n | kernel | gradient, max relative gap | fitted theta, max relative gap | LL gap |
|---|---|---|---|---|
| 150 | both | 0 (both sequential) | 0 | 0 |
| 300 | gauss | 3.9e-9 | 6.5e-8 | 1.9e-6 |
| 300 | matern5_2 | 6.9e-14 | 1.9e-4 | 2.2e-5 |
| 500 | gauss | 3.4e-8 | 3.6e-7 | 1.2e-5 |
| 500 | matern5_2 | 1.3e-12 | 3.7e-4 | 2.5e-5 |
| 700 | gauss | 5.6e-8 | 1.3e-8 | 5.9e-6 |
| 700 | matern5_2 | 1.9e-12 | 3.4e-5 | 1.5e-5 |

So for 200 <= n < 800 the fitted model depended on the OS (the gauss gradient differs from the
8th–9th digit: ill-conditioned R, the trace term cancels).

### Change

`compute_ll_grad_theta_vecs` now uses the per-row partial sums on every path: one loop,
`#pragma omp parallel for ... if (n >= kOmpMinRows)`, the same code without OpenMP / under TSan.
Rows are summed in index order after the loop. Memory: 2·d·n doubles (as before above the
threshold). The function body is otherwise unchanged (T1 will replace it; the T1 rewrite should
keep one summation order for every thread count and threshold, which its spec already requires).

`kOmpMinRows` stays 200 on Linux / 800 elsewhere: it is now a performance setting only.

### Result: identical on every path

Same probe, d = 3, n = 150, 300, 500, 700, gauss and matern5_2; outputs printed with `%a`
(exact) and compared byte for byte:

| build / setting | output vs default build (threshold 200) |
|---|---|
| `-DLK_OMP_MIN_ROWS=800` (macOS / Windows default) | identical |
| `-DLK_OMP_MIN_ROWS=1000000000` (never parallel) | identical |
| `-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON` (no OpenMP: same path as TSan) | identical |
| thread budget 100 % / 200 % (4 OpenMP threads) | identical |
| thread budget 25 % (1 thread: OpenMP **and OpenBLAS** at 1) | gradient at fixed theta identical, fitted theta differs (≤ 2e-4 relative, matern5_2 n = 300) |

The last row is the BLAS thread count (OpenBLAS 1 vs 4 threads; `OPENBLAS_NUM_THREADS=1` with
a 100 % budget gives the same fit as the 25 % budget): the known, documented dependence
(`parallel.md` 4, FAQ), outside T5's scope.

For n < 200 the gradient now sums in the per-row order instead of the pair order: last-bit
changes of the gradient (n = 150: 5e-10 relative for gauss, 5e-14 for matern5_2) and of the
fitted theta (n = 150: 6e-9 gauss, 1.1e-4 matern5_2). The full `ctest` passes unchanged (226 / 226).

## 2. Threshold: timings (bench/bench-threads)

`bench-threads <n> 4 25 matern5_2`, budget 4 (no environment variable), medians of the per-run
medians over 3 runs (`timing2`) and 2 runs (`timing`); cov = covariance loop of `cholCov`,
llgrad = one LL + gradient evaluation, ms. `base` = before the change (threshold 200), `new` =
after (threshold 200), `t800` = after with threshold 800, `seq` = after, never parallel.

| n | base cov / llgrad | new cov / llgrad | t800 cov / llgrad | seq cov / llgrad |
|---|---|---|---|---|
| 50 | 0.25 / 0.50 | 0.25 / 0.50 | 0.23 / 0.46 | 0.22 / 0.48 |
| 100 | 0.98 / 1.73 | 0.92 / 1.73 | 0.94 / 1.67 | 0.94 / 1.69 |
| 150 | 1.76 / 3.78 | 1.79 / 3.85 | 1.81 / 3.35 | 1.79 / 3.50 |
| 200 | 5.79 / 9.71 | 6.54 / 9–20 (bimodal) | 3.32 / 6.14 | 3.12 / 6.29 |
| 250 | 6.77 / 11.78 | 6.43 / 10.55 | | 4.95 / 10.68 |
| 300 | 7.57 / 14.60 | 7.67 / 14.02 | 6.80 / 14.49 | 6.92 / 14.63 |
| 400 | 7.61 / 20.38 | 7.40 / 21.07 | | 11.56 / 28.47 |
| 500 | 10.26 / 39.20 | 8.37 / 32.43 | 18.47 / 48.49 | 18.16 / 44.16 |
| 800 | 13.73 / 83.11 | 17.21 / 88.83 | 15.56 / 72.88 | 48.15 / 131.81 |
| 1200 | 28.01 / 207.63 | 31.20 / 200.94 | 29.33 / 201.27 | 99.81 / 322.85 |

- Below the threshold the single per-row path costs nothing measurable (n = 50–150: `new` =
  `base` within noise; the `if (false)` parallel region is negligible next to the O(n²) loop).
- At n = 200 the parallel loops lose on this 4-core VM (llgrad 9–20 ms vs 6 ms sequential;
  repeated runs of the same binary are bimodal, base and new alike); break-even ~250–300, gain
  ×1.4 at 400, ×1.4–1.5 at 500, ×1.5–1.8 at 800–1200. On the i9-10900K (10 cores) the gain at
  n = 200 was ×3 (`bench/thread-budget/RESULTS.md`). The best threshold depends on the core
  count: 200 is kept on Linux (a fraction of a millisecond lost per evaluation on small machines
  at n = 200–250, against ×3 on 10 cores).
- Decision: keep the per-OS defaults, since results no longer depend on them; one threshold
  everywhere would change macOS / Windows timings without measurement there (#416, #417).

## 3. BLAS backends

### MKL: implemented, per thread

- `MKL_Get_Max_Threads` / `MKL_Set_Num_Threads_Local` (C entry points, int by value), weak
  symbols on ELF (as the OpenBLAS ones: they bind to the MKL that the library's BLAS calls go
  to, linked, `libblas.so.3` alternative or preloaded), `dlsym(RTLD_DEFAULT)` on macOS (no
  undefined weak symbols without a definition at link time on Mach-O), `GetProcAddress` on
  `mkl_rt.2.dll` / `mkl_rt.dll` / `mkl_rt.1.dll` on Windows.
- Pitfall found while checking the exports of `libmkl_rt.so.2` (oneMKL 2024.2.2 wheel):
  the lower-case `mkl_set_num_threads_local` has the same address as `MKL_SET_NUM_THREADS_LOCAL`,
  the Fortran entry point (argument by reference); `mkl_service.h` maps the lower-case C name
  to `MKL_Set_Num_Threads_Local` by a macro. Calling the lower-case symbol with an int by value
  would pass a garbage pointer. The code uses the C names only.
- Rule (`mkl_threads_for(share)`): as OpenBLAS, the count is lowered to the share only when MKL
  was loaded with more (`MKL_NUM_THREADS` below the share is kept; MKL's default is the physical
  cores, so in practice it acts in multistart workers and with a budget below 100 %).
- Where: `ScopedShare` (each pool worker, thread-local: no race between workers, unlike the
  process-global OpenBLAS count, cf. T6) and `BudgetScope` (the calling thread of each public
  method). Restored with the previous thread-local value (0 = back to the global count).
- Not run here: executing the downloaded MKL binaries was not permitted in the session, and no
  MKL is in CI. To verify (Linux, OpenBLAS build, MKL preloaded so that `dgemm_` etc. and the
  `MKL_*` weak symbols bind to it):

  ```
  pip download --no-deps mkl==2024.2.2 && unzip mkl-*.whl
  export MKL_THREADING_LAYER=GNU LD_LIBRARY_PATH=$PWD/mkl-2024.2.2.data/data/lib
  MKL_VERBOSE=1 LD_PRELOAD=$LD_LIBRARY_PATH/libmkl_rt.so.2 <program fitting "BFGS4">
  ```

  Expected: `MKL_VERBOSE` lines with `NThr:1` for the calls of the 4 workers on 4 cores
  (before: `NThr:4`, i.e. 16 threads on 4 cores), and the caller's `MKL_Get_Max_Threads()`
  unchanged after the fit.

### Accelerate, FlexiBLAS, BLIS: documented

- Accelerate has no per-thread count (`BLASSetThreading`, macOS 15+, is process-wide and only
  single / multi); `VECLIB_MAXIMUM_THREADS` is read at load. Documented: set
  `VECLIB_MAXIMUM_THREADS=1` for multistart fits (FAQ, `parallel.md` 5).
- FlexiBLAS / BLIS: `flexiblas_set_num_threads` / `bli_thread_set_num_threads` are
  process-global, as OpenBLAS; not implemented (no test machine), documented.

## 4. cgroup v1 quotas

`physical_cores()` now reads, for the v2 hierarchy and for the v1 hierarchy holding the `cpu`
controller (both on a hybrid system), the mount point and root from `/proc/self/mountinfo`, the
cgroup path from `/proc/self/cgroup`, and takes the smallest quota of the cgroup and of its
ancestors visible under the mount point (v2 `cpu.max`, v1 `cpu.cfs_quota_us` /
`cpu.cfs_period_us`, ceil). Before: v2 only, and only the process's cgroup and the root.

Checked on this machine (cgroup v1 `cpu` at `/sys/fs/cgroup/cpu`), `Optim::get_num_threads()`
at 100 % in a child cgroup:

| quota | before | after |
|---|---|---|
| none | 4 | 4 |
| 2 CPUs (200000 / 100000) | 4 | 2 |
| parent 2 CPUs, child unlimited | 4 | 2 |
| 1.5 CPUs | 4 | 2 |
| parent 1.5, child 0.5 CPUs | 4 | 1 |

## 5. Checks

- `ctest` (Release, OpenBLAS pthreads, 4 cores): 226 / 226 passed.
- TSan build (`-DSANITIZE=THREAD`, RelWithDebInfo, GCC 13), CI settings
  (`OMP_NUM_THREADS=1`, `.tsan-suppressions`, `tools/linux-macos/test.sh`): `BlasThreadsTest`
  (9 test cases, including the new MKL one, skipped without MKL) and `KrigingTest` pass, 0
  reports. Without `OMP_NUM_THREADS=1` (budget 4, OpenBLAS at 4 threads) TSan reports races
  between the uninstrumented OpenBLAS threads and Armadillo (`syrk`, `chol_block`); no frame in
  the code changed here; the CI job runs with 1 thread for that reason.
- New test `BlasThreadsTest` "A fit leaves the caller's MKL thread count unchanged" (warns and
  returns without MKL, i.e. in CI).
- `clang-format` (18) on the changed C++ files.

## Decisions

1. One summation order for the LL gradient on every path instead of one threshold everywhere:
   it also covers the builds without OpenMP and under TSan, and leaves the threshold free to
   be tuned per machine.
2. `kOmpMinRows` stays 200 (Linux) / 800 (elsewhere), a performance setting.
3. MKL driven per thread; Accelerate, FlexiBLAS, BLIS documented as not driven.
