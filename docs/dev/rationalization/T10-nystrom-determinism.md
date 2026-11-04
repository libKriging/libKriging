# T10. Intermittent Nystrom LOO equality — work log

Task specification: `docs/dev/Rationalization.md`, section T10.

## Status

Done (test changed, no library change). Draft PR #451.

## 1. The failure

`tests/KrigingApproxObjectivesTest.cpp:230` (ctest "approximate LOO / LMP fits are consistent
with the exact fits"), job "Octave macOS" (macOS-latest arm64, Release, AppleClang 21), run
41168746602 (job 124441923746), commit 74d4d7f (#448):

```
CHECK( ka.leaveOneOut(method) == ka.leaveOneOut(method.substr(0, 7)) )
with expansion:
  0.0000000158 == 0.0000000158
LOO / nystrom(80): theta = 3.5527 1.5385   exact theta = 3.4154 1.1300
```

The backend of that job, reported by `LinearAlgebra::check` (ctest 7 / 127 of the same log):
`Apple Accelerate (.../vecLib.framework/Versions/A/libBLAS.dylib)`. The other run of the same
commit (41168750296) passed.

## 2. Do both calls take the same path?

Yes (`src/lib/Kriging.cpp`, `Kriging::leaveOneOut(const std::string&)`):

- `"nystrom(80)"` and `"nystrom"` both parse to `ApproxMethod::Nystrom`; the rank only goes
  through `check_nystrom_rank`, which raises when it differs from the fitted k (80) and does
  nothing else. `"nystrom"` without k means "the fitted rank" here (the default k = 50 applies
  only to `fit`).
- Both then call `ensure_approx_sets(false, ...)` (no-op for Nystrom: the landmarks are those
  of the fit, selected once by `make_nystrom_landmarks`, deterministic pivoted Cholesky, no RNG)
  and `_leaveOneOutApprox(false, m_theta, nullptr, nullptr) * loo_scale2()`, a `const` member:
  no cache, no RNG, no state written.
- `build_nystrom_precision` / `NystromChol` / `approx_loo`: every buffer allocated with
  `arma::fill::none` is fully written (`R_ss` by `covMat_sym_X` off-diagonal + diagonal,
  `R_ns` by `covMat_rect`, `x` and `diag_inv` on S then on N = complement of S).
- OpenMP: at n = 150, k = 80 none of the library loops is parallel (`covMat_sym_X` needs
  n >= 200, `covMat_rect` n1·n2 >= 40000, here 80² and 150·80 = 12000), whatever
  `kOmpMinRows` (200 on Linux, 800 elsewhere); those loops have no reduction anyway (each entry
  written by one thread). Armadillo's own OpenMP is off (`ARMA_DONT_USE_OPENMP`), and its
  `accu` / `sum` are sequential and alignment-independent without `-ffast-math` (not used).

So the two calls run the same deterministic code on the same data: bit identity is expected
**as long as the BLAS / LAPACK is reproducible from one call to the next**. The only part
whose order of operations is not fixed by libKriging is the BLAS / LAPACK (`dgemm`, `dtrtrs`,
`dpotrf`, `dgeqrf`, ...).

## 3. Measurements (Linux, Xeon 2.1 GHz, 4 cores, OpenBLAS 0.3.26 pthreads, Release)

Scripts: `repro.cpp` (repeated calls, bitwise comparison, optional heap shifts between calls
to change the buffer alignments) and `sens.cpp` (Nystrom / exact / Vecchia LOO at the fitted
theta 3.5521374017423248, 1.53830560290489, printed to 17 digits); both are reproduced in the
appendix.

### 3.1 Repeated calls

| Setting | Calls | Bitwise differences |
|---|---|---|
| fit then `leaveOneOut("nystrom(80)")` / `("nystrom")` alternated | 2000 | 0 |
| same, a `malloc` of 8–512 bytes kept between calls (alignment of the buffers changes) | 2000 | 0 |
| `OPENBLAS_NUM_THREADS` 1, 2, 4 × `OMP_NUM_THREADS` 1, 4 × `LK_NUM_THREADS_PERCENT` 0, 50, heap shifts | 500 each (12 settings) | 0 |
| `ctest -R "...consistent with the exact fits" --repeat until-fail:200`, `OPENBLAS_NUM_THREADS` 1 and 4 (original exact check) | 2 × 200 | see 3.3 |

Within one process, at a fixed BLAS thread count, the criterion is bit-reproducible on Linux.

### 3.2 Sensitivity to the BLAS summation order

Same theta, `OPENBLAS_CORETYPE` × `OPENBLAS_NUM_THREADS` (each line one process):

| Kernel | Threads 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| auto (SkylakeX) | 1.5780633317199439e-08 | 1.5780626811749182e-08 | 1.5780626811750452e-08 | 1.5780626811749182e-08 |
| Haswell | 1.578063670984146e-08 | 1.5780638307942843e-08 | 1.578064149738918e-08 | 1.5780639830218392e-08 |
| Sandybridge | 1.5780622998189305e-08 | 1.5780633829611372e-08 | 1.5780633829611372e-08 | 1.5780633829611372e-08 |
| Nehalem | 1.578064151742728e-08 | 1.5780632165633706e-08 | 1.5780631078285448e-08 | 1.5780633173263382e-08 |
| Prescott | 1.5780631377942712e-08 | 1.5780633118069629e-08 | 1.5780632957059688e-08 | 1.5780632718040532e-08 |

- Nystrom LOO: spread 1.17e-6 relative over the 20 configurations (max − min over min); the
  thread count alone changes it (SkylakeX: 1 vs 2 threads, 4.1e-7; 2 vs 3 threads, 8e-14).
- Exact LOO at the same theta (same runs): 1.2125057e-08 to 1.2125112e-08, spread 4.5e-6:
  the sensitivity is that of the problem (LOO errors ~1e-4 for y ~ 1, long ranges, R close to
  singular, e_i = (Q y)_i / Q_ii from large cancelling terms), not of the Nystrom code.
- Vecchia LOO (30 neighbours): independent of the thread count, 1.1e-8 relative across kernels.
- Discrimination: at the same theta, `nystrom(79)` gives 1.6101e-08 (+2.0 %), `nystrom(81)`
  1.7630e-08 (+11.7 %), `nystrom(70)` 2.3564e-08 (+49 %), exact 1.2125e-08 (−23 %).

### 3.3 Memory checks

- valgrind memcheck (`--track-origins=yes`), the test case, `OPENBLAS_NUM_THREADS=1`: see
  section 6 (results).
- ASan + UBSan build (RelWithDebInfo) of `KrigingApproxObjectivesTest`: see section 6.
- MSan: not run (needs every dependency, OpenBLAS and libstdc++ included, instrumented);
  valgrind covers uninitialized reads.

## 4. Cause

Two calls of the same deterministic code differ only if the BLAS / LAPACK changes its order of
operations between them. OpenBLAS does it between thread counts (3.2) but keeps the count of
the process, so it never shows within a run. Accelerate's default threading
(`BLAS_THREADING_MULTI_THREADED`, Apple's `BLASSetThreading` documentation) lets the
framework decide per call whether and how to split a call over threads; Apple documents no
run-to-run reproducibility. A different split of a reduction gives a different rounding, which
this criterion amplifies to ~1e-7–1e-6 relative (3.2). This matches the observations: macOS /
Accelerate only, intermittent, the two values equal to the 3 digits printed by Catch.

Not reproduced bitwise on macOS: no macOS machine here, and GitHub Actions ends every run in
`startup_failure` since 2026-11-02. The attribution to Accelerate's per-call partition is an
inference from (a) the code reading of section 2, (b) the measured order sensitivity of 3.2 and
(c) the Accelerate threading model; it is not a measurement on Accelerate.

## 5. Decision

- No library change: the library code is deterministic (section 2, 3.1); making Accelerate
  reproducible would need `BLASSetThreading(BLAS_THREADING_SINGLE_THREADED)` (macOS 15+),
  a process-global setting with a performance cost, out of the scope of this task.
- The test compares the two values relatively, at 1e-5: 8.5 × the largest spread measured
  (1.17e-6), 2000 × below the nearest wrong criterion (k = 79, 2 %). Its purpose (the string
  returned by `method()` is accepted and evaluates the fitted criterion) is kept.
- Same class, not changed (no failure seen, better-conditioned values, but still bitwise checks
  of two BLAS-dependent evaluations): `KrigingNystromTest.cpp:172` and `:816`,
  `KrigingVecchiaTest.cpp:356` and `:390`, `KrigingApproxObjectivesTest.cpp:301-302`,
  `KrigingNormalizeScaleTest.cpp:812`. To relax the same way if one fails on Accelerate.

## 6. Results of the checks

(filled below)
