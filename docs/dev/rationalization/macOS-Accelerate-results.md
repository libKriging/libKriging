# macOS / Accelerate — results (issue #452)

Machine: Apple M4 (4 P + 6 E cores, 10 logical), macOS 15.7.4, Apple clang 17.0.0, cmake 4.4.0, libomp (Homebrew), Accelerate BLAS.
Branch under test: `origin/ccr-54a50343-4jupn9-t9-blas-backends-ci` (2983509f) unless stated; Q1/Q2 on `...-t10-nystrom-determinism` (e841a604) with the exact check of #451 restored locally.
**CI finding (checked in the logs of PR #455's run):** every macOS CI job (Debug, R, Octave, Julia, ...) prints "OpenMP not found - multistart optimization will run sequentially": the macOS CI has no OpenMP, so it never exercised libomp (no fork deadlock, no `kOmpMinRows` effect, no multistart threads). The Q4/Q5 "macos-15 3 cores" numbers of the issue were therefore sequential. Fix: `brew install libomp` + `-DOpenMP_ROOT=$(brew --prefix libomp)` in the macOS jobs (decision needed: it will also run the new fork tests under libomp, which #455 makes pass).

Release builds: `cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$(brew --prefix libomp) -DOpenMP_ROOT=$(brew --prefix libomp)` (bindings OFF).
Raw data: `macOS-q4-bench-threads.csv`, `macOS-q4-fit-profile.txt`, `macOS-q5-timing.txt`, `macOS-fork125-sample.txt`, `macOS-rng-seed407.txt`; progress log in `macOS-Accelerate-TODO-updated.md`.

**OpenMP is NOT found by default on macOS** (Homebrew libomp is keg-only): without `OpenMP_ROOT` the build is sequential ("multistart optimization will run sequentially"). Check what the CI does.

## Q1 / Q2 — bitwise checks (inconclusive: not reproduced)
| Test | Result |
|---|---|
| `consistent with the exact fits`, exact equality restored, `--repeat until-fail:500`, default threads | 500/500 pass |
| `repro.cpp`, 2000 alternated calls, with and without heap shifts | 0 bitwise differences (LOO = 1.5778344766038358e-08, theta = 3.5521543413903967, 1.538402235248628) |
| Q2: `--repeat until-fail:500` on the 5 test cases holding the 6 checks (`KrigingNystromTest.cpp:172` "Nystrom fit is a permanent light fit", `:816` "Nystrom noise-model save/load round-trip", `KrigingVecchiaTest.cpp:356`/`:390` "Vecchia fit keeps no exact factorization", `KrigingApproxObjectivesTest.cpp:301-302` "method argument checks", `KrigingNormalizeScaleTest.cpp:812` "normalize=false: accessors ... bit for bit") | **500/500 pass for each** (17, 19, 548, 8, 9 s) |

No failure, so the `VECLIB_MAXIMUM_THREADS=1` / `BLASSetThreading` variants were not run. Attribution to Accelerate not confirmed nor refuted on a 10-core M4 (the failure was seen on a 3-core runner). Proposal: keep the 1e-5 relative tolerance.

## Q3 — MLPKriging calibration (#407): reproduced, hypothesis refuted
Deterministic failure on T9: coverage 0.707 < 0.8 (Kriging Q2 0.8438, uncalibrated Q2 -0.6847, var_scale 5.66757), identical in Release (10/10), **Debug (3/3)** and Release with `-ffp-contract=off` (3/3). So neither Release rounding nor FMA contraction is the cause, and the test does fail in Debug here. T8 passes here, so the data are not the problem on this machine; the earlier hypothesis (random sequences differing between libc++ and libstdc++) is no longer needed to explain it; Linux comparison with `macOS-rng-seed407.txt` is optional. 

Branches (10 runs each, Release and Debug, same M4): 
| Branch | Release | Debug |
|---|---|---|
| T9 | fails 10/10 (cov 0.707) | fails 3/3 |
| `origin/master` (a866c440) | fails 10/10 (`MLPKrigingCalibrationTest.cpp:100`, same Kriging Q2 0.843773) | fails 10/10 |
| T8 (`...-t8-mlp-calibration-cost`, K = 3, ca7f5dd7) | **passes 10/10** | **passes 10/10** |

T8 `[calibration]` output, identical in Release and Debug: BFGS calibrated NLPD 5.2393 cov 0.96 var_scale 101.623; BFGS4 uncalibrated Q2 0.996307 cov 0.269, calibrated NLPD 1.18539 cov 0.993 var_scale 642.905. So the macOS failure is specific to the T9/master version of the test and is fixed by T8 here, regardless of build type. Proposal (not applied): merge T8 (K = 3); no test change needed for macOS.

## Q4 — `kOmpMinRows` (d = 4, 10 threads, median of 3 rounds x 9 reps; threshold 800 = default, 200 = `-DLK_OMP_MIN_ROWS=200`)
| n | LL 800 (ms) | LL 200 (ms) | LL+grad 800 | LL+grad 200 |
|---|---|---|---|---|
| 100 | 0.23 | 0.24 | 0.36 | 0.37 |
| 200 | 0.90 | 0.43 | 1.39 | 0.75 |
| 400 | 4.12 | 1.97 | 6.54 | 3.44 |
| 800 | 9.20 | 9.19 | 19.16 | 17.55 |
| 1500 | 29.57 | 29.52 | 72.05 | 72.50 |

Full fits (`bench-fit-profile n 4 optim`, ms, median of 3):
| n | BFGS 800 | BFGS 200 | BFGS4 800 | BFGS4 200 |
|---|---|---|---|---|
| 200 | 54 | 38 | 32 | 28 |
| 500 | 180 | 133 | 253 | 224 |
| 1000 | 554 | 555 | 1443 | 1435 |

Conclusion: parallel loops pay off from n = 200 on this Mac (x2 at n = 200-400, -30 % on a BFGS fit); nothing changes from 800. Break-even is below 200 (not measured; n = 100 is below both thresholds). Proposal: `LK_OMP_MIN_ROWS` 200 on macOS too (drop the `__linux__` condition in `src/lib/OmpLoops.hpp`), provided OpenMP is actually enabled in the macOS build.

## Q5 — multistart threading (`BlasBackendTest "[.timing]"`, n = 200/400/800, 3 rounds)
BFGS4 / BFGS wall ratio: n=200 0.54-0.57, n=400 0.44-0.46, n=800 0.58-0.61 in every configuration (default, `LK_NUM_THREADS_PERCENT=50`, `VECLIB_MAXIMUM_THREADS=1`); BFGS times differ by < 2 %. `BLASSetThreading(single)` (built into the test): multi/single ratio 0.97-1.02. No gain above ~2 % noise from any setting, so no oversubscription is visible on 10 cores. Proposal: remove `VECLIB_MAXIMUM_THREADS=1` advice from the FAQ / `parallel.md` for macOS (needs a many-core check ideally >10 cores). Not done: BFGS10, n = 1500, `/usr/bin/time -l` CPU times.

## Q6 — ConcurrentCallsTest
20/20 pass at default budget and at `LK_NUM_THREADS_PERCENT=50`. Per-quantity gaps are not printed by the test (not measured).

## Q7 — shared reference
`LK_EXPECT_BLAS=accelerate ./tests/BlasBackendTest`: all pass. Gaps to the reference: theta 7e-8..1e-4, LL 7e-10..2e-8; at fixed theta LL <= 7.5e-10, gradient <= 2.7e-8; BFGS4 pool 1 vs 4: 0.

## Full suite, Release (`ctest -j4`): 233/237 before the fixes below
- #145 MLPKriging calibration: see Q3 (fixed by T8).
- #125, #115, #230: one root cause for the first two, one for the third; fixed in `macOS-fix-fork-libomp.patch` (local commit 62df3170 on branch `fix/atfork-libomp-deadlock`, from T9; not pushed). **Not applied to the results branch** (maintainer decision).

### #125 and #115: libomp's own atfork handlers (deadlock)
libomp registers `pthread_atfork` handlers that take its internal locks (`initz`, `forkjoin`) in `prepare` and keep them until the parent/child handlers; the child handler resets the runtime. Registered after libKriging's (lazily, at the first parallel region), they run first in `prepare`; so `libkriging_atfork_prepare` -> `omp_get_max_threads` -> `__kmp_get_global_thread_id_reg` -> `__kmp_acquire_ticket_lock` spins on a lock the same thread already holds (self-deadlock), and threads exiting at the same time spin behind it. In the child, our handler's `omp_set_num_threads(1)` hangs the same way. Stand-alone repro, no libKriging: `macOS-atfork-libomp-repro.cpp` (`./a.out 1` = handler calling omp_get_max_threads in prepare: hangs; without handler: the child sees omp_get_max_threads = 10 because libomp re-initializes itself, i.e. the test expectation "child <= 1" does not hold for libomp). Evidence: `macOS-fork125-sample.txt`.
Fix tried: if the runtime is libomp (`dlsym(RTLD_DEFAULT, "__kmpc_fork_call")`, evaluated once when the handlers are registered), the handlers only touch OpenBLAS and never call omp_*; the child handler does `setenv("OMP_NUM_THREADS", "1")`, which libomp reads when it re-initializes after our handler (checked in the repro: child sees 1, parent keeps 4). Results: #125 passes in 0.4 s (was a 1500 s timeout), #115 passes, both 30/30 in a row; full Release suite 236/237 in 21 s (was 1501 s), only #145 remaining.
**Is it macOS-specific?** Not to macOS itself but to libomp: AppleClang always uses it, so it hits every Mac build; the same should happen on Linux/Windows with clang + libomp (inferred, not tested). libgomp (gcc, Linux CI) installs no atfork handlers as far as I know, so the previous code works there and the new code takes the unchanged path (dlsym returns null). The T6/T9 fork tests were designed and passed on libgomp.

### #230 `cmake_package`
The installed `libKrigingConfig.cmake` calls `find_dependency(OpenMP)`; with AppleClang, Homebrew's keg-only libomp is not found without hints (`OpenMP_ROOT`...), the same reason the library itself needs `-DOpenMP_ROOT` to be configured with OpenMP. The test consumer was not given the hints. Fix tried (test only): forward `OpenMP_ROOT`, `OpenMP_CXX_FLAGS`, `OpenMP_CXX_LIB_NAMES`, `OpenMP_libomp_LIBRARY`, `OpenMP_omp_LIBRARY` from the build to the consumer configure; the test passes. **macOS/AppleClang-specific** (Linux gcc finds libgomp by itself). Real users of the installed package on a Mac also need `-DOpenMP_ROOT=$(brew --prefix libomp)` (or the Config could record the library path): to document or decide. This also means CI macOS jobs either pass such a hint or have no OpenMP (relevant to Q4/Q5).
Debug full suite: same four failures as Release.

## Full suite, Debug (`ctest -j4 --timeout 120`, T9 branch): 233/237
Same four failures as Release: #115, #125 (timeout), #145, #230. No Debug-only failure.

## Q8 — possibly wrong `dpotri` from Accelerate (found on #449): not reproduced on the M4; the guard is too slow, a cheaper one is proposed
Branch `origin/ccr-54a50343-4jupn9-t9-blas-backends-ci` at 6b84215e (guard of #449 in `inv_sympd`), Apple M4 (10 cores), macOS 15.7.4, Accelerate.

| Check | Result |
|---|---|
| `BlasBackendTest "[backend]"` x 20 runs, default threading (each run prints `accelerate potri, n = 750: k / 20`) | k = 0 in all 20 runs: 0 / 400 faulty `dpotri`, worst relative gap 1.56e-14 (`macOS-q8-potri-backendtest.txt`) |
| same with `VECLIB_MAXIMUM_THREADS=1` | k = 0 in all 20 runs: 0 / 400 |
| Standalone reproducer (`macOS-q8-potri_repro.cpp`, no libKriging: gauss correlation matrix of the test, +1e-9, `dpotrf` then 1000 x `dpotri` on a fresh copy, against `dpotrs` of the identity) n = 250, 500, 750, 1000 (theta 0.3 0.5 0.7); n = 750 with theta 0.4048 0.6122 0.5592 (the one quoted in the issue; the test itself uses 0.3 0.5 0.7); `VECLIB_MAXIMUM_THREADS=1`; `BLASSetThreading(single)`; 5 and 10 background threads running `dgemm` | **0 faults in 9000 calls**, worst gap 2.8e-14 (n = 1000), `info` always 0 (`macOS-q8-potri-repro.txt`) |

Conclusion: **not reproduced** (inconclusive, not refuted): the failures were seen on 5-core GitHub runners (job 129621767306), none of the conditions tried here (threads, contention, size, theta) triggers it on a 10-core M4. No Apple report possible without a reproducer.

**Guard cost: not negligible.** `inv_sympd` with the guard against `potri` + `symmatl` alone (`macOS-q8-guard-cost.txt`): n = 500: +62 %, n = 1000: +100 % (6.9 ms), n = 1500: +81 %, n = 2000: +59 %, n = 3000: +35 % (58 ms). Roughly 12-14 % of an LL + gradient evaluation at n = 800-1500. The cost is in the two `arma::solve(trimat...)` on the vector (condition estimate and copy of L, plus `L.t()`), not in the O(n^2) arithmetic (`macOS-q8-guard-parts.txt`, ms at n = 500 / 1000 / 2000 / 3000: `solve` x 2 1.51 / 6.44 / 28.7 / 64.1; `cblas_dtrsv` x 2 0.05 / 0.21 / 0.98 / 3.13; `Rinv * v` 0.01 / 0.02 / 0.71 / 1.02).
**Proposal** (`macOS-q8-guard-trtrs.patch`, 21 lines, not applied: the library belongs to #449): the two solves with `arma::lapack::trtrs` ('N' then 'T', on L itself), same inequality. Cost: +8 % (n = 500), +9 % (1000), +10 % (1500), +5 % (2000), +7 % (3000) of `potri` (`macOS-q8-guard-cost-trtrs.txt`); the same vector as before (agreement of the two ways to 3e-14 at n = 1000, 1.7e-12 at n = 3000). Detection (`macOS-q8-guard-detect.txt`, n = 750): a top-left half block left unprocessed (the interposer of the issue) gives a gap of 0.91 (threshold 1e-8): detected; a correct inverse 2e-14; a single entry off by 1e-6 relative gives 7.6e-9: not detected, as before (one vector only checks the aggregate). With the patch: `BlasBackendTest` 5 cases pass, full suite on the T9 branch (without `fork()` / `cmake_package`, which fail there for other reasons: #455) 239 / 240, the failure being the MLP calibration test (#407).

**`CholJitterTest` tolerance (1e-3 on Apple only, #449):** the three fits (budget 10 %, 100 %, 200 %) are **bit-identical on the M4**, 20 runs with default threading and 20 with `VECLIB_MAXIMUM_THREADS=1`, 3 cases: theta gap and LL gap exactly 0 (`macOS-q8-choljitter.txt`). Nothing to say about 1e-4 vs 1e-3 from this machine. Also: a `dpotri` fault of the kind of the issue (gradient of order 1e10) would send the optimizer elsewhere, not give the 1.7e-4 theta gap that justified 1e-3, so the two are probably unrelated.
