# Thread budget — step 1 results

Machine: Intel i9-10900K, **10 cores / 20 threads** (hyperthreading), 1 NUMA node,
no AVX-512, Ubuntu 24.04, GCC 13.3, OpenBLAS 0.3.26 (pthreads and OpenMP builds),
reference BLAS/LAPACK 3.12 (`machine.txt`). Syncthing and other services stopped
during the runs. `bench-threads`: one exact log-likelihood + gradient evaluation
(`t_llgrad_ms`), median of 7 reps, Matérn 5/2. BLAS variant switched with
`LD_LIBRARY_PATH` (see the TODO). Builds: `build-tb` (`kOmpMinRows = 800`, as on
master) and `build-tb200` (`-DLK_OMP_MIN_ROWS=200`). Tables: `summarize.py`.

## Files

| File | BLAS | `kOmpMinRows` | Grid |
|---|---|---|---|
| `pthreads.csv` | OpenBLAS pthreads | 800 | n ∈ {200, 500, 1000, 2000, 3000}, d ∈ {2, 10}, BLAS × OpenMP ∈ {1, 2, 4, 10, 20}² |
| `pthreads-min200.csv` | OpenBLAS pthreads | 200 | n ∈ {200, 500, 1000} |
| `pthreads-handoff.csv` | OpenBLAS pthreads, `LK_BENCH_BLAS_HANDOFF=1` (variant a) | 200 | idem |
| `pthreads-timeout.csv` | OpenBLAS pthreads, `OPENBLAS_THREAD_TIMEOUT=4` (variant b) | 200 | idem |
| `openmp.csv`, `openmp-min200.csv` | OpenBLAS OpenMP | 800 / 200 | as above; BLAS count follows `OMP_NUM_THREADS` (5 repeats per OpenMP count) |
| `reference.csv`, `reference-min200.csv` | reference | 800 / 200 | OpenMP ∈ {1, 2, 4, 10, 20} only |
| `fits.csv` | OpenBLAS pthreads | 800 / 200 | full fits, see below |
| `reference-4cores-pthreads.csv` | — | 800 | 4-core VM of the plan, for comparison |

Spread: for the repeated OpenMP-OpenBLAS configurations, max / min of the
medians ≤ 1.36 (≤ 1.19 with `kOmpMinRows = 200`).

## 1. OpenBLAS pthreads (the R binary's variant)

`t_llgrad_ms`, d = 10 (d = 2 similar):

| n | default 20/20, min 800 | best, min 800 | 10/10, min 200 | 20/20, min 200 |
|---|---|---|---|---|
| 200 | 6.4 | 4.9 (4/4) | **1.5** | 41 |
| 500 | 44.6 | 32.4 (10/1) | **10.3** | 28 |
| 1000 | 74 | 49.3 (10/10) | **48.6** | 76 |
| 2000 | 306 | 218 (10/10) | — | — |
| 3000 | 827 | 572 (10/20) | — | — |

- Best split for n ≥ 1000: BLAS = OpenMP = 10 (physical cores), 1.4–2.0× faster
  than the default 20/20. BLAS at 20 threads is slower than at 10 in every row
  (n = 1000, d = 10, OpenMP 1: 180 vs 145 ms).
- No split of the cores between BLAS and OpenMP helps: BLAS calls and OpenMP loops
  run one after the other, each takes the whole budget.
- Break-even of the OpenMP loops: below n = 200 with 10/10 (×3 at n = 200 and 500);
  with either count at 20 (logical CPUs) the loops collapse at every n ≤ 1000
  (spinning OpenBLAS threads on the hyperthread siblings).
- `compute_dX` (threshold 200, capped at 2 threads): 0.25 ms, but 6 ms whenever
  BLAS is at 20 and OpenMP at 2–10 (same contention as on the 4-core VM).
- Covariance loop scales well (n = 1000: 91 → 11 ms at 10 threads); the Cholesky
  barely (21 → 16.6 ms at 10 BLAS threads, 23 ms at 20).

## 2. OpenBLAS OpenMP

- No collapse at 20 / 20, even with `kOmpMinRows = 200` (BLAS and loops share the
  libgomp pool): the contention is specific to the pthreads build.
- 10 threads still best; 20 is 1.2–1.35× slower for n ≥ 1000.
- `kOmpMinRows = 200`: n = 200, d = 10: 4.9 → 1.3 ms; n = 500, d = 10: 32 → 8.3 ms.
- vs pthreads (best splits, min 800): within ±1 % for n ≤ 500, 5–13 % faster at
  n = 1000–2000.

## 3. Reference BLAS

The single-threaded BLAS dominates (n = 3000: 15–20 s per evaluation); the OpenMP
loops save 5–25 % (n = 1000, d = 10: 534 → 443 ms at 10 threads). With
`kOmpMinRows = 200`, 20 threads is again worse than 10 at n = 200
(11.6 vs 4.8 ms, d = 10).

## 4. Variants (OpenBLAS pthreads, `kOmpMinRows = 200`)

Best split per variant, `t_llgrad_ms`:

| n, d | min200 | (a) handoff | (b) timeout |
|---|---|---|---|
| 200, 2 | 1.37 | 1.40 | 1.25 |
| 200, 10 | 1.54 | 1.51 | 1.35 |
| 500, 2 | 9.80 | 9.83 | 8.97 |
| 500, 10 | 10.24 | 10.21 | 8.79 |
| 1000, 2 | 43.4 | 43.8 | 47.0 |
| 1000, 10 | 48.6 | 48.6 | 48.9 |

Oversubscribed 20/20: min200 41–107 ms, (a) 35–126 ms, (b) 11.5–64 ms.

- (a) BLAS to 1 thread around the OpenMP loops: no effect, sometimes worse;
  `openblas_set_num_threads(1)` does not stop the threads already spinning.
- (b) `OPENBLAS_THREAD_TIMEOUT=4`: 10–15 % at n = 200–500 at 10/10, −8 % at
  n = 1000, d = 2; limits the damage when oversubscribed (20/20: 53 → 11.5 ms).
  Read by OpenBLAS at init only, no API: the library cannot set it; document it.
- (c) reduced OpenMP team: every split below 10/10 (4/4, 10/20, 20/10, 1/10) is
  slower than 10/10; nothing to implement.

## 5. Full fits

See `fits.csv` (Kriging gauss, d = 4, `bench-fit-profile`, 3 reps):
A = `build-tb` default 20/20 (today), B = `build-tb` 10/10 (budget only),
C = `build-tb200` 10/10 (budget + threshold 200), D = `build-tb200` 20/20.

Median fit time (ms), spread max / min ≤ 1.17:

| n | optim | A (today) | B (budget) | C (budget + 200) | D (200, no budget) | A / C |
|---|---|---|---|---|---|---|
| 500 | BFGS | 240 | 262 | **134** | 3275 | 1.79 |
| 500 | BFGS10 | 10222 | 8873 | **7096** | 29364 | 1.44 |
| 2000 | BFGS | 6272 | 3755 | **3748** | 6287 | 1.67 |
| 2000 | BFGS10 | 156478 | 75050 | **74857** | 158215 | 2.09 |

- The single-evaluation results carry over: the budget alone gives ×1.7–2.1 at
  n = 2000; the threshold 200 adds ×1.2–2.0 at n = 500 on top of it.
- Without the budget, the threshold 200 is a regression (D: ×13.6 slower at
  n = 500, BFGS): the two must ship together.

## 6. predict / simulate / update

`bench-threads-ops` (`ops.csv`, `bench-threads-ops.sh`): Kriging Matérn 5/2 with
fixed theta, d = 4, one call timed (median of 7–51 reps), BLAS = OpenMP threads
∈ {1, 10, 20}. Builds: `tb` (threshold 800, cap 2 = master), `tb200` (threshold
200, cap 2), `tbnc` (threshold 200, `-DLK_OMP_THREAD_CAP=1024`: no cap of 2
threads on compute_dX / covMat_* / R_on). Times in ms:

| op | n | n_new | nsim | today (tb, 20) | budget (tb, 10) | budget + no cap (tbnc, 10) | no cap, no budget (tbnc, 20) | today / tbnc 10 |
|---|---|---|---|---|---|---|---|---|
| predict | 500 | 1 | | 0.62 | 0.44 | 0.44 | 0.62 | 1.41 |
| predict | 500 | 100 | | 4.26 | 3.60 | **1.29** | 38.1 | 3.30 |
| predict | 500 | 1000 | | 31.5 | 29.9 | **8.0** | 21.8 | 3.92 |
| predict | 2000 | 1 | | 6.16 | 6.28 | 6.08 | 6.18 | 1.01 |
| predict | 2000 | 100 | | 34.9 | 22.3 | **14.7** | 40.6 | 2.38 |
| predict | 2000 | 1000 | | 151 | 121 | **43.7** | 80.5 | 3.47 |
| predict + cov | 500 | 1000 | | 81.0 | 78.7 | **43.5** | 88.8 | 1.86 |
| predict + cov | 2000 | 1000 | | 203 | 173 | **103** | 148 | 1.98 |
| simulate | 500 | 100 | 100 | 4.87 | 4.45 | **2.15** | 32.6 | 2.27 |
| simulate | 500 | 1000 | 100 | 98.9 | 89.3 | **55.7** | 114 | 1.77 |
| simulate | 2000 | 100 | 100 | 36.2 | 23.0 | **16.0** | 54.4 | 2.25 |
| simulate | 2000 | 1000 | 100 | 230 | 182 | **104** | 162 | 2.21 |
| update (no refit) | 500 | 10 | | 8.24 | 9.02 | 8.82 | 23.1 | 0.93 |
| update (no refit) | 500 | 100 | | 34.6 | 17.3 | **12.9** | 55.0 | 2.69 |
| update (no refit) | 2000 | 10 | | 180 | 181 | 179 | 192 | 1.00 |
| update (no refit) | 2000 | 100 | | 237 | 223 | **204** | 235 | 1.16 |

- Same rule as the objective: at the physical-core budget, predict and simulate
  gain ×1.8–3.9 once the cap of 2 threads is lifted; the budget alone gives
  ×1.0–1.6. Lifting the cap **without** the budget is a regression (up to ×30
  slower, predict n = 500, n_new = 100): cap removal and budget ship together.
- `update(refit = true)` is a warm-restarted BFGS: same objective as `fit`,
  covered by sections 1 and 5.
- Not thread-bound, separate performance items:
  - `predict` of 1 point at n = 2000 takes 6.1 ms whatever the threads (1 / 10 /
    20); cause not profiled yet (a triangular solve of n × 1 alone is ~n² = 4·10⁶
    flops). Matters for sequential design loops (one predict per candidate).
  - `update` without refit at n = 2000 takes ~180 ms for 10 new points whatever
    the threads: `update_no_refit_impl` rebuilds the whole `m_dX` (d × n² =
    128 MB at n = 2000) instead of appending the new rows.
- Not measured: `update_simulate` (its cache keyed on `X_u` makes repeated calls
  hit the cache), Vecchia / Nystrom variants, NuggetKriging / NoiseKriging.

## 7. Kriging multistart in the thread pool

Kriging multistart moved from a sequential loop to `detail::run_task_pool`
(`TaskPool.cpp`, shared with WarpKriging). `fits-multistart.csv`:
`bench-fit-profile` (gauss, d = 4), `build-tb` (threshold 800, cap 2), median of 3
reps (1 at n = 2000), spread max / min ≤ 1.41. `LK_THREAD_POOL_SIZE=1` = the former
sequential behaviour (one worker with all the BLAS / OpenMP threads). Pool shares
still computed from the 20 logical CPUs (step 2 not done).

| n | optim | sequential (pool 1) | pool 10 (×2 threads) | default (pool = starts ≤ 20) | sequential / best |
|---|---|---|---|---|---|
| 100 | BFGS10 | 236 | 99 | 100 (10 × 2) | 2.4 |
| 100 | BFGS20 | 640 | **127** | 271 (20 × 1) | 5.0 |
| 500 | BFGS10 | 10915 | 2391 | 2356 (10 × 2) | 4.6 |
| 500 | BFGS20 | 29767 | 5407 | **4067** (20 × 1) | 7.3 |
| 2000 | BFGS10 | 163382 | 62250 | 63497 (10 × 2) | 2.6 |

- The multistart-first split pays: ×2.4–7.3 over the sequential multistart.
- n = 500, BFGS20: 20 single-threaded workers (all logical CPUs) beat 10 × 2. With
  1 BLAS thread per worker there are no spinning OpenBLAS threads, and the
  hyperthreads add throughput: the "physical cores" rule is for BLAS / OpenMP
  teams; for the pool workers the logical CPUs may be better. To measure again in
  step 2 (budget at 10 vs 20 workers).
- n = 100, BFGS20: 20 workers are 2× slower than 10. The pool staggers the first
  task of worker k by k × `Optim::thread_start_delay_ms` (10 ms): 190 ms for the
  last of 20 workers, more than a whole n = 100 start. The delay should scale
  with the problem or go (it was introduced for WarpKriging).
  Done: `run_task_pool(..., stagger)`, off for the Kriging multistart: n = 100,
  BFGS20, 20 workers 271 → 101–113 ms (3 runs), on par with 10 workers.

## 8. Step 2 build (thread budget implemented), default environment

`fits-step2.csv`, `ops-step2.csv`: `build-tb` at the step 2 commit, no environment
variable (budget = 10 physical cores; OpenBLAS starts at its default, 20, and
each public call sets it to 10 for its duration).

Full fits (ms, median of 3; 1 at n = 2000 BFGS10):

| n | optim | A (before) | C (budget by env, 10/10) | step 2 default | A / step 2 |
|---|---|---|---|---|---|
| 500 | BFGS | 240 | 134 | 228 | 1.05 |
| 500 | BFGS10 | 10222 | 7096 | 2993 | 3.4 |
| 2000 | BFGS | 6272 | 3748 | 3740 | 1.68 |
| 2000 | BFGS10 | 156478 | 74857 | 71610 | 2.19 |

predict / simulate / update: step 2 default matches the best configuration of
section 6 (`tbnc` 10/10) within the noise, ×1.5–3.8 over before, except the
single-point predict and the small updates (unchanged, not thread-bound).

**~~OpenBLAS pthreads cannot be reduced at run time.~~ Superseded by section 9**:
the gap below is a start-up transient (every row is the first fit of a fresh
process), not a steady-state cost. n = 500, BFGS (single start), 5 runs each:

| OpenBLAS count | fit (ms) |
|---|---|
| `OPENBLAS_NUM_THREADS=10` (set before OpenBLAS loads) | 127–176 |
| default 20, set to 10 by the library (`BudgetScope`) | 199–256 |
| default 20, `openblas_set_num_threads(10)` once at program start | 229–254 |
| same + `blas_thread_shutdown_()` (pool recreated) | 142–208; the pool comes back with 20 threads (29 threads in the process) |
| default 20 + `OPENBLAS_THREAD_TIMEOUT=4` | 122–137 |

OpenBLAS 0.3.26 pthreads creates its threads at load time; lowering the count
afterwards does not stop the extra threads from spinning on every call. The
library can only reach the env-level performance if `OPENBLAS_NUM_THREADS` (or
`OPENBLAS_THREAD_TIMEOUT`) is set before OpenBLAS is loaded: documentation for
C++ users, and possibly the R / Python bindings setting it before loading the
bundled OpenBLAS (step 3). Larger problems (n = 2000, multistart) are not
affected: the step 2 default already matches C there.

Multistart pool size (BFGS20, median of 3): default pool = budget (10 workers ×
1) vs `LK_THREAD_POOL_SIZE=20` (20 workers × 1, on the hyperthreads): n = 100:
123 vs 104 ms; n = 500: 3463 vs 3258 ms (−6 to −15 % with 20 workers). With one
thread per worker, the workers may use the logical CPUs (decision pending).
Sequential (before): 640 / 29767 ms.

## Decision (gate of step 1)

Gate: a variant gains > 10 % over the baseline for 200 ≤ n ≤ 1000 with OpenBLAS
pthreads. The gain does not come from an arbitration variant but from **never
exceeding the physical cores** (BLAS and OpenMP both at the budget), which then
lets the OpenMP loops run from n = 200: ×1.6–3.2 at n = 200–500, ×1.0 at n = 1000
(where `kOmpMinRows = 800` no longer applies).

- Step 4 (BLAS / OpenMP arbitration): **dropped**. (a) and (c) bring nothing;
  (b) cannot be set by the library and is documented only (step 6).
- Step 2: budget = physical cores × percentage; OpenMP loops and BLAS both at the
  budget (no split); the cap of 2 threads (`get_optimal_threads(2)`) removed and
  `kOmpMinRows` → 200, both together with the budget (not before:
  without it, 20/20 at n = 200 is 27–38× slower than 10/10).
- The library has to set the OpenBLAS count to the budget when OpenBLAS's own
  default (logical CPUs) exceeds it, and restore it afterwards
  (`BlasThreadGuard`), since OpenBLAS pthreads at 20 threads costs 1.3–2× by itself.

## 9. Start-up transient of OpenBLAS pthreads (2026-10-19)

The section 8 gap (single-start fit, n = 500: ~215 ms with OpenBLAS loaded at 20
threads and set to 10 by the library vs ~125 ms with `OPENBLAS_NUM_THREADS=10`)
was investigated:

- Same optimizer path: 8 BFGS iterations, same optimum, in every configuration.
- Only the **first fit of the process** is slow: three fits in a row (default
  OpenBLAS) take 215–225, 123–151, 121–133 ms; with `OPENBLAS_NUM_THREADS=10`:
  132–135, 127–135, 126–127 ms.
- Not page faults (`GLIBC_TUNABLES` mmap / trim thresholds: 3× fewer minor faults,
  same first fit). Involuntary context switches: 959 (default) vs 135 (env 10):
  more runnable threads than CPUs.
- OpenBLAS alone (C program, `dpotrf` n = 500) shows no such cost.
- Busy-waiting before the first fit (main thread, CPU kept at full clock), median
  of 5 runs:

  | wait | default (20, set to 10) | `OPENBLAS_NUM_THREADS=10` | `OPENBLAS_THREAD_TIMEOUT=4` |
  |---|---|---|---|
  | 0 ms | 204–250 ms | 127–151 ms | 111–126 ms |
  | 150 ms | 113–147 ms | 112–121 ms | 111–122 ms |
  | 500 ms | 113–169 ms | 112–140 ms | 109–132 ms |

  (`sleep` instead of busy-wait gives noisy results for every configuration: the
  CPU clocks down while idle.)

Cause: OpenBLAS pthreads starts its threads when it is loaded; they spin for
`OPENBLAS_THREAD_TIMEOUT` (~2^28 cycles, ~50–100 ms here) before sleeping. Work
done in that window competes with 19 spinning threads. Afterwards, the threads
beyond the count set by the library are never woken again: **lowering the
OpenBLAS count at run time is as fast as `OPENBLAS_NUM_THREADS`**. Per
evaluation (`bench-threads`, OpenBLAS loaded with 20, set once to 10 vs 1):

| n | set to 10 | set to 1 | `OPENBLAS_NUM_THREADS=10` |
|---|---|---|---|
| 200 | 1.17–1.22 ms | 1.62–1.68 ms | 1.20 ms |
| 500 | 8.2–9.2 ms | 11.5–13.7 ms | 8.2 ms |
| 1000 | 32.8–33.8 ms | 61.9–63.4 ms | 37.3 ms |
| 2000 | 203–208 ms | 438–442 ms | — |

Setting OpenBLAS to 1 thread is slower everywhere (`blas-switch.csv`: also
predict / simulate, ×1.3–5.7); the size-dependent switch to 1 thread considered
for decision 8.1 is dropped. Full fits compared across fresh processes also differ
by their optimizer path when the BLAS thread count changes (rounding), which made
that file's fit columns unusable for a crossover.

Effect on the earlier measurements: every bench run is a fresh process, and the
short ones (n = 200–500 single evaluations, a few ms each) fall inside the window.
`warmup-check.csv` re-measures the key cells with and without a 300 ms busy-wait
(`LK_BENCH_WARMUP_MS`, now 300 by default in `bench-threads`, `bench-threads-ops`,
`bench-fit-profile`):

- The step 1 conclusion holds in steady state: with the loops from n = 200,
  (BLAS 10, OpenMP 10) gives 1.5 ms at n = 200 and 10 ms at n = 500; with either at
  20, 19–43 ms and 31–50 ms; at n = 1000, 43–50 ms vs 79–103 ms.
- The `compute_dX` slow-down (6 ms instead of 0.25 ms with BLAS at 20) was mostly
  the transient: 0.04–1.0 ms after the warm-up in every configuration but one.

