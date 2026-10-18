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

## Decision (gate of step 1)

Gate: a variant gains > 10 % over the baseline for 200 ≤ n ≤ 1000 with OpenBLAS
pthreads. The gain does not come from an arbitration variant but from **never
exceeding the physical cores** (BLAS and OpenMP both at the budget), which then
lets the OpenMP loops run from n = 200: ×1.6–3.2 at n = 200–500, ×1.0 at n = 1000
(where `kOmpMinRows = 800` no longer applies).

- Step 4 (BLAS / OpenMP arbitration): **dropped**. (a) and (c) bring nothing;
  (b) cannot be set by the library and is documented only (step 6).
- Step 2: budget = physical cores × percentage; OpenMP loops and BLAS both at the
  budget (no split); `kOmpMinRows` → 200 together with the budget (not before:
  without it, 20/20 at n = 200 is 27–38× slower than 10/10).
- The library has to set the OpenBLAS count to the budget when OpenBLAS's own
  default (logical CPUs) exceeds it, and restore it afterwards
  (`BlasThreadGuard`), since OpenBLAS pthreads at 20 threads costs 1.3–2× by itself.
