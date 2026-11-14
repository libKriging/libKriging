# #467. MLPKriging default optimizer — work log

Issue: [#467](https://github.com/libKriging/libKriging/issues/467) (part of #464, follow-up of
#407). Tools and raw results: [`mlp-default-467/`](mlp-default-467/).

## Status

Done (pending review). Decision: the default optimizer of `MLPKriging` becomes `"BFGS4"` (was
`"BFGS"`) in C++ and every binding (Python, sklearn wrapper, R, Octave / MATLAB, Julia).
`WarpKriging` keeps `"BFGS"` for every warping, `mlp_joint` included (documented: pass
`"BFGS4"`). `calibration_folds` stays 3 (decision D4 of T8); the measured gain of K = 5 with the
new default is reported below for the maintainer to decide.

## Protocol

- **Benchmark base.** The T7 base (`bench/fit-protocol`) fits `Kriging` only: it has no MLP
  family. The MLP base used here is the 15-function benchmark of T8
  (`T8/bench_t8.cpp`: `branin_proj5`, `branin2`, `hartmann3`, `hartmann6`, `ishigami`,
  `borehole`, `friedman`, `rosenbrock4`, `michalewicz2`, `goldstein_log`, `otlcircuit`,
  `piston`, `forrester`, `sin_inter2`, `branin_proj10`; n 15–120, d 1–10, LHS design, 1000
  uniform test points; hidden `{8}`, `{16, 8}` for d 10; d_out 2; matern5_2; constant trend),
  same data (seed 1000 + 17 r + d), **designs 0–19** (T8: 0–8), **tanh and selu** (T8: tanh):
  600 cases per variant. Driver `mlp-default-467/bench_467.cpp` (the T8 cases, one line per fit:
  wall, CPU, var_scale, LL, test Q², NLPD, 95 % coverage, calibrated and raw).
- **#407 case.** `mlp-default-467/case407.py`: the Python script of #407 (projected Branin
  10D, n 120, W from seed 43, 2000 test points, `hidden_dims=[16, 8]`, `d_out=2`, matern5_2),
  design 0 = the design of #407 (LHS seed 44), designs 1–9 = LHS seeds 44 + 1000 k; tanh and
  selu.
- **Variants.** `optim` `"BFGS"`, `"BFGS2"`, `"BFGS4"` with `calibration_folds` K = 3 (default)
  and K = 5; `"BFGS10"` with K = 5 as the reference; plain `Kriging(matern5_2)` with its
  default (`"BFGS4"`) on the same data, as the baseline the class has to beat.
- **Cores.** 1 core = `taskset -c k` + `LK_NUM_THREADS_PERCENT=25` (budget 1), 4 processes at
  a time on the 4 cores (pass A, quality and CPU time; T3 checked that this does not inflate
  the times). 4 cores = `taskset -c 0-3`, default budget, one process alone (pass B: 15
  functions x designs 0–2, tanh; #407 case: designs 0–2). `OMP_NUM_THREADS` not set (an
  explicit value caps the budget, #407 comment).
- **CPU time** = process time (`getrusage` / `time.process_time`, all threads) of the fit,
  calibration included; wall = steady clock. Python times include the binding conversions.
- Machine: 4-vCPU Intel Xeon @ 2.10 GHz (cloud VM, 1 thread per core), 15 GB, Ubuntu 24.04,
  gcc 13.3, OpenBLAS 0.3.26 pthreads (as T3 / T7 / T8), Release, numpy 2.5.3 / scipy 1.18.1;
  library at `a8a89e3` (master after #466) with the default change only.
- Reproduction: build, compile `bench_467.cpp` with the flags of `bench/bench-mlpkriging`,
  `run.sh <bench_467> <out>` (about 1 h 15 min), `python3 report.py <out>` (=
  `mlp-default-467/report.md`). Raw outputs: `mlp-default-467/results.tar.gz` (extract, then
  `report.py` on the directory).

Statistics: Wilson 95 % intervals on rates; paired differences on the same case, median with a
bootstrap 95 % interval of the median (the means of NLPD differences are dominated by a few
fits with a collapsed variance, NLPD up to 1e13, in every variant, `Kriging` included).

## Measurements

### 15-function base, tanh + selu pooled, 1 core (600 cases per variant)

| optim:K | median Q² | Q² < 0.5, % [95 % CI] | Q² < Kriging − 0.05 | median NLPD | ΔNLPD vs BFGS:3, median [95 % CI] | NLPD > 10 | mean cov. | cov. < 0.8 | 1-core CPU, sum | x BFGS:3 |
|---|---|---|---|---|---|---|---|---|---|---|
| `Kriging` (BFGS4) | 0.953 | 7.0 [5.2, 9.3] | – | 0.24 | – | 4 | 0.923 | 30 | 26 s | 0.11 |
| BFGS:3 | -0.033 | 68.5 [64.7, 72.1] | 71 % | 2.90 | 0 | 21 | 0.958 | 14 | 231 s | 1.00 |
| BFGS:5 | -0.033 | 68.5 [64.7, 72.1] | 71 % | 2.70 | -0.03 [-0.08, -0.00] | 20 | 0.957 | 8 | 406 s | 1.76 |
| BFGS2:3 | 0.743 | 39.7 [35.8, 43.6] | 55 % | 2.21 | -0.29 [-0.55, -0.02] | 14 | 0.974 | 9 | 314 s | 1.36 |
| BFGS2:5 | 0.743 | 39.7 [35.8, 43.6] | 55 % | 2.10 | -0.96 [-1.13, -0.81] | 10 | 0.968 | 8 | 475 s | 2.06 |
| **BFGS4:3** | **0.901** | **29.3 [25.8, 33.1]** | 45 % | 1.72 | -1.17 [-1.38, -0.92] | 19 | 0.971 | 14 | 486 s | **2.10** |
| BFGS4:5 | 0.901 | 29.3 [25.8, 33.1] | 45 % | 1.45 | -1.70 [-1.88, -1.50] | 20 | 0.961 | 13 | 649 s | 2.81 |
| BFGS10:5 | 0.956 | 25.3 [22.0, 29.0] | 41 % | 1.41 | -1.90 [-2.14, -1.72] | 19 | 0.961 | 18 | 1205 s | 5.21 |

Per activation (300 cases each): tanh median Q² BFGS / BFGS2 / BFGS4 / BFGS10 −0.100 / 0.715 /
0.906 / 0.951, Q² < 0.5 69 / 40 / 29 / 25 %; selu −0.001 / 0.770 / 0.887 / 0.963, 68 / 39 / 30 /
25 %. The two activations agree; the per-function tables are in `mlp-default-467/report.md`.

Paired Q² (600 cases, a variant "better" when its Q² exceeds the other's by more than 0.05):

| comparison | first better | second better | share of discordant cases won by the second |
|---|---|---|---|
| BFGS:3 vs BFGS4:3 | 45 | 340 | 88 % |
| BFGS2:3 vs BFGS4:3 | 87 | 215 | 71 % |
| BFGS4:3 vs BFGS10:5 | 98 | 170 | 63 % |

The single start fails on the functions with d ≥ 3 and a non-trivial structure (median Q²:
borehole −0.73 / −0.29 (tanh / selu), otlcircuit −0.39 / 0.19, piston −2.97 / −0.39,
branin_proj5 −0.30 / 0.01, branin_proj10 −4.79 / −1.14, friedman −0.25 / −0.15), where `"BFGS4"`
reaches 0.84–0.999 (branin_proj10 0.64 / 0.86, the remaining gap to `"BFGS10"` 0.88 / 0.995).
It is as good on the easy 1–2D functions (branin2, forrester, sin_inter2). The K-fold
calibration keeps the coverage near 0.95 in every variant: a bad mean shows in Q² and NLPD, not
in the coverage.

### Wall and CPU time, 1 core against 4 cores (15 functions x designs 0–2, tanh, one process alone)

| optim:K | 1 core wall | 4 cores wall | speed-up | 4 cores CPU | wall x BFGS:3 (1 core / 4 cores) |
|---|---|---|---|---|---|
| BFGS:3 | 18.1 s | 13.4 s | 1.35 | 20.4 s | 1.00 / 1.00 |
| BFGS:5 | 31.9 s | 18.2 s | 1.76 | 35.8 s | 1.76 / 1.35 |
| BFGS2:3 | 24.9 s | 13.9 s | 1.80 | 27.1 s | 1.38 / 1.03 |
| BFGS2:5 | 37.1 s | 18.7 s | 1.98 | 41.4 s | 2.05 / 1.39 |
| **BFGS4:3** | 37.9 s | 14.8 s | 2.56 | 41.5 s | **2.09 / 1.10** |
| BFGS4:5 | 50.6 s | 18.9 s | 2.68 | 53.7 s | 2.79 / 1.41 |
| BFGS10:5 | 89.8 s | 31.1 s | 2.88 | 95.4 s | 4.96 / 2.32 |

The calibration folds are single-start refits (from the winning start), so a multistart adds
the starts only: `"BFGS4"` costs x 2.1 the single start on one core, not x 4.

### #407 case (projected Branin 10D, n 120, hidden [16, 8]), 1 core, 10 designs: median [min, max]

| activation | optim:K | Q² | NLPD | cov. 95 % | wall |
|---|---|---|---|---|---|
| `Kriging` (BFGS4) | – | 0.954 [0.940, 0.962] | 2.99 [2.81, 3.10] | 0.867 [0.824, 0.907] | 0.25 s |
| tanh | BFGS:3 | **−5.96 [−10.20, −0.09]** | 6.45 [5.24, 8.73] | 0.940 [0.886, 1.000] | 1.52 s |
| tanh | BFGS2:3 | 0.973 [0.867, 0.992] | 2.96 [2.14, 3.88] | 0.999 [0.980, 1.000] | 2.03 s |
| tanh | **BFGS4:3** | **0.983 [0.930, 0.993]** | 2.78 [1.74, 2.95] | 0.998 [0.981, 1.000] | 3.08 s |
| tanh | BFGS4:5 | 0.983 [0.930, 0.993] | 2.06 [1.42, 2.57] | 0.988 [0.975, 1.000] | 4.13 s |
| tanh | BFGS10:5 | 0.987 [0.976, 0.990] | 1.99 [1.61, 2.48] | 0.980 [0.919, 1.000] | 7.43 s |
| selu | BFGS:3 | **−2.48 [−11.62, −0.07]** | 6.31 [5.75, 8.29] | 0.979 [0.854, 1.000] | 1.39 s |
| selu | BFGS2:3 | 0.978 [0.731, 0.996] | 3.09 [1.53, 5.22] | 1.000 [0.990, 1.000] | 1.82 s |
| selu | **BFGS4:3** | **0.993 [0.989, 0.997]** | 1.84 [1.38, 2.51] | 1.000 [0.996, 1.000] | 2.88 s |
| selu | BFGS4:5 | 0.993 [0.989, 0.997] | 1.34 [0.88, 1.96] | 0.991 [0.975, 1.000] | 3.70 s |
| selu | BFGS10:5 | 0.996 [0.987, 0.998] | 1.60 [0.94, 1.93] | 0.997 [0.975, 1.000] | 6.72 s |

Design 0 (the design of #407): `"BFGS"` Q² −7.3 (tanh) / −3.1 (selu), as in #407 (−8.1 / −3.0
on another build); `"BFGS4"` 0.983 / 0.993. On 4 cores (designs 0–2) `"BFGS4:3"` takes 1.11 s
(tanh) / 0.95 s (selu) wall against 1.05 / 0.98 s for the single start, Q² 0.989 / 0.991;
`"BFGS10:5"` 2.5 / 2.3 s. Full tables (K = 5 rows, 4 cores, CPU): `mlp-default-467/report.md`.

### Calibration folds: K = 5 against K = 3 (paired, 600 cases, 1 core)

| optim | ΔNLPD K5 − K3, median [95 % CI] | share of cases K5 better | cov. K3 / K5 | cases cov. < 0.8 K3 / K5 | mean abs(cov − 0.95) K3 / K5 | CPU K5 / K3 |
|---|---|---|---|---|---|---|
| BFGS | −0.031 [−0.075, −0.003] | 54 % | 0.958 / 0.957 | 14 / 8 | 0.042 / 0.041 | 1.76 |
| BFGS2 | −0.090 [−0.157, −0.042] | 58 % | 0.974 / 0.968 | 9 / 8 | 0.045 / 0.043 | 1.51 |
| **BFGS4** | **−0.147 [−0.196, −0.100]** | 64 % | 0.971 / 0.961 | 14 / 13 | 0.048 / 0.044 | **1.33** (wall 4 cores: 1.28) |

K does not change the mean (Q² identical, only the variance factor differs). On the #407 case,
K = 5 with `"BFGS4"`: NLPD 2.78 → 2.06 (tanh), 1.84 → 1.34 (selu), coverage 0.998 → 0.988 and
1.000 → 0.991 (the K = 3 intervals are too wide there).

## Decisions

1. **`MLPKriging` default `optim = "BFGS4"`** (C++ constructor and `fit`, Python `MLPKriging`
   and `MLPKrigingRegressor`, R `MLPKriging()` / `fit.MLPKriging`, Octave / MATLAB, Julia).
   - The single start is not acceptable as a default: test Q² < 0.5 on 68.5 % [64.7, 72.1] of
     the cases (plain `Kriging`: 7 %), median Q² −0.03, Q² −6.0 / −2.5 on #407.
   - `"BFGS4"`: 29.3 % [25.8, 33.1], median Q² 0.90; wins 88 % of the cases where it and the
     single start disagree; #407 0.983 / 0.993 (better than `Kriging`, 0.954). Cost x 2.1 CPU on
     one core (calibration included), x 1.1 wall on 4 cores.
   - Why not `"BFGS10"`: 25.3 % [22.0, 29.0], overlapping `"BFGS4"`'s interval; it wins 63 % of
     the discordant cases (170 : 98) at x 2.5 the CPU time of `"BFGS4:3"` on one core and x 2.1
     the wall time on 4 cores. Same reasoning as T3 for `Kriging` (a default must not make the
     1–2-core cases 5 x slower); `"BFGS10"` stays the documented recommendation for a hard fit
     (it is the variant that closes the branin_proj10 gap).
   - Why not `"BFGS2"`: 39.7 % [35.8, 43.6], loses 71 % of the discordant cases to `"BFGS4"`
     for a saving of 35 % CPU on one core and none on 4 cores.
   - Consistency: same default as `Kriging` (T3), same fixed k on every machine.
2. **`WarpKriging` keeps `"BFGS"`, `mlp_joint` included.** Its default is one value for every
   warping (`kumaraswamy`, `boxcox`, `knots`, `mlp`, categorical, ...), none of which was
   measured here, and a warping-dependent default would need a sentinel value in every binding.
   The `mlp_joint` path is the code of `MLPKriging`, so the measurements above apply to it: the
   docs (skill, `Warping-MLPJoint.md`, C++ reference) tell to pass `"BFGS4"` with `mlp_joint`.
   Follow-up for the maintainer: measure the other warpings on the same protocol before
   changing the `WarpKriging` default.
3. **`calibration_folds` stays 3** (T8, decision D4 revised: K = 3 everywhere, no rule
   depending on the optimizer). Measured with the new default: K = 5 lowers the test NLPD by
   0.15 (median, CI [0.10, 0.20], better on 64 % of the cases; coverage 0.961 vs 0.971) for x 1.33
   CPU (x 1.28 wall on 4 cores); the T8 reason for K = 3 (the cost of the single-start default)
   no longer applies to the default path. Not changed here, as D4 was a maintainer decision; the
   options are (a) keep K = 3 and document K = 5 (done), (b) K = 5 everywhere (the single start,
   now explicit, then costs x 1.76 for a −0.03 NLPD), (c) a rule depending on the optimizer
   (rejected in D4). The change is one line (`m_calibration_folds`).
4. **Not experimental.** The default fit is usable (above), but the class is no general
   replacement for `Kriging`: even with `"BFGS10"`, plain `Kriging` has a test Q² higher by
   more than 0.05 on 41 % of the cases (45 % with `"BFGS4"`), at a tenth of the cost; `MLPKriging`
   wins where the response depends on a few linear combinations of the inputs (branin_proj5 /
   10, and #407). Documented in the skill (compare on held-out data before recommending it)
   rather than labelling the class experimental.

## Limits of the sample (adversarial review)

- **Not the T7 base.** #467 asks for "the MLP families of the T7 base": there are none (the
  T7 driver fits `Kriging` only). The T8 15-function base is the MLP benchmark of the project;
  its designs are small (n 15–120), which is the regime where an MLP feature map has more
  weights than observations. Larger n (500–1000), real data and other architectures are not
  measured.
- **Q² < 0.5 is a threshold of convenience**; the median Q², the paired comparisons and the NLPD
  give the same ordering. BFGS10 is a reference, not the global optimum: the likelihood of
  `MLPKriging` is highly multimodal (each start ends elsewhere), and the miss definition of T3 /
  T7 (LL 10 below the best) was not used because a higher LL does not mean a better predictor
  here (#407: more interpolating feature maps).
- **20 designs x 15 functions x 2 activations** treated as 600 independent cases: the cases of a
  function share the function, so the pooled intervals overstate the precision. Per function,
  the median Q² of `"BFGS4"` exceeds the single start's by more than 0.05 on 9 (tanh) / 10
  (selu) of the 15 functions; it is lower by more than 0.05 on two, both tanh, both functions
  where no MLP variant is usable: hartmann6 (−2.30 → −2.98; `"BFGS10"` −1.50, `Kriging` 0.57)
  and michalewicz2 (0.35 → 0.27; `"BFGS10"` 0.12, `Kriging` 0.41).
- **Timing**: 4-core wall times on 3 designs, one run each, on a shared VM (±20 %); 1-core CPU
  times on all 600 cases. Fits differ between 1 and 4 cores (thread-dependent rounding, T8):
  the 4-core Q² of the #407 case (3 designs) are not the 1-core ones.
- **One machine, one BLAS** (Linux, OpenBLAS); macOS / Windows / MKL not run (#464 lists them).
- **The K = 3 vs 5 comparison** is on the variance factor only; with both K the coverage is
  ≥ 0.95 on average, so the K = 3 cost is wider intervals, not under-coverage.
- The `MLPKriging` predictive variance remains overconfident before calibration (raw coverage
  in `results.tar.gz`, `cov95_raw` column): the feature-map regularisation of #407 is not part of
  this change.

## Tests

- `MLPKrigingTest` (Branin 2D, #382): the default is `"BFGS4"`; the explicit single start
  `"BFGS"` still reaches Q² > 0.95 there.
- `MLPKrigingCalibrationTest`, new case `MLPKriging default optimizer is the multistart BFGS4
  (#467)`: on the 5D projected Branin of the calibration test (n 60, seed 407), the default fit
  equals the explicit `"BFGS4"` fit (LL and `var_scale`), its Q² is above the single start's and
  above 0.75 (Linux: 0.860 vs −0.684, `Kriging` 0.866). Scores printed on every platform.
- `ctest` (C++ and Python binding tests): see the PR. R, Octave / MATLAB and Julia are not
  installed on the machine: their binding tests are left to the CI.
