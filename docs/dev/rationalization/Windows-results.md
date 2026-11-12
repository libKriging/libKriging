# Windows (physical machine): results of the #473 session

Session on a physical Windows machine for #473 (part of #464, takes over the TODO
of #417). Branch `windows-session-1` from `origin/master` 6e008b24. Raw numbers,
commands and conclusions; the summary goes on #473.

Status: sections 0–2 done for MSVC; MinGW, R and section 3 not done (see "Not done").

Conventions: no `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` in the environment
unless stated; `OPENBLAS_L2_SIZE=2048` (`tools/windows/loadenv.sh`) always.

## 0. Setup

### Machine

| | |
|---|---|
| Model | Alienware Area-51m R2 (laptop, on mains power, battery 100 %, charging status 2 = AC) |
| CPU | Intel Core i9-10900K @ 3.70 GHz (Comet Lake), 10 cores / 20 logical CPUs, 1 socket, 1 processor group |
| SMT layout | logical CPUs 2k and 2k+1 are siblings (checked with affinity masks, section 2.1) |
| RAM | 64 GB |
| OS | Windows 10 Famille 22H2, build 19045.6456 |
| Compiler | MSVC 19.29.30159 (Visual Studio 2019 Build Tools 16.11.52, toolset 14.29.30133), generator "Visual Studio 16 2019", `CMAKE_GENERATOR_PLATFORM=x64` |
| CMake / Ninja | CMake 3.31.1, Ninja (conda-forge, `~/Miniconda3/Library/bin`) |
| Shells | Git Bash (Git 2.51.0.windows.2); MSYS2 at `C:\msys64` |
| BLAS | conda-forge `openblas` / `libopenblas` 0.3.33 `pthreads_*` (the CI pin), `openblas.dll` from `~/Miniconda3/Library/bin`; `libblas` / `liblapack` resolved to the MKL variant (3.9.0 `*_mkl`, mkl 2024.2.2) by the CI install command, **not linked**: Armadillo finds `openblas.lib` first |
| Other | Python 3.14.0 (python.org), R 4.5.1 + Rtools45 |

Install as `tools/windows/install.sh` does, without admin rights (no Chocolatey
packages: MinGW / make are only needed for the Octave toolchain):

```sh
curl -o ~/Downloads/Miniconda3-latest-Windows-x86_64.exe https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
(cd ~/Downloads && cmd //c tools\\windows\\install_conda.bat)
~/Miniconda3/Scripts/conda.exe install -y -n base --override-channels -c conda-forge \
  "openblas=0.3.33" "libopenblas=0.3.33" liblapack pkg-config cmake ninja
git submodule update --init --recursive
```

### MSVC build (C++ only)

```sh
export PATH="$HOME/Miniconda3/Library/bin:$PATH" OPENBLAS_L2_SIZE=2048   # tools/windows/loadenv.sh
MODE=Release BUILD_DIR=build ENABLE_PYTHON_BINDING=off ENABLE_OCTAVE_BINDING=off \
  ENABLE_MATLAB_BINDING=off tools/windows/build.sh
```

| | |
|---|---|
| OpenMP | found, MSVC `vcomp` (`-openmp`, OpenMP 2.0); Armadillo's own OpenMP disabled |
| BLAS / LAPACK linked | `openblas.lib` (OpenBLAS provides LAPACK) |
| `LinearAlgebra::check()` | `backend: OpenBLAS 0.3.33 DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=128 \| core Haswell (C:\Users\riche\Miniconda3\Library\bin\openblas.dll)`, all checks passed |
| OpenBLAS count at load | 20 (logical CPUs; also read from `OMP_NUM_THREADS` when `OPENBLAS_NUM_THREADS` is unset) |
| Thread budget | 10 (`Optim::get_num_threads()`, physical cores) |
| `BlasBackendTest` | prints `backend: unknown, OpenBLAS count: 0`: the test has no Windows path (weak symbols / `dladdr` only), see 1.4 |

## 1. Master on Windows

### 1.1 Full `ctest`, MSVC Release

241 tests, 239 passed, **2 failed** (both `ConcurrentCallsTest`), 2 runs:

| Test | Result | Cause | Action |
|---|---|---|---|
| 124 `Concurrent calls at a budget of 1 thread give the sequential results` (tolerance 0) | fails 5/5, deterministic | job `matern3_2 BFGS LOO`: theta, sigma2, predictions, simulations bit-identical, `leaveOneOut()` 1–2 ulp apart (relative 1.06e-22 of max(1, \|LOO\|)). The public objective functions open no `BudgetScope`: they run at the OpenBLAS count of the moment, 20 (loaded) in the sequential reference, 1 while another thread's call holds its scope | **PR #474** |
| 125 `Concurrent fit / predict / simulate from user threads` (tolerance 1e-4) | fails 5/5, the same numbers to the last digit | job `matern5_2 BFGS4 LL`: theta 1.93e-4, sigma2 8.06e-4, stdev 2.13e-4, objective 2.5e-9, mean 2.4e-7 relative; see below | open |

Both pass with `OPENBLAS_NUM_THREADS=1`, and with `OMP_NUM_THREADS=1`: this OpenBLAS
reads `OMP_NUM_THREADS` when `OPENBLAS_NUM_THREADS` is unset. In the Windows CI the
MSVC C++ suite runs only inside the Python jobs (`main.yml`: Python 3.10 / 3.14
Windows Debug; no C++-only Windows job), which export `OMP_NUM_THREADS=1`
(`tools/windows/build.sh`, `test.sh`; log of run 43271745830, 6e008b24): OpenBLAS is
loaded with 1 thread there, and these cases **never run with a multithreaded
OpenBLAS in CI**. The Octave Windows job (MinGW) sets it too.

Test 125, isolated by environment (5 runs each, identical numbers in every run):

| Environment | OpenBLAS count | largest relative gaps (theta, sigma2) | result |
|---|---|---|---|
| none | 20 | 1.93e-4, 8.06e-4 | fail (tolerance 1e-4) |
| `OPENBLAS_NUM_THREADS=10` | 10 | 1.93e-4, 8.06e-4 | fail |
| `LK_BLAS_THREADS_MIN_N=0` | 20 | 1.93e-4, 8.06e-4 | fail |
| `LK_THREAD_POOL_SIZE=1` | 20 | 1.93e-4, 8.06e-4 | fail |
| `OPENBLAS_NUM_THREADS=1` | 1 | 0, 0 | pass (bit-identical required) |
| `OMP_NUM_THREADS=1` | 1 | 0, 0 | pass (bit-identical required) |

Reading: n = 50, but the automatic small-problem rule (`blas + budget > logical
CPUs`) is off on this SMT machine (10 + 10 = 20), so the sequential `BFGS4` fit runs
OpenBLAS at the worker share (2), the concurrent one at 1 (budget 10 / 6 calls).
The objective agrees to 2.5e-9: the optimizer stops at different points of a flat
optimum, the documented amplification of BLAS rounding (`docs/dev/parallel.md`,
section 3.1). Not a race (deterministic, races give order-1 gaps). The tolerance
1e-4 had a margin of 1.4x on Linux on the same CPU (7.2e-5); on Windows with
OpenBLAS 0.3.33 the gap is 8x the tolerance.

MSVC **Debug** (`MODE=Debug BUILD_DIR=build-debug tools/windows/build.sh`, 180 s of
tests): 239/241, the same two failures with the same gaps to the last digit.

`fork()` tests: compiled out on Windows (`#if defined(LK_TEST_FORK)`, POSIX only),
so not registered: neither run nor failed (not reported as skipped either).

Repetitions (Release, `ConcurrentCallsTest.exe` / `BlasThreadsTest.exe` run 20 times
each, default environment and `LK_NUM_THREADS_PERCENT=50`, 600 s timeout):

| Executable | environment | runs | result | seconds (min–max) |
|---|---|---|---|---|
| BlasThreadsTest | default | 20 | 20 pass | 1.80–2.00 |
| BlasThreadsTest | `LK_NUM_THREADS_PERCENT=50` | 20 | 20 pass | 1.83–2.07 |
| ConcurrentCallsTest | default | 20 | 20 fail (124, 125, same gaps) | 0.58–0.67 |
| ConcurrentCallsTest | `LK_NUM_THREADS_PERCENT=50` | 20 | 20 fail (124, 125, same gaps) | 0.57–0.66 |

No hang, no timing outlier. `LK_NUM_THREADS_PERCENT` does not reach the
`ConcurrentCallsTest` cases, which set their budget themselves (1e-6 % and 100 %).

### 1.3 Python (`pylibkriging`, MSVC Debug as the CI job)

Python 3.14.0 venv, numpy 2.5.4 (no scipy: not a requirement), pybind11 2.13.6,
`MODE=Debug BUILD_DIR=build-py-debug ENABLE_PYTHON_BINDING=on tools/windows/build.sh`,
`LIBKRIGING_DLL_PATH=~/Miniconda3/Library/bin`, **no** `OMP_NUM_THREADS`.

- Full `ctest` (C++ and Python, 262 tests, `--timeout 1500`): 258 passed, 2 skipped
  (`Python/sklearn_estimator`, `Python/sklearn_multimodel`: scikit-learn not
  installed), 2 failed (the `ConcurrentCallsTest` pair of 1.1); 212 s. Every
  Python test passed, without hang.
- Reproducer of `WindowsPythonHangDiagnostic354.md`
  (`python -m pytest WrappedPyKriging_parametric_test.py`, 8 tests), 20 runs:
  20 passed, 1.08–1.13 s each. Its n = 40 stays below `kOmpMinRows` = 800: no
  parallel region is entered; the threshold-200 check is in section 2.
- OpenBLAS DLLs in the process after `import numpy, pylibkriging`:
  `numpy.libs\libscipy_openblas64_-ed4f167a….dll` (numpy's: OpenBLAS 0.3.34.106.0
  `USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=24`, 20 threads) and
  `Miniconda3\Library\bin\openblas.dll` (libKriging's, LP64, 0.3.33).
  `GetModuleHandle("openblas.dll")` resolves in the process, so
  `get_openblas_set_num_threads()` finds **libKriging's own** OpenBLAS (the one
  its BLAS calls go to); `libopenblas.dll` is absent. numpy's DLL exports both
  `openblas_set_num_threads` and `scipy_openblas_set_num_threads64_`, but under a
  hashed file name the lookup never matches.
- Conclusion for #417: the lookup does not need the `scipy_openblas*` names.
  numpy's OpenBLAS is a second, ILP64 thread pool that libKriging never calls;
  driving it would change numpy's own thread count for no gain. The only overlap
  is numpy work and libKriging work interleaved in one Python thread, which run
  one after the other. Proposal: no change, document it (numpy's pool is
  `threadpoolctl`'s business).
- The binding holds the GIL during `fit` / `predict` / `simulate` (only
  `linearalgebra_check` releases it): Python threads calling libKriging run one
  after the other, so the concurrent-calls budget split of `parallel.md` 3.1 never
  happens from Python threads (joblib processes are not concerned). Not
  Windows-specific; reported, not changed here.
- `BFGS4` fit with numpy loaded (n = 600, d = 4, gauss, Debug): 2.62 s, theta
  0.5946 / 0.5949 / 0.5940 / 0.5961, LL 1652.7256. The OpenBLAS count cannot be
  observed from Python during the call (GIL held); it is checked in C++ by
  `ConcurrentCallsTest`'s monitor thread (the same `GetProcAddress` lookup).

### 1.2 Objective functions and the BLAS count (PR #474)

Adding `BudgetScope(n)` to the 17 public objective functions of `Kriging` and to
`WarpKriging::logLikelihoodFun` fixes test 124 (3/3), and makes 3 reproducibility
tests fail. Those tests assumed the BLAS count did not move across budgets, and on
master it never did, because the objective ignored the budget; with the budget
applied, this OpenBLAS rounds differently with its count:

| Test | quantity | relative gap, budget 1 vs 4 / 10 threads (n = 900, matern5_2) |
|---|---|---|
| `BlasThreadsTest` OpenMP count | ll | 2.7e-10 (6819.4289880089 vs 6819.4289898641) |
| | updated ll | 6.3e-10 |
| `BlasThreadsTest` budget | ll (budget 2) | 2.8e-10 |
| `GradientThreadsTest` | LL / LMP gradient, n = 100 gauss | last bits |

They now compare to the existing MKL tolerances when OpenBLAS / MKL is loaded with
more than 1 thread; bit-identity stays in the `*_NUM_THREADS=1` variants (decision:
maintainer, 2026-11-12). Full `ctest` with #474: 241/242 (only 125 fails).

### 1.4 Gaps in the Windows test / bench tooling

- `BlasBackendTest`: no Windows path to find the BLAS provider or read the OpenBLAS /
  MKL counts (weak symbols and `dladdr` only): `backend: unknown`, `LK_EXPECT_BLAS`
  cannot be checked on Windows.
- `bench-threads`: `blas_threads` / `blas_kind` read through weak symbols only:
  `-1` under MSVC (the sweep script sets `OPENBLAS_NUM_THREADS`, so the value is
  known from the command line).

## 2. Thread budget and OpenMP threshold

### 2.1 Physical cores under an affinity mask (PR #475)

Budget printed by `BlasBackendTest` under `start /b /wait /affinity <mask>`:

| mask | logical CPUs | cores | master | #475 |
|---|---|---|---|---|
| 1 | 0 | 1 | 10 | 1 |
| 3 | 0,1 | 1 | 10 | 1 |
| 5 | 0,2 | 2 | 10 | 2 |
| F | 0–3 | 2 | 10 | 2 |
| FF | 0–7 | 4 | 10 | 4 |
| AAAAA | odd CPUs | 10 | 10 | 10 |
| FFFFF | all | 10 | 10 | 10 |

Without a mask the count is right (10 with SMT on; SMT off not tested: it needs a
firmware setting and a reboot). On master the
mask was ignored (`physical_cores_os()` counted every core; `hardware_concurrency()`
ignores the mask too). Single processor group here: the multi-group path is
unchanged and not tested.

### 2.2 `bench-threads` sweep (MSVC, `vcomp`, OpenBLAS 0.3.33)

`bench/thread-budget/windows-sweep.sh` (the loop of `bench/bench-threads.sh`, with
the environment recorded in each row, because `blas_threads` reads `-1` under
MSVC): d = 4, matern5_2, n ∈ {100, 200, 400, 800, 1500}, OpenBLAS × OpenMP
threads ∈ {1, 2, 4, 10, 20}², 7 reps per process (median), **5 processes per
configuration** interleaved in time, for the default build (threshold 800) and a
`-DLK_OMP_MIN_ROWS=200` build (`build-omp200`, `CMAKE_CXX_FLAGS="-DWIN32 -D_WINDOWS
-GR -EHsc -DLK_OMP_MIN_ROWS=200"`: from Git Bash, `/D...` flags are mangled into
paths), without and with `OPENBLAS_THREAD_TIMEOUT=4`. 2500 rows:
`bench/thread-budget/windows-msvc.csv`. Idle machine (CPU < 3 % before), mains
power, "Utilisation normale" power plan.

Noise: (max − min) / median over the 5 processes, across the 500 configurations:
median 3.3 %, p90 18 %, max 79 % (n = 200 cells, < 1 ms phases).

One log-likelihood + gradient evaluation (`t_llgrad_ms`), median (min–max) of the 5
processes, no extra environment variable:

| n | threshold | BLAS 1 / OMP 1 | 10 / 1 | 1 / 10 | **10 / 10** (budget) | 20 / 10 | 10 / 20 | 20 / 20 | best |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 800 | 1.03 (1.02–1.05) | 1.45 | 1.03 | 1.37 (1.31–1.55) | 1.49 | 1.64 | 1.47 | 1.03 (1 / 10) |
| 100 | 200 | 1.06 | 1.41 | 1.03 | 1.42 (1.36–1.69) | 1.52 | 1.33 | 1.52 | 1.03 (1 / 4) |
| 200 | 800 | 3.94 (3.88–3.98) | 5.05 | 6.00 | 5.66 (5.22–7.07) | 6.09 | 5.64 | 6.05 | 3.94 (1 / 1) |
| 200 | 200 | 3.90 | 5.03 | 2.36 (2.24–3.83) | **3.54** (3.49–3.67) | 4.13 | 3.69 | 4.21 | 2.32 (1 / 20) |
| 400 | 800 | 14.82 | 16.85 | 14.77 | 16.82 (16.53–17.61) | 18.64 | 16.84 | 18.65 | 14.69 (1 / 20) |
| 400 | 200 | 14.84 | 16.84 | 7.97 (7.65–8.06) | **11.21** (10.30–11.53) | 12.66 | 10.51 | 12.76 | 7.97 (1 / 10) |
| 800 | 800 | 62.78 | 65.50 | 34.12 | 34.21 (33.24–35.94) | 38.53 | 34.01 | 37.67 | 33.62 (4 / 20) |
| 800 | 200 | 62.75 | 65.17 | 34.35 | 34.36 (33.80–35.52) | 38.18 | 34.42 | 38.08 | 32.49 (2 / 20) |
| 1500 | 800 | 247.88 | 211.35 | 152.25 | **117.70** (111.92–128.96) | 204.10 | 113.86 | 203.36 | 113.86 (10 / 20) |
| 1500 | 200 | 247.97 | 211.76 | 153.72 | 115.44 (112.76–118.77) | 203.52 | 113.15 | 203.43 | 113.15 (10 / 20) |

Covariance matrix alone (`t_cov_ms`, the OpenMP loop), n = 1500: 92–98 ms with
OpenMP 1, 12.9–14.7 ms with OpenMP 10 or 20 **if OpenBLAS has ≤ 10 threads**, back
to 93–98 ms with OpenBLAS 20 (20 / 10, 20 / 20) although no BLAS call runs during
that loop.

`OPENBLAS_THREAD_TIMEOUT=4`: every ratio "none / timeout" of the table above lies
in 0.92–1.04 (one outlier, 1.48, in the noisy n = 200 / threshold 800 / 1 / 10
cell); the collapse with OpenBLAS 20 is unchanged. `KMP_BLOCKTIME`: not applicable,
MSVC links `vcomp`, not LLVM libomp.

Reading:

- **Budget**: confirmed. At n = 1500, BLAS = OpenMP = logical CPUs (20 / 20, what
  an unbudgeted process gets: OpenBLAS defaults to the logical CPUs) costs ×1.7
  versus the budget (10 / 10); with OpenBLAS at 20 the OpenMP loops run at
  sequential speed. Unlike Linux, a shorter spin (`OPENBLAS_THREAD_TIMEOUT=4`)
  does not help.
- **Threshold 200**: confirmed per evaluation at the budget: ×1.60 (n = 200),
  ×1.50 (n = 400); neutral at n = 100 (below 200) and n ≥ 800 (both parallel).
- **Small problems want one BLAS thread here**: at n = 200–400, 1 / 10 beats
  10 / 10 by ×1.4–1.5 with threshold 200 (2.36 vs 3.54, 7.97 vs 11.21), and at
  n = 100 by ×1.33. The automatic small-problem rule (`Optim::blas_threads_min_n`,
  #436) is off on this machine because BLAS + budget (20) does not exceed the
  logical CPUs (20); on Linux with SMT the BLAS at the budget was faster at every
  size (`RESULTS.md` section 9). On Windows OpenBLAS the reverse holds below
  n ≈ 800. See 2.4.

### 2.3 Full fits (`bench-fit-profile`, gauss, d = 4)

`bench/thread-budget/windows-fits.ps1` (wall time of the fit as printed, process
CPU time from `TotalProcessorTime`), 5 runs interleaved per configuration, no
environment variable (budget 10, OpenBLAS loaded with 20). `bench-fit-profile`
now prints LL and theta (local change on this branch): **the optimum is
bit-identical between the two thresholds** in every case, so the work done is the
same. `bench/thread-budget/windows-msvc-fits.csv`.

| n | optim | threshold 800: fit ms, median (min–max) | threshold 200 | gain (wall) | process CPU 800 → 200 |
|---|---|---|---|---|---|
| 500 | BFGS | 411.0 (403.3–417.7) | 341.8 (336.5–351.8) | ×1.20 | 1.81 s → 4.08 s |
| 500 | BFGS4 | 491.0 (472.8–511.3) | 439.3 (432.8–455.1) | ×1.12 | 3.11 s → 4.59 s |
| 2000 | BFGS | 4878 (4861–5010) | 4916 (4841–5418) | ×0.99 | 41.3 s → 42.0 s |
| 2000 | BFGS4 | 14387 (14220–14492) | 14347 (14305–14433) | ×1.00 | 70.7 s → 70.7 s |

LL: 1163.4590957113 (500, BFGS), 1163.4590953625 (500, BFGS4), 11714.6222567942
(2000, BFGS), 11714.6226657303 (2000, BFGS4).

Reading: the threshold 200 gains ×1.12–1.20 of wall time on full fits at n = 500,
for ×1.5–2.3 more CPU time; nothing at n = 2000. Smaller than the per-evaluation
×1.5–1.6 (a fit also spends time in BLAS-bound phases and in the multistart pool).

### 2.4 Hang check at threshold 200 (no `OMP_NUM_THREADS`)

C++ (`build-omp200`, MSVC Release, full tree):

- full `ctest --timeout 1500`, 3 runs: 239/241 each (58.6–62.2 s), the two
  `ConcurrentCallsTest` failures of 1.1 only;
- `ConcurrentCallsTest.exe` / `BlasThreadsTest.exe`, 20 runs each, default and
  `LK_NUM_THREADS_PERCENT=50`: `BlasThreadsTest` 40/40 pass (1.80–2.00 s),
  `ConcurrentCallsTest` 40/40 the same two failures (0.55–0.65 s). No hang, no
  timing outlier: identical to threshold 800 (1.1).

Python (`build-py-omp200`: MSVC Debug as the CI job, threshold 200, no
`OMP_NUM_THREADS`):

- Python suite (`ctest -L Python`) × 5: 19 passed + 2 skipped (scikit-learn)
  each time, 32.8–33.6 s. Three Python tests cross the threshold (n = 200, 200,
  400: `pylibkriging_nested_test.py`, `test_new_features.py`,
  `threading_test.py`).
- #354 reproducer × 20: 20 passed, 1.08–1.13 s.
- Stress: 200 `BFGS4` fits + predict (n = 300, d = 4, matern5_2, 250 prediction
  points) in one Python process: 85.9 s, per fit median 0.370 s (0.317–1.321). No
  hang.

### 2.5 One BLAS thread for small problems

The sweep (2.2) shows OpenBLAS at 1 thread beating 10 below n ≈ 800. Full fits,
threshold-200 build, default (rule off: SMT headroom) vs `LK_BLAS_THREADS_MIN_N=500`,
5 runs, `bench/thread-budget/windows-msvc-smalln.csv`:

| n | optim | default ms (min–max) | 1 BLAS thread | gain | CPU ms | LL default / 1 thread |
|---|---|---|---|---|---|---|
| 200 | BFGS | 215.6 (197.0–225.3) | 135.5 (125.6–146.9) | ×1.59 | 2484 → 1703 | 104.9913221612 / 105.0070262614 |
| 200 | BFGS4 | 115.7 (115.3–120.2) | 104.7 (101.0–112.0) | ×1.10 | 2125 → 2016 | 106.3404412663 / …657 |
| 300 | BFGS | 185.8 (169.9–190.6) | 123.2 (111.7–164.6) | ×1.51 | 2203 → 1578 | 336.4893950428 / …49941 |
| 300 | BFGS4 | 314.8 (313.3–326.4) | 291.1 (284.3–298.9) | ×1.08 | 3125 → 2875 | 336.4893920341 / …20145 |
| 400 | BFGS | 260.9 (255.4–269.3) | 216.1 (183.6–233.5) | ×1.21 | 3125 → 2484 | 683.3786642404 / …39158 |
| 400 | BFGS4 | 329.4 (315.6–332.8) | 306.4 (301.0–315.0) | ×1.08 | 3641 → 3484 | 683.3786728010 / …29527 |

At threshold 800 (master vs the PR build, 5 runs): BFGS ×1.38 / ×1.33 / ×1.16,
BFGS4 ×1.09 / ×1.12 / ×1.06 at n = 200 / 300 / 400. Opposite to Linux, where with
SMT the BLAS at the budget won at every size: **PR #478** (rule always on with
MSVC).

## Decisions and PRs

| # | PR | What | Based on |
|---|---|---|---|
| 1 | #474 | public objective functions open a `BudgetScope`; tests compare to a tolerance when the BLAS count can change | 1.1, 1.2 |
| 2 | #475 | Windows: cores of the affinity mask | 2.1 |
| 3 | #476 | `kOmpMinRows` 200 with MSVC (MinGW, macOS keep 800) | 2.2–2.4 |
| 4 | #477 (on #474) | Windows Python CI without `OMP_NUM_THREADS=1`; `ConcurrentCallsTest` full-budget tolerance 1e-4 → 2e-3 | 1.1, 2.4 |
| 5 | #478 | small-problem rule always on with MSVC | 2.2, 2.5 |

Proposals without PR:

- Python OpenBLAS lookup: no change (1.3).
- `BlasBackendTest` and `bench-threads`: add the `GetProcAddress` path on Windows
  (1.4), so that `LK_EXPECT_BLAS` and the sweep's `blas_threads` column work there.
- `pylibkriging` holds the GIL in fit / predict / simulate: releasing it would
  make Python threads really concurrent (1.3; not Windows-specific).

## Not done

- MinGW build (Octave toolchain; Rtools45 is installed): no sweep, threshold and
  hang behaviour of libgomp on Windows unmeasured; the Octave job keeps
  `OMP_NUM_THREADS=1`, MinGW keeps threshold 800 and the former small-problem rule.
- R binding build and tests.
- Section 3 (optional): MKL on Windows (`LK_EXPECT_BLAS=mkl` needs 1.4 first),
  Octave package.
- SMT off; several processor groups.
