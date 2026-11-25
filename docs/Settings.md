# Settings

Process-wide settings of libKriging: the optimizer (`Optim`), the threads
(`Parallel`) and the Cholesky factorizations (`LinearAlgebra`). This is the one
table of the `LK_*` environment variables read by the library (#468).

## Rule

- **Every user setting has a setter and a getter**, with the same name and the
  same meaning in C++ and in every binding (table below). The binding API check
  (`.github/workflows/binding-api.yml`) fails when one of them is missing from a
  binding, and its smoke test sets and reads each one back
  (`tools/checks/binding_api/smoke_setters.tsv`).
- **An environment variable only gives the default value.** It is read **once,
  when the library is loaded** (C++ program start, `import pylibkriging`,
  `library(rlibkriging)`, first call of the Octave / MATLAB MEX file, first
  `jlibkriging` call): changing it afterwards (`os.environ`, `Sys.setenv`,
  `setenv`, `ENV`) has no effect in that process.
- **A setter call wins** over the environment, for the rest of the process (or
  until the next setter call). The getter returns the value in effect.
- An unparsable value is ignored (the built-in default is used, without a
  message). Integers are read with `std::stoi` (`"3.7"` gives 3), booleans are
  true for `1`, `true`, `TRUE`, `True` and false for any other value.
- The settings are process-wide, configuration-before-use: set them before
  starting fits from several threads, not while calls run
  (`docs/dev/rationalization/T6-thread-safety.md`). A forked child
  (`fork()`, R's `parallel::mclapply`) inherits the values of its parent, except
  its thread budget (1 thread, see `docs/dev/parallel.md`).
- Variables marked **unsupported** below are diagnostics for the developers:
  no setter, may change or disappear in any release.

## Names in each binding

| C++ | Python | R | Julia | Octave / MATLAB |
|---|---|---|---|---|
| `Optim::set_x(v)` / `Optim::get_x()` | `pylibkriging.Optim.set_x(v)` | `optim_set_x(v)` | `jlibkriging.Optim.set_x(v)` | `Optim.set_x(v)` |
| `Parallel::set_x(v)` | `pylibkriging.Parallel.set_x(v)` | `parallel_set_x(v)` | `jlibkriging.Parallel.set_x(v)` | `Parallel.set_x(v)` |
| `LinearAlgebra::set_x(v)` | `pylibkriging.LinearAlgebra.set_x(v)` | `linalg_set_x(v)` | `jlibkriging.set_x(v)` (exported) | `LinearAlgebra.set_x(v)` |

The thread settings of `Optim` (`set_num_threads_percent`, `get_num_threads`,
`set_thread_pool_size`, `set_thread_start_delay_ms`) are the same settings as
those of `Parallel`, kept for compatibility.

## Settings and environment variables

Kind: **result** — the fitted model may change; **performance** — only the time
changes (up to the rounding of the BLAS); **output** — messages only.

| Environment variable | Setter / getter | Default | Kind | Effect |
|---|---|---|---|---|
| `LK_MAX_RESTART` | `Optim::set_max_restart` / `get_max_restart` | 10 | result | restarts of one optimizer start (Kriging, WarpKriging, MLPKriging) that converged to a bound or to a worse objective than the previous run; 0: none. Setter and getter new in 2.0 (environment only before) |
| `LK_MAX_ITERATION` | `Optim::set_max_iteration` / `get_max_iteration` | 20 | result | iterations of each optimizer run |
| `LK_GRADIENT_TOLERANCE` | `Optim::set_gradient_tolerance` / `get_gradient_tolerance` | 0.001 | result | stopping tolerance on the projected gradient |
| `LK_OBJECTIVE_REL_TOLERANCE` | `Optim::set_objective_rel_tolerance` / `get_objective_rel_tolerance` | 0.001 | result | stopping tolerance on the relative change of the objective |
| `LK_THETA_LOWER_FACTOR` | `Optim::set_theta_lower_factor` / `get_theta_lower_factor` | 0.02 | result | lower bound of the range parameters, times the range of each input |
| `LK_THETA_UPPER_FACTOR` | `Optim::set_theta_upper_factor` / `get_theta_upper_factor` | 10 | result | upper bound of the range parameters, times the range of each input |
| `LK_VARIOGRAM_BOUNDS_HEURISTIC` | `Optim::use_variogram_bounds_heuristic` / `variogram_bounds_heuristic_used` | true | result | narrows the range bounds with the slope of the empirical variogram |
| `LK_REPARAMETRIZE` | `Optim::use_reparametrize` / `is_reparametrized` | true | result | optimize log(theta) instead of theta |
| `LK_CHOL_JITTER` | `LinearAlgebra::set_chol_jitter` / `get_chol_jitter` | -1 (per kernel) | result | jitter on the correlation diagonal of noise-free fits: < 0 per kernel (gauss 1e-9, matern5_2 1e-10, others 0), 0 off, > 0 this value (`docs/dev/CholJitter.md`) |
| `LK_LOG_LEVEL` | `Optim::set_log_level` / `get_log_level` | 0 | output | 0 none, 1 error, 2 warning, 3 info, 4 debug, 5 trace |
| `LK_NUM_THREADS_PERCENT` | `Parallel::set_num_threads_percent` / `get_num_threads_percent` (also `Optim::`) | 0 (100 %) | performance | thread budget: share of the physical cores, x <= 1 a ratio, x > 1 a percentage; `get_num_threads()` gives the resulting count (`docs/dev/parallel.md`) |
| `LK_THREAD_POOL_SIZE` | `Parallel::set_thread_pool_size` / `get_thread_pool_size` (also `Optim::`) | 0 (the budget) | performance | number of multistart workers |
| `LK_THREAD_START_DELAY_MS` | `Parallel::set_thread_start_delay_ms` / `get_thread_start_delay_ms` (also `Optim::`) | 10 | performance | delay of the first start of worker k by k times this, in ms (WarpKriging, MLPKriging multistart) |
| `LK_BLAS_THREADS_MIN_N` | `Parallel::set_blas_threads_min_n` / `get_blas_threads_min_n` | -1 (automatic) | performance | OpenBLAS on 1 thread below this number of observations; 0 off |
| `LK_OMP_MIN_ROWS` | `Parallel::set_omp_min_rows` / `get_omp_min_rows` | build value (200 on Linux / MSVC, 800 elsewhere) | performance | rows from which the covariance and gradient loops run on the OpenMP threads |
| `LK_LINEARALGEBRA_CHECK_TRACE` | none | unset | **unsupported** | any value: `LinearAlgebra.check()` / `linearalgebra_check` prints a line per check and per worker on stderr, to locate a BLAS crash or stall (read at its first call) |
| `LK_ACCELERATE_POOL_SINGLE` | none | unset | **unsupported** | macOS 15+ only, `0`: the workers of a multistart pool keep Accelerate's multithreaded mode (measurements; read at the first multistart) |

Removed in 2.0: `LK_LONG_RANGE_RESTART` and `LK_MAX_LONG_RANGE_RESTART`, with
the long-range restart of a single start (#466); they are ignored. Pass
`optim = "BFGS2"` instead.

Not read by the library, listed here so that they are not mistaken for
settings: `LK_BENCH_WARMUP_MS` and the other `LK_*` variables of the
benchmarks and studies under `bench/`; `LK_OMP_LOOPS`, `LK_TSAN_OPENMP` and
`kOmpMinRows` are compile-time macros / constants of the build, not variables of
the environment (`docs/dev/parallel.md`). The
variables of the BLAS and OpenMP runtimes (`OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, ...) are in `docs/dev/parallel.md`, section 9.

## Examples

```python
import pylibkriging as lk
lk.Optim.set_max_restart(0)
lk.Parallel.set_num_threads_percent(0.5)
lk.LinearAlgebra.set_chol_jitter(0.0)
```

```r
library(rlibkriging)
optim_set_max_restart(0)
parallel_set_num_threads_percent(0.5)
linalg_set_chol_jitter(0)
```

```julia
using jlibkriging
jlibkriging.Optim.set_max_restart(0)
jlibkriging.Parallel.set_num_threads_percent(0.5)
set_chol_jitter(0.0)
```

```matlab
Optim.set_max_restart(0);
Parallel.set_num_threads_percent(0.5);
LinearAlgebra.set_chol_jitter(0);
```

```sh
# the same defaults for one run, without changing the code
LK_MAX_RESTART=0 LK_NUM_THREADS_PERCENT=0.5 LK_CHOL_JITTER=0 python3 my_script.py
```
