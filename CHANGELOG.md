# Changelog

All notable changes to libKriging are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/).

This file was introduced during the 1.x cycle. For the detailed notes of each
past release, see the corresponding entry on the
[GitHub releases page](https://github.com/libKriging/libKriging/releases).

## [2.0.0] - unreleased

First major version since 1.0. The date is set at the release. 2.0 changes defaults, fitted
results and APIs: every such change since 1.2.2 is listed under **Breaking changes** first, and
the migration guide, [docs/Migration-2.0.md](docs/Migration-2.0.md), gives for each one the way back to the 1.x behaviour where
there is one, the code before / after, and which results move. The other sections follow
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); developer-only items (CI, tests,
benchmarks, studies) are in the last section. References: issues and pull requests (`#n`); `T1`
to `T10` are the tasks of the rationalization program (#440,
[docs/dev/Rationalization.md](docs/dev/Rationalization.md)).

### Breaking changes

Compared with 1.2.2 (§ = section of [docs/Migration-2.0.md](docs/Migration-2.0.md)). Several of these
items have **no released deprecation period**: they were the documented 1.2.x API, deprecated
only during the 2.0 development cycle (never in a release), and 1.2.x code using them fails with
2.0 without having seen a warning (marked *no deprecation in 1.2.x*).

*Defaults and fitted results*

- **`Kriging` default optimizer `"BFGS4"`** (was `"BFGS"`; T3, #443): 4 starts in parallel, the
  best kept. Every fit without an explicit `optim` changes. Back: `optim = "BFGS"` (§1).
- **`MLPKriging` default optimizer `"BFGS4"`** (#467, #489; 1.2.2: `"BFGS+Adam"`, then `"BFGS"`
  during the cycle, #382). Back: `optim = "BFGS+Adam"` with the 1.2 Adam knobs, only
  approximately (§2).
- **`WarpKriging` default optimizer `"BFGS"`** (was `"BFGS+Adam"`, #382), and `"BFGS+Adam"`
  itself changed (Adam defaults 200 steps at 3e-2 instead of 10 at 1e-3, inner restarts, early
  stop), as did the joint L-BFGS-B budget, `update(refit = true)` and the multistart draws
  (#382, #418). `NestedKriging` keeps `"BFGS"`. **Known inconsistency**: `WarpKriging` with
  `mlp_joint`, which shares the code path of `MLPKriging`, keeps `"BFGS"` while `MLPKriging` uses
  `"BFGS4"` (maintainer decision 2026-11-27, #464): pass `optim = "BFGS4"` with `mlp_joint` (§3,
  §27).
- **`MLPKriging` / `mlp_joint` predictive variance calibrated** (#407, T8 #448, #489): new in
  2.0, on by default with 5 folds; `var_scale()` is not clamped to >= 1. Predicted stdev,
  `simulate` and `update_simulate` change; the mean and `logLikelihood()` do not. Back:
  `calibration_folds = "0"`. (1.2.2 had no calibration: "5 folds" is not "the 1.x value".) (§4)
- **Search jitter of noise-free `gauss` / `matern5_2` fits** (#415): 1e-9 / 1e-10 on the
  correlation diagonal during the optimization; theta, predictions and `logLikelihood()` move.
  Back: `LinearAlgebra.set_chol_jitter(0)` / `LK_CHOL_JITTER=0` (§5).
- **One jitter rule** (#465, #479): `logLikelihood()` of a `WarpKriging` / `MLPKriging` after
  `update(refit = false)`, or loaded from a file without `objective_ll`, is the fit objective,
  not the committed exact model's log-likelihood (up to 907 units apart). No way back (§6).
- **Exact committed model, variance floor** (#428): the `WarpKriging` / `MLPKriging` model used by
  `predict` / `simulate` / `update` has no 1e-10 on its diagonal any more; the predicted
  variance is floored at 4 eps sigma2 instead of 0 (§7).
- **Noise-free committed model checked by its interpolation residual** (#496, #498) instead of
  the rcond estimate: predictions of near-singular noise-free fits move (mostly better), a few
  nugget fits too. No way back (§8).
- **Restarts no longer cascade** (T2, #442): a restart started because the previous run was
  worse does not trigger another one. A few fits end at a slightly different point (§9).
- **Other changes of the optimization path** (results move, no setting brings 1.2.2 back
  exactly): log-uniform `theta` starts and the `Random` generator for `WarpKriging` (#418), nugget
  fits warm-started from the noise-free fit (#383), bounded `kumaraswamy` and the input-collapse
  penalty of monotone warps (#406), a fixed `theta` (`is_theta_estim = false`) kept fixed with
  `"BFGS*"` (#388), the stable Nystrom factorization (#410) (§10).
- **`sigma2` / `nugget` exactly 0 when `y` lies in the span of the trend** (#450): the fit
  commits the trend with zero variance instead of failing or returning the rounding residual
  (§11).
- **Thread budget** (#415, #446, #460): one budget bounds the multistart pool, the OpenMP loops
  and the BLAS; `Kriging` multistart starts now run in parallel; a forked child (R
  `parallel::mclapply`) gets a budget of 1 thread until `Parallel.set_num_threads_percent` is
  called in it. Times change; results may move in the last bits with the BLAS thread count, as
  for the reordered sums of the gradients and OpenMP loops (T1, #441) (§12).

*API*

- **Thread settings moved from `Optim` to `Parallel`** (#468, #500). Of these, only
  `set/get_thread_pool_size` and `set/get_thread_start_delay_ms` (C++, Python, Octave / MATLAB, R
  `rlibkriging:::optim_*`) were in 1.2.2; the others (`set_num_threads_percent`, ...) were added
  and moved within the cycle. `set_thread_pool_size(n < 0)` now raises (1.2.2: any n <= 0 meant
  the logical CPUs; now 0 means the thread budget). *No deprecation in 1.2.x* (§13).
- **Multistart start delay 0 ms by default** (was 10 ms; #468, #500): results unchanged, small
  multistart fits faster. Back: `Parallel.set_thread_start_delay_ms(10)` (§14).
- **`LK_*` environment variables** (#468, #500): an invalid value is ignored with a `[WARNING]`
  line (1.2.2: silently ignored, or a prefix taken: `LK_MAX_ITERATION=3.7` gave 3); booleans are
  `1/true/yes/on` and `0/false/no/off` (1.2.2: `LK_REPARAMETRIZE=yes` turned it off). They are
  read once, when the library is loaded, and a setter wins; `docs/Settings.md` lists them (§15).
- **Removed, *no deprecation in 1.2.x*** (#469, #499): `objective = "LLVecchia(m)"` /
  `"LLNystrom(k)"` (use `objective = "LL", method = "vecchia(m)"` / `"nystrom(k)"`; a fit with the
  old spelling raises), C++ `Kriging::set_vecchia_exact_commit` / `vecchia_exact_commit`, and R
  `simulate(..., with_nugget = )` (a silent alias of `with_noise` in 1.2.2, now an error). Also
  removed, *deprecated in 1.2.2* (`Base.depwarn`): Julia `get_*` / `is_normalize` (§16).
- **Approximate methods explicit at every step** (#375): `method = "exact" | "vecchia(m)" |
  "nystrom(k)"` in `fit`, `predict`, `simulate`, `logLikelihood`, ...; `predict` after a Nystrom
  fit no longer switches to the approximation by itself (an error names the method); a Vecchia
  fit keeps no exact factorization and its beta / sigma2 come from the Vecchia likelihood (§17).
- **`Kriging` with `normalize = true`: accessors and objective functions in the original scale of
  `X` / `y`** (#379): `theta()`, `sigma2()`, `nugget()`, `noise()`, `beta()`, `covMat()`,
  `logLikelihoodFun(theta)`, ... (§18).
- **Python: vector outputs are 1-D arrays** of shape `(n,)` instead of `(n, 1)` / `(1, d)`
  (`predict` mean / stdev, `theta()`, `beta()`, ...) for `Kriging`, `WarpKriging`, `MLPKriging`
  (§19).
- **`simulate(..., with_noise)` takes noise variances** (was standard deviations) for `Kriging`,
  `WarpKriging` and `MLPKriging`; R: a numeric scalar `with_noise` is a variance, not the logical
  nugget flag (§20).
- **Python `Kriging.simulate` on a nugget model includes the nugget by default** (as R / Julia);
  `with_noise = False` for the latent process (§21).
- **`logLikelihoodFun` without Hessian** in C++ and every binding: Python returns `(ll, grad)`
  (was `(ll, grad, hess)`) and the third positional argument of `Kriging.logLikelihoodFun` is
  `bench`; `WarpKriging` / `MLPKriging` C++ return a pair; Octave / MATLAB two outputs (§22).
- **Errors instead of silent or unclear results**: approximate `leaveOneOut` / `logMargPost` on a
  noise model; `update(y_u, X_u)` / `update_simulate(y_u, X_u)` on a heterogeneous model;
  negative or non-finite noise variances in `fit` (#292); `WarpKriging` noise-argument errors are
  `std::invalid_argument` (was `std::runtime_error`) (§23).
- **Python 3.10 to 3.14** (`python_requires >= 3.10`; 1.2.2 wheels: 3.9 to 3.12, 3.7 on Windows)
  (#386, #389) (§24).
- **Octave / MATLAB: integer results are `int32`** (were `uint64`, #468) (§25).
- **At the version bump (#471)**: shared library `libKriging.so.2` (SONAME = major), CMake package
  `SameMajorVersion` (`find_package(libKriging 1.2)` does not accept 2.0), R package `2.0-0`:
  rebuild what links libKriging (§26).

Not breaking with respect to 1.2.2, although changed during the 2.0 cycle: the long-range restart
of a single start (added by #426, removed by #466 / #485 before any release: `"BFGS"` in 2.0 is the
1.2.2 single start without the cascading restarts), `calibration_folds` (3 then back to 5 during the
cycle, #448, #489) and the `Optim` thread settings other than the four named above.

### Added
- **Settings: one table, one rule, the same setters in every binding** (#468): `docs/Settings.md`
  lists the 15 `LK_*` environment variables read by the library, with the setter / getter of
  each in C++, Python, R, Julia and Octave / MATLAB, its default and its kind (result /
  performance / output). Rule: each setting belongs to one class by domain (`Optim` optimizer,
  `Parallel` threads, `LinearAlgebra` Cholesky), never duplicated; an environment variable only
  gives the default, read once when the library is loaded (changing it afterwards has no
  effect), and a setter call overrides it. The `LK_*` rows of `docs/dev/parallel.md` section 9
  moved there.
- **`Optim::set_max_restart` / `get_max_restart`** in C++ and every binding (Python / Octave
  `Optim.set_max_restart`, R `optim_set_max_restart`, Julia `jlibkriging.Optim.set_max_restart`):
  the restarts of one optimizer start were set by `LK_MAX_RESTART` only (default 10 unchanged;
  0: no restart; a negative value is an error). Migration: none, `LK_MAX_RESTART` still sets the
  default.
- **Thread budget and `Parallel`** (#415, #480, #483): one setting bounds the threads of the multistart pool, the OpenMP loops
  and the BLAS together: `Parallel::set_num_threads_percent(x)` (x <= 1 a ratio, x > 1 a
  percentage of the physical cores, 0 = 100 %) or `LK_NUM_THREADS_PERCENT`;
  `Parallel::get_num_threads()` returns the resolved number of threads. The starts of a multistart
  get the budget first, each start's BLAS / OpenMP the rest; an explicit `OMP_NUM_THREADS` or
  `OPENBLAS_NUM_THREADS` is never exceeded. All the thread settings are in one
  class, `Parallel` (C++ `libKriging/Parallel.hpp`, Python / Octave / MATLAB `Parallel.*`, Julia
  `jlibkriging.Parallel.*`, R `parallel_*`): the budget (`set_num_threads_percent`,
  `get_num_threads`), its distribution (`set_thread_pool_size`, `set_thread_start_delay_ms`,
  `set_blas_threads_min_n`, `set_omp_min_rows`) and the state of each level, read only
  (`blas_backend`, `blas_num_threads`, `omp_enabled`). No separate OpenMP / BLAS thread count.
  See [docs/dev/parallel.md](docs/dev/parallel.md) and [docs/Settings.md](docs/Settings.md).
- **OpenMP threshold at run time**: `kOmpMinRows` stays the build default; `LK_OMP_MIN_ROWS` in
  the environment or `Parallel::set_omp_min_rows` set it without recompiling (a performance
  setting: the results do not depend on it, #441).
- **Small-problem BLAS rule** (OpenBLAS only): `fit`, `predict` and `update` of `Kriging`,
  `WarpKriging` and `MLPKriging` on fewer than 500 training observations run OpenBLAS on
  1 thread instead of the thread budget, restored after the call, when the BLAS threads and the
  OpenMP team (both at the budget) outnumber the logical CPUs (no SMT headroom: VMs, CI runners).
  There the idle OpenBLAS threads spin next to the OpenMP team of the covariance / gradient
  loops: on a 4-vCPU VM, fits at n = 200 are 1.7-3.6x faster (all classes), n = 400 up to 1.8x
  (MLP warping / MLPKriging), with 1.5-5x less CPU time; from n ~ 500-700 on the BLAS threads pay
  off again and are kept. `Parallel::set_blas_threads_min_n(n)` / `LK_BLAS_THREADS_MIN_N`: -1
  automatic (default), 0 off, n > 0 explicit threshold. Every binding: `Parallel.set_blas_threads_min_n`
  (R `parallel_set_blas_threads_min_n`). Not measured on a machine with SMT, where the rule stays off. See
  [REPORT-blas-threads.md](REPORT-blas-threads.md).
- **Linear algebra self-test**: `LinearAlgebra::check(n_threads = 0)` calls the `LinearAlgebra`
  routines libKriging relies on (crossprod, diagABA, Cholesky, triangular / general solves,
  inverse, rcond, QR, Woodbury solve, symmetric eigen) and compares each result with a long
  double oracle computed without the BLAS, whatever the BLAS (OpenBLAS, MKL, Accelerate, BLIS,
  FlexiBLAS, reference), from one thread then from `n_threads` concurrent threads, and returns
  `ok`, the BLAS identification (library libKriging's `dgemm` is bound to, and vendor details) and a report with
  the workaround of a known faulty BLAS (OpenBLAS 0.3.19 / 0.3.20 `Cooperlake` kernels, MKL 2020 with its Intel
  threading layer); the backend is detected, not chosen. Bindings: `linearalgebra_check()`
  (Python `pylibkriging.linearalgebra_check` dict, R exported `linearalgebra_check` list,
  Octave `[ok, backend, report] = linearalgebra_check()`, Julia named tuple). See
  [docs/dev/FAQ.md](docs/dev/FAQ.md) ([#429](https://github.com/libKriging/libKriging/issues/429)).
- **R**: `Kriging`, `NestedKriging`, `WarpKriging` and `MLPKriging` objects survive R
  serialization: `saveRDS()` / `readRDS()`, `save()` / `load()` of a workspace,
  `serialize()`, and shipping to PSOCK / `future` workers. The current model is stored as its
  JSON save file and reloaded into a new C++ object, which the object and its `$` methods
  share ([#327](https://github.com/libKriging/libKriging/issues/327)). Objects serialized by
  an older rlibkriging cannot be restored (use `$save()` / `load.*()`).
- **Bindings**, argument differences found by the binding API check (#419, #423, #431):
  - **Julia**: `fit!(::Kriging, …; parameters=)` and `fit!(::NestedKriging, …; parameters=)`
    (a dict with the keys `sigma2`, `theta`, `beta`, `nugget`, `is_<name>_estim`, as in Python / R;
    C API `lk_kriging_fit` takes the parameters); `simulate(model, X; nsim=1, seed=123, …)` for
    `Kriging`, `WarpKriging`, `MLPKriging` (the defaults of Python / R; `simulate(model, nsim, seed, X)`
    unchanged); `simulate(::Kriging, …; with_noise=true/false)` = `with_nugget` (as in Python / R);
    `bench=` of `log_likelihood_fun`, `leave_one_out_fun`, `log_marg_post_fun` (`Kriging`).
  - **Python**: `Kriging.logLikelihoodFun`, `leaveOneOutFun`, `logMargPostFun` take `theta`,
    `return_grad`, `bench` by name; `WarpKriging.logLikelihoodFun(theta, return_grad)` (was `**kw`);
    `WarpKriging.simulate` without `X` raises `TypeError` (passed `None` down).
  - **R**: `update()` / `update_simulate()` of a `Kriging`: `X_u` has no default in the
    signature (it was `NULL`, then an error); calls unchanged.
- **Bindings**, gaps found by the same check:
  - **Julia**: the `Optim` settings, as `jlibkriging.Optim.<name>` (not exported, not to clash
    with Optim.jl): iterations, tolerances, log level, reparametrization, theta bounds
    (the thread settings are in `jlibkriging.Parallel`) ([#423](https://github.com/libKriging/libKriging/issues/423)); `model(k)` of a
    `Kriging` (a `Dict`, as `model()` in Python / Octave); `objective` of an `MLPKriging`.
  - **Octave**: `objective` of an `MLPKriging`.
  - **R**: `summary()` of a `Kriging` / `NestedKriging` (their description, as printed).
  - **Python, Octave, Julia**: `LinearAlgebra` `set_chol_warning`, `check_chol_rcond`,
    `chol_rcond_checked`, as in R.
  - **All four**: `chol_jitter()` / `model_chol_jitter()` of a fitted `Kriging`, `WarpKriging`,
    `MLPKriging` (jitter of the fit / of the committed model, saved with it; C++ `WarpKriging` /
    `MLPKriging` getters made public) ([#424](https://github.com/libKriging/libKriging/issues/424)).
- **WarpKriging / MLPKriging bindings: `simulate(…, will_update, with_noise)`**
  in Python, R, Julia and Octave / MATLAB, after `will_update` (its existing
  position is kept): `with_noise` holds observation-noise VARIANCES (a
  scalar or one per row of `X`), added as independent N(0, `with_noise`)
  draws to the simulated latent process (and again by `update_simulate`
  after a `will_update=true` simulate). Default (Python `None`, R `NULL`,
  Julia `nothing`, Octave omitted / `[]`): no noise, unchanged. Negative /
  NaN values raise (Python `ValueError`). There is no "fitted nugget"
  shortcut as with `Kriging`: a logical `TRUE` / `True` raises. Julia C
  shim: `lk_warp_kriging_simulate` / `lk_mlp_kriging_simulate` take
  `(…, with_noise, noise_n, will_update, sim_out)`. R:
  `simulate.WarpKriging` / `simulate.MLPKriging` warn about unused
  arguments instead of silently ignoring them (e.g. a `with_noise=` passed
  through `...` was dropped).
- **Kriging: `objective="LOO"` and `"LMP"` with `method="vecchia(m)"` /
  `"nystrom(k)"`** (noise-free models). The LOO criterion is the virtual
  leave-one-out of the approximate joint law (Vecchia precision BᵀD⁻¹B,
  Nystrom precision by Woodbury), O(n m³) / O(n k²); the LMP criterion uses
  log det R = Σ log Dᵢ (Vecchia) or the determinant lemma (Nystrom). Both
  have analytic gradients (finite-difference verified) and are exact for
  m = n − 1 / k = n. Fits commit β (GLS under the approximation) and σ²
  (SSE/n, SSE/(n − p) for LMP) like the exact ones, keep no exact
  factorization, and survive `update` and `save` / `load`; `predictNystrom`
  applies the LMP variance factor (n−p)/(n−p−2) like `predict` /
  `predictVecchia`. New `leaveOneOut(method)`, `logMargPost(method)`, `leaveOneOutVec(theta,
  method)` overloads (C++ also `leaveOneOutVecchiaFun`,
  `leaveOneOutNystromFun`, `logMargPostVecchiaFun`,
  `logMargPostNystromFun`); `method` argument in every binding (Python /
  R / Julia keyword, Octave/Matlab trailing argument). With
  `normalize=true` they follow the original-scale convention: θ in the
  units of X, LOO in squared y units, LMP of the original y, chain-ruled
  gradients, so `leaveOneOut("vecchia") == leaveOneOutVecchiaFun(theta())`
  and they match a `normalize=false` model at the same θ (Vecchia with all
  predecessors as neighbors; Nystrom up to its numerical jitter). See
  docs/math/LOO.md and docs/math/LMP.md.
- `method="nystrom(k)"` with the noise models and `objective="LL"`
  (`noise="nugget"` and a heterogeneous `noise` vector; C++
  `NoiseModel::Nugget` / `NoiseModel::Heterogeneous`; `"LOO"` / `"LMP"` with
  an approximate method still need the noise-free model): the Nystrom correlation C̃ = U Uᵀ + diag(D)
  replaces ρ(θ) in R = α C̃ + (1 − α) I (nugget) or R = C̃ + diag(noise)/σ²
  (heterogeneous), still "rank k + diagonal" (Woodbury, O(n k²)), with the
  same profiling as the exact noise-model likelihoods and an analytic
  gradient in θ and α / σ². `predict` / `simulate` / `logLikelihood` with
  `method="nystrom"`, `update` and `save` / `load` follow the exact
  noise-model conventions (see docs/math/Nystrom.md, "Noise models").
  With `normalize=true` they follow the original-scale convention:
  `logLikelihoodNystromFun` takes an original-scale `[theta, alpha]` /
  `[theta, sigma2]`. `simulate` with `method="nystrom"` draws the latent
  process only: the C++ `simulate(…, with_nugget | with_noise, will_update,
  method)` overloads and the bindings raise for `with_nugget=true` / a
  non-zero `with_noise` with it.
- Kriging: `method="vecchia(m)"` with the noise models (`NoiseModel::Nugget`
  / `noise="nugget"` and `NoiseModel::Heterogeneous` / known noise
  variances), by response Vecchia (Katzfuss & Guinness 2021): Vecchia
  likelihood on the noisy correlation blocks with the exact noise-model
  profiling of sigma2 / nugget, analytic gradient in theta and in alpha /
  sigma2, light commit (no exact factorization), `predict` / `simulate` /
  `logLikelihood` with `method="vecchia"`, `update` and `save` / `load`.
  `predictVecchia` / `simulateVecchia` follow the exact noise-model
  conventions (also after an exact noise-model fit); `simulateVecchia` takes
  `with_nugget` / `with_noise`, and C++ `simulate(nsim, seed, X_n,
  with_nugget | with_noise, will_update, method)` overloads (used by every
  binding) accept `"vecchia"`. New C++ accessor `Kriging::alpha()`
  (scale-free). With `normalize=true` these follow the original-scale
  convention above: `logLikelihoodVecchiaFun` takes an original-scale
  `[theta, alpha]` / `[theta, sigma2]`, `with_noise` (noise variances since
  the breaking change above) is in the scale of `y`, and a Vecchia noise-model fit with all predecessors as
  neighbors reports the same parameters, likelihood, gradient, predictions
  and simulations as a `normalize=false` one (with fewer neighbors they
  differ by design: the neighbors are searched in the normalized inputs).
  Accuracy degrades when the noise is large relative to the signal (latent
  Vecchia is not implemented); `predictVecchia` now also applies the LMP variance rescaling of
  `predict` after an exact LMP fit.
- `Kriging::simulateVecchia(nsim, seed, X_n, m)` (C++): sequential
  (response-first) Vecchia conditional simulation, O(q (n + q) d + q m³)
  instead of O(q³), exact when m ≥ n + q − 1; `simulate(…, method=
  "vecchia")` in every binding (usable after any fit, including a Vecchia
  fit, where the exact `simulate` is not available).
- NestedKriging API parity across bindings: `X`, `y`, `groups`, `warping`,
  `set_predict_chunk` and `set_warp_subsample` in Julia
  (`set_predict_chunk!` / `set_warp_subsample!`) and Octave/Matlab; `fit` on
  an existing object (refit), `set_predict_chunk` and `set_warp_subsample` in
  R. `groups` are 1-based in R, Julia and Octave/Matlab, 0-based in Python.
- `vecchia_neighbors()` in every binding (Python, R, Julia, Octave/Matlab),
  alongside the already exposed `nystrom_rank()`: number `m` of conditioning
  neighbors of a `method="vecchia(m)"` fit, 0 otherwise.
- A Vecchia fit (without exact factorization) supports `update` (Vecchia
  refit) and `save` / `load` (both used to raise on the 1.2 "light" fit);
  the Vecchia sets are rebuilt on demand after `load`, and
  `vecchia_neighbors()` is restored.
- Octave/Matlab: `Kriging(kernel)` / `Kriging(kernel, noise_model)` build an
  unfitted model (fitted later by `fit`), as `Kriging(kernel)` already did
  in Python, R and Julia.
- Python: `pickle` support for `Kriging`, `WarpKriging`, `MLPKriging` and
  `NestedKriging` (through their JSON save/load), so the scikit-learn
  wrappers can be cloned, cached and sent to worker processes.
- `NestedKriging::save` / `NestedKriging::load` (C++, Python, R, Julia,
  Octave/Matlab): configuration, data, partition, common prior and submodels
  (`Kriging` or `WarpKriging`) in one JSON file; the NK precomputations are
  rebuilt at load. `KrigingLoader::describe` reports
  `KrigingType::NestedKriging`, so the generic `load` of each binding
  dispatches to it.
- **Python**: a source distribution (sdist) is published with the wheels, so `pip install
  pylibkriging` builds from source (CMake, a C++17 compiler and BLAS / LAPACK) where no
  wheel matches, instead of failing with "no matching distribution". It is built from the
  new root `MANIFEST.in` / `pyproject.toml`, and its install is checked on every push
  ([#389](https://github.com/libKriging/libKriging/issues/389)). New CMake option
  `LIBKRIGING_BUILD_TESTS` (default `ON`; `OFF` in `setup.py`, which no longer builds the
  C++ tests nor benchmarks, and links armadillo statically).
- **CMake package** for the C++ library (#127): after installation,
  `find_package(libKriging CONFIG)` and `target_link_libraries(app PRIVATE libKriging::Kriging)`
  work, with version checking (`SameMajorVersion`) and transitive dependencies
  (Armadillo, lbfgsb_cpp, OpenMP).
- **MATLAB package in the releases** (#180, #497): `release-matlab.yml` attaches
  `mLibKriging_<version>_Linux-x86_64_MATLAB-R2024b.tgz` (MEX file and `.m` classes) to the
  release, from the release following 1.2.2 on; the MEX file links MATLAB's own BLAS / LAPACK.
  Windows and macOS: build from source.

### Changed
*Defaults and fitted results (see Breaking changes)*
- **Default optimizer of `Kriging`: `"BFGS4"`** (was `"BFGS"`), in C++ and every binding (Python
  `Kriging` / `KrigingRegressor`, R `Kriging()` / `fit()`, Octave / MATLAB, Julia): 4 random
  starts run in parallel, the best kept. On 4080 benchmark data sets (sin, Hartmann6, Branin and
  sinhalf with inactive inputs; gauss / matern5_2; noise-free / nugget; LL / LOO / LMP; n
  100–1000; 30 sets per configuration) a single start ends more than 10 units below the best
  optimum found on 18 % [17, 19] of the sets, `"BFGS4"` on 5 % [4, 6], `"BFGS10"` on 1 % [1, 1]
  (Wilson 95 %); the cost of `"BFGS4"` is x 1.6–3.8 the single start on one core, x 0.3–2.6 its
  wall time on 4 cores (`docs/dev/rationalization/T3-default-multistart.md`). Results of fits
  that relied on the default change; pass `optim = "BFGS"` for the previous behaviour.
  `WarpKriging` and `NestedKriging` keep `"BFGS"`; `MLPKriging`: `"BFGS4"` too (below).
- **MLPKriging: default optimizer `"BFGS4"`** (#467, #489; 1.2.2: `"BFGS+Adam"`; `"BFGS"`, a single start, during the 2.0 cycle, #382): C++
  `MLPKriging` constructor and `fit`, Python `MLPKriging` and `MLPKrigingRegressor`, R
  `MLPKriging()` / `fit.MLPKriging`, Octave / MATLAB, Julia. Measured on the 15-function MLP
  benchmark of T8 (20 designs, tanh and selu, 600 cases; plain `Kriging` 7 %) and on the
  projected-Branin 10D case of #407 (10 designs), `docs/dev/rationalization/mlp-default-467.md`:
  test Q² < 0.5 on 68.5 % of the cases with the single start against 29.3 % with `"BFGS4"`
  (25.3 % with `"BFGS10"`), #407 median Q² −6.0 / −2.5 (tanh / selu) against 0.983 / 0.993, for
  x 1.6 the CPU time of the single start (`"BFGS"`, 5 folds) on one core and x 1.04 its wall time
  on 4 cores (calibration included). Changed results: every `MLPKriging` fit without an explicit
  `optim` (different mean, variance factor and `logLikelihood()`). `WarpKriging` keeps `"BFGS"`
  for every warping, `mlp_joint` included (pass `"BFGS4"` there). `calibration_folds` stays 5
  (see the calibration entry below). Migration: `optim = "BFGS"` gives the single start of the
  2.0 cycle, not the 1.2.2 fit; the closest to 1.2.2 is `optim = "BFGS+Adam"` with
  `max_iter_adam = "10"`, `adam_lr = "0.001"` and `calibration_folds = "0"`
  ([docs/Migration-2.0.md](docs/Migration-2.0.md) §2); `"BFGS10"` for a hard fit. Saved models are loaded unchanged (`update(refit = true)`
  stays a warm-started single start).
- **MLPKriging: `"BFGS"` (joint L-BFGS-B on the MLP weights and θ) replaced `"BFGS+Adam"`
  during the cycle** (#382; superseded by `"BFGS4"`, above) in C++ and every binding: `"BFGS+Adam"`'s 10 Adam steps of size 1e-3 left the network at its random
  initialisation (#382).
- **WarpKriging: the default optimiser is now `"BFGS"`** (#382, C++ and every
  binding), instead of `"BFGS+Adam"`: on every warp tested (Branin 2D,
  kumaraswamy, boxcox, knots, neural_mono, mlp, mlp_joint) the joint
  L-BFGS-B reaches an equal or better log-likelihood than the (now
  effective, see below) bi-level Adam phase, in less time. `"BFGS+Adam"`
  stays available on request. `mlp_joint` included: see the known inconsistency in
  Breaking changes (pass `optim = "BFGS4"` there).
- WarpKriging `"BFGS+Adam"` (#382): the Adam phase now
  actually moves the warp parameters. Defaults `max_iter_adam` 10 → 200 and
  `adam_lr` 1e-3 → 3e-2 (each Adam step moves a parameter by about
  `adam_lr`, so 10 × 1e-3 left the warps at their initialisation), with an
  early stop after 20 steps without a relative log-likelihood gain > 1e-6.
  On Branin 2D (n = 30) the log-likelihood rises e.g. from -131 to -94
  (`mlp`), -120 to -99 (`neural_mono`), -155 to -83 (`mlp_joint`), for
  30–200 ms instead of 5–20 ms. Models saved by earlier versions carry
  `max_iter_adam = 10`, `adam_lr = 0.001`, used as-is after `load`.
- WarpKriging/MLPKriging joint `"BFGS"` (#382): the iteration budget grows with the
  number of optimised parameters (20 per parameter, capped at 1000, never
  below `Optim::max_iteration`) instead of the global `Optim::max_iteration`
  (20), sized for a handful of θ; a run that hits the budget while still
  improving is resumed from the point reached rather than restarted from a
  contracted θ (a joint MLP, `MLPKriging` / `mlp_joint`, keeps `Optim::max_iteration` per call
  instead, #407: see the calibration entry). The `max_iter_bfgs` tuning knob now defaults to 0 (automatic:
  100 for the inner θ L-BFGS of `"BFGS+Adam"`, the budget above for the
  joint L-BFGS-B) and, when set, applies to both (models saved by earlier
  versions carry `max_iter_bfgs = 100`, used as-is after `load`). Fits get slower but reach
  the optimum: ~0.2 s instead of ~5 ms on the #382 case (Branin, n = 30,
  `hidden_dims=[8]`), with log-likelihood -80 instead of -172.
- WarpKriging/MLPKriging `update(refit=true)` (#382) of a model fitted with
  `"BFGS"`/`"BFGSk"` re-runs the joint L-BFGS-B (warm start) instead of
  always using the bi-level `"BFGS+Adam"` path.
- WarpKriging/MLPKriging multistart (`"BFGSk"`, `"BFGSk+Adam"`, #382): starts
  2..k also re-draw the MLP weights (`mlp_joint`, per-dimension `mlp`), not
  only θ; start 1 still uses the current parameters.
- **WarpKriging** / **MLPKriging**, optimisers `"Adam"` / `"BFGS+Adam"`: the first inner
  `theta` L-BFGS-B is restarted as in `Kriging` when it stops abnormally, in <= 2 iterations or
  at the lower bound (a start whose gradient points out of the bounds stopped at their corner,
  LL −329 instead of 831 on a categorical case). The Adam loop no longer stops at its first
  plateau: it goes back to the best point with half the step, twice. mlp warps: LL +167
  (median of 15 data sets), fit time x2; every Adam-optimised warp is up to 1.4x slower. See
  [docs/dev/CholJitter.md](docs/dev/CholJitter.md), Adam phase.
- **MLPKriging / `mlp_joint`: the predictive stdev is calibrated on held-out points** (#407, new in 2.0). With more weights
  than observations, the likelihood-optimal network learns features in which the training
  outputs are almost exactly smooth, and the GP on top was overconfident: on `Branin(Wx)` in
  10D (n = 120, `hidden_dims=[16, 8]`, `BFGS10`) 95 % intervals covered 1–2 % of test points
  (NLPD 3.7e4–9.6e5 vs 2.93 for `Kriging`). After the fit, a K-fold cross-validation (K = `calibration_folds`, 5 by
  default; each fold refitted from the starting point of the winning start, the folds run
  concurrently in the multistart pool) scales σ² for `predict` /
  `simulate` / `update_simulate` (`var_scale()`, reported in `summary()`, saved with the
  model); the mean, `sigma2()` and `logLikelihood()` are unchanged, `calibration_folds = "0"`
  disables it (also a `WarpKriging` tuning parameter for `mlp_joint`). The joint L-BFGS-B of
  a joint MLP keeps `Optim::max_iteration` iterations per call, resumed while it improves,
  instead of up to 1000 (#382): longer runs only overfitted. On that benchmark (4 designs,
  tanh / selu) `BFGS10` takes 2.3–2.5 s instead of 23–50 s, with Q2 0.986–0.998, NLPD
  0.75–2.25 and 95 % coverage 0.95–1.00. A single start still often stops at a poor
  mean in 10D (Q2 < 0; hence the `"BFGS4"` default, #489); its coverage is now 0.84–1.00
  ([#407](https://github.com/libKriging/libKriging/issues/407)).
- **MLPKriging variance calibration: 5 folds, factor not clamped** (T8 #448, #467, #489):
  `calibration_folds` defaults to 5 (the value of the calibration when it was introduced, #407).
  T8 lowered it to 3 for the single-start default (same coverage and test NLPD at x 0.57 the CPU
  time of the calibrated fit); with the `"BFGS4"` default (#467) 3 folds widen the intervals more
  (median test NLPD +0.15, 95 % CI [+0.10, +0.20], 600 paired cases; coverage 0.971 against 0.961)
  for x 0.75 the CPU time, so 5 was restored. Pass `calibration_folds = "3"` to trade interval
  width for fit time. The folds run concurrently (#407): on 12 cores the calibrated fit of the
  #407 case takes 1.3 s with `BFGS+Adam` and 2.9 s with `BFGS` instead of 6.4 s / 12.4 s run one
  after the other. The factor is no longer
  clamped to >= 1 (`var_scale()` = mean squared standardised held-out residual, the
  maximum-likelihood scale; the clamp was never active on the 260 K = 5 fits measured).
  Models saved before keep their number of folds. See
  `docs/dev/rationalization/T8-mlp-calibration-cost.md`.
- **Search jitter: Kriging** / **WarpKriging** / **NestedKriging**, noise-free fits with `gauss` or
  `matern5_2` (#415): a fixed jitter is added to the diagonal of the correlation matrix during the
  optimization (1e-9 for `gauss`, 1e-10 for `matern5_2`, none for `matern3_2` / `exp`). On
  smooth noise-free data the optimizer reaches the near-singular region, where the stepwise
  numerical nugget added when a Cholesky fails made the objective discontinuous: a
  single-start fit ended at a different optimum depending on the BLAS thread count, BLAS build
  or CPU (18 % of 240 `gauss` test cases across thread counts; 1.7 % with the jitter, `theta`
  otherwise equal within the optimizer tolerance). `Kriging` factorizes the model at the
  optimum exactly, so `predict` still interpolates the observations, and `logLikelihood()` stays the
  optimized objective; `WarpKriging` / `MLPKriging` too since #428 (below; 1.2.2: 1e-10 on their
  diagonal). The approximate methods
  (`vecchia(m)`, `nystrom(k)`) use the same jitter for their `LOO` / `LMP` objectives as for `LL`. The jitter is saved with the model (`chol_jitter`,
  `model_chol_jitter`; older files load unchanged). Fitted `theta`, predictions and
  log-likelihoods of these fits change; accuracy is within a few % of the former fits, except
  on very smooth functions (Branin, n = 50: median test RMSE 5e-4 → 0.058).
  `LinearAlgebra::set_chol_jitter(0)` or `LK_CHOL_JITTER=0` restores the former behaviour
  (Python / Octave `LinearAlgebra.set_chol_jitter`, Julia `set_chol_jitter`, R
  `linalg_set_chol_jitter`). See [docs/dev/parallel.md](docs/dev/parallel.md) 8.6.
- **One jitter rule for Kriging, WarpKriging / MLPKriging and NestedKriging** (#465, part of #464):
  the search and the objective accessors (`logLikelihood()`, `leaveOneOut()`, `logMargPost()`) use
  the search diagonal 1 + J (`LinearAlgebra::kernel_chol_jitter`, `LK_CHOL_JITTER`), the committed
  model (`predict`, `simulate`, `update`) is exact, and an accessor always equals its objective
  function at the committed parameters (`logLikelihood() == logLikelihoodFun(theta())`). Kriging
  and NestedKriging already did (no change of their results). WarpKriging / MLPKriging: after
  `update(refit = false)`, and for a file saved without `objective_ll`, `logLikelihood()` returned
  the log-likelihood of the committed exact model, another quantity (MLPKriging gauss /
  matern5_2, n 100-200: up to 907 / 812 units from the objective); it is now the objective there
  too. Fits, predictions and the
  value after `fit` / `update(refit = true)` / `load` are unchanged. Rules (i) "logLikelihood() of
  the exact model" and (ii) "commit with the search jitter" were measured and not kept
  (`docs/dev/CholJitter.md`, one rule for every class). Migration: code that read
  `logLikelihood()` after `update(refit = false)` of a WarpKriging / MLPKriging as the exact
  model's log-likelihood gets the objective's (the one the fit maximizes); the exact model's value
  is not exposed (it is dominated by rounding where R is near singular).
- **Predicted variance floor, exact committed model** (#428): `predict` floors the variance at 4 eps x sigma2 instead of
  clamping it to 0; the stdev is never 0 (no NaN NLPD) and its derivative is 0 (was NaN) where the
  variance is at the floor. The **WarpKriging / MLPKriging committed model is now exact**, as
  Kriging's (it kept 1e-10 on its diagonal).
- **rcond test of the Cholesky retry: the noise-free committed model is checked by its
  interpolation residual** (#496, follow-up of #465). The retry added a nugget when the 1-norm
  rcond estimate of the factor was below 1e-18, about 4 orders of magnitude pessimistic: on the
  exact committed model of Kriging / WarpKriging (gauss, matern5_2, n >= 200) it was a false alarm
  that stopped the model from interpolating (max error up to 4.8e-3 sd(y)). Now the noise-free
  committed model is factorized without the test and kept when its interpolation residual
  max |y - F b - (R + delta I) alpha| / sd(y) (delta: the nugget of a failing Cholesky) is
  <= 1e-5, else factorized again with it (MLPKriging, where the
  learned features make the matrix genuinely ill-conditioned: the protection is kept); the test
  is skipped wherever the diagonal already bounds rcond (search jitter, nugget, noise).
  NestedKriging keeps the rcond test in its own group factors (its NK aggregation is fragile with
  exact factors): NK with one group now differs from its exact submodel by the retry nugget. Measured (`docs/dev/CholJitter.md`,
  `bench/thread-budget/rcond_study.cpp`): noise-free Kriging (n 100-1000) and WarpKriging,
  34 fits of 388 change, RMSE better by more than 10 % on 22 (down to × 0.40), never worse by
  more than 5 %; MLPKriging (n 100-500, 6 functions), 6 fits of 156 change, 3 better by more
  than 10 % (down to × 0.60), none worse by more than 5 %. A Cholesky of the committed model that
  succeeds with its smallest pivot at the rounding level (min L_ii^2 < 4 eps ||R||_1) counts as
  failing: the matrix is numerically singular and the outcome would depend on the BLAS. Results change: predictions of the
  noise-free models above, the fitted theta of Kriging at n 1000 (13 of 40 fits, LL never
  lower), and with a nugget a few fits where 1 - alpha now bounds rcond (3 of 560 moved in the
  study of #465, LL −30 to +5). Not covered: the variance of
  gauss, overconfident (#425), lost the widening of the retry nugget on 5 fits of 444 (coverage
  down to 0.62, WarpKriging kumaraswamy). `LinearAlgebra::set_chol_residual_tol` (C++) sets the
  tolerance; `check_chol_rcond(false)` still removes the rcond test everywhere.
- **Single L-BFGS-B restart policy** (T2, [docs/dev/rationalization/T2-restart-policy.md](docs/dev/rationalization/T2-restart-policy.md)):
  the restart criteria (abnormal stop, <= 2 iterations, at the lower bound, worse than the best
  run beyond factr x epsmch; the long-range criterion added during the cycle was removed by #485) and the contracted restart point, copied in
  `Kriging::fit`, the WarpKriging joint BFGS and `AdamBFGS`, are now one private component
  (`src/lib/RestartPolicy.hpp`). The **"worse" criterion no longer cascades**: a restart started
  only because the previous run was worse does not trigger another one by being worse again
  (before: up to `max_restart` = 10 restarts, each from a point closer to the lower bound). Kriging
  and WarpKriging (joint BFGS) fits that went through such a cascade end with fewer restarts and
  may end at a slightly different point (4 of 1728 measured Kriging fits, |ΔLL| <= 1.1e-5, 12 %
  fewer restarts; 216 WarpKriging fits unchanged). `AdamBFGS::inner_restart` now decides the restart itself
  (it receives the run's result, end and start points; `AdamBFGS::max_restart` is removed).
- **Kriging** / **WarpKriging** multistart: the random starting points of `theta` are drawn
  log-uniformly in their bounds instead of uniformly (`theta` is a scale parameter): with the
  default bounds `[0.02, 10] × max |dX|` a uniform draw put 90 % of the starts in the
  long-range region where the likelihood is flat. The joint MLP of `MLPKriging` / `mlp_joint`
  keeps the uniform draw (long-range starts let its weights adapt). `WarpKriging` now draws
  them from the same generator as `Kriging` (`Random::reset_seed` applies), not Armadillo's.
  The restart of a start whose result is "not better" now ignores differences below the
  optimizer tolerance ([#418](https://github.com/libKriging/libKriging/issues/418)).
- **Kriging (nugget)**: the BFGS fit seeds its first start with the theta of the
  noise-free fit on the same data and a small nugget share (alpha = 0.99), instead of a
  random start that often ended in the all-noise region (nugget >> sigma2) with the
  default single start. On Branin (n = 30, low noise) the mean LL gap to `BFGS10` drops
  from ~8 to ~0.5 ([#383](https://github.com/libKriging/libKriging/issues/383)).
- **`sigma2` / `nugget` exactly 0 when y lies in the span of the trend** (#450, found in #449): the fit
  without optimization committed on such data (e.g. constant y with a constant trend) is documented
  to estimate sigma2 = 0 (and nugget = 0), but stored SSE / n, the rounding residual of the BLAS:
  constant y, gauss, n = 10, `normalize = false`: 2.4e-32 without noise (OpenBLAS and MKL), 2.4e-32 /
  2.4e-35 with a nugget (MKL), 2.0e-22 with `method = "vecchia(5)"`, 2.5e-32 / 2.5e-35 with
  `method = "nystrom(5)"` and a nugget (OpenBLAS). The estimated variances are now set to 0 on this
  path (exact, Vecchia and Nystrom methods). Heterogeneous noise is unchanged (still rejected).
*API (see Breaking changes)*
- **Thread settings of `Optim`, moved to `Parallel`** (#468, #500; listed under Removed for `Optim`): `set_num_threads_percent`,
  `get_num_threads_percent`, `get_num_threads`, `set_thread_pool_size`, `get_thread_pool_size`,
  `set_thread_start_delay_ms`, `get_thread_start_delay_ms` (and in C++ `set_blas_threads_min_n`,
  `get_blas_threads_min_n`, plus the public static members `Optim::num_threads_percent`,
  `thread_pool_size`, `thread_start_delay_ms`, `blas_threads_min_n`). They were the same settings
  as those of `Parallel` (#480). In 1.2.2 only `set/get_thread_pool_size` and
  `set/get_thread_start_delay_ms` existed (C++, Python, Octave / MATLAB, R internal
  `optim_*`); `set_thread_pool_size(n)` then took any n <= 0 as "the logical CPUs", now 0 is the
  thread budget and n < 0 raises. Migration: same names in `Parallel` — C++ `Parallel::<name>`
  (`#include "libKriging/Parallel.hpp"`), Python / Octave / MATLAB `Parallel.<name>`, Julia
  `jlibkriging.Parallel.<name>`, R `parallel_<name>` instead of `optim_<name>`.
- **Start delay of the multistart workers: 0 ms by default (was 10 ms)** (#468,
  `Parallel::set_thread_start_delay_ms`, `LK_THREAD_START_DELAY_MS`). The first start of
  WarpKriging / MLPKriging worker k waited k × 10 ms, against start-up races that no longer
  exist: with reentrant tasks and a deterministic choice of the best start, 800 multistart fits
  (WarpKriging kumaraswamy n 60 / 150, `BFGS8` and `BFGS4+Adam`, MLPKriging `BFGS8`; delays 0 to
  10 ms; pools of 4 and 8 workers on 4 cores) are all bit-identical to the sequential run
  (`bench/thread-budget/start_delay_study.cpp`, OpenBLAS); `MultistartDeterminismTest` checks on
  every CI platform that the fit does not depend on the delay at a given pool size (with MKL it
  does depend on the pool size, through the BLAS thread count of each start). Results unchanged; small multistart fits faster (WarpKriging n 60 `BFGS8`: 26
  instead of 42 ms, 28 instead of 81 ms with 8 workers). Migration: none;
  `set_thread_start_delay_ms(10)` restores the former stagger.
- **Invalid `LK_*` values: a warning instead of silence, and stricter parsing** (#468). A value
  that is not entirely a valid number or boolean, or below the minimum of the setting, is
  ignored with a line `[WARNING] LK_X="..." ignored (expected ...), default ... used` when the
  library is loaded; before, it was silently ignored, and a prefix was accepted (`LK_MAX_ITERATION=3.7`
  gave 3, `"12abc"` gave 12: now the default and a warning). Booleans accept `1/true/yes/on` and
  `0/false/no/off` in any case (before: `1`, `true`, `TRUE`, `True` true, anything else false,
  so `LK_REPARAMETRIZE=yes` turned it off). The thread setters reject invalid values with an
  error: `set_thread_pool_size(n < 0)`, `set_thread_start_delay_ms(ms < 0)`,
  `set_blas_threads_min_n(n < -1)` (before: accepted, a negative pool size meaning automatic).
  Migration: pass 0 for an automatic pool size.
- **Kriging (breaking): exact / Vecchia / Nystrom methods are explicit at
  every step, fit included, never inferred.**
  - `fit` (and the fitting constructors) take a `method`: `"exact"` (the
    default), `"vecchia"` / `"vecchia(m)"` (Vecchia approximation, m
    neighbors, default 30) or `"nystrom"` / `"nystrom(k)"` (rank-k
    Nystrom, default 50). `objective` is now only the criterion (`"LL"`,
    `"LOO"`, `"LMP"`); the approximate methods exist for every objective
    (see Added) without a noise model, and for `"LL"` with the noise models
    too; other combinations raise. `method()` returns the fitted
    method with its explicit size (`"vecchia(30)"`), and is saved / loaded.
    C++ trailing `method` argument of `fit` and of the constructor; Python /
    R / Julia `method=` keyword; Octave/Matlab trailing argument.
  - `objective="LLVecchia(m)"` / `"LLNystrom(k)"` are replaced by
    `objective="LL"` with `method="vecchia(m)"` / `"nystrom(k)"` (the
    former spellings raise, see Removed). Save files of earlier versions
    load with the matching method.
  - A Vecchia fit no longer performs the final exact O(n³) factorization:
    like a Nystrom fit, it keeps theta and the Vecchia-profiled beta /
    sigma2 only, so a large-n Vecchia fit no longer needs an n × n matrix.
    The exact model at the Vecchia (or Nystrom) estimate is a separate fit
    (`method="exact"`, `optim="none"`, fitted theta given). In 1.2, an
    `LLVecchia` fit kept that factorization by default and its beta / sigma2
    came from the exact likelihood; they now come from the Vecchia one.
    Save files of 1.2 keep their stored factorization until the next
    `update`.
  - `predict` and `simulate` used to switch silently to `predictNystrom` /
    `simulateNystrom` after an `LLNystrom` fit. They now take a `method`
    (`"exact"`, the default, `"vecchia"` / `"vecchia(m)"` or `"nystrom"`,
    also the fitted `"nystrom(k)"`): C++ `predict(X, stdev, cov, deriv,
    method)` / `simulate(nsim, seed, X, method, will_update)`. Method
    `"exact"` on a Vecchia or Nystrom fit raises an error naming the method
    to use, instead of answering with an approximation.
    `logLikelihood(method)` gives the Vecchia / Nystrom objective of such a
    fit.
  - Code predicting after an `LLNystrom` / `LLVecchia` fit must pass
    `method="nystrom"` / `"vecchia"`; the scikit-learn `KrigingRegressor`
    `method` parameter applies to `fit`, `predict` and `sample_y`.
  - `set_vecchia_exact_commit` / `vecchia_exact_commit()` no longer exist
    (see Removed): a Vecchia fit is always light.
  - NestedKriging: `method="vecchia(m)"` or `"nystrom(k)"` (after
    `warping`) estimates the common prior with one global Vecchia or Nystrom
    fit (formerly only `objective="LLVecchia(m)"`); `objective()` and
    `method()` accessors. WarpKriging and MLPKriging accept `method` for a
    uniform API (only `"exact"` is implemented).
- **Kriging (breaking): with `normalize=true`, every parameter accessor and
  objective function is in the ORIGINAL scale of `X` and `y`**, like the
  `parameters` given to `fit` (which already were), so that input and output
  round-trip and a `normalize=true` model reports the same values as a
  `normalize=false` one on the same data:
  - `theta()` = internal theta × `scaleX`; `sigma2()`, `nugget()` and the
    heterogeneous `noise()` = internal value × `scaleY²`; `beta()` = the
    coefficients `b` such that `F(x) b = scaleY · F(x_n) b_n + centerY`
    (exact affine basis change for every trend: constant, linear,
    interactive and quadratic; the `"none"` trend has no coefficient, its
    implicit mean is `centerY`); `covMat()` is scaled by `scaleY²`;
    `summary()` prints these values. Previously they were the internal,
    normalized ones (e.g. a fixed `theta = [0.3, 0.5]` read back as
    `[0.3096, 0.5112]`).
  - `logLikelihoodFun`, `leaveOneOutFun`, `logMargPostFun`,
    `logLikelihoodVecchiaFun`, `logLikelihoodNystromFun` and
    `leaveOneOutVec` take an original-scale `theta` (and an original-scale
    `sigma2` as the last component with a heterogeneous noise; the nugget
    `alpha` is scale-free) and return original-scale gradients (chain rule,
    `d/dθ_i = d/dθ_n,i / scaleX_i`). Their values are those of the original
    data: log-likelihood (all methods) − n·log(scaleY) relative to the
    normalized one, leave-one-out error × scaleY², log marginal posterior
    − (n−p)·log(scaleY) + log|det A| (A the trend basis change), so they
    match a `normalize=false` model at the same parameters, and
    `logLikelihood()` == `logLikelihoodFun(theta())` (same for LOO / LMP).
    `leaveOneOutVec` returns mean / stdev in the scale of `y`.
  - Fixed: a fixed `parameters.beta` with `normalize=true` was only divided
    by `scaleY` (ignoring `centerY` and, for non-constant trends, the `X`
    normalization); it now uses the exact inverse transform (a wrong length
    raises `std::invalid_argument`). The heterogeneous `noise` given to `fit`
    was not normalized (`R = C + diag(noise / sigma2)` mixed an original-scale
    noise with a normalized `sigma2`, which could even crash the fit); it is
    now stored as `noise / scaleY²`, consistently with `update` /
    `update_simulate`. A given heterogeneous-noise `sigma2` starting value is
    now normalized by `scaleY²` (was `scaleY`).
  - `normalize=false` is unchanged, bit for bit. The save format is
    unchanged (internal, normalized members); a heterogeneous-noise model
    saved by an earlier version with `normalize=true` loads with the same
    predictions, but its `noise()` now reports the stored noise × `scaleY²`.
    `X()`, `y()`, `F()`, `T()`, `M()`, `z()` (and the matching `model()`
    entries) still expose the internal normalized data and factorization.
  - Out of scope: `WarpKriging` / `MLPKriging` (their `theta` lives in the
    warped feature space) are unchanged; `NestedKriging` has no `normalize`.
- **Python (breaking):** every vector-valued output of `Kriging`,
  `WarpKriging` and `MLPKriging` is now a 1-D numpy array of shape `(n,)`
  instead of `(n, 1)` (or `(1, d)` for `centerX` / `scaleX`): `predict` mean
  and stdev, `theta`, `beta`, `y`, `z`, `noise`, `warp_params`, gradients of
  `logLikelihoodFun` / `leaveOneOutFun` / `logMargPostFun`, `leaveOneOutVec`,
  `subsetOfData` indices and the matching `model()` entries. This matches
  `NestedKriging` and scikit-learn; matrices (`X`, `F`, `T`, `M`, covariances,
  derivatives, simulations) are unchanged. Code indexing `[:, 0]` on those
  outputs must drop the index; `ravel()` / `flatten()` / `reshape(...)` keep
  working.
- **Kriging (breaking): `simulate(…, with_noise, …)` takes noise VARIANCES**,
  consistent with the `noise` of `fit` / `update(y_u, noise_u, X_u)` /
  `update_simulate(y_u, noise_u, X_u)` (which already were variances):
  the exact `simulate`, `simulateVecchia` / `simulate(…, method="vecchia")`
  and the noise `update_simulate` re-adds now draw N(0, `with_noise`)
  observation noise, where they used `with_noise` as a standard deviation.
  Existing code passing standard deviations must pass their squares. Negative
  (or NaN) values raise `std::invalid_argument`. Applies to every binding
  (Python `with_noise=` array, R numeric `with_noise`, Julia `with_noise=`,
  Octave / MATLAB `simulate(nsim, seed, X_n, with_noise, …)`). An empty
  `with_noise` in the exact path no longer adds an uninitialized matrix.
  R: a numeric scalar `with_noise` (e.g. `0.3`) is now a noise variance for
  every point; it was taken as the logical nugget flag `TRUE`.
  Same for **WarpKriging and MLPKriging**: `simulate(nsim, seed, X_n,
  will_update, with_noise)` and the `update_simulate` re-add take noise
  VARIANCES (like `WarpKriging`'s fit `noise` / `noise_u`), where the C++
  `WarpKriging` used standard deviations; negative / NaN values raise
  `std::invalid_argument` (was `std::runtime_error` for negative values).
  `MLPKriging::simulate` gains the optional trailing `with_noise` argument
  (forwarded to `WarpKriging`). The bindings now expose `with_noise` for
  both classes with these semantics (see Added). `NestedKriging` has no
  `simulate`.
- **Python (breaking): `Kriging.simulate` on a nugget model**: without `with_noise`, the draws include the nugget
  (as in R / Julia; Octave takes `with_nugget` explicitly), except with `method="nystrom"` (latent
  process only). Before, Python simulated the latent process: pass `with_noise=False` for it.
- **Kriging: LOO / LMP with an approximate method reject noise models.**
  `leaveOneOut(method)`, `logMargPost(method)`, `leaveOneOutVec(theta,
  method)` with `"vecchia"` / `"nystrom"`, and `leaveOneOutVecchiaFun`,
  `leaveOneOutNystromFun`, `logMargPostVecchiaFun`, `logMargPostNystromFun`
  raise `std::invalid_argument` (Python `ValueError`, R / Julia / Octave
  error; same wording as the `fit` check) on a Nugget / Heterogeneous model,
  where they silently returned the noise-free approximate criterion. The
  exact `leaveOneOut()` / `logMargPost()` and their `*Fun` are unchanged
  (they use the noisy covariance at the committed nugget / noise).
- **Kriging: `update(y_u, X_u)` on a Heterogeneous model** raises
  `std::invalid_argument("heterogeneous noise model: use update(y_u,
  noise_u, X_u)")` before any state change (Python `ValueError`, R / Julia /
  Octave error), for every method; the exact path used to fail with an
  unclear dimension error after extending the data.
- **Kriging (breaking): `update_simulate(y_u, X_u)` on a Heterogeneous model**
  raises `std::invalid_argument("heterogeneous noise model: use
  update_simulate(y_u, noise_u, X_u)")` before any state change (Python
  `ValueError`, R / Julia / Octave error), as `update(y_u, X_u)`; it used to
  condition on the update points as noise-free observations of the latent
  process. Use `update_simulate(y_u, noise_u, X_u)` (Octave / MATLAB: the
  4-argument form). Nugget and None models are unchanged. Likewise,
  `WarpKriging::update_simulate` / `update` on a model fitted with `noise`
  and an empty `noise_u` raise `std::invalid_argument` ("heterogeneous noise
  model: use update_simulate(y_u, X_u, noise_u)" / "use update(y_u, X_u,
  refit, noise_u)"), where they raised `std::runtime_error`; a `noise_u` of
  the wrong length or with negative values still raises `std::runtime_error`.
- **Kriging: heterogeneous noise variances checked** (#292): `fit(y, noise, X, ...)` raises
  `std::invalid_argument("noise variances must be finite and non-negative")` (Python
  `ValueError`, an error elsewhere) on a negative, NaN or infinite variance, which 1.2.2 accepted.
- **Python**: supported CPython versions are 3.10 to 3.14: the CI tests each of them
  (3.10 and 3.14 on Linux and Windows), the release builds wheels for all of them, and
  `python_requires` is now `>=3.10` (the end-of-life 3.7 and 3.9 jobs are dropped)
  ([#386](https://github.com/libKriging/libKriging/issues/386),
  [#389](https://github.com/libKriging/libKriging/issues/389)).
*Performance and threads*
- **Multistart memory** (#444): a `Kriging` BFGS multistart no longer allocates and keeps the
  n x n model (R, L, R^-1, ...) of every start, nor deep-copies it into a result per start: each
  start builds its model in its task (at most min(multistart, pool size) alive at once) and only
  the best start's model is kept, moved (not copied). `WarpKriging` multistart clones no longer
  carry the GP state (R, Cholesky, R^-1, features, dX) outside their running start, and only the
  best start's state is kept (moved). Fitted results are bit-identical; peak RSS at n = 2000,
  BFGS20 (4 cores): 3369 → 1501 MB; n = 1000, BFGS10: 564 → 330 MB; WarpKriging n = 1000,
  BFGS10: 1112 → 578 MB. See
  [docs/dev/rationalization/T4-kmodel-per-worker.md](docs/dev/rationalization/T4-kmodel-per-worker.md).
- **Faster theta gradients** (T1, `docs/dev/rationalization/T1-vectorized-gradients.md`): the
  LL and LMP theta gradients use per-dimension kernel factors in one pair loop (no
  `std::function` / `arma::vec` per pair), and the LMP gradient is reduced algebraically to one
  R^-1 and an O(d n^2) loop instead of d O(n^3) solves; `LinearAlgebra::inv_sympd` uses LAPACK
  `*potri` (x3 fewer flops). Single thread, d = 4, n = 500 / 1500: LL gradient x2.2-3.7, LMP
  gradient x10-15. The pair sums now have one order whatever the thread count, n and OS (the
  `kOmpMinRows` dependence of the gradient is removed). Results change at rounding level
  (< 3e-12 relative for cond(R) < 1e8; more on ill-conditioned R).
- **Kriging** / **WarpKriging**: from n = 200 on Linux (n = 800 on macOS and Windows, not
  measured there yet: #416, #417), the covariance matrix of the exact
  objectives (and its new rows in `update`) is built in an OpenMP loop instead of one thread,
  and the θ part of the log-likelihood gradient too, with per-row partial sums added in a
  fixed order. Neither depends on the OpenMP thread count. 4 cores, d = 4, OpenBLAS and
  OpenMP at 4 threads, log-likelihood + gradient: 75 → 54 ms at n = 800, 364 → 215 ms at
  n = 1500. Without the thread budget the OpenMP team competed with the spinning OpenBLAS
  threads below n = 800 and was slower than one thread (3.9 → 17 ms at n = 200); within the
  budget (both at the physical cores) the loops pay off from n = 200 (x3 at n = 200-500 on
  10 cores). Each worker of a
  parallel multistart gets its share of OpenMP threads, as of BLAS threads. The parallel
  gradient sum differs from the sequential one: on an ill-conditioned covariance matrix the
  gradient, hence the optimum reached, may change from that threshold on (see
  [docs/dev/FAQ.md](docs/dev/FAQ.md)).
- **Python: the GIL is released during fit / predict / simulate / update** (#473): the
  `Kriging`, `WarpKriging`, `MLPKriging` and `NestedKriging` methods convert their arguments
  with the GIL, run the libKriging call without it and convert the results with it again
  (`call_guard.hpp`). Calls on different models from Python threads now run at the same time
  and share the thread budget (docs/dev/parallel.md 3.1); calls on the same model object are
  serialized by a per-object mutex (the GIL did it before). New `gil_release_test.py`.
- **Objective evaluations run at the thread budget** (#473): the public objective functions of
  `Kriging` (`logLikelihood()`, `leaveOneOut()`, `logMargPost()`, with or without a method, their
  `*Fun`, `*VecchiaFun`, `*NystromFun`, `leaveOneOutVec`) and `WarpKriging::logLikelihoodFun`
  (so `MLPKriging`'s) open the same budget scope as fit / predict / simulate / update: OpenBLAS
  at the budget (or 1 thread below `Parallel::blas_threads_min_n`), MKL per thread, restored after.
  Before, they ran at whatever count OpenBLAS had: the logical CPUs it was loaded with, or the
  count another thread's call had set, so the same call on the same model could round
  differently depending on the concurrent calls (1–2 ulp on `leaveOneOut()`,
  `ConcurrentCallsTest` budget-1 case on Windows with OpenBLAS 0.3.33 loaded with 20 threads).
  Migration: none; values may change in the last bits with a multithreaded OpenBLAS.
- **Thread budget of 1 in forked children, whatever the OpenMP runtime** (#420): the
  `pthread_atfork` child handler caps the budget at 1 (a plain store, no call into the OpenMP
  runtime) until `Parallel::set_num_threads_percent` is called in the child. Before, only the
  OpenMP count bounded it there: builds without OpenMP (R on macOS, the "without OpenMP" CI
  builds) ran the child's multistart pool on all the physical cores. The C++ and R fork tests
  check the budget in the child, before and after a fit.
- **Concurrent calls from user threads** (T6, #446): the save / restore of the process-global
  OpenBLAS thread count is shared by all threads under a mutex (count in effect = smallest live
  request, the last scope restores the caller's count); concurrent public calls split the thread
  budget (budget / running calls) instead of each taking all of it; `Random`'s engine and seed
  are per thread (`reset_seed` applies to the calling thread; a new thread starts from seed 123);
  the fork handlers hold the library's mutexes, so a `fork()` made while another thread fits no
  longer deadlocks the child. `Optim` / `LinearAlgebra` settings are documented as
  configuration-before-use. See [docs/dev/parallel.md](docs/dev/parallel.md), section 3.1.
- **MKL thread control** (#445, T5): with Intel MKL / oneMKL, each multistart worker sets its own
  MKL thread count to its share of the budget (`MKL_Set_Num_Threads_Local`), and each public
  method the calling thread's count to the budget, restored after (before: every worker kept
  all MKL threads). Apple Accelerate, FlexiBLAS and BLIS are still not driven (documented in
  the FAQ: `VECLIB_MAXIMUM_THREADS=1` etc. for multistart fits).
- **cgroup v1 CPU quotas** (#445, T5): the thread budget also reads `cpu.cfs_quota_us` /
  `cpu.cfs_period_us`, and the quotas of the parent cgroups (v1 and v2).
- **macOS: multistart fits ×1.4-1.9 faster, with Accelerate** (#491, #490): the workers of a
  multistart pool (`BFGS2`, `BFGS4`, `BFGS10`, …) run Accelerate single-threaded by themselves
  (`BLASSetThreading`, macOS 15+, per thread), where their own threads and Accelerate's used to
  oversubscribe the cores: on an Apple M4, `BFGS4` fits take 0.53-0.64 of the time at n = 500-1500
  with 3-4x less CPU, the same fits bit for bit; single-start fits and the caller's thread keep
  Accelerate's threads. This is what `VECLIB_MAXIMUM_THREADS=1` did (still the way on macOS < 15,
  FAQ); always on since #468 (the `LK_ACCELERATE_POOL_SINGLE=0` switch of the measurements was
  removed). `kOmpMinRows` is also 200 on Apple (libomp
  builds; the default macOS build has no OpenMP loops): ×1.6-1.8 per evaluation at n = 200-400.
- **OpenMP threshold 200 with MSVC on Windows** (#473): `kOmpMinRows` (covariance matrix and
  log-likelihood gradient loops) goes from 800 to 200 in MSVC builds, as on Linux. Measured on an
  i9-10900K (10 cores / 20 threads, `vcomp`, OpenBLAS 0.3.33) at the thread budget: one LL +
  gradient evaluation ×1.6 faster at n = 200 and ×1.5 at n = 400, full fits ×1.20 (`BFGS`) /
  ×1.12 (`BFGS4`) at n = 500 for ×1.5–2.3 more CPU time, unchanged at n = 2000; the fitted model is
  bit-identical (the gradients sum in one order whatever the threshold, #441). No hang with the C++
  and Python suites without `OMP_NUM_THREADS=1`. MinGW builds (Octave, R on Windows) and macOS
  keep 800 until measured. Migration: none.
- **One OpenBLAS thread below 500 observations with MSVC on Windows** (#473): the automatic
  small-problem rule (`Parallel::blas_threads_min_n` = -1, #436) applies in MSVC builds even when the
  machine has SMT headroom (on Linux it stays off there: the BLAS at the budget was faster). Measured
  on an i9-10900K (10 cores / 20 threads, OpenBLAS 0.3.33): full fits at n = 200 / 300 / 400 ×1.59 /
  ×1.51 / ×1.21 faster with `BFGS`, ×1.10 / ×1.08 / ×1.08 with `BFGS4`, with less CPU time and the
  same optimum (to 1e-9 in LL, except one `BFGS` path at n = 200); per evaluation ×1.4–1.5 at
  n = 200–400. Migration: `Parallel::set_blas_threads_min_n(0)` or `LK_BLAS_THREADS_MIN_N=0` restores
  the former behaviour; results may change in the last bits (the BLAS count changes).
- **Thread budget under a CPU affinity mask on Windows** (#473): the physical cores are counted
  among the CPUs of the process affinity mask (`start /affinity`, job objects,
  `SetProcessAffinityMask`), as on Linux. Before, every core of the machine counted: a process
  restricted to one core of a 10-core i9-10900K got a budget of 10 threads. A process spread over
  several processor groups still counts every core. New test `AffinityTest` (Windows, Linux).
- **R: global settings exported**, as in the other bindings: the `Optim` settings
  (`optim_set_max_iteration`, `optim_set_max_restart`, ...), the `Parallel` settings
  (`parallel_set_num_threads_percent`, `parallel_get_num_threads`, `parallel_set_thread_pool_size`,
  ...) and the `LinearAlgebra` settings (`linalg_set_chol_jitter`,
  `linalg_set_num_nugget`, `linalg_set_chol_warning`, `linalg_check_chol_rcond`, ...) are
  exported from `rlibkriging` instead of being reached through `rlibkriging:::` (which still
  works). Documented in `R/settings.R`; the other `linalg_*` and the `random_*` helpers stay
  internal, as in Python.

### Removed
- **Objectives `"LLVecchia(m)"` / `"LLNystrom(k)"` and the Vecchia exact-commit setter** (#469).
  The objective aliases were the 1.2 way to ask for an approximate fit, deprecated during the
  2.0 cycle and never deprecated in a release: 1.2.x code using them fails with 2.0. A fit with
  one of them now raises (C++ `std::invalid_argument`, Python `ValueError`, R / Julia /
  Octave / Matlab error) with a message naming the replacement, e.g. `objective "LLVecchia(7)"
  was removed in libKriging 2.0; use objective="LL", method="vecchia(7)"`; never reinterpreted,
  whatever `method` is given. Save files of libKriging <= 1.2 storing such an objective still
  load (as `method="vecchia(m)"` / `"nystrom(k)"`). Migration:
  - `objective="LLVecchia(m)"` (or `"LLVecchia"`) → `objective="LL", method="vecchia(m)"`
    (`method="vecchia"`, m = 30); the same in `NestedKriging`.
  - `objective="LLNystrom(k)"` (or `"LLNystrom"`) → `objective="LL", method="nystrom(k)"`
    (`method="nystrom"`, k = 50).
  - C++ `Kriging::set_vecchia_exact_commit(bool)` / `vecchia_exact_commit()` → nothing to call
    for `false` (a Vecchia fit is always light); for the exact model at the Vecchia estimate,
    a separate fit with `method="exact"`, `optim="none"` and the fitted theta.
  - C++ `Kriging::set_deprecation_handler` / `Kriging::deprecated_objective_message` (and the
    `DeprecationHandler` type), R `kriging_deprecation_message` (internal), Julia C symbol
    `lk_deprecated_objective_message`: no replacement, no deprecated objective is left to
    report. The native deprecation warnings of the bindings for these objectives (Python
    `DeprecationWarning`, R `deprecatedWarning`, Julia `Base.depwarn`, Octave / Matlab
    `mLibKriging:deprecated`) become the errors above: code catching or silencing those
    warnings must use the new spelling.
- **Julia `get_*` / `is_normalize` getter aliases and the R `simulate(..., with_nugget = )`
  alias** (#469). Migration:
  - Julia (deprecated with `Base.depwarn` in 1.2.2) `get_X`, `get_y`, `get_theta`, `get_sigma2`, `get_beta`, `get_nugget`, `get_noise`,
    `get_centerX`, `get_scaleX`, `get_centerY`, `get_scaleY`, `get_F`, `get_T`, `get_M`,
    `get_z` (Kriging), `get_warping` (WarpKriging), `get_hidden_dims` (MLPKriging) → the same
    name without `get_` (`X(k)`, `theta(k)`, `warping(wk)`, `hidden_dims(mk)`, ...), already
    exported; `is_normalize(k)` → `normalize(k)`. The old names are no longer defined
    (`UndefVarError`). `get_chol_jitter` / `get_num_nugget` (global settings) are unchanged.
  - R `simulate(k, ..., with_nugget = b)` (silent alias of the former `NuggetKriging` class in
    1.2.2, no deprecation warning) →
    `with_noise = b`; `with_nugget` now raises an error naming `with_noise` instead of being
    taken as `with_noise`. The Julia `with_nugget` keyword of `simulate` (the C++ argument
    name) is unchanged.
- **Thread settings of `Optim`** (#468): moved to `Parallel`, see Changed.
- **Log-likelihood Hessian** (breaking): `logLikelihoodFun` has no `return_hess` argument
  nor Hessian output any more, in C++ and every binding; it was never computed (an empty matrix,
  since the analytic Hessian of the former Newton optimizer was dropped).
  - C++: `WarpKriging::logLikelihoodFun(theta, return_grad)` and `MLPKriging::logLikelihoodFun`
    return `(ll, grad)` (was `(ll, grad, hess)`); `Kriging::logLikelihoodFun(theta, return_grad,
    bench)` unchanged; the unused `Optim::reparam_from_deriv2` is removed.
  - Python: `logLikelihoodFun` returns `(ll, grad)` (was `(ll, grad, hess)`), arguments
    `(theta, return_grad=False, bench=False)` for `Kriging` (`want_hess` removed: a third positional
    argument is now `bench`), `(theta, return_grad=False)` for `WarpKriging` / `MLPKriging`.
  - R: no `return_hess` (`logLikelihoodHess` / `hessian` no longer returned).
  - Octave: `[ll, llgrad] = logLikelihoodFun(theta, return_grad[, bench])` (`Kriging`),
    `logLikelihoodFun(theta, return_grad)` (`WarpKriging` / `MLPKriging`); a third output raises.
  - Julia: no `return_hess`; `log_likelihood_fun` returns `(ll=, grad=)`; C API
    `lk_*_log_likelihood_fun` without `return_hess` / `hess_out`.

### Fixed
- **Octave / MATLAB: integer getters returned a wrong value for a negative number** (#468):
  an `int` result was a `uint64` array with only its low 4 bytes written, so
  `Parallel.get_blas_threads_min_n()` returned 4294967295 instead of -1 (automatic) and could not
  be restored with its setter. Integer results are now `int32` (they were `uint64`): code that
  tested `isa(x, 'uint64')` must test `isinteger(x)`.
- **MATLAB: `"..."` strings accepted everywhere** (#180): a MATLAB string (what a double-quoted
  literal is in MATLAB) in a cell (`NestedKriging` warping), in a parameter struct (`WarpKriging`,
  `MLPKriging`) or as a categorical label of `WarpKriging` was rejected; only char arrays
  (`'...'`) worked. Octave is unchanged.
- **macOS: forked workers** (R `parallel::mclapply`, #455, #459, #487): libomp's own fork handlers
  deadlocked the library's (no `omp_*` call in the handlers any more), and a forked child crashed
  (SIGSEGV) in its first large Accelerate call, then in its first `BFGS4` fit: Accelerate runs
  single-threaded in the child, in its forking thread and in the pool threads it creates
  (the mode is per thread). `cmake_package` also finds a Homebrew libomp (`OpenMP_ROOT`, documented).
- **Fork and BLAS-backend fixes** (T9, #449), found by running MKL in CI: the fork prepare handler
  initializes the lazy statics of the call path (a `fork()` during another thread's first call
  deadlocked the child with MKL), the fork child sets MKL to 1 thread (sigma2 = nugget = 0 on
  degenerate data: see Changed, #450), and
  `LinearAlgebra::inv_sympd` checks its LAPACK `potri` result in O(n^2) against triangular
  solves and falls back to them (a wrong `R^-1` gave a LL gradient off by 7e6 relative on a
  macOS Accelerate runner, #452). Known limitation:
  with MKL, `fork()` while other threads run MKL needs `MKL_DISABLE_FAST_MM=1`
  ([docs/dev/parallel.md](docs/dev/parallel.md), section 7).
- **Verbose output of the Kriging multistart** (#420): the log lines of the concurrent starts
  (`Optim::set_log_level` > 0) are written under a mutex; Armadillo's `operator<<` changes the
  stream's format flags, a data race between pool workers found by the Clang + Archer TSan job
  (the GCC TSan job runs on 1 thread, i.e. a pool of 1 worker, and does not see it).
- **Octave**: `Optim.set_log_level`, `set_max_iteration` and the thread settings
  (`set_thread_start_delay_ms`, `set_thread_pool_size`, now in `Parallel`) failed with "is not an unsigned 64bits int" when given a plain
  number (`Optim.set_log_level(2)`); they now convert it to `int32`. Their tests
  (`test_new_features.m`) were not registered in CMake and never ran; they now do.
- **Kriging** multistart `"BFGS<k>"`: in 1.2.2 the starts ran one after the other in the calling
  thread, each with only ncpu/k BLAS/OpenMP threads (1 thread from k ≥ ncpu on), leaving the
  other cores idle. They now run in the multistart pool and share the thread budget (#415,
  #444).
  With a fixed seed the fit stays bit-reproducible for a given BLAS thread count, but its
  result may change with that count (as it already did with the number of cores): see
  [docs/dev/FAQ.md](docs/dev/FAQ.md).
- BLAS/OpenMP thread counts are restored after a fit instead of being left at 1 thread: a
  `Kriging` multistart or a parallel `WarpKriging` / `MLPKriging` multistart made every later
  `predict`, `simulate` or `update` run single-threaded. The parent process of a `fork()`
  (R `parallel::mclapply`) also gets its thread counts back once the fork returns; the
  children still start single-threaded (#325).
- WarpKriging/MLPKriging parallel multistart: each worker now gets
  `ncpu / workers` BLAS/OpenMP threads (as `Kriging` already did) instead of
  all of them; the nested thread teams contended (`"BFGS4"`, `mlp_joint`,
  n = 500: 19.6 s → 10.1 s, system time 24 s → 1 s). The staggered start
  delay (`Parallel::thread_start_delay_ms`, 0 by default since #468) applies only to
  each worker's first start instead of every start (`"BFGS10"`, n = 30, 10 ms delay: 175 ms →
  43 ms).
- WarpKriging/MLPKriging joint `"BFGS"`: when the objective cannot be
  evaluated (correlation matrix not factorisable even with the numerical
  nugget), the fit keeps the best point evaluated so far instead of throwing.
- WarpKriging: intermittent heap corruption (`malloc(): corrupted top size`)
  and degraded optimum in the parallel multistart of `fit()`
  (`BFGS<k>`, `BFGS<k>+Adam`). Concurrent L-BFGS-B runs raced on the `static`
  locals of the f2c BLAS/LINPACK helpers of `lbfgsb_cpp`; the submodule is
  bumped to `b6eff96`, which makes them reentrant. The multistart thread
  joiner is also constructed before the threads are spawned, so they are
  joined even if a later spawn throws.
- Python: deadlock in code paths where libKriging allocates from worker
  threads while the calling Python thread holds the GIL: NestedKriging NK
  `predict` (OpenMP pair loop) and WarpKriging parallel multistart
  (`BFGS<k>`). The numpy allocator handed to libKriging reports to
  tracemalloc, which takes the GIL; it is now used only from threads holding
  the GIL, other threads use `std::malloc` (each block is freed by the C
  runtime that allocated it). The NestedKriging Python tests are now
  registered in CTest, with a subprocess/timeout regression test.
- **Nystrom** (`method="nystrom(k)"`): the log-likelihood, predicted mean and LOO / LMP
  objectives no longer depend on rounding, i.e. on the CPU / BLAS kernel. At the landmark
  rows the residual diagonal is at the `num_nugget` level (1e-10), and the Woodbury solve
  weighted those rows by 1e10 through a Gram matrix squaring their conditioning: with
  k = n the log-likelihood was off the exact one by 1e-7 to 4e-6, and with an
  ill-conditioned landmark block (smooth Gaussian kernel, k = 10) a 1-ulp perturbation of
  `X` moved the predicted mean by ~60 % of sd(y). The rank-k + diagonal matrix is now
  factorized by block Cholesky with the landmark rows first (QR, no Gram matrix), and the
  landmark diagonal is set to `num_nugget` instead of computed as 1 − (≈1): k = n now
  reproduces the exact log-likelihood and mean to ~1e-12, and the 1-ulp sensitivity drops to
  ~1e-6. Cost: Nystrom `predict` +30–40 %, log-likelihood and gradient +10–20 %
  ([#410](https://github.com/libKriging/libKriging/issues/410)).
- **WarpKriging (monotone warps)**: the fit no longer reaches degenerate warps squeezing a
  sub-range of the training inputs onto one point, where the noise-free likelihood is
  unbounded (Gramacy & Lee 2008, `kumaraswamy`: logLikelihood 207 vs 50 for `Kriging`, Q2
  -0.25, min predictive stdev 3e-7). `kumaraswamy` parameters are bounded to a, b in
  [1/3, 3], and `boxcox` / `kumaraswamy` / `neural_mono` / `knots` fits maximise the
  log-likelihood minus a penalty on gaps between consecutive training inputs where the warp
  is much flatter than on average (`logLikelihood()` stays the plain LL). On that benchmark
  (8 designs, n = 40) the median Q2 of `kumaraswamy` is 0.81 (Kriging 0.75) with no negative
  Q2 under the default optimiser; `boxcox` goes from -0.03 to 0.81. A joint `BFGS` line-search
  step to a point where the correlation matrix cannot be factorised now backtracks instead of
  ending the run there, and non-finite warped features raise before `chol()` (the spurious
  "chol(): given matrix is not symmetric" warnings). `neural_mono` uses an overflow-safe
  softplus (it gave NaN predictions for large weights)
  ([#406](https://github.com/libKriging/libKriging/issues/406)).
- **Kriging**: `parameters` with a `theta` and `is_theta_estim = false` now keep
  theta fixed with `optim = "BFGS[#]"` too (it was silently re-estimated unless
  `optim = "none"`); the nugget / heterogeneous-noise `sigma2` are still
  optimized per their own flags, and `update(…, refit = true)` keeps theta
  fixed. Several `theta` rows with `is_theta_estim = false` are rejected
  ([#388](https://github.com/libKriging/libKriging/issues/388)).
- `Kriging::fit` on degenerate data:
  - `y` in the span of the trend (constant `y` with a constant trend, affine
    `y` with a linear one, ...): the residual is zero for every theta, so
    sigma2 = 0 and theta is not identifiable; the optimizer drove theta to a
    bound where R is singular and failed ("All 1 optimization attempts
    failed"). The fit now skips the optimization and commits the model at the
    given theta (or a deterministic default): least-squares trend, sigma2 = 0
    (and nugget = 0), i.e. trend predictions with zero variance. With a known
    heterogeneous noise, sigma2 = 0 is not representable: explicit error.
  - a constant input column got [0, 0] theta bounds (theta = 0: division by
    zero in the kernel); its theta is now pinned to 1.
  - `normalize = true` divided by a zero range for a constant output or input
    column (shared by Kriging, WarpKriging, MLPKriging); it is now only
    centered.
  - BFGS restarts started from `(theta_start + theta_lower) / 2^k`, which can
    lie below `theta_lower`; L-BFGS-B does not project its starting point, so
    the returned theta could be outside its bounds. The restart point is now
    clamped to the bounds, and the "stuck at a bound" test ignores pinned
    dimensions.
- **`fit` / `update` now drop the state of the last `simulate(…,
  will_update=true)`** (and the update-points cache) for `Kriging` (every
  noise model and method, refit or not), `WarpKriging` and `MLPKriging`: a
  later `update_simulate` mixed matrices built for the previous model with
  the new one, failing with a dimension error ("matrix multiplication:
  incompatible matrix dimensions") or, after a refit on as many points or
  when the update-points cache was already filled, silently returning paths
  of the previous model. `update_simulate` now raises `std::runtime_error("No
  previous simulation data available")` until a new `simulate(…,
  will_update=true)`; a call rejected on its input (e.g. `update(y_u, X_u)`
  on a heterogeneous model) leaves the state unchanged. Also,
  `Kriging::simulate(…, will_update=true)` without `with_noise` dropped
  nothing of an earlier `simulate(…, with_noise, true)`, whose with_noise
  draws `update_simulate(y_u, noise_u, X_u)` then added again.
- **`Kriging::update_simulate(y_u, X_u)` after `update_simulate(y_u, noise_u,
  X_u)`** with the same `X_u` reused the noisy factorisation left in the
  update-points cache (keyed on `X_u` only) and returned the noisy result. The
  cache is now keyed on `(X_u, noise_u)`: it is reused across `y_u` for the
  same `X_u` and `noise_u` (as for WarpKriging), and never between the noisy
  and noise-free forms.
- `update_simulate` (Kriging, with nugget or noise, and WarpKriging): the
  conditioned paths mixed the raw and normalized output scales, giving a wrong
  mean and stdev with `normalize=true`; extending the paths to `X_u` inverted
  the near-singular joint correlation of the simulation design, which left an
  absolute stdev floor and mean shifts for smooth kernels. Both are fixed (the
  extension is now a block-Cholesky extension driven by the normals of
  `simulate()`), and `noise_u` is normalized by `scaleY^2` in
  `Kriging::update_simulate(y, noise, X)`.
- `Kriging::simulate` nugget (`with_nugget`) and heterogeneous-noise
  (`with_noise`) overloads skipped the light-fit checks: on a Nystrom or
  Vecchia fit without factorization they went to the exact simulator, which has no
  factorization to use. They now raise a clear error there, like the base
  overload (see the explicit `method` entry above).
- **Octave / MATLAB `Kriging.update_simulate`** dispatches on its argument
  count, as `update`: `update_simulate(y_u, X_u)` and
  `update_simulate(y_u, noise_u, X_u)`, whatever the noise model. On a
  heterogeneous model the 3-argument form failed with an unclear
  missing-argument error; it now reaches the C++ `update_simulate(y_u, X_u)`,
  which raises the explicit heterogeneous-noise error (see Changed).
  `WarpKriging.update_simulate` with a
  cell (categorical) `X_u` forwards `noise_u` (it was dropped).
- **Octave / MATLAB `Kriging.logLikelihoodFun`**: the 4th argument
  (`want_hess`) was passed to the C++ `logLikelihoodFun(theta, return_grad,
  bench)` as `bench` (printing timings instead of returning a hessian). Since
  the Hessian was removed (see Removed) the signature is
  `[ll, llgrad] = logLikelihoodFun(theta, return_grad[, bench])`.
- Octave/Matlab `Kriging.load` built and printed a throw-away
  `Kriging([1], [1], "gauss")` model, whose mex reference was then
  overwritten and never released. It now wraps the loaded reference directly,
  like `WarpKriging.load` / `MLPKriging.load`.
- Octave/Matlab `Kriging.copy` returned the raw mex reference of the copy
  instead of a `Kriging` object (and leaked it when the result was not
  assigned). It now returns a `Kriging` object, like `WarpKriging.copy` /
  `MLPKriging.copy`.
- Python `pylibkriging.load()` and Octave/Matlab `class_saved` /
  `load_kriging` failed on a saved `Kriging` with a nugget or noise channel
  (described as `NuggetKriging` / `NoiseKriging` by the loader, which these
  two bindings did not map back to `Kriging`).
- **Octave/MATLAB**: `NestedKriging.predict(X, return_stdev)` accepts the optional
  `return_stdev` flag like `Kriging.predict` (it raised "Input requires exactly 2 arguments",
  [#385](https://github.com/libKriging/libKriging/issues/385)).
- **R**: `str()`/`as.list()` on an unfitted `Kriging("gauss")` no longer segfaults
  (the trend `regmodel` was left uninitialized, and a `str.Kriging` method now displays the model content, [#249](https://github.com/libKriging/libKriging/issues/249)).
- R: `WarpKriging` and `MLPKriging` had no `k$F()` / `k$T()` (only
  `k$F_()` / `k$T_()`), although `bindings/README.md` documents `obj$F()`,
  and `Kriging` had no `F_()` / `T_()`. All three classes now provide both
  `k$F()` / `k$T()` and `F_(k)` / `T_(k)` (the S3 generics stay named `F_` /
  `T_` so as not to mask `base::F` / `base::T`).
- Python scikit-learn wrapper `NestedKrigingRegressor`: `nb_groups` is
  capped to `n // (d + 2)` (value used stored in `nb_groups_`, with a warning)
  and fewer than `d + 2` samples raise a clear `ValueError`, instead of the
  native "nb_groups should be in [1, n/(d+2)]" error on small samples.
- R: `utils` moves from `Suggests` to `Imports` in `rlibkriging`'s
  `DESCRIPTION`, since the `NAMESPACE` imports it (`@importFrom utils methods`);
  `R CMD check` reported a NOTE ("Base package in Suggests/Enhances imported in
  NAMESPACE").
- **R (Linux binary)**: the release archive bundles OpenBLAS and its Fortran runtime
  (`$ORIGIN` RUNPATH), which a base R install lacks, so `library(rlibkriging)` no longer fails
  with `libopenblas.so.0: cannot open shared object file`; the README documents the
  workaround for earlier archives ([#387](https://github.com/libKriging/libKriging/issues/387)).
- Windows: OpenBLAS is pinned to 0.3.33 (conda-forge) and
  `OPENBLAS_L2_SIZE=2048` is exported in the Windows build environments.
  OpenBLAS 0.3.34 overflows the stack of `dgemm_kernel_ZEN`/`HASWELL` on AMD
  Zen 4/5 hosts exposing AVX-512 (OpenMathLib/OpenBLAS#6013, #6021), which
  crashed the R, Python and Julia Windows builds; the bundled DLL ships with
  the Python wheel and the R release.
- **Build**: the `install.lib` target no longer fails on a fresh build directory
  because of the bench executables it does not build: their install is now `OPTIONAL`
  ([#384](https://github.com/libKriging/libKriging/issues/384)).
- Documentation, skills and plugin commands checked against the bindings
  code (calls, argument names, return shapes; Python snippets executed, C++
  snippets compiled):
  - `bindings/README.md`: Octave/Matlab constructors are `Kriging(...)`,
    `WarpKriging(...)`, ... (not `build(...)`); Python loads are
    `Kriging.load(f)` / `pylibkriging.load(f)` (no `load_kriging`); R object
    methods are `feature_dim()`, `is_fitted()`, `hidden_dims()`; Python
    `subsetOfData` returns a 1-D array; the "C++ only" list now has
    `simulateVecchia` and no longer `vecchia_neighbors`.
  - `predict` unpacked as a pair for `Kriging` / `WarpKriging` /
    `MLPKriging` (5-tuple) in `docs/math/Kriging.md`,
    `Warping-MLPJoint.md`, `Warping-NeuralMono.md` and the C++ reference.
  - R examples passed `stdev = TRUE` (argument is `return_stdev`),
    `noise_model =` (argument is `noise`) and `simulate(..., X = )`
    (argument is `x`); Julia README examples passed `stdev=` / `cov=`
    (keywords are `return_stdev` / `return_cov`).
  - C++ reference: `WarpKriging` lives in `libKriging::`, its `fit` takes no
    warping list, and the `NestedKriging` constructor takes the aggregation,
    partition and seed before the trend.
  - `docs/math/Nystrom.md` said `predict` used an exact factorization after
    an `LLNystrom` fit; there is none (see the explicit `method` entry above).
  - `commands/simulate.md`: results are `npred × nsim` (one column per
    path) in every binding.
  - Every language reference now shows save / load and
    `simulate(..., will_update)` + `update_simulate`.

### Development (CI, tests, benchmarks, studies)
Not visible to users of the library; summarized, see the pull requests for details.
- **Benchmarks and studies**: statistical benchmark base of the fit, `bench/fit-protocol`
  (T7, #447; [docs/dev/rationalization/T7-benchmark-base.md](docs/dev/rationalization/T7-benchmark-base.md)),
  on which the 2.0 defaults were measured (T3, #485, #489, #479, #498); work logs of T1-T10 in
  `docs/dev/rationalization/`. The long-range restart of a single start (#426, #418) was added and
  removed (#466, #485) within the cycle, with `Optim::set_(max_)long_range_restart` and
  `LK_(MAX_)LONG_RANGE_RESTART` (never released; now ignored).
- **CI**: binding API check `binding-api.yml` (#419, #435: methods, arguments and setters
  compared across bindings and against C++); BLAS backends `blas-backends.yml` (T9, #449:
  oneMKL GNU / INTEL layers, macOS Accelerate, `BlasBackendTest`); Thread Sanitizer with Clang +
  Archer on the OpenMP loops (#420, #460); Memcheck with a zero threshold (#461, #463); the
  "without OpenMP" jobs really built without OpenMP (`omp_*` symbol check); MATLAB build and tests
  back (#180, #497, through `matlab-actions`, `tools/matlab/run_mex_tests.m`); runs on pull
  requests only, job timeouts, robust apt installs (#453, #454); Python 3.10-3.14 matrix
  (#386); runner diagnostics; Windows Python without `OMP_NUM_THREADS=1` (#477).
- **Tests**: Python tests and the Octave `test_new_features.m` registered in CTest (they never
  ran); the Nystrom LOO comparison of `KrigingApproxObjectivesTest` relative instead of bitwise
  (T10, #451); noise-aware finite-difference tolerance (#393); R fork regression test (#325);
  `AffinityTest`, `ConcurrentCallsTest`, `MultistartDeterminismTest`, `SettingsTest`,
  `JitterRuleTest`; Windows BLAS thread counts read at run time in `BlasBackendTest` and the thread
  benchmarks (#473, #481).
- **Developer diagnostics removed** (#468): `LK_LINEARALGEBRA_CHECK_TRACE` (stderr trace of
  `LinearAlgebra.check()`) and `LK_ACCELERATE_POOL_SINGLE` (macOS measurement switch of #491), both
  added during the cycle; now ignored.
- **Internal refactors**: one `parallel_for` for the OpenMP loops (#456, #462); one restart
  policy component `RestartPolicy.hpp` (T2, `AdamBFGS::max_restart` removed); KModel buffers per
  running start (T4).

## [1.2.2] - 2026-09-22

Documentation, packaging and release-process consolidation: no change to the
C++ core and no numerical change. Julia is now installed through
JLibKriging.jl.

### Changed
- Julia: the installable, registered Julia package is now
  [JLibKriging](https://github.com/libKriging/JLibKriging.jl), which builds
  libKriging from source at `Pkg.add` time (the way `rlibkriging` does for R)
  and follows libKriging `master` through `sync-as-submodule.yml`; the Julia
  binding itself stays in `bindings/Julia/jlibkriging`. Registration on the
  General registry moves out of `release-julia.yml`, which only tests the
  binding on release tags now (`jlibkriging` had been rejected by the registry's
  naming and licence guidelines). The README and the Julia binding pages now
  document this install path (`Pkg.add(url=...)` until the package is on the
  General registry).
- Python: the PyPI project page now shows the package README instead of a
  one-line description, and the `sklearn` extra that `pylibkriging.sklearn`
  asks users to install now exists (`pip install pylibkriging[sklearn]`). A
  `multifidelity` extra, added by mistake with the scikit-learn estimators and
  pointing at a module that does not exist, is removed.
- Benchmarks: the C++ benchmark report and executables name the noise model
  (`Kriging (nugget)`, `Kriging (heterogeneous noise)`) instead of the
  `NuggetKriging` / `NoiseKriging` classes merged into `Kriging` before 1.0.

### Documentation
- The README advertised `objective="VLL(m)"`, which was renamed
  `LLVecchia(m)` in 1.2.0 and now raises; it lists `LLVecchia(m)` and
  `LLNystrom(k)`. If you are coming from 1.1.x, see the *Breaking* entry of
  1.2.0 for the migration.
- The method reference (`bindings/README.md`) documents the 1.2 API:
  `subsetOfData` (whose indices are 0-based in Python and Julia, 1-based in R
  and Octave/Matlab), `nystrom_rank`, the objectives, the `WarpKriging`
  accessors and the scikit-learn estimators, and lists the Vecchia / Nystrom
  methods that are C++-only (`predictVecchia`, `predictNystrom`,
  `simulateNystrom`, `set_vecchia_exact_commit`), which `docs/math` did not say.
  All 58 binding notebooks are now linked, and `docs/math` and
  `docs/comparisons` have index pages.
- The libkriging skill and its per-language references were wrong for
  `WarpKriging`: it always fits with `"LL"` (any `objective` is ignored, so the
  Vecchia / Nystrom objectives apply to `Kriging` only) and has no
  `noise="nugget"` mode. They now cover `subsetOfData`, `nystrom_rank` and the
  scikit-learn estimators.
- README: complete list of input warpings, large-design methods, tested
  platforms taken from the CI configuration, and install links that were frozen
  on 0.x releases; same for `CITATION.cff`. The Nystrom, k-means and
  distributed-GP references and the `USE_JEMALLOC` CMake option were missing
  from `docs/dev`. The `pylibkriging` README is rewritten.

### CI/Release process
- `tools/release/check_versions.py` checks that `cmake/version.cmake`,
  `CITATION.cff`, `.claude-plugin/plugin.json`, `jlibkriging/Project.toml`,
  `rlibkriging/DESCRIPTION` and the changelog agree on the version and on the
  release date. It runs on every push and in the six release workflows, which
  used to compare the tag with `major.minor` of `cmake/version.cmake` only.
- `RELEASE.md` describes the release procedure.

## [1.2.1] - 2026-09-19

### Fixed
- Release process: `bindings/Julia/jlibkriging/Project.toml` and
  `.claude-plugin/plugin.json` had been left at 1.1.0 by the 1.2.0 release
  preparation, so the Julia release workflow rejected the `v1.2.0` tag and
  `jlibkriging` was never registered on Julia's General registry for 1.2.0.
  Both now carry the release version. 1.2.1 is otherwise identical to 1.2.0 for
  the C++ core, Python, R and Octave/Matlab bindings, and is the first 1.2
  release available for Julia (#365).

## [1.2.0] - 2026-09-19

### Added
- Python: scikit-learn compatible estimators for all four Kriging classes —
  `KrigingRegressor`, `WarpKrigingRegressor`, `MLPKrigingRegressor`,
  `NestedKrigingRegressor` in `pylibkriging.sklearn`, implementing the
  scikit-learn estimator API (`fit`/`predict`, `get_params`/`set_params`,
  `clone`) so they drop into `Pipeline` and `GridSearchCV` (#338).
- Cross-package comparison benchmark (`bench/comparison/`): libKriging vs.
  scikit-learn/GPy/SMT/OpenTURNS (Python) and DiceKriging/RobustGaSP (R) on
  shared randomized LHS designs (Branin, Hartmann-3/6, Borehole), reporting
  fit/predict time, RMSE, Q², NLPD; runs on a manual/monthly CI workflow (#335).
- `Kriging::subsetOfData`: k-means (or random) pre-fit row-subsetting for
  large designs. Available in the core C++ API and all four bindings
  (Python/R/Julia/Octave-MATLAB); see `docs/math/SubsetOfData.md` and
  `Scalability.md` (#358).
- `nystrom_rank()` accessor exposed in the Julia, Octave/MATLAB and R
  bindings (Python already had it); worked notebooks
  `docs/math/llnystrom_vs_cholesky.ipynb` / `llvecchia_vs_cholesky.ipynb`
  comparing `LLNystrom`/`LLVecchia` against exact Cholesky (#358).
- `LLNystrom(k)` objective (fixed-landmark Nystrom low-rank approximation of
  the covariance, greedy pivoted-Cholesky landmarks held fixed across theta so
  the objective stays smooth) for large-`n` scalability: `O(n·k²)` fit via the
  Woodbury identity with an analytic gradient, `predictNystrom`, and a
  Nystrom-specific `update()` (`refit=false` at fixed theta/landmarks, or
  `refit=true` warm-restarting theta over the same landmarks); such fits skip
  the `O(n²)` pairwise-difference cube (#346).
- `WarpKriging`: binding surface brought to parity with `Kriging` across the
  Python/R/Octave/Julia bindings — `noise()`, `warp_params()`, `optim()`,
  `objective()` and `covMat(X1, X2)` accessors, numeric `parameters` seeds
  (`theta` / `warp_params` / `noise`) with `optim="none"` to rebuild a model
  with frozen hyper-parameters, `noise=` and `parameters=` no longer mutually
  exclusive, and `update(..., noise_u=)` / `update_simulate(..., noise_u=)`;
  the R/Python docs no longer advertise an unimplemented `noise="nugget"`
  mode (#361).
- Claude Code plugin packaging: `.claude-plugin/` manifests and
  `fit`/`predict`/`simulate`/`update`/`build` commands driving the libKriging
  skill, installable through `/plugin marketplace add libKriging/libKriging`.

### Changed
- **Breaking**: the Vecchia objective `"VLL"` / `"VLL(m)"` introduced in 1.1.0
  is renamed `"LLVecchia"` / `"LLVecchia(m)"` (and the Nystrom objective is
  spelled `"LLNystrom"` / `"LLNystrom(k)"`, never released under its former
  `"LLNys"` name), in the C++ core, `NestedKriging` and all bindings. No alias
  was kept: `objective="VLL(m)"` now raises `Unsupported fit objective`.
  Migration: replace `"VLL"` by `"LLVecchia"` and `"VLL(m)"` by `"LLVecchia(m)"`
  in your calls; results are unchanged (#346).
- Python: dropped the `numpy<2` pin — `pylibkriging` now supports NumPy 2.x.
  Required bumping the vendored `pybind11` (2.10.1 → 2.13.6) and `carma`
  submodules, since both hardcode offsets into NumPy's C-API function table
  and predated the NumPy 2.0 ABI changes; also fixed a bug in carma's own
  NumPy-2.0 fix where `PyArray_CopyInto`'s table offset (which differs
  between NumPy 1.x and 2.x) was hardcoded to the NumPy-2-only value instead
  of being picked at runtime (#339, libKriging/carma#1).
- Performance: `R^-1` is now computed lazily in `populate_Model`. It is only
  consumed by an analytic theta gradient, so plain `fit()`, `predict()`,
  `logLikelihoodFun(theta, grad=false)` and, above all, `update(refit=false)`
  no longer pay a dense `O(n^3)` `inv_sympd` at the full size on every call;
  `update(refit=false)`'s incremental Cholesky is `O(n_old^2 * n_u)` again
  (#363).

### Fixed
- `WarpKriging`: every per-variable warping whose parametrisation assumes an
  `O(1)` / `[0, 1]` input — `knots(k)` (Xiong et al. 2007, on `[0, 1]`),
  `kumaraswamy` (a CDF on `[0, 1]`), `boxcox` (needs `x > 0`), `neural_mono`,
  `mlp` and `mlp_joint` (weight init + softplus / tanh) — now maps inputs
  from their training range onto `[0, 1]` internally, like `DiceKriging`'s
  `knots` argument. Previously inputs on any other scale collapsed onto the clamp /
  positivity / saturation boundary: `knots` and `kumaraswamy` diverged
  (`knots`: `|params|` to ~15, `theta` to its bound) instead of settling on
  the identity, the fit was >100 nats / 40x–300x RMSE worse than a plain
  stationary GP, and `neural_mono` could fail the Cholesky outright. The
  default range `[0, 1]` is the identity map, so a model fit on inputs
  spanning exactly `[0, 1]` is unchanged (`knots` / `kumaraswamy` / `boxcox`
  bit-for-bit; the neural warps to numerical precision); for any other range
  the transform is calibrated to the data range. New regression tests
  `test_warp_input_scale_invariance` / `test_warp_input_scale_hardening` in
  `WarpKrigingTest` (invariance across `[0,1]`, `[100,300]`, `[-50,50]`; plus
  save/load, `update()`, multistart, 2-D, `normalize=true` and derivative
  checks). `NestedKriging`'s warped submodels, which share one
  `(theta, warp_params)`, now also share one input-range calibration
  (`WarpKriging::recalibrate_warps()`), so the aggregate still interpolates
  the design (#362).
- `optim="none"` silently fell through to a plain exact factorization for
  a light Vecchia fit (`set_vecchia_exact_commit(false)`), ignoring the
  requested `LLVecchia(m)` objective entirely instead of committing a
  genuine light fit at the given theta (the same class of bug already
  fixed for `LLNystrom` in #353) (#358).
- R: `.match_kriging_objective`'s internal validator no longer hijacks
  `Kriging`'s roxygen `@export` documentation (#329).
- Python: `loading_test`'s version check no longer hardcodes the expected
  version, reading it from `cmake/version.cmake` instead so it doesn't need
  updating on every release (#328).
- CI: Windows jobs retry the `choco install` step to absorb transient
  community-feed 504s (#326); `rlibkriging`'s `tools/gitmodules-shas` is kept
  in sync with submodule bumps, staged in the right order (#330, #331).
- `predict(..., return_deriv=true)` returned wrong derivatives
  (`dyhat/dx`, `dysd2/dx`) whenever the model was fitted with
  `normalize=true`: the per-dimension division by `scaleX` required by the
  chain rule was missing, so every derivative was off by a factor of
  `scaleX` (#345).
- `WarpKriging`: the analytical warp-parameter gradient was silently wrong
  for every continuous warp (`knots`, `kumaraswamy`, `boxcox`, `affine`,
  `neural_mono`, `mlp`), which kept the bi-level optimizer (`BFGS+Adam` and
  joint `BFGS`) from ever discovering a non-trivial warp — inconsistent
  `sigma2` scaling, a sign flip from `compute_dX()` not being antisymmetric,
  and a missing factor 2. `warp_gradient()` now matches finite differences
  to ~1e-5, with a permanent regression test per warp type (#342).
- `LLNystrom`: deterministic `optim="none"` fits and a landmark-seeded BFGS
  warm start, fixing outlier fits in the comparison benchmark (#353).
- Windows/Python: fixed heap corruption (issue #354) caused by NumPy's and
  Armadillo's own allocators coexisting in one process — the
  `ARMA_ALIEN_MEM_*` defines that route Armadillo through `lkalloc` (and so
  through NumPy's allocator) had been disabled; re-enabled, with `lkalloc`
  falling back to `_aligned_malloc` / plain `malloc` for every other binding
  (#357). Debug builds also redirect CRT debug-heap errors to stderr, so a
  corruption fails in seconds instead of hanging CI on a blocking message
  box (#356); diagnosis in `docs/dev/WindowsPythonHangDiagnostic354.md`.
- R: `simulate.WarpKriging` no longer self-qualifies with `:::` (an
  `R CMD check` NOTE), `WarpKriging` is registered with `setOldClass` (no
  load-time warning), and the `save`/`load` examples clean up their
  temporary file (#360).

### Documentation
- Added a coding-agent skill covering libKriging usage patterns (#336) and a
  "Known pitfalls" section to `AGENTS.md` (#333).
- Refreshed the comparison notebooks and the READMEs (#343, #340).

### CI/Release process
- Automated `jlibkriging` registration on Julia's General registry (#332).
- GitHub release notes are now filled in from this changelog (#334).
- Windows CI: single-threaded OpenMP for the Octave (NestedKriging hang, #349)
  and Python (#351, #352) jobs; Coverage mode no longer flaky on an undefined
  `PROCESSOR_COUNT` (#359).
- Comparison benchmark: data-range-aware length-scale initialisation for
  GPy / scikit-learn / OpenTURNS, which previously failed on some designs (#341).
- `jlibkriging`: `[compat]` bounds and package README required by the Julia
  General registry's AutoMerge, so the next tagged release registers without
  manual review (#344).

## [1.1.0] - 2026-07-08

### Added
- `NestedKriging`: divide-and-conquer Gaussian process for large designs —
  partition of `(X, y)` into groups with one Kriging submodel each, unified
  hyperparameters, and aggregated predictions (PoE / gPoE / BCM / rBCM and the
  optimal nested-kriging `NK` aggregation), with Python/R/Octave/Matlab/Julia
  bindings (#317).
- Vecchia approximated log-likelihood objective `VLL(m)`, with local prediction
  and a factorization-free "light" mode (#318).

### Fixed
- Fork-after-threads deadlock in forked child processes (#319).
- Windows CI on the `windows-2025-vs2026` runner image: CMake pinned to the
  version providing the "Visual Studio 18 2026" generator, and Octave/conda
  setup (#320).
- Thread Sanitizer job: removed false-positive data races caused by GCC's
  uninstrumented OpenMP runtime (libgomp) (#320).
- Constructor argument consistency across bindings: R `Kriging` now accepts
  `objective="VLL(m)"` and the `"quadratic"` trend, with `noise` as the last
  argument (aligned with Python/WarpKriging); Julia `Kriging`/`NestedKriging`
  accept a `parameters` dict like the other classes (#323).

### Documentation
- Documentation, licensing and metadata review: fixed stale dependency and
  architecture docs, added scientific and input-warping references, added
  `CITATION.cff`, `NOTICE` and this changelog, and README features/license/
  citation sections (#321).

## Released versions

| Version | Date | Notes |
|:--------|:-----|:------|
| [1.2.2](https://github.com/libKriging/libKriging/releases/tag/v1.2.2) | 2026-09-22 | Julia installs through JLibKriging.jl; documentation, PyPI page and release-process consolidation (version consistency check, `RELEASE.md`); no API or numerical change. |
| [1.2.1](https://github.com/libKriging/libKriging/releases/tag/v1.2.1) | 2026-09-19 | Julia registration fix (version metadata); otherwise identical to 1.2.0. |
| [1.2.0](https://github.com/libKriging/libKriging/releases/tag/v1.2.0) | 2026-09-19 | `LLNystrom` objective; scikit-learn estimators; `subsetOfData`; NumPy 2; WarpKriging binding parity and gradient/input-range fixes; `predict` derivative fix under `normalize`; Windows/Python heap-corruption fix; lazy `R^-1` (faster `update`). |
| [1.1.0](https://github.com/libKriging/libKriging/releases/tag/v1.1.0) | 2026-07-08 | NestedKriging for large designs; Vecchia VLL objective; fork/threads, Windows CI and TSan fixes; docs & licensing review. |
| [1.0.0](https://github.com/libKriging/libKriging/releases/tag/v1.0.0) | 2026-05-13 | First stable 1.0 release. |
| [0.9.3](https://github.com/libKriging/libKriging/releases/tag/v0.9.3) | 2026-01-18 | |
| [0.9.2](https://github.com/libKriging/libKriging/releases/tag/v0.9.2) | 2025-12-17 | |
| [0.9.1](https://github.com/libKriging/libKriging/releases/tag/v0.9.1) | 2025-01-14 | |
| [0.9.0](https://github.com/libKriging/libKriging/releases/tag/v0.9.0) | 2024-09-04 | |
| [0.8.3](https://github.com/libKriging/libKriging/releases/tag/v0.8.3) | 2023-12-10 | |
| [0.8.2](https://github.com/libKriging/libKriging/releases/tag/v0.8.2) | 2023-12-10 | |
| [0.8.0](https://github.com/libKriging/libKriging/releases/tag/v0.8.0) | 2023-05-23 | |
| [0.7.4](https://github.com/libKriging/libKriging/releases/tag/v0.7.4) | 2023-01-13 | |
| [0.7.3](https://github.com/libKriging/libKriging/releases/tag/v0.7.3) | 2023-01-09 | |
| [0.7.2](https://github.com/libKriging/libKriging/releases/tag/v0.7.2) | 2022-12-23 | |
| [0.7.1](https://github.com/libKriging/libKriging/releases/tag/v0.7.1) | 2022-12-23 | |
| [0.7.0](https://github.com/libKriging/libKriging/releases/tag/v0.7.0) | 2022-10-06 | |
| [0.6.0](https://github.com/libKriging/libKriging/releases/tag/v0.6.0) | 2022-05-24 | |
| [0.5.1](https://github.com/libKriging/libKriging/releases/tag/v0.5.1) | 2022-04-07 | |
| [0.4.8](https://github.com/libKriging/libKriging/releases/tag/v0.4.8) | 2021-12-05 | |
| [0.4.7](https://github.com/libKriging/libKriging/releases/tag/v0.4.7) | 2021-09-05 | |
| [0.4.5](https://github.com/libKriging/libKriging/releases/tag/v0.4.5) | 2021-09-02 | |
| [0.4.4](https://github.com/libKriging/libKriging/releases/tag/v0.4.4) | 2021-09-02 | |
| [0.4.3](https://github.com/libKriging/libKriging/releases/tag/v0.4.3) | 2021-08-30 | |
| [0.4.2](https://github.com/libKriging/libKriging/releases/tag/v0.4.2) | 2021-06-01 | |
| [0.4.1](https://github.com/libKriging/libKriging/releases/tag/v0.4.1) | 2021-05-31 | First public pre-releases. |

[2.0.0]: https://github.com/libKriging/libKriging/compare/v1.2.2...master
[1.2.2]: https://github.com/libKriging/libKriging/compare/v1.2.1...v1.2.2
[1.2.1]: https://github.com/libKriging/libKriging/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/libKriging/libKriging/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/libKriging/libKriging/compare/v1.0.0...v1.1.0
