# Changelog

All notable changes to libKriging are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/).

This file was introduced during the 1.x cycle. For the detailed notes of each
past release, see the corresponding entry on the
[GitHub releases page](https://github.com/libKriging/libKriging/releases).

## [Unreleased]

### Added
- **CI**: `binding-api` job and "check binding API" step: the names exposed by the Python, R,
  Octave and Julia bindings are compared to the manifest `bindings/api.yaml`, where each gap is
  listed as `not exposed: <reason>`, and every binding test file must be run by ctest or
  testthat ([#419](https://github.com/libKriging/libKriging/issues/419)).
- **R**: `Kriging`, `NestedKriging`, `WarpKriging` and `MLPKriging` objects survive R
  serialization: `saveRDS()` / `readRDS()`, `save()` / `load()` of a workspace,
  `serialize()`, and shipping to PSOCK / `future` workers. The current model is stored as its
  JSON save file and reloaded into a new C++ object, which the object and its `$` methods
  share ([#327](https://github.com/libKriging/libKriging/issues/327)). Objects serialized by
  an older rlibkriging cannot be restored (use `$save()` / `load.*()`).

### Fixed
- **MLPKriging** / `mlp_joint`: the K refits of the predictive-variance calibration (#407)
  run concurrently in the multistart thread pool (`Optim::set_thread_pool_size`) instead of
  one after the other. The calibration made the fit 5-10x slower (n = 200, d = 4,
  `hidden_dims=[16, 8]`: 6.4 s instead of 0.6 s with `BFGS+Adam`, 12.4 s instead of 1.4 s
  with `BFGS`), which pushed the C++ benchmark CI job past its 3 h timeout; on 12 cores it
  now takes 1.3 s and 2.9 s. The folds are independent and their residuals summed in fold
  order, so the result depends on the pool size only through the number of BLAS threads
  each fold gets (as a multistart).
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
- **MLPKriging**: the predictive stdev is calibrated on held-out points. With more weights
  than observations, the likelihood-optimal network learns features in which the training
  outputs are almost exactly smooth, and the GP on top was overconfident: on `Branin(Wx)` in
  10D (n = 120, `hidden_dims=[16, 8]`, `BFGS10`) 95 % intervals covered 1–2 % of test points
  (NLPD 3.7e4–9.6e5 vs 2.93 for `Kriging`). After the fit, a 5-fold cross-validation (each
  fold refitted from the starting point of the winning start) scales σ² for `predict` /
  `simulate` / `update_simulate` (`var_scale()`, reported in `summary()`, saved with the
  model); the mean, `sigma2()` and `logLikelihood()` are unchanged, `calibration_folds = "0"`
  disables it (also a `WarpKriging` tuning parameter for `mlp_joint`). The joint L-BFGS-B of
  a joint MLP keeps `Optim::max_iteration` iterations per call, resumed while it improves,
  instead of up to 1000 (#382): longer runs only overfitted. On that benchmark (4 designs,
  tanh / selu) `BFGS10` takes 2.3–2.5 s instead of 23–50 s, with Q2 0.986–0.998, NLPD
  0.75–2.25 and 95 % coverage 0.95–1.00. The single-start default still often stops at a poor
  mean in 10D (Q2 < 0); its coverage is now 0.84–1.00
  ([#407](https://github.com/libKriging/libKriging/issues/407)).
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
- **R**: regression test for forked workers (`parallel::mcparallel`) after a multistart
  fit, and the fork / threading constraints documented in `bindings/R/README.md`
  ([#325](https://github.com/libKriging/libKriging/issues/325)).
- **R (Linux binary)**: the release archive bundles OpenBLAS and its Fortran runtime
  (`$ORIGIN` RUNPATH), which a base R install lacks, so `library(rlibkriging)` no longer fails
  with `libopenblas.so.0: cannot open shared object file`; the README documents the
  workaround for earlier archives ([#387](https://github.com/libKriging/libKriging/issues/387)).
- **Kriging (nugget)**: the BFGS fit seeds its first start with the theta of the
  noise-free fit on the same data and a small nugget share (alpha = 0.99), instead of a
  random start that often ended in the all-noise region (nugget >> sigma2) with the
  default single start. On Branin (n = 30, low noise) the mean LL gap to `BFGS10` drops
  from ~8 to ~0.5 ([#383](https://github.com/libKriging/libKriging/issues/383)).
- **Build**: the `install.lib` target no longer fails on a fresh build directory
  because of the bench executables it does not build: their install is now `OPTIONAL`
  ([#384](https://github.com/libKriging/libKriging/issues/384)).
- **Octave/MATLAB**: `NestedKriging.predict(X, return_stdev)` accepts the optional
  `return_stdev` flag like `Kriging.predict` (it raised "Input requires exactly 2 arguments",
  [#385](https://github.com/libKriging/libKriging/issues/385)).
- **R**: `str()`/`as.list()` on an unfitted `Kriging("gauss")` no longer segfaults
  (the trend `regmodel` was left uninitialized, and a `str.Kriging` method now displays the model content, [#249](https://github.com/libKriging/libKriging/issues/249)).
- **Kriging**: `parameters` with a `theta` and `is_theta_estim = false` now keep
  theta fixed with `optim = "BFGS[#]"` too (it was silently re-estimated unless
  `optim = "none"`); the nugget / heterogeneous-noise `sigma2` are still
  optimized per their own flags, and `update(…, refit = true)` keeps theta
  fixed. Several `theta` rows with `is_theta_estim = false` are rejected
  ([#388](https://github.com/libKriging/libKriging/issues/388)).

### Changed
- **Python**: supported CPython versions are 3.10 to 3.14: the CI tests each of them
  (3.10 and 3.14 on Linux and Windows), the release builds wheels for all of them, and
  `python_requires` is now `>=3.10` (the end-of-life 3.7 and 3.9 jobs are dropped)
  ([#386](https://github.com/libKriging/libKriging/issues/386),
  [#389](https://github.com/libKriging/libKriging/issues/389)).
- **MLPKriging: the default optimiser is now `"BFGS"`** (joint L-BFGS-B on
  the MLP weights and θ) in C++ and every binding, instead of `"BFGS+Adam"`,
  whose 10 Adam steps of size 1e-3 left the network at its random
  initialisation (#382).
- **WarpKriging: the default optimiser is now `"BFGS"`** too (C++ and every
  binding), instead of `"BFGS+Adam"`: on every warp tested (Branin 2D,
  kumaraswamy, boxcox, knots, neural_mono, mlp, mlp_joint) the joint
  L-BFGS-B reaches an equal or better log-likelihood than the (now
  effective, see below) bi-level Adam phase, in less time. `"BFGS+Adam"`
  stays available on request.
- WarpKriging/MLPKriging `update(refit=true)` of a model fitted with
  `"BFGS"`/`"BFGSk"` re-runs the joint L-BFGS-B (warm start) instead of
  always using the bi-level `"BFGS+Adam"` path.
- WarpKriging/MLPKriging joint `"BFGS"`: the iteration budget grows with the
  number of optimised parameters (20 per parameter, capped at 1000, never
  below `Optim::max_iteration`) instead of the global `Optim::max_iteration`
  (20), sized for a handful of θ; a run that hits the budget while still
  improving is resumed from the point reached rather than restarted from a
  contracted θ. The `max_iter_bfgs` tuning knob now defaults to 0 (automatic:
  100 for the inner θ L-BFGS of `"BFGS+Adam"`, the budget above for the
  joint L-BFGS-B) and, when set, applies to both (models saved by earlier
  versions carry `max_iter_bfgs = 100`, used as-is after `load`). Fits get slower but reach
  the optimum: ~0.2 s instead of ~5 ms on the #382 case (Branin, n = 30,
  `hidden_dims=[8]`), with log-likelihood -80 instead of -172.
- WarpKriging `"BFGS+Adam"`: the Adam phase now
  actually moves the warp parameters. Defaults `max_iter_adam` 10 → 200 and
  `adam_lr` 1e-3 → 3e-2 (each Adam step moves a parameter by about
  `adam_lr`, so 10 × 1e-3 left the warps at their initialisation), with an
  early stop after 20 steps without a relative log-likelihood gain > 1e-6.
  On Branin 2D (n = 30) the log-likelihood rises e.g. from -131 to -94
  (`mlp`), -120 to -99 (`neural_mono`), -155 to -83 (`mlp_joint`), for
  30–200 ms instead of 5–20 ms. Models saved by earlier versions carry
  `max_iter_adam = 10`, `adam_lr = 0.001`, used as-is after `load`.
- WarpKriging/MLPKriging multistart (`"BFGSk"`, `"BFGSk+Adam"`): starts
  2..k also re-draw the MLP weights (`mlp_joint`, per-dimension `mlp`), not
  only θ; start 1 still uses the current parameters.
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
  - `objective="LLVecchia(m)"` / `"LLNystrom(k)"` are deprecated aliases of
    `objective="LL"` with `method="vecchia(m)"` / `"nystrom(k)"`; a
    conflicting `method` raises. Save files of earlier versions load with the
    matching method. Using an alias raises the native deprecation warning of
    each language: Python `DeprecationWarning`, R `warning` of class
    `deprecatedWarning`, Julia `Base.depwarn` (shown without `--depwarn`),
    Octave/Matlab warning `mLibKriging:deprecated`. They follow each
    language's filters (`warnings.simplefilter("error")`, `options(warn = 2)`,
    `--depwarn=error`, `warning("error", ...)`). In C++, the message goes to
    `Kriging::set_deprecation_handler` (default: printed once on
    `arma::cout`), and `Kriging::deprecated_objective_message` gives it.
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
  - `set_vecchia_exact_commit(false)` is a deprecated no-op in C++ and
    `set_vecchia_exact_commit(true)` raises; `vecchia_exact_commit()` is
    deprecated.
  - NestedKriging: `method="vecchia(m)"` or `"nystrom(k)"` (after
    `warping`) estimates the common prior with one global Vecchia or Nystrom
    fit (formerly only `objective="LLVecchia(m)"`); `objective()` and
    `method()` accessors. WarpKriging and MLPKriging accept `method` for a
    uniform API (only `"exact"` is implemented).

### Added
- **Python**: a source distribution (sdist) is published with the wheels, so `pip install
  pylibkriging` builds from source (CMake, a C++17 compiler and BLAS / LAPACK) where no
  wheel matches, instead of failing with "no matching distribution". It is built from the
  new root `MANIFEST.in` / `pyproject.toml`, and its install is checked on every push
  ([#389](https://github.com/libKriging/libKriging/issues/389)). New CMake option
  `LIBKRIGING_BUILD_TESTS` (default `ON`; `OFF` in `setup.py`, which no longer builds the
  C++ tests nor benchmarks, and links armadillo statically).
- CMake package for the C++ library (#127): after installation,
  `find_package(libKriging CONFIG)` and `target_link_libraries(app PRIVATE libKriging::Kriging)`
  work, with version checking (`SameMajorVersion`) and transitive dependencies
  (Armadillo, lbfgsb_cpp, OpenMP).
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

### Fixed
- WarpKriging/MLPKriging parallel multistart: each worker now gets
  `ncpu / workers` BLAS/OpenMP threads (as `Kriging` already did) instead of
  all of them; the nested thread teams contended (`"BFGS4"`, `mlp_joint`,
  n = 500: 19.6 s → 10.1 s, system time 24 s → 1 s). The staggered start
  delay (`Optim::thread_start_delay_ms`) now applies only to each worker's
  first start instead of every start (`"BFGS10"`, n = 30: 175 ms → 43 ms).
- WarpKriging/MLPKriging joint `"BFGS"`: when the objective cannot be
  evaluated (correlation matrix not factorisable even with the numerical
  nugget), the fit keeps the best point evaluated so far instead of throwing.
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
  bench)` as `bench` (printing timings instead of returning a hessian). The
  signature is now `[ll, llgrad, llhess] = logLikelihoodFun(theta,
  return_grad, return_hess, bench)`, as in Python / R: `Kriging` computes no
  hessian, so `llhess` is empty (as in Python / R), and `bench` is the 5th
  argument. Julia `log_likelihood_fun(::Kriging; return_hess=true)` returns
  an empty hessian instead of an uninitialized `n × n` matrix.
- `Kriging::simulate` nugget (`with_nugget`) and heterogeneous-noise
  (`with_noise`) overloads skipped the light-fit checks: on a Nystrom or
  Vecchia fit without factorization they went to the exact simulator, which has no
  factorization to use. They now raise a clear error there, like the base
  overload (see the explicit `method` entry above).
- Python `pylibkriging.load()` and Octave/Matlab `class_saved` /
  `load_kriging` failed on a saved `Kriging` with a nugget or noise channel
  (described as `NuggetKriging` / `NoiseKriging` by the loader, which these
  two bindings did not map back to `Kriging`).
- Python: deadlock in code paths where libKriging allocates from worker
  threads while the calling Python thread holds the GIL: NestedKriging NK
  `predict` (OpenMP pair loop) and WarpKriging parallel multistart
  (`BFGS<k>`). The numpy allocator handed to libKriging reports to
  tracemalloc, which takes the GIL; it is now used only from threads holding
  the GIL, other threads use `std::malloc` (each block is freed by the C
  runtime that allocated it). The NestedKriging Python tests are now
  registered in CTest, with a subprocess/timeout regression test.
- Octave/Matlab `Kriging.load` built and printed a throw-away
  `Kriging([1], [1], "gauss")` model, whose mex reference was then
  overwritten and never released. It now wraps the loaded reference directly,
  like `WarpKriging.load` / `MLPKriging.load`.
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
- R: `WarpKriging` and `MLPKriging` had no `k$F()` / `k$T()` (only
  `k$F_()` / `k$T_()`), although `bindings/README.md` documents `obj$F()`,
  and `Kriging` had no `F_()` / `T_()`. All three classes now provide both
  `k$F()` / `k$T()` and `F_(k)` / `T_(k)` (the S3 generics stay named `F_` /
  `T_` so as not to mask `base::F` / `base::T`).
- Python scikit-learn wrapper `NestedKrigingRegressor`: `nb_groups` is
  capped to `n // (d + 2)` (value used stored in `nb_groups_`, with a warning)
  and fewer than `d + 2` samples raise a clear `ValueError`, instead of the
  native "nb_groups should be in [1, n/(d+2)]" error on small samples.
- Python tests `test_new_features.py`, `sklearn_estimator_test.py` and
  `sklearn_multimodel_test.py` were never run by CTest (and the scikit-learn
  ones failed); they are now registered and pass.
- CI: the "Linux Debug without OpenMP" and "R Linux without OpenMP" jobs
  built with OpenMP. They only removed `libomp-dev` (LLVM's runtime), while
  GCC ships its own (`libgomp`, `omp.h`), so `find_package(OpenMP)` still
  found `-fopenmp`. They now configure with
  `-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON`, and a new step fails if the
  installed libKriging references any `omp_*` / `GOMP_*` symbol.
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
    `Warping-MLPJoint.md`, `Warping-NeuralMono.md` and the C++ reference;
    Python `logLikelihoodFun` returns `(ll, grad, hess)`.
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
- Octave/Matlab `Kriging.copy` returned the raw mex reference of the copy
  instead of a `Kriging` object (and leaked it when the result was not
  assigned). It now returns a `Kriging` object, like `WarpKriging.copy` /
  `MLPKriging.copy`.
- R: `utils` moves from `Suggests` to `Imports` in `rlibkriging`'s
  `DESCRIPTION`, since the `NAMESPACE` imports it (`@importFrom utils methods`);
  `R CMD check` reported a NOTE ("Base package in Suggests/Enhances imported in
  NAMESPACE").
- WarpKriging: intermittent heap corruption (`malloc(): corrupted top size`)
  and degraded optimum in the parallel multistart of `fit()`
  (`BFGS<k>`, `BFGS<k>+Adam`). Concurrent L-BFGS-B runs raced on the `static`
  locals of the f2c BLAS/LINPACK helpers of `lbfgsb_cpp`; the submodule is
  bumped to `b6eff96`, which makes them reentrant. The multistart thread
  joiner is also constructed before the threads are spawned, so they are
  joined even if a later spawn throws.
- Windows: OpenBLAS is pinned to 0.3.33 (conda-forge) and
  `OPENBLAS_L2_SIZE=2048` is exported in the Windows build environments.
  OpenBLAS 0.3.34 overflows the stack of `dgemm_kernel_ZEN`/`HASWELL` on AMD
  Zen 4/5 hosts exposing AVX-512 (OpenMathLib/OpenBLAS#6013, #6021), which
  crashed the R, Python and Julia Windows builds; the bundled DLL ships with
  the Python wheel and the R release.
- `update_simulate` (Kriging, with nugget or noise, and WarpKriging): the
  conditioned paths mixed the raw and normalized output scales, giving a wrong
  mean and stdev with `normalize=true`; extending the paths to `X_u` inverted
  the near-singular joint correlation of the simulation design, which left an
  absolute stdev floor and mean shifts for smooth kernels. Both are fixed (the
  extension is now a block-Cholesky extension driven by the normals of
  `simulate()`), and `noise_u` is normalized by `scaleY^2` in
  `Kriging::update_simulate(y, noise, X)`.

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

[Unreleased]: https://github.com/libKriging/libKriging/compare/v1.2.2...master
[1.2.2]: https://github.com/libKriging/libKriging/compare/v1.2.1...v1.2.2
[1.2.1]: https://github.com/libKriging/libKriging/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/libKriging/libKriging/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/libKriging/libKriging/compare/v1.0.0...v1.1.0
