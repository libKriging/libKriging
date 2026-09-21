# Changelog

All notable changes to libKriging are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/).

This file was introduced during the 1.x cycle. For the detailed notes of each
past release, see the corresponding entry on the
[GitHub releases page](https://github.com/libKriging/libKriging/releases).

## [Unreleased]

### Changed
- Julia: the installable, registered Julia package is now
  [JLibKriging](https://github.com/libKriging/JLibKriging.jl), which builds
  libKriging from source at `Pkg.add` time (the way `rlibkriging` does for R)
  and follows libKriging `master` through `sync-as-submodule.yml`; the Julia
  binding itself stays in `bindings/Julia/jlibkriging`. Registration on the
  General registry moves out of `release-julia.yml`, which only tests the
  binding on release tags now (`jlibkriging` had been rejected by the registry's
  naming and licence guidelines).

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

[Unreleased]: https://github.com/libKriging/libKriging/compare/v1.2.1...master
[1.2.1]: https://github.com/libKriging/libKriging/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/libKriging/libKriging/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/libKriging/libKriging/compare/v1.0.0...v1.1.0
