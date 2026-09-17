# Changelog

All notable changes to libKriging are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/).

This file was introduced during the 1.x cycle. For the detailed notes of each
past release, see the corresponding entry on the
[GitHub releases page](https://github.com/libKriging/libKriging/releases).

## [Unreleased]

### Added
- CUDA iterative backend: `build_cov_kernel` (the dense fast-path builder
  behind `LinearAlgebraCuda`'s `R`/`dR` cache) now computes each covariance
  pair once instead of twice. `R` and every `∂R/∂θₖ` block are symmetric
  (`Cov(Xi,Xj,θ) == Cov(Xj,Xi,θ)`), but the kernel launched one thread per
  `(i,j)` over the FULL n×n grid, each independently paying the
  transcendental-heavy (`exp`/`log1p`) evaluation for its own cell — i.e.
  every pair's cost twice for no reason. Now only the upper triangle
  (`i <= j`) does the evaluation, writing the result to both `R[i,j]` and
  `R[j,i]` (same for `dR`); lower-triangle threads return immediately.
  Bit-identical output (cross-checked), 527 assertions green. This was
  plan item #6, expected to be worth ×3-5 on the SLQ path when originally
  scoped — but that estimate predates the device-side `R`/`dR` cache (plan
  item #3, already shipped) that stopped `R` from being rebuilt on every
  Lanczos/CG step: measured directly (H100, n=8000, isolating the build by
  forcing a cache miss vs. hit), the one-time build now costs only ~3 ms
  out of a ~1.3 s evaluation, so halving it is real but currently
  invisible in aggregate timing — kept because it's a correct, free,
  zero-downside fix that matters more wherever the build itself dominates
  (very large `n`, or few-CG-iteration regimes).
- CUDA iterative backend: `_logLikelihoodIterative`'s gradient path now
  fuses its two separate batched CG solves (`[F|y]`, then the Hutchinson
  probes) into ONE Krylov pass on `[F|y|probes]` (plan item #5, phase A —
  see `todo`-style discussion in the PR; the harder phase B, reading the
  SLQ log-determinant directly off this fused solve's own CG coefficients
  instead of running a separate dedicated Lanczos, is NOT done here — that
  was tried once and deliberately reverted, per the comment at
  `Kriging.cpp`'s SLQ dispatch, because without full reorthogonalization
  the CG-derived tridiagonal degrades as `cond(R)` grows with `n`). Both
  `LinearAlgebraCuda::conjugateGradient` and
  `LinearAlgebra::conjugateGradientBatched` gained a per-column-tolerance
  overload (`tol` as an `arma::vec` instead of `double`, existing scalar
  callers unaffected via a thin broadcast overload) so the fused call can
  still converge `[F|y]` to `cg_tol` and probes to the looser
  `probes_cg_tol` independently, each column freezing at its own rate —
  bit-for-bit what two separate calls would produce, just paying one
  matvec-per-iteration loop's kernel-launch/host-sync overhead instead of
  two. CUDA-only for now (mirrors the SLQ device-residency work just
  above): HIP/SYCL/Metal keep the two-separate-calls path, since only
  `LinearAlgebraCuda::conjugateGradient` has the vector-tol overload.
  Values cross-checked against the pre-fusion path (differences at the
  1e-4-1e-5 relative level, consistent with floating-point reassociation
  from batching order, not a correctness regression — the SLQ term's own
  ~5% stochastic bias already swamps this). Measured on an H100 (isolated
  timings, shared-node noise still present): ~10-24% faster on
  `logLik`+gradient across n=2000-16000, on top of the SLQ device-residency
  gain above — notably this is the first item in this series to move the
  needle at n=16000, where the SLQ optimization alone showed nothing
  because the probe CG solve dominated total time regardless.
- CUDA iterative backend: the SLQ log-determinant's batched Lanczos
  recurrence (`LLIterative`'s log-det term) is now fully device-resident
  (`LinearAlgebraCuda::stochasticLogDetBatched`) instead of round-tripping
  through the host once per Lanczos step. Previously, even with a GPU
  backend bound, the SLQ term called the CPU-orchestrated
  `LinearAlgebra::stochasticLogDetBatched` with the GPU matvec as its
  `AmulBatched` callback: every one of the 40 (default) Lanczos steps
  uploaded that step's probe vectors, computed `R*V` on device, downloaded
  the result, then did reorthogonalization/dot-products/bookkeeping on the
  host in Armadillo before uploading again for the next step. The new path
  keeps every probe's entire Krylov history resident in one device buffer
  for the whole recurrence: step j's vectors are a contiguous n×nprobe
  block (feeds the matvec directly, no gather needed) and, in the SAME
  buffer, probe p's history across steps is a standard column-major matrix
  with leading dimension `nprobe*n` (feeds `cublasDgemmStridedBatched`
  directly) — full reorthogonalization against every prior step becomes two
  batched cuBLAS calls instead of `nprobe*(j+1)` separate host-side
  dot/axpy pairs. Only `nprobe*lanczos_steps` scalars (α/β) come back to
  the host, once, for the final small tridiagonal eigendecompositions.
  Same estimator/numerics as before (bit-matching cross-checked against the
  prior implementation; +10 assertions in `KrigingIterativeTest.cpp`), pure
  performance change. Measured on an H100 (isolated, uncontended timings —
  the shared node's GPUs otherwise carry heavy multi-tenant noise): the
  no-grad path (`[F|y]` CG + SLQ, unaffected by the separate probe-CG cost)
  is 1.6-2.4x faster at n=2000/4000/8000; on the full log-likelihood+
  gradient evaluation the gain shrinks to ~10% at n=8000 and becomes
  negligible at n=16000/32000, because the gradient's Hutchinson-probe CG
  solve (untouched by this change) dominates total time at that scale —
  the next lever for those sizes is fusing `[F|y|probes]` into one Krylov
  pass (mBCG) rather than three, not the SLQ term. Scoped to the
  unpreconditioned CUDA case for now: HIP/SYCL/Metal and the
  Nystrom-preconditioned SLQ path still use the prior CPU-orchestrated
  ping-pong (documented in the new function's doc comment).
- Python: scikit-learn compatible estimators for all four Kriging classes —
  `KrigingRegressor`, `WarpKrigingRegressor`, `MLPKrigingRegressor`,
  `NestedKrigingRegressor` in `pylibkriging.sklearn`, implementing the
  scikit-learn estimator API (`fit`/`predict`, `get_params`/`set_params`,
  `clone`) so they drop into `Pipeline` and `GridSearchCV` (#338).
- Cross-package comparison benchmark (`bench/comparison/`): libKriging vs.
  scikit-learn/GPy/SMT/OpenTURNS (Python) and DiceKriging/RobustGaSP (R) on
  shared randomized LHS designs (Branin, Hartmann-3/6, Borehole), reporting
  fit/predict time, RMSE, Q², NLPD; runs on a manual/monthly CI workflow (#335).
- `predictIterative`: matrix-free conjugate-gradient alternative to `predict()`
  that solves each prediction on the fly instead of keeping a dense Cholesky
  factor resident, with optional Nystrom-preconditioned CG. `LLIterative(m)`:
  matching matrix-free fit objective (CG linear solves, stochastic Lanczos
  quadrature log-determinant, Hutchinson trace-gradient estimator), also with
  optional Nystrom preconditioning, plus `updateIterative` for incremental
  refits. The objective string also takes an optional third argument,
  `LLIterative(m,precond_rank,lanczos_steps)`, exposing the SLQ Lanczos
  step count per probe (default 20) on every binding — raise it when the
  stochastic log-determinant drifts from the exact objective on an
  ill-conditioned covariance; `precond_rank` now also accepts 0 (= no
  preconditioner). `Kriging::subsetOfData`: k-means (or random) pre-fit row-subsetting
  for large designs. Available in the core C++ API and all four bindings
  (Python/R/Julia/Octave-MATLAB); see `docs/math/Iterative.md`,
  `PredictIterative.md`, `SubsetOfData.md`, `Scalability.md` and the
  comparison-vs-GPyTorch notebooks (#347).
- GPU acceleration for the iterative path (`objective="LLIterative(m)"` /
  `predictIterative`), opt-in only (`-DENABLE_CUDA_ITERATIVE=ON`; also the
  UNVERIFIED `-DENABLE_HIP_ITERATIVE` / `-DENABLE_SYCL_ITERATIVE` (Intel
  oneAPI) / `-DENABLE_METAL_ITERATIVE` (Apple, float32-only — MSL has no
  double) ports, none of which have a toolchain to build/run against): the
  matrix-free CG solves, the
  Stochastic Lanczos Quadrature log-determinant (via a lockstep-Lanczos
  `LinearAlgebra::stochasticLogDetBatched`), the Hutchinson trace-gradient
  `dR/dtheta` matvec, and a device-side Nystrom/Woodbury CG preconditioner
  all run batched on the device, with the CG's per-iteration alpha/beta and
  convergence scalars kept on-device (no host round-trip per iteration).
  New `pylibkriging` binding `logLikelihoodIterativeFun` (the O(n^2)
  iterative objective, distinct from `logLikelihoodFun`'s exact O(n^3) one).
  New standalone (non-CI) GPU benchmark `bench/gpu/`: one fixed sweep at
  `theta=0.15` comparing `libKriging-Cholesky-<BLAS>` (reference),
  `libKriging-Iterative-CUDA` / `-OpenMP` and `GPyTorch-BBMM-CUDA` / `-<BLAS>`
  on fit / logLik / predict time and accuracy, writing a per-machine
  Markdown/CSV to `bench/gpu/results/`.
- `nystrom_rank()` accessor exposed in the Julia, Octave/MATLAB and R
  bindings (Python already had it); worked notebooks
  `docs/math/llnystrom_vs_cholesky.ipynb` / `llvecchia_vs_cholesky.ipynb`
  comparing `LLNystrom`/`LLVecchia` against exact Cholesky (#358).

### Changed
- Iterative path (`LLIterative` / `predictIterative`), CPU as well as GPU:
  an `optim="none"` light fit no longer builds the dense `d x n^2` pairwise
  distance tensor it never reads (was several GB at n=8000); the matrix-free
  matvecs are no longer capped at 2/8 OpenMP threads (`OMP_NUM_THREADS`
  governs them now); `predictIterative`'s posterior-variance CG solves use
  `sqrt(tol)` (the variance needs far less accuracy than the mean).
- `LLIterative` CPU objective/gradient: the matrix-free `R * V` is now
  evaluated once per iteration for the *whole* right-hand-side block
  (`LinearAlgebra::conjugateGradientBatched`, and the SLQ log-determinant's
  lockstep Lanczos through the same engine), instead of one full
  covariance sweep per column — every transcendental `Cov(x_i - x_j, theta)`
  is reused across all `nprobe` probes / all `d` gradient directions. The
  `dR/dtheta` matvec is batched the same way.
- `LLIterative` CPU objective/gradient, dense fast path: for a separable
  kernel (`gauss` / `exp` / `matern3_2` / `matern5_2`) and an `n` whose
  dense `n x n` `R` — plus the `d` `dR/dtheta_k` blocks when a gradient is
  wanted — fits a memory budget (6 GiB default, `LK_ITERATIVE_DENSE_MAX_MB`;
  `0` forces the strictly matrix-free path), `R` is materialized ONCE with
  an inlined, parallelized symmetric build and every subsequent CG
  iteration / Lanczos step is a BLAS-3 `R * V` instead of a fresh
  transcendental-heavy covariance sweep. Combined with the batching above,
  ~100x faster `logLik` + gradient at `n = 2000` (235 s → ~2 s at
  `OMP_NUM_THREADS=32`); results unchanged to the SLQ noise floor. GPU
  path unaffected (it has its own device matvecs). The builder
  (`KrigingImpl::build_separable_cov`) is shared with `predictIterative`
  (below).
- `predictIterative`: gets the same CPU speedups as `LLIterative` above —
  its mean/stdev/GLS-correction CG solves now share one batched matvec
  (`LinearAlgebra::conjugateGradientBatched`, replacing the old
  thread-per-column `conjugateGradient`) and, within the same
  `LK_ITERATIVE_DENSE_MAX_MB` budget, run against a dense `R` materialized
  once per call instead of a matrix-free sweep per CG iteration. ~40x
  faster at `n = 4000` (50.7 s → 1.3 s); results unchanged to ~1e-7.
- CUDA iterative path gets its own dense fast path (`build_cov_kernel` in
  `CudaLinearAlgebraKernel.cu` + `cublasDgemm`, one CUDA thread per `(i,j)`
  pair, governed by `LK_ITERATIVE_CUDA_DENSE_MAX_MB` (default 4096 MiB,
  independent of the CPU budget)): `LinearAlgebraCuda::conjugateGradient`/
  `rmulBatched`/`dRmulBatched` materialize `R` (or the `dR/dtheta_k`
  blocks) once per call instead of recomputing every covariance entry's
  transcendentals on every CG iteration / Lanczos step — the hand-written
  `rmul_batched_kernel`/`drmul_batched_kernel` had never gotten this
  treatment, so `libKriging-Iterative-CUDA` had fallen *behind* the CPU's
  new dense path at every `n` in `bench/gpu`. Now ~25x faster at `n=2000`
  logLik+gradient, ~114x at `n=4000` (8.0 s → 0.32 s, 137.5 s → 1.2 s);
  `predictIterative` ~1.4-3x. Adds a `CUDA::cublas` link dependency
  (`ENABLE_CUDA_ITERATIVE` builds only). Results unchanged to the SLQ/CG
  noise floor.
- `bench/gpu`: GPyTorch's `max_cholesky_size` (default 800) was left at
  its default, so the `GPyTorch-BBMM-*` rows at `n=250`/`500` — below that
  threshold — were silently running an exact dense Cholesky solve instead
  of BBMM despite the raised CG/Lanczos settings, confirmed empirically
  (single-digit-millisecond eval time; `-mll` differing from a forced-BBMM
  run by ~0.06-0.16, an order of magnitude more than `n=900`/`2000`
  disagreed with themselves). Now forces `max_cholesky_size(0)` so every
  `n` in the sweep genuinely runs BBMM, matching the `-BBMM` name; RMSE/Q²
  are unchanged (BBMM already converges to the same accuracy at these
  `n`/θ), only the previously-too-fast small-`n` timings correct upward.
- `bench/gpu`: adds `GPyTorch-Cholesky-CUDA`/`-<BLAS>` backends — the same
  model as `GPyTorch-BBMM-*` with `max_cholesky_size` forced far above
  every `n` instead of `0`, so GPyTorch always solves exactly. A second,
  GPyTorch-only reference (independent of libKriging's Cholesky) that
  isolates whether BBMM has converged from the unrelated
  covariance-argument-convention offset between the two libraries; the
  generated report's Verdict section now reports the max BBMM-vs-its-own-
  Cholesky posterior-mean gap directly. `n=4000` also added to the sweep
  (`--sizes`, default unchanged).
- `bench/gpu`: GPyTorch's `fit` column was not a real fit — `gpytorch.models
  .ExactGP.__init__` does no linear algebra (confirmed empirically: ~1ms
  flat regardless of `n`), so all of GPyTorch's kernel/solve cost was
  silently deferred to the first `logLik` call. `run_gpytorch` now forces
  one no-grad `mll(model(x), y)` forward (a Cholesky, or one BBMM CG+SLQ
  solve) inside the timed `fit` step, mirroring what libKriging's
  `Kriging(...)` constructor already does at a fixed theta (one dense
  Cholesky, or `LLIterative`'s one CG+SLQ commit) — `fit` now genuinely
  scales with `n` for every GPyTorch backend (e.g. `GPyTorch-BBMM-CUDA`:
  0.11s → 0.52s from `n=250` to `4000`, was flat ~0.001s before). `logLik`
  still independently re-solves from scratch (GPyTorch's train-mode
  forward has no cache), same as before.
- `LLIterative(m,precond_rank)`: the Nystrom preconditioner is now applied
  to the SLQ log-determinant too, not only the CG solves — the Lanczos
  quadrature runs on the whitened `Rtilde = L^-1 R L^-T` (`L L' = P`, via
  the new `WoodburyFactorization::whitenL`/`whitenLt`) and `log|P|` is added
  back exactly, so a preconditioner also curbs the `log|R|` bias on an
  ill-conditioned R (previously only `lanczos_steps` did). GPU path
  unchanged. See `docs/math/Iterative.md`.
- Python: dropped the `numpy<2` pin — `pylibkriging` now supports NumPy 2.x.
  Required bumping the vendored `pybind11` (2.10.1 → 2.13.6) and `carma`
  submodules, since both hardcode offsets into NumPy's C-API function table
  and predated the NumPy 2.0 ABI changes; also fixed a bug in carma's own
  NumPy-2.0 fix where `PyArray_CopyInto`'s table offset (which differs
  between NumPy 1.x and 2.x) was hardcoded to the NumPy-2-only value instead
  of being picked at runtime (#339, libKriging/carma#1).

### Fixed
- R binding: `Kriging`'s `objective` validator rejected every `LLIterative` form
  with more than three fields *before* the string ever reached C++, so the
  `cg_max_iter_mult` and `cg_tol` fields were unusable from R. The regex now
  accepts all five, including the real-valued `cg_tol` (`1e-6`, `0.01`), and
  carries a comment pointing at `Kriging::parse_iterative_m` as the thing it
  must stay in sync with.
- `bench/gpu`: the GPyTorch side of the comparison was not fitting libKriging's
  model, which made the harness's `dMean/rms` accuracy column meaningless. Two
  mismatches: (1) `MaternKernel(nu=2.5, ard_num_dims=d)` is *radial after
  scaling* — a single Matern function of `||dx/l||_2` — whereas libKriging's
  `matern5_2` is *separable*, `prod_k f(|dx_k|/theta_k)`; at d=4, theta=0.15,
  dx=(0.05,0.10,0.02,0.07) these are 0.589472 vs 0.556929. (2) GPyTorch's fixed
  `ConstantMean` was set to `mean(y)`, the OLS constant, while libKriging does
  universal kriging and profiles out the *GLS* trend `beta`. Together these put
  a constant O(1e-2) posterior-mean offset on every GPyTorch row — including
  the **exact** `GPyTorch-Cholesky` ones, which is the tell — drowning the
  ~1e-3 iterative-convergence signal the column exists to show. GPyTorch now
  runs a `ProductKernel` of one-dimensional Matern-5/2 factors with
  `ConstantMean = beta`; `GPyTorch-Cholesky`'s `dMean/rms` against the
  libKriging Cholesky reference drops from 4.2e-02 to **4.1e-08**, and RMSE/Q2
  now agree to six digits across all seven backends. Costs GPyTorch ~13% on
  `predict` (d lazily-evaluated kernels instead of one fused one).
- `LLIterative` objective: the CG relative-residual tolerance was hardcoded to
  `1e-8` and unreachable from any binding, so every evaluation drove its linear
  solves eight orders of magnitude tighter than the *stochastic* SLQ
  log-determinant sitting next to them — accuracy the returned log-likelihood
  cannot use. Measured on an H100 at n=4000, d=4, matern5_2, theta=0.15, that
  was 6550 CG iterations on the 30 Hutchinson probes against ~100 at `1e-4`,
  for a log-likelihood identical to 4 significant digits. The default is now
  `1e-4` (GPyTorch's BBMM makes the same trade with its `cg_tolerance`), and it
  is settable as a 5th field:
  `objective="LLIterative(m,precond_rank,lanczos_steps,cg_max_iter_mult,cg_tol)"`.
  Log-likelihood
  + gradient at n=4000 on an H100: 0.94 s -> 0.40 s; on the OpenMP backend the
  same benchmark point went 15.9 s -> 7.1 s.
- `LLIterative` objective: `cg_tol` (5th field, above) was shared by the
  `[F|y]` solve AND the gradient's Hutchinson-probe solve
  (`W = R⁻¹·probes`), but the two need very different iteration counts to
  reach the SAME relative-residual tolerance — CG iterations-to-tolerance
  depend on the right-hand side's spectral content, not just `cond(R)`:
  `F`/`y` are smooth and project mostly onto `R`'s dominant eigenmodes, the
  isotropic Rademacher probes don't, and the gap widens as `R` gets more
  ill-conditioned (fixed θ, growing `n` in a fixed-volume domain). Measured
  on an H100 at n=2000/4000/8000, d=4, matern5_2, θ=0.15, `cg_tol=1e-4`:
  `[F|y]` iterations go 180 → 400 → 1100 (×2.2, ×2.75), the probe solve's go
  900 → 2270 → 13470 (×2.5, ×5.9) — at n=8000 the probe solve alone
  dominates the evaluation, and the shared `cg_tol` gave no way to loosen it
  without also loosening `predictIterative`'s own (unrelated) tolerance.
  `probes_cg_tol` is now a separate, optional 6th field
  (`objective="LLIterative(m,precond_rank,lanczos_steps,cg_max_iter_mult,cg_tol,probes_cg_tol)"`,
  defaults to `cg_tol` when omitted, so existing specs are unaffected). It
  feeds the same stochastic Hutchinson trace estimate whose own sampling
  error already swamps a tight tolerance (same argument as `cg_tol` itself),
  so loosening it costs no measurable accuracy: at n=8000,
  `probes_cg_tol=1e-2` keeps the relative gradient error (0.06%) an order of
  magnitude below the 0.66% already accepted for `cg_tol` itself, while the
  log-likelihood value is bit-identical (it does not depend on
  `probes_cg_tol` at all — only the gradient does). `bench/gpu/bench_gpu.py`
  exposes it as `--lk-probes-cg-tol`, now **defaulting to `1e-2`** (was:
  tracked `--cg-tol`, libKriging-only, no GPyTorch equivalent, so it need
  not be budget-matched) — regenerated H100 results:
  `libKriging-Iterative-CUDA`'s `logLik` (value+gradient) drops from 3.19 s
  to **1.49 s** at n=8000 (−53%) and 0.28 s to 0.19 s at n=4000 (−33%);
  `dMean/rms`/`dLogLik/n` unchanged at every `n` in the sweep. Confirmed
  this is unlike loosening the shared `--cg-tol`, which degrades
  `predictIterative`'s `dMean/rms` by up to 35× with no compensating gain
  (`predictIterative`'s CG has no stochastic floor to hide behind). See
  `docs/math/Iterative.md`.
- `bench/gpu/bench_gpu.py`: extended the sweep to n=16000 and n=32000
  (`SWEEP_EXTRA_SIZES_BY_KEY`) for the two backends built to scale past
  n=8000 — `libKriging-Iterative-CUDA` reaches n=32000,
  `GPyTorch-BBMM-CUDA` reaches n=16000. The exact-Cholesky backends
  (`libKriging-Cholesky-*`, `GPyTorch-Cholesky-*`) stay capped at n=8000
  regardless of `--sizes`: an O(n³) dense factorization is not meant to
  scale past that, and no `chol` reference row exists past it either, so
  `dMean/rms`/`dLogLik/n` are simply absent for n>8000 (pure scalability
  numbers, not an accuracy comparison). Two fixes were needed to get there:
  - `libKriging-Iterative-CUDA`'s device dense-R cache
    (`LK_ITERATIVE_CUDA_DENSE_MAX_MB`, default 4096 MiB) is too small for
    the gradient's `dR/dtheta_k` blocks past n≈11585 (d=4) — at n=32000
    they alone need ~31 GiB, so both the CG solves AND the gradient
    silently fell back to a matrix-free kernel that recomputes the full
    O(n²) covariance sum on every iteration instead of one dense build
    reused via `cublasDgemm`. Raising the budget
    (`LK_ITERATIVE_CUDA_DENSE_MAX_MB=40960`) cut n=32000's `logLik` from
    open-ended (still running after 4+ hours) to **864 s**, and n=16000's
    from 50.9 s to 38.6 s, on an H100.
  - `GPyTorch-BBMM-CUDA`'s `max_cg_iterations` was hardcoded to 5000,
    tuned for n≤8000 and silently meaningless past it: at n=32000 CG
    terminated with an average residual norm of 0.857 against a 1e-4
    target (essentially unsolved, not just under-tolerance). It now scales
    with `n` the same way libKriging's own `cg_max_iter_mult` does
    (`LK_ITER_CG_MAX_ITER_MULT * n`, floored at 5000), making the shared
    n=16000 row an apples-to-apples comparison of the same iteration
    budget. `GPyTorch-BBMM-CUDA` still stops at n=16000: n=32000 hits a
    hard `CUDA out of memory` building the BBMM pipeline itself (~78 GiB,
    confirmed independent of the CG iteration budget — 38x more
    iterations changed nothing) on a card libKriging's matrix-free kernels
    use ~1.5 GiB on at the same n; not something this script's settings
    can address.
- CUDA iterative backend: `conjugateGradient`, `rmulBatched` and `dRmulBatched`
  were stateless, each re-uploading `X`/`theta` and rebuilding the dense `R`
  from scratch (a 128 MB allocation and 16 M fp64 `exp`/`log1p` at n=4000) on
  every call. Since one objective evaluation runs three Krylov passes at a fixed
  `theta`, and the SLQ recurrence calls `rmulBatched` once per Lanczos step,
  that constant matrix was being rebuilt 40+ times per evaluation for ~0.02 ms
  of useful GEMM each time. `R` is now cached on the device and keyed by value
  on `(n, dimX, covType, X, theta)`, so it is built once per `theta`.
- CUDA iterative backend: the Nyström/Woodbury preconditioner applied its
  `k x k` triangular solve with one thread per right-hand-side column running
  an `O(k^2)` strictly serial substitution (`trisolve_MMt_kernel`), plus two
  hand-written GEMMs — so a 300-column solve kept ~300 threads busy on a
  132-SM device and the preconditioner cost an order of magnitude more than
  the CG iterations it was meant to save. It is now `cublasDgemm` /
  `cublasDtrsm` / `cublasDgemm` around two elementwise kernels. Measured on an
  H100 at n=4000, d=4, matern5_2, theta=0.15: `predictIterative` with a rank-128
  Nyström preconditioner 1.78 s -> 0.46 s, rank-512 9.20 s -> 0.80 s;
  `LLIterative(30,128,40)` log-likelihood + gradient 12.08 s -> 2.21 s. The
  preconditioned path now costs ~2.2x the unpreconditioned one instead of ~13x,
  for bit-comparable results (log-likelihood value unchanged; CUDA-vs-CPU
  posterior means agree to 6e-08, i.e. to the requested CG tolerance).
- `bench/gpu/bench_gpu.py` benchmarked libKriging's `predictIterative` at
  `tol=1e-8` against GPyTorch's BBMM at `eval_cg_tolerance=1e-4` — a solver
  converged four orders of magnitude tighter against a deliberately truncated
  one, with the difference reported as speed. The CG tolerance is now a single
  `--cg-tol` option (default `1e-4`) driving **both** libraries, the Nyström
  rank is `--lk-precond-rank` instead of a hardcoded 128, the generated report
  states the shared budget, and the "the dense Cholesky path is fastest on all
  three operations" verdict — which no longer holds — is computed from the
  measurements instead of hardcoded. The `run_libkriging_iter` warm-up call was
  also silently failing, so the first GPyTorch point absorbed CUDA context
  initialisation.
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
  the design.
- Windows: Python binding processes silently hanging (looking like ~1h CI
  timeouts) were actually undetected heap corruption from Armadillo's
  aligned allocator never being routed through libKriging's own allocator
  indirection (`lkalloc`) despite the Python binding requesting it at module
  init — re-enabled the wiring (#354, #357). The MSVC Debug CRT's blocking
  error dialog is now also redirected to stderr so any future corruption
  fails fast with a diagnosable message instead of hanging CI (#356).
- Windows: Python binding CI jobs hanging on OpenMP thread-pool churn from
  repeated parallel regions during BFGS, the same mechanism as an earlier
  Octave Windows fix — forced `OMP_NUM_THREADS=1` for Windows Python builds
  (#351, #352).
- Octave Windows: flaky `predictNystrom` test assertion caused by free-BFGS
  convergence varying across platforms/compilers — compared against a
  deterministic fixed-theta fit instead; also fixed `optim="none"` silently
  ignoring `LLNystrom(k)` and doing an exact fit instead of honoring the
  requested objective (#353).
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

### Documentation
- Added a coding-agent skill covering libKriging usage patterns (#336) and a
  "Known pitfalls" section to `AGENTS.md` (#333).

### CI/Release process
- Automated `jlibkriging` registration on Julia's General registry (#332).
- GitHub release notes are now filled in from this changelog (#334).

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

[Unreleased]: https://github.com/libKriging/libKriging/compare/v1.1.0...master
[1.1.0]: https://github.com/libKriging/libKriging/compare/v1.0.0...v1.1.0
