# GPU comparison benchmark — GPyTorch vs libKriging (n > 1000)

A **local**, GPU-only counterpart to `../comparison` (which does default
MLE fits on CPU and stops at n = 1000). This one runs on a single CUDA
device — an **NVIDIA H100** here — and compares the two packages'
*matrix-free conjugate-gradient* linear algebra at training sizes the CPU
benchmark can't reach:

| | scalable GP path exercised |
|---|---|
| **GPyTorch** | BBMM — CG solves via `LinearOperator` matvecs + pivoted-Cholesky preconditioning, on GPU |
| **libKriging** | `objective="LLIterative(m)"` fit + `predictIterative` — matrix-free CG + SLQ log-det, with the opt-in **CUDA backend** (`-DENABLE_CUDA_ITERATIVE=ON`, `src/lib/cuda/`) |

It is the runnable, larger-n, GPU version of the story in
[`docs/comparisons/libKriging_vs_GPyTorch.ipynb`](../../docs/comparisons/libKriging_vs_GPyTorch.ipynb) §4.

## Protocol

- **Functions**: `sine_sum` (`Σ sin(2πxᵢ)`, d=4), `hartmann3` (d=3),
  `hartmann6` (d=6), `borehole` (d=8, raw physical units) — from
  `../comparison`. Training sizes configurable in `make_datasets.py`'s
  `CASES` (**2000, 5000** by default); common 2000-point LHS test set;
  repetitions seeded per `(n, rep)`, all designs shared bitwise across
  backends.
- **Sweeps** (`run_gpu.py`): `--theta-frac` (comma list — reference
  length-scale as a fraction of the input range; small = well-conditioned
  R, large = ill-conditioned), `--kernel` (`matern5_2`/`matern3_2`/
  `gauss`/`exp`), `--backends` (`gpytorch`, `gpytorch-cpu`,
  `libkriging-gpu`, `libkriging-cpu`).
- **Fixed, shared hyperparameters.** Every backend is evaluated at the
  same `theta_ref` = `0.3 × (per-dimension input range)`, `sigma2 = var(y)`
  (`theta_ref.csv`, written by `make_datasets.py`). The benchmark measures
  each package's CG linear algebra, **not** its optimizer. The MLE optimum
  is deliberately *not* used: for these smooth surfaces it sits at a very
  long length-scale where the pure-interpolation kernel has condition
  number ~1e16 and *neither* side's CG converges (a real effect, already
  documented in the notebook) — see `make_datasets.py`'s docstring.
- **Interpolation only**, jitter 1e-10, no nugget — matching
  `../comparison`; libKriging's `LLIterative` is `NoiseModel::None`-only.
- **Three phases** per case (mirrors the CI benchmark's fit / predict /
  update):
  - `fit` — one likelihood(+grad) evaluation at `theta_ref`
    (one optimizer-step-equivalent): GPyTorch `-mll(model(x),y).backward()`;
    libKriging `Kriging(..., optim="none", "LLIterative(m)")` +
    `logLikelihoodFun(theta, grad=True)`.
  - `predict` — posterior **mean** on the full test set (→ RMSE, Q²), plus
    **stdev** on a `--stdev-n`-point subsample (→ NLPD, coverage90).
    stdev is one CG solve *per test point* on the libKriging side, hence
    subsampled.
  - `update` — re-condition on `n + ⌈0.25n⌉` fresh LHS points at the same
    `theta_ref`, then predict the test mean again (full rebuild on both
    sides, so directly comparable).
- **Backends** (`--backends`): `gpytorch`, `libkriging-gpu`,
  `libkriging-cpu` (CPU-CG baseline — same binary, toggled at runtime via
  `pylibkriging.set_cuda_iterative_enabled`). Each `(func, n, rep,
  backend)` runs in its own subprocess with a wall-clock budget
  (`--budget`, default 1800 s); timeouts/errors are recorded, never fatal.
- **Report**: `results/summary.md` — median [q25; q75] over repetitions per
  `(func, n)`, plus `results/all.csv`.

## Build & run

Not wired into CI (needs a specific GPU + a CUDA build of libKriging).
Run it locally:

```sh
# 1. libKriging with the CUDA iterative backend + Python binding
cmake -B build-cuda -S . -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_PYTHON_BINDING=ON -DENABLE_JULIA_BINDING=OFF \
      -DENABLE_CUDA_ITERATIVE=ON -DCMAKE_CUDA_ARCHITECTURES="90;89" \
      -DPYTHON_EXECUTABLE="$VENV/bin/python"
cmake --build build-cuda --target install -j

# 2. Python deps (CUDA-matched torch wheel!)
"$VENV/bin/pip" install "numpy<2" scipy pandas gpytorch
"$VENV/bin/pip" install --index-url https://download.pytorch.org/whl/cu126 torch==2.8.0

# 3. environment: pin the GPU, expose the fresh pylibkriging
export CUDA_VISIBLE_DEVICES=1                 # the H100
export PYTHONPATH=build-cuda/installed/bindings/Python
export LD_LIBRARY_PATH=build-cuda/installed/lib64:$LD_LIBRARY_PATH

# 4. datasets + run + report
cd bench/comparison-gpu
./run.sh                                      # REPEATS/QUICK/BUDGET/BACKENDS env knobs
# or step by step:
python make_datasets.py --repeats 3           # or --quick
python run_gpu.py --budget 1800 --out results/gpu.csv
python aggregate.py --results results --out results/summary.md
```

`run_gpu.py --backends gpytorch` (etc.) restricts to a subset;
`--sizes` is fixed in `make_datasets.py`'s `CASES`.

## Fairness caveats

- Fixed shared `theta` isolates the **linear algebra**; this is not a
  comparison of optimizers or of end-to-end MLE fits.
- GPyTorch's `max_cg_iterations` / `cg_tolerance` and libKriging's
  `predictIterative` `tol` are each a speed vs convergence knob; the
  `CG conv` column flags whether GPyTorch's CG hit its iteration cap
  before tolerance. libKriging's *fit-side* CG budget isn't tunable from
  Python (see `docs/math/Iterative.md`) and is reported as converged.
- libKriging's `predictIterative` matvec is not BLAS-batched (a per-pair
  covariance call); the CUDA backend (`src/lib/cuda/CudaLinearAlgebraKernel.cu`)
  batches/tiles it, but it is still a different implementation strategy
  from GPyTorch's fused tensor ops — any gap reflects that, not one side
  doing structurally less work.
- The GPUs on this machine are shared; absolute times carry contention
  noise. Compare **within** a run, not across runs.
