# `bench/gpu/` — manual GPU vs CPU vs GPyTorch benchmark

`bench_gpu.py` is a **standalone, run-by-hand** benchmark (not wired into
CI) for libKriging's iterative matrix-free CG path
(`objective="LLIterative(m)"` → `logLikelihoodIterativeFun`). It runs one
fixed sweep — `sine_sum`, d=4, matern5_2, shared `theta=0.3`,
n ∈ {250, 500, 1000, 2000, 4000, 8000} — for four backends:

| backend | what it exercises |
|---|---|
| `libkriging-gpu` | `set_cuda_iterative_enabled(True)` — device-batched CG + SLQ log-det + Hutchinson gradient |
| `libkriging-cpu` | `set_cuda_iterative_enabled(False)` — the OpenMP CPU path |
| `gpytorch-gpu` | GPyTorch BBMM (`-mll` forward+backward) on `cuda` |
| `gpytorch-cpu` | GPyTorch BBMM on `cpu` |

Each run writes `results/<GPU-slug>__<CPU-slug>.md` (+ a `.csv`) — the file
name encodes the machine, so results from several machines can be committed
side by side and compared. Re-running on the same machine overwrites its
file.

## Running it

Needs `pylibkriging` (built with a GPU iterative backend for the `*-gpu`
rows — `-DENABLE_CUDA_ITERATIVE=ON`, etc.), plus `torch` and `gpytorch`
importable from the same environment. See
[`bench/comparison-gpu/README.md`](../comparison-gpu/README.md) and the
`gpu-bench-env` project note for the build/env recipe used on the reference
machine.

```sh
# whole sweep, all backends, auto-named output
python bench/gpu/bench_gpu.py

# quicker smoke run
python bench/gpu/bench_gpu.py --sizes 250,500,1000 --backends libkriging-gpu,libkriging-cpu

# machine with no CUDA: the *-gpu backends are dropped automatically
python bench/gpu/bench_gpu.py
```

Useful flags: `--sizes`, `--theta`, `--nprobe`, `--backends`,
`--cpu-budget` / `--gpu-budget` (seconds; a backend stops once the next
size projects to exceed it — the CPU iterative path is O(n³)-ish in this
ill-conditioned regime), `--tag`, `--outdir`.

## Reading the output

* `fit` — one `logLikelihoodIterativeFun` (libKriging) or one `-mll`
  forward+backward (GPyTorch). This is the head-to-head number.
* `build` — libKriging's `Kriging(...)` constructor (always factorizes the
  exact dense R at `optim="none"`, regardless of objective).
* `RMSE`/`Q²`/`LOO` — libKriging's **exact** `predict()` / `leaveOneOut()`,
  *not* the iterative path; the iterative objective's own fidelity is the
  `llErr/pt` column (|iterative − exact| concentrated log-likelihood per
  training point, n ≤ 2000).
* The `theta=0.3` regime is deliberately ill-conditioned — see the Notes
  section each report emits, and
  [`docs/comparisons/libKriging_vs_GPyTorch.ipynb`](../../docs/comparisons/libKriging_vs_GPyTorch.ipynb)
  §4 / [`../comparison-gpu/ANALYSIS.md`](../comparison-gpu/ANALYSIS.md).
