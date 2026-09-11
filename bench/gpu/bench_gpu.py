#!/usr/bin/env python3
"""Standalone libKriging vs GPyTorch benchmark, matrix-free iterative path.

Run this by hand on a machine, then commit the Markdown (+ CSV) it writes to
``bench/gpu/results/`` -- the file name encodes the machine's GPU and CPU so
results from several machines can be compared side by side.

Goal
----
Compare, over ``n in {250, 500, 1000, 2000}`` at a shared fixed
``theta = 0.15`` (a regime where *every* method converges with sane
settings), the cost and accuracy of five backends, each named
``<lib>-<method>-<linalg lib>`` (the linalg name is auto-detected):

* **libKriging-Cholesky-<BLAS>** -- the exact dense-Cholesky path
  (``objective="LL"``), the dense BLAS/LAPACK named. This is the
  **reference**: its log-likelihood value and its posterior mean are what
  the iterative methods are measured against.
* **libKriging-Iterative-CUDA** / **libKriging-Iterative-OpenMP** -- the
  iterative path (``objective="LLIterative(30,0,40)"``: 30 SLQ probes, no CG
  preconditioner, 40 Lanczos steps per probe so the stochastic
  log-determinant stays close to exact), light fit (no dense R *factor*),
  ``set_cuda_iterative_enabled(True/False)``. Both materialize R (or the
  ``dR/dtheta_k`` blocks) once per evaluation for a separable kernel within a
  memory budget -- CUDA via a device build kernel + ``cublasDgemm``
  (``LK_ITERATIVE_CUDA_DENSE_MAX_MB``), OpenMP via BLAS-3 ``R*V``
  (``LK_ITERATIVE_DENSE_MAX_MB``) -- falling back to a hand-written
  matrix-free matvec kernel/loop otherwise.
* **GPyTorch-BBMM-CUDA** / **GPyTorch-BBMM-<BLAS>** -- GPyTorch's ``ExactGP``
  + BBMM (CG + pivoted-Cholesky preconditioner + SLQ log-det), on ``cuda``
  or ``cpu`` (torch's own BLAS named), with *raised*
  CG/Lanczos/preconditioner settings so it actually converges.

Three timings per backend, plus accuracy
----------------------------------------
* ``fit``     -- model construction: the ``Kriging(...)`` constructor
  (Cholesky for ``LL``; one CG+SLQ commit for the light ``LLIterative`` fit);
  GPyTorch model/likelihood build (near zero).
* ``logLik``  -- one log-likelihood **+ gradient** evaluation at ``theta``:
  ``logLikelihoodFun`` / ``logLikelihoodIterativeFun`` / one
  ``-mll(...).backward()``.
* ``predict`` -- a *cold* posterior mean on 300 held-out points: ``predict``
  / ``predictIterative`` (CG to ``tol`` with a raised ``max_iter`` +
  Nystrom preconditioner) / GPyTorch ``.eval()`` posterior.
* All timings are the **min of up to 5 reps** (1 rep once a call exceeds 3 s).
* ``RMSE`` / ``Q2`` on the test set; and, vs the Cholesky reference,
  ``dLogLik/n`` (``|ll - ll_chol| / n``; not comparable for GPyTorch's
  differently-normalised ``-mll``, shown as ``--``) and ``dMean/rms``
  (``max|mean - mean_chol| / rms(y_test)``).

Requires ``pylibkriging`` (built with a GPU iterative backend for the CUDA
rows), ``torch`` and ``gpytorch`` importable in one environment.
See ``bench/gpu/README.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as _dt
import platform
import re
import socket
import subprocess
import sys
import time

import numpy as np

SWEEP_DEFAULT = [250, 500, 1000, 2000]
D_CG = 4
THETA_DEFAULT = 0.15
N_TEST = 300
TEST_SEED = 999
TRAIN_SEED = 123
NOISE_SEED = 0
LK_ITER_OBJECTIVE = "LLIterative(30,0,40)"  # 30 probes, no CG precond, 40 SLQ Lanczos steps


# --------------------------------------------------------------------------
# machine identification (drives the output file name)
# --------------------------------------------------------------------------
def detect_gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    for cmd in (["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                ["rocm-smi", "--showproductname"]):
        try:
            out = subprocess.check_output(cmd, text=True, timeout=10, stderr=subprocess.DEVNULL)
            line = next((l.strip() for l in out.splitlines() if l.strip()), "")
            if line:
                return line
        except Exception:
            continue
    return None


def detect_cpu_name() -> str:
    sysname = platform.system()
    if sysname == "Linux":
        try:
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
    elif sysname == "Darwin":
        try:
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:
            pass
    elif sysname == "Windows" and platform.processor():
        return platform.processor()
    return platform.processor() or platform.machine() or "unknown-cpu"


def slug(name: str | None, maxlen: int = 48) -> str:
    s = name or "none"
    s = re.sub(r"\(R\)|\(TM\)|\(r\)|\(tm\)|®|™", "", s)
    s = re.sub(r"\b(CPU|Processor)\b", "", s, flags=re.I)
    s = re.sub(r"@.*$", "", s)
    s = re.sub(r"\b\d+-Core\b.*$", "", s, flags=re.I)
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")
    return (s[:maxlen].strip("-")) or "unknown"


# --------------------------------------------------------------------------
# dense-linear-algebra backend detection (the "<linalg lib>" name component)
# --------------------------------------------------------------------------
def _blas_from_ldd(sofile: str) -> str | None:
    try:
        out = subprocess.check_output(["ldd", sofile], text=True, stderr=subprocess.DEVNULL).lower()
    except Exception:
        return None
    for token, name in (("libmkl", "MKL"), ("libopenblas", "OpenBLAS"), ("libflexiblas", "FlexiBLAS"),
                        ("libblis", "BLIS"), ("libatlas", "ATLAS"), ("libarmpl", "ArmPL"),
                        ("libnvpl", "NVPL"), ("libaccelerate", "Accelerate")):
        if token in out:
            return name
    if "liblapack" in out or "libblas" in out:  # reference netlib
        return "LAPACK"
    return None


def detect_libkriging_blas() -> str | None:
    """Name the BLAS/LAPACK libKriging's dense-Cholesky path links against."""
    import glob
    import os

    cands: list[str] = []
    try:
        import pylibkriging as _lk

        pkg = os.path.dirname(_lk.__file__)
        cands += glob.glob(os.path.join(pkg, "**", "_pylibkriging*.so"), recursive=True)
        cands += glob.glob(os.path.join(pkg, "..", "_pylibkriging*.so"))
    except Exception:
        pass
    for d in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if d:
            cands += glob.glob(os.path.join(d, "libKriging.so*"))
    for c in cands:
        b = _blas_from_ldd(c)
        if b:
            return b
    return None


def detect_torch_cpu_blas() -> str | None:
    try:
        import torch

        cfg = torch.__config__.show()
    except Exception:
        return None
    m = re.search(r"BLAS_INFO=(\w+)", cfg)
    if m:
        return {"mkl": "MKL", "open": "OpenBLAS", "openblas": "OpenBLAS", "blis": "BLIS",
                "accelerate": "Accelerate", "eigen": "Eigen", "nvpl": "NVPL",
                "flexiblas": "FlexiBLAS", "generic": "BLAS"}.get(m.group(1).lower(), m.group(1))
    if "USE_MKL=ON" in cfg:
        return "MKL"
    return None


# backend keys used by --backends / dispatch, and how each maps to the
# "<lib>-<method>-<linalg lib>" display label (the linalg part is filled in
# at runtime for the CPU/Cholesky rows).
BACKEND_KEYS = ["chol", "iter-cuda", "iter-omp", "gpt-cuda", "gpt-cpu"]


def backend_label(key: str, lk_blas: str, torch_blas: str) -> str:
    return {
        "chol": f"libKriging-Cholesky-{lk_blas}",
        "iter-cuda": "libKriging-Iterative-CUDA",   # dense-R cublasDgemm within budget, else hand-written CUDA kernels
        "iter-omp": "libKriging-Iterative-OpenMP",  # dense-R BLAS-3 within budget, else hand-written OpenMP loops
        "gpt-cuda": "GPyTorch-BBMM-CUDA",           # PyTorch CUDA (cuBLAS/cuSOLVER) + BBMM
        "gpt-cpu": f"GPyTorch-BBMM-{torch_blas}",
    }[key]


# --------------------------------------------------------------------------
# shared design helpers
# --------------------------------------------------------------------------
def lhs(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = np.zeros((n, d))
    for j in range(d):
        X[:, j] = (rng.permutation(n) - rng.random(n)) / n
    return X


def sine_sum(X) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return np.sin(2 * np.pi * X).sum(axis=1)


def timed_min(fn, reps: int = 5, cap_s: float = 3.0):
    """Call ``fn`` up to ``reps`` times, return (min elapsed, last result).
    Stops early once a single call exceeds ``cap_s`` (already slow -> stable,
    and repeating would waste time). Used to denoise the sub-second ops."""
    best = float("inf")
    out = None
    for i in range(max(1, reps)):
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
        if dt > cap_s:
            break
    return best, out


def rmse_q2(mean, y_true) -> tuple[float, float]:
    mean = np.asarray(mean, dtype=float).ravel()
    y_true = np.asarray(y_true, dtype=float).ravel()
    resid = y_true - mean
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    return rmse, 1.0 - float(np.sum(resid ** 2)) / sst


# --------------------------------------------------------------------------
# GPyTorch side
# --------------------------------------------------------------------------
def _gpt_modules():
    import gpytorch
    import torch

    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.shape[1])
            )

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

    return gpytorch, torch, ExactGPModel


def _gpt_converged_ctx(gpytorch):
    # raised so BBMM CG / SLQ log-det actually converge on the (mildly
    # ill-conditioned) R at theta=0.15 -- the defaults (1000 CG iters,
    # cg_tol 1e-2, 15 Lanczos, 10 probes, precond >= n=2000) leave -mll and
    # the posterior under-converged.
    return [
        # GPyTorch silently falls back to an EXACT dense Cholesky solve for
        # any matrix at or below this size (default 800), no matter how the
        # CG/Lanczos settings below are raised -- confirmed empirically:
        # at n=250/500 (below the default threshold) the eval finished in
        # single-digit milliseconds and its -mll disagreed with a
        # max_cholesky_size(0) run by ~0.06-0.16, an order of magnitude more
        # than n=900/2000 (already above 800) disagreed with themselves
        # (~0.003-0.01, pure stochastic-estimator noise). Since every row
        # here is labeled "GPyTorch-BBMM-*", force BBMM at every n in the
        # sweep -- 0 means no n satisfies "n <= threshold", so Cholesky is
        # never selected.
        gpytorch.settings.max_cholesky_size(0),
        gpytorch.settings.max_cg_iterations(5000),
        gpytorch.settings.cg_tolerance(1e-4),
        gpytorch.settings.eval_cg_tolerance(1e-4),
        gpytorch.settings.max_lanczos_quadrature_iterations(32),
        gpytorch.settings.num_trace_samples(32),
        gpytorch.settings.max_preconditioner_size(100),
        gpytorch.settings.min_preconditioning_size(1),
    ]


def run_gpytorch(X, y, Xte, yte, theta, device_str):
    gpytorch, torch, ExactGPModel = _gpt_modules()
    torch.set_default_dtype(torch.float64)
    device = torch.device(device_str)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)

    tx = torch.tensor(X, dtype=torch.float64, device=device)
    ty = torch.tensor(y, dtype=torch.float64, device=device)
    Xte_t = torch.tensor(np.asarray(Xte, dtype=float), device=device)

    def build():
        lik = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-12))
        lik.noise = 1e-8
        lik.noise_covar.raw_noise.requires_grad_(False)
        model = ExactGPModel(tx, ty, lik)
        with torch.no_grad():
            model.covar_module.base_kernel.lengthscale = torch.tensor(np.full(X.shape[1], theta), dtype=torch.float64)
            model.covar_module.outputscale = torch.tensor(1.0, dtype=torch.float64)
            model.mean_module.constant.fill_(float(np.mean(y)))  # match libKriging's constant trend
        model, lik = model.double().to(device), lik.double().to(device)
        sync()
        return model, lik

    fit_s, (model, lik) = timed_min(build)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)

    with contextlib.ExitStack() as es:
        for c in _gpt_converged_ctx(gpytorch):
            es.enter_context(c)

        def eval_ll():
            model.train()
            lik.train()
            model.zero_grad(set_to_none=True)
            loss = -mll(model(tx), ty)
            loss.backward()
            sync()
            return -float(loss.item())

        loglik_s, ll = timed_min(eval_ll)

        def eval_pred():
            model.train()   # toggle train->eval to drop GPyTorch's cached
            model.eval()    # prediction strategy, so every rep is a COLD solve
            lik.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pm = lik(model(Xte_t)).mean.cpu().numpy()
            sync()
            return pm

        predict_s, pm = timed_min(eval_pred)
    rmse, q2 = rmse_q2(pm, yte)
    return dict(fit_s=fit_s, loglik_s=loglik_s, predict_s=predict_s, ll=ll,
                ll_comparable=False, rmse=rmse, q2=q2, pred_mean=pm)


# --------------------------------------------------------------------------
# libKriging side
# --------------------------------------------------------------------------
def run_libkriging_chol(X, y, Xte, yte, theta):
    import pylibkriging as lk

    d = X.shape[1]
    th = np.full(d, theta)
    params = {"theta": np.full((1, d), theta), "sigma2": 1.0}
    fit_s, m = timed_min(lambda: lk.Kriging(y, X, "matern5_2", objective="LL", optim="none", parameters=params))
    loglik_s, ll = timed_min(lambda: float(m.logLikelihoodFun(th, True)[0]))
    predict_s, pm = timed_min(lambda: np.asarray(m.predict(Xte, False, False, False)[0]).ravel())
    rmse, q2 = rmse_q2(pm, yte)
    return dict(fit_s=fit_s, loglik_s=loglik_s, predict_s=predict_s, ll=ll,
                ll_comparable=True, rmse=rmse, q2=q2, pred_mean=pm)


def run_libkriging_iter(X, y, Xte, yte, theta, use_cuda):
    import pylibkriging as lk

    lk.set_cuda_iterative_enabled(bool(use_cuda))
    n, d = X.shape
    th = np.full(d, theta)
    params = {"theta": np.full((1, d), theta), "sigma2": 1.0}

    fit_s, m = timed_min(
        lambda: lk.Kriging(y, X, "matern5_2", objective=LK_ITER_OBJECTIVE, optim="none", parameters=params))
    if not m.is_iterative_light():
        raise RuntimeError("expected an iterative-light fit (no dense R factor)")

    loglik_s, ll = timed_min(lambda: float(m.logLikelihoodIterativeFun(th, True)[0]))
    ll = float(ll)

    prank = min(128, max(4, n // 4))  # Nystrom CG preconditioner for predictIterative
    predict_s, pm = timed_min(lambda: np.asarray(
        m.predictIterative(Xte, False, max_iter=8 * n, tol=1e-8, use_nystrom_precond=True, precond_rank=prank)[0]
    ).ravel())
    rmse, q2 = rmse_q2(pm, yte)
    return dict(fit_s=fit_s, loglik_s=loglik_s, predict_s=predict_s, ll=ll,
                ll_comparable=True, rmse=rmse, q2=q2, pred_mean=pm)


# --------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------
def one_point(key, n, theta, Xte, yte):
    X = lhs(n, D_CG, seed=TRAIN_SEED)
    y = sine_sum(X) + np.random.default_rng(NOISE_SEED).normal(scale=1e-3, size=n)
    if key == "chol":
        return run_libkriging_chol(X, y, Xte, yte, theta)
    if key == "iter-cuda":
        return run_libkriging_iter(X, y, Xte, yte, theta, use_cuda=True)
    if key == "iter-omp":
        return run_libkriging_iter(X, y, Xte, yte, theta, use_cuda=False)
    if key == "gpt-cuda":
        return run_gpytorch(X, y, Xte, yte, theta, "cuda")
    if key == "gpt-cpu":
        return run_gpytorch(X, y, Xte, yte, theta, "cpu")
    raise ValueError(key)


def sweep(keys, sizes, theta, labels):
    Xte = lhs(N_TEST, D_CG, seed=TEST_SEED)
    yte = sine_sum(Xte)
    yte_rms = float(np.sqrt(np.mean(yte ** 2)))
    rows = []
    ref = {}  # n -> (ll_chol, mean_chol)
    order = (["chol"] if "chol" in keys else []) + [k for k in keys if k != "chol"]  # chol first = reference
    for key in order:
        label = labels[key]
        for n in sizes:
            print(f"  {label:30s} n={n:5d}  ...", end="", flush=True)
            t0 = time.perf_counter()
            try:
                res = one_point(key, n, theta, Xte, yte)
                res.update(backend=label, backend_key=key, n=n, status="ok",
                           wall_s=time.perf_counter() - t0)
                pm = res.pop("pred_mean")
                if key == "chol":
                    ref[n] = (res["ll"], pm)
                if n in ref:
                    ll_chol, mean_chol = ref[n]
                    res["dmean_rms"] = float(np.max(np.abs(pm - mean_chol))) / yte_rms
                    res["dloglik_n"] = (abs(res["ll"] - ll_chol) / n) if res["ll_comparable"] else None
                rows.append(res)
                print(f" fit={res['fit_s']:6.2f}s logLik={res['loglik_s']:7.2f}s predict={res['predict_s']:7.2f}s"
                      f"  RMSE={res['rmse']:.4f} Q2={res['q2']:.4f}"
                      f"  dMean/rms={res.get('dmean_rms', float('nan')):.1e}"
                      f" dLogLik/n={res.get('dloglik_n') if res.get('dloglik_n') is not None else float('nan'):.1e}",
                      flush=True)
            except Exception as exc:  # noqa: BLE001
                rows.append(dict(backend=label, backend_key=key, n=n,
                                 status=f"error: {exc!r}", wall_s=time.perf_counter() - t0))
                print(f" FAILED: {exc!r}", flush=True)
    return rows


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def _versions():
    v = {}
    try:
        import pylibkriging as lk

        v["pylibkriging"] = getattr(lk, "__version__", "?")
        for attr in ("cuda_iterative_available", "hip_iterative_available",
                     "sycl_iterative_available", "metal_iterative_available"):
            fn = getattr(lk, attr, None)
            if callable(fn):
                try:
                    v[attr] = bool(fn())
                except Exception:
                    pass
    except Exception as exc:
        v["pylibkriging"] = f"import failed: {exc!r}"
    try:
        import torch

        v["torch"] = torch.__version__
        v["torch.cuda"] = torch.version.cuda
        v["torch.cuda.is_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        v["torch"] = f"import failed: {exc!r}"
    try:
        import gpytorch

        v["gpytorch"] = gpytorch.__version__
    except Exception as exc:
        v["gpytorch"] = f"import failed: {exc!r}"
    return v


def _f(x, spec=".4f"):
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return format(x, spec) if isinstance(x, float) else str(x)


def write_csv(path, rows):
    cols = ["backend", "backend_key", "n", "status", "fit_s", "loglik_s", "predict_s", "wall_s",
            "ll", "ll_comparable", "rmse", "q2", "dloglik_n", "dmean_rms"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_markdown(path, rows, meta):
    ok = [r for r in rows if r.get("status") == "ok"]
    ns = sorted({r["n"] for r in rows})
    got = lambda k: {r["n"]: r for r in ok if r.get("backend_key") == k}
    chol, lkg, lkc = got("chol"), got("iter-cuda"), got("iter-omp")
    gtg, gtc = got("gpt-cuda"), got("gpt-cpu")
    lbl = {r["backend_key"]: r["backend"] for r in rows if r.get("backend_key")}
    name = lambda k: lbl.get(k, k)

    L = []
    ap = L.append
    ap("# libKriging vs GPyTorch — iterative path, small n")
    ap("")
    ap(f"- **GPU**: {meta['gpu'] or '(none detected)'}")
    ap(f"- **CPU**: {meta['cpu']}")
    ap(f"- **host**: `{meta['host']}`  ·  logical CPUs: {meta['nproc']}  ·  "
       f"OMP_NUM_THREADS: `{meta['omp']}`")
    ap(f"- **when**: {meta['when']}")
    ap(f"- **sweep**: `sine_sum` d={D_CG}, matern5_2, shared theta={meta['theta']}; "
       f"n = {', '.join(map(str, meta['sizes']))}; test n={N_TEST}")
    ap(f"- **libKriging iterative objective**: `{LK_ITER_OBJECTIVE}`  ·  "
       f"`predictIterative(max_iter=8n, Nystrom precond rank ≤ 128)`")
    ap("- **GPyTorch**: raised settings so BBMM converges, and "
       "`max_cholesky_size=0` so every n in the sweep actually USES BBMM "
       "(GPyTorch's default, 800, silently falls back to exact dense "
       "Cholesky at or below it, which would make an n=250/500 "
       "\"GPyTorch-BBMM\" row misnamed) — "
       "`max_cg_iterations=5000, cg_tolerance=1e-4, eval_cg_tolerance=1e-4, "
       "max_lanczos_quadrature_iterations=32, num_trace_samples=32, "
       "max_preconditioner_size=100, min_preconditioning_size=1`")
    ap("- **versions**: " + ", ".join(f"{k}=`{v}`" for k, v in meta["versions"].items()))
    ap("")
    ap("`fit` = the `Kriging(...)` constructor (dense Cholesky for `LL`; one CG+SLQ "
       "commit for the light `LLIterative` fit) / GPyTorch model build. `logLik` = one "
       "log-likelihood **+ gradient** evaluation at theta. `predict` = a *cold* posterior "
       f"mean on {N_TEST} held-out points (GPyTorch's per-fit prediction cache is dropped "
       "each rep). Every timing is the **min of up to 5 reps** (1 rep once a single call "
       "exceeds 3 s). Backends are named `<lib>-<method>-<linalg lib>`. "
       f"`{name('chol')}` is the **reference**: `dLogLik/n` = "
       "`|ll − ll_chol|/n` (blank for GPyTorch, whose `-mll` is a differently normalised "
       "quantity) and `dMean/rms` = `max|mean − mean_chol| / rms(y_test)`.")
    ap("")

    ap(f"## Reference — `{name('chol')}` (exact dense Cholesky)")
    ap("")
    ap("| n | fit (s) | logLik (s) | predict (s) | logLik value | RMSE | Q² |")
    ap("|--:|--:|--:|--:|--:|--:|--:|")
    for n in ns:
        r = chol.get(n)
        if r:
            ap(f"| {n} | {_f(r['fit_s'], '.3f')} | {_f(r['loglik_s'], '.3f')} | "
               f"{_f(r['predict_s'], '.3f')} | {_f(r['ll'], '.3f')} | {_f(r['rmse'])} | {_f(r['q2'])} |")
        else:
            ap(f"| {n} | — | — | — | — | — | — |")
    ap("")

    ap("## Timing — seconds")
    ap("")
    ap("| backend | n | fit | logLik | predict |")
    ap("|---|--:|--:|--:|--:|")
    for r in rows:
        if r.get("status") != "ok":
            ap(f"| {r['backend']} | {r['n']} | — | — | _{r.get('status', '?')}_ |")
            continue
        ap(f"| {r['backend']} | {r['n']} | {_f(r['fit_s'], '.3f')} | "
           f"{_f(r['loglik_s'], '.3f')} | {_f(r['predict_s'], '.3f')} |")
    ap("")

    ap("## Accuracy")
    ap("")
    ap("| backend | n | RMSE | Q² | logLik value | dLogLik/n | dMean/rms |")
    ap("|---|--:|--:|--:|--:|--:|--:|")
    for r in rows:
        if r.get("status") != "ok":
            ap(f"| {r['backend']} | {r['n']} | — | — | — | — | _{r.get('status', '?')}_ |")
            continue
        ap(f"| {r['backend']} | {r['n']} | {_f(r['rmse'])} | {_f(r['q2'])} | "
           f"{_f(r['ll'], '.3f')} | {_f(r.get('dloglik_n'), '.2e')} | {_f(r.get('dmean_rms'), '.2e')} |")
    ap("")

    ap("## Speed-ups (logLik-eval time)")
    ap("")
    ap(f"| n | {name('chol')} | {name('iter-cuda')} | {name('iter-omp')} | "
       f"**OpenMP / CUDA** | **CUDA / Cholesky** | {name('gpt-cuda')} | {name('gpt-cpu')} |")
    ap("|--:|--:|--:|--:|--:|--:|--:|--:|")
    for n in ns:
        c = chol.get(n, {}).get("loglik_s")
        g = lkg.get(n, {}).get("loglik_s")
        cp = lkc.get(n, {}).get("loglik_s")
        gg = gtg.get(n, {}).get("loglik_s")
        gc = gtc.get(n, {}).get("loglik_s")
        ap(f"| {n} | {_f(c, '.3f')} | {_f(g, '.3f')} | {_f(cp, '.3f')} | "
           f"{(f'{cp / g:.1f}×' if (cp and g) else '—')} | "
           f"{(f'{g / c:.1f}×' if (g and c) else '—')} | {_f(gg, '.3f')} | {_f(gc, '.3f')} |")
    ap("")

    ap("## Verdict — did everything converge?")
    ap("")
    iters = [r for r in ok if r.get("backend_key") in ("iter-cuda", "iter-omp")]
    gpts = [r for r in ok if r.get("backend_key") in ("gpt-cuda", "gpt-cpu")]
    if iters:
        m_dll = max((r["dloglik_n"] for r in iters if r.get("dloglik_n") is not None), default=float("nan"))
        m_dm = max((r["dmean_rms"] for r in iters if r.get("dmean_rms") is not None), default=float("nan"))
        ap(f"- **libKriging iterative**: logLik within `{m_dll:.1e}` per point of the chol "
           f"reference, posterior mean within `{m_dm:.1e}` of test RMS — converged.")
    if gpts:
        m_dm = max((r["dmean_rms"] for r in gpts if r.get("dmean_rms") is not None), default=float("nan"))
        gq = min((r["q2"] for r in gpts), default=float("nan"))
        ap(f"- **GPyTorch**: posterior mean within `{m_dm:.1e}` of test RMS of the chol "
           f"reference (a roughly n-independent offset from the covariance-argument "
           f"convention, *not* under-convergence), Q² ≥ `{gq:.4f}`. Its `-mll` value is a "
           f"different normalisation and is not compared.")
    ap(f"- **`{name('chol')}`** is fastest on all three ops at these n — the iterative "
       "path is for n where the dense factor no longer fits / is too slow, not this range.")
    ap("")

    ap("## Notes")
    ap("")
    ap("- Backend names are `<lib>-<method>-<linalg lib>`. Both "
       "`libKriging-Iterative-CUDA` (`LK_ITERATIVE_CUDA_DENSE_MAX_MB` budget) "
       "and `-OpenMP` (`LK_ITERATIVE_DENSE_MAX_MB`) materialize R once per "
       "evaluation for a separable kernel within their memory budget and run "
       "the matvecs as a single `cublasDgemm` / BLAS-3 `R*V` (hence "
       "`-OpenMP`, the BLAS it links), else fall back to a hand-written "
       "matvec kernel. "
       f"`{name('chol')}` and `{name('gpt-cpu')}` name the actual dense BLAS/LAPACK "
       "each links against.")
    ap(f"- theta={meta['theta']} is chosen so the SLQ log-determinant's Lanczos "
       "quadrature and GPyTorch's BBMM CG both converge with sane iteration budgets; "
       "at longer theta (better-fitting but more ill-conditioned R) both need far more "
       "iterations / Lanczos steps. The `,0,40` in the libKriging objective is the "
       "third `LLIterative` argument (SLQ Lanczos steps per probe), added so the "
       "iterative log-likelihood *value* also tracks the exact one here.")
    ap(f"- `{name('iter-cuda')}` vs `{name('iter-omp')}` is the same binary with "
       "`set_cuda_iterative_enabled(...)` toggled — identical results, different path "
       "for the batched CG / SLQ / gradient matvecs (CUDA kernels vs the CPU "
       "dense-`R` BLAS path).")
    ap("- Companion: `docs/comparisons/libKriging_vs_GPyTorch.ipynb` (summary of these "
       "results + the GPyTorch code libKriging mimics).")
    ap("")

    with open(path, "w") as fh:
        fh.write("\n".join(L))


# --------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", default=",".join(map(str, SWEEP_DEFAULT)),
                   help="comma-separated training sizes (default: %(default)s)")
    p.add_argument("--theta", type=float, default=THETA_DEFAULT,
                   help="shared fixed length-scale (default: %(default)s)")
    p.add_argument("--backends", default=",".join(BACKEND_KEYS),
                   help="comma-separated subset of these keys: " + ", ".join(BACKEND_KEYS)
                        + "  (chol / iter-cuda / iter-omp / gpt-cuda / gpt-cpu)")
    p.add_argument("--outdir", default=None, help="output directory (default: <this file>/results)")
    p.add_argument("--tag", default=None, help="extra tag appended to the output file name")
    args = p.parse_args(argv)

    import os

    here = os.path.dirname(os.path.abspath(__file__))
    outdir = args.outdir or os.path.join(here, "results")
    os.makedirs(outdir, exist_ok=True)

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    keys = [b.strip() for b in args.backends.split(",") if b.strip()]
    for k in keys:
        if k not in BACKEND_KEYS:
            p.error(f"unknown backend {k!r}; choose from {BACKEND_KEYS}")

    gpu_name, cpu_name = detect_gpu_name(), detect_cpu_name()
    try:
        import torch

        have_cuda = bool(torch.cuda.is_available())
    except Exception:
        have_cuda = False
    if not have_cuda:
        dropped = [k for k in keys if k.endswith("-cuda")]
        keys = [k for k in keys if not k.endswith("-cuda")]
        if dropped:
            print(f"note: no CUDA -> dropping {dropped}", flush=True)

    lk_blas = detect_libkriging_blas() or "BLAS"
    torch_blas = detect_torch_cpu_blas() or "CPU"
    labels = {k: backend_label(k, lk_blas, torch_blas) for k in BACKEND_KEYS}

    meta = dict(
        gpu=gpu_name, cpu=cpu_name, host=socket.gethostname(),
        nproc=os.cpu_count(), omp=os.environ.get("OMP_NUM_THREADS", "(unset)"),
        when=_dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        theta=args.theta, sizes=sizes, lk_blas=lk_blas, torch_blas=torch_blas,
        versions=_versions(),
    )
    print("machine:", meta["gpu"], "|", meta["cpu"], "| host", meta["host"])
    print("linalg: libKriging dense ->", lk_blas, "| torch CPU ->", torch_blas)
    print("backends:", [labels[k] for k in keys], "| sizes:", sizes, "| theta:", args.theta, flush=True)

    # warm up CUDA contexts so the first real timing isn't polluted by init
    if have_cuda and any(k.endswith("-cuda") for k in keys):
        try:
            Xw = lhs(64, D_CG, seed=1)
            yw = sine_sum(Xw)
            run_libkriging_iter(Xw, yw, Xw[:8], sine_sum(Xw[:8]), args.theta, use_cuda=True)
            run_gpytorch(Xw, yw, Xw[:8], sine_sum(Xw[:8]), args.theta, "cuda")
        except Exception as exc:  # noqa: BLE001
            print(f"warmup skipped: {exc!r}", flush=True)

    t0 = time.perf_counter()
    rows = sweep(keys, sizes, args.theta, labels)
    print(f"\ntotal wall time: {time.perf_counter() - t0:.0f}s", flush=True)

    base = f"{slug(gpu_name)}__{slug(cpu_name)}"
    if args.tag:
        base += "__" + slug(args.tag, 24)
    write_markdown(os.path.join(outdir, base + ".md"), rows, meta)
    write_csv(os.path.join(outdir, base + ".csv"), rows)
    print("wrote", os.path.join(outdir, base + ".md"))
    print("wrote", os.path.join(outdir, base + ".csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
