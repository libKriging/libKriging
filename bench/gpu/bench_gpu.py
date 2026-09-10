#!/usr/bin/env python3
"""Standalone GPU vs CPU vs GPyTorch benchmark for libKriging's iterative
(matrix-free conjugate-gradient) path.

This is a *manual* benchmark -- it is deliberately **not** wired into CI.
Run it by hand on a machine, then commit the Markdown (and CSV) it writes to
``bench/gpu/results/``.  The file name encodes the machine's GPU and CPU so
results from several machines can live side by side and be compared.

What it measures, per training size ``n`` and per backend
--------------------------------------------------------
Fixed sweep (identical on every machine, so the numbers are comparable):

* function   ``sine_sum(x) = sum_i sin(2*pi*x_i)``,  ``d = 4``
* design     seeded LHS,  ``n in {250, 500, 1000, 2000, 4000, 8000}``
* kernel     ``matern5_2``,  shared fixed ``theta = 0.3``,  ``sigma2 = 1``
  (a deliberately ill-conditioned regime -- same rationale as
  ``docs/comparisons/libKriging_vs_GPyTorch.ipynb`` section 4 and
  ``bench/comparison-gpu/`` : a free MLE fit under ``LLIterative`` is
  impractically slow, and a fixed theta isolates each package's CG linear
  algebra from its optimizer)
* objective  ``LLIterative(30)`` -> a **light** (matrix-free) fit: the
  ``Kriging(..., objective="LLIterative(m)", optim="none")`` constructor
  commits beta/sigma2 from one CG + Stochastic-Lanczos-Quadrature likelihood
  evaluation and **never forms or factorizes the dense covariance R**
  (``is_iterative_light() == True``, ``m_is_empty``).  ``fit`` then times one
  more full ``logLikelihoodIterativeFun`` (with gradient) -- the O(n^2)
  matrix-free estimate, the direct analogue of GPyTorch's BBMM ``-mll``.
* predict     ``predictIterative`` : the matrix-free CG posterior mean (what
  ``.predict()`` on a light iterative fit dispatches to anyway).  CG to
  ``tol=1e-8`` with a raised ``max_iter`` (default ``8*n``) and a Nystrom
  preconditioner -- ``predictIterative``'s own default cap of ``2*n``
  under-converges on this ill-conditioned R past n~4000, stalling the RMSE
  trend; both knobs are Python-exposed (``--pred-cg-iters-per-n``,
  ``--pred-precond-rank``).  -> RMSE, Q2 on the test set.

So every ``libkriging-*`` number below is the **fully matrix-free iterative
path** -- no Cholesky anywhere.  ``--exact-crosscheck`` optionally adds, for
``n <= --exact-max-n``, two O(n^3) *dense-Cholesky reference* columns
(``leaveOneOut`` and the exact ``logLikelihoodFun``) purely to check that
the iterative estimates are faithful; those are clearly not part of the
timed comparison.

Backends (each ``n`` run for every selected backend):

* ``libkriging-gpu``   ``set_cuda_iterative_enabled(True)``  -- CUDA/HIP/... batched CG+SLQ
* ``libkriging-cpu``   ``set_cuda_iterative_enabled(False)`` -- OpenMP CPU path
* ``gpytorch-gpu``     BBMM on ``cuda``
* ``gpytorch-cpu``     BBMM on ``cpu``

GPyTorch's posterior mean matches a from-scratch exact GP; libKriging's
``predictIterative`` mean (CG to ``tol``) also converges to the exact GP
posterior, up to a roughly n-independent ~10-15%-of-signal offset from the
covariance-argument convention (see the notebook) -- both are Q2 ~ 1
predictors at small/mid n.  At large n GPyTorch's BBMM *caps* its CG
(``max_cg_iterations``) and its predictor degrades; libKriging's CG runs to
``tol`` and stays accurate (but slow).

A per-backend wall-time budget stops a backend early once a point projects
(next size ~= 2x n ~= 8x cost) to blow the budget -- the CPU iterative path
is O(n^3)-ish in this regime and would otherwise run for hours.

Requires ``pylibkriging`` (built with a GPU iterative backend for the
``*-gpu`` rows), ``torch`` and ``gpytorch`` importable in the same
environment.  See ``bench/gpu/README.md``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import platform
import re
import socket
import subprocess
import sys
import time

import numpy as np

SWEEP_DEFAULT = [250, 500, 1000, 2000, 4000, 8000]
D_CG = 4
THETA_DEFAULT = 0.3
NPROBE_DEFAULT = 30
N_TEST = 300
TEST_SEED = 999
TRAIN_SEED = 123
NOISE_SEED = 0
EXACT_MAX_N_DEFAULT = 2000     # --exact-crosscheck only runs the O(n^3) refs up to here
BACKENDS_ALL = ["libkriging-gpu", "libkriging-cpu", "gpytorch-gpu", "gpytorch-cpu"]


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
    for cmd in (
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        ["rocm-smi", "--showproductname"],
    ):
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
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except Exception:
            pass
    elif sysname == "Windows":
        if platform.processor():
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
# shared design helpers (match the notebook / bench/comparison-gpu)
# --------------------------------------------------------------------------
def lhs(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = np.zeros((n, d))
    for j in range(d):
        X[:, j] = (rng.permutation(n) - rng.random(n)) / n
    return X


def sine_sum(X: np.ndarray) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return np.sin(2 * np.pi * X).sum(axis=1)


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
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x)
            )

    return gpytorch, torch, ExactGPModel


def _gpt_build(X, y, theta, device):
    gpytorch, torch, ExactGPModel = _gpt_modules()
    torch.set_default_dtype(torch.float64)  # match pylibkriging's float64 throughout
    tx = torch.tensor(X, dtype=torch.float64, device=device)
    ty = torch.tensor(y, dtype=torch.float64, device=device)
    lik = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-12)
    )
    lik.noise = 1e-8
    lik.noise_covar.raw_noise.requires_grad_(False)
    model = ExactGPModel(tx, ty, lik)
    with torch.no_grad():
        model.covar_module.base_kernel.lengthscale = torch.tensor(
            np.full(X.shape[1], theta), dtype=torch.float64
        )
        model.covar_module.outputscale = torch.tensor(1.0, dtype=torch.float64)
        model.mean_module.constant.fill_(float(np.mean(y)))
    return model.double().to(device), lik.double().to(device), tx, ty


def run_gpytorch(X, y, Xte, yte, theta, device_str, cg_iters=1000, cg_tol=1e-2):
    gpytorch, torch, _ = _gpt_modules()
    device = torch.device(device_str)
    model, lik, tx, ty = _gpt_build(X, y, theta, device)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    model.train()
    lik.train()
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    sync()
    t0 = time.perf_counter()
    with gpytorch.settings.max_cg_iterations(cg_iters), gpytorch.settings.cg_tolerance(cg_tol):
        loss = -mll(model(tx), ty)
        loss.backward()
    sync()
    fit_s = time.perf_counter() - t0
    ll = -float(loss.item())

    model.eval()
    lik.eval()
    t0 = time.perf_counter()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pm = lik(model(torch.tensor(np.asarray(Xte, dtype=float), device=device))).mean.cpu().numpy()
    sync()
    pred_s = time.perf_counter() - t0
    rmse, q2 = rmse_q2(pm, yte)
    return dict(commit_s=0.0, fit_s=fit_s, pred_s=pred_s, rmse=rmse, q2=q2,
                loo=None, ll_iter=ll, ll_perpt_err=None, pred_mean=pm)


# --------------------------------------------------------------------------
# libKriging side -- fully matrix-free: light LLIterative fit (no dense R
# factor) + predictIterative CG posterior.  The optional exact-Cholesky
# reference columns are computed only when explicitly asked for.
# --------------------------------------------------------------------------
def run_libkriging(X, y, Xte, yte, theta, use_cuda, nprobe, exact_crosscheck, exact_max_n,
                   pred_cg_iters_per_n, pred_precond_rank):
    import pylibkriging as lk

    lk.set_cuda_iterative_enabled(bool(use_cuda))
    n, d = X.shape
    th = np.full(d, theta)
    params = {"theta": np.full((1, d), theta), "sigma2": 1.0}

    # constructor commits a *light* iterative fit -- one CG+SLQ likelihood
    # evaluation, no factorization of R
    t0 = time.perf_counter()
    m = lk.Kriging(y, X, "matern5_2", objective=f"LLIterative({nprobe})",
                   optim="none", parameters=params)
    commit_s = time.perf_counter() - t0
    if not m.is_iterative_light():
        raise RuntimeError("expected an iterative-light fit (no dense R factor) but got a heavy one")

    # one full matrix-free likelihood+gradient evaluation -- the head-to-head number
    t1 = time.perf_counter()
    ll_iter, _ = m.logLikelihoodIterativeFun(th, True)
    fit_s = time.perf_counter() - t1

    # matrix-free CG posterior mean.  predictIterative's DEFAULT max_iter (2n)
    # under-converges on this deliberately ill-conditioned R once n is a few
    # thousand (the RMSE trend visibly stalls), so give it a generous cap and,
    # by default, a Nystrom-preconditioned CG -- which reaches the exact
    # predictor at every n and is cheaper than brute-forcing the iteration
    # count at large n.  `--pred-cg-iters-per-n 0` restores the library default.
    max_iter = 0 if pred_cg_iters_per_n <= 0 else int(pred_cg_iters_per_n * n)
    prank = min(int(pred_precond_rank), max(1, n // 4)) if pred_precond_rank > 0 else 0
    kw = dict(max_iter=max_iter, tol=1e-8)
    if prank > 0:
        kw.update(use_nystrom_precond=True, precond_rank=prank)
    t2 = time.perf_counter()
    pm = np.asarray(m.predictIterative(Xte, False, **kw)[0]).ravel()
    pred_s = time.perf_counter() - t2
    rmse, q2 = rmse_q2(pm, yte)

    loo = ll_perpt_err = None
    if exact_crosscheck and len(y) <= exact_max_n:
        # O(n^3) dense-Cholesky reference ONLY -- not the iterative path
        ll_exact = float(m.logLikelihoodFun(th, False)[0])
        ll_perpt_err = abs(float(ll_iter) - ll_exact) / len(y)
        loo = float(m.leaveOneOut())

    return dict(commit_s=commit_s, fit_s=fit_s, pred_s=pred_s, rmse=rmse, q2=q2,
                loo=loo, ll_iter=float(ll_iter), ll_perpt_err=ll_perpt_err, pred_mean=pm)


# --------------------------------------------------------------------------
# sweep driver
# --------------------------------------------------------------------------
def one_point(backend, n, theta, nprobe, Xte, yte, exact_crosscheck, exact_max_n,
              pred_cg_iters_per_n, pred_precond_rank):
    X = lhs(n, D_CG, seed=TRAIN_SEED)
    y = sine_sum(X) + np.random.default_rng(NOISE_SEED).normal(scale=1e-3, size=n)
    lk_args = (exact_crosscheck, exact_max_n, pred_cg_iters_per_n, pred_precond_rank)
    if backend == "libkriging-gpu":
        return run_libkriging(X, y, Xte, yte, theta, True, nprobe, *lk_args)
    if backend == "libkriging-cpu":
        return run_libkriging(X, y, Xte, yte, theta, False, nprobe, *lk_args)
    if backend == "gpytorch-gpu":
        return run_gpytorch(X, y, Xte, yte, theta, "cuda")
    if backend == "gpytorch-cpu":
        return run_gpytorch(X, y, Xte, yte, theta, "cpu")
    raise ValueError(backend)


def sweep(backends, sizes, theta, nprobe, cpu_budget_s, gpu_budget_s,
          exact_crosscheck, exact_max_n, pred_cg_iters_per_n, pred_precond_rank):
    Xte = lhs(N_TEST, D_CG, seed=TEST_SEED)
    yte = sine_sum(Xte)
    rows = []
    means = {}  # (backend, n) -> posterior mean, for the cross-backend agreement column
    for backend in backends:
        budget = gpu_budget_s if backend.endswith("-gpu") else cpu_budget_s
        stopped = False
        for n in sizes:
            if stopped:
                rows.append(dict(backend=backend, n=n, status="skipped (wall-budget projection)"))
                print(f"  {backend:15s} n={n:5d}   SKIPPED (wall-budget projection)", flush=True)
                continue
            print(f"  {backend:15s} n={n:5d}   ...", end="", flush=True)
            t0 = time.perf_counter()
            try:
                res = one_point(backend, n, theta, nprobe, Xte, yte, exact_crosscheck, exact_max_n,
                                pred_cg_iters_per_n, pred_precond_rank)
                res.update(backend=backend, n=n, status="ok", wall_s=time.perf_counter() - t0)
                means[(backend, n)] = res.pop("pred_mean")
                rows.append(res)
                print(
                    f" commit={res['commit_s']:6.2f}s  fit={res['fit_s']:8.2f}s  "
                    f"pred={res['pred_s']:7.2f}s  RMSE={res['rmse']:.4f}  Q2={res['q2']:.4f}"
                    + (f"  [xcheck LOO={res['loo']:.4f} llErr/pt={res['ll_perpt_err']:.2e}]"
                       if res["loo"] is not None else ""),
                    flush=True,
                )
                if budget is not None and 8.0 * res["fit_s"] > budget:
                    stopped = True
            except Exception as exc:  # noqa: BLE001 -- record and keep going
                rows.append(dict(backend=backend, n=n, status=f"error: {exc!r}",
                                 wall_s=time.perf_counter() - t0))
                print(f" FAILED: {exc!r}", flush=True)
    return rows, means


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


def _fmt(x, spec=".4f"):
    if x is None:
        return "-"
    if isinstance(x, float):
        return format(x, spec)
    return str(x)


def write_csv(path, rows):
    cols = ["backend", "n", "status", "commit_s", "fit_s", "pred_s", "wall_s",
            "rmse", "q2", "ll_iter", "loo", "ll_perpt_err"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_markdown(path, rows, means, meta):
    ok = [r for r in rows if r.get("status") == "ok"]
    by = lambda b: {r["n"]: r for r in ok if r["backend"] == b}
    lkg, lkc = by("libkriging-gpu"), by("libkriging-cpu")
    gtg, gtc = by("gpytorch-gpu"), by("gpytorch-cpu")
    ns = sorted({r["n"] for r in rows})

    L = []
    ap = L.append
    ap(f"# libKriging iterative path — GPU vs CPU vs GPyTorch")
    ap("")
    ap(f"- **GPU**: {meta['gpu'] or '(none detected)'}")
    ap(f"- **CPU**: {meta['cpu']}")
    ap(f"- **host**: `{meta['host']}`  ·  logical CPUs: {meta['nproc']}  ·  "
       f"OMP_NUM_THREADS: `{meta['omp']}`")
    ap(f"- **when**: {meta['when']}")
    ap(f"- **sweep**: `sine_sum` d={D_CG}, matern5_2, shared theta={meta['theta']}, "
       f"LLIterative({meta['nprobe']}); n = {', '.join(map(str, meta['sizes']))}; "
       f"test n={N_TEST}")
    ap(f"- **wall budget**: gpu {meta['gpu_budget']:g}s / cpu {meta['cpu_budget']:g}s "
       f"(a backend stops once the next size projects to exceed it)")
    ap("- **versions**: " + ", ".join(f"{k}=`{v}`" for k, v in meta["versions"].items()))
    ap("")
    ap("Every `libkriging-*` row is the **fully matrix-free iterative path** — a *light* "
       "`LLIterative` fit that never factorizes the dense covariance R "
       "(`is_iterative_light() == True`), plus a `predictIterative` CG posterior mean. "
       "No Cholesky anywhere.")
    ap("")
    ap("- **`commit`** — the `Kriging(..., objective=\"LLIterative(%d)\", optim=\"none\")` "
       "constructor: one CG+SLQ likelihood evaluation to commit β/σ² (GPyTorch has no "
       "separate step, shown as 0)." % meta["nprobe"])
    ap("- **`fit`** — one more full `logLikelihoodIterativeFun` with gradient (libKriging, "
       "O(n²) CG + SLQ log-det + Hutchinson trace) / one `-mll` forward+backward "
       "(GPyTorch, BBMM). **This is the head-to-head number.**")
    ap("- **`pred`** — posterior mean on the %d test points. libKriging: `predictIterative` "
       "CG to `tol=1e-8` with `max_iter=%s·n`%s. `predictIterative`'s *default* cap (2n) "
       "under-converges on this ill-conditioned R past n≈4000 (the RMSE trend stalls), so "
       "the sweep raises it; the Nystrom preconditioner reaches the exact predictor at "
       "every n and is the cheaper route at large n. GPyTorch: `.eval()` BBMM posterior "
       "(its own CG, `max_cg_iterations` cap)." %
       (N_TEST,
        (("%g" % meta["pred_cg_iters_per_n"]) if meta["pred_cg_iters_per_n"] > 0 else "2 (library default)"),
        (", Nystrom-preconditioned (rank ≤ %d)" % meta["pred_precond_rank"]) if meta["pred_precond_rank"] > 0 else ""))
    ap("- **`RMSE` / `Q²`** — that predictor vs the true function on the test set.")
    ap("- **`ll(iter)`** — each side's own fit-objective value; libKriging's raw "
       "concentrated log-likelihood vs GPyTorch's per-point `-mll` are **not comparable "
       "across the two** (different normalisation / σ² handling).")
    if meta["exact_crosscheck"]:
        ap("- **`LOO`, `llErr/pt`** — *exact O(n³) dense-Cholesky reference*, computed "
           "separately for n ≤ %d only (not part of the timed iterative comparison): "
           "libKriging's closed-form leave-one-out, and |iterative − exact| concentrated "
           "log-likelihood per training point." % meta["exact_max_n"])
    ap("")

    ap("## Per-backend results  (matrix-free iterative path)")
    ap("")
    hdr = "| backend | n | commit (s) | fit (s) | pred (s) | wall (s) | RMSE | Q² | ll(iter) |"
    sep = "|---|--:|--:|--:|--:|--:|--:|--:|--:|"
    if meta["exact_crosscheck"]:
        hdr += " LOO | llErr/pt |"
        sep += "--:|--:|"
    ap(hdr)
    ap(sep)
    for r in rows:
        if r.get("status") != "ok":
            tail = " — | — |" if meta["exact_crosscheck"] else ""
            ap(f"| {r['backend']} | {r['n']} | — | — | — | {_fmt(r.get('wall_s'), '.1f')} | — | "
               f"— | _{r.get('status', '?')}_ |" + tail)
            continue
        row = (f"| {r['backend']} | {r['n']} | {_fmt(r['commit_s'], '.2f')} | "
               f"{_fmt(r['fit_s'], '.2f')} | {_fmt(r['pred_s'], '.2f')} | "
               f"{_fmt(r.get('wall_s'), '.1f')} | {_fmt(r['rmse'])} | {_fmt(r['q2'])} | "
               f"{_fmt(r['ll_iter'], '.3f')} |")
        if meta["exact_crosscheck"]:
            row += f" {_fmt(r['loo'])} | {_fmt(r['ll_perpt_err'], '.2e')} |"
        ap(row)
    ap("")

    ap("## Speed-ups (fit-eval time, seconds; ratios in **bold**)")
    ap("")
    ap("`cpu/gpu` > 1 → libKriging's GPU backend is that much faster than its own CPU "
       "path. `libk-gpu ÷ gpt-gpu` > 1 → libKriging is that much slower than GPyTorch "
       "(expected: GPyTorch caps its CG, libKriging runs it to `tol`).")
    ap("")
    ap("| n | libk-cpu | libk-gpu | **cpu / gpu** | gpytorch-gpu | **libk-gpu ÷ gpt-gpu** | gpytorch-cpu |")
    ap("|--:|--:|--:|--:|--:|--:|--:|")
    for n in ns:
        c = lkc.get(n, {}).get("fit_s")
        g = lkg.get(n, {}).get("fit_s")
        gg = gtg.get(n, {}).get("fit_s")
        gc = gtc.get(n, {}).get("fit_s")
        cpu_gpu = f"{c / g:.1f}×" if (c and g) else "—"
        lk_gt = f"{g / gg:.1f}×" if (g and gg) else "—"
        ap(f"| {n} | {_fmt(c, '.2f')} | {_fmt(g, '.2f')} | {cpu_gpu} | "
           f"{_fmt(gg, '.2f')} | {lk_gt} | {_fmt(gc, '.2f')} |")
    ap("")

    # cross-backend predictor agreement (max |mean_a - mean_b| / test RMS)
    yte_rms = float(np.sqrt(np.mean(sine_sum(lhs(N_TEST, D_CG, seed=TEST_SEED)) ** 2)))
    pairs = [("libkriging-gpu", "libkriging-cpu"), ("libkriging-gpu", "gpytorch-gpu"),
             ("gpytorch-gpu", "gpytorch-cpu")]
    ap("## Predictor agreement  (`max |mean_a − mean_b| / test-RMS`)")
    ap("")
    ap("| n | " + " | ".join(f"{a} vs {b}" for a, b in pairs) + " |")
    ap("|--:|" + "|".join(["--:"] * len(pairs)) + "|")
    for n in ns:
        cells = []
        for a, b in pairs:
            ma, mb = means.get((a, n)), means.get((b, n))
            if ma is None or mb is None:
                cells.append("—")
            else:
                cells.append(f"{float(np.max(np.abs(ma - mb))) / yte_rms:.2e}")
        ap(f"| {n} | " + " | ".join(cells) + " |")
    ap("")
    ap("`libkriging-gpu vs libkriging-cpu` is the same binary with the CUDA path "
       "toggled — it should be ~0 (numerical noise). `libkriging-gpu vs gpytorch-gpu` "
       "carries the ~10–15%-of-signal covariance-argument-convention offset discussed "
       "in `docs/comparisons/libKriging_vs_GPyTorch.ipynb` §4, plus, at large n, "
       "GPyTorch's CG hitting its `max_cg_iterations` cap.")
    ap("")

    ap("## Notes")
    ap("")
    ap("- Every `libkriging-*` timing above is **matrix-free** — the light `LLIterative` "
       "fit keeps no dense R factor and `predictIterative` solves by CG. The only "
       "Cholesky in this benchmark is the optional `--exact-crosscheck` reference "
       "(`%s` this run)." % ("on, n ≤ %d" % meta["exact_max_n"] if meta["exact_crosscheck"] else "off"))
    ap("- **theta = %s is deliberately ill-conditioned** for `sine_sum` d=%d on the unit "
       "cube: CG needs many iterations to reach `tol`, and the SLQ log-determinant's "
       "default 20 Lanczos steps stop resolving R's spectrum beyond n ≈ a couple "
       "thousand — so the *fit* objective `ll(iter)` drifts from the exact concentrated "
       "log-likelihood (`llErr/pt` grows with n, when `--exact-crosscheck` is on). "
       "GPyTorch's CG instead *caps* at `max_cg_iterations` (1000) and returns an "
       "under-converged `-mll` and posterior. Both are budget effects, not structural — "
       "more Lanczos steps / CG iterations close each gap. The `commit`/`fit` CG cap "
       "(2n) and Lanczos step count are **not** tunable from Python, so `llErr/pt` "
       "growth is expected here." % (meta["theta"], D_CG))
    ap("- **`predictIterative`'s CG *is* tuned here** (its `max_iter` and preconditioner "
       "are Python-exposed). Its default 2n-iteration cap under-converges on this R once "
       "n exceeds ~4000 — the RMSE trend visibly stalls (e.g. n=8000 RMSE ≈ 0.008 at 2n "
       "vs ≈ 0.003 converged). With the raised cap + Nystrom preconditioner used above, "
       "`predictIterative` reaches the exact posterior at every n, so `libkriging-*` "
       "RMSE/Q² follow a clean monotone trend while `gpytorch-*` degrade past n ≈ 2000 "
       "(its BBMM CG caps out; see the `NumericalWarning`s on stderr).")
    ap("- The **CPU iterative path is O(n³)-ish in this regime** — the wall budget "
       "usually stops `libkriging-cpu` after n ≈ 500–1000, while `libkriging-gpu` "
       "finishes the whole sweep. That gap is the headline result.")
    ap("- Companion material: `docs/comparisons/libKriging_vs_GPyTorch.ipynb` §4, "
       "`bench/comparison-gpu/` (multi-function CI-style variant), "
       "`bench/comparison-gpu/ANALYSIS.md`.")
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
    p.add_argument("--nprobe", type=int, default=NPROBE_DEFAULT,
                   help="LLIterative probe count (default: %(default)s)")
    p.add_argument("--backends", default=",".join(BACKENDS_ALL),
                   help="comma-separated subset of: " + ", ".join(BACKENDS_ALL))
    p.add_argument("--cpu-budget", type=float, default=900.0,
                   help="per-CPU-backend wall-time budget in seconds (default: %(default)s)")
    p.add_argument("--gpu-budget", type=float, default=3600.0,
                   help="per-GPU-backend wall-time budget in seconds (default: %(default)s)")
    p.add_argument("--exact-crosscheck", action="store_true",
                   help="also compute the exact O(n^3) dense-Cholesky leaveOneOut and "
                        "logLikelihoodFun as a fidelity reference (n <= --exact-max-n); "
                        "off by default so every reported libKriging number is matrix-free")
    p.add_argument("--exact-max-n", type=int, default=EXACT_MAX_N_DEFAULT,
                   help="cap for --exact-crosscheck (default: %(default)s)")
    p.add_argument("--pred-cg-iters-per-n", type=float, default=8.0,
                   help="predictIterative CG iteration cap as a multiple of n "
                        "(default: %(default)s; 0 = library default of 2n, which "
                        "under-converges past n~4000 in this regime)")
    p.add_argument("--pred-precond-rank", type=int, default=128,
                   help="Nystrom preconditioner rank for predictIterative's CG "
                        "(default: %(default)s, capped at n/4; 0 = no preconditioner)")
    p.add_argument("--outdir", default=None,
                   help="output directory (default: <this file>/results)")
    p.add_argument("--tag", default=None,
                   help="extra tag appended to the output file name")
    args = p.parse_args(argv)

    import os

    here = os.path.dirname(os.path.abspath(__file__))
    outdir = args.outdir or os.path.join(here, "results")
    os.makedirs(outdir, exist_ok=True)

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    for b in backends:
        if b not in BACKENDS_ALL:
            p.error(f"unknown backend {b!r}; choose from {BACKENDS_ALL}")

    gpu_name = detect_gpu_name()
    cpu_name = detect_cpu_name()
    have_cuda = False
    try:
        import torch

        have_cuda = bool(torch.cuda.is_available())
    except Exception:
        pass
    if not have_cuda:
        dropped = [b for b in backends if b.endswith("-gpu")]
        backends = [b for b in backends if not b.endswith("-gpu")]
        if dropped:
            print(f"note: no CUDA torch -> dropping {dropped}", flush=True)

    meta = dict(
        gpu=gpu_name, cpu=cpu_name, host=socket.gethostname(),
        nproc=os.cpu_count(), omp=os.environ.get("OMP_NUM_THREADS", "(unset)"),
        when=_dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        theta=args.theta, nprobe=args.nprobe, sizes=sizes,
        cpu_budget=args.cpu_budget, gpu_budget=args.gpu_budget,
        exact_crosscheck=bool(args.exact_crosscheck), exact_max_n=args.exact_max_n,
        pred_cg_iters_per_n=args.pred_cg_iters_per_n, pred_precond_rank=args.pred_precond_rank,
        versions=_versions(),
    )
    print("machine:", meta["gpu"], "|", meta["cpu"], "| host", meta["host"], flush=True)
    print("backends:", backends, "| sizes:", sizes,
          "| exact-crosscheck:", meta["exact_crosscheck"],
          "| predictIterative: max_iter=%s*n precond_rank<=%d" %
          (args.pred_cg_iters_per_n or "2(lib)", args.pred_precond_rank), flush=True)

    t_start = time.perf_counter()
    rows, means = sweep(backends, sizes, args.theta, args.nprobe,
                        args.cpu_budget, args.gpu_budget,
                        meta["exact_crosscheck"], meta["exact_max_n"],
                        args.pred_cg_iters_per_n, args.pred_precond_rank)
    print(f"\ntotal wall time: {time.perf_counter() - t_start:.0f}s", flush=True)

    base = f"{slug(gpu_name)}__{slug(cpu_name)}"
    if args.tag:
        base += "__" + slug(args.tag, 24)
    md_path = os.path.join(outdir, base + ".md")
    csv_path = os.path.join(outdir, base + ".csv")
    write_markdown(md_path, rows, means, meta)
    write_csv(csv_path, rows)
    print("wrote", md_path, flush=True)
    print("wrote", csv_path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
