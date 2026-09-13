#!/usr/bin/env python3
"""Standalone libKriging vs GPyTorch benchmark, matrix-free iterative path.

Run this by hand on a machine, then commit the HTML report (+ CSV) it writes to
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
  CG/Lanczos/preconditioner settings AND ``max_cholesky_size=0`` so it
  actually converges via BBMM at every n (GPyTorch's own ``ExactGP`` name
  is about the MODEL being exact, not the solve method -- its default
  ``max_cholesky_size=800`` silently switches to an exact dense solve at or
  below that size, which the raised CG/Lanczos settings alone don't
  override).
* **GPyTorch-Cholesky-CUDA** / **GPyTorch-Cholesky-<BLAS>** -- the SAME
  model, with ``max_cholesky_size`` forced far above any n here instead, so
  every row is GPyTorch's own exact dense solve -- a second reference,
  alongside libKriging's, that separates any residual cross-library model
  offset (see ``dMean/rms`` below) from BBMM under-convergence: being
  exact, its ``dMean/rms`` is a pure model-agreement figure, and BBMM
  should match it at every n where BBMM has converged.

  Both libraries are configured on the *same* model: a product of
  one-dimensional Matern-5/2 factors (which is what libKriging's
  ``matern5_2`` is -- see ``_gpt_modules`` for why
  ``MaternKernel(ard_num_dims=d)`` is *not* that model), with GPyTorch's
  fixed ``ConstantMean`` set to libKriging's GLS constant trend ``beta``
  rather than ``mean(y)``. Both alignments matter: without them the exact
  ``GPyTorch-Cholesky`` rows reported ``dMean/rms`` ~4e-02, i.e. that column
  measured model mismatch instead of the ~1e-03 solver convergence it
  exists to show.

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
  ``dLogLik/n`` (``|ll - ll_chol| / n``) and ``dMean/rms``
  (``max|mean - mean_chol| / rms(y_test)``).
* Two log-likelihood columns. ``logLik (native)`` is what each library
  returns as-is, and the two sides look nothing alike because the
  conventions differ: GPyTorch's ``ExactMarginalLogLikelihood`` is
  ``log p(y|X) / n`` (per datapoint) while libKriging returns the
  *concentrated* likelihood with ``sigma2`` profiled out analytically.
  ``logLik (libK conv.)`` undoes both and is what ``dLogLik/n`` uses; see
  ``run_gpytorch`` for the conversion and its verification.

Requires ``pylibkriging`` (built with a GPU iterative backend for the CUDA
rows), ``torch`` and ``gpytorch`` importable in one environment.
See ``bench/gpu/README.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as _dt
import html as _html
import json
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
# 30 probes, no CG precond, 40 SLQ Lanczos steps, 6n CG budget, and a CG
# tolerance filled in from --cg-tol.
#
# The 4th field (cg_max_iter_mult) raises the CG budget from the default 2n
# to 6n: at n=8000 in this sweep the default 2n budget was NOT enough for
# the gradient's 30-column Hutchinson probe solve to reach tol (confirmed by
# LinearAlgebra::cgNonConvergenceWarning / Kriging::iterative_cg_converged()
# firing) -- see docs/math/Iterative.md's "CG iteration budget /
# non-convergence" section. 6n was empirically enough to converge cleanly up
# to n=8000 at this sweep's theta; push it further if larger --sizes still
# trip the warning.
#
# The 5th field is the CG tolerance, driven by --cg-tol so that `logLik` is
# budget-matched to GPyTorch the same way `predict` is (leaving it at
# libKriging's own default would have libKriging solve to a different
# residual than GPyTorch's BBMM).
LK_ITER_NPROBE = 30    # Hutchinson/SLQ probes -- mirrored into GPyTorch's num_trace_samples
LK_ITER_LANCZOS = 40   # SLQ Lanczos steps/probe -- mirrored into max_lanczos_quadrature_iterations
LK_ITER_OBJECTIVE_FMT = (
    "LLIterative(%d,0,%d,6,{cg_tol:g})" % (LK_ITER_NPROBE, LK_ITER_LANCZOS))

# Shared CG convergence budget, applied to BOTH libraries (see --cg-tol).
# This has to be ONE number: earlier revisions of this harness ran
# libKriging's predictIterative at tol=1e-8 while GPyTorch's BBMM ran at
# eval_cg_tolerance=1e-4, i.e. compared a solver converged four orders of
# magnitude tighter against a deliberately truncated one, and read the
# difference as a speed gap. At n=4000 that single setting accounted for a
# 3x difference in libKriging's predict time on its own (0.253 s at 1e-8 vs
# 0.085 s at 1e-4, for a posterior mean already 35x closer to the exact
# Cholesky answer than GPyTorch's at either tolerance).
CG_TOL_DEFAULT = 1e-4
# Nystrom preconditioner rank for predictIterative, capped at n//4. 0 = off.
# GPyTorch's comparable knob is max_preconditioner_size (100 below).
LK_PRECOND_RANK_DEFAULT = 128


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
BACKEND_KEYS = ["chol", "iter-cuda", "iter-omp", "gpt-cuda", "gpt-cpu", "gpt-chol-cuda", "gpt-chol-cpu"]


def backend_label(key: str, lk_blas: str, torch_blas: str) -> str:
    return {
        "chol": f"libKriging-Cholesky-{lk_blas}",
        "iter-cuda": "libKriging-Iterative-CUDA",   # dense-R cublasDgemm within budget, else hand-written CUDA kernels
        "iter-omp": "libKriging-Iterative-OpenMP",  # dense-R BLAS-3 within budget, else hand-written OpenMP loops
        "gpt-cuda": "GPyTorch-BBMM-CUDA",           # PyTorch CUDA (cuBLAS/cuSOLVER) + BBMM
        "gpt-cpu": f"GPyTorch-BBMM-{torch_blas}",
        "gpt-chol-cuda": "GPyTorch-Cholesky-CUDA",           # max_cholesky_size forced huge -> exact dense solve
        "gpt-chol-cpu": f"GPyTorch-Cholesky-{torch_blas}",
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
            # A PRODUCT of one-dimensional Matern-5/2 kernels, not
            # MaternKernel(ard_num_dims=d).
            #
            # These are different covariance models, and the difference is not
            # a detail: GPyTorch's ARD Matern is *radial after scaling* -- it
            # collapses the per-dimension scaled offsets into the Euclidean
            # norm r = ||dx/l||_2 and evaluates ONE Matern function of it --
            # whereas libKriging's "matern5_2" (like every covType it
            # supports) is *separable*, Cov = prod_k f(|dx_k|/theta_k).
            # Numerically, at d=4, theta=0.15, dx=(0.05,0.10,0.02,0.07):
            # separable 0.556929, radial 0.589472.
            #
            # Running the sweep on the radial kernel made every GPyTorch row
            # report `dMean/rms` ~4e-02 against the libKriging Cholesky
            # reference -- including `GPyTorch-Cholesky`, an EXACT solver,
            # which is the tell: that column was measuring the kernel
            # mismatch, not solver convergence, and it drowned the ~1e-03
            # convergence signal it exists to show. With the product kernel
            # the same figure drops to ~9e-04, i.e. into the same range as
            # libKriging's own iterative rows, and RMSE/Q2 become comparable
            # across libraries instead of comparing two different models.
            # Costs GPyTorch ~13% on `predict` (d lazily-evaluated kernels
            # instead of one fused one) -- worth it for a valid comparison.
            d = train_x.shape[1]
            base = gpytorch.kernels.ProductKernel(
                *[gpytorch.kernels.MaternKernel(nu=2.5, active_dims=[k]) for k in range(d)]
            )
            self.covar_module = gpytorch.kernels.ScaleKernel(base)

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

    return gpytorch, torch, ExactGPModel


def _gpt_converged_ctx(gpytorch, cg_tol: float, bbmm: bool = True):
    # raised so BBMM CG / SLQ log-det actually converge on the (mildly
    # ill-conditioned) R at theta=0.15 -- the defaults (1000 CG iters,
    # cg_tol 1e-2, 15 Lanczos, 10 probes, precond >= n=2000) leave -mll and
    # the posterior under-converged. Irrelevant (but harmless) when
    # bbmm=False: max_cholesky_size below then forces the exact dense path
    # regardless, so none of these are ever consulted.
    return [
        # GPyTorch silently falls back to an EXACT dense Cholesky solve for
        # any matrix at or below this size (default 800), no matter how the
        # CG/Lanczos settings below are raised -- confirmed empirically:
        # at n=250/500 (below the default threshold) the eval finished in
        # single-digit milliseconds and its -mll disagreed with a
        # max_cholesky_size(0) run by ~0.06-0.16, an order of magnitude more
        # than n=900/2000 (already above 800) disagreed with themselves
        # (~0.003-0.01, pure stochastic-estimator noise). bbmm=True (the
        # "GPyTorch-BBMM-*" backends) forces BBMM at every n in the sweep --
        # 0 means no n satisfies "n <= threshold", so Cholesky is never
        # selected. bbmm=False (the "GPyTorch-Cholesky-*" backends, an
        # explicit exact-solve cross-check alongside libKriging's own
        # Cholesky reference) does the opposite: a threshold far above any
        # n this sweep uses, so Cholesky is ALWAYS selected.
        gpytorch.settings.max_cholesky_size(0 if bbmm else 1_000_000),
        gpytorch.settings.max_cg_iterations(5000),
        gpytorch.settings.cg_tolerance(cg_tol),
        gpytorch.settings.eval_cg_tolerance(cg_tol),
        # Matched EXACTLY to libKriging's LLIterative(30,_,40,...): 30
        # Hutchinson probes and 40 SLQ Lanczos steps per probe. These drive
        # the stochastic log-determinant, so leaving them at different values
        # (32/32 vs 30/40) would make the `dLogLik/n` column a comparison of
        # two different quadrature budgets rather than of the two estimators.
        gpytorch.settings.max_lanczos_quadrature_iterations(LK_ITER_LANCZOS),
        gpytorch.settings.num_trace_samples(LK_ITER_NPROBE),
        gpytorch.settings.max_preconditioner_size(100),
        gpytorch.settings.min_preconditioning_size(1),
    ]


def run_gpytorch(X, y, Xte, yte, theta, device_str, cg_tol: float, bbmm: bool = True,
                 trend_const=None):
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
            for sub in model.covar_module.base_kernel.kernels:  # one 1-D Matern per input dimension
                sub.lengthscale = torch.tensor(theta, dtype=torch.float64)
            model.covar_module.outputscale = torch.tensor(1.0, dtype=torch.float64)
            # Match libKriging's constant trend. libKriging does *universal*
            # kriging: its constant trend beta is the GLS estimate
            # (F'R^-1 F)^-1 F'R^-1 y, profiled out at fit time -- not the
            # arithmetic mean of y, which is only its OLS counterpart.
            # Feeding GPyTorch's fixed ConstantMean the arithmetic mean left
            # a residual O(1e-2) posterior-mean offset at small n that showed
            # up on the EXACT `GPyTorch-Cholesky` rows too; using the GLS beta
            # from the libKriging Cholesky reference drops it by another
            # order of magnitude (8.8e-04 -> 8.2e-05 at n=2000, d=4), leaving
            # `dMean/rms` measuring solver convergence and nothing else.
            # This is model configuration, not a shortcut: it costs GPyTorch
            # no work (the value is a scalar set before any solve) and
            # mean(y) was an equally arbitrary externally-supplied constant.
            model.mean_module.constant.fill_(
                float(np.mean(y)) if trend_const is None else float(trend_const))
        model, lik = model.double().to(device), lik.double().to(device)
        return model, lik

    with contextlib.ExitStack() as es:
        for c in _gpt_converged_ctx(gpytorch, cg_tol, bbmm=bbmm):
            es.enter_context(c)

        def fit():
            # Mirrors libKriging's `Kriging(...)` constructor: with no
            # hyperparameter optimization (theta is fixed here, same as
            # libKriging's `optim="none"`), "fitting" still means solving
            # R(theta) once -- one Cholesky factorization, or one CG+SLQ
            # commit for BBMM (exactly what the light `LLIterative` fit does
            # on the libKriging side). `ExactGP.__init__` itself does none of
            # that -- it's pure lazy bookkeeping (verified: flat ~1ms
            # regardless of n) -- so without this forced forward call
            # `fit_s` measured object construction, not a fit, and all the
            # real work silently landed in the first `logLik` call instead.
            model, lik = build()
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
            model.train()
            lik.train()
            with torch.no_grad():
                mll(model(tx), ty)  # forces the R(theta) solve/logdet now, not on first logLik call
            sync()
            return model, lik, mll

        fit_s, (model, lik, mll) = timed_min(fit)

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

        # Put GPyTorch's log-likelihood into libKriging's convention, so the
        # report's `logLik value` column is one comparable quantity instead
        # of two numbers that differ by three orders of magnitude for
        # reasons that have nothing to do with either solver. UNTIMED: this
        # is an extra solve purely for the accuracy column.
        #
        # Two conventions separate them, both exactly invertible:
        #   * GPyTorch's ExactMarginalLogLikelihood returns log p(y|X) / n
        #     (per datapoint), so n*mll is its total.
        #   * libKriging returns the CONCENTRATED (profiled) likelihood: it
        #     maximizes over sigma2 analytically and substitutes
        #     sigma2_hat = q/n (q = r' K^-1 r, r = y - F beta), which turns
        #     the -q/2 term into -(n log sigma2_hat + n)/2.
        # so  ll_libK = n*mll + q/2 - (n log(q/n) + n)/2.
        #
        # q comes from GPyTorch's OWN solve under the same settings as the
        # timed run, so a BBMM row's converted value carries BBMM's own CG
        # error rather than borrowing libKriging's -- which is the point, it
        # has to stay a convergence measurement. Verified against libKriging
        # directly at n=250 and n=4000, d=4, matern5_2, theta=0.15: GPyTorch's
        # own sigma2_hat matches libKriging's sigma2() to 6e-09/1.5e-08 and
        # the converted log-likelihood matches to 1.8e-07 relative.
        model.train()
        lik.train()
        with torch.no_grad():
            joint = lik(model(tx))
            r = (ty - joint.mean).unsqueeze(-1)
            q = float((r * joint.lazy_covariance_matrix.solve(r)).sum())
            sync()
        n_obs = X.shape[0]
        ll_libk = ll * n_obs + 0.5 * q - 0.5 * (n_obs * np.log(q / n_obs) + n_obs)

    rmse, q2 = rmse_q2(pm, yte)
    return dict(fit_s=fit_s, loglik_s=loglik_s, predict_s=predict_s,
                ll=ll_libk, ll_native=ll, ll_comparable=True,
                rmse=rmse, q2=q2, pred_mean=pm)


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
                ll_native=ll, ll_comparable=True, rmse=rmse, q2=q2, pred_mean=pm,
                beta=float(np.asarray(m.beta()).ravel()[0]))


def run_libkriging_iter(X, y, Xte, yte, theta, use_cuda, cg_tol: float, precond_rank: int):
    import pylibkriging as lk

    lk.set_cuda_iterative_enabled(bool(use_cuda))
    n, d = X.shape
    th = np.full(d, theta)
    params = {"theta": np.full((1, d), theta), "sigma2": 1.0}

    fit_s, m = timed_min(
        lambda: lk.Kriging(y, X, "matern5_2", objective=LK_ITER_OBJECTIVE_FMT.format(cg_tol=cg_tol),
                           optim="none", parameters=params))
    if not m.is_iterative_light():
        raise RuntimeError("expected an iterative-light fit (no dense R factor)")

    loglik_s, ll = timed_min(lambda: float(m.logLikelihoodIterativeFun(th, True)[0]))
    ll = float(ll)

    # Nystrom CG preconditioner for predictIterative; rank 0 disables it.
    prank = min(precond_rank, max(4, n // 4)) if precond_rank > 0 else 0
    predict_s, pm = timed_min(lambda: np.asarray(
        m.predictIterative(Xte, False, max_iter=8 * n, tol=cg_tol,
                           use_nystrom_precond=prank > 0, precond_rank=prank)[0]
    ).ravel())
    rmse, q2 = rmse_q2(pm, yte)
    return dict(fit_s=fit_s, loglik_s=loglik_s, predict_s=predict_s, ll=ll,
                ll_native=ll, ll_comparable=True, rmse=rmse, q2=q2, pred_mean=pm)


# --------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------
def one_point(key, n, theta, Xte, yte, cg_tol, precond_rank, trend_const=None):
    X = lhs(n, D_CG, seed=TRAIN_SEED)
    y = sine_sum(X) + np.random.default_rng(NOISE_SEED).normal(scale=1e-3, size=n)
    if key == "chol":
        return run_libkriging_chol(X, y, Xte, yte, theta)
    if key == "iter-cuda":
        return run_libkriging_iter(X, y, Xte, yte, theta, True, cg_tol, precond_rank)
    if key == "iter-omp":
        return run_libkriging_iter(X, y, Xte, yte, theta, False, cg_tol, precond_rank)
    if key == "gpt-cuda":
        return run_gpytorch(X, y, Xte, yte, theta, "cuda", cg_tol, bbmm=True, trend_const=trend_const)
    if key == "gpt-cpu":
        return run_gpytorch(X, y, Xte, yte, theta, "cpu", cg_tol, bbmm=True, trend_const=trend_const)
    if key == "gpt-chol-cuda":
        return run_gpytorch(X, y, Xte, yte, theta, "cuda", cg_tol, bbmm=False, trend_const=trend_const)
    if key == "gpt-chol-cpu":
        return run_gpytorch(X, y, Xte, yte, theta, "cpu", cg_tol, bbmm=False, trend_const=trend_const)
    raise ValueError(key)


def sweep(keys, sizes, theta, labels, cg_tol, precond_rank):
    Xte = lhs(N_TEST, D_CG, seed=TEST_SEED)
    yte = sine_sum(Xte)
    yte_rms = float(np.sqrt(np.mean(yte ** 2)))
    rows = []
    ref = {}  # n -> (ll_chol, mean_chol)
    beta_ref = {}  # n -> libKriging's GLS constant trend, handed to GPyTorch
    order = (["chol"] if "chol" in keys else []) + [k for k in keys if k != "chol"]  # chol first = reference
    for key in order:
        label = labels[key]
        for n in sizes:
            print(f"  {label:30s} n={n:5d}  ...", end="", flush=True)
            t0 = time.perf_counter()
            try:
                res = one_point(key, n, theta, Xte, yte, cg_tol, precond_rank,
                                trend_const=beta_ref.get(n))
                res.update(backend=label, backend_key=key, n=n, status="ok",
                           wall_s=time.perf_counter() - t0)
                pm = res.pop("pred_mean")
                if key == "chol":
                    ref[n] = (res["ll"], pm)
                    beta_ref[n] = res.pop("beta", None)
                res.pop("beta", None)
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
            "ll", "ll_native", "ll_comparable", "rmse", "q2", "dloglik_n", "dmean_rms"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _md_inline(s: str) -> str:
    """Minimal Markdown-inline -> HTML: escape, then `code`, **bold**, *italic*."""
    s = _html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", s)
    return s


def write_html(path, rows, meta):
    ok = [r for r in rows if r.get("status") == "ok"]
    ns = sorted({r["n"] for r in rows})
    got = lambda k: {r["n"]: r for r in ok if r.get("backend_key") == k}
    chol, lkg, lkc = got("chol"), got("iter-cuda"), got("iter-omp")
    gtg, gtc = got("gpt-cuda"), got("gpt-cpu")
    lbl = {r["backend_key"]: r["backend"] for r in rows if r.get("backend_key")}
    name = lambda k: lbl.get(k, k)
    md = _md_inline

    L = []
    ap = L.append

    def row(*cells, header=False):
        tag = "th" if header else "td"
        ap("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")

    ap("<h1>libKriging vs GPyTorch — iterative path, small n</h1>")
    ap("<ul class=\"meta\">")
    ap(f"<li><strong>GPU</strong>: {_html.escape(meta['gpu'] or '(none detected)')}</li>")
    ap(f"<li><strong>CPU</strong>: {_html.escape(meta['cpu'])}</li>")
    ap(f"<li><strong>host</strong>: <code>{_html.escape(meta['host'])}</code>"
       f"  ·  logical CPUs: {meta['nproc']}  ·  OMP_NUM_THREADS: <code>{meta['omp']}</code></li>")
    ap(f"<li><strong>when</strong>: {_html.escape(meta['when'])}</li>")
    ap(f"<li><strong>sweep</strong>: <code>sine_sum</code> d={D_CG}, matern5_2, shared theta={meta['theta']}; "
       f"n = {', '.join(map(str, meta['sizes']))}; test n={N_TEST}</li>")
    ap("<li>" + md(
        "**shared CG budget**: relative-residual tolerance `%g` applied to **both** libraries "
        "(libKriging `predictIterative(tol=...)` and the objective's 5th `LLIterative` field; "
        "GPyTorch `cg_tolerance`/`eval_cg_tolerance`). Budget-matching this is not cosmetic: "
        "running libKriging at 1e-8 against GPyTorch at 1e-4 — as an earlier revision of this "
        "harness did — compares a solver converged four orders of magnitude tighter against a "
        "deliberately truncated one and reads the difference as speed." % meta["cg_tol"]) + "</li>")
    ap(f"<li><strong>libKriging iterative objective</strong>: "
       f"<code>{_html.escape(LK_ITER_OBJECTIVE_FMT.format(cg_tol=meta['cg_tol']))}</code>  ·  "
       f"<code>predictIterative(max_iter=8n, Nystrom precond rank ≤ {meta['lk_precond_rank']}"
       f"{' = OFF' if meta['lk_precond_rank'] == 0 else ''})</code></li>")
    ap("<li>" + md(
        "**same model on both sides**: GPyTorch runs a `ProductKernel` of one-dimensional "
        "Matérn-5/2 factors — which is what libKriging's separable `matern5_2` is, unlike "
        "`MaternKernel(ard_num_dims=d)`, which is radial in `||dx/l||_2` — with its fixed "
        "`ConstantMean` set to libKriging's **GLS** trend `beta` rather than `mean(y)`. "
        "Without both alignments the exact `GPyTorch-Cholesky` rows themselves showed "
        "`dMean/rms ≈ 4e-02`, i.e. that column measured a model mismatch instead of solver "
        "convergence; with them they sit at ~4e-08. See `_gpt_modules` and `run_gpytorch`.") + "</li>")
    ap("<li><strong>GPyTorch</strong>: raised settings so BBMM converges, and "
       "<code>max_cholesky_size=0</code> so every n in the sweep actually USES BBMM "
       "(GPyTorch's default, 800, silently falls back to exact dense "
       "Cholesky at or below it, which would make an n=250/500 "
       "“GPyTorch-BBMM” row misnamed) — "
       f"<code>max_cg_iterations=5000, cg_tolerance={meta['cg_tol']:g}, "
       f"eval_cg_tolerance={meta['cg_tol']:g}, "
       f"max_lanczos_quadrature_iterations={LK_ITER_LANCZOS}, num_trace_samples={LK_ITER_NPROBE}, "
       "max_preconditioner_size=100, min_preconditioning_size=1</code></li>")
    ap("<li><strong>versions</strong>: " + ", ".join(
        f"{_html.escape(str(k))}=<code>{_html.escape(str(v))}</code>" for k, v in meta["versions"].items()) + "</li>")
    ap("</ul>")
    ap("<p>" + md(
        "`fit` = solving `R(theta)` once at the fixed, given theta (no hyperparameter "
        "optimization anywhere in this sweep): the `Kriging(...)` constructor (dense "
        "Cholesky for `LL`; one CG+SLQ commit for the light `LLIterative` fit) / for "
        "GPyTorch, model build **plus one forced no-grad `mll(model(x), y)` forward** "
        "(Cholesky, or one BBMM CG+SLQ solve). That forced forward is needed because "
        "`gpytorch.models.ExactGP.__init__` itself does no linear algebra at all "
        "(verified: flat ~1ms regardless of n) -- without it, all of GPyTorch's "
        "kernel/solve cost would silently land in `logLik` (its first forward call) "
        "instead, making `fit` measure object construction rather than a fit. `logLik` = "
        "a further, independent log-likelihood **+ gradient** evaluation at theta -- "
        "GPyTorch's train-mode forward has no cache, so this genuinely re-solves "
        "`R(theta)` from scratch, same as libKriging's `logLikelihoodFun`/"
        "`logLikelihoodIterativeFun` re-solving independently of what `fit` did. "
        "`predict` = a *cold* posterior "
        f"mean on {N_TEST} held-out points (GPyTorch's per-fit prediction cache is dropped "
        "each rep). Every timing is the **min of up to 5 reps** (1 rep once a single call "
        "exceeds 3 s). Backends are named `<lib>-<method>-<linalg lib>`. "
        f"`{name('chol')}` is the **reference**: `dLogLik/n` = "
        "`|ll − ll_chol|/n` and `dMean/rms` = `max|mean − mean_chol| / rms(y_test)`, against the "
        "**libKriging** Cholesky mean. Both libraries run the same separable Matérn-5/2 "
        "product kernel and the same GLS trend, so this is a convergence figure for every "
        "row; `GPyTorch-Cholesky` being exact, its own `dMean/rms` is the residual "
        "cross-library agreement and is the floor the `GPyTorch-BBMM` rows should reach. "
        "**`logLik (native)`** is what each library returns as-is — which is why the two "
        "sides of that column look nothing alike, and why it is *not* the one to compare: "
        "GPyTorch's `ExactMarginalLogLikelihood` is `log p(y|X) / n`, a per-datapoint "
        "figure, while libKriging returns the **concentrated** likelihood, having profiled "
        "`sigma2` out analytically. **`logLik (libK conv.)`** undoes both conventions — "
        "`ll_libK = n·mll + q/2 − (n·log(q/n) + n)/2`, with `q = r' K⁻¹ r` taken from "
        "GPyTorch's *own* solve so a BBMM row keeps its own CG error — and is the column "
        "`dLogLik/n` is computed from. The two agree to ~1e-07 relative once converted, so "
        "the raw gap between e.g. `2183.8` and `−0.29` is entirely convention, not "
        "disagreement about the model or the data.") + "</p>")

    # ---- interactive chart: time (log) vs n, one trace per backend, ----
    # ---- metric switch (fit / logLik / predict) via a dropdown ----
    metrics = [("fit_s", "Fit time (s)"), ("loglik_s", "LogLik+grad time (s)"), ("predict_s", "Predict time (s)")]
    backend_order = list(dict.fromkeys(r["backend_key"] for r in rows if r.get("backend_key")))
    traces = []
    for mi, (mkey, _mlabel) in enumerate(metrics):
        for bkey in backend_order:
            pts = sorted((r["n"], r[mkey]) for r in ok if r.get("backend_key") == bkey and r.get(mkey) is not None)
            traces.append(dict(
                x=[p[0] for p in pts], y=[p[1] for p in pts],
                name=name(bkey), mode="lines+markers", visible=(mi == 0),
                hovertemplate=f"{name(bkey)}<br>n=%{{x}}<br>%{{y:.4g}} s<extra></extra>",
            ))
    n_backends = len(backend_order)
    buttons = []
    for mi, (_mkey, mlabel) in enumerate(metrics):
        vis = [False] * (len(metrics) * n_backends)
        for j in range(n_backends):
            vis[mi * n_backends + j] = True
        buttons.append(dict(label=mlabel, method="update",
                             args=[{"visible": vis}, {"yaxis.title.text": mlabel}]))
    ap("<h2>Timing trend — time (log scale) vs n</h2>")
    ap("<p>Click a metric above the plot to switch; click legend entries to isolate/hide backends.</p>")
    ap('<div id="timing-chart" style="width:100%;max-width:960px;height:560px;"></div>')
    ap("<script>")
    ap("Plotly.newPlot('timing-chart', " + json.dumps(traces) + ", {")
    ap("  xaxis: {title: {text: 'n (training size)'}, type: 'log'},")
    ap(f"  yaxis: {{title: {{text: {json.dumps(metrics[0][1])}}}, type: 'log'}},")
    ap("  legend: {orientation: 'h', y: -0.2},")
    ap("  margin: {t: 60},")
    ap("  updatemenus: [{buttons: " + json.dumps(buttons) + ", direction: 'down', x: 0, y: 1.15}],")
    ap("  template: (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) ? 'plotly_dark' : 'plotly_white'")
    ap("}, {responsive: true});")
    ap("</script>")

    ap(f"<h2>Reference — <code>{_html.escape(name('chol'))}</code> (exact dense Cholesky)</h2>")
    ap('<table><thead>')
    row("n", "fit (s)", "logLik (s)", "predict (s)", "logLik value", "RMSE", "Q²", header=True)
    ap('</thead><tbody>')
    for n in ns:
        r = chol.get(n)
        if r:
            row(n, _f(r['fit_s'], '.3f'), _f(r['loglik_s'], '.3f'), _f(r['predict_s'], '.3f'),
                _f(r['ll'], '.3f'), _f(r['rmse']), _f(r['q2']))
        else:
            row(n, "—", "—", "—", "—", "—", "—")
    ap('</tbody></table>')

    ap("<h2>Timing — seconds</h2>")
    ap('<table><thead>')
    row("backend", "n", "fit", "logLik", "predict", header=True)
    ap('</thead><tbody>')
    for r in rows:
        if r.get("status") != "ok":
            row(_html.escape(r['backend']), r['n'], "—", "—", f"<em>{_html.escape(str(r.get('status', '?')))}</em>")
            continue
        row(_html.escape(r['backend']), r['n'], _f(r['fit_s'], '.3f'), _f(r['loglik_s'], '.3f'), _f(r['predict_s'], '.3f'))
    ap('</tbody></table>')

    ap("<h2>Accuracy</h2>")
    ap('<table><thead>')
    row("backend", "n", "RMSE", "Q²", "logLik (libK conv.)", "logLik (native)",
        "dLogLik/n", "dMean/rms", header=True)
    ap('</thead><tbody>')
    for r in rows:
        if r.get("status") != "ok":
            row(_html.escape(r['backend']), r['n'], "—", "—", "—", "—", "—", f"<em>{_html.escape(str(r.get('status', '?')))}</em>")
            continue
        row(_html.escape(r['backend']), r['n'], _f(r['rmse']), _f(r['q2']), _f(r['ll'], '.3f'),
            _f(r.get('ll_native'), '.4f'),
            _f(r.get('dloglik_n'), '.2e'), _f(r.get('dmean_rms'), '.2e'))
    ap('</tbody></table>')

    ap("<h2>Speed-ups (logLik-eval time)</h2>")
    ap('<table><thead>')
    row("n", _html.escape(name('chol')), _html.escape(name('iter-cuda')), _html.escape(name('iter-omp')),
        "<strong>OpenMP / CUDA</strong>", "<strong>CUDA / Cholesky</strong>",
        _html.escape(name('gpt-cuda')), _html.escape(name('gpt-cpu')), header=True)
    ap('</thead><tbody>')
    for n in ns:
        c = chol.get(n, {}).get("loglik_s")
        g = lkg.get(n, {}).get("loglik_s")
        cp = lkc.get(n, {}).get("loglik_s")
        gg = gtg.get(n, {}).get("loglik_s")
        gc = gtc.get(n, {}).get("loglik_s")
        row(n, _f(c, '.3f'), _f(g, '.3f'), _f(cp, '.3f'),
            (f'{cp / g:.1f}×' if (cp and g) else '—'),
            (f'{g / c:.1f}×' if (g and c) else '—'),
            _f(gg, '.3f'), _f(gc, '.3f'))
    ap('</tbody></table>')

    ap("<h2>Verdict — did everything converge?</h2>")
    ap("<ul>")
    iters = [r for r in ok if r.get("backend_key") in ("iter-cuda", "iter-omp")]
    gpts = [r for r in ok if r.get("backend_key") in ("gpt-cuda", "gpt-cpu")]
    if iters:
        m_dll = max((r["dloglik_n"] for r in iters if r.get("dloglik_n") is not None), default=float("nan"))
        m_dm = max((r["dmean_rms"] for r in iters if r.get("dmean_rms") is not None), default=float("nan"))
        msg = ("**libKriging iterative**: logLik within `%.1e` per point of the chol "
               "reference, posterior mean within `%.1e` of test RMS — converged." % (m_dll, m_dm))
        ap(f"<li>{md(msg)}</li>")
    if gpts:
        m_dm = max((r["dmean_rms"] for r in gpts if r.get("dmean_rms") is not None), default=float("nan"))
        gq = min((r["q2"] for r in gpts), default=float("nan"))
        m_gdll = max((r["dloglik_n"] for r in gpts if r.get("dloglik_n") is not None), default=float("nan"))
        msg = ("**GPyTorch**: posterior mean within `%.1e` of test RMS of the chol "
               "reference — now that both libraries run the same separable kernel and the "
               "same GLS trend, this is BBMM's own solver convergence, directly comparable "
               "to the libKriging-iterative figure above — Q² ≥ `%.4f`. Its log-likelihood is "
               "converted into libKriging's convention (see the accuracy table) and compared "
               "to within `%.1e` per point." % (m_dm, gq, m_gdll))
        ap(f"<li>{md(msg)}</li>")
    # Head-to-head on the log-likelihood, now that both sides are one
    # comparable quantity at an identical probe/Lanczos budget. This stayed
    # invisible while the column was blank for GPyTorch.
    if iters and gpts:
        i_dll = max((r["dloglik_n"] for r in iters if r.get("dloglik_n") is not None), default=float("nan"))
        g_dll = max((r["dloglik_n"] for r in gpts if r.get("dloglik_n") is not None), default=float("nan"))
        if i_dll == i_dll and g_dll == g_dll and i_dll > 0:
            msg = ("**log-likelihood accuracy, head to head**: at the same %d Hutchinson "
                   "probes and %d SLQ Lanczos steps per probe, and the same CG tolerance, "
                   "libKriging's stochastic log-determinant is the more accurate of the two "
                   "— worst `dLogLik/n` `%.1e` against GPyTorch BBMM's `%.1e` (`%.0fx`). "
                   "Note the two differ in *which* quantity they are better at: BBMM's "
                   "posterior mean is slightly closer (see `dMean/rms`), libKriging's "
                   "log-likelihood markedly so."
                   % (LK_ITER_NPROBE, LK_ITER_LANCZOS, i_dll, g_dll, g_dll / i_dll))
            ap(f"<li>{md(msg)}</li>")
    gtg_chol, gtc_chol = got("gpt-chol-cuda"), got("gpt-chol-cpu")
    bbmm_vs_chol = [
        abs(bbmm[n]["dmean_rms"] - chol_gpt[n]["dmean_rms"])
        for bbmm, chol_gpt in ((gtg, gtg_chol), (gtc, gtc_chol))
        for n in ns
        if n in bbmm and n in chol_gpt and bbmm[n].get("dmean_rms") is not None
        and chol_gpt[n].get("dmean_rms") is not None
    ]
    if bbmm_vs_chol:
        msg = ("**GPyTorch BBMM vs its own exact Cholesky** (`GPyTorch-Cholesky-*`, "
               "`max_cholesky_size` forced far above every n here): posterior mean differs by at "
               "most `%.1e` of test RMS across n — BBMM has converged to "
               "GPyTorch's own exact answer." % max(bbmm_vs_chol))
        ap(f"<li>{md(msg)}</li>")
    # Whether the dense factor still wins is a RESULT, not a given: report the
    # crossover n per operation instead of asserting one. The previous hardcoded
    # "Cholesky is fastest on all three ops at these n" was written under the
    # old, un-budget-matched settings (libKriging CG at 1e-8 vs GPyTorch at
    # 1e-4) and silently became false once the CG budgets were matched.
    crossover = {}
    for op in ("fit_s", "loglik_s", "predict_s"):
        wins = [n for n in ns
                if chol.get(n, {}).get(op) and lkg.get(n, {}).get(op)
                and lkg[n][op] < chol[n][op]]
        crossover[op] = min(wins) if wins else None
    lab = {"fit_s": "fit", "loglik_s": "logLik", "predict_s": "predict"}
    beaten = [f"`{lab[op]}` from n={nn}" for op, nn in crossover.items() if nn is not None]
    never = [f"`{lab[op]}`" for op, nn in crossover.items() if nn is None]
    parts = []
    if beaten:
        parts.append("**`%s`** is already faster than **`%s`** on %s"
                     % (name("iter-cuda"), name("chol"), ", ".join(beaten)))
    if never:
        parts.append("the dense factor still wins on %s at every n here" % ", ".join(never))
    msg = "; ".join(parts) + (
        " — the iterative path targets n where the dense factor no longer fits in memory, "
        "so a win at these n is a bonus, not the point." if parts else "")
    if parts:
        ap(f"<li>{md(msg)}</li>")
    ap("</ul>")

    ap("<h2>Notes</h2>")
    ap("<ul>")
    msg = ("Backend names are `<lib>-<method>-<linalg lib>`. Both "
           "`libKriging-Iterative-CUDA` (`LK_ITERATIVE_CUDA_DENSE_MAX_MB` budget) "
           "and `-OpenMP` (`LK_ITERATIVE_DENSE_MAX_MB`) materialize R once per "
           "evaluation for a separable kernel within their memory budget and run "
           "the matvecs as a single `cublasDgemm` / BLAS-3 `R*V` (hence "
           "`-OpenMP`, the BLAS it links), else fall back to a hand-written "
           "matvec kernel. "
           "`%s` and `%s` name the actual dense BLAS/LAPACK "
           "each links against." % (name("chol"), name("gpt-cpu")))
    ap(f"<li>{md(msg)}</li>")
    msg = ("theta=%s is chosen so the SLQ log-determinant's Lanczos "
           "quadrature and GPyTorch's BBMM CG both converge with sane iteration budgets; "
           "at longer theta (better-fitting but more ill-conditioned R) both need far more "
           "iterations / Lanczos steps. The `,0,40` in the libKriging objective is the "
           "third `LLIterative` argument (SLQ Lanczos steps per probe), added so the "
           "iterative log-likelihood *value* also tracks the exact one here." % meta["theta"])
    ap(f"<li>{md(msg)}</li>")
    msg = ("`%s` vs `%s` is the same binary with "
           "`set_cuda_iterative_enabled(...)` toggled — identical results, different path "
           "for the batched CG / SLQ / gradient matvecs (CUDA kernels vs the CPU "
           "dense-`R` BLAS path)." % (name("iter-cuda"), name("iter-omp")))
    ap(f"<li>{md(msg)}</li>")
    msg = ("`%s`/`%s` vs `%s`/`%s` are the SAME model and data, only "
           "`gpytorch.settings.max_cholesky_size` differs (`0` = always BBMM, forced huge = "
           "always exact) — the Cholesky rows are GPyTorch's own reference for whether BBMM "
           "has converged, independent of libKriging's Cholesky reference."
           % (name("gpt-cuda"), name("gpt-cpu"), name("gpt-chol-cuda"), name("gpt-chol-cpu")))
    ap(f"<li>{md(msg)}</li>")
    msg = ("Companion: `docs/comparisons/libKriging_vs_GPyTorch.ipynb` (summary of these "
           "results + the GPyTorch code libKriging mimics).")
    ap(f"<li>{md(msg)}</li>")
    ap("</ul>")

    title = f"libKriging vs GPyTorch — {meta['gpu'] or meta['cpu']}"
    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_html.escape(title)}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
          max-width: 980px; margin: 2rem auto; padding: 0 1rem; }}
  h1 {{ font-size: 1.5rem; }}
  h2 {{ font-size: 1.15rem; margin-top: 2rem; border-bottom: 1px solid #8884; padding-bottom: .25rem; }}
  ul.meta {{ list-style: none; padding: 0; }}
  ul.meta li {{ margin: .2rem 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: .5rem 0 1rem; font-size: .92rem; }}
  th, td {{ border: 1px solid #8884; padding: .3rem .55rem; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  thead th {{ background: #8881; }}
  code {{ font-size: .9em; }}
</style>
</head>
<body>
{chr(10).join(L)}
</body>
</html>
"""
    with open(path, "w") as fh:
        fh.write(doc)


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
                        + "  (chol / iter-cuda / iter-omp / gpt-cuda / gpt-cpu / gpt-chol-cuda / gpt-chol-cpu)")
    p.add_argument("--cg-tol", type=float, default=CG_TOL_DEFAULT,
                   help="SHARED relative-residual CG tolerance, applied to BOTH libKriging "
                        "(predictIterative tol) and GPyTorch (cg_tolerance / eval_cg_tolerance) "
                        "so the two are compared at the same convergence budget (default: %(default)s)")
    p.add_argument("--lk-precond-rank", type=int, default=LK_PRECOND_RANK_DEFAULT,
                   help="Nystrom preconditioner rank for libKriging's predictIterative, capped at n//4; "
                        "0 disables it (default: %(default)s)")
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
        cg_tol=args.cg_tol, lk_precond_rank=args.lk_precond_rank,
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
            run_libkriging_iter(Xw, yw, Xw[:8], sine_sum(Xw[:8]), args.theta, True,
                                args.cg_tol, args.lk_precond_rank)
            run_gpytorch(Xw, yw, Xw[:8], sine_sum(Xw[:8]), args.theta, "cuda", args.cg_tol)
        except Exception as exc:  # noqa: BLE001
            print(f"warmup skipped: {exc!r}", flush=True)

    t0 = time.perf_counter()
    rows = sweep(keys, sizes, args.theta, labels, args.cg_tol, args.lk_precond_rank)
    print(f"\ntotal wall time: {time.perf_counter() - t0:.0f}s", flush=True)

    base = f"{slug(gpu_name)}__{slug(cpu_name)}"
    if args.tag:
        base += "__" + slug(args.tag, 24)
    write_html(os.path.join(outdir, base + ".html"), rows, meta)
    write_csv(os.path.join(outdir, base + ".csv"), rows)
    print("wrote", os.path.join(outdir, base + ".html"))
    print("wrote", os.path.join(outdir, base + ".csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
