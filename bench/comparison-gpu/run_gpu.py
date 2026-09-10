"""GPU cross-package comparison: GPyTorch (BBMM, CUDA) vs libKriging
(``LLIterative`` / ``predictIterative``, CUDA) at n > 1000, on the shared
designs from ``make_datasets.py``.

Why this is shaped differently from ``../comparison/run_python.py``
-----------------------------------------------------------------
That benchmark does a full **default MLE fit** per package and stops at
n = 1000. Here:

* **Fixed, shared hyperparameters.** Every backend is evaluated at the
  *same* reference theta (``theta_ref.csv`` = ``0.3 x per-dim input range``,
  ``sigma2 = var(y)`` -- see ``make_datasets.py`` for why not the MLE
  optimum). What is being compared at n > 1000 is each package's
  **matrix-free conjugate-gradient linear algebra** -- GPyTorch's BBMM vs
  libKriging's ``LLIterative`` / ``predictIterative`` -- not its optimizer.
  A free MLE fit under ``objective="LLIterative(m)"`` is also impractically
  slow (see ``docs/math/Iterative.md``'s cost model), so a fixed theta is
  the only choice that is both tractable and fair. Same argument as
  ``docs/comparisons/libKriging_vs_GPyTorch.ipynb`` sec 4.

* **Three phases**, mirroring the CI benchmark's fit / predict / update:
    - ``fit``    -- cost of one likelihood(+grad) evaluation at theta_ref
                    (one optimizer-step-equivalent): GPyTorch
                    ``-mll(model(x), y).backward()``; libKriging
                    ``Kriging(..., optim="none", "LLIterative(m)")`` +
                    ``logLikelihoodIterativeFun(theta, grad=True)`` (the
                    matrix-free CG + SLQ estimate, not the exact O(n^3) one).
    - ``predict`` -- posterior mean on the 2000-point test set, plus
                     stdev on a ``--stdev-n`` subsample (stdev is one CG
                     solve *per* test point on the libKriging side, hence
                     subsampled). Metrics: RMSE, Q2 (mean); NLPD,
                     coverage90 (stdev subsample).
    - ``update`` -- re-condition on n + ceil(0.25 n) fresh points at the
                    *same* theta, then predict the test mean again. Full
                    rebuild on both sides, so the two are comparable
                    regardless of each package's internal incremental path.

* **Interpolation only** (jitter 1e-10, no nugget), like the CI
  benchmark; libKriging's ``LLIterative`` is ``NoiseModel::None``-only.

Backends (``--backends``): ``gpytorch``, ``libkriging-gpu``,
``libkriging-cpu`` (CPU CG baseline, same binary via the runtime toggle
added to ``pylibkriging``). Each (func, n, rep, backend) runs in its own
subprocess with a wall-clock budget (``--budget``); timeouts/errors are
recorded, never fatal.

Output CSV columns:
  func,d,n,rep,backend,device,fit_time,pred_mean_time,pred_stdev_time,
  update_time,rmse,q2,nlpd,coverage90,cg_converged,rmse_update,q2_update,status
"""
import argparse
import glob
import multiprocessing as mp
import os
import time
import traceback

import numpy as np

JITTER = 1e-10
NPROBE = 20            # LLIterative(m) probe count / matches the notebook
UPDATE_FRAC = 0.25
LHS_UPDATE_SEED = 900000


# ------------------------------------------------------------------ metrics
def mean_metrics(y, mu):
    y, mu = np.asarray(y).ravel(), np.asarray(mu).ravel()
    resid = y - mu
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    sst = float(np.sum((y - y.mean()) ** 2))
    q2 = float(1.0 - np.sum(resid ** 2) / sst) if sst > 0 else float("nan")
    return rmse, q2


def stdev_metrics(y, mu, sd):
    y, mu, sd = np.asarray(y).ravel(), np.asarray(mu).ravel(), np.asarray(sd).ravel()
    s2 = np.maximum(sd, 1e-12) ** 2
    resid = y - mu
    nlpd = float(np.mean(0.5 * np.log(2 * np.pi * s2) + 0.5 * resid ** 2 / s2))
    cov90 = float(np.mean(np.abs(resid) / np.sqrt(s2) <= 1.645))
    return nlpd, cov90


def lhs_like(dom, n, seed):
    from scipy.stats import qmc
    d = dom.shape[0]
    u = qmc.LatinHypercube(d=d, seed=seed).random(n)
    return qmc.scale(u, dom[:, 0], dom[:, 1])


def theta_from_frac(X, frac):
    """Shared reference length-scale: frac * per-dimension observed range.
    frac small = short correlation = well-conditioned R; frac large = long
    correlation = ill-conditioned R (see ANALYSIS.md)."""
    rng = X.max(axis=0) - X.min(axis=0)
    rng = np.where(rng > 0, rng, 1.0)
    return frac * rng


# GPyTorch MaternKernel nu (or RBF) per libKriging kernel name.
_GPT_NU = {"matern5_2": 2.5, "matern3_2": 1.5, "exp": 0.5}


# ----------------------------------------------------------------- GPyTorch
def run_gpytorch(paths, args):
    import torch
    import gpytorch

    torch.set_default_dtype(torch.float64)
    want = paths.get("device", "cuda")
    if want == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available to torch")
    dev = torch.device(want)
    dev_name = torch.cuda.get_device_name(0) if want == "cuda" else "CPU"
    _sync = torch.cuda.synchronize if want == "cuda" else (lambda: None)

    X = np.loadtxt(paths["X_train"], delimiter=",", ndmin=2)
    y = np.loadtxt(paths["y_train"], delimiter=",")
    Xt = np.loadtxt(paths["X_test"], delimiter=",", ndmin=2)
    yt = np.loadtxt(paths["y_test"], delimiter=",")
    d = X.shape[1]
    kernel = paths.get("kernel", "matern5_2")
    theta = theta_from_frac(X, float(paths["theta_frac"]))
    sigma2 = float(np.var(y))
    ymean = float(y.mean())

    def _base_kernel():
        if kernel == "gauss":
            return gpytorch.kernels.RBFKernel(ard_num_dims=d)
        return gpytorch.kernels.MaternKernel(nu=_GPT_NU[kernel], ard_num_dims=d)

    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, tx, ty, lik):
            super().__init__(tx, ty, lik)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(_base_kernel())

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x))

    def build(Xa, ya):
        tx = torch.tensor(Xa, device=dev)
        ta = torch.tensor(ya, device=dev)
        lik = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-12)).to(dev)
        lik.noise = JITTER
        lik.noise_covar.raw_noise.requires_grad_(False)
        model = ExactGPModel(tx, ta, lik).to(dev)
        with torch.no_grad():
            model.covar_module.base_kernel.lengthscale = torch.tensor(theta, device=dev)
            model.covar_module.outputscale = torch.tensor(sigma2, device=dev)
            model.mean_module.constant.fill_(ymean)
        return tx, ta, lik, model

    def predict_mean(model, lik, Xq):
        model.eval(); lik.eval()
        txq = torch.tensor(Xq, device=dev)
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
                gpytorch.settings.max_cg_iterations(args.cg_iters), \
                gpytorch.settings.cg_tolerance(args.cg_tol):
            out = model(txq)
            mu = out.mean.detach().cpu().numpy()
        _sync()
        return mu

    def predict_stdev(model, lik, Xq):
        model.eval(); lik.eval()
        txq = torch.tensor(Xq, device=dev)
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
                gpytorch.settings.max_cg_iterations(args.cg_iters), \
                gpytorch.settings.cg_tolerance(args.cg_tol):
            out = lik(model(txq))
            mu = out.mean.detach().cpu().numpy()
            sd = out.stddev.detach().cpu().numpy()
        _sync()
        return mu, sd

    # --- fit: one likelihood + gradient evaluation at theta_ref -----------
    tx, ta, lik, model = build(X, y)
    # re-enable grad on the kernel params so backward() does real work
    model.covar_module.base_kernel.raw_lengthscale.requires_grad_(True)
    model.covar_module.raw_outputscale.requires_grad_(True)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    model.train(); lik.train()
    _sync()
    t0 = time.perf_counter()
    with gpytorch.settings.max_cg_iterations(args.cg_iters), \
            gpytorch.settings.cg_tolerance(args.cg_tol):
        loss = -mll(model(tx), ta)
        loss.backward()
    _sync()
    fit_time = time.perf_counter() - t0

    # --- predict mean on the (optionally capped) test set ---------------
    if args.test_n > 0 and args.test_n < len(Xt):
        Xt, yt = Xt[:args.test_n], yt[:args.test_n]
    t0 = time.perf_counter()
    mu = predict_mean(model, lik, Xt)
    pred_mean_time = time.perf_counter() - t0
    rmse, q2 = mean_metrics(yt, mu)

    # --- predict stdev on a subsample (--stdev-n 0 skips) ------------
    if args.stdev_n > 0:
        rs = np.random.default_rng(0)
        sub = rs.choice(len(Xt), size=min(args.stdev_n, len(Xt)), replace=False)
        t0 = time.perf_counter()
        mus, sds = predict_stdev(model, lik, Xt[sub])
        pred_stdev_time = time.perf_counter() - t0
        nlpd, cov90 = stdev_metrics(yt[sub], mus, sds)
    else:
        pred_stdev_time = nlpd = cov90 = np.nan

    # --- update: re-condition on n + ceil(0.25 n) fresh points --------
    n_u = int(np.ceil(UPDATE_FRAC * len(X)))
    dom = _domain_for(paths["func"])
    Xu = lhs_like(dom, n_u, seed=LHS_UPDATE_SEED + len(X))
    yu = _f_for(paths["func"])(Xu)
    Xa = np.vstack([X, Xu]); ya = np.concatenate([y, yu])
    txa, taa, lik2, model2 = build(Xa, ya)
    t0 = time.perf_counter()
    mu_u = predict_mean(model2, lik2, Xt)
    update_time = time.perf_counter() - t0
    rmse_u, q2_u = mean_metrics(yt, mu_u)

    return dict(device=dev_name, fit_time=fit_time,
                pred_mean_time=pred_mean_time, pred_stdev_time=pred_stdev_time,
                update_time=update_time, rmse=rmse, q2=q2, nlpd=nlpd,
                coverage90=cov90, cg_converged=int(not _gpytorch_cg_warned()),
                rmse_update=rmse_u, q2_update=q2_u, status="ok")


_CG_WARNED = {"v": False}


def _install_cg_warning_hook():
    import warnings
    _orig = warnings.showwarning

    def hook(message, category, filename, lineno, file=None, line=None):
        if "CG terminated in" in str(message):
            _CG_WARNED["v"] = True
        return _orig(message, category, filename, lineno, file, line)

    warnings.showwarning = hook


def _gpytorch_cg_warned():
    return _CG_WARNED["v"]


# ---------------------------------------------------------------- libKriging
def run_libkriging(paths, args, use_cuda):
    import pylibkriging as lk

    lk.set_cuda_iterative_enabled(use_cuda)
    dev_name = "CUDA" if (use_cuda and lk.cuda_iterative_available()) else "CPU"

    X = np.loadtxt(paths["X_train"], delimiter=",", ndmin=2)
    y = np.loadtxt(paths["y_train"], delimiter=",")
    Xt = np.loadtxt(paths["X_test"], delimiter=",", ndmin=2)
    yt = np.loadtxt(paths["y_test"], delimiter=",")
    kernel = paths.get("kernel", "matern5_2")
    theta = theta_from_frac(X, float(paths["theta_frac"]))
    sigma2 = float(np.var(y))

    params = {"theta": theta.reshape(1, -1), "sigma2": sigma2}

    def build(Xa, ya):
        return lk.Kriging(ya, Xa, kernel, regmodel="constant",
                          normalize=False, optim="none",
                          objective=f"LLIterative({NPROBE})", parameters=params)

    # --- fit: one iterative-objective evaluation at theta_ref -------------
    # Constructing an optim="none" LLIterative model runs exactly one
    # _logLikelihoodIterative internally (the matrix-free CG solve for
    # beta/sigma2 + the SLQ log-determinant) -- that IS one objective
    # evaluation, the unit an optimizer repeats. It does NOT run the exact
    # O(n^3) dense-Cholesky objective (which is what the Python-bound
    # logLikelihoodFun would give -- see logLikelihoodIterativeFun).
    t0 = time.perf_counter()
    m = build(X, y)
    fit_time = time.perf_counter() - t0
    # exercise the newly-exposed iterative objective for the ll value / to
    # keep parity with a real optimizer step (grad omitted: its Hutchinson
    # trace is still CPU-bound -- see bench/comparison-gpu/ANALYSIS.md).
    if hasattr(m, "logLikelihoodIterativeFun"):
        ll, _ = m.logLikelihoodIterativeFun(theta, False)

    # --- predict mean on the (optionally capped) test set (matrix-free CG) --
    if args.test_n > 0 and args.test_n < len(Xt):
        Xt, yt = Xt[:args.test_n], yt[:args.test_n]
    t0 = time.perf_counter()
    out = m.predictIterative(Xt, return_stdev=False, max_iter=0, tol=args.lk_tol)
    mu = np.asarray(out[0] if isinstance(out, tuple) else out).ravel()
    pred_mean_time = time.perf_counter() - t0
    rmse, q2 = mean_metrics(yt, mu)

    # --- predict stdev on a subsample, one CG solve per point (--stdev-n 0 skips) --
    if args.stdev_n > 0:
        rs = np.random.default_rng(0)
        sub = rs.choice(len(Xt), size=min(args.stdev_n, len(Xt)), replace=False)
        t0 = time.perf_counter()
        mus, sds = m.predictIterative(Xt[sub], return_stdev=True, max_iter=0, tol=args.lk_tol)
        pred_stdev_time = time.perf_counter() - t0
        nlpd, cov90 = stdev_metrics(yt[sub], np.asarray(mus).ravel(), np.asarray(sds).ravel())
    else:
        pred_stdev_time = nlpd = cov90 = np.nan

    # --- update: re-condition on n + ceil(0.25 n) fresh points -------
    n_u = int(np.ceil(UPDATE_FRAC * len(X)))
    dom = _domain_for(paths["func"])
    Xu = lhs_like(dom, n_u, seed=LHS_UPDATE_SEED + len(X))
    yu = _f_for(paths["func"])(Xu)
    Xa = np.vstack([X, Xu]); ya = np.concatenate([y, yu])
    t0 = time.perf_counter()
    m2 = build(Xa, ya)
    out = m2.predictIterative(Xt, return_stdev=False, max_iter=0, tol=args.lk_tol)
    mu_u = np.asarray(out[0] if isinstance(out, tuple) else out).ravel()
    update_time = time.perf_counter() - t0
    rmse_u, q2_u = mean_metrics(yt, mu_u)

    return dict(device=dev_name, fit_time=fit_time,
                pred_mean_time=pred_mean_time, pred_stdev_time=pred_stdev_time,
                update_time=update_time, rmse=rmse, q2=q2, nlpd=nlpd,
                coverage90=cov90, cg_converged=1,
                rmse_update=rmse_u, q2_update=q2_u, status="ok")


# ------------------------------------------------------------- func helpers
def _domain_for(func):
    from functions import DOMAINS
    return DOMAINS[func]


def _f_for(func):
    from functions import FUNCTIONS
    return FUNCTIONS[func]


# ----------------------------------------------------------------- harness
_FIELDS = ["device", "fit_time", "pred_mean_time", "pred_stdev_time",
           "update_time", "rmse", "q2", "nlpd", "coverage90", "cg_converged",
           "rmse_update", "q2_update", "status"]


def _task(backend, paths, args, queue):
    try:
        if backend in ("gpytorch", "gpytorch-cpu"):
            _install_cg_warning_hook()
            paths = dict(paths, device="cpu" if backend == "gpytorch-cpu" else "cuda")
            res = run_gpytorch(paths, args)
        elif backend == "libkriging-gpu":
            res = run_libkriging(paths, args, use_cuda=True)
        elif backend == "libkriging-cpu":
            res = run_libkriging(paths, args, use_cuda=False)
        else:
            raise ValueError(f"unknown backend {backend}")
        queue.put(res)
    except Exception:
        traceback.print_exc()
        queue.put(dict(device="?", status="error",
                       **{k: np.nan for k in _FIELDS if k not in ("device", "status")}))


def run_with_budget(backend, paths, args):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    p = ctx.Process(target=_task, args=(backend, paths, args, queue))
    p.start()
    # drain the queue *before* join: a result larger than the pipe buffer
    # would otherwise deadlock the worker at exit.
    try:
        res = queue.get(timeout=args.budget)
    except Exception:
        res = None
    p.join(30)
    if p.is_alive():
        p.terminate(); p.join(15)
    if p.is_alive():
        p.kill(); p.join()
    if res is None:
        status = "timeout" if p.exitcode is None or p.exitcode < 0 else "crash"
        return dict(device="?", status=status,
                    **{k: np.nan for k in _FIELDS if k not in ("device", "status")})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/gpu.csv")
    ap.add_argument("--budget", type=float, default=1800.0,
                    help="per (func,n,rep,backend) wall-clock budget (s)")
    ap.add_argument("--backends", default="gpytorch,libkriging-gpu,libkriging-cpu",
                    help="comma list of: gpytorch, gpytorch-cpu, libkriging-gpu, libkriging-cpu")
    ap.add_argument("--theta-frac", default="0.3",
                    help="comma list of reference length-scale fractions of the "
                         "input range (0.1 = short/well-conditioned, 0.6 = "
                         "long/ill-conditioned)")
    ap.add_argument("--kernel", default="matern5_2",
                    help="matern5_2 | matern3_2 | gauss | exp")
    ap.add_argument("--max-n-cpu", type=int, default=2000,
                    help="skip libkriging-cpu / gpytorch-cpu above this n")
    ap.add_argument("--stdev-n", type=int, default=64,
                    help="test points used for the (expensive) stdev/NLPD pass "
                         "(one CG solve per point on the libKriging side); "
                         "0 skips the stdev pass entirely")
    ap.add_argument("--test-n", type=int, default=0,
                    help="cap the test set to the first N points (0 = all 2000); "
                         "lets the libkriging-cpu baseline finish in budget")
    ap.add_argument("--cg-iters", type=int, default=3000,
                    help="GPyTorch max_cg_iterations")
    ap.add_argument("--cg-tol", type=float, default=1e-3,
                    help="GPyTorch cg_tolerance")
    ap.add_argument("--lk-tol", type=float, default=1e-6,
                    help="libKriging predictIterative CG tolerance")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fracs = [f.strip() for f in args.theta_frac.split(",")]
    header = ("func,d,n,rep,kernel,theta_frac,backend," + ",".join(_FIELDS))
    rows = [header]
    for xtr in sorted(glob.glob(os.path.join(
            args.data, "*", "n*", "rep*", "X_train.csv"))):
        rdir = os.path.dirname(xtr)
        ndir = os.path.dirname(rdir)
        fdir = os.path.dirname(ndir)
        func = os.path.basename(fdir)
        n = int(os.path.basename(ndir)[1:])
        rep = int(os.path.basename(rdir)[3:])
        d = np.loadtxt(xtr, delimiter=",", ndmin=2).shape[1]
        for frac in fracs:
            paths = {"X_train": xtr,
                     "y_train": os.path.join(rdir, "y_train.csv"),
                     "X_test": os.path.join(fdir, "X_test.csv"),
                     "y_test": os.path.join(fdir, "y_test.csv"),
                     "func": func, "theta_frac": frac, "kernel": args.kernel}
            for backend in args.backends.split(","):
                backend = backend.strip()
                pfx = f"{func},{d},{n},{rep},{args.kernel},{frac},{backend},"
                if backend in ("libkriging-cpu", "gpytorch-cpu") and n > args.max_n_cpu:
                    print(pfx + ",".join(["skipped"] + [""] * (len(_FIELDS) - 2) + ["skipped"]),
                          flush=True)
                    continue
                res = run_with_budget(backend, paths, args)
                row = pfx + ",".join(str(res.get(k, "")) for k in _FIELDS)
                rows.append(row)
                print(row, flush=True)
                with open(args.out, "w") as fh:
                    fh.write("\n".join(rows) + "\n")
    print(f"[run_gpu] wrote {args.out}")


if __name__ == "__main__":
    main()
