"""Benchmark pylibkriging against GPy, scikit-learn, SMT and OpenTURNS on the
shared datasets produced by make_datasets.py.

Common modelling choices (as close as each API allows):
  * Matern 5/2 anisotropic (ARD) kernel
  * constant trend / mean, estimated
  * interpolation (no nugget beyond numerical jitter)
  * hyperparameters by maximum likelihood, package defaults for the optimizer

Each fit runs in a subprocess with a wall-clock budget (--budget, default
300 s); failures and timeouts are recorded, never fatal.

Output: results/python.csv with columns
  func,d,n,rep,package,fit_time,pred_time,rmse,q2,nlpd,status
"""
import argparse
import glob
import multiprocessing as mp
import os
import time
import traceback

import numpy as np

JITTER = 1e-10


# ----------------------------------------------------------------- adapters
def fit_pylibkriging(X, y):
    import pylibkriging as lk
    t0 = time.perf_counter()
    m = lk.Kriging(y, X, "matern5_2", "constant", False, "BFGS", "LL")
    return m, time.perf_counter() - t0


def pred_pylibkriging(m, Xt):
    p = m.predict(Xt, True, False, False)
    return np.asarray(p[0]).ravel(), np.asarray(p[1]).ravel()


def fit_sklearn(X, y):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    k = ConstantKernel(1.0, (1e-5, 1e7)) * Matern(
        length_scale=np.ones(X.shape[1]),
        length_scale_bounds=(1e-6, 1e6), nu=2.5)
    t0 = time.perf_counter()
    m = GaussianProcessRegressor(kernel=k, alpha=JITTER, normalize_y=True,
                                 n_restarts_optimizer=0).fit(X, y)
    return m, time.perf_counter() - t0


def pred_sklearn(m, Xt):
    mu, sd = m.predict(Xt, return_std=True)
    return mu.ravel(), sd.ravel()


def fit_gpy(X, y):
    import GPy
    k = GPy.kern.Matern52(input_dim=X.shape[1], ARD=True)
    t0 = time.perf_counter()
    m = GPy.models.GPRegression(X, y.reshape(-1, 1), k)
    m.Gaussian_noise.variance = JITTER
    m.Gaussian_noise.variance.fix()
    m.optimize()
    return m, time.perf_counter() - t0


def pred_gpy(m, Xt):
    mu, var = m.predict(Xt, include_likelihood=False)
    return mu.ravel(), np.sqrt(np.maximum(var.ravel(), 0))


def fit_smt(X, y):
    from smt.surrogate_models import KRG
    t0 = time.perf_counter()
    m = KRG(corr="matern52", theta0=[1e-2] * X.shape[1],
            print_global=False)
    m.set_training_values(X, y)
    m.train()
    return m, time.perf_counter() - t0


def pred_smt(m, Xt):
    mu = m.predict_values(Xt).ravel()
    var = m.predict_variances(Xt).ravel()
    return mu, np.sqrt(np.maximum(var, 0))


def fit_openturns(X, y):
    import openturns as ot
    d = X.shape[1]
    t0 = time.perf_counter()
    cov = ot.MaternModel([1.0] * d, [1.0], 2.5)
    basis = ot.ConstantBasisFactory(d).build()
    algo = ot.KrigingAlgorithm(ot.Sample(X), ot.Sample(y.reshape(-1, 1)),
                               cov, basis)
    algo.run()
    return algo.getResult(), time.perf_counter() - t0


def pred_openturns(res, Xt):
    import openturns as ot
    meta = res.getMetaModel()
    mu = np.asarray(meta(ot.Sample(Xt))).ravel()
    var = np.asarray(
        res.getConditionalMarginalVariance(ot.Sample(Xt))).ravel()
    return mu, np.sqrt(np.maximum(var, 0))


PACKAGES = {
    "pylibkriging": (fit_pylibkriging, pred_pylibkriging),
    "sklearn": (fit_sklearn, pred_sklearn),
    "GPy": (fit_gpy, pred_gpy),
    "SMT": (fit_smt, pred_smt),
    "OpenTURNS": (fit_openturns, pred_openturns),
}


# ----------------------------------------------------------------- harness
def metrics(y, mu, sd):
    rmse = float(np.sqrt(np.mean((y - mu) ** 2)))
    q2 = float(1 - np.sum((y - mu) ** 2) / np.sum((y - np.mean(y)) ** 2))
    s2 = np.maximum(sd, 1e-12) ** 2
    nlpd = float(np.mean(0.5 * np.log(2 * np.pi * s2)
                         + 0.5 * (y - mu) ** 2 / s2))
    return rmse, q2, nlpd


def one_task(pkg, paths, queue):
    try:
        X = np.loadtxt(paths["X_train"], delimiter=",", ndmin=2)
        y = np.loadtxt(paths["y_train"], delimiter=",")
        Xt = np.loadtxt(paths["X_test"], delimiter=",", ndmin=2)
        yt = np.loadtxt(paths["y_test"], delimiter=",")
        fit, pred = PACKAGES[pkg]
        model, fit_time = fit(X, y)
        t0 = time.perf_counter()
        mu, sd = pred(model, Xt)
        pred_time = time.perf_counter() - t0
        rmse, q2, nlpd = metrics(yt, mu, sd)
        queue.put((fit_time, pred_time, rmse, q2, nlpd, "ok"))
    except Exception:
        traceback.print_exc()
        queue.put((np.nan,) * 5 + ("error",))


def run_with_budget(pkg, paths, budget):
    queue = mp.Queue()
    p = mp.Process(target=one_task, args=(pkg, paths, queue))
    p.start()
    p.join(budget)
    if p.is_alive():
        p.terminate()
        p.join()
        return (np.nan,) * 5 + ("timeout",)
    return queue.get() if not queue.empty() else (np.nan,) * 5 + ("crash",)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/python.csv")
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--packages", default=",".join(PACKAGES))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = ["func,d,n,rep,package,fit_time,pred_time,rmse,q2,nlpd,status"]
    for xtr in sorted(glob.glob(
            os.path.join(args.data, "*", "n*", "rep*", "X_train.csv"))):
        rdir = os.path.dirname(xtr)
        ndir = os.path.dirname(rdir)
        fdir = os.path.dirname(ndir)
        func = os.path.basename(fdir)
        n = int(os.path.basename(ndir)[1:])
        rep = int(os.path.basename(rdir)[3:])
        paths = {"X_train": xtr,
                 "y_train": os.path.join(rdir, "y_train.csv"),
                 "X_test": os.path.join(fdir, "X_test.csv"),
                 "y_test": os.path.join(fdir, "y_test.csv")}
        d = np.loadtxt(xtr, delimiter=",", ndmin=2).shape[1]
        for pkg in args.packages.split(","):
            res = run_with_budget(pkg, paths, args.budget)
            rows.append(f"{func},{d},{n},{rep},{pkg},"
                        + ",".join(str(v) for v in res))
            print(rows[-1], flush=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
