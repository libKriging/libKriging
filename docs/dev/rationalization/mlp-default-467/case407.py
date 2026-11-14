"""#467: the projected-Branin 10D case of #407 (n 120, d 10, d_out 2, hidden [16, 8]).

Design 0 is the design of #407 (LHS seed 44, W from seed 43, 2000 test points seed 10045);
design k > 0 changes the LHS seed to 44 + 1000 k (same W and test set).
Usage: python3 case407.py <out.csv> <optim,...> <folds,...> <activation,...> <design0> <ndesigns>
(optim "Kriging": plain Kriging(matern5_2) with its default optimizer, the reference.)
One CSV row per fit: design, activation, optim, folds, wall, cpu, LL, Q2, NLPD, cov95.
The thread budget comes from the environment (taskset + LK_NUM_THREADS_PERCENT), as for any
libKriging program.
"""
import os
import sys
import time

import numpy as np
from scipy.stats import qmc

import pylibkriging as lk

out, optims, foldss, acts = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(","), sys.argv[4].split(",")
d0, nd = int(sys.argv[5]), int(sys.argv[6])

rng = np.random.default_rng(43)
W = rng.normal(size=(2, 10))
W /= np.abs(W).sum(axis=1, keepdims=True)


def branin(U):
    x1, x2 = 15 * U[:, 0] - 5, 15 * U[:, 1]
    return (x2 - 5.1 / (4 * np.pi**2) * x1**2 + 5 / np.pi * x1 - 6) ** 2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10


def f(X):
    return branin((X @ W.T - W.clip(max=0).sum(axis=1)) / np.abs(W).sum(axis=1))


Xt = np.random.default_rng(10045).random((2000, 10))
yt = f(Xt)
new = not os.path.exists(out)
with open(out, "a") as fo:
    if new:
        fo.write("design,activation,optim,folds,wall,cpu,LL,Q2,NLPD,cov95\n")
    for design in range(d0, d0 + nd):
        X = qmc.LatinHypercube(d=10, seed=44 + 1000 * design, optimization="random-cd").random(120)
        y = f(X)
        for act in acts:
            for optim in optims:
                for folds in foldss:
                    time.sleep(0.2)  # let the BLAS threads of the previous predict go idle
                    c0, w0 = time.process_time(), time.perf_counter()
                    if optim == "Kriging":  # reference: plain Kriging, its default optimizer
                        k = lk.Kriging(y, X, "matern5_2")
                    else:
                        k = lk.MLPKriging(y, X, hidden_dims=[16, 8], d_out=2, activation=act, kernel="matern5_2",
                                          optim=optim, parameters={"calibration_folds": folds})
                    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
                    m, s, *_ = k.predict(Xt, return_stdev=True)
                    m, s = m.ravel(), s.ravel()
                    e = yt - m
                    q2 = 1 - np.sum(e**2) / np.sum((yt - yt.mean()) ** 2)
                    nlpd = np.mean(0.5 * np.log(2 * np.pi) + np.log(s) + 0.5 * (e / s) ** 2)
                    cov = np.mean(np.abs(e) <= 1.96 * s)
                    fo.write(f"{design},{act},{optim},{folds},{wall:.4f},{cpu:.4f},{k.logLikelihood():.6g},"
                             f"{q2:.6g},{nlpd:.6g},{cov:.4f}\n")
                    fo.flush()
