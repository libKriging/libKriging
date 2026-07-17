"""Generate shared, seeded LHS designs so that every package (Python and R)
is fitted on *identical* conditioning points.

Layout:
  data/<func>/X_test.csv, y_test.csv                  (common test set)
  data/<func>/n<N>/rep<K>/X_train.csv, y_train.csv    (K = 0..repeats-1)

All CSVs have no header; full float precision (repr) so R reads bitwise-
comparable values.
"""
import argparse
import os

import numpy as np
from scipy.stats import qmc

from functions import DOMAINS, FUNCTIONS

CASES = {  # function -> training sizes
    "branin": [50, 200, 1000],
    "hartmann3": [100, 300, 1000],
    "hartmann6": [200, 500, 1000],
    "borehole": [200, 500, 1000],
}
QUICK_CASES = {"branin": [50], "hartmann6": [200]}
N_TEST = 2000
TEST_SEED = 20260717


def lhs(n, dom, seed):
    d = dom.shape[0]
    u = qmc.LatinHypercube(d=d, seed=seed).random(n)
    return qmc.scale(u, dom[:, 0], dom[:, 1])


def save(path, arr):
    np.savetxt(path, np.atleast_2d(arr.T).T, delimiter=",", fmt="%.17g")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    cases = QUICK_CASES if args.quick else CASES
    for func, sizes in cases.items():
        f, dom = FUNCTIONS[func], DOMAINS[func]
        fdir = os.path.join(args.out, func)
        os.makedirs(fdir, exist_ok=True)
        Xt = lhs(N_TEST, dom, TEST_SEED)
        save(os.path.join(fdir, "X_test.csv"), Xt)
        save(os.path.join(fdir, "y_test.csv"), f(Xt))
        for n in sizes:
            for rep in range(args.repeats):
                rdir = os.path.join(fdir, f"n{n}", f"rep{rep}")
                os.makedirs(rdir, exist_ok=True)
                X = lhs(n, dom, seed=1000 * n + rep)  # deterministic per (n, rep)
                save(os.path.join(rdir, "X_train.csv"), X)
                save(os.path.join(rdir, "y_train.csv"), f(X))
        print(f"[make_datasets] {func}: sizes={sizes} repeats={args.repeats}")


if __name__ == "__main__":
    main()
