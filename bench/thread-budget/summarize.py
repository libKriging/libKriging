#!/usr/bin/env python3
"""Summarize bench-threads CSVs (docs/dev/parallel.md, bench/thread-budget/RESULTS.md).

    python3 bench/thread-budget/summarize.py file.csv [baseline.csv]

For each (n, d): the serial time (BLAS = OpenMP = 1), the default time
(both at nproc), and the best (BLAS, OpenMP) split, for t_llgrad_ms (one
log-likelihood + gradient evaluation). With a baseline file, also the
speed-up of the best split of `file` over the best split of the baseline.
"""
import os
import sys

import pandas as pd

COL = "t_llgrad_ms"


def summary(df):
    # Repeated configurations (e.g. OpenMP OpenBLAS, whose BLAS count follows
    # OMP_NUM_THREADS whatever OPENBLAS_NUM_THREADS says): median of the repeats.
    df = df.groupby(["n", "d", "blas_threads", "omp_threads"], as_index=False)[COL].median()
    ncpu = df["omp_threads"].max()
    rows = []
    for (n, d), g in df.groupby(["n", "d"]):
        serial = g[(g.blas_threads == 1) & (g.omp_threads == 1)][COL]
        default = g[(g.blas_threads == ncpu) & (g.omp_threads == ncpu)][COL]
        best = g.loc[g[COL].idxmin()]
        rows.append(
            dict(
                n=n,
                d=d,
                serial=serial.iloc[0] if len(serial) else float("nan"),
                default=default.iloc[0] if len(default) else float("nan"),
                best=best[COL],
                best_blas=int(best.blas_threads),
                best_omp=int(best.omp_threads),
            )
        )
    s = pd.DataFrame(rows)
    s["default/best"] = s["default"] / s["best"]
    s["serial/best"] = s["serial"] / s["best"]
    return s


def main():
    df = pd.read_csv(sys.argv[1])
    s = summary(df)
    if len(sys.argv) > 2:
        b = summary(pd.read_csv(sys.argv[2]))
        s = s.merge(b[["n", "d", "best"]], on=["n", "d"], suffixes=("", "_base"))
        s["gain_vs_base_%"] = 100 * (s["best_base"] / s["best"] - 1)
    pd.set_option("display.width", 200)
    print(f"# {os.path.basename(sys.argv[1])}: {COL}, blas_kind={sorted(df.blas_kind.unique())}")
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
