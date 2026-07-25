"""Merge python.csv and r.csv into results/all.csv and a markdown summary
(median over repetitions, with [q25; q75]) suitable for $GITHUB_STEP_SUMMARY.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd


def fmt(series, digits=3):
    s = series.dropna()
    if s.empty:
        return "—"
    med, q1, q3 = s.median(), s.quantile(0.25), s.quantile(0.75)
    return f"{med:.{digits}g} [{q1:.{digits}g}; {q3:.{digits}g}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/summary.md")
    args = ap.parse_args()

    frames = [pd.read_csv(f) for f in glob.glob(
        os.path.join(args.results, "*.csv"))
        if os.path.basename(f) not in ("all.csv",)]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(args.results, "all.csv"), index=False)

    lines = ["# libKriging comparison benchmark",
             "",
             f"Repetitions per case: {df.groupby(['func','n','package'])['rep'].count().max()}"
             " — identical LHS designs shared by all packages.",
             "Median [q25; q75] over repetitions. `—` = no successful run.",
             ""]
    for (func, n), sub in df.groupby(["func", "n"]):
        d = int(sub["d"].iloc[0])
        lines += [f"## {func} (d={d}, n={n})", "",
                  "| package | fit time (s) | pred time (s) | RMSE | Q² | NLPD | ok/total |",
                  "|---|---|---|---|---|---|---|"]
        for pkg, g in sub.groupby("package"):
            ok = (g["status"] == "ok").sum()
            lines.append(
                f"| {pkg} | {fmt(g['fit_time'])} | {fmt(g['pred_time'])} "
                f"| {fmt(g['rmse'])} | {fmt(g['q2'], 4)} | {fmt(g['nlpd'])} "
                f"| {ok}/{len(g)} |")
        lines.append("")

    lines += ["### Caveats",
              "",
              "Packages differ in optimizer, restarts, parameter bounds and",
              "internal scaling; this compares *default MLE fits* with a",
              "Matern 5/2 ARD kernel and constant trend, not tuned setups.",
              ""]
    out = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(out)
    print(out)


if __name__ == "__main__":
    main()
