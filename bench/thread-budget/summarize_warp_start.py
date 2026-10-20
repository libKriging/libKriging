#!/usr/bin/env python3
"""Summarize warp_start_study.cpp CSVs: log-uniform (warpstart-log-*) against
uniform (warpstart-unif-*) starting points, paired by data set."""
import csv, glob, statistics as st, sys

def load(p):
    with open(p) as f:
        return {int(r["set"]): r for r in csv.DictReader(f)}

def med(v):
    return st.median(v) if v else float("nan")

print("| config | LL log − unif (median) | log better / worse (> 0.1) | Q2 log / unif (median) | NLPD log / unif (median) | ms log / unif |")
print("|---|---|---|---|---|---|")
for pl in sorted(glob.glob("warpstart-log-*.csv")):
    cfg = pl[len("warpstart-log-"):-4]
    a, b = load(pl), load("warpstart-unif-" + cfg + ".csv")
    keys = [k for k in a if k in b and a[k]["ll"] != "nan" and b[k]["ll"] != "nan"]
    dll = [float(a[k]["ll"]) - float(b[k]["ll"]) for k in keys]
    better = sum(x > 0.1 for x in dll); worse = sum(x < -0.1 for x in dll)
    g = lambda r, c: [float(r[k][c]) for k in keys]
    print(f"| {cfg} | {med(dll):.3g} | {better} / {worse} (of {len(keys)}) | {med(g(a,'q2')):.3g} / {med(g(b,'q2')):.3g} "
          f"| {med(g(a,'nlpd')):.3g} / {med(g(b,'nlpd')):.3g} | {med(g(a,'ms')):.0f} / {med(g(b,'ms')):.0f} |")
