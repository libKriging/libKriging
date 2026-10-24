#!/usr/bin/env python3
"""Summarize warp_start_study.cpp CSVs, paired by data set: warpstart-<prefix>-*
(default prefix "log": log-uniform draw for all warps; "final": the committed
rule, log-uniform for per-variable warps, uniform for the joint MLP, from the
Random generator) against warpstart-unif-* (uniform, Armadillo generator).
Usage: summarize_warp_start.py [prefix]"""
import csv, glob, statistics as st, sys

def load(p):
    with open(p) as f:
        return {int(r["set"]): r for r in csv.DictReader(f)}

def med(v):
    return st.median(v) if v else float("nan")

print("| config | LL new − unif (median) | new better / worse (> 0.1) | Q2 new / unif (median) | NLPD new / unif (median) | ms new / unif |")
print("|---|---|---|---|---|---|")
prefix = sys.argv[1] if len(sys.argv) > 1 else "log"
for pl in sorted(glob.glob(f"warpstart-{prefix}-*.csv")):
    cfg = pl[len(f"warpstart-{prefix}-"):-4]
    a, b = load(pl), load("warpstart-unif-" + cfg + ".csv")
    keys = [k for k in a if k in b and a[k]["ll"] != "nan" and b[k]["ll"] != "nan"]
    dll = [float(a[k]["ll"]) - float(b[k]["ll"]) for k in keys]
    better = sum(x > 0.1 for x in dll); worse = sum(x < -0.1 for x in dll)
    g = lambda r, c: [float(r[k][c]) for k in keys]
    print(f"| {cfg} | {med(dll):.3g} | {better} / {worse} (of {len(keys)}) | {med(g(a,'q2')):.3g} / {med(g(b,'q2')):.3g} "
          f"| {med(g(a,'nlpd')):.3g} / {med(g(b,'nlpd')):.3g} | {med(g(a,'ms')):.0f} / {med(g(b,'ms')):.0f} |")
