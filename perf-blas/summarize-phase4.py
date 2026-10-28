#!/usr/bin/env python3
"""Median wall / CPU time and final log-likelihood per (class, optim, n, variant)."""
import csv, collections, statistics as st, sys
path = sys.argv[1] if len(sys.argv) > 1 else "perf-blas/phase4-fits.csv"
G = collections.defaultdict(list)
for r in csv.DictReader(open(path)):
    if r["seconds"] in ("FAIL", ""):
        continue
    G[(r["class"], r["optim"], int(r["n"]))].append(r)
print("| class | optim | n | seeds | variant | median wall (s) | median CPU (s) | CPU/wall | median LL | best LL (count) | wall vs master |")
print("|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|")
for k in sorted(G, key=lambda k: (k[0], k[1], k[2])):
    rows = G[k]
    V = collections.defaultdict(list)
    for r in rows:
        V[r["variant"]].append(r)
    seeds = set.intersection(*[set(r["seed"] for r in v) for v in V.values()])
    best = {}
    for s in seeds:
        lls = {v: float(next(r["ll"] for r in V[v] if r["seed"] == s)) for v in V}
        # MLPKriging logLikelihood is a log-likelihood too: higher is better
        m = max(lls.values())
        for v, ll in lls.items():
            best[v] = best.get(v, 0) + (abs(ll - m) <= 1e-6 * max(1.0, abs(m)))
    ref = None
    for v in ["master", "A", "B", "B1000"]:
        if v not in V:
            continue
        sel = [r for r in V[v] if r["seed"] in seeds]
        w = st.median(float(r["seconds"]) for r in sel)
        c = st.median(float(r["cpu_seconds"]) for r in sel)
        ll = st.median(float(r["ll"]) for r in sel)
        if v == "master":
            ref = w
        rel = f"x{ref / w:.2f}" if ref else ""
        print(f"| {k[0]} | {k[1]} | {k[2]} | {len(seeds)} | {v} | {w:.3f} | {c:.3f} | {c / w:.1f} | {ll:.2f} | {best.get(v, 0)} | {rel} |")
