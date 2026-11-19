#!/usr/bin/env python3
"""Long-range restart study (#466): paired comparisons of two variants on the same data sets
(restart on / off, BFGS + restart against BFGS2 / BFGS4) on the fit-protocol results.

    long_range_466.py results/long-range-466.csv.gz [--by noise,objective,n] [--tol 10]

Labels: optimizer, ":off" for a run with LK_LONG_RANGE_RESTART=0. Where the restart was off by
construction (noise-free LOO / LMP, noise-free multistarts; bit-identical, checked on a smoke run)
the ":off" fit is the default one. Miss: as summarize.py (best of all the variants of the set).
Columns: misses with Wilson 95 % intervals, discordant sets (only A misses : only B misses) with
an exact two-sided sign test, wall and CPU time ratios (sums and median per set), test RMSE ratio
B / A (geometric mean, share of sets > 10 % worse). The restart was removed in 2.0 (#466): the
script is kept to reproduce docs/dev/rationalization/long-range-restart-466.md."""
import sys, math, argparse, os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summarize as S

ap = argparse.ArgumentParser()
ap.add_argument("inp", nargs="+")
ap.add_argument("--by", default="noise,objective,n")
ap.add_argument("--tol", type=float, default=10)
a = ap.parse_args()
rows = S.read(a.inp)
OFF = "LK_LONG_RANGE_RESTART=0"
def label(r):
    return r["optim"] + (":off" if OFF in r["variant"] else "")
sets = defaultdict(dict)
for r in rows:
    k = tuple(r[c] for c in S.KEY)
    sets[k][label(r)] = r
# for noise-free LOO/LMP and noise-free multistarts the restart is off by construction
# (bit-identical, checked on the smoke run): the off variant = the on variant there
for k, v in sets.items():
    for o in ("BFGS", "BFGS2", "BFGS4"):
        if o + ":off" not in v and o in v and (k[1] == "none" and (k[2] != "LL" or o != "BFGS")):
            v[o + ":off"] = v[o]
def sc(r): return S.score(r)
best = {k: max(sc(r) for r in v.values() if not math.isnan(sc(r))) for k, v in sets.items()}
def miss(k, r):
    s = sc(r); return math.isnan(s) or s < best[k] - a.tol
bykeys = a.by.split(",") if a.by else []
idx = {c: i for i, c in enumerate(S.KEY)}
def grp(k): return tuple(k[idx[c]] for c in bykeys)
def binom_two_sided(x, n):
    if n == 0: return 1.0
    k = min(x, n - x)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)
def pct(k, n):
    lo, hi = S.wilson(k, n); return "%.0f [%.0f, %.0f]" % (100 * k / n, 100 * lo, 100 * hi)
def gm(v):
    v = [x for x in v if x > 0 and not math.isnan(x)]
    return math.exp(sum(map(math.log, v)) / len(v)) if v else float("nan")

def paired(A, B, title, only_active=False):
    """A, B: labels; rows: per group, miss A / B, discordant, p, time ratios, RMSE ratio B/A"""
    print("\n### " + title + "\n")
    print("| " + " | ".join(bykeys) + " | sets | identical | miss %s | miss %s | only %s : only %s | p (sign) | wall %s / %s (sum) | CPU (sum) | wall median ratio | RMSE %s / %s geo. mean | > 10 %% worse %s / %s |" % (A, B, A, B, A, B, B, A, B, A))
    print("|" + "---|" * (len(bykeys) + 11))
    G = defaultdict(list)
    for k, v in sets.items():
        if A in v and B in v: G[grp(k)].append(k)
    tot = defaultdict(float)
    for g in sorted(G, key=lambda g: tuple(int(x) if x.isdigit() else x for x in g)):
        ks = G[g]; n = len(ks)
        ma = sum(miss(k, sets[k][A]) for k in ks); mb = sum(miss(k, sets[k][B]) for k in ks)
        oa = sum(miss(k, sets[k][A]) and not miss(k, sets[k][B]) for k in ks)
        ob = sum(miss(k, sets[k][B]) and not miss(k, sets[k][A]) for k in ks)
        ident = sum(sets[k][A]["objective_value"] == sets[k][B]["objective_value"] for k in ks)
        wa = sum(float(sets[k][A]["wall_s"]) for k in ks); wb = sum(float(sets[k][B]["wall_s"]) for k in ks)
        ca = sum(float(sets[k][A]["cpu_s"]) for k in ks); cb = sum(float(sets[k][B]["cpu_s"]) for k in ks)
        wr = S.median([float(sets[k][A]["wall_s"]) / float(sets[k][B]["wall_s"]) for k in ks])
        rr = [float(sets[k][B]["rmse"]) / float(sets[k][A]["rmse"]) for k in ks]
        worse = sum(x > 1.1 for x in rr)
        for kk, vv in (("n", n), ("ma", ma), ("mb", mb), ("oa", oa), ("ob", ob), ("wa", wa), ("wb", wb), ("ca", ca), ("cb", cb)): tot[kk] += vv
        print("| " + " | ".join(g) + " | %d | %d | %s | %s | %d : %d | %.2g | %.2f | %.2f | %.2f | %.3f | %d %% |" % (
            n, ident, pct(ma, n), pct(mb, n), oa, ob, binom_two_sided(oa, oa + ob), wa / wb, ca / cb, wr, gm(rr), round(100 * worse / n)))
    n = int(tot["n"])
    if n:
        print("| " + " | ".join(["all"] * len(bykeys)) + " | %d | | %s | %s | %d : %d | %.2g | %.2f | %.2f | | | |" % (
            n, pct(int(tot["ma"]), n), pct(int(tot["mb"]), n), tot["oa"], tot["ob"],
            binom_two_sided(int(tot["oa"]), int(tot["oa"] + tot["ob"])), tot["wa"] / tot["wb"], tot["ca"] / tot["cb"]))

pairs = [("BFGS", "BFGS:off", "BFGS: restart on vs off"),
         ("BFGS4", "BFGS4:off", "BFGS4 (default): restart on vs off (nugget: the warm start)"),
         ("BFGS2", "BFGS2:off", "BFGS2: restart on vs off (nugget: the warm start)"),
         ("BFGS", "BFGS2", "BFGS + restart vs BFGS2 (equal-time candidate)"),
         ("BFGS:off", "BFGS2", "BFGS without restart vs BFGS2"),
         ("BFGS", "BFGS4", "BFGS + restart vs BFGS4"),
         ("BFGS2:off", "BFGS4:off", "BFGS2 vs BFGS4 without the restart")]
for A, B, t in pairs:
    paired(A, B, t)
