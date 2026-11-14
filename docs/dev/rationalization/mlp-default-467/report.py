"""Tables of #467 from the outputs of run.sh. Usage: python3 report.py <out dir>"""
import glob
import os
import sys

import numpy as np

d = sys.argv[1]
COLS = "name rep optim folds act wall cpu vs ll q2 nlpd cov nlpd0 cov0".split()
ORDER = ["BFGS:3", "BFGS:5", "BFGS2:3", "BFGS2:5", "BFGS4:3", "BFGS4:5", "BFGS10:5"]
rng = np.random.default_rng(0)


def load(pattern):
    rows, fails = [], 0
    for f in sorted(glob.glob(os.path.join(d, pattern))):
        for line in open(f):
            t = line.split()
            if t[0] == "FAIL":
                fails += 1
            elif t[0] == "FIT":
                r = dict(zip(COLS, t[1:]))
                for k in COLS[5:]:
                    r[k] = float(r[k])
                r["v"] = f"{r['optim']}:{r['folds']}"
                rows.append(r)
    return rows, fails


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * p, 100 * (c - h), 100 * (c + h)


def key(r):
    return (r["name"], r["rep"], r["act"])


def bmed(x, n=4000):
    """median and bootstrap 95 % CI of the median"""
    x = np.asarray(x)
    m = np.median(rng.choice(x, (n, len(x))), axis=1)
    return np.median(x), np.quantile(m, 0.025), np.quantile(m, 0.975)


def quality(rows, title, ref="BFGS:3", kr=None):
    by = {}
    for r in rows:
        by.setdefault(r["v"], {})[key(r)] = r
    common = set.intersection(*[set(v) for v in by.values()])
    krk = {(r["name"], r["rep"]): r for r in (kr or [])}
    print(f"\n### {title} ({len(common)} cases per variant)\n")
    print("| optim:K | median Q² | Q² < 0.5, % [95 % CI] | Q² < Kriging − 0.05 | median NLPD | "
          f"ΔNLPD vs {ref}, median [95 % CI] | NLPD > 10 | mean cov. | cov. < 0.8 | 1-core CPU, sum | x {ref} |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    base = by[ref]
    cpu0 = sum(base[k]["cpu"] for k in common)
    if krk:
        ks = [(k[0], k[1]) for k in common]
        q2 = np.array([krk[k]["q2"] for k in ks])
        nl = np.array([krk[k]["nlpd"] for k in ks])
        cov = np.array([krk[k]["cov"] for k in ks])
        f5 = wilson(int((q2 < 0.5).sum()), len(q2))
        print(f"| `Kriging` (BFGS4) | {np.median(q2):.3f} | {f5[0]:.1f} [{f5[1]:.1f}, {f5[2]:.1f}] | – | {np.median(nl):.2f} | – | "
              f"{(nl > 10).sum()} | {cov.mean():.3f} | {(cov < 0.8).sum()} | {sum(krk[k]['cpu'] for k in ks):.0f} s | "
              f"{sum(krk[k]['cpu'] for k in ks) / cpu0:.2f} |")
    for v in ORDER:
        if v not in by:
            continue
        b = by[v]
        q2 = np.array([b[k]["q2"] for k in common])
        dn = [b[k]["nlpd"] - base[k]["nlpd"] for k in common]
        nl = np.array([b[k]["nlpd"] for k in common])
        cov = np.array([b[k]["cov"] for k in common])
        cpu = sum(b[k]["cpu"] for k in common)
        f5 = wilson(int((q2 < 0.5).sum()), len(q2))
        m, lo, hi = bmed(dn)
        worse = "–"
        if krk:
            w = np.array([b[k]["q2"] < krk[(k[0], k[1])]["q2"] - 0.05 for k in common])
            worse = f"{100 * w.mean():.0f} %"
        print(f"| {v} | {np.median(q2):.3f} | {f5[0]:.1f} [{f5[1]:.1f}, {f5[2]:.1f}] | {worse} | {np.median(nl):.2f} | "
              f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}] | {(nl > 10).sum()} | {cov.mean():.3f} | {(cov < 0.8).sum()} | "
              f"{cpu:.0f} s | {cpu / cpu0:.2f} |")
    return by, common


def per_function(by, common, kr, vs=("BFGS:3", "BFGS2:3", "BFGS4:3", "BFGS4:5", "BFGS10:5")):
    names = sorted({k[0] for k in common})
    krk = {(r["name"], r["rep"]): r for r in kr}
    print("\nMedian Q² / mean coverage per function:\n")
    print("| function | Kriging | " + " | ".join(vs) + " |")
    print("|---" * (len(vs) + 2) + "|")
    for n in names:
        ks = [k for k in common if k[0] == n]
        kq = np.median([krk[(k[0], k[1])]["q2"] for k in ks]) if krk else np.nan
        kc = np.mean([krk[(k[0], k[1])]["cov"] for k in ks]) if krk else np.nan
        cells = [f"{kq:.3f} / {kc:.2f}"]
        for v in vs:
            q = np.median([by[v][k]["q2"] for k in ks])
            c = np.mean([by[v][k]["cov"] for k in ks])
            cells.append(f"{q:.3f} / {c:.2f}")
        print(f"| {n} | " + " | ".join(cells) + " |")


def folds(by, common):
    print("\nK = 5 against K = 3 (paired, same optimizer):\n")
    print("| optim | ΔNLPD K5 − K3, median [95 % CI] | share K5 better | cov. K3 / K5 | cases cov. < 0.8 K3 / K5 | "
          "|cov − 0.95| K3 / K5 | CPU K5 / K3 |")
    print("|---|---|---|---|---|---|---|")
    for o in ["BFGS", "BFGS2", "BFGS4"]:
        a, b = by.get(o + ":3"), by.get(o + ":5")
        if not a or not b:
            continue
        dn = [b[k]["nlpd"] - a[k]["nlpd"] for k in common]
        m, lo, hi = bmed(dn)
        ca = np.array([a[k]["cov"] for k in common])
        cb = np.array([b[k]["cov"] for k in common])
        cpu = sum(b[k]["cpu"] for k in common) / sum(a[k]["cpu"] for k in common)
        print(f"| {o} | {m:+.3f} [{lo:+.3f}, {hi:+.3f}] | {100 * np.mean(np.array(dn) < 0):.0f} % | {ca.mean():.3f} / {cb.mean():.3f} | "
              f"{(ca < 0.8).sum()} / {(cb < 0.8).sum()} | {np.abs(ca - 0.95).mean():.3f} / {np.abs(cb - 0.95).mean():.3f} | "
              f"{cpu:.2f} |")


rowsA, failsA = load("A_*.txt")
kr, _ = load("K_*.txt")
print(f"Pass A: {len(rowsA)} fits, {failsA} failed")
for act in ["tanh", "selu"]:
    rows = [r for r in rowsA if r["act"] == act]
    by, common = quality(rows, f"15 functions, {act}, 1 core", kr=kr)
    per_function(by, common, kr)
    folds(by, common)
by, common = quality(rowsA, "15 functions, tanh + selu pooled, 1 core", kr=kr)
folds(by, common)

# pass B: wall 1 core vs 4 cores
r1, f1 = load("B1_*.txt")
r4, f4 = load("B4_*.txt")
if r1 and r4:
    print(f"\n### Wall / CPU time, 1 core vs 4 cores (tanh, designs 0-2, 15 functions; {f1 + f4} failed)\n")
    print("| optim:K | 1 core wall | 4 cores wall | speed-up | 4 cores CPU | wall x BFGS:3 (1 core / 4 cores) |")
    print("|---|---|---|---|---|---|")
    w1 = {v: sum(r["wall"] for r in r1 if r["v"] == v) for v in ORDER}
    w4 = {v: sum(r["wall"] for r in r4 if r["v"] == v) for v in ORDER}
    c4 = {v: sum(r["cpu"] for r in r4 if r["v"] == v) for v in ORDER}
    for v in ORDER:
        if w1[v] > 0:
            print(f"| {v} | {w1[v]:.1f} s | {w4[v]:.1f} s | {w1[v] / w4[v]:.2f} | {c4[v]:.1f} s | "
                  f"{w1[v] / w1['BFGS:3']:.2f} / {w4[v] / w4['BFGS:3']:.2f} |")

# pass C: #407 Python case
import csv

for tag, pat in [("1 core", "C1_*.csv"), ("4 cores", "C4.csv")]:
    rows = []
    for f in sorted(glob.glob(os.path.join(d, pat))):
        rows += list(csv.DictReader(open(f)))
    if not rows:
        continue
    nd = len({r["design"] for r in rows})
    print(f"\n### #407 case (projected Branin 10D, n 120, [16, 8]), {tag}, {nd} designs: median [min, max]\n")
    print("| activation | optim:K | Q² | NLPD | cov. 95 % | wall | CPU |")
    print("|---|---|---|---|---|---|---|")
    if tag == "1 core":
        rows += list(csv.DictReader(open(os.path.join(d, "CK.csv")))) if os.path.exists(os.path.join(d, "CK.csv")) else []
    for act in ["none", "tanh", "selu"]:
        for v in (["Kriging:0"] if act == "none" else ORDER):
            o, k = v.split(":")
            s = [r for r in rows if r["activation"] == act and r["optim"] == o and r["folds"] == k]
            if not s:
                continue

            def mmm(c, fmt):
                x = np.array([float(r[c]) for r in s])
                return f"{np.median(x):{fmt}} [{x.min():{fmt}}, {x.max():{fmt}}]"

            print(f"| {act if act != 'none' else '`Kriging` (BFGS4)'} | {v if act != 'none' else '–'} | {mmm('Q2', '.3f')} | {mmm('NLPD', '.2f')} | {mmm('cov95', '.3f')} | "
                  f"{mmm('wall', '.2f')} | {mmm('cpu', '.2f')} |")
