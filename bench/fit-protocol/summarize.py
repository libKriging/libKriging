#!/usr/bin/env python3
"""Tables of the fit protocol (T7): miss rate with Wilson 95 % intervals, test
RMSE, wall and CPU time per configuration and variant.

    summarize.py <dir | file.csv | file.csv.gz> ... [--by kernel,noise,objective,n] [--detail] [--tol 10]

Rows of every input are merged; a data set is (kernel, noise, objective,
function, d, n, set) and its fits by the variants ("optimizer:settings") are
compared to the best of them:
  - LL / LMP: miss if the objective is more than --tol below the best one;
  - LOO (mean squared LOO error e, minimized): on the same log-density scale,
    miss if (n / 2) log(e / e_best) > --tol;
  - a fit that failed (status error) is a miss.
Columns: sets compared, misses, miss rate and Wilson 95 % interval, median
test RMSE (over sd(y_test)), median and 90 % quantile of RMSE / RMSE of the best fit of the set, median wall and CPU time per fit, total wall time. Only stdlib (no pandas).
"""
import argparse
import csv
import gzip
import io
import math
import os
import re
import sys
from collections import OrderedDict, defaultdict

KEY = ("kernel", "noise", "objective", "function", "d", "n", "set")


def inputs(paths):
    for p in paths:
        if os.path.isdir(p):
            for r, _, fs in sorted(os.walk(p)):
                for f in sorted(fs):
                    if f.endswith((".csv", ".csv.gz")):
                        yield os.path.join(r, f)
        else:
            yield p


def read(paths):
    rows = []
    for p in inputs(paths):
        op = gzip.open if p.endswith(".gz") else open
        with op(p, "rt", newline="") as fh:
            rows += list(csv.DictReader(fh))
    return rows


def wilson(k, n, z=1.959964):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def quantile(v, q):
    v = sorted(x for x in v if not math.isnan(x))
    if not v:
        return float("nan")
    h = (len(v) - 1) * q
    i = int(math.floor(h))
    return v[i] if i + 1 >= len(v) else v[i] + (h - i) * (v[i + 1] - v[i])


def median(v):
    v = sorted(x for x in v if not math.isnan(x))
    if not v:
        return float("nan")
    m = len(v) // 2
    return v[m] if len(v) % 2 else 0.5 * (v[m - 1] + v[m])


def score(r):
    """Objective on a 'larger is better' log-density scale (nan if failed)."""
    if r["status"] != "ok":
        return float("nan")
    v = float(r["objective_value"])
    if r["objective"] == "LOO":
        return -0.5 * int(r["n"]) * math.log(v) if v > 0 else float("nan")
    return v


def fmt(x, digits=3):
    if isinstance(x, float) and math.isnan(x):
        return "-"
    if isinstance(x, float):
        return ("%." + str(digits) + "g") % x
    return str(x)


def analyse(rows, tol):
    sets = defaultdict(list)
    for r in rows:
        sets[tuple(r[k] for k in KEY)].append(r)
    out = []  # one record per fit, with miss and RMSE ratio
    for key, fits in sets.items():
        scores = [score(r) for r in fits]
        ok = [s for s in scores if not math.isnan(s)]
        best = max(ok) if ok else float("nan")
        ib = scores.index(best) if ok else None
        rmse_best = float(fits[ib]["rmse"]) if ib is not None else float("nan")
        for r, s in zip(fits, scores):
            rmse = float(r["rmse"]) if r["status"] == "ok" else float("nan")
            out.append(dict(r, miss=math.isnan(s) or s < best - tol, gap=best - s,
                            rmse_ratio=rmse / rmse_best if rmse_best > 0 else float("nan"),
                            rmse_v=rmse, wall=float(r["wall_s"]), cpu=float(r["cpu_s"])))
    return out


def col_key(v):
    return (0, int(v), "") if v.isdigit() else (1, 0, v)


def variant_key(v):
    """BFGS < BFGS4 < BFGS10 (number of starts), default settings first."""
    m = re.match(r"([A-Za-z_]+)(\d*):(.*)", v)
    if not m:
        return (v, 0, 1, "")
    return (m.group(1), int(m.group(2) or 1), m.group(3) != "default", m.group(3))


def table(recs, by, title):
    groups = OrderedDict()
    for r in sorted(recs, key=lambda r: ([col_key(r[k]) for k in by], variant_key(r["variant"]))):
        groups.setdefault(tuple(r[k] for k in by), OrderedDict()).setdefault(r["variant"], []).append(r)
    lines = ["### " + title, "",
             "| " + " | ".join(by) + " | variant | sets | miss | miss % [95 % CI] | RMSE | RMSE / best, median / p90 | wall s | CPU s | wall total s |",
             "|" + "---|" * (len(by) + 9)]
    for g, variants in groups.items():
        for v, rs in variants.items():
            n, k = len(rs), sum(r["miss"] for r in rs)
            lo, hi = wilson(k, n)
            err = sum(r["status"] != "ok" for r in rs)
            lines.append("| " + " | ".join(g) + " | %s | %d | %d%s | %.0f [%.0f, %.0f] | %s | %s | %s | %s | %s |" % (
                v, n, k, " (%d err)" % err if err else "", 100 * k / n, 100 * lo, 100 * hi,
                fmt(median(r["rmse_v"] for r in rs)), "%s / %s" % (fmt(median(r["rmse_ratio"] for r in rs)), fmt(quantile([r["rmse_ratio"] for r in rs], 0.9))),
                fmt(median(r["wall"] for r in rs)), fmt(median(r["cpu"] for r in rs)),
                fmt(sum(r["wall"] for r in rs if not math.isnan(r["wall"])))))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--by", default="kernel,noise,objective,n", help="grouping columns of the main table")
    ap.add_argument("--detail", action="store_true", help="add the table per configuration (with function)")
    ap.add_argument("--tol", type=float, default=10.0, help="miss threshold (log-density units)")
    a = ap.parse_args()
    rows = read(a.paths)
    if not rows:
        raise SystemExit("no rows")
    recs = analyse(rows, a.tol)
    nsets = len({tuple(r[k] for k in KEY) for r in rows})
    out = io.StringIO()
    out.write("## Fit protocol summary\n\n%d fits, %d data sets, %d variants; miss: objective more than %g "
              "(log-density units) below the best variant of the data set.\n\n"
              % (len(rows), nsets, len({r["variant"] for r in rows}), a.tol))
    out.write(table(recs, a.by.split(","), "By " + ", ".join(a.by.split(","))) + "\n")
    out.write(table(recs, ["noise", "objective", "function"], "By noise, objective, function (kernels and n pooled)") + "\n")
    if a.detail:
        out.write(table(recs, ["kernel", "noise", "objective", "function", "n"], "Per configuration") + "\n")
    sys.stdout.write(out.getvalue())


if __name__ == "__main__":
    main()
