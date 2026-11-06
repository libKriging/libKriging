#!/usr/bin/env python3
"""Phase 2 summary: per (op, d, n), median over repetitions of the time per call
for each OpenBLAS thread count, speed-up vs 1 thread and CPU / wall ratio."""
import collections, csv, statistics as st, sys

path = sys.argv[1] if len(sys.argv) > 1 else "perf-blas/phase2-microbench.csv"
D = collections.defaultdict(list)
for r in csv.DictReader(open(path)):
    if r["ms_per_call"] in ("", "FAIL"):
        continue
    D[(r["op"], int(r["d"]), int(r["n"]), int(r["blas_threads"]))].append(
        (float(r["ms_per_call"]), float(r["cpu_ms_per_call"])))
keys = sorted({k[:3] for k in D})
threads = sorted({k[3] for k in D})
print("| op | d | n | reps | " + " | ".join(f"{t} thr: ms (x vs 1, CPU/wall)" for t in threads) + " | best |")
print("|---|---:|---:|---:|" + "---:|" * len(threads) + "---:|")
for op, d, n in keys:
    if (op, d, n, 1) not in D:
        continue
    w1 = st.median(w for w, _ in D[(op, d, n, 1)])
    cells, best = [], (w1, 1)
    for t in threads:
        v = D.get((op, d, n, t))
        if not v:
            cells.append("")
            continue
        w = st.median(x for x, _ in v)
        c = st.median(c / x for x, c in v)
        best = min(best, (w, t))
        cells.append(f"{w:.2f} (x{w1 / w:.2f}, {c:.1f})")
    print(f"| {op} | {d} | {n} | {len(D[(op, d, n, 1)])} | " + " | ".join(cells) + f" | {best[1]} |")
