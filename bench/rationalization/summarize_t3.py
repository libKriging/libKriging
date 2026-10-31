"""Summary of the T3 study CSVs (run_t3.sh, t3_multistart_study.cpp):
   summarize_t3.py <out dir> [pass prefix, default A] [threshold, default 10]

miss: the optimized objective of a variant more than <threshold> below the best
of the variants on the same data set (LL, LMP: log densities). LOO
(leaveOneOut(), a mean squared error, minimized) is put on the same scale as a
Gaussian log density of the LOO residuals: miss when n/2 log(LOO / LOO_best)
exceeds the threshold. A failed fit (nan) counts as a miss.
Intervals: Wilson 95 %. RMSE ratio: test RMSE / test RMSE of the best variant of
the set, median and 90 % quantile. Times: median per fit, and the ratio of the
summed CPU times to the single start's (BFGS)."""
import csv, glob, math, statistics as st, sys
from collections import defaultdict

out = sys.argv[1]
prefix = sys.argv[2] if len(sys.argv) > 2 else 'A'
thr = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
VARIANTS = ['BFGS', 'BFGS-nolr', 'BFGS4', 'BFGS10']


def wilson(k, n, z=1.96):
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def fmt_rate(k, n):
    lo, hi = wilson(k, n)
    return f'{100 * k / n:.0f} % [{100 * lo:.0f}, {100 * hi:.0f}]' if n else '-'


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


# sets[(family, kernel, noise, objective, n, d, pct)][set][variant] = row
sets = defaultdict(lambda: defaultdict(dict))
for fn in sorted(glob.glob(f'{out}/{prefix}-*.csv')):
    for r in csv.DictReader(open(fn)):
        key = (r['family'], r['kernel'], r['noise'], r['objective'], int(r['n']), int(r['d']), r['pct'])
        sets[key][int(r['set'])][r['variant']] = r


def score(r, objective, n):
    """Objective on a 'higher is better' log-density scale (nan: failure)."""
    o = float(r['obj'])
    if math.isnan(o):
        return float('nan')
    if objective == 'LOO':
        return -n / 2 * math.log(max(o, 1e-300))
    return o


def analyse(key):
    fam, ker, nz, obj, n, d, pct = key
    res = {v: dict(miss=0, tot=0, rr=[], rmse=[], cpu=[], wall=[], fail=0) for v in VARIANTS}
    pairs = defaultdict(int)  # (variant a misses, variant b misses) for BFGS vs BFGS4 / BFGS-nolr
    ties = 0
    for s, vr in sets[key].items():
        sc = {v: score(r, obj, n) for v, r in vr.items()}
        good = [x for x in sc.values() if not math.isnan(x)]
        if not good:
            continue
        best = max(good)
        bestv = max((v for v in sc if not math.isnan(sc[v])), key=lambda v: sc[v])
        rbest = float(vr[bestv]['rmse'])
        m = {}
        for v, r in vr.items():
            a = res[v]
            a['tot'] += 1
            m[v] = math.isnan(sc[v]) or sc[v] < best - thr
            a['miss'] += m[v]
            if math.isnan(sc[v]):
                a['fail'] += 1
                continue
            a['rmse'].append(float(r['rmse']))
            a['rr'].append(float(r['rmse']) / rbest if rbest > 0 else 1.0)
            a['cpu'].append(float(r['cpu_ms']))
            a['wall'].append(float(r['wall_ms']))
        for b in ('BFGS4', 'BFGS-nolr', 'BFGS10'):
            if 'BFGS' in m and b in m:
                pairs[(b, m['BFGS'], m[b])] += 1
        if 'BFGS' in vr and 'BFGS-nolr' in vr and vr['BFGS']['obj'] == vr['BFGS-nolr']['obj']:
            ties += 1
    return res, pairs, ties


def main():
    keys = sorted(sets, key=lambda k: (k[3], k[2], k[0], k[1], k[4]))
    print(f'# T3 study, pass {prefix}, miss threshold {thr:g}\n')
    print('| objective | noise | data | kernel | n | sets | ' + ' | '.join(f'{v} miss' for v in VARIANTS)
          + ' | RMSE ratio BFGS / BFGS4 / BFGS10 (median, q90) | CPU s BFGS / nolr / 4 / 10 (median) |'
          + ' nolr == BFGS |')
    print('|' + '---|' * (6 + len(VARIANTS) + 3))
    pooled = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    cpu_sum = defaultdict(lambda: defaultdict(float))
    pair_tot = defaultdict(lambda: defaultdict(int))
    for key in keys:
        fam, ker, nz, obj, n, d, pct = key
        res, pairs, ties = analyse(key)
        cells = []
        for v in VARIANTS:
            a = res[v]
            cells.append(fmt_rate(a['miss'], a['tot']) if a['tot'] else '-')
            g = (obj, nz)
            pooled[g][v][0] += a['miss']
            pooled[g][v][1] += a['tot']
            pooled[('all', 'all')][v][0] += a['miss']
            pooled[('all', 'all')][v][1] += a['tot']
            if a['tot']:
                cpu_sum[(obj, nz, n)][v] += sum(a['cpu'])
        for pk, c in pairs.items():
            pair_tot[(obj, nz)][pk] += c
        rr = ' / '.join(f"{st.median(res[v]['rr']):.3g}, {q(res[v]['rr'], 0.9):.3g}" if res[v]['rr'] else '-'
                        for v in ('BFGS', 'BFGS4', 'BFGS10'))
        cpu = ' / '.join(f"{st.median(res[v]['cpu']) / 1e3:.3g}" if res[v]['cpu'] else '-' for v in VARIANTS)
        tot = res['BFGS']['tot']
        tie = f'{ties} / {tot}' if res['BFGS-nolr']['tot'] else '-'
        print(f'| {obj} | {nz} | {fam} d{d} | {ker} | {n} | {tot} | ' + ' | '.join(cells) + f' | {rr} | {cpu} | {tie} |')

    print('\n## Pooled miss rates (Wilson 95 %)\n')
    print('| objective | noise | ' + ' | '.join(VARIANTS) + ' |')
    print('|' + '---|' * (2 + len(VARIANTS)))
    for g in sorted(pooled):
        print(f'| {g[0]} | {g[1]} | ' + ' | '.join(
            f'{pooled[g][v][0]} / {pooled[g][v][1]} = {fmt_rate(*pooled[g][v])}' if pooled[g][v][1] else '-'
            for v in VARIANTS) + ' |')

    print('\n## Paired: BFGS against X (sets where only one of the two misses)\n')
    print('| objective | noise | X | only BFGS misses | only X misses | both | neither |')
    print('|---|---|---|---|---|---|---|')
    for g in sorted(pair_tot):
        for b in ('BFGS-nolr', 'BFGS4', 'BFGS10'):
            p = pair_tot[g]
            if sum(c for k, c in p.items() if k[0] == b) == 0:
                continue
            print(f'| {g[0]} | {g[1]} | {b} | {p[(b, True, False)]} | {p[(b, False, True)]} | '
                  f'{p[(b, True, True)]} | {p[(b, False, False)]} |')

    print('\n## Summed CPU time relative to BFGS (all data sets of the (objective, noise, n) cell)\n')
    print('| objective | noise | n | ' + ' | '.join(VARIANTS) + ' |')
    print('|' + '---|' * (3 + len(VARIANTS)))
    for c in sorted(cpu_sum):
        b = cpu_sum[c]['BFGS']
        print(f'| {c[0]} | {c[1]} | {c[2]} | ' + ' | '.join(
            f'{cpu_sum[c][v] / b:.2f}' if cpu_sum[c][v] else '-' for v in VARIANTS) + f' (BFGS {b / 1e3:.0f} s) |')


main()
