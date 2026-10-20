"""Summary of jitter_split_study.cpp CSVs (jitter-split-<kernel>-<family>-<n>-<d>-<optim>.csv):
median normalized test RMSE per variant (pct = 100) and % of data sets whose
fit diverges across thread budgets (relative spread of the first theta
> 1e-3 over pct 10 / 100 / 200: the optimizer took another path)."""
import glob, csv, statistics as st
V = ['off', 'on', 'on>0', 'on>10', 'on~0']
print('| kernel | data | n | d | optim | ' + ' | '.join(f'RMSE {v}' for v in V) + ' | ' + ' | '.join(f'div {v}' for v in V) + ' |')
print('|' + '---|' * (5 + 2 * len(V)))
for fn in sorted(glob.glob('jitter-split-*.csv')):
    _, _, k, fam, n, d, o = fn[:-4].split('-')
    D = {}
    for r in csv.DictReader(open(fn)):
        if r['variant'] == 'error':
            continue
        D.setdefault((r['variant'], int(r['set'])), {})[float(r['pct'])] = (float(r['theta0']), float(r['rmse']), float(r['pred0']))
    sets = sorted({s for (_, s) in D})
    rm, dv = [], []
    for v in V:
        rm.append(st.median(D[(v, s)][100.0][1] for s in sets if (v, s) in D))
        div = 0
        for s in sets:
            x = D.get((v, s), {})
            if len(x) < 3:
                continue
            th = [x[p][0] for p in x]
            if (max(th) - min(th)) > 1e-3 * abs(max(th)):
                div += 1
        dv.append(100 * div / len(sets))
    print(f'| {k} | {fam} | {n} | {d} | {o} | ' + ' | '.join(f'{x:.3g}' for x in rm) + ' | ' + ' | '.join(f'{x:.0f}%' for x in dv) + ' |')
