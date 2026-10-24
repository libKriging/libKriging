"""Summary of jitter_multistart_study.cpp CSVs (jitter-ms-<kernel>-<family>-<n>-<d>.csv).
miss: BFGS LL < BFGS10 LL - 0.01 (same jitter setting); rmse: median normalized
test RMSE of the BFGS / BFGS10 fit."""
import glob, csv, statistics as st, sys
rows = []
for fn in sorted(glob.glob('jitter-ms-*.csv')):
    _, _, kernel, family, n, d = fn[:-4].split('-')
    data = {}
    for r in csv.DictReader(open(fn)):
        data[(int(r['set']), r['jitter'], r['optim'])] = (float(r['ll']), float(r['rmse']))
    sets = sorted({k[0] for k in data})
    out = [kernel, family, n, d]
    for j in ('off', 'on'):
        miss = sum(1 for s in sets if data[(s, j, 'BFGS')][0] < data[(s, j, 'BFGS10')][0] - 0.01)
        r1 = st.median(data[(s, j, 'BFGS')][1] for s in sets)
        r10 = st.median(data[(s, j, 'BFGS10')][1] for s in sets)
        out += [f'{100*miss/len(sets):.0f}%', f'{r1:.3g}', f'{r10:.3g}']
    rows.append(out)
print('| kernel | data | n | d | miss off | RMSE BFGS off | RMSE BFGS10 off | miss on | RMSE BFGS on | RMSE BFGS10 on |')
print('|---|---|---|---|---|---|---|---|---|---|')
for r in rows:
    print('| ' + ' | '.join(r) + ' |')
