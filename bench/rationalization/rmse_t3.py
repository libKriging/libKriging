"""Paired test RMSE of the T3 study (run_t3.sh): per (objective, noise) and
variant, against BFGS10 on the same data set: geometric mean of RMSE / RMSE_BFGS10,
share of sets more than 10 % worse / better than BFGS10.
   rmse_t3.py <out dir> [pass prefix, default A]"""
import csv, glob, math, sys
from collections import defaultdict
out, prefix = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else 'A')
sets = defaultdict(dict)
for fn in glob.glob(f'{out}/{prefix}-*.csv'):
    for r in csv.DictReader(open(fn)):
        sets[(r['objective'], r['noise'], r['family'], r['kernel'], r['n'], r['set'])][r['variant']] = float(r['rmse'])
acc = defaultdict(lambda: [0, 0.0, 0, 0])
for k, v in sets.items():
    if 'BFGS10' not in v or not v['BFGS10'] > 0:
        continue
    for var, x in v.items():
        if math.isnan(x):
            continue
        a = acc[(k[0], k[1], var)]
        r = x / v['BFGS10']
        a[0] += 1; a[1] += math.log(r); a[2] += r > 1.1; a[3] += r < 1 / 1.1
print('| objective | noise | variant | sets | RMSE / RMSE(BFGS10), geometric mean | > 10 % worse | > 10 % better |')
print('|---|---|---|---|---|---|---|')
for (o, nz, var), a in sorted(acc.items()):
    if var == 'BFGS10':
        continue
    print(f'| {o} | {nz} | {var} | {a[0]} | {math.exp(a[1] / a[0]):.3f} | {100 * a[2] / a[0]:.0f} % | {100 * a[3] / a[0]:.0f} % |')
