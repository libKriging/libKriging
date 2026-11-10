"""Wall time of the T3 timing pass (run_t3.sh B): per (objective, noise, n) and
variant, summed wall time of the data sets run at both budgets, 1 core (B25)
against the whole machine (B100), and the same for CPU time; then per data
family at n 1000 (whole machine).
   threads_t3.py <out dir | results/t3-B.csv.gz>"""
import sys
import t3_rows
from collections import defaultdict
out = sys.argv[1]
V = ['BFGS', 'BFGS-nolr', 'BFGS4', 'BFGS10']
rows = {}
for p, r in t3_rows.rows(out):
    if p.startswith('B'):
        rows[(p, r['family'], r['kernel'], r['noise'], r['objective'], r['n'], r['set'], r['variant'])] = r
both = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0, 0.0]))
for k, r in rows.items():
    if k[0] != 'B25':
        continue
    r4 = rows.get(('B100',) + k[1:])
    if r4 is None:
        continue
    a = both[(k[4], k[3], k[5])][k[7]]
    a[0] += float(r['wall_ms']); a[1] += float(r4['wall_ms']); a[2] += float(r['cpu_ms']); a[3] += float(r4['cpu_ms'])
print('| objective | noise | n | variant | wall 1 core | wall 4 cores | speed-up | CPU 4 cores / CPU 1 core |')
print('|---|---|---|---|---|---|---|---|')
for c in sorted(both):
    for v in V:
        a = both[c][v]
        if a[0]:
            print(f'| {c[0]} | {c[1]} | {c[2]} | {v} | {a[0] / 1e3:.1f} s | {a[1] / 1e3:.1f} s | {a[0] / a[1]:.2f} | {a[3] / a[2]:.2f} |')
fam = defaultdict(lambda: defaultdict(float))
for k, r in rows.items():
    if k[0] == 'B100' and k[5] == '1000':
        fam[(k[1], k[2])][k[7]] += float(r['wall_ms'])
print('\n| n 1000, 4 cores | ' + ' | '.join(V) + ' |')
print('|---|' + '---|' * len(V))
for f in sorted(fam):
    print(f'| {f[0]} {f[1]} | ' + ' | '.join(f'{fam[f][v] / 1e3:.1f} s' for v in V) + ' |')
