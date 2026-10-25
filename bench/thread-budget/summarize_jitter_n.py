"""Summary of jitter_n_study.cpp CSVs (run_jitter_n.sh: jitter-n-<kernel>-<family>-<n>-<d>.csv).
Each rule J(n) picks, per n, one jitter of the CSV:
  off 0; c10 1e-10; c9 1e-9 (current gauss); c8 1e-8;
  lin 1e-9 n/100; sqrt 1e-9 sqrt(n/100); quad 1e-9 (n/100)^2.
Per configuration and rule:
  div   % of data sets whose theta differs by > 1e-3 relative (any component)
        across the thread budgets (pct 10 / 100 / 200);
  mat   % of data sets with a material divergence: the LL differs by > 1 or the
        test RMSE by > 10 % relative across the budgets (else the paths end at
        equivalent points of a flat objective);
  rmse  median normalized test RMSE (pct 100);
  r/c9  median per-set ratio of that RMSE to the c9 one;
  plat  % of fits (pct 100) with a theta component at the upper bound region
        (>= 0.99 max over the configuration), i.e. stuck on the long-range plateau.
Then the totals per rule and n, and the conditioning at the c9 optimum
(median log10 lambda_max, % with lambda_min <= 1e-12 lambda_max).
Usage: summarize_jitter_n.py [dir] [kernel]"""
import csv, glob, math, os, statistics as st, sys

d = sys.argv[1] if len(sys.argv) > 1 else '.'
kernel = sys.argv[2] if len(sys.argv) > 2 else 'gauss'
RULES = {
    'off': lambda n: 0.0, 'c10': lambda n: 1e-10, 'c9': lambda n: 1e-9, 'c8': lambda n: 1e-8,
    'lin': lambda n: 1e-9 * n / 100, 'sqrt': lambda n: 1e-9 * math.sqrt(n / 100),
    'quad': lambda n: 1e-9 * (n / 100) ** 2,
}


def load(fn):
    D = {}
    for r in csv.DictReader(open(fn)):
        if r['theta'] == 'nan':
            continue
        th = [float(x) for x in r['theta'].split(';')]
        D.setdefault((int(r['set']), float(r['J'])), {})[float(r['pct'])] = dict(
            th=th, ll=float(r['ll']), rmse=float(r['rmse']), ms=float(r['ms']), lmax=float(r['lmax']), lmin=float(r['lmin']))
    return D


def pick(Js, v):
    for J in Js:
        if (J == 0 and v == 0) or (v > 0 and abs(J - v) <= 1e-4 * v):
            return J
    return None


rows, tot = [], {}
for fn in sorted(glob.glob(os.path.join(d, f'jitter-n-{kernel}-*.csv'))):
    fam, n, dim = os.path.basename(fn)[len(f'jitter-n-{kernel}-'):-4].split('-')
    n, dim = int(n), int(dim)
    D = load(fn)
    sets = sorted({s for (s, _) in D})
    Js = sorted({J for (_, J) in D})
    thmax = max(max(x['th']) for v in D.values() for x in v.values())
    res = {}
    for name, rule in RULES.items():
        J = pick(Js, rule(n))
        if J is None:
            continue
        div = mat = cnt = plat = 0
        rm = {}
        for s in sets:
            x = D.get((s, J), {})
            if len(x) < 3:
                continue
            cnt += 1
            th = [x[p]['th'] for p in x]
            div += any(max(c) - min(c) > 1e-3 * abs(max(c)) for c in zip(*th))
            ll = [x[p]['ll'] for p in x]
            rr = [x[p]['rmse'] for p in x]
            mat += (max(ll) - min(ll) > 1) or (max(rr) - min(rr) > 0.1 * max(rr))
            rm[s] = x[100.0]['rmse']
            plat += max(x[100.0]['th']) >= 0.99 * thmax
        res[name] = (J, 100 * div / max(cnt, 1), rm, 100 * plat / max(cnt, 1), cnt, 100 * mat / max(cnt, 1))
        t = tot.setdefault((name, n), [0, 0, 0])
        t[0] += div
        t[1] += cnt
        t[2] += mat
    c9 = res['c9'][2]
    lmx = [D[(s, 1e-9)][100.0]['lmax'] for s in sets if (s, 1e-9) in D and 100.0 in D[(s, 1e-9)]]
    lmn = [D[(s, 1e-9)][100.0]['lmin'] for s in sets if (s, 1e-9) in D and 100.0 in D[(s, 1e-9)]]
    cond = None
    if lmx and not math.isnan(lmx[0]):
        cond = (st.median(math.log10(a) for a in lmx), 100 * sum(b <= 1e-12 * a for a, b in zip(lmx, lmn)) / len(lmx))
    rows.append((fam, dim, n, len(sets), res, c9, cond))

rows.sort(key=lambda r: (r[0], r[1], r[2]))
names = list(RULES)
print('## div / mat (% of sets, theta / LL or RMSE differ across budgets) / median RMSE (pct 100)\n')
print('| data | d | n | sets | ' + ' | '.join(names) + ' | lg lmax(c9) | singular(c9) |')
print('|---|---|---|---|' + '---|' * len(names) + '---|---|')
for fam, dim, n, ns, res, c9, cond in rows:
    cells = []
    for k in names:
        if k not in res:
            cells.append('')
            continue
        J, dv, rm, pl, _, mt = res[k]
        cells.append(f'{dv:.0f} / {mt:.0f}% / {st.median(rm.values()):.3g}')
    cs = f'{cond[0]:.2f} | {cond[1]:.0f}%' if cond else ' | '
    print(f'| {fam} | {dim} | {n} | {ns} | ' + ' | '.join(cells) + f' | {cs} |')

print('\n## median per-set RMSE ratio to c9 / % on the plateau (pct 100)\n')
print('| data | d | n | ' + ' | '.join(names) + ' |')
print('|---|---|---|' + '---|' * len(names))
for fam, dim, n, ns, res, c9, cond in rows:
    cells = []
    for k in names:
        if k not in res:
            cells.append('')
            continue
        J, dv, rm, pl, _, mt = res[k]
        r = [rm[s] / c9[s] for s in rm if s in c9 and c9[s] > 0]
        cells.append(f'{st.median(r):.2f} / {pl:.0f}%')
    print(f'| {fam} | {dim} | {n} | ' + ' | '.join(cells) + ' |')

ns = sorted({n for (_, n) in tot})
for i, what in ((0, 'div'), (2, 'mat')):
    print(f'\n## {what} per n, all configurations (% of sets)\n')
    print('| rule | ' + ' | '.join(str(n) for n in ns) + ' | all |')
    print('|---|' + '---|' * (len(ns) + 1))
    for k in names:
        c = [tot.get((k, n), [0, 0, 0]) for n in ns]
        a = [sum(x[i] for x in c), sum(x[1] for x in c)]
        print(f'| {k} | ' + ' | '.join(f'{100 * x[i] / x[1]:.1f}' if x[1] else '' for x in c)
              + f' | {100 * a[0] / max(a[1], 1):.1f} |')
