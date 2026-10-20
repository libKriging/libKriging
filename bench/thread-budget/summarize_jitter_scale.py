"""Summary of jitter_scale_study.cpp CSVs
(jitter-scale-<kernel>-<family>-<n>-<d>-<optim>-<noise>.csv).
Per configuration and jitter (off / on, the current default):
  div   % of data sets whose theta differs by > 1e-3 relative across the
        thread budgets (pct 10 / 100 / 200): another optimizer path;
  ident % of data sets with a bit-identical theta across the budgets;
  rmse  median normalized test RMSE (pct 100);
and the median per-set ratios on / off of the RMSE and of the fit time."""
import csv, glob, statistics as st, sys

prefix = sys.argv[1] if len(sys.argv) > 1 else 'jitter-scale'

def load(fn):
    D = {}
    for r in csv.DictReader(open(fn)):
        if r['theta0'] == 'nan':
            continue
        D.setdefault((int(r['set']), r['jitter']), {})[float(r['pct'])] = (
            float(r['theta0']), float(r['rmse']), float(r['ms']))
    return D

rows = []
for fn in sorted(glob.glob(prefix + '-*.csv')):
    k, fam, n, d, optim, noise = fn[len(prefix) + 1:-4].split('-', 5)
    D = load(fn)
    sets = sorted({s for (s, _) in D})
    out = {}
    for j in ('off', 'on'):
        div = ident = 0
        cnt = 0
        for s in sets:
            x = D.get((s, j), {})
            if len(x) < 3:
                continue
            cnt += 1
            th = [x[p][0] for p in x]
            div += (max(th) - min(th)) > 1e-3 * abs(max(th))
            ident += max(th) == min(th)
        out[j] = (100 * div / max(cnt, 1), 100 * ident / max(cnt, 1),
                  st.median(D[(s, j)][100.0][1] for s in sets if (s, j) in D and 100.0 in D[(s, j)]))
    both = [s for s in sets if 100.0 in D.get((s, 'on'), {}) and 100.0 in D.get((s, 'off'), {})]
    r_rmse = st.median(D[(s, 'on')][100.0][1] / D[(s, 'off')][100.0][1] for s in both)
    r_time = st.median(D[(s, 'on')][100.0][2] / D[(s, 'off')][100.0][2] for s in both)
    rows.append((k, fam, int(n), int(d), optim, noise, len(sets), out, r_rmse, r_time))

rows.sort(key=lambda r: (r[0], r[4], r[5], r[1], r[3], r[2]))
print('| kernel | data | n | d | optim | noise | sets | div off | div on | ident off | ident on | RMSE off | RMSE on | RMSE on/off | time on/off |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for k, fam, n, d, optim, noise, ns, out, rr, rt in rows:
    print(f'| {k} | {fam} | {n} | {d} | {optim} | {noise} | {ns} | {out["off"][0]:.0f}% | {out["on"][0]:.0f}% | '
          f'{out["off"][1]:.0f}% | {out["on"][1]:.0f}% | {out["off"][2]:.3g} | {out["on"][2]:.3g} | {rr:.2f} | {rt:.2f} |')
