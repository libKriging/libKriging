from analyze import *
V=['K0','K5s','K3s','K5f','K3f','K3s10','K5s10','K10s']
R={t:{v:load_res(f'res_{t}_{v}.txt') for v in V} for t in ['c1','c4']}
Z={v:load_z(f'z_c4_{v}.txt') for v in V}
print("== timing (sum over cases), optim BFGS: 45 fits / BFGS4: 3 fits branin_proj5")
for opt in ['BFGS','BFGS4']:
  for t in ['c1','c4']:
    b=R[t]['K0']; keys=[k for k in b if k[2]==opt]
    bw=sum(b[k]['wall'] for k in keys); bc=sum(b[k]['cpu'] for k in keys)
    row=[f"{opt:5s} {t}: K0 wall {bw:7.2f} cpu {bc:7.2f}"]
    for v in V[1:]:
      w=sum(R[t][v][k]['wall'] for k in keys); c=sum(R[t][v][k]['cpu'] for k in keys)
      row.append(f"{v} x{w/bw:.2f}/x{c/bc:.2f}")
    print(" | ".join(row))
# per-case timing ratio median
print("== median per-case ratio of CPU (c1) to K0")
for v in V[1:]:
  rs=[R['c1'][v][k]['cpu']/R['c1']['K0'][k]['cpu'] for k in R['c1']['K0'] if k[2]=='BFGS']
  print(v, "median x%.2f  q10 x%.2f q90 x%.2f"%(np.median(rs),np.quantile(rs,.1),np.quantile(rs,.9)))
print("== quality (c4 fits, all 48 cases). cov: mean | median | #<0.8 | #<0.5 | mean|cov-.95| ; dNLPD vs K5s mean_clamp: median, mean")
base={}
for k,f in R['c4']['K5s'].items(): base[k]=scores(f,f['vs'])[1]
raw=R['c4']['K0']
covs=[scores(f,1)[0] for f in raw.values()]
nl=[scores(f,1)[1]-base[k] for k,f in raw.items()]
print("%-14s cov %.3f | %.3f | %2d | %2d | %.3f ; dNLPD med %+.3f mean %+.2f"%("uncalibrated",np.mean(covs),np.median(covs),sum(c<.8 for c in covs),sum(c<.5 for c in covs),np.mean(np.abs(np.array(covs)-.95)),np.median(nl),np.mean(nl)))
for v in V[1:]:
  for e,fn in EST.items():
    covs=[];nl=[]
    for k,f in R['c4'][v].items():
      z=Z[v].get(k)
      c=fn(z) if z is not None and 2*len(z)>=len(f['r'])*0+1 else 1
      cv,nlp=scores(f,c); covs.append(cv); nl.append(nlp-base[k])
    covs=np.array(covs)
    print("%-5s %-10s cov %.3f | %.3f | %2d | %2d | %.3f ; dNLPD med %+.3f mean %+.2f"%(v,e,covs.mean(),np.median(covs),(covs<.8).sum(),(covs<.5).sum(),np.mean(np.abs(covs-.95)),np.median(nl),np.mean(nl)))
