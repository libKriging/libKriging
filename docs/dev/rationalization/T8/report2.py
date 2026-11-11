from analyze import *
rng=np.random.default_rng(0)
V=['K0','K5s','K3s','K3s10','K5s10']
R={};Z={}
for v in V:
    a=load_res(f'res_c4_{v}.txt'); a={k:x for k,x in a.items() if k[2]=='BFGS'}; a.update(load_res(f'res_x_{v}.txt')); R[v]=a
    z=load_z(f'z_c4_{v}.txt'); z.update(load_z(f'z_x_{v}.txt')); Z[v]=z
keys=list(R['K5s'].keys()); print(len(keys),'cases')
def per(v,est='mean_clamp'):
    cov=[];nl=[];cpu=[]
    for k in keys:
        f=R[v][k]; z=Z[v].get(k)
        c=1.0 if v=='K0' else (f['vs'] if est=='lib' else (EST[est](z) if z is not None else 1.0))
        a,b=scores(f,c); cov.append(a); nl.append(b); cpu.append(f['cpu'])
    return np.array(cov),np.array(nl),np.array(cpu)
def ci(x):
    b=[np.mean(rng.choice(x,len(x))) for _ in range(2000)]; return np.quantile(b,[.025,.975])
c5,n5,_=per('K5s')
for v in V:
  for est in (['mean_clamp','mean','q95'] if v!='K0' else ['-']):
    cv,nl,_=per(v,est if est!='-' else 'mean')
    d=nl-n5; lo,hi=ci(d); err=np.abs(cv-.95)
    print(f"{v:6s} {est:10s} cov mean {cv.mean():.3f} med {np.median(cv):.3f} <0.8:{(cv<.8).sum():3d} <0.9:{(cv<.9).sum():3d} |cov-.95| {err.mean():.3f} | dNLPD vs K5s: med {np.median(d):+.3f} mean {d.mean():+.2f} CI[{lo:+.2f},{hi:+.2f}]  frac better {np.mean(d<0):.2f}")
# per dataset coverage
names=sorted(set(k[0] for k in keys))
print("per-dataset mean coverage (K0,K5s,K3s,K3s10)")
for nm in names:
    row=[]
    for v in ['K0','K5s','K3s','K3s10']:
        cv,_,_=per(v,'mean_clamp'); row.append(np.mean([cv[i] for i,k in enumerate(keys) if k[0]==nm]))
    print(f"  {nm:14s} "+" ".join(f"{x:.3f}" for x in row))
