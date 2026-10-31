import sys, glob, math, collections
import numpy as np
def load_res(fn):
    out=collections.OrderedDict(); cur=None
    for line in open(fn):
        p=line.split()
        if p[0]=='FIT':
            key=(p[1],int(p[2]),p[3]); cur=dict(wall=float(p[5]),cpu=float(p[6]),vs=float(p[7]),r=[],ls=[]); out[key]=cur
        else:
            cur['r'].append(float(p[1])); cur['ls'].append(float(p[2]))
    for v in out.values(): v['r']=np.array(v['r']); v['ls']=np.array(v['ls'])
    return out
def load_z(fn):
    out={}; cur=None
    try:
        for line in open(fn):
            p=line.split()
            if p[0]=='CASE': key=(p[1],int(p[2]),p[3]); cur=out.setdefault(key,[])
            else: cur.append(float(p[2]))
    except FileNotFoundError: pass
    return {k:np.array(v) for k,v in out.items()}
# truncated chi2_1 mean for trimmed estimator
a=1.6448536269514722; phi=math.exp(-a*a/2)/math.sqrt(2*math.pi); tm90=(0.9-2*a*phi)/0.9
MED=0.454936423119572
EST={
 'mean_clamp': lambda z: max(1,np.mean(z**2)),
 'mean': lambda z: np.mean(z**2),
 'median': lambda z: np.median(z**2)/MED,
 'trim10': lambda z: np.mean(np.sort(z**2)[:int(math.ceil(0.9*len(z)))])/tm90,
 'q95': lambda z: (np.quantile(np.abs(z),0.95)/1.96)**2,
}
def scores(fit,c):
    r=fit['r']; ls=fit['ls']
    cov=np.mean(np.abs(r)<=1.96*math.sqrt(c))
    nlpd=np.mean(0.5*np.log(2*np.pi*c)+ls+0.5*r**2/c)
    return cov,nlpd
