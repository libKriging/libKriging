import subprocess,sys
rows=[]
worst_rel=0;worst_norm=0
for n in [int(a) for a in (sys.argv[2].split(",") if len(sys.argv)>2 else ["60","300"])]:
  for kern in ["gauss","exp","matern3_2","matern5_2","whitenoise"]:
    for d in range(1,11):
      out=[]
      for exe in ["./gc-base","./gc-new"]:
        r=subprocess.run([exe,str(n),str(d),kern,"check"],capture_output=True,text=True,env=dict(__import__("os").environ,GC_COND="1"))
        out.append([(l.split()[0],float(l.split()[1])) for l in r.stdout.split("\n") if l.strip()])
        if r.returncode: print(kern,d,n,r.stderr[-200:])
      cond=[v for t,v in out[1] if t=="COND"][0]
      for tag in ["LL","LMP"]:
        a=[v for t,v in out[0] if t==tag]; b=[v for t,v in out[1] if t==tag]
        if not a: continue
        nrm=max(abs(x) for x in a) or 1.0
        rel=max((abs(x-y)/abs(x) if x!=0 else (0 if y==0 else float('inf'))) for x,y in zip(a,b))
        rn=max(abs(x-y) for x,y in zip(a,b))/nrm
        worst_rel=max(worst_rel,rel); worst_norm=max(worst_norm,rn)
        rows.append((n,kern,d,tag,cond,rel,rn))
for r in rows:
  if r[5]>1e-12: print("%4d %-10s d=%2d %-3s cond=%.1e comp-rel=%.2e norm-rel=%.2e"%r)
print("worst component-relative: %.3e, worst norm-relative: %.3e over %d gradients"%(worst_rel,worst_norm,len(rows)))

wc=[r for r in rows if r[4]<1e8]
print("cond(R) < 1e8: %d gradients, worst component-relative %.3e"%(len(wc),max(r[5] for r in wc)))
import json; json.dump(rows,open("cmp-%s.json"%sys.argv[1] if len(sys.argv)>1 else "cmp.json","w"))
