# timing.py <runs> <n,...> <d> : single-thread (env) gradient timings, base vs new
import subprocess,sys,statistics as st
runs=int(sys.argv[1]); ns=[int(x) for x in sys.argv[2].split(",")]; d=sys.argv[3]
kerns=sys.argv[4].split(",") if len(sys.argv)>4 else ["matern5_2","gauss","exp","matern3_2"]
LMPKEYS=("gradR_k [optimized]","Wb_k =","ans[k]","u = M y","R^-1")
print("n,d,kernel,build,ll_ms,llgrad_ms,lmp_ms,lmpgrad_ms,llloop_ms,lmploop_ms,lmp_Rinv_ms")
for n in ns:
  for k in kerns:
    for b in ["base","new"]:
      exe="./gc-"+b
      t=subprocess.run([exe,str(n),d,k,"time",str(runs)],capture_output=True,text=True).stdout.strip().split(",")[3:]
      ll=[];lmp=[];rinv=[]
      for r in range(runs):
        out=subprocess.run([exe,str(n),d,k,"bench"],capture_output=True,text=True).stdout.split("\n")
        # the LL table comes first (8 lines), then the LMP table
        sect=0; s_lmp=0; s_rinv=0
        for line in out:
          if not line.startswith("|"): continue
          key,val=line.split("|")[1].strip(),float(line.split("|")[2])
          if key.startswith("CL ="): sect=1
          if sect==0 and key.startswith("gradR computation"): ll.append(val)
          if sect==1 and key.startswith(LMPKEYS):
            if key.startswith("R^-1"): s_rinv+=val
            else: s_lmp+=val
        lmp.append(s_lmp); rinv.append(s_rinv)
      print("%d,%s,%s,%s,%s,%.2f,%.2f,%.2f"%(n,d,k,b,",".join(t),st.median(ll),st.median(lmp),st.median(rinv)),flush=True)
