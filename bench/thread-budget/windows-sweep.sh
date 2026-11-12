#!/usr/bin/env bash
# sweep.sh <out.csv>: bench-threads grid of #473 section 2, both thresholds, with and without OPENBLAS_THREAD_TIMEOUT=4
OUT=$1; R=${R:-5}
cd /c/Users/riche/libKriging
export PATH="$HOME/Miniconda3/Library/bin:$PATH" OPENBLAS_L2_SIZE=2048
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS LK_NUM_THREADS_PERCENT
echo "toolchain,threshold,env,rep,blas_env,omp_env,$(build/bin/bench-threads.exe 10 2 1 matern5_2 --header | head -1)" > "$OUT"
for envv in none OPENBLAS_THREAD_TIMEOUT=4; do
 for rep in $(seq 1 $R); do
  for b in build:800 build-omp200:200; do
   dir=${b%%:*}; thr=${b##*:}
   for n in 100 200 400 800 1500; do
    for tb in 1 2 4 10 20; do
     for to in 1 2 4 10 20; do
      if [ "$envv" = none ]; then row=$(OPENBLAS_NUM_THREADS=$tb OMP_NUM_THREADS=$to $dir/bin/bench-threads.exe $n 4 7 matern5_2)
      else row=$(env $envv OPENBLAS_NUM_THREADS=$tb OMP_NUM_THREADS=$to $dir/bin/bench-threads.exe $n 4 7 matern5_2); fi
      echo "msvc,$thr,$envv,$rep,$tb,$to,$row" >> "$OUT"
     done
    done
   done
  done
 done
done
echo done >> "$OUT.done"
