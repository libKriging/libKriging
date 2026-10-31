#!/bin/bash
# run.sh <tag> <cpus> <pct> ; writes res_<tag>_<variant>.txt and z_<tag>_<variant>.txt
tag=$1; cpus=$2; pct=$3
cd "$(dirname "$0")"
run() { v=$1; folds=$2; optim=$3; filt=$4; shift 4
  env "$@" LK_NUM_THREADS_PERCENT=$pct LK_T8_DUMP=$PWD/z_${tag}_$v.txt taskset -c $cpus ./bench_t8 res_${tag}_$v.txt $folds $optim $filt 3; }
rm -f res_${tag}_* z_${tag}_*
for optim in BFGS BFGS4; do
  filt=all; [ $optim = BFGS4 ] && filt=branin_proj5
  run K0 0 $optim $filt
  run K5s 5 $optim $filt
  run K3s 3 $optim $filt
  run K5f 5 $optim $filt LK_T8_START=fit
  run K3f 3 $optim $filt LK_T8_START=fit
  run K3s10 3 $optim $filt LK_T8_ITER=10
  run K5s10 5 $optim $filt LK_T8_ITER=10
  run K10s 10 $optim $filt
done
