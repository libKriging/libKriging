#!/bin/bash
cd "$(dirname "$0")"
tag=x
run() { v=$1; folds=$2; shift 2
  env "$@" LK_T8_REP0=3 LK_T8_DUMP=$PWD/z_${tag}_$v.txt taskset -c 0-3 ./bench_t8 res_${tag}_$v.txt $folds BFGS all 9; }
rm -f res_${tag}_* z_${tag}_*
run K0 0
run K5s 5
run K3s 3
run K3s10 3 LK_T8_ITER=10
run K5s10 5 LK_T8_ITER=10
