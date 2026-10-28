#!/bin/bash
# Phase 2 grid: fixed-parameter micro-benchmarks over OPENBLAS_NUM_THREADS.
# Usage: run-phase2.sh <bench-blas-threads binary> <out.csv> [threads list]
BIN=$1; OUT=$2; THREADS=${3:-"1 2 4"}
echo "op,n,d,blas_threads,omp_threads,calls,ms_per_call,cpu_ms_per_call" > "$OUT"
run() { # op n d
  for t in $THREADS; do
    OPENBLAS_NUM_THREADS=$t timeout 900 "$BIN" "$1" "$2" "$3" 1.5 >> "$OUT" || echo "$1,$2,$3,$t,,,FAIL" >> "$OUT"
  done
}
for n in 50 100 200 400 1000 2000 4000; do
  for op in kriging_ll kriging_predict mlp_ll mlp_predict warp_ll; do run $op $n 4; done
done
for n in 100 400 1000 2000; do
  for op in kriging_ll kriging_predict mlp_ll mlp_predict warp_ll; do run $op $n 10; done
done
