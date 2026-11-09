#!/bin/bash
# Phase 2: fixed-parameter micro-benchmarks (bench-blas-threads) over the
# OpenBLAS thread count, OpenMP left at its default (the thread budget).
# Usage: run-phase2.sh <bench-blas-threads> <out.csv> <reps> "<n list>" <d> ["<threads>"] ["<ops>"] [env...]
#   one CSV row per (rep, op, n, d, threads); rows already present are skipped (resume)
BIN=$1; OUT=$2; REPS=$3; NS=$4; D=$5; THREADS=${6:-"1 2 4"}
OPS=${7:-"kriging_ll kriging_predict mlp_ll mlp_predict warp_ll"}; shift 7 2>/dev/null || shift $#
[ -f "$OUT" ] || echo "rep,op,n,d,blas_threads,omp_threads,calls,ms_per_call,cpu_ms_per_call" > "$OUT"
for r in $(seq 1 "$REPS"); do
  for n in $NS; do
    for op in $OPS; do
      for t in $THREADS; do
        grep -q "^$r,$op,$n,$D,$t," "$OUT" && continue
        line=$(env "$@" OPENBLAS_NUM_THREADS=$t timeout 1800 "$BIN" $op $n $D 1.5)
        [ -n "$line" ] || line="$op,$n,$D,$t,,,,FAIL"
        echo "$r,$line" >> "$OUT"
      done
    done
  done
done
