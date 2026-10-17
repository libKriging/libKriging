#!/bin/bash
# Repetitions of the phase 2 grid (d = 4, n <= 2000), appended to <out.csv>
# with the repetition number in the first column, to estimate the noise.
BIN=$1; OUT=$2; REPS=${3:-2}; THREADS=${4:-"1 2 4"}
[ -f "$OUT" ] || echo "rep,op,n,d,blas_threads,omp_threads,calls,ms_per_call,cpu_ms_per_call" > "$OUT"
for r in $(seq 1 "$REPS"); do
  for n in 50 100 200 400 700 1000 2000; do
    for op in kriging_ll kriging_predict mlp_ll mlp_predict warp_ll; do
      for t in $THREADS; do
        echo "$r,$(OPENBLAS_NUM_THREADS=$t "$BIN" $op $n 4 1.5)" >> "$OUT"
      done
    done
  done
done
