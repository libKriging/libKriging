#!/usr/bin/env bash
# Thread-budget benchmark sweep (docs/dev/parallel.md, bench/thread-budget/RESULTS.md).
#
#   bench/bench-threads.sh <build-dir> [out.csv]
#
# Sweeps n, d and the (BLAS, OpenMP) thread counts with bench-threads and
# appends one CSV row per configuration. The BLAS variant in use (OpenBLAS
# pthreads / OpenMP / sequential) is recorded in each row (blas_kind): switch
# it between runs (see the TODO) and append to the same file.
# Override the grids with N_LIST, D_LIST, T_LIST, REPS.
set -euo pipefail

BUILD=${1:?usage: $0 <build-dir> [out.csv]}
OUT=${2:-bench-threads.csv}
BIN="$BUILD/bench/bench-threads"
[ -x "$BIN" ] || { echo "missing $BIN (ninja -C $BUILD bench-threads)" >&2; exit 1; }

NCPU=$(nproc)
N_LIST=${N_LIST:-"200 500 1000 2000 3000"}
D_LIST=${D_LIST:-"2 10"}
T_LIST=${T_LIST:-"1 2 4 $((NCPU / 2)) $NCPU"}
REPS=${REPS:-7}

[ -s "$OUT" ] || "$BIN" 10 2 1 matern5_2 --header | head -1 > "$OUT"
for n in $N_LIST; do
  for d in $D_LIST; do
    for tb in $T_LIST; do
      for to in $T_LIST; do
        OPENBLAS_NUM_THREADS=$tb OMP_NUM_THREADS=$to "$BIN" "$n" "$d" "$REPS" matern5_2 >> "$OUT"
        echo "n=$n d=$d blas=$tb omp=$to: $(tail -1 "$OUT")"
      done
    done
  done
done
