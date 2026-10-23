#!/usr/bin/env bash
# Thread-budget sweep of predict / simulate / update (docs/dev/parallel.md).
#
#   bench/bench-threads-ops.sh <build-dir> [out.csv]
#
# One CSV row per (op, n, n_new, nsim, BLAS = OpenMP thread count), the build
# dir recorded in the first column. Override the grids with N_LIST, T_LIST.
set -euo pipefail

BUILD=${1:?usage: $0 <build-dir> [out.csv]}
OUT=${2:-bench-threads-ops.csv}
BIN="$BUILD/bench/bench-threads-ops"
[ -x "$BIN" ] || { echo "missing $BIN (ninja -C $BUILD bench-threads-ops)" >&2; exit 1; }

NCPU=$(nproc)
N_LIST=${N_LIST:-"500 2000"}
T_LIST=${T_LIST:-"1 $((NCPU / 2)) $NCPU"}
D=4

[ -s "$OUT" ] || echo "build,$("$BIN" predict 1 1 1 --header)" > "$OUT"
run() {  # op n_new nsim reps
  for t in $T_LIST; do
    row=$(OPENBLAS_NUM_THREADS=$t OMP_NUM_THREADS=$t "$BIN" "$1" "$n" $D "$2" "$3" "$4")
    echo "$(basename "$BUILD"),$row" >> "$OUT"
    echo "$(basename "$BUILD") $row"
  done
}
for n in $N_LIST; do
  run predict 1 1 51
  run predict 100 1 15
  run predict 1000 1 7
  run predict_cov 100 1 15
  run predict_cov 1000 1 7
  run simulate 100 1 15
  run simulate 100 100 15
  run simulate 1000 1 7
  run simulate 1000 100 7
  run update 10 1 15
  run update 100 1 7
done
