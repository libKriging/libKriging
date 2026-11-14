#!/usr/bin/env bash
# Pool workers single-threaded for Accelerate (branch perf/pool-workers-accelerate-single): same binary,
#   old  = LK_ACCELERATE_POOL_SINGLE=0 (Accelerate's default mode in the workers)
#   new  = default of the branch
#   veclib1 = LK_ACCELERATE_POOL_SINGLE=0 VECLIB_MAXIMUM_THREADS=1 (every thread single, the reference)
# usage: macOS-M4-session-2-pool-single.sh <bench-fit-profile> <out.csv> [rounds]
set -uo pipefail
BIN=${1:?bench-fit-profile}; OUT=${2:?csv}; ROUNDS=${3:-5}
echo "round,n,optim,variant,fit_ms,proc_real_s,proc_cpu_s" > "$OUT"
run() { # variant n optim -> row
  local v=$1 n=$2 o=$3 tf of; tf=$(mktemp); of=$(mktemp)
  case $v in
    old) LK_ACCELERATE_POOL_SINGLE=0 /usr/bin/time -l "$BIN" $n 4 $o > "$of" 2> "$tf";;
    new) /usr/bin/time -l "$BIN" $n 4 $o > "$of" 2> "$tf";;
    veclib1) LK_ACCELERATE_POOL_SINGLE=0 VECLIB_MAXIMUM_THREADS=1 /usr/bin/time -l "$BIN" $n 4 $o > "$of" 2> "$tf";;
  esac
  local l; l=$(grep -m1 " real " "$tf")
  echo "$round,$n,$o,$v,$(grep -m1 '^Total:' "$of" | awk '{print $2}'),$(echo "$l" | awk '{print $1}'),$(echo "$l" | awk '{print $3+$5}')" >> "$OUT"
  rm -f "$tf" "$of"
}
for round in $(seq 1 $ROUNDS); do
  for n in 500 800 1500; do for o in BFGS BFGS2 BFGS4 BFGS10; do for v in old new veclib1; do run $v $n $o; done; done; done
  echo "round $round done $(date -u +%T)" >&2
done
touch "$OUT.done"
