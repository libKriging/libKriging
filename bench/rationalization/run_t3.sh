#!/bin/sh
# Grid of the T3 study (docs/dev/rationalization/T3-default-multistart.md):
#   run_t3.sh <t3_multistart_study binary> <out dir> [A|B] [jobs]
# pass A (default): every configuration, 30 data sets, one core per process
#   (OPENBLAS_NUM_THREADS=1, Optim budget 25 % of 4 cores), <jobs> processes
#   (default 4) at a time: miss rate, RMSE, single-core CPU time.
# pass B: timing, one process at a time, OpenBLAS threads as installed:
#   n 200, LL, gauss / matern5_2, none / nugget, 10 data sets, 1 core (budget
#   25 %) and the whole machine (100 %); n 1000, LL, noise-free, 5 data sets,
#   whole machine (+ sin gauss on 1 core, contention check against pass A);
#   n 500, LOO / LMP, gauss, sin / sinhalf, 5 data sets, whole machine.
# Output: <out dir>/<pass>-<family>-<kernel>-<noise>-<objective>-<n>-<set0>.csv
# (skipped when present); then summarize_t3.py <out dir>.
set -e
if [ "$1" = --one ]; then # --one out pass bin family kernel noise objective n d set0 set1 pct variants
  out=$2; pass=$3; bin=$4; shift 4
  f="$out/$pass-$1-$2-$3-$4-$5-$7.csv"
  [ -s "$f" ] && exit 0
  "$bin" "$@" > "$f.tmp" 2> "$f.log" && mv "$f.tmp" "$f"
  exit 0
fi
self=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
bin=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); out=$(mkdir -p "$2" && cd "$2" && pwd); pass=${3:-A}; jobs=${4:-4}
fams="sin:4 hartmann6:6 branin:4 sinhalf:8"
if [ "$pass" = A ]; then
  export OPENBLAS_NUM_THREADS=1
  for n in 100 200 500 1000; do
    for nz in none nugget; do for obj in LL LOO LMP; do
      [ $nz = nugget ] && [ $obj = LOO ] && continue  # LOO: noise model None only
      [ $obj != LL ] && [ $n = 1000 ] && continue     # LOO / LMP: n <= 500 (cost)
      v=BFGS,BFGS4,BFGS10
      # long-range restart: active for a noise-free LL single start, and for a
      # nugget fit through its noise-free BFGS warm start; noise-free LOO / LMP:
      # BFGS-nolr == BFGS, checked at n 100
      if [ $obj = LL ] || [ $nz = nugget ] || [ $n = 100 ]; then v=BFGS,BFGS-nolr,BFGS4,BFGS10; fi
      step=10; [ $n -ge 500 ] && step=5
      for k in gauss matern5_2; do for fd in $fams; do
        s=0; while [ $s -lt 30 ]; do
          echo "$out A $bin ${fd%:*} $k $nz $obj $n ${fd#*:} $s $((s + step)) 25 $v"; s=$((s + step))
        done
      done; done
    done; done
  done > "$out/jobs-A.txt"
  xargs -P "$jobs" -L 1 "$self" --one < "$out/jobs-A.txt"
else
  V=BFGS,BFGS-nolr,BFGS4,BFGS10
  b() { "$self" --one "$out" "B$1" "$bin" "$2" "$3" "$4" "$5" "$6" "$7" 0 "$8" "$1" "$9"; }
  for pct in 25 100; do for nz in none nugget; do for k in gauss matern5_2; do for fd in $fams; do
    b $pct "${fd%:*}" $k $nz LL 200 "${fd#*:}" 10 $V
  done; done; done; done
  for k in gauss matern5_2; do for fd in $fams; do
    b 100 "${fd%:*}" $k none LL 1000 "${fd#*:}" 5 $V
  done; done
  b 25 sin gauss none LL 1000 4 5 $V
  for obj in LOO LMP; do for fd in sin:4 sinhalf:8; do
    b 100 "${fd%:*}" gauss none $obj 500 "${fd#*:}" 5 BFGS,BFGS4,BFGS10
  done; done
fi
