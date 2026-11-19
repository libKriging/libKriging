#!/bin/bash
# #467 measurement grid. Usage: run.sh <bench_467 binary> <out dir>
# Pass A (quality, 1-core CPU time): 4 processes at a time, each pinned to one core
#   (taskset -c k, LK_NUM_THREADS_PERCENT=25: budget 1), designs 0-19, tanh and selu.
# Pass B (wall time on 4 cores vs 1 core): one process alone, designs 0-2, tanh;
#   4 cores = taskset -c 0-3 with the default budget (100 %), 1 core = taskset -c 0, budget 25 %.
# Pass C (#407, Python): case407.py, designs 0-9 on 1 core (4 at a time), designs 0-2 on 4 cores.
set -e
bin=$(realpath "$1"); out=$(realpath -m "$2"); here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
variants="BFGS:3 BFGS:5 BFGS2:3 BFGS2:5 BFGS4:3 BFGS4:5 BFGS10:5"
# pass A: one job per (variant, activation, core)
jobsA() { for v in $variants; do for a in tanh selu; do for c in 0 1 2 3; do
  echo "$c ${v%%:*} ${v##*:} $a $((5 * c)) 5"; done; done; done; }
jobsA | xargs -P 4 -L 1 bash -c 'LK_NUM_THREADS_PERCENT=25 taskset -c $0 '"$bin"' '"$out"'/A_$1_$2_$3_$0.txt $2 $1 $3 all $4 $5'
# reference: plain Kriging on the same data (quality only)
for c in 0 1 2 3; do LK_NUM_THREADS_PERCENT=25 taskset -c $c "$bin" "$out/K_$c.txt" 0 Kriging none all $((5 * c)) 5 & done; wait
# pass B
for v in $variants; do o=${v%%:*}; k=${v##*:}
  LK_NUM_THREADS_PERCENT=25 taskset -c 0 "$bin" "$out/B1_${o}_$k.txt" "$k" "$o" tanh all 0 3
  taskset -c 0-3 "$bin" "$out/B4_${o}_$k.txt" "$k" "$o" tanh all 0 3
done
# pass C
for c in 0 1 2 3; do d0=$((c * 3)); nd=3; [ $c = 3 ] && nd=1
  LK_NUM_THREADS_PERCENT=25 taskset -c $c python3 "$here/case407.py" "$out/C1_$c.csv" BFGS,BFGS2,BFGS4,BFGS10 3,5 tanh,selu $d0 $nd &
done; wait
taskset -c 0-3 python3 "$here/case407.py" "$out/C4.csv" BFGS,BFGS2,BFGS4,BFGS10 3,5 tanh,selu 0 3
LK_NUM_THREADS_PERCENT=25 taskset -c 0 python3 "$here/case407.py" "$out/CK.csv" Kriging 0 none 0 10
