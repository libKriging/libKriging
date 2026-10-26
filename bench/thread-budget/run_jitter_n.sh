#!/bin/sh
# Grid of the jitter-versus-n study (#426, docs/dev/CholJitter.md):
#   run_jitter_n.sh <jitter_n_study binary> <out dir> [kernel]
# Per n, the jitters are the constants 0, 1e-10, 1e-9, 1e-8 and the values at
# that n of the rules anchored at J(100) = 1e-9 (the current gauss value):
#   lin  1e-9 n / 100      (J / lambda_max constant: lambda_max <= n)
#   sqrt 1e-9 sqrt(n / 100)
#   quad 1e-9 (n / 100)^2  (Cholesky backward error n eps cond, cond <= n / J)
# Output: <out dir>/jitter-n-<kernel>-<family>-<n>-<d>.csv
set -e
bin=$1; out=$2; kernel=${3:-gauss}
mkdir -p "$out"
jlist() {
  python3 -c "
n=$1; a=1e-9
v=[0,1e-10,1e-9,1e-8,a*n/100,a*(n/100)**0.5,a*(n/100)**2]
print(','.join(sorted({'%.6g'%x for x in v}, key=float)))"
}
sets() { case $1 in 50|100|200) echo 20;; 500) echo 10;; 1000) echo 6;; *) echo 2;; esac; }
run() { # family d n
  f="$out/jitter-n-$kernel-$1-$3-$2.csv"
  [ -s "$f" ] && return 0
  "$bin" "$1" "$kernel" "$3" "$2" "$(sets $3)" BFGS "$(jlist $3)" > "$f.tmp" && mv "$f.tmp" "$f"
}
for n in 50 100 200 500 1000; do
  run branin 2 $n; run sin 2 $n; run sin 4 $n; run sin 8 $n; run hartmann6 6 $n
done
for n in 2000; do
  run sin 4 $n; run sin 8 $n; run hartmann6 6 $n
done
