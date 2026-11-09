#!/bin/bash
# Phase 4 plan: seeded fits, master (rule off) vs the small-problem rule
# (LK_BLAS_THREADS_MIN_N=$TH), both with the branch library (the rule off is
# master's code path). Seeds reduced where a fit takes minutes (see the report).
# Resumable: fits already in $OUT are skipped.
D=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$D/phase4-fits.csv}; LIB=${LIB:-/home/user/branchlib}; BIN=${BIN:-$LIB/bench-blas-threads}
TH=${TH:-500}
M=master:$LIB; R=rule$TH:$LIB:LK_BLAS_THREADS_MIN_N=$TH
P="$D/run-phase4.sh $OUT $BIN"
$P kriging   BFGS      "100 400 1000" 10 $M $R
$P kriging   BFGS10    "100 400" 10 $M $R
$P nugget    BFGS      "100 400 1000" 10 $M $R
$P nugget    BFGS10    "100 400" 10 $M $R
$P warp_kuma BFGS      "100 400" 10 $M $R
$P warp_kuma BFGS10    "100 400" 10 $M $R
$P mlp       BFGS+Adam "100 400" 10 $M $R
$P mlp       BFGS      "100" 10 $M $R
$P warp_mlp  BFGS      "100" 10 $M $R
$P mlp       BFGS      "400" 5 $M $R
$P warp_mlp  BFGS      "400" 5 $M $R
$P kriging   BFGS      "2000" 5 $M $R
$P kriging   BFGS10    "1000" 5 $M $R
$P nugget    BFGS10    "1000" 3 $M $R
$P warp_kuma BFGS      "1000" 3 $M $R
$P mlp       BFGS+Adam "1000" 3 $M $R
# added: n = 200, where the OpenMP loops start (kOmpMinRows) and phase 2 shows the largest gap
$P kriging   BFGS      "200" 10 $M $R
$P nugget    BFGS      "200" 10 $M $R
$P warp_kuma BFGS      "200" 10 $M $R
$P mlp       BFGS+Adam "200" 10 $M $R
$P mlp       BFGS      "200" 10 $M $R
$P warp_mlp  BFGS      "200" 10 $M $R
