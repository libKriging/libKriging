#!/bin/bash
# Phase 4 plan actually run (seeds reduced where a fit takes minutes, see the report).
D=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$D/phase4-fits.csv}; BIN=${BIN:-/home/user/baseline/bench-blas-threads}
M=master:${MASTER_LIB:-/home/user/baseline}; B=B:${BRANCH_LIB:-/home/user/branchlib}
A=A:${BRANCH_LIB:-/home/user/branchlib}:LK_BLAS_THREADS_MIN_N=-1
R="$D/run-phase4.sh $OUT $BIN"
$R kriging   BFGS      "100 400 1000" 10 $M $B
$R kriging   BFGS10    "100 400" 10 $M $A $B
$R nugget    BFGS      "100 400 1000" 10 $M $B
$R nugget    BFGS10    "100 400" 10 $M $A $B
$R warp_kuma BFGS      "100 400" 10 $M $B
$R warp_kuma BFGS10    "100 400" 10 $M $B
$R mlp       BFGS+Adam "100 400" 10 $M $B
$R mlp       BFGS      "100" 10 $M $B
$R warp_mlp  BFGS      "100" 10 $M $B
$R mlp       BFGS      "400" 5 $M $B
$R warp_mlp  BFGS      "400" 5 $M $B
$R kriging   BFGS      "2000" 5 $M $B
$R kriging   BFGS10    "1000" 5 $M $A $B
$R nugget    BFGS10    "1000" 3 $M $B
$R warp_kuma BFGS      "1000" 3 $M $B
$R warp_kuma BFGS10    "1000" 3 $M $B
$R mlp       BFGS+Adam "1000" 3 $M $B
