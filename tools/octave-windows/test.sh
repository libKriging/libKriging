#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

BASEDIR=$(dirname "$0")
BASEDIR=$(cd "$BASEDIR" && pwd -P)
test -f "${BASEDIR}"/loadenv.sh && . "${BASEDIR}"/loadenv.sh 

# Tests (LinearAlgebra varying sizes / rapid fire) time out on Windows
export CTEST_EXCLUDE="varying sizes|rapid fire"

# NestedKriging tests (fit on n=400 with matern5_2) hang for ~25min under
# ctest's default TIMEOUT on this job specifically, while taking <1s on
# Linux/macOS -- a >1000x slowdown far beyond normal cross-platform BLAS/CPU
# variance. Suspected cause: MinGW's libgomp has expensive thread-pool churn
# on Windows when OpenMP parallel regions (LinearAlgebra.cpp) are entered
# repeatedly inside the BFGS optimization loop. Forcing single-threaded
# OpenMP avoids that churn.
export OMP_NUM_THREADS=1
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS} (Octave Windows job: avoid libgomp thread-pool churn)"

"${BASEDIR}"/../linux-macos/test.sh
