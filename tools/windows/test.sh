#!/usr/bin/env bash
set -eo pipefail

BASEDIR=$(dirname "$0")
BASEDIR=$(cd "$BASEDIR" && pwd -P)
test -f "${BASEDIR}"/loadenv.sh && . "${BASEDIR}"/loadenv.sh 

if [[ "$DEBUG_CI" == "true" ]]; then
  CTEST_FLAGS="--verbose --output-on-failure"
  set -x
else
  CTEST_FLAGS=--output-on-failure
fi

MODE=${MODE:-Debug}

if [[ "$ENABLE_COVERAGE" == "on" ]]; then
    echo "Coverage not supported for Windows"
    exit 1
fi

if [[ "$ENABLE_OCTAVE_BINDING" == "on" ]]; then
  # NestedKriging tests (fit on n=400 with matern5_2) hang for ~25min under
  # ctest's default TIMEOUT on this job specifically, while taking <1s on
  # Linux/macOS -- a >1000x slowdown far beyond normal cross-platform BLAS/CPU
  # variance. Suspected cause: MinGW's libgomp has expensive thread-pool
  # churn on Windows when OpenMP parallel regions (LinearAlgebra.cpp) are
  # entered repeatedly inside a BFGS optimization loop. Forcing single-
  # threaded OpenMP avoids that churn.
  export OMP_NUM_THREADS=1
  echo "OMP_NUM_THREADS=${OMP_NUM_THREADS} (Octave Windows job: avoid libgomp thread-pool churn)"
fi

cd ${BUILD_DIR:-build}

# Cleanup compiled libs to check right path finding
rm -fr src bindings

ctest -C "${MODE}" ${CTEST_FLAGS}
