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

# Same fix as build.sh's embedded ctest call -- see the comment there for
# the Octave Windows precedent (commit b6cbf97a) this mirrors.
if [[ "$ENABLE_PYTHON_BINDING" == "on" ]]; then
  export OMP_NUM_THREADS=1
  echo "OMP_NUM_THREADS=${OMP_NUM_THREADS} (Windows Python job: avoid libgomp thread-pool churn)"
fi

cd ${BUILD_DIR:-build}

# Cleanup compiled libs to check right path finding
rm -fr src bindings

ctest -C "${MODE}" ${CTEST_FLAGS}
