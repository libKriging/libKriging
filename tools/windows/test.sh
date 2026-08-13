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

# TEMP DEBUG (issue #351): same fix attempt as build.sh's embedded ctest
# call -- see the comment there for the Octave Windows precedent. Restore
# both this and the --timeout below before merge.
if [[ "$ENABLE_PYTHON_BINDING" == "on" ]]; then
  export OMP_NUM_THREADS=1
  echo "OMP_NUM_THREADS=${OMP_NUM_THREADS} (Windows Python job: avoid libgomp thread-pool churn, see issue #351)"
fi
CTEST_FLAGS="${CTEST_FLAGS} --timeout 180"

cd ${BUILD_DIR:-build}

# Cleanup compiled libs to check right path finding
rm -fr src bindings

ctest -C "${MODE}" ${CTEST_FLAGS}
