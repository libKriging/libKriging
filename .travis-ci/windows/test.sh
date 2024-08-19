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
    travis_terminate 1
fi

cd ${BUILD_DIR:-build}

# Cleanup compiled libs to check right path finding
rm -fr src bindings

ctest -C "${MODE}" ${CTEST_FLAGS}
