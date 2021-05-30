#!/usr/bin/env bash
set -eo pipefail

BASEDIR=$(dirname "$0")
BASEDIR=$(readlink -f "${BASEDIR}")

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

. ${BASEDIR}/loadenv.sh

cd build

# Cleanup compiled libs to check right path finding
rm -fr src/lib
# add library directory search PATH for executables
export PATH=$PWD/installed/bin:$PATH

ctest -C "${MODE}" ${CTEST_FLAGS}
