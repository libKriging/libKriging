#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == true ]]; then
  CTEST_FLAGS=--verbose
  set -x
fi

cd build

if [[ "$MODE" == "Coverage" ]]; then
    cmake --build . --target coverage --config Coverage
else
    ctest -C "${MODE}" ${CTEST_FLAGS}

    # Cleanup compiled libs to check right path finding
    rm -fr src/lib
    # add library directory search PATH for executables
    export LD_LIBRARY_PATH=$PWD/installed/lib:${LD_LIBRARY_PATH}

    ctest -C "${MODE}" ${CTEST_FLAGS}
fi

