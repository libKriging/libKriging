#!/usr/bin/env bash
set -eo pipefail

echo "DEBUG_CI=$DEBUG_CI" 

if [[ "$DEBUG_CI" == "true" ]]; then
  CTEST_FLAGS=--verbose
  set -x
else
  CTEST_FLAGS=--output-on-failure
fi

if [[ "$MODE" == "Coverage" ]]; then
    echo "Coverage not supported for Windows"
    travis_terminate 1
fi

cd build

# Cleanup compiled libs to check right path finding
rm -fr src/lib
# add library directory search PATH for executables
export PATH=$PWD/installed/bin:$PATH
# add OpenBLAS DLL library path
export PATH=$HOME/Miniconda3/Library/bin:$PATH

ctest -C "${MODE}" ${CTEST_FLAGS}
