#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == true ]]; then
  set -x
fi

cd build

# Cleanup compiled libs to check right path finding
rm -fr src/lib
# add library directory search PATH for executables
export LD_LIBRARY_PATH=$PWD/installed/lib:${LD_LIBRARY_PATH}

ctest -C ${MODE} # --verbose
