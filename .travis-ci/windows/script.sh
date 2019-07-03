#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == true ]]; then
  set -x
fi

# Default configuration when used out of travis-ci
MODE=${MODE:-Debug}
if [[ -n ${TRAVIS_BUILD_DIR:+x} ]]; then
echo
    cd ${TRAVIS_BUILD_DIR}
fi

# OpenBLAS installation
export EXTRA_SYSTEM_LIBRARY_PATH=${HOME}/Miniconda3/Library/lib

mkdir build
cd build
cmake .. -DCMAKE_GENERATOR_PLATFORM=x64 -DEXTRA_SYSTEM_LIBRARY_PATH=${EXTRA_SYSTEM_LIBRARY_PATH}
cmake --build . --target ALL_BUILD --config ${MODE}
# add library directory search PATH for executables
export PATH=$PWD/src/lib/${MODE}:$PATH

if [[ "$MODE" == "Coverage" ]]; then
    echo "Coverage not supported for Windows"
    travis_terminate 1
fi

cmake --build . --target install --config ${MODE}
# Show installation directory content
# find installed
# Cleanup compiled libs to check right path finding
rm -fr src/lib
# add library directory search PATH for executables
export PATH=$PWD/installed/bin:$PATH
# add OpenBLAS DLL library path
export PATH=$HOME/Miniconda3/Library/bin:$PATH

ctest -C ${MODE}
