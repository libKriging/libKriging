#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == true ]]; then
  export VERBOSE=true
  set -x
fi

# Default configuration when used out of travis-ci
MODE=${MODE:-Debug}
EXTRA_CMAKE_OPTIONS=${EXTRA_CMAKE_OPTIONS:-}
BUILD_TEST=${BUILD_TEST:-true}

if [[ -n ${TRAVIS_BUILD_DIR:+x} ]]; then
echo
    cd "${TRAVIS_BUILD_DIR}"
fi

# OpenBLAS installation
export EXTRA_SYSTEM_LIBRARY_PATH=${HOME}/Miniconda3/Library/lib

mkdir -p build
cd build
cmake \
  -DCMAKE_GENERATOR_PLATFORM=x64 \
  -DEXTRA_SYSTEM_LIBRARY_PATH=${EXTRA_SYSTEM_LIBRARY_PATH} \
  ${EXTRA_CMAKE_OPTIONS} \
  ..

if [[ "$BUILD_TEST" == "true" ]]; then
    cmake --build . --target ALL_BUILD --config "${MODE}"
    # add library directory search PATH for executables
    export PATH=$PWD/src/lib/${MODE}:$PATH

    cmake --build . --target install --config "${MODE}"
else
    # faster install target if tests are not required
    cmake --build . --target install.lib --config "${MODE}"
fi