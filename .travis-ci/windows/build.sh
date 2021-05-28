#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  export VERBOSE=true
  set -x
fi

BASEDIR=$(dirname "$0")
BASEDIR=$(readlink -f "${BASEDIR}")

# Default configuration when used out of travis-ci
MODE=${MODE:-Debug}
EXTRA_CMAKE_OPTIONS=${EXTRA_CMAKE_OPTIONS:-}
BUILD_TEST=${BUILD_TEST:-true}

export ENABLE_OCTAVE_BINDING=${ENABLE_OCTAVE_BINDING:-auto}
export ENABLE_PYTHON_BINDING=${ENABLE_PYTHON_BINDING:-auto}

if [[ -n ${TRAVIS_BUILD_DIR:+x} ]]; then
    cd "${TRAVIS_BUILD_DIR}"
fi

. ${BASEDIR}/loadenv.sh

# OpenBLAS installation
export EXTRA_SYSTEM_LIBRARY_PATH=${HOME}/Miniconda3/Library/lib

mkdir -p build
cd build
cmake \
  -DCMAKE_GENERATOR_PLATFORM=x64 \
  -DEXTRA_SYSTEM_LIBRARY_PATH=${EXTRA_SYSTEM_LIBRARY_PATH} \
  -DENABLE_OCTAVE_BINDING=${ENABLE_OCTAVE_BINDING} \
  -DENABLE_PYTHON_BINDING=${ENABLE_PYTHON_BINDING} \
  -DUSE_COMPILER_CACHE="${USE_COMPILER_CACHE}" \
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