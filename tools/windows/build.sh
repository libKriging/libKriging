#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  export VERBOSE=true
  set -x
fi

# Default configuration when used out of CI
MODE=${MODE:-Debug}
EXTRA_CMAKE_OPTIONS=${EXTRA_CMAKE_OPTIONS:-}
BUILD_TEST=${BUILD_TEST:-true}

export ENABLE_OCTAVE_BINDING=${ENABLE_OCTAVE_BINDING:-auto}
export ENABLE_MATLAB_BINDING=${ENABLE_MATLAB_BINDING:-auto}
export ENABLE_PYTHON_BINDING=${ENABLE_PYTHON_BINDING:-auto}
export ENABLE_JULIA_BINDING=${ENABLE_JULIA_BINDING:-off}

BASEDIR=$(dirname "$0")
BASEDIR=$(cd "$BASEDIR" && pwd -P)
test -f "${BASEDIR}"/loadenv.sh && . "${BASEDIR}"/loadenv.sh 

# OpenBLAS installation
export EXTRA_SYSTEM_LIBRARY_PATH=${HOME}/Miniconda3/Library/lib

# mwblas and mwlapack are provided by Matlab/extern on Windows
# export EXTRA_SYSTEM_LIBRARY_PATH="C:/Program Files/MATLAB/R2022a/extern/lib/win64/microsoft"
# EXTRA_CMAKE_OPTIONS="${EXTRA_CMAKE_OPTIONS} -DBLAS_NAMES=libmwblas -DLAPACK_NAMES=libmwlapack"
STATIC_LIB=on
MAKE_SHARED_LIBS=off
EXTRA_CMAKE_OPTIONS="-DBUILD_SHARED_LIBS=${MAKE_SHARED_LIBS} -DSTATIC_LIB=${STATIC_LIB}"

mkdir -p ${BUILD_DIR:-build}
cd ${BUILD_DIR:-build}
cmake \
  -DCMAKE_GENERATOR_PLATFORM=x64 \
  -DEXTRA_SYSTEM_LIBRARY_PATH="${EXTRA_SYSTEM_LIBRARY_PATH}" \
  -DENABLE_OCTAVE_BINDING=${ENABLE_OCTAVE_BINDING} \
  -DENABLE_MATLAB_BINDING=${ENABLE_MATLAB_BINDING} \
  -DENABLE_PYTHON_BINDING=${ENABLE_PYTHON_BINDING} \
  -DENABLE_JULIA_BINDING=${ENABLE_JULIA_BINDING} \
  -DUSE_COMPILER_CACHE="${USE_COMPILER_CACHE}" \
  $(eval echo ${EXTRA_CMAKE_OPTIONS}) \
  ..

if [[ "$BUILD_TEST" == "true" ]]; then
    cmake --build . --target ALL_BUILD --config "${MODE}"

    if [[ "$DEBUG_CI" == "true" ]]; then
      CTEST_FLAGS="--verbose --output-on-failure"
      set -x
    else
      CTEST_FLAGS=--output-on-failure
    fi

    # TEMP DEBUG (issue #351): fix attempt, based on the identical symptom
    # already diagnosed and fixed for Octave Windows (see
    # tools/octave-windows/test.sh / commit b6cbf97a) -- MinGW/Windows
    # libgomp thread-pool churn when OpenMP parallel regions
    # (LinearAlgebra.cpp) are entered repeatedly inside an optimization loop,
    # causing a >1000x slowdown on Windows specifically vs Linux/macOS for
    # otherwise-identical work. That fix scoped OMP_NUM_THREADS=1 to the
    # Octave Windows job; trying the same here for Python Windows, since the
    # observed pattern (fine on Linux/macOS, fine as a compiled C++ .exe,
    # hangs specifically for Python tests that fit/optimize repeatedly) matches.
    if [[ "$ENABLE_PYTHON_BINDING" == "on" ]]; then
      export OMP_NUM_THREADS=1
      echo "OMP_NUM_THREADS=${OMP_NUM_THREADS} (Windows Python job: avoid libgomp thread-pool churn, see issue #351)"
    fi

    # TEMP DEBUG (issue #351): also cap the per-test timeout so that if the
    # OMP_NUM_THREADS fix above does NOT fully resolve the hang, ctest moves
    # on and we still get the full picture (which tests, if any, still hang)
    # in one CI run instead of eating the whole job budget on the first one.
    # Restore both of the above before merge.
    CTEST_FLAGS="${CTEST_FLAGS} --timeout 180"

    # Test on fresh build lib (before installation)
    ctest -C "${MODE}" ${CTEST_FLAGS}

    cmake --build . --target install --config "${MODE}"
else
    # faster install target if tests are not required
    cmake --build . --target install.lib --config "${MODE}"
fi