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

    # Python tests (fit/predict calls that repeatedly enter OpenMP parallel
    # regions in LinearAlgebra.cpp, e.g. inside a BFGS loop) hang for ~1h
    # under ctest on this job specifically, while the identical computation
    # takes seconds on Linux/macOS or as a compiled C++ .exe -- the same
    # MinGW/Windows libgomp thread-pool churn already diagnosed and fixed for
    # Octave Windows (see tools/octave-windows/test.sh, commit b6cbf97a).
    # Forcing single-threaded OpenMP avoids that churn here too. Verified
    # across Python 3.7/3.9/3.10/3.11/3.12 (see issue #351).
    if [[ "$ENABLE_PYTHON_BINDING" == "on" ]]; then
      export OMP_NUM_THREADS=${DEBUG_354_OMP_NUM_THREADS:-1}
      echo "OMP_NUM_THREADS=${OMP_NUM_THREADS} (Windows Python job: avoid libgomp thread-pool churn) [DEBUG #354 override]"
    fi

    # TEMP DEBUG (issue #354): bisecting the WrappedPyKrigingParametricTest
    # Windows hang -- restrict to just that test and cut the timeout from
    # 1500s to 120s so a hang fails fast instead of eating the whole job
    # budget. Restore before merge.
    ctest -C "${MODE}" ${CTEST_FLAGS} --timeout 120 -R WrappedPyKrigingParametricTest

    cmake --build . --target install --config "${MODE}"
else
    # faster install target if tests are not required
    cmake --build . --target install.lib --config "${MODE}"
fi