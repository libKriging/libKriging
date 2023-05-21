#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  export VERBOSE=true
  set -x
fi

# Default configuration when used out of travis-ci
MODE=${MODE:-Debug}
EXTRA_CMAKE_OPTIONS=${EXTRA_CMAKE_OPTIONS:-}
BUILD_TEST=${BUILD_TEST:-true}

export ENABLE_OCTAVE_BINDING=${ENABLE_OCTAVE_BINDING:-auto}
export ENABLE_MATLAB_BINDING=${ENABLE_MATLAB_BINDING:-auto}
export ENABLE_PYTHON_BINDING=${ENABLE_PYTHON_BINDING:-auto}

BASEDIR=$(dirname "$0")
BASEDIR=$(cd "$BASEDIR" && pwd -P)
test -f "${BASEDIR}"/loadenv.sh && . "${BASEDIR}"/loadenv.sh 

if [[ -n ${TRAVIS_BUILD_DIR:+x} ]]; then
    cd "${TRAVIS_BUILD_DIR}"
fi

# OpenBLAS installation
export EXTRA_SYSTEM_LIBRARY_PATH=${HOME}/Miniconda3/Library/lib

# mwblas and mwlapack are provided by Matlab/extern on Windows
# export EXTRA_SYSTEM_LIBRARY_PATH="C:/Program Files/MATLAB/R2022a/extern/lib/win64/microsoft"
# EXTRA_CMAKE_OPTIONS="${EXTRA_CMAKE_OPTIONS} -DBLAS_NAMES=libmwblas -DLAPACK_NAMES=libmwlapack"

if [[ -z ${CMAKE_Fortran_COMPILER} ]]; then
  if ( command -v R >/dev/null 2>&1 ); then
    CMAKE_Fortran_COMPILER="$(R CMD config FC | awk '{ print $1 }')"
  else
    echo "R not available; fall back to naive gfortran"
    CMAKE_Fortran_COMPILER=gfortran
  fi
fi

if [[ -z ${Fortran_LINK_FLAGS} ]]; then
  if ( command -v R >/dev/null 2>&1 ); then
    Fortran_LINK_FLAGS="$(R CMD config FLIBS)"
  else
    Fortran_LINK_FLAGS="-lgfortran -lquadmath"
  fi
fi


mkdir -p ${BUILD_DIR:-build}
cd ${BUILD_DIR:-build}
cmake \
  -DCMAKE_Fortran_COMPILER="${CMAKE_Fortran_COMPILER}" \
  -DFortran_LINK_FLAGS="${Fortran_LINK_FLAGS}" \
  -DCMAKE_GENERATOR_PLATFORM=x64 \
  -DEXTRA_SYSTEM_LIBRARY_PATH="${EXTRA_SYSTEM_LIBRARY_PATH}" \
  -DENABLE_OCTAVE_BINDING=${ENABLE_OCTAVE_BINDING} \
  -DENABLE_MATLAB_BINDING=${ENABLE_MATLAB_BINDING} \
  -DENABLE_PYTHON_BINDING=${ENABLE_PYTHON_BINDING} \
  -DUSE_COMPILER_CACHE="${USE_COMPILER_CACHE}" \
  $(eval echo ${EXTRA_CMAKE_OPTIONS}) \
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