#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == true ]]; then
  set -x
fi

# Default configuration when used out of travis-ci
if [[ -n ${TRAVIS_BUILD_DIR:+x} ]]; then
    cd "${TRAVIS_BUILD_DIR}"
fi

# make
export PATH="/c/Program Files/make/make-4.3/bin":${PATH}

# R shortcut (and tools with absolute path) by Chocolatey
export PATH="/c/Program Files/R/R-3.6.0/bin":$PATH
# Required to disambiguate with Chocolatey GCC installation
export PATH="/c/Rtools/mingw_64/bin":$PATH

## R shortcut (and tools) by Anaconda
#export PATH=${HOME}/Miniconda3/Scripts:${PATH}
#export PATH=$HOME/Miniconda3/Rtools/mingw_64/bin:$PATH

# OpenBLAS installation
export EXTRA_SYSTEM_LIBRARY_PATH=${HOME}/Miniconda3/Library/lib
# and libomp.dll, flang.dll, flangrti.dll, openblas.dll
export PATH=${HOME}/Miniconda3/Library/bin:$PATH

# Windows + Shared : OK
# Windows + Static : OK
MAKE_SHARED_LIBS=on

BASEDIR=$(dirname "$0")
BASEDIR=$(readlink -f "${BASEDIR}")

BUILD_TEST=false \
    MODE=Release \
    CC=$(R CMD config CC) \
    CXX=$(R CMD config CXX) \
    EXTRA_CMAKE_OPTIONS="-DBUILD_SHARED_LIBS=${MAKE_SHARED_LIBS} -DEXTRA_SYSTEM_LIBRARY_PATH=${EXTRA_SYSTEM_LIBRARY_PATH}" \
    "${BASEDIR}"/../linux-macos/build.sh

export LIBKRIGING_PATH=${PWD}/build/installed
export PATH=${LIBKRIGING_PATH}/bin:${PATH}

cd bindings/R
make uninstall || true
make clean
MAKE_SHARED_LIBS=${MAKE_SHARED_LIBS} make

