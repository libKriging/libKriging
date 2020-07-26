#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

BASEDIR=$(dirname "$0")
BASEDIR=$(readlink -f "${BASEDIR}")

# Default configuration when used out of travis-ci
if [[ -n ${TRAVIS_BUILD_DIR:+x} ]]; then
    cd "${TRAVIS_BUILD_DIR}"
fi

. ${BASEDIR}/loadenv.sh

# OpenBLAS installation
export EXTRA_SYSTEM_LIBRARY_PATH=${HOME}/Miniconda3/Library/lib

# Windows + Shared : OK
# Windows + Static : OK
MAKE_SHARED_LIBS=on

BUILD_TEST=false \
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

