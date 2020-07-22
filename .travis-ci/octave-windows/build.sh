#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

# Default configuration when used out of travis-ci
if [[ -n ${TRAVIS_BUILD_DIR:+x} ]]; then
    cd "${TRAVIS_BUILD_DIR}"
fi

# make
export PATH="/c/Program Files/make/make-4.3/bin":${PATH}

# Octave shortcut (and tools with absolute path) by Chocolatey
export PATH=/c/Octave/Octave-5.2.0/mingw64/bin/:$PATH

# OpenBLAS installation comes from Octave installation

BASEDIR=$(dirname "$0")
BASEDIR=$(readlink -f "${BASEDIR}")

CC=gcc CXX=g++ \
    EXTRA_CMAKE_OPTIONS="-DBUILD_SHARED_LIBS=${MAKE_SHARED_LIBS} -DEXTRA_SYSTEM_LIBRARY_PATH=${EXTRA_SYSTEM_LIBRARY_PATH}" \
    "${BASEDIR}"/../linux-macos/build.sh

