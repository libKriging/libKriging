#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

# Default configuration when used out of travis-ci
if [[ -n ${TRAVIS_BUILD_DIR:+x} ]]; then
    cd "${TRAVIS_BUILD_DIR}"
fi

# MacOS + Shared : OK
# MacOS + Static : there is a bug with dependant libs not found
# Linux + Shared : OK
# Linux + Static : OK
MAKE_SHARED_LIBS=on
if [[ "$(uname -s)" == "Linux" ]]; then
  MAKE_SHARED_LIBS=off # Quick workaround for not found armadillo lib
fi

# to get readlink on MacOS (no effect on Linux)
if [[ -e /usr/local/opt/coreutils/libexec/gnubin/readlink ]]; then
    export PATH=/usr/local/opt/coreutils/libexec/gnubin:$PATH
fi
BASEDIR=$(dirname "$0")
BASEDIR=$(readlink -f "${BASEDIR}")
MODE=${MODE:-Release}

BUILD_TEST=false \
    MODE=${MODE} \
    CC=$(R CMD config CC) \
    CXX=$(R CMD config CXX) \
    EXTRA_CMAKE_OPTIONS="-DBUILD_SHARED_LIBS=${MAKE_SHARED_LIBS} ${EXTRA_CMAKE_OPTIONS}" \
    "${BASEDIR}"/../linux-macos/build.sh

export LIBKRIGING_PATH=${PWD}/build/installed
 
cd bindings/R
make uninstall || true
make clean
MAKEFLAGS=-j$(nproc)
MAKE_SHARED_LIBS=${MAKE_SHARED_LIBS} make
