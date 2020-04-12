#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == true ]]; then
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

# to get readlink on MacOS (no effect on Linux)
if [[ -e /usr/local/opt/coreutils/libexec/gnubin/readlink ]]; then
    export PATH=/usr/local/opt/coreutils/libexec/gnubin:$PATH
fi
BASEDIR=$(dirname "$0")
BASEDIR=$(readlink -f "${BASEDIR}")

BUILD_TEST=false \
    MODE=Release \
    CC=$(R CMD config CC) \
    CXX=$(R CMD config CXX) \
    EXTRA_CMAKE_OPTIONS="-DBUILD_SHARED_LIBS=${MAKE_SHARED_LIBS} ${EXTRA_CMAKE_OPTIONS}" \
    "${BASEDIR}"/../linux-macos/build.sh

export LIBKRIGING_PATH=${PWD}/build/installed

cd bindings/R
make uninstall || true
make clean
MAKE_SHARED_LIBS=${MAKE_SHARED_LIBS} make
