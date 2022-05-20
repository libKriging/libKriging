#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

BASEDIR=$(dirname "$0")
BASEDIR=$(cd "$BASEDIR" && pwd -P)
test -f "${BASEDIR}"/loadenv.sh && . "${BASEDIR}"/loadenv.sh 

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

MODE=${MODE:-Release}

BUILD_TEST=false \
    MODE=${MODE} \
    CC=$(R CMD config CC) \
    CXX=$(R CMD config CXX) \
    EXTRA_CMAKE_OPTIONS="-DBUILD_SHARED_LIBS=${MAKE_SHARED_LIBS} ${EXTRA_CMAKE_OPTIONS}" \
    "${BASEDIR}"/../linux-macos/build.sh

export LIBKRIGING_PATH=${PWD}/${BUILD_DIR:-build}/installed

NPROC=1
if ( command -v nproc >/dev/null 2>&1 ); then
  NPROC=$(nproc)
fi

cd bindings/R
make uninstall || true
make clean
MAKEFLAGS=-j${NPROC}
MAKE_SHARED_LIBS=${MAKE_SHARED_LIBS} make
