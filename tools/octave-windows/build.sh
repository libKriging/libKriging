#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

BASEDIR=$(dirname "$0")
BASEDIR=$(cd "$BASEDIR" && pwd -P)
test -f "${BASEDIR}"/loadenv.sh && . "${BASEDIR}"/loadenv.sh 

STATIC_LIB=on
MAKE_SHARED_LIBS=off

CC=gcc CXX=g++ \
    EXTRA_CMAKE_OPTIONS="-DBUILD_SHARED_LIBS=${MAKE_SHARED_LIBS} -DSTATIC_LIB=${STATIC_LIB} -DEXTRA_SYSTEM_LIBRARY_PATH=${EXTRA_SYSTEM_LIBRARY_PATH}" \
    "${BASEDIR}"/../linux-macos/build.sh

