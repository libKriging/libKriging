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

export LIBKRIGING_PATH=${PWD}/${BUILD_DIR:-build}/installed
export PATH=${LIBKRIGING_PATH}/bin:${PATH}

cd bindings/R
make test
