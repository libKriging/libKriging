#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

BASEDIR=$(dirname "$0")
BASEDIR=$(cd "$BASEDIR" && pwd -P)
test -f "${BASEDIR}"/loadenv.sh && . "${BASEDIR}"/loadenv.sh 

export LIBKRIGING_PATH=${PWD}/${BUILD_DIR:-build}/installed
export PATH=${LIBKRIGING_PATH}/bin:${PATH}

cd bindings/R
make test
