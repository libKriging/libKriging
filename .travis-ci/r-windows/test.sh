#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

# Default configuration when used out of travis-ci
if [[ -n ${TRAVIS_BUILD_DIR:+x} ]]; then
    cd "${TRAVIS_BUILD_DIR}"
fi

export PATH="/c/Program Files/R/R-3.6.0/bin":$PATH
export PATH=${HOME}/Miniconda3/Library/bin:$PATH
export LIBKRIGING_PATH=${PWD}/build/installed
export PATH=${LIBKRIGING_PATH}/bin:${PATH}

cd bindings/R
make test
