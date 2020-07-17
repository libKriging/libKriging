#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == true ]]; then
  export VERBOSE=true
  set -x
fi

# Default configuration when used out of travis-ci
MODE=${MODE:-Debug}
EXTRA_CMAKE_OPTIONS=${EXTRA_CMAKE_OPTIONS:-}
BUILD_TEST=${BUILD_TEST:-true}

if [[ -n ${TRAVIS_BUILD_DIR:+x} ]]; then
    cd "${TRAVIS_BUILD_DIR}"
fi

mkdir -p build
cd build
cmake \
  -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE="${MODE}" \
  $(eval echo ${EXTRA_CMAKE_OPTIONS}) \
  ..

if [[ "$BUILD_TEST" == "true" ]]; then
    cmake --build . --target install --config "${MODE}"
else
    # faster install target if tests are not required
    cmake --build . --target install.lib --config "${MODE}"
fi
