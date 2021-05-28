#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  export VERBOSE=true
  set -x
fi

# Default configuration when used out of travis-ci
MODE=${MODE:-Debug}
EXTRA_CMAKE_OPTIONS=${EXTRA_CMAKE_OPTIONS:-}
BUILD_TEST=${BUILD_TEST:-true}

export ENABLE_OCTAVE_BINDING=${ENABLE_OCTAVE_BINDING:-auto}
export ENABLE_PYTHON_BINDING=${ENABLE_PYTHON_BINDING:-auto}
export ENABLE_COVERAGE=${ENABLE_COVERAGE:-off}
export ENABLE_MEMCHECK=${ENABLE_MEMCHECK:-off}
export ENABLE_STATIC_ANALYSIS=${ENABLE_STATIC_ANALYSIS:-off}

if [[ -n ${TRAVIS_BUILD_DIR:+x} ]]; then
    cd "${TRAVIS_BUILD_DIR}"
fi

if [[ "$ENABLE_OCTAVE_BINDING" == "on" ]]; then
  EXTRA_CMAKE_OPTIONS="${EXTRA_CMAKE_OPTIONS} -DBUILD_SHARED_LIBS=off"
fi

case "$COMPILER" in 
  gcc*)
    export CXX=g++${COMPILER#gcc} 
    export CC=gcc${COMPILER#gcc}
    ;;
  clang*)
    export CXX=clang++${COMPILER#clang} 
    export CC=clang${COMPILER#clang}
    # initially was only for clang ≥ 7
    # CXXFLAGS="-stdlib=libc++"
    ;;
  "")
    # not defined: use default configuration
    ;;
  *)
    echo "${COMPILER} not supported compiler"
    exit 1
    ;;
esac

mkdir -p build
cd build
cmake \
  -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE="${MODE}" \
  -DENABLE_COVERAGE="${ENABLE_COVERAGE}" \
  -DENABLE_MEMCHECK="${ENABLE_MEMCHECK}" \
  -DENABLE_OCTAVE_BINDING=${ENABLE_OCTAVE_BINDING} \
  -DENABLE_PYTHON_BINDING=${ENABLE_PYTHON_BINDING} \
  -DENABLE_STATIC_ANALYSIS=${ENABLE_STATIC_ANALYSIS} \
  -DUSE_COMPILER_CACHE="${USE_COMPILER_CACHE}" \
  $(eval echo ${EXTRA_CMAKE_OPTIONS}) \
  ..

if [[ "$BUILD_TEST" == "true" ]]; then
    cmake --build . --target install --config "${MODE}"
else
    # faster install target if tests are not required
    cmake --build . --target install.lib --config "${MODE}"
fi
