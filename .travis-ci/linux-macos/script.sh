#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == true ]]; then
  set -x
fi

# Default configuration when used out of travis-ci
MODE=${MODE:-Debug}
if [[ -n ${TRAVIS_BUILD_DIR:+x} ]]; then
echo
    cd ${TRAVIS_BUILD_DIR}
fi

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=${MODE}
cmake --build .

if [[ "$MODE" == "Coverage" ]]; then
  cmake --build . --target coverage --config Coverage
else
  ctest -C ${MODE} # --verbose
fi

cmake --build . --target install --config ${MODE}
# Show installation directory content
find installed
# Cleanup compiled libs to check right path finding
rm -fr src/lib
# add library directory search PATH for executables
export LD_LIBRARY_PATH=$PWD/installed/lib:${LD_LIBRARY_PATH}
ctest -C ${MODE}
