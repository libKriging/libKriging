#!/usr/bin/env bash
# Compile a diagnostic harness against an existing libKriging build tree.
# Usage: LK_SRC=/path/to/libKriging LK_BUILD=$LK_SRC/build ./build.sh ws   (-> ./ws)
# Reuses the exact compile flags/defines/includes of the KrigingPredictIterativeTest target.
set -euo pipefail
name=$1
LK_SRC=${LK_SRC:-$(cd "$(dirname "$0")/../../../.." && pwd)}
LK_BUILD=${LK_BUILD:-$LK_SRC/build}
FM=$LK_BUILD/tests/CMakeFiles/KrigingPredictIterativeTest.dir/flags.make
D=$(grep CXX_DEFINES "$FM" | cut -d= -f2-)
I=$(grep CXX_INCLUDES "$FM" | cut -d= -f2-)
F=$(grep CXX_FLAGS "$FM" | cut -d= -f2-)
ARMA=$LK_BUILD/dependencies/armadillo-code
g++ $F $D $I "$(dirname "$0")/$name.cpp" -o "$name" \
  -L"$LK_BUILD/src/lib" -lKriging -Wl,-rpath,"$LK_BUILD/src/lib" \
  "$ARMA"/libarmadillo.so -Wl,-rpath,"$ARMA" -fopenmp
