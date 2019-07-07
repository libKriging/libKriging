#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == true ]]; then
  set -x
fi

# Default configuration when used out of travis-ci
if [[ -n ${TRAVIS_BUILD_DIR:+x} ]]; then
echo
    cd ${TRAVIS_BUILD_DIR}
fi

# make
export PATH="/c/Program Files/make/make-4.2.1/bin":${PATH}

# R shortcut (and tools with absolute path) by Chocolatey
export PATH="/c/Program Files/R/R-3.6.0/bin":$PATH
# Required to disambiguate with Chocolatey GCC installation
export PATH="/c/Rtools/mingw_64/bin":$PATH

## R shortcut (and tools) by Anaconda
#export PATH=${HOME}/Miniconda3/Scripts:${PATH}
#export PATH=$HOME/Miniconda3/Rtools/mingw_64/bin:$PATH

# OpenBLAS installation
export EXTRA_SYSTEM_LIBRARY_PATH=${HOME}/Miniconda3/Library/lib
# and libomp.dll, flang.dll, flangrti.dll, openblas.dll
export PATH=${HOME}/Miniconda3/Library/bin:$PATH

cd bindings/R
make veryclean
make
make test