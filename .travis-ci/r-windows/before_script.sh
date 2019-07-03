#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == true ]]; then
  set -x
fi

if [[ "$DEBUG_CI" == true ]]; then
    echo $PATH
    c++ --version
    cmake --version
    export PATH="/c/Program Files/make/make-4.2.1/bin":${PATH}
    make --version
    export PATH=${HOME}/Miniconda3/Scripts:${PATH}
    R --version
fi