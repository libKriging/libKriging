#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == true ]]; then
  set -x
fi

if [[ "$DEBUG_CI" == true ]]; then
    echo $PATH
    cmake --version
fi
