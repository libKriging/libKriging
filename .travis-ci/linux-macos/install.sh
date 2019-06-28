#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == true ]]; then
  set -x
fi

if [[ "$MODE" == "Coverage" ]]; then
    gem install coveralls-lcov
fi
