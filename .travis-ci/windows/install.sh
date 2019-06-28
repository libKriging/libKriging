#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == true ]]; then
  set -x
fi

if [[ "$MODE" == "Coverage" ]]; then
    echo "Coverage not supported for Windows"
    travis_terminate 1
fi
