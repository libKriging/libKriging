#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

if [[ "$MODE" == "Coverage" ]]; then
    gem install coveralls-lcov
fi

if ( command -v python3 >/dev/null 2>&1 && 
     python3 -m pip --version >/dev/null 2>&1 ); then
  # python3 -m pip install pip # --upgrade # --progress-bar off
  python3 -m pip install pytest numpy # --upgrade # --progress-bar off
fi