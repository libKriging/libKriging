#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == true ]]; then
  set -x
fi

if [[ "$DEBUG_CI" == true ]]; then
    echo "$PATH"
    c++ --version
    cmake --version
    
    if ( command -v octave >/dev/null 2>&1 ); then
      octave --version
    fi

    if ( command -v R >/dev/null 2>&1 ); then
      R --version
    fi

    echo "EXTRA_CMAKE_OPTIONS = ${EXTRA_CMAKE_OPTIONS}"
fi