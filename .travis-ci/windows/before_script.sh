#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == true ]]; then
    echo "PATH=$PATH"

    echo "CMake config: $(command -v cmake)"
    cmake --version
    
    if ( command -v octave >/dev/null 2>&1 ); then
      echo "Octave config: $(command -v octave)"
      octave --version
    fi

    if ( command -v R >/dev/null 2>&1 ); then
      echo "R config: $(command -v R)"
      R --version
    fi

    if ( command -v python3 >/dev/null 2>&1 ); then
      echo "Python3 config: $(command -v python3)"
      python3 --version
    fi

    echo "EXTRA_CMAKE_OPTIONS = ${EXTRA_CMAKE_OPTIONS}"
fi