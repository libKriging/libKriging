#!/usr/bin/env bash
set -eo pipefail

BASEDIR=$(dirname "$0")
BASEDIR=$(readlink -f "${BASEDIR}")

if [[ "$DEBUG_CI" == "true" ]]; then
    . ${BASEDIR}/loadenv.sh
    echo "PATH=$PATH"

    echo "CMake config: $(command -v cmake)"
    cmake --version | sed 's/^/  /'
    
    if ( command -v octave-cli >/dev/null 2>&1 ); then
      echo "Octave config: $(command -v octave-cli)"
      octave --version | sed 's/^/  /'
    fi

    if ( command -v R >/dev/null 2>&1 ); then
      echo "R config: $(command -v R)"
      R --version | sed 's/^/  /'
    fi

    # Python3 is named python in Windows, but we add a symlink
    if ( command -v python3 >/dev/null 2>&1 ); then
      echo "Python3 config: $(command -v python3)"
      python3 --version 2>&1 | sed 's/^/  /'
    fi

    echo "EXTRA_CMAKE_OPTIONS = ${EXTRA_CMAKE_OPTIONS}"
fi