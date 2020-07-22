#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
    # export PATH=${HOME}/Miniconda3:$PATH
    export PATH=/c/Python37:$PATH
    #export PATH=/c/Octave/Octave-5.2.0/mingw64/bin/:$PATH
    export PATH=/c/ProgramData/chocolatey/lib/octave.portable/tools/octave-5.2.0-w64/mingw64/bin/:$PATH

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

    # Python3 is named python in Windows 
    if ( command -v python >/dev/null 2>&1 ); then
      echo "Python config: $(command -v python)"
      python --version | sed 's/^/  /'
    fi

    echo "EXTRA_CMAKE_OPTIONS = ${EXTRA_CMAKE_OPTIONS}"
fi