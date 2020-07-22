#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
    echo "PATH=$PATH"
    #export PATH=/c/Octave/Octave-5.2.0/mingw64/bin/:$PATH
    export PATH=/c/ProgramData/chocolatey/lib/octave.portable/tools/octave-5.2.0-w64/mingw64/bin/:$PATH

    echo "C++ config: $(command -v c++)"
    c++ --version | sed 's/^/  /'

    echo "CMake config: $(command -v cmake)"
    cmake --version | sed 's/^/  /'
    
    export PATH="/c/Program Files/make/make-4.3/bin":${PATH}
    if ( command -v make >/dev/null 2>&1 ); then
      echo "Make config: $(command -v make)"
      make --version | sed 's/^/  /'
    fi

    if ( command -v octave-cli >/dev/null 2>&1 ); then
      echo "Octave config: $(command -v octave-cli)"
      octave-cli --version | sed 's/^/  /'
    fi

    # export PATH=${HOME}/Miniconda3/Scripts:${PATH}
    export PATH="/c/Program Files/R/R-3.6.0/bin":$PATH
    if ( command -v R >/dev/null 2>&1 ); then
      echo "R config: $(command -v R)"
      R --version | sed 's/^/  /'
    fi

    if ( command -v python3 >/dev/null 2>&1 ); then
      echo "Python3 config: $(command -v python3)"
      python3 --version | sed 's/^/  /'
    fi

    echo "EXTRA_CMAKE_OPTIONS = ${EXTRA_CMAKE_OPTIONS}"
fi
