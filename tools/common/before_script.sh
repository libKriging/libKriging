#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
    echo "------------------------------------"

    echo "$(uname -o) system running on $(uname -m) processors"

    echo "------------------------------------"

    BASEDIR=$(dirname "$0")
    if [ -e "${BASEDIR}"/../${BUILD_NAME}/loadenv.sh ]; then
      set -x
      . "${BASEDIR}"/../${BUILD_NAME}/loadenv.sh
      set +x
    else
      echo "No custom loadenv.sh"
    fi
    echo "PATH=$PATH"

    echo "------------------------------------"

    if ( command -v c++ >/dev/null 2>&1 ); then
      echo "C++ config: $(command -v c++)"
      c++ --version 2>&1 | sed 's/^/  /'
    else
      echo "No C++ command found"
    fi

    echo "------------------------------------"

    if ( command -v clang-format >/dev/null 2>&1 ); then
      echo "clang-format config: $(command -v clang-format)"
      clang-format --version 2>&1 | sed 's/^/  /'
    else
      echo "No clang-format command found"
    fi

    echo "------------------------------------"

    if ( command -v cmake >/dev/null 2>&1 ); then
      echo "CMake config: $(command -v cmake)"
      cmake --version 2>&1 | sed 's/^/  /'
    else
      echo "No cmake command found"
    fi
          
    echo "------------------------------------"

    if ( command -v make >/dev/null 2>&1 ); then
      echo "Make config: $(command -v make)"
      make --version 2>&1 | sed 's/^/  /'
    else
      echo "No make command found"
    fi

    echo "------------------------------------"

    if ( command -v octave-config >/dev/null 2>&1 ); then
      echo "Octave config: $(command -v octave-config)"
      octave-config --version 2>&1 | sed 's/^/  /'
    else
      echo "No octave-config command found"
    fi

    echo "------------------------------------"

    if ( command -v matlab >/dev/null 2>&1 ); then
      echo "Matlab config: $(command -v matlab)"
      # Read the version without starting MATLAB: outside the matlab-actions/run-*
      # actions, MATLAB is not licensed on GitHub runners (see tools/matlab/run_mex_tests.m)
      MATLAB_ROOT=$(dirname "$(dirname "$(readlink -f "$(command -v matlab)")")")
      sed -n 's:.*<\(version\|release\)>\(.*\)</.*:  \1 \2:p' "${MATLAB_ROOT}/VersionInfo.xml" 2>/dev/null || echo "  (no VersionInfo.xml in ${MATLAB_ROOT})"
    else
      echo "No matlab command found"
    fi

    echo "------------------------------------"
    if ( command -v R >/dev/null 2>&1 ); then
      echo "R config: $(command -v R)"
      R --version 2>&1 | sed 's/^/  /'
    else
      echo "No R command found"
    fi

    echo "------------------------------------"

    # Python3 is named python in Windows, but we add a symlink
    ROOT_DIR=$(git rev-parse --show-toplevel)
    if [[ -f "${ROOT_DIR}"/venv/bin/activate ]]; then
      echo "Loading virtual environment from ${ROOT_DIR}/venv"
      . "${ROOT_DIR}"/venv/bin/activate
    fi

    if ( command -v python3 >/dev/null 2>&1 ); then
      echo "Python3 config: $(command -v python3)"
      {
        python3 --version 2>&1 | sed 's/^/  /'
      } || {
        echo "Cannot execute python3 --version"
      }
    else
      echo "No python3 command found"
    fi

    echo "------------------------------------"

    echo "EXTRA_CMAKE_OPTIONS = ${EXTRA_CMAKE_OPTIONS}"

    echo "------------------------------------"

fi
