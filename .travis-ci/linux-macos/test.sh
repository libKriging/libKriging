#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  CTEST_FLAGS="--verbose --output-on-failure"
  set -x
else
  CTEST_FLAGS=--output-on-failure
fi

ROOT_DIR=$(git rev-parse --show-toplevel)
if [[ "$ENABLE_PYTHON_BINDING" == "on" && ! -d "$VIRTUAL_ENV" ]]; then
  echo "Loading virtual environment from ${ROOT_DIR}/venv"
  . "${ROOT_DIR}"/venv/bin/activate
fi

cd "${BUILD_DIR:-build}"

# jemalloc is statically linked into libKriging (no need for LD_PRELOAD)

# Add MKL library path if MKL is installed
if [[ "$(uname -s)" == "Linux" && -d /opt/intel/oneapi/mkl/latest/lib ]]; then
  export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib:${LD_LIBRARY_PATH}
  echo "Added MKL library path to LD_LIBRARY_PATH"
elif [[ "$(uname -s)" == "Linux" && -d /opt/intel/mkl/lib/intel64 ]]; then
  export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:${LD_LIBRARY_PATH}
  echo "Added MKL library path to LD_LIBRARY_PATH"
fi

# Fix MATLAB libstdc++ compatibility issue
# MATLAB bundles an old libstdc++ that may conflict with system libraries
# Force MATLAB to use system libstdc++ instead
if [[ "${ENABLE_MATLAB_BINDING}" == "on" && "$(uname -s)" == "Linux" ]]; then
  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
  echo "Forcing MATLAB to use system libstdc++.so.6 via LD_PRELOAD"
fi

if [[ "$ENABLE_COVERAGE" == "on" ]]; then
    cmake --build . --target coverage --config "${MODE}"
elif [[ "$ENABLE_MEMCHECK" == "on" ]]; then
    ctest -C "${MODE}" ${CTEST_FLAGS} -T memcheck
else
    ctest -C "${MODE}" ${CTEST_FLAGS}

    # Cleanup compiled libs to check right path finding
    rm -fr src/lib
        
    # add library directory search PATH for executables
    case "$(uname -s)" in
     Darwin)
      export DYLD_LIBRARY_PATH=$PWD/installed/lib:${DYLD_LIBRARY_PATH}
       ;;
    
     *)
      export LD_LIBRARY_PATH=$PWD/installed/lib:${LD_LIBRARY_PATH}
       ;;
    esac

    ctest -C "${MODE}" ${CTEST_FLAGS}
fi
