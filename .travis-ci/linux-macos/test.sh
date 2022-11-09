#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  CTEST_FLAGS="--verbose --output-on-failure"
  set -x
else
  CTEST_FLAGS=--output-on-failure
fi

cd ${BUILD_DIR:-build}

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
