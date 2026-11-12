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

# ThreadSanitizer + GCC/libgomp: libgomp is not TSan-instrumented and keeps a
# pool of parked worker threads across parallel regions, so TSan cannot see the
# implicit end-of-parallel barrier and reports false races between a worker's
# write inside an omp region (e.g. filling dX in LinearAlgebra::compute_dX) and
# a later read on the main thread (e.g. max(abs(...)) in fit_setup_impl). These
# are spurious (the non-TSan parallel builds pass with correct results).
# Run this job with OMP_NUM_THREADS=1 so libgomp executes parallel regions
# inline (no worker threads spawned) -> no OpenMP false positives, while TSan
# still instruments the code and would report any genuine non-OpenMP race.
# The .tsan-suppressions file is kept as a documented fallback and is loaded
# too. Real multi-threaded OpenMP race detection would require building this
# job with Clang + the TSan-aware OpenMP runtime (libarcher).
if [[ "$(printf %s "${SANITIZE:-}" | tr "[:upper:]" "[:lower:]")" == "thread" ]]; then
  export OMP_NUM_THREADS=1
  export TSAN_OPTIONS="suppressions=${ROOT_DIR}/.tsan-suppressions${TSAN_OPTIONS:+ ${TSAN_OPTIONS}}"
  echo "TSan job: OMP_NUM_THREADS=${OMP_NUM_THREADS}, TSAN_OPTIONS=${TSAN_OPTIONS}"
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
    rc=0
    ctest -C "${MODE}" ${CTEST_FLAGS} -T memcheck || rc=$?
    # The records behind the "Defects" counts of the summary (which only counts
    # them), printed whether the tests pass or not: leak records of every kind
    # CTest counts and valgrind errors, each up to the blank line ending it.
    defects=0
    for f in Testing/Temporary/MemoryChecker.*.log; do
      if grep -qE "blocks are (definitely lost|possibly lost|still reachable) in loss record|^==[0-9]+== (Invalid |Conditional jump|Use of uninitialised|Syscall param|Mismatched |Source and destination overlap)" "$f"; then
        echo "==== $f ($(grep -m1 -o 'Command: .*' "$f" | cut -c1-200))"
        awk '/^==[0-9]+== .*(blocks are (definitely lost|possibly lost|still reachable) in loss record|Invalid |Conditional jump|Use of uninitialised|Syscall param|Mismatched |Source and destination overlap)/ {p=1}
             p {print}
             /^==[0-9]+== $/ {p=0}' "$f"
        defects=$((defects + 1))
      fi
    done
    [[ $rc -eq 0 ]] || exit $rc
    # Zero threshold: every record of the runtime and of the test harness is
    # either fixed or suppressed with its reason in .valgrind-suppressions; a
    # new one (a leak, or a library upgrade bringing a benign one) fails the
    # job until it is fixed or classified there.
    if [[ $defects -gt 0 ]]; then
      echo "Memcheck: records in ${defects} test log(s) (printed above); fix them or add a commented suppression to .valgrind-suppressions"
      exit 1
    fi
else
    CTEST_EXCLUDE_FLAGS=()
    if [[ -n "${CTEST_EXCLUDE}" ]]; then
      CTEST_EXCLUDE_FLAGS=(-E "${CTEST_EXCLUDE}")
    fi

    ctest -C "${MODE}" ${CTEST_FLAGS} "${CTEST_EXCLUDE_FLAGS[@]}"

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

    ctest -C "${MODE}" ${CTEST_FLAGS} "${CTEST_EXCLUDE_FLAGS[@]}"
fi
