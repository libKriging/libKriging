#!/usr/bin/env bash
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

# Use jemalloc for multi-threaded stability (Linux only)
if [[ "$(uname -s)" == "Linux" && -f /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 ]]; then
  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
  echo "Using jemalloc for thread-safe memory allocation"
fi

# Add MKL library path if MKL is installed
if [[ "$(uname -s)" == "Linux" && -d /opt/intel/oneapi/mkl/latest/lib ]]; then
  export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib:${LD_LIBRARY_PATH}
  echo "Added MKL library path to LD_LIBRARY_PATH"
elif [[ "$(uname -s)" == "Linux" && -d /opt/intel/mkl/lib/intel64 ]]; then
  export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:${LD_LIBRARY_PATH}
  echo "Added MKL library path to LD_LIBRARY_PATH"
fi

cd bindings/R
make test
