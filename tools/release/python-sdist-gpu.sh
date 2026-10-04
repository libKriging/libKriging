#!/usr/bin/env bash
# Build the `pylibkriging-gpu` source distribution (dist/pylibkriging_gpu-<version>.tar.gz).
# Installing it (`pip install pylibkriging-gpu`) compiles libKriging on the
# installing machine with every GPU backend found there (CUDA, HIP, SYCL,
# Metal: ENABLE_GPU_ITERATIVE=AUTO), falling back to CPU when there is none.
# The default `pylibkriging` package stays the CPU-only binary wheels.
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

ROOT_DIR=$(cd "$(dirname "$0")/../.." && pwd -P)
cd "${ROOT_DIR}"

if [ ! -f dependencies/armadillo-code/CMakeLists.txt ]; then
  echo "Submodules are missing: run 'git submodule update --init --recursive' first" >&2
  exit 1
fi

VARIANT_FILE=bindings/Python/pylibkriging/VARIANT
trap 'rm -f "${VARIANT_FILE}"; rm -rf pylibkriging_gpu.egg-info' EXIT
echo gpu > "${VARIANT_FILE}"
LIBKRIGING_PY_VARIANT=gpu python3 setup.py sdist
ls -l dist/pylibkriging_gpu-*.tar.gz
