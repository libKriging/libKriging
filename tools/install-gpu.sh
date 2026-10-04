#!/usr/bin/env bash
# Build and install libKriging from source on THIS machine with every GPU
# backend found here (CUDA, HIP/ROCm, SYCL/oneAPI, Apple Metal:
# ENABLE_GPU_ITERATIVE=AUTO), for the requested bindings. This is the "-gpu"
# variant of every binding; the published binary packages (PyPI wheels, CRAN,
# release archives) are CPU-only. Without any usable GPU toolchain the result
# is an ordinary CPU build.
#
# Usage: tools/install-gpu.sh [--prefix DIR] [--build-dir DIR] [--jobs N] [BINDING...]
#   BINDING: cpp (always built), python, r, octave, matlab, julia
#   Run from a git checkout with submodules, or from the libKriging-gpu
#   source archive attached to each GitHub release.
#   Extra CMake options can be passed through LIBKRIGING_CMAKE_ARGS, e.g.
#   LIBKRIGING_CMAKE_ARGS="-DCMAKE_CUDA_HOST_COMPILER=g++-12".
#
# Python users can instead run `pip install pylibkriging-gpu` (same build,
# driven by pip).
set -eo pipefail

if [[ "$DEBUG_CI" == "true" ]]; then
  set -x
fi

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd -P)
PREFIX="${ROOT_DIR}/build-gpu/installed"
BUILD_DIR="${ROOT_DIR}/build-gpu"
JOBS=$( (command -v nproc >/dev/null && nproc) || (sysctl -n hw.ncpu 2>/dev/null) || echo 2)
BINDINGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) PREFIX=$(mkdir -p "$2" && cd "$2" && pwd -P); shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    cpp|python|r|octave|matlab|julia) BINDINGS+=("$1"); shift ;;
    *) echo "Unknown argument '$1' (see --help)" >&2; exit 1 ;;
  esac
done

case "$(uname -s)" in
  Linux|Darwin) ;;
  *) echo "This script supports Linux and macOS. On Windows, configure with CMake directly:" >&2
     echo "  cmake -DENABLE_GPU_ITERATIVE=AUTO -DCMAKE_GENERATOR_PLATFORM=x64 ..." >&2
     exit 1 ;;
esac

if [ ! -f "${ROOT_DIR}/dependencies/armadillo-code/CMakeLists.txt" ]; then
  echo "Submodules are missing: run 'git submodule update --init --recursive' first" >&2
  exit 1
fi

has() { local b; for b in "${BINDINGS[@]}"; do [ "$b" == "$1" ] && return 0; done; return 1; }
onoff() { if has "$1"; then echo ON; else echo OFF; fi; }

if has octave && has matlab; then
  echo "octave and matlab bindings are mutually exclusive: install them in two runs (different --build-dir)" >&2
  exit 1
fi

# The Python binding is built for the python3 found in PATH (CMake would
# otherwise pick any interpreter it finds), whose modules (pytest included)
# the CMake configuration checks.
if has python; then
  PYTHON_EXE=$(command -v python3) || { echo "python3 not found in PATH" >&2; exit 1; }
  REQ_DIR="${ROOT_DIR}/bindings/Python/pylibkriging"
  if ! "${PYTHON_EXE}" "${REQ_DIR}/check_requirements.py" requirements.txt dev-requirements.txt; then
    echo "Missing Python modules for the python binding: run" >&2
    echo "  ${PYTHON_EXE} -m pip install -r ${REQ_DIR}/requirements.txt -r ${REQ_DIR}/dev-requirements.txt" >&2
    exit 1
  fi
fi

CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_PREFIX="${PREFIX}"
  -DCMAKE_INSTALL_LIBDIR=lib
  -DENABLE_GPU_ITERATIVE=AUTO
  -DENABLE_PYTHON_BINDING="$(onoff python)"
  -DENABLE_OCTAVE_BINDING="$(onoff octave)"
  -DENABLE_MATLAB_BINDING="$(onoff matlab)"
  -DENABLE_JULIA_BINDING="$(onoff julia)"
)
has python && CMAKE_ARGS+=(-DPYTHON_EXECUTABLE="${PYTHON_EXE}")
# The Octave/MATLAB mex and the R package (Linux) link a static libKriging,
# as their release builds do (see tools/r-linux-macos/build.sh).
if has octave || has matlab || { has r && [ "$(uname -s)" == "Linux" ]; }; then
  CMAKE_ARGS+=(-DBUILD_SHARED_LIBS=OFF -DSTATIC_LIB=ON)
  R_SHARED=off
else
  R_SHARED=on
fi
# R packages must be compiled with R's own toolchain.
if has r; then
  command -v R >/dev/null || { echo "R not found in PATH" >&2; exit 1; }
  CMAKE_ARGS+=(-DCMAKE_C_COMPILER="$(R CMD config CC | awk '{print $1}')"
               -DCMAKE_CXX_COMPILER="$(R CMD config CXX | awk '{print $1}')")
fi
# shellcheck disable=SC2206
CMAKE_ARGS+=(${LIBKRIGING_CMAKE_ARGS})

echo "== Configuring libKriging (GPU auto-detection) in ${BUILD_DIR}"
if ! cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" "${CMAKE_ARGS[@]}" > "${BUILD_DIR}.configure.log" 2>&1; then
  tail -n 30 "${BUILD_DIR}.configure.log" >&2
  echo "CMake configuration failed (full log: ${BUILD_DIR}.configure.log)" >&2
  exit 1
fi
grep -E "iterative backend|binding (enabled|available)" "${BUILD_DIR}.configure.log" || true
echo "== Building and installing into ${PREFIX}"
cmake --build "${BUILD_DIR}" --target install --parallel "${JOBS}"

if has r; then
  echo "== Installing the R package (rlibkriging)"
  (
    cd "${ROOT_DIR}/bindings/R"
    Rscript -e "Rcpp::compileAttributes(pkgdir = 'rlibkriging')"
    LIBKRIGING_PATH="${PREFIX}" MAKE_SHARED_LIBS="${R_SHARED}" R CMD INSTALL --no-multiarch rlibkriging
  )
fi

echo
echo "== Done. GPU backends:"
grep -E "iterative backend" "${BUILD_DIR}.configure.log" | sed 's/^-- /   /'
echo
echo "Use it:"
echo "  C++:    headers/libs in ${PREFIX} (CMAKE_PREFIX_PATH=${PREFIX})"
has python && echo "  Python: export PYTHONPATH=${PREFIX}/bindings/Python LD_LIBRARY_PATH=${PREFIX}/lib:\$LD_LIBRARY_PATH; python -c 'import pylibkriging as lk; print(lk.gpu_backend())'"
has r      && echo "  R:      library(rlibkriging); gpu_backend()"
has octave && echo "  Octave: addpath('${PREFIX}/bindings/Octave'); Gpu.backend()"
has matlab && echo "  MATLAB: addpath('${PREFIX}/bindings/Matlab'); Gpu.backend()"
has julia  && echo "  Julia:  ENV[\"JLIBKRIGING_LIB_PATH\"]=\"${PREFIX}/bindings/Julia/libkriging_c.$( [ "$(uname -s)" == Darwin ] && echo dylib || echo so)\"; using jlibkriging; gpu_backend()"
exit 0
