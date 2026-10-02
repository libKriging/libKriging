#!/bin/bash
# Run bench/gpu/bench_gpu.py (libKriging vs GPyTorch) on this machine.
#
# Wires up LD_LIBRARY_PATH/PYTHONPATH for this repo's pylibkriging build,
# (re)builds the _pylibkriging extension if it's missing or stale relative
# to the C++ sources, and checks the python env has torch/gpytorch/numpy
# before running.
#
# Usage: ./bench/bench_gpu.sh [bench_gpu.py options...]
#   e.g. ./bench/bench_gpu.sh --sizes 250,500 --backends chol,iter-cuda,iter-omp
#   e.g. (macOS/Metal) ./bench/bench_gpu.sh --sizes 250,500 --backends chol,iter-metal,gpt-cpu
#
# Env overrides:
#   BUILD_DIR    : build dir to use/create for pylibkriging
#                  (default: ./build_cuda_iterative if it already has
#                  ENABLE_CUDA_ITERATIVE=ON; else, on Darwin,
#                  ./build_metal_iterative; else ./build_dev3)
#   PYTHON_BIN   : python interpreter to use (default: /home/richet/.local/bin/python3.12
#                  if present -- the repo-root venv/.venv on that host are
#                  stale symlinks, see memory env-binding-toolchains --
#                  otherwise the first `python3` on PATH)
#   METAL_CPP_DIR: path to the metal-cpp headers (macOS only; default:
#                  <repo>/.deps/metal-cpp if present). Only used when a NEW
#                  build dir is configured on Darwin, to pass
#                  -DENABLE_METAL_ITERATIVE=ON -DMETAL_CPP_DIR=...
#   SKIP_BUILD   : set to 1 to skip the freshness check/rebuild entirely
#   CUDA_VISIBLE_DEVICES : forwarded as-is (select GPU on a multi-GPU host)
#
# See bench/gpu/README.md for what the sweep does and how to read the output.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

IS_DARWIN=0
[ "$(uname -s)" = "Darwin" ] && IS_DARWIN=1

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "/home/richet/.local/bin/python3.12" ]; then
        PYTHON_BIN="/home/richet/.local/bin/python3.12"
    else
        PYTHON_BIN="$(command -v python3 || true)"
    fi
fi

if [ -z "${BUILD_DIR:-}" ]; then
    if [ -f "${PROJECT_ROOT}/build_cuda_iterative/CMakeCache.txt" ] \
        && grep -q "^ENABLE_CUDA_ITERATIVE:BOOL=ON" "${PROJECT_ROOT}/build_cuda_iterative/CMakeCache.txt" 2>/dev/null; then
        BUILD_DIR="${PROJECT_ROOT}/build_cuda_iterative"
    elif [ "$IS_DARWIN" = 1 ]; then
        BUILD_DIR="${PROJECT_ROOT}/build_metal_iterative"
    else
        BUILD_DIR="${PROJECT_ROOT}/build_dev3"
    fi
fi

if [ -z "${METAL_CPP_DIR:-}" ] && [ -d "${PROJECT_ROOT}/.deps/metal-cpp" ]; then
    METAL_CPP_DIR="${PROJECT_ROOT}/.deps/metal-cpp"
fi

# nproc is Linux-only; macOS has neither it nor coreutils by default.
if command -v nproc > /dev/null 2>&1; then
    NPROC="$(nproc)"
elif [ "$IS_DARWIN" = 1 ]; then
    NPROC="$(sysctl -n hw.ncpu)"
else
    NPROC="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
fi

if [ -z "$PYTHON_BIN" ] || [ ! -x "$PYTHON_BIN" ]; then
    echo "Error: PYTHON_BIN not found/executable: ${PYTHON_BIN:-<empty>}" >&2
    exit 1
fi

PYLIBKRIGING_DIR="${BUILD_DIR}/bindings/Python/pylibkriging"

# --- 1. (re)configure + (re)build pylibkriging if needed -------------------
if [ "${SKIP_BUILD:-0}" != "1" ]; then
    if [ ! -d "$BUILD_DIR" ]; then
        echo "No build dir at $BUILD_DIR -- configuring a new one..."
        EXTRA_CMAKE_ARGS=()
        if [ "$IS_DARWIN" = 1 ]; then
            if [ -n "${METAL_CPP_DIR:-}" ]; then
                EXTRA_CMAKE_ARGS+=(-DENABLE_METAL_ITERATIVE=ON -DMETAL_CPP_DIR="$METAL_CPP_DIR")
            else
                echo "Note: no METAL_CPP_DIR found -- configuring without -DENABLE_METAL_ITERATIVE" \
                     "(iter-metal backend won't be available; set METAL_CPP_DIR to enable it)." >&2
            fi
        fi
        cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" \
            -DCMAKE_BUILD_TYPE=Release \
            -DENABLE_PYTHON_BINDING=ON \
            -DPYTHON_EXECUTABLE="$PYTHON_BIN" \
            "${EXTRA_CMAKE_ARGS[@]}"
    fi

    SO_PATH="$(compgen -G "${PYLIBKRIGING_DIR}/_pylibkriging*.so" || true)"

    NEEDS_BUILD=0
    if [ -z "$SO_PATH" ]; then
        NEEDS_BUILD=1
    else
        # Rebuild if any C++ source under src/ or the Python binding is newer
        # than the compiled extension (cheap staleness check; cmake/make would
        # also catch this, but this avoids paying the "nothing to do" cmake
        # invocation cost on every single run).
        NEWEST_SRC="$(find "${PROJECT_ROOT}/src" "${PROJECT_ROOT}/bindings/Python/pylibkriging" \
            \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name 'CMakeLists.txt' \) \
            -newer "$SO_PATH" -print -quit 2>/dev/null || true)"
        if [ -n "$NEWEST_SRC" ]; then
            echo "Source newer than built _pylibkriging ($NEWEST_SRC) -- rebuilding..."
            NEEDS_BUILD=1
        fi
    fi

    if [ "$NEEDS_BUILD" = "1" ]; then
        echo "Building _pylibkriging in $BUILD_DIR ..."
        cmake --build "$BUILD_DIR" --target _pylibkriging -j"$NPROC"
    fi
fi

if ! compgen -G "${PYLIBKRIGING_DIR}/_pylibkriging*.so" > /dev/null; then
    echo "Error: no _pylibkriging*.so found under $PYLIBKRIGING_DIR after build." >&2
    echo "Configure/build it manually (target: _pylibkriging, in $BUILD_DIR) and re-run." >&2
    exit 1
fi

export LD_LIBRARY_PATH="${BUILD_DIR}/src/lib:${BUILD_DIR}/dependencies/armadillo-code:${BUILD_DIR}/dependencies/lbfgsb_cpp${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [ "$IS_DARWIN" = 1 ]; then
    # macOS's dynamic linker uses DYLD_LIBRARY_PATH, not LD_LIBRARY_PATH.
    export DYLD_LIBRARY_PATH="${BUILD_DIR}/src/lib:${BUILD_DIR}/dependencies/armadillo-code:${BUILD_DIR}/dependencies/lbfgsb_cpp${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi
export PYTHONPATH="${PYLIBKRIGING_DIR}:${PROJECT_ROOT}/bindings/Python/pylibkriging/src${PYTHONPATH:+:$PYTHONPATH}"

# --- 2. sanity-check the python env -----------------------------------------
ENV_CHECK="$("$PYTHON_BIN" - <<'PYEOF' 2>&1
import sys
missing = []
for mod in ("numpy", "torch", "gpytorch"):
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
try:
    import pylibkriging  # noqa: F401
except ImportError as e:
    print(f"PYLIBKRIGING_IMPORT_ERROR: {e}")
    sys.exit(2)
if missing:
    print("MISSING: " + ",".join(missing))
    sys.exit(1)
print("OK")
PYEOF
)" && ENV_STATUS=0 || ENV_STATUS=$?

if [ "$ENV_STATUS" = 2 ]; then
    echo "Error: built _pylibkriging could not be imported by $PYTHON_BIN:" >&2
    echo "  $ENV_CHECK" >&2
    echo "(check PYTHON_EXECUTABLE used to configure $BUILD_DIR matches PYTHON_BIN)" >&2
    exit 1
elif [ "$ENV_STATUS" = 1 ]; then
    MISSING="${ENV_CHECK#MISSING: }"
    echo "Error: $PYTHON_BIN is missing: $MISSING" >&2
    echo "Install with: $PYTHON_BIN -m pip install ${MISSING//,/ }" >&2
    echo "(not done automatically -- torch's CUDA build depends on your GPU/driver, pick the right index-url)" >&2
    exit 1
elif [ "$ENV_STATUS" != 0 ]; then
    echo "Error: python env check failed unexpectedly:" >&2
    echo "  $ENV_CHECK" >&2
    exit 1
fi

echo "Using BUILD_DIR=$BUILD_DIR"
echo "Using PYTHON_BIN=$PYTHON_BIN"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

exec "$PYTHON_BIN" "${PROJECT_ROOT}/bench/gpu/bench_gpu.py" "$@"
