#!/bin/bash

# Benchmark script for NoiseKriging BFGS multistart optimization
# Tests parallelism with different numbers of starting points

set -e

# Default parameters
DATASET_SIZE=${1:-500}
N_RUNS=3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║       NoiseKriging Multistart Benchmark (n=$DATASET_SIZE)               ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory not found at $BUILD_DIR"
    exit 1
fi

# Check library exists
if [ ! -f "$BUILD_DIR/src/lib/libKriging.a" ] && [ ! -f "$BUILD_DIR/src/lib/libKriging.so" ]; then
    echo "Error: libKriging library not found"
    exit 1
fi

# Compile benchmark
echo "Compiling NoiseKriging benchmark..."

JEMALLOC_ARG=""
if [ -f "$BUILD_DIR/jemalloc-install/lib/libjemalloc.a" ]; then
    JEMALLOC_ARG="$BUILD_DIR/jemalloc-install/lib/libjemalloc.a"
fi

g++ -std=c++17 -O2 -fopenmp -DARMA_32BIT_WORD \
    "$SCRIPT_DIR/bench_noisekriging_multistart.cpp" \
    -o "$BUILD_DIR/bench_noisekriging_multistart" \
    -I"$PROJECT_ROOT/src/lib/include" \
    -I"$BUILD_DIR/dependencies/armadillo-code/tmp/include" \
    -I"$BUILD_DIR/dependencies/lbfgsb_cpp-prefix/src/lbfgsb_cpp/include" \
    -I"$BUILD_DIR/src/lib" \
    "$BUILD_DIR/src/lib/libKriging.a" \
    "$BUILD_DIR/dependencies/armadillo-code/libarmadillo.a" \
    "$BUILD_DIR/dependencies/lbfgsb_cpp/liblbfgsb_cpp.a" \
    $JEMALLOC_ARG \
    -lblas -llapack -lpthread -ldl -lgfortran

if [ $? -ne 0 ]; then
    echo "Error: Compilation failed"
    exit 1
fi

echo "✓ Compilation successful"
echo ""

# Run benchmark
echo "Running benchmark..."
echo ""

export LD_LIBRARY_PATH="$BUILD_DIR/src/lib:$LD_LIBRARY_PATH"

"$BUILD_DIR/bench_noisekriging_multistart" "$DATASET_SIZE" "$N_RUNS"

echo ""
echo "Benchmark complete!"
