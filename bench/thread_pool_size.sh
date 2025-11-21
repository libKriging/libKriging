#!/bin/bash

# Benchmark script for testing different thread pool sizes with BFGS20
# Tests the impact of thread_pool_size parameter on parallel optimization performance
#
# Usage:
#   ./bench/thread_pool_size.sh [dataset_size]
#
# Example:
#   ./bench/thread_pool_size.sh 500
#
# Environment variables:
#   LK_THREAD_POOL_SIZE - Override thread pool size (default: auto-detect)
#   LK_LOG_LEVEL - Set logging level (default: 0)

set -e

# Default parameters
DATASET_SIZE=${1:-500}
N_RUNS=10
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
TEMP_DIR=$(mktemp -d)

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║   Thread Pool Size Benchmark - BFGS20 (n=$DATASET_SIZE)                 ║"
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

# Detect number of CPUs
N_CPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "4")
echo "Detected CPUs: $N_CPU"
echo ""

# Detect OpenMP threading in Armadillo
echo "Armadillo OpenMP support:"
OMP_THREADS=$(grep -i openmp_threads "$BUILD_DIR/dependencies/armadillo-code/tmp/include/armadillo_bits/config.hpp" | grep -oP '(?<=#define ARMA_OPENMP_THREADS )[0-9]+')
if [ -n "$OMP_THREADS" ]; then
    echo " - Default OpenMP threads: $OMP_THREADS"
else
    echo " - Default OpenMP threads: not set"
fi  
echo ""

# Create benchmark source code
cat > "$TEMP_DIR/bench_thread_pool.cpp" << 'EOF'
#include "libKriging/Kriging.hpp"
#include "libKriging/Optim.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

void generate_branin_data(arma::mat& X, arma::vec& y, int n) {
  arma::arma_rng::set_seed(123);
  X = arma::randu(n, 2);
  X.col(0) = X.col(0) * 15.0 - 5.0;
  X.col(1) = X.col(1) * 15.0;
  y.set_size(n);
  for (int i = 0; i < n; i++) {
    double x1 = X(i, 0), x2 = X(i, 1);
    double a = 1.0, b = 5.1 / (4.0 * M_PI * M_PI), c = 5.0 / M_PI;
    double r = 6.0, s = 10.0, t = 1.0 / (8.0 * M_PI);
    y(i) = a * std::pow(x2 - b * x1 * x1 + c * x1 - r, 2.0) + s * (1.0 - t) * std::cos(x1) + s;
  }
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <dataset_size> <n_runs> <pool_size>" << std::endl;
    return 1;
  }
  
  int n = std::stoi(argv[1]);
  int n_runs = std::stoi(argv[2]);
  int pool_size = std::stoi(argv[3]);
  
  arma::mat X;
  arma::vec y;
  generate_branin_data(X, y, n);
  
  // Set thread pool size
  Optim::set_thread_pool_size(pool_size);
  
  std::vector<double> times;
  double ll = 0;
  
  for (int r = 0; r < n_runs; r++) {
    Kriging km("gauss");
    auto start = std::chrono::high_resolution_clock::now();
    km.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS20", "LL");
    auto end = std::chrono::high_resolution_clock::now();
    times.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0);
    if (r == n_runs - 1) ll = km.logLikelihood();
  }
  
  double avg = 0;
  for (double t : times) avg += t;
  avg /= times.size();
  
  double std_dev = 0;
  for (double t : times) std_dev += (t - avg) * (t - avg);
  std_dev = std::sqrt(std_dev / times.size()) * 1000.0;
  
  std::cout << std::setw(8) << pool_size
            << " │ " << std::setw(12) << std::fixed << std::setprecision(3) << avg
            << " │ " << std::setw(12) << std::fixed << std::setprecision(1) << std_dev
            << " │ " << std::setw(16) << std::fixed << std::setprecision(4) << ll
            << std::endl;
  
  return 0;
}
EOF

# Compile benchmark
echo "Compiling benchmark..."

JEMALLOC_ARG=""
if [ -f "$BUILD_DIR/jemalloc-install/lib/libjemalloc.a" ]; then
    JEMALLOC_ARG="$BUILD_DIR/jemalloc-install/lib/libjemalloc.a"
fi

g++ -std=c++17 -O2 -fopenmp -DARMA_32BIT_WORD \
    "$TEMP_DIR/bench_thread_pool.cpp" \
    -o "$TEMP_DIR/bench_thread_pool" \
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
    rm -rf "$TEMP_DIR"
    exit 1
fi

echo "✓ Compilation successful"
echo ""

# Run benchmarks with different pool sizes
echo "Running benchmarks..."
echo ""
echo "Dataset: n=$DATASET_SIZE points, d=2 dimensions"
echo "Running $N_RUNS iterations per pool size"
echo "BFGS20 method (20 optimization starts)"
echo ""

export LD_LIBRARY_PATH="$BUILD_DIR/src/lib:$LD_LIBRARY_PATH"

echo "┌──────────┬──────────────┬──────────────┬──────────────────┐"
echo "│ Pool Size│  Avg Time(s) │   Std(ms)    │  Log-Likelihood  │"
echo "├──────────┼──────────────┼──────────────┼──────────────────┤"

# Test different pool sizes
# 0 = auto-detect (ncpu)
# 1 = single worker (sequential multistart)
# 2, 4, 8, etc = specific pool sizes
# ncpu = one worker per start (maximum parallelism)

POOL_SIZES=(0 1 2 4 8 $N_CPU)

# Remove duplicates and sort
POOL_SIZES=($(printf '%s\n' "${POOL_SIZES[@]}" | sort -nu))

for pool_size in "${POOL_SIZES[@]}"; do
    # Skip if pool_size exceeds 20 (BFGS20 has 20 starts)
    if [ "$pool_size" -gt 20 ] && [ "$pool_size" -ne "$N_CPU" ]; then
        continue
    fi
    
    "$TEMP_DIR/bench_thread_pool" "$DATASET_SIZE" "$N_RUNS" "$pool_size" | sed 's/^/│ /'
done

echo "└──────────┴──────────────┴──────────────┴──────────────────┘"
echo ""

# Analysis
echo "Analysis:"
echo "─────────"
echo "• Pool size 0 (auto): Uses ncpu workers by default"
echo "• Pool size 1: Sequential execution (one start at a time)"
echo "• Pool size 2-8: Limited parallelism (good for nested BLAS)"
echo "• Pool size $N_CPU: Maximum parallelism (one worker per CPU)"
echo ""
echo "Recommendations:"
echo "• For best throughput: Use pool_size = ncpu/4 to ncpu/2"
echo "• For nested parallelism: Use pool_size = 2-4"
echo "• For debugging: Use pool_size = 1"

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "Benchmark complete!"
