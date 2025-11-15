#!/bin/bash

# Benchmark script for BFGS multistart optimization
# Tests parallelism with different numbers of starting points
#
# Requirements:
#   - libKriging built with OpenBLAS (not MKL) for best parallel performance
#   - jemalloc enabled (USE_JEMALLOC=ON)
#   - Sufficient CPU cores (tested on 20-core system)
#
# Usage:
#   ./bench/bench_multistart_optim.sh [dataset_size]
#
# Example:
#   ./bench/bench_multistart_optim.sh 500
#
# Note on BLAS library:
#   - OpenBLAS provides 4-6x better parallel performance than MKL
#   - MKL_NUM_THREADS=1 still has internal serialization causing bottlenecks
#   - Use OPENBLAS_NUM_THREADS=1 to avoid nested parallelism

set -e

# Default parameters
DATASET_SIZE=${1:-500}
N_RUNS=3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
TEMP_DIR=$(mktemp -d)

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║       BFGS Multistart Optimization Benchmark (n=$DATASET_SIZE)           ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory not found at $BUILD_DIR"
    echo "Please build the project first with:"
    echo "  cd $PROJECT_ROOT && mkdir -p build && cd build"
    echo "  cmake .. -DUSE_JEMALLOC=ON && make -j4"
    exit 1
fi

# Check library exists
if [ ! -f "$BUILD_DIR/src/lib/libKriging.so" ]; then
    echo "Error: libKriging.so not found"
    echo "Please build the library first: cd $BUILD_DIR && make Kriging -j4"
    exit 1
fi

# Check BLAS library being used
echo "Checking BLAS library..."
BLAS_LIB=$(ldd "$BUILD_DIR/src/lib/libKriging.so" | grep -i "blas\|lapack" | head -3)
echo "$BLAS_LIB"
echo ""

if echo "$BLAS_LIB" | grep -q "mkl"; then
    echo "⚠️  WARNING: MKL detected!"
    echo "   MKL has internal serialization that limits parallel performance."
    echo "   For best results, rebuild with OpenBLAS instead."
    echo ""
    USING_MKL=1
else
    echo "✓ Using OpenBLAS - good for parallel performance"
    echo ""
    USING_MKL=0
fi

# Create benchmark source code
cat > "$TEMP_DIR/bench_multistart.cpp" << 'EOF'
#include "libKriging/Kriging.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <algorithm>

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
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <dataset_size> <n_runs>" << std::endl;
    return 1;
  }
  
  int n = std::stoi(argv[1]);
  int n_runs = std::stoi(argv[2]);
  
  arma::mat X;
  arma::vec y;
  generate_branin_data(X, y, n);
  
  std::cout << "Dataset: n=" << n << " points, d=2 dimensions" << std::endl;
  std::cout << "Running " << n_runs << " iterations per method" << std::endl;
  std::cout << std::endl;
  
  std::vector<std::string> methods = {"BFGS", "BFGS2", "BFGS10", "BFGS20"};
  std::vector<int> n_starts = {1, 2, 10, 20};
  
  std::cout << "┌──────────┬─────────┬──────────────┬──────────────┬──────────────────┬─────────────────┐" << std::endl;
  std::cout << "│  Method  │ Starts  │  Avg Time(s) │   Std(ms)    │  Log-Likelihood  │  Parallel Eff.  │" << std::endl;
  std::cout << "├──────────┼─────────┼──────────────┼──────────────┼──────────────────┼─────────────────┤" << std::endl;
  
  double base_time = 0;
  
  for (size_t m = 0; m < methods.size(); m++) {
    std::vector<double> times;
    double ll = 0;
    
    for (int r = 0; r < n_runs; r++) {
      Kriging km("gauss");
      auto start = std::chrono::high_resolution_clock::now();
      km.fit(y, X, Trend::RegressionModel::Constant, false, methods[m], "LL");
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
    
    if (m == 0) base_time = avg;
    
    double speedup = (base_time * n_starts[m]) / avg;
    double efficiency = speedup / n_starts[m] * 100.0;
    
    std::cout << "│ " << std::setw(8) << std::left << methods[m]
              << " │ " << std::setw(7) << std::right << n_starts[m]
              << " │ " << std::setw(12) << std::fixed << std::setprecision(3) << avg
              << " │ " << std::setw(12) << std::fixed << std::setprecision(1) << std_dev
              << " │ " << std::setw(16) << std::fixed << std::setprecision(4) << ll
              << " │ " << std::setw(15) << std::fixed << std::setprecision(1) << efficiency << "%"
              << " │" << std::endl;
  }
  
  std::cout << "└──────────┴─────────┴──────────────┴──────────────┴──────────────────┴─────────────────┘" << std::endl;
  
  return 0;
}
EOF

# Compile benchmark
echo "Compiling benchmark..."
g++ -std=c++17 -O2 -fopenmp -DARMA_32BIT_WORD \
    "$TEMP_DIR/bench_multistart.cpp" \
    -o "$TEMP_DIR/bench_multistart" \
    -I"$PROJECT_ROOT/src/lib/include" \
    -I"$BUILD_DIR/dependencies/armadillo-code/tmp/include" \
    -I"$BUILD_DIR/src/lib" \
    -L"$BUILD_DIR/src/lib" \
    -lKriging \
    -Wl,-rpath,"$BUILD_DIR/src/lib"

if [ $? -ne 0 ]; then
    echo "Error: Compilation failed"
    rm -rf "$TEMP_DIR"
    exit 1
fi

echo "✓ Compilation successful"
echo ""

# Run benchmark
echo "Running benchmark..."
echo ""

export LD_LIBRARY_PATH="$BUILD_DIR/src/lib:$LD_LIBRARY_PATH"

"$TEMP_DIR/bench_multistart" "$DATASET_SIZE" "$N_RUNS"

if [ $USING_MKL -eq 1 ]; then
    echo "⚠️  Performance limited by MKL internal serialization"
    echo ""
    echo "   To improve performance:"
    echo "     1. Remove MKL from system"
    echo "     2. Rebuild libKriging (it will use OpenBLAS)"
    echo "     3. Re-run this benchmark"
fi

# Cleanup
rm -rf "$TEMP_DIR"

echo "Benchmark complete!"
