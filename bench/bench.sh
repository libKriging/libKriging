#!/bin/bash

# Script to compile and run benchmarks with explicit parameters
# Usage: ./bench.sh [iterations=N] [n=N] [d=N] [filter]
#
# Arguments (all optional, order-independent):
#   iterations=N  : Number of iterations per test (default: 10)
#   n=N          : Number of training points (default: 100)
#   d=N          : Number of dimensions (default: 4)
#   filter       : Benchmark type filter (kriging/nugget/noise) or
#                  Operation filter (fit/predict/simulate/update/update_simulate)
#                  Default: kriging
#
# Examples:
#   ./bench.sh                                    # n=100 d=4 iterations=10 kriging
#   ./bench.sh iterations=20                      # n=100 d=4 iterations=20 kriging
#   ./bench.sh n=50 d=2                          # n=50 d=2 iterations=10 kriging
#   ./bench.sh iterations=5 n=1000 d=4 fit       # Only fit operation
#   ./bench.sh nugget                             # NuggetKriging benchmark
#   ./bench.sh noise iterations=15 predict       # NoiseKriging predict only

# Default values
ITERATIONS=10
N_POINTS=100
D_DIMS=4
BENCHMARK_FILTER="kriging"
OPERATION_FILTER=""

# Parse arguments
for arg in "$@"; do
  case "$arg" in
    iterations=*)
      ITERATIONS="${arg#*=}"
      ;;
    n=*)
      N_POINTS="${arg#*=}"
      ;;
    d=*)
      D_DIMS="${arg#*=}"
      ;;
    kriging|nugget|noise)
      BENCHMARK_FILTER="$arg"
      ;;
    fit|predict|simulate|update|update_simulate)
      OPERATION_FILTER="$arg"
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Valid arguments: iterations=N, n=N, d=N, kriging|nugget|noise, fit|predict|simulate|update|update_simulate"
      exit 1
      ;;
  esac
done

OUTPUT_FILE="benchmark-results.log"

# Get the script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default build directory
BUILD_DIR="${PROJECT_ROOT}/build"

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory not found at $BUILD_DIR"
    echo "Please run cmake and build the project first"
    exit 1
fi

cd "$BUILD_DIR"

# Build the benchmarks (silently)
if ! cmake --build . --target all_benchmarks -j$(nproc) 2>&1 > /dev/null; then
    echo "Error: Failed to build benchmarks"
    exit 1
fi

# Define available benchmarks
declare -A BENCHMARKS
BENCHMARKS["kriging"]="bench/bench-kriging"
BENCHMARKS["nugget"]="bench/bench-nugget-kriging"
BENCHMARKS["noise"]="bench/bench-noise-kriging"

# Function to filter output by operation
filter_operation() {
    local operation_filter=$1
    if [ -z "$operation_filter" ]; then
        cat
    else
        awk -v op="$operation_filter" '
        BEGIN { printing=0; header_printed=0 }
        /^[A-Za-z].*Benchmark/ { print; next }
        /^n=/ { print; next }
        /^Operation.*Mean.*Std.*Min.*Max.*Median/ {
            if (!header_printed) {
                print
                header_printed=1
            }
            next
        }
        $1 == op { printing=1; print; next }
        /^[a-z_]+[ ]*\|/ {
            if (printing && $1 == op) print
            printing=0
        }
        '
    fi
}

# Function to run a single benchmark
run_benchmark() {
    local name=$1
    local executable=$2
    local iterations=$3
    local n=$4
    local d=$5
    local op_filter=$6

    if [ ! -f "$executable" ]; then
        echo "Warning: Benchmark executable not found: $executable"
        return 1
    fi

    # Run benchmark and filter output
    "$executable" "$iterations" "$n" "$d" 2>&1 | filter_operation "$op_filter"
    local exit_code=$?

    return $exit_code
}

# Initialize output file
OUTPUT_PATH="${BUILD_DIR}/${OUTPUT_FILE}"
{
    echo "Date: $(date)"
    echo "Configuration: n=$N_POINTS d=$D_DIMS iterations=$ITERATIONS benchmark=$BENCHMARK_FILTER"
    if [ -n "$OPERATION_FILTER" ]; then
        echo "Operation filter: $OPERATION_FILTER"
    fi
    echo ""
} > "$OUTPUT_PATH"

# Run benchmarks based on filter
FAILED=0
SUCCESS_COUNT=0
TOTAL_COUNT=0

# Check if benchmark filter is valid
if [ -z "${BENCHMARKS[$BENCHMARK_FILTER]}" ]; then
    echo "Error: Unknown benchmark filter: $BENCHMARK_FILTER"
    echo ""
    echo "Available benchmarks:"
    for name in "${!BENCHMARKS[@]}"; do
        echo "  - $name"
    done
    exit 1
fi

TOTAL_COUNT=1
executable="${BENCHMARKS[$BENCHMARK_FILTER]}"

if run_benchmark "$BENCHMARK_FILTER" "$executable" "$ITERATIONS" "$N_POINTS" "$D_DIMS" "$OPERATION_FILTER" | tee -a "$OUTPUT_PATH"; then
    SUCCESS_COUNT=1
else
    FAILED=1
fi

# Print summary
echo ""
echo "Results saved to: $OUTPUT_PATH"

# Append summary to output file
{
    echo ""
    echo "Completed: $SUCCESS_COUNT/$TOTAL_COUNT benchmarks - $(date)"
} >> "$OUTPUT_PATH"

if [ $FAILED -eq 1 ]; then
    exit 1
else
    exit 0
fi
