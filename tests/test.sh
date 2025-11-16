#!/bin/bash

# Script to compile and run a specific test by name
# Usage: ./test.sh "test_name"
#   Example: ./test.sh "LinearAlgebra::safe_chol_lower - correlation-like matrix near singular"

if [ $# -eq 0 ]; then
    echo "Usage: $0 \"test_name\""
    echo "Example: $0 \"LinearAlgebra::safe_chol_lower - correlation-like matrix near singular\""
    exit 1
fi

TEST_NAME="$1"

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

# Build the tests
echo "Building tests..."
cmake --build . --target all_test_binaries -j$(nproc)

if [ $? -ne 0 ]; then
    echo "Error: Build failed"
    exit 1
fi

echo ""
echo "Running test: $TEST_NAME"
echo "----------------------------------------"

# Try to find and run the test in common test executables
TEST_EXECUTABLES=(
    "tests/KrigingTest"
    "tests/NuggetKrigingTest"
    "tests/NoiseKrigingTest"
    "tests/KrigingLogLikTest"
    "tests/NuggetKrigingLogLikTest"
    "tests/NoiseKrigingLogLikTest"
    "tests/LinearAlgebraTest"
    "tests/CacheTest"
    "tests/JsonTest"
    "tests/catch2_unit_test"
    "tests/regression_unit_test"
)

FOUND=0
for TEST_EXEC in "${TEST_EXECUTABLES[@]}"; do
    if [ -f "$TEST_EXEC" ]; then
        # Run the test and capture output
        OUTPUT=$("$TEST_EXEC" "$TEST_NAME" 2>&1)
        if echo "$OUTPUT" | grep -q "All tests passed"; then
            echo "$OUTPUT"
            FOUND=1
            echo ""
            echo "✓ Test passed successfully"
            exit 0
        elif echo "$OUTPUT" | grep -q "test cases:.*|.*1 passed"; then
            echo "$OUTPUT"
            FOUND=1
            echo ""
            echo "✓ Test passed successfully"
            exit 0
        elif echo "$OUTPUT" | grep -q "test cases:"; then
            echo "$OUTPUT"
            FOUND=1
            echo ""
            echo "✗ Test failed or had errors"
            exit 1
        fi
    fi
done

if [ $FOUND -eq 0 ]; then
    echo "Error: Test '$TEST_NAME' not found in any test executable"
    echo ""
    echo "Available test executables:"
    for TEST_EXEC in "${TEST_EXECUTABLES[@]}"; do
        if [ -f "$TEST_EXEC" ]; then
            echo "  - $TEST_EXEC"
        fi
    done
    exit 1
fi
