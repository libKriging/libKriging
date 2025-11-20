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

# Build the tests (but don't fail if some tests fail to build)
echo "Building tests..."
cmake --build . --target all_test_binaries -j$(nproc) 2>&1 | grep -v "LinearAlgebraTest" || true

echo ""
echo "Running test: $TEST_NAME"
echo "----------------------------------------"

# Dynamically discover all test executables from *Test.cpp files in the tests directory
TEST_EXECUTABLES=()
for test_file in "$SCRIPT_DIR"/*Test*.cpp; do
    if [ -f "$test_file" ]; then
        # Extract basename without extension
        test_name=$(basename "$test_file" .cpp)
        test_exec="tests/$test_name"
        # Check if the executable exists in build directory
        if [ -f "$test_exec" ]; then
            TEST_EXECUTABLES+=("$test_exec")
        fi
    fi
done

# Add other test executables that don't follow the *Test.cpp pattern
OTHER_EXECUTABLES=(
    "tests/catch2_unit_test"
    "tests/regression_unit_test"
)

for TEST_EXEC in "${OTHER_EXECUTABLES[@]}"; do
    if [ -f "$TEST_EXEC" ]; then
        TEST_EXECUTABLES+=("$TEST_EXEC")
    fi
done

FOUND=0
for TEST_EXEC in "${TEST_EXECUTABLES[@]}"; do
    if [ -f "$TEST_EXEC" ]; then
        # Run the test and capture output (with -s flag to show INFO messages)
        OUTPUT=$("$TEST_EXEC" "$TEST_NAME" -s 2>&1)
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
