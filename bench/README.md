# libKriging Benchmarks

This directory contains performance benchmarks for the libKriging C++ library.

## Overview

The benchmark suite tests the performance of three Kriging model types across various data sizes and dimensions:

- **Kriging** - Standard Kriging models (default)
- **NuggetKriging** - Kriging with nugget effect
- **NoiseKriging** - Kriging with heteroscedastic noise

Each benchmark tests five key operations:
1. **fit** - Model fitting with BFGS optimization using LL (Log-Likelihood) objective
2. **predict** - Prediction at new points (with derivatives)
3. **simulate** - Conditional simulation
4. **update** - Model update with new data
5. **update_simulate** - Combined update and simulation

## Default Configuration

**By default, benchmarks run with:**
- **Benchmark**: Kriging only (standard Kriging with LL method)
- **Training points (n)**: 100
- **Dimensions (d)**: 4
- **Iterations**: 10

This provides a standard, focused benchmark for typical use cases.

## Test Configurations

All parameters are configurable via command-line arguments:

- **Training points (n)**: Any positive integer (default: 100, typical: 10-1000)
- **Dimensions (d)**: Any positive integer (default: 4, typical: 1-10)
- **Iterations**: Any positive integer (default: 10, typical: 5-50)

## Usage

### Using bench.sh (Recommended)

The `bench.sh` script provides an easy way to run benchmarks with explicit named parameters:

```bash
# Run default benchmark (iterations=10 n=100 d=4 kriging)
./bench/bench.sh

# Run with custom parameters (all optional, order-independent)
./bench/bench.sh iterations=20 n=50 d=2 kriging
./bench/bench.sh n=200 d=4 nugget
./bench/bench.sh iterations=5 noise

# Filter by operation (show only specific operation)
./bench/bench.sh kriging fit
./bench/bench.sh n=50 d=2 nugget predict
./bench/bench.sh iterations=3 noise update_simulate
```

**Named Parameters (all optional):**
- `iterations=N` - Number of iterations per test (default: `10`)
- `n=N` - Number of training points (default: `100`)
- `d=N` - Number of dimensions (default: `4`)

**Benchmark Type Filter:**
- `kriging` - Standard Kriging (uses LL objective) [default]
- `nugget` - NuggetKriging (uses LMP objective)
- `noise` - NoiseKriging (uses LL objective)

**Operation Filter (optional):**
- `fit` - Show only fit operation
- `predict` - Show only predict operation
- `simulate` - Show only simulate operation
- `update` - Show only update operation
- `update_simulate` - Show only update_simulate operation
- (omit for all operations)

### Running Directly

You can also run the benchmark executables directly:

```bash
cd build/bench

# Run with default configuration (n=100, d=4, 10 iterations)
./bench-kriging

# Run with custom parameters: [iterations] [n] [d]
./bench-kriging 20              # 20 iterations, n=100, d=4
./bench-kriging 15 50 2         # 15 iterations, n=50, d=2
./bench-nugget-kriging 10 200 4 # 10 iterations, n=200, d=4
./bench-noise-kriging 5 30 1    # 5 iterations, n=30, d=1
```

### Running Full Test Suite

To run the complete benchmark suite with all combinations of n and d values, use a shell script:

```bash
# Test all combinations of n and d
for n in 10 100 1000; do
  for d in 1 2 4; do
    ./bench/bench.sh iterations=10 n=$n d=$d kriging | tee -a full-results.log
  done
done

# Or create a custom script
for benchmark in kriging nugget noise; do
  for n in 50 100 200; do
    ./bench/bench.sh iterations=5 n=$n d=4 $benchmark
  done
done
```

### Building Benchmarks

```bash
cd build
cmake --build . --target all_benchmarks -j8
```

## Output Format

Each benchmark produces minimal formatted output with statistics for each operation:

```
Kriging Benchmark (BFGS, LL objective)

n=100 d=2 iterations=10
Operation                 |  Mean (ms) |   Std (ms) |   Min (ms) |   Max (ms) | Median (ms)
fit                       |     50.385 |      3.168 |     46.177 |     53.820 |     51.158
predict                   |      1.236 |      0.447 |      0.909 |      1.868 |      0.931
simulate                  |      0.216 |      0.049 |      0.176 |      0.285 |      0.188
update                    |    991.258 |     50.069 |    929.696 |   1052.337 |    991.743
update_simulate           |      0.502 |      0.275 |      0.301 |      0.892 |      0.313

Results saved to: /path/to/benchmark-results.log
```

**Statistics computed:**
- **Mean** - Average execution time
- **Std** - Standard deviation
- **Min** - Minimum execution time
- **Max** - Maximum execution time
- **Median** - Median execution time

## CI Integration

Benchmarks are integrated with GitHub Actions via `.github/workflows/bench-cpp.yml`.

**Configuration:**
- **Platform:** Ubuntu (latest)
- **Build Type:** Release
- **Test configurations:**
  - n ∈ {100, 200, 400}
  - d ∈ {2, 4, 8}
  - All three benchmark types (Kriging, NuggetKriging, NoiseKriging)
- **Total:** 27 configurations per run

**Triggers:**
- Push to master/develop branches
- Tags matching `bench-cpp*`
- Pull requests to master/develop
- Manual workflow dispatch (with custom iteration count)

**Generated Report:**
The CI automatically generates a comprehensive markdown report (`BENCHMARK_REPORT.md`) containing:
- Summary tables with mean times for fit/predict/update operations
- Detailed results for each configuration
- Comparison across all benchmark types
- Available in GitHub Actions artifacts and job summary

See `bench/SAMPLE_REPORT.md` for an example of the generated report format.

**Artifacts:**
- Individual benchmark logs for all 27 configurations
- Aggregated markdown report (`BENCHMARK_REPORT.md`)
- Results retained for 90 days

## Testing CI Report Generation Locally

To test the CI report generation locally without running all 27 configurations:

```bash
cd build
bash ../bench/test-report-generation.sh
```

This script will:
1. Run all benchmark combinations (n ∈ {100, 200, 400}, d ∈ {2, 4, 8})
2. Generate the same markdown report as CI
3. Display the report preview

**Note:** This will take significant time to complete all 27 configurations.

## Files

- **bench-kriging.cpp** - Kriging benchmark implementation
- **bench-nugget-kriging.cpp** - NuggetKriging benchmark implementation
- **bench-noise-kriging.cpp** - NoiseKriging benchmark implementation
- **bench.sh** - Convenience script to run benchmarks
- **test-report-generation.sh** - Local test script for CI report generation
- **CMakeLists.txt** - Build configuration
- **README.md** - This file

## Notes

- All benchmarks use BFGS optimization without parallelization for consistency
- The synthetic test function is: `f(x) = sum_{i=1}^{d} sin(2*pi*x_i)`
- Random seed is set to 123 for reproducibility
- Update tests use 10% of training data size for updates
- Prediction tests use 10 prediction points
- Simulation tests generate 10 trajectories

## Performance Tips

For best benchmark results:

1. **Build in Release mode:**
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release
   ```

2. **Use optimized BLAS:**
   - OpenBLAS
   - Intel MKL
   - Apple Accelerate (macOS)

3. **Minimize system load:**
   - Close other applications
   - Disable CPU frequency scaling (if possible)
   - Run multiple iterations to reduce variance

4. **Run from command line, not IDE**
   - Avoids debugger overhead

## Example Workflow

```bash
# Build the project in release mode
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target all_benchmarks -j8

# Run benchmarks with custom parameters
./bench/bench.sh iterations=20 n=100 d=4 kriging
./bench/bench.sh iterations=20 n=100 d=4 nugget
./bench/bench.sh iterations=20 n=100 d=4 noise

# Test specific operation
./bench/bench.sh iterations=50 n=200 d=4 kriging fit

# Compare performance at different scales
./bench/bench.sh iterations=10 n=50 d=2 kriging
./bench/bench.sh iterations=10 n=500 d=2 kriging

# View results
cat build/benchmark-results.log
```

## Contributing

When adding new benchmarks:

1. Create a new `bench-*.cpp` file following the existing pattern
2. Update `bench/CMakeLists.txt` to add the new executable
3. Update `bench.sh` to include the new benchmark
4. Update this README with the new benchmark description
