# Covariance Computation Optimization Benchmark

This benchmark tests different implementations for computing covariance matrices and their gradients, which are critical operations in Kriging methods.

## Target Operations

### 1. Covariance Matrix Filling
Current implementation in `*Kriging.cpp`:
```cpp
for (arma::uword i = 0; i < n; i++) {
  R.at(i, i) = 1.0;
  for (arma::uword j = 0; j < i; j++) {
    R.at(i, j) = R.at(j, i) = _Cov((X.row(i) - X.row(j)).t(), theta);
  }
}
```

### 2. Gradient Covariance Computation
Current implementation in gradient calculations (lines ~240-246 in Kriging.cpp):
```cpp
for (arma::uword i = 0; i < n; i++) {
  gradR.tube(i, i) = zeros;
  for (arma::uword j = 0; j < i; j++) {
    gradR.tube(i, j) = m.R.at(i, j) * _DlnCovDtheta(m_dX.col(i * n + j), _theta);
    gradR.tube(j, i) = gradR.tube(i, j);
  }
}
```

## Tested Methods

### Covariance Matrix Methods:
1. **Current nested loops**: Direct implementation with nested loops
2. **Vectorized**: Pre-fetch rows to reduce indexing overhead
3. **Precompute differences**: Separate computation of differences and covariances
4. **OpenMP**: Parallel computation with OpenMP (if available)
5. **Cache-friendly**: Column-major traversal for better cache utilization
6. **Blocked**: Block-wise computation for improved cache behavior

### GradR Methods:
1. **Current nested loops**: Direct implementation
2. **Pre-multiply R**: Extract R values before loop
3. **Column-wise**: Cache-friendly column traversal

## Building

```bash
cd bench
./build_bench_cov.sh
```

## Running

```bash
./bench_cov_computation [n_points] [dimension] [n_runs]
```

Examples:
```bash
# Default run (n=100, d=2, runs=5)
./bench_cov_computation

# Custom parameters
./bench_cov_computation 200 3 10

# Large test
./bench_cov_computation 500 5 5
```

## Expected Output

The benchmark will test multiple problem sizes and report:
- Execution time for each method (averaged over multiple runs)
- Maximum difference from reference implementation (for correctness)

## Interpretation

- **Speed**: Look for methods with lowest execution time
- **Accuracy**: All methods should have max_diff ≈ 0 (numerical precision)
- **Scalability**: Test with increasing n to see how methods scale

## OpenMP Notes

If OpenMP is available, parallel methods will be tested. To control threads:
```bash
export OMP_NUM_THREADS=4
./bench_cov_computation
```

## Integration

After identifying the fastest method:
1. Implement the optimized version in `LinearAlgebra` as a static method
2. Update `Kriging.cpp`, `NuggetKriging.cpp`, `NoiseKriging.cpp` to use the new method
3. Verify with existing tests that results are unchanged
4. Measure overall performance improvement with existing benchmarks
