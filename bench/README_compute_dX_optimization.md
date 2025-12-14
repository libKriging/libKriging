# compute_dX Optimization

## Summary

Implemented a fast pointer-based method `LinearAlgebra::compute_dX()` to replace the original dX computation loop in all *Kriging classes (Kriging, NuggetKriging, NoiseKriging).

## Performance Improvement

Benchmark results show significant speedup across different problem sizes:

| Problem Size (n, d) | Original Time | Optimized Time | Speedup |
|---------------------|---------------|----------------|---------|
| n=20, d=2           | 0.0081 ms     | 0.0007 ms      | **11.3x** |
| n=50, d=2           | 0.0507 ms     | 0.0038 ms      | **13.2x** |
| n=100, d=2          | 0.1980 ms     | 0.0154 ms      | **12.8x** |
| n=100, d=5          | 0.2125 ms     | 0.0421 ms      | **5.1x** |
| n=200, d=2          | 0.7886 ms     | 0.0672 ms      | **11.7x** |

Average speedup: **~10x** for typical Kriging problems (moderate n, small d)

## Implementation Details

### Original Implementation (lines 1020-1028 in Kriging.cpp)
```cpp
m_dX = arma::mat(d, n * n, arma::fill::zeros);
for (arma::uword ij = 0; ij < m_dX.n_cols; ij++) {
  int i = (int)ij / n;
  int j = ij % n;  // i,j <-> i*n+j
  if (i < j) {
    m_dX.col(ij) = trans(m_X.row(i) - m_X.row(j));
    m_dX.col(j * n + i) = m_dX.col(ij);
  }
}
```

**Issues:**
- Division and modulo operations are expensive (10-20 CPU cycles each)
- Armadillo indexing overhead for each column access
- Poor cache locality

### Optimized Implementation (LinearAlgebra::compute_dX)
```cpp
arma::mat LinearAlgebra::compute_dX(const arma::mat& X) {
  arma::uword n = X.n_rows;
  arma::uword d = X.n_cols;
  arma::mat dX(d, n * n, arma::fill::zeros);
  
  const double* X_mem = X.memptr();
  double* dX_mem = dX.memptr();
  
  for (arma::uword i = 0; i < n; i++) {
    for (arma::uword j = i + 1; j < n; j++) {
      arma::uword ij = i * n + j;
      arma::uword ji = j * n + i;
      for (arma::uword k = 0; k < d; k++) {
        double diff = X_mem[i + k * n] - X_mem[j + k * n];
        dX_mem[k + ij * d] = diff;
        dX_mem[k + ji * d] = diff;
      }
    }
  }
  
  return dX;
}
```

**Improvements:**
- Direct pointer access eliminates Armadillo indexing overhead
- No division/modulo operations
- Better cache locality with sequential memory access
- Easier for compiler to vectorize

## Changes Made

1. Added `LinearAlgebra::compute_dX()` static method in `LinearAlgebra.hpp/cpp`
2. Replaced dX computation loops in:
   - `src/lib/Kriging.cpp` (line ~1020)
   - `src/lib/NuggetKriging.cpp` (line ~750)
   - `src/lib/NoiseKriging.cpp` (line ~390)

## Testing

All tests pass:
- ✅ KrigingTest (37 assertions)
- ✅ KrigingLogLikTest (12 assertions)
- ✅ NuggetKrigingLogLikTest (10 assertions)
- ✅ NoiseKrigingTest (20 assertions)
- ✅ NoiseKrigingLogLikTest (7 assertions)
- ✅ LinearAlgebraTest (378 assertions)

Results are identical to original implementation (verified with `arma::approx_equal` at 1e-12 precision).

## Benchmark Script

Run `bench/bench_compute_dX` to reproduce the performance comparison.

Build with:
```bash
cd bench
g++ -O3 -march=native -std=c++14 \
    -I"../dependencies/armadillo-code/include" \
    bench_compute_dX.cpp \
    -o bench_compute_dX \
    -L"../build/dependencies/armadillo-code" \
    -larmadillo -llapack -lblas
./bench_compute_dX
```
