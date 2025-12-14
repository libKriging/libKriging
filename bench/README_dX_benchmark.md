# dX Computation Benchmark Results

## Summary

Benchmarked 6 different implementations of the dX computation (Kriging.cpp lines 1020-1028) with various problem sizes.

## Key Findings

### Best Implementation: **Pointer-based Direct Access**

Speedup over original implementation:
- n=50, d=2: **24.7x faster** (0.089ms → 0.004ms)
- n=100, d=2: **12.9x faster** (0.200ms → 0.015ms)
- n=200, d=2: **6.5x faster** (1.028ms → 0.158ms)
- n=100, d=5: **5.5x faster** (0.218ms → 0.040ms)
- n=100, d=10: **2.0x faster** (0.350ms → 0.172ms)

### Performance Ranking (Best to Worst)

1. **Pointer** - Direct pointer access with manual indexing
2. **Preallocate** - Pre-allocated difference vector
3. **Direct loop** - Loop over i,j directly (avoid div/mod)
4. **Memcpy** - Using memcpy for symmetric copy
5. **Original** - Current implementation with division/modulo
6. **Vectorized (OpenMP)** - Only beneficial for very large n

### Why Pointer Method Wins

1. **No division/modulo operations** - These are expensive CPU operations
2. **Direct memory access** - No Armadillo overhead for indexing
3. **Cache-friendly** - Sequential memory access pattern
4. **Compiler-friendly** - Easy to vectorize automatically

### Implementation Details

The winning implementation:
```cpp
void dX_pointer(const arma::mat& X, arma::mat& dX) {
  arma::uword n = X.n_rows;
  arma::uword d = X.n_cols;
  dX = arma::mat(d, n * n, arma::fill::zeros);
  
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
}
```

## Notes

- OpenMP parallelization is NOT beneficial for this computation due to overhead
- The speedup is most significant for smaller problems (higher relative overhead)
- All implementations produce identical results (verified with `arma::approx_equal`)

## Recommendation

Replace the current implementation in Kriging.cpp (lines 1020-1028) with the pointer-based version for significant performance improvement, especially for problems with moderate n and small d (typical Kriging scenarios).
