#include <armadillo>
#include <chrono>
#include <iostream>
#include <iomanip>

// Original implementation from Kriging.cpp lines 1020-1028
void dX_original(const arma::mat& X, arma::mat& dX) {
  arma::uword n = X.n_rows;
  arma::uword d = X.n_cols;
  dX = arma::mat(d, n * n, arma::fill::zeros);
  
  for (arma::uword ij = 0; ij < dX.n_cols; ij++) {
    int i = (int)ij / n;
    int j = ij % n;  // i,j <-> i*n+j
    if (i < j) {
      dX.col(ij) = trans(X.row(i) - X.row(j));
      dX.col(j * n + i) = dX.col(ij);
    }
  }
}

// Alternative 1: Loop over i,j directly (avoid division/modulo)
void dX_direct_loop(const arma::mat& X, arma::mat& dX) {
  arma::uword n = X.n_rows;
  arma::uword d = X.n_cols;
  dX = arma::mat(d, n * n, arma::fill::zeros);
  
  for (arma::uword i = 0; i < n; i++) {
    for (arma::uword j = i + 1; j < n; j++) {
      arma::uword ij = i * n + j;
      arma::uword ji = j * n + i;
      dX.col(ij) = trans(X.row(i) - X.row(j));
      dX.col(ji) = dX.col(ij);
    }
  }
}

// Alternative 2: Pre-allocate difference vector
void dX_preallocate(const arma::mat& X, arma::mat& dX) {
  arma::uword n = X.n_rows;
  arma::uword d = X.n_cols;
  dX = arma::mat(d, n * n, arma::fill::zeros);
  arma::vec diff(d);
  
  for (arma::uword i = 0; i < n; i++) {
    for (arma::uword j = i + 1; j < n; j++) {
      arma::uword ij = i * n + j;
      arma::uword ji = j * n + i;
      for (arma::uword k = 0; k < d; k++) {
        diff(k) = X(i, k) - X(j, k);
      }
      dX.col(ij) = diff;
      dX.col(ji) = diff;
    }
  }
}

// Alternative 3: Use memcpy for copy
void dX_memcpy(const arma::mat& X, arma::mat& dX) {
  arma::uword n = X.n_rows;
  arma::uword d = X.n_cols;
  dX = arma::mat(d, n * n, arma::fill::zeros);
  
  for (arma::uword i = 0; i < n; i++) {
    for (arma::uword j = i + 1; j < n; j++) {
      arma::uword ij = i * n + j;
      arma::uword ji = j * n + i;
      dX.col(ij) = trans(X.row(i) - X.row(j));
      std::memcpy(dX.colptr(ji), dX.colptr(ij), d * sizeof(double));
    }
  }
}

// Alternative 4: Direct pointer access
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

// Alternative 5: Vectorized with OpenMP (if available)
void dX_vectorized(const arma::mat& X, arma::mat& dX) {
  arma::uword n = X.n_rows;
  arma::uword d = X.n_cols;
  dX = arma::mat(d, n * n, arma::fill::zeros);
  
  #pragma omp parallel for schedule(static)
  for (arma::uword i = 0; i < n; i++) {
    for (arma::uword j = i + 1; j < n; j++) {
      arma::uword ij = i * n + j;
      arma::uword ji = j * n + i;
      arma::vec diff = trans(X.row(i) - X.row(j));
      dX.col(ij) = diff;
      dX.col(ji) = diff;
    }
  }
}

template<typename Func>
double benchmark(Func func, const arma::mat& X, arma::mat& dX, int iterations = 100) {
  auto start = std::chrono::high_resolution_clock::now();
  
  for (int iter = 0; iter < iterations; iter++) {
    func(X, dX);
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  return elapsed.count() / iterations;
}

void verify_correctness(const arma::mat& X) {
  arma::mat dX1, dX2, dX3, dX4, dX5, dX6;
  
  dX_original(X, dX1);
  dX_direct_loop(X, dX2);
  dX_preallocate(X, dX3);
  dX_memcpy(X, dX4);
  dX_pointer(X, dX5);
  dX_vectorized(X, dX6);
  
  std::cout << "Correctness check:\n";
  std::cout << "  direct_loop vs original: " 
            << (arma::approx_equal(dX1, dX2, "absdiff", 1e-12) ? "PASS" : "FAIL") << "\n";
  std::cout << "  preallocate vs original: " 
            << (arma::approx_equal(dX1, dX3, "absdiff", 1e-12) ? "PASS" : "FAIL") << "\n";
  std::cout << "  memcpy vs original: " 
            << (arma::approx_equal(dX1, dX4, "absdiff", 1e-12) ? "PASS" : "FAIL") << "\n";
  std::cout << "  pointer vs original: " 
            << (arma::approx_equal(dX1, dX5, "absdiff", 1e-12) ? "PASS" : "FAIL") << "\n";
  std::cout << "  vectorized vs original: " 
            << (arma::approx_equal(dX1, dX6, "absdiff", 1e-12) ? "PASS" : "FAIL") << "\n";
}

int main() {
  std::cout << "Benchmarking dX computation implementations\n";
  std::cout << "============================================\n\n";
  
  // Test different sizes
  std::vector<std::pair<arma::uword, arma::uword>> sizes = {
    {50, 2},    // Small problem
    {100, 2},   // Medium problem
    {200, 2},   // Large problem
    {100, 5},   // Medium with more dimensions
    {100, 10}   // Medium with high dimensions
  };
  
  for (const auto& size : sizes) {
    arma::uword n = size.first;
    arma::uword d = size.second;
    
    std::cout << "Problem size: n=" << n << ", d=" << d 
              << " (dX matrix: " << d << "x" << n*n << ")\n";
    
    // Generate random data
    arma::arma_rng::set_seed(42);
    arma::mat X = arma::randu<arma::mat>(n, d);
    arma::mat dX;
    
    // Verify correctness once
    if (n <= 100) {
      verify_correctness(X);
    }
    
    std::cout << "\nBenchmark results (average time per iteration in ms):\n";
    std::cout << std::fixed << std::setprecision(4);
    
    int iterations = (n <= 100) ? 100 : 10;
    
    double t1 = benchmark(dX_original, X, dX, iterations);
    std::cout << "  Original (with div/mod): " << t1 << " ms\n";
    
    double t2 = benchmark(dX_direct_loop, X, dX, iterations);
    std::cout << "  Direct loop:             " << t2 << " ms (speedup: " 
              << std::setprecision(2) << t1/t2 << "x)\n";
    
    double t3 = benchmark(dX_preallocate, X, dX, iterations);
    std::cout << "  Preallocate:             " << std::setprecision(4) << t3 
              << " ms (speedup: " << std::setprecision(2) << t1/t3 << "x)\n";
    
    double t4 = benchmark(dX_memcpy, X, dX, iterations);
    std::cout << "  Memcpy:                  " << std::setprecision(4) << t4 
              << " ms (speedup: " << std::setprecision(2) << t1/t4 << "x)\n";
    
    double t5 = benchmark(dX_pointer, X, dX, iterations);
    std::cout << "  Pointer:                 " << std::setprecision(4) << t5 
              << " ms (speedup: " << std::setprecision(2) << t1/t5 << "x)\n";
    
    double t6 = benchmark(dX_vectorized, X, dX, iterations);
    std::cout << "  Vectorized (OpenMP):     " << std::setprecision(4) << t6 
              << " ms (speedup: " << std::setprecision(2) << t1/t6 << "x)\n";
    
    // Find best
    double best = std::min({t1, t2, t3, t4, t5, t6});
    std::cout << "\n  Best: ";
    if (best == t1) std::cout << "Original";
    else if (best == t2) std::cout << "Direct loop";
    else if (best == t3) std::cout << "Preallocate";
    else if (best == t4) std::cout << "Memcpy";
    else if (best == t5) std::cout << "Pointer";
    else std::cout << "Vectorized";
    std::cout << " (" << std::setprecision(4) << best << " ms)\n";
    
    std::cout << "\n" << std::string(60, '-') << "\n\n";
  }
  
  return 0;
}
