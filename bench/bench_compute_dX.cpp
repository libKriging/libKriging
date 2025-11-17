#include <armadillo>
#include <chrono>
#include <iostream>
#include <iomanip>

// Original implementation from Kriging.cpp (before optimization)
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

// Optimized pointer-based implementation (new LinearAlgebra::compute_dX)
arma::mat compute_dX_optimized(const arma::mat& X) {
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

template<typename Func>
double benchmark(Func func, const arma::mat& X, int iterations = 100) {
  auto start = std::chrono::high_resolution_clock::now();
  
  for (int iter = 0; iter < iterations; iter++) {
    auto result = func(X);
    // Prevent compiler from optimizing away
    if (result.n_elem == 0) std::cout << "";
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  return elapsed.count() / iterations;
}

int main() {
  std::cout << "Benchmark: Original vs Optimized compute_dX\n";
  std::cout << "============================================\n\n";
  
  // Test different sizes typical for Kriging
  std::vector<std::pair<arma::uword, arma::uword>> sizes = {
    {20, 2},    // Small
    {50, 2},    // Small-Medium
    {100, 2},   // Medium
    {100, 5},   // Medium with more dimensions
    {200, 2},   // Large
  };
  
  for (const auto& size : sizes) {
    arma::uword n = size.first;
    arma::uword d = size.second;
    
    std::cout << "Problem size: n=" << n << ", d=" << d 
              << " (dX matrix: " << d << "x" << n*n << ")\n";
    
    // Generate random data
    arma::arma_rng::set_seed(42);
    arma::mat X = arma::randu<arma::mat>(n, d);
    
    // Verify correctness
    arma::mat dX_old, dX_new;
    dX_original(X, dX_old);
    dX_new = compute_dX_optimized(X);
    
    bool correct = arma::approx_equal(dX_old, dX_new, "absdiff", 1e-12);
    std::cout << "  Correctness: " << (correct ? "PASS" : "FAIL") << "\n";
    
    if (!correct) {
      std::cout << "  ERROR: Results differ!\n";
      std::cout << "  Max diff: " << arma::max(arma::max(arma::abs(dX_old - dX_new))) << "\n";
      continue;
    }
    
    // Benchmark
    int iterations = (n <= 100) ? 100 : 20;
    
    double t_old = benchmark([](const arma::mat& X) {
      arma::mat dX;
      dX_original(X, dX);
      return dX;
    }, X, iterations);
    
    double t_new = benchmark([](const arma::mat& X) {
      return compute_dX_optimized(X);
    }, X, iterations);
    
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Original:  " << t_old << " ms\n";
    std::cout << "  Optimized: " << t_new << " ms\n";
    std::cout << "  Speedup:   " << std::setprecision(2) << t_old/t_new << "x\n";
    std::cout << "\n";
  }
  
  return 0;
}
