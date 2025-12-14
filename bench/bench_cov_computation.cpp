// Benchmark for covariance computation loops optimization
// This benchmark compares different implementations of covariance matrix filling

#include <armadillo>
#include <iostream>
#include <chrono>
#include <vector>
#include <functional>
#include <cmath>

// Timing helper
class Timer {
  std::chrono::high_resolution_clock::time_point start_time;
public:
  Timer() : start_time(std::chrono::high_resolution_clock::now()) {}
  
  double elapsed_ms() const {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(now - start_time).count();
  }
  
  void reset() {
    start_time = std::chrono::high_resolution_clock::now();
  }
};

// Method 1: Current implementation - nested loops with individual Cov calls
void method1_nested_loops(arma::mat& R, const arma::mat& X, const arma::vec& theta, 
                          const std::function<double(const arma::vec&, const arma::vec&)>& Cov) {
  arma::uword n = X.n_rows;
  for (arma::uword i = 0; i < n; i++) {
    R.at(i, i) = 1.0;
    for (arma::uword j = 0; j < i; j++) {
      R.at(i, j) = R.at(j, i) = Cov((X.row(i) - X.row(j)).t(), theta);
    }
  }
}

// Method 2: Vectorized computation with column-wise operations
void method2_vectorized(arma::mat& R, const arma::mat& X, const arma::vec& theta,
                        const std::function<double(const arma::vec&, const arma::vec&)>& Cov) {
  arma::uword n = X.n_rows;
  R.diag().ones();
  
  for (arma::uword i = 1; i < n; i++) {
    arma::rowvec Xi = X.row(i);
    for (arma::uword j = 0; j < i; j++) {
      double val = Cov((Xi - X.row(j)).t(), theta);
      R.at(i, j) = val;
      R.at(j, i) = val;
    }
  }
}

// Method 3: Pre-compute differences, then apply covariance
void method3_precompute_diff(arma::mat& R, const arma::mat& X, const arma::vec& theta,
                              const std::function<double(const arma::vec&, const arma::vec&)>& Cov) {
  arma::uword n = X.n_rows;
  R.diag().ones();
  
  // Pre-compute all differences
  std::vector<arma::vec> diffs;
  diffs.reserve(n * (n - 1) / 2);
  
  for (arma::uword i = 1; i < n; i++) {
    for (arma::uword j = 0; j < i; j++) {
      diffs.push_back((X.row(i) - X.row(j)).t());
    }
  }
  
  // Compute covariances
  arma::uword idx = 0;
  for (arma::uword i = 1; i < n; i++) {
    for (arma::uword j = 0; j < i; j++) {
      double val = Cov(diffs[idx++], theta);
      R.at(i, j) = val;
      R.at(j, i) = val;
    }
  }
}

// Method 4: OpenMP parallelization (if enabled)
void method4_openmp(arma::mat& R, const arma::mat& X, const arma::vec& theta,
                    const std::function<double(const arma::vec&, const arma::vec&)>& Cov) {
  arma::uword n = X.n_rows;
  R.diag().ones();
  
  #ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic)
  for (arma::uword i = 1; i < n; i++) {
    arma::rowvec Xi = X.row(i);
    for (arma::uword j = 0; j < i; j++) {
      double val = Cov((Xi - X.row(j)).t(), theta);
      R.at(i, j) = val;
      R.at(j, i) = val;
    }
  }
  #else
  method2_vectorized(R, X, theta, Cov);
  #endif
}

// Method 5: Cache-friendly traversal (column-major for Armadillo)
void method5_cache_friendly(arma::mat& R, const arma::mat& X, const arma::vec& theta,
                             const std::function<double(const arma::vec&, const arma::vec&)>& Cov) {
  arma::uword n = X.n_rows;
  R.diag().ones();
  
  // Traverse by columns (cache-friendly for Armadillo)
  for (arma::uword j = 0; j < n; j++) {
    arma::rowvec Xj = X.row(j);
    for (arma::uword i = j + 1; i < n; i++) {
      double val = Cov((X.row(i) - Xj).t(), theta);
      R.at(i, j) = val;
      R.at(j, i) = val;
    }
  }
}

// Method 6: Blocked computation for better cache utilization
void method6_blocked(arma::mat& R, const arma::mat& X, const arma::vec& theta,
                     const std::function<double(const arma::vec&, const arma::vec&)>& Cov,
                     arma::uword block_size = 64) {
  arma::uword n = X.n_rows;
  R.diag().ones();
  
  for (arma::uword bi = 0; bi < n; bi += block_size) {
    arma::uword block_end_i = std::min(bi + block_size, n);
    for (arma::uword bj = 0; bj < bi; bj += block_size) {
      arma::uword block_end_j = std::min(bj + block_size, n);
      
      // Process block
      for (arma::uword i = bi; i < block_end_i; i++) {
        arma::rowvec Xi = X.row(i);
        arma::uword j_start = (bj == bi) ? bi : bj;
        arma::uword j_end = (bj == bi) ? i : block_end_j;
        
        for (arma::uword j = j_start; j < j_end; j++) {
          double val = Cov((Xi - X.row(j)).t(), theta);
          R.at(i, j) = val;
          R.at(j, i) = val;
        }
      }
    }
  }
}

// Benchmark for gradR computation (used in gradient calculations)
void benchmark_gradR_computation(arma::uword n, arma::uword d, int n_runs = 5) {
  std::cout << "\n=== GradR Computation Benchmark (n=" << n << ", d=" << d << ") ===" << std::endl;
  
  // Setup
  arma::mat dX(d, n * n);
  arma::mat R(n, n);
  arma::vec theta = arma::vec(d, arma::fill::ones) * 0.5;
  
  for (arma::uword i = 0; i < n * n; i++) {
    dX.col(i) = arma::randn<arma::vec>(d);
  }
  R.randn();
  R = 0.5 * (R + R.t());
  R.diag().ones();
  
  // Simple derivative of log Gaussian covariance for testing
  auto DlnCovDtheta = [](const arma::vec& dx, const arma::vec& theta) {
    arma::vec grad(theta.n_elem);
    for (arma::uword i = 0; i < dx.n_elem; i++) {
      double r = dx[i] / theta[i];
      grad[i] = r * r / theta[i];
    }
    return grad;
  };
  
  // Method 1: Current nested loop implementation
  {
    Timer timer;
    double total_time = 0.0;
    
    for (int run = 0; run < n_runs; run++) {
      arma::cube gradR(n, n, d, arma::fill::none);
      const arma::vec zeros = arma::vec(d, arma::fill::zeros);
      
      timer.reset();
      for (arma::uword i = 0; i < n; i++) {
        gradR.tube(i, i) = zeros;
        for (arma::uword j = 0; j < i; j++) {
          gradR.tube(i, j) = R.at(i, j) * DlnCovDtheta(dX.col(i * n + j), theta);
          gradR.tube(j, i) = gradR.tube(i, j);
        }
      }
      total_time += timer.elapsed_ms();
    }
    
    std::cout << "Method 1 (current nested loops):  " << total_time / n_runs << " ms" << std::endl;
  }
  
  // Method 2: Pre-multiply R values
  {
    Timer timer;
    double total_time = 0.0;
    
    for (int run = 0; run < n_runs; run++) {
      arma::cube gradR(n, n, d, arma::fill::none);
      const arma::vec zeros = arma::vec(d, arma::fill::zeros);
      
      timer.reset();
      for (arma::uword i = 0; i < n; i++) {
        gradR.tube(i, i) = zeros;
      }
      
      for (arma::uword i = 1; i < n; i++) {
        double R_ij;
        for (arma::uword j = 0; j < i; j++) {
          R_ij = R.at(i, j);
          arma::vec grad_ij = R_ij * DlnCovDtheta(dX.col(i * n + j), theta);
          gradR.tube(i, j) = grad_ij;
          gradR.tube(j, i) = grad_ij;
        }
      }
      total_time += timer.elapsed_ms();
    }
    
    std::cout << "Method 2 (pre-multiply R):         " << total_time / n_runs << " ms" << std::endl;
  }
  
  // Method 3: Column-wise traversal
  {
    Timer timer;
    double total_time = 0.0;
    
    for (int run = 0; run < n_runs; run++) {
      arma::cube gradR(n, n, d, arma::fill::zeros);
      
      timer.reset();
      for (arma::uword j = 0; j < n; j++) {
        for (arma::uword i = j + 1; i < n; i++) {
          arma::vec grad_ij = R.at(i, j) * DlnCovDtheta(dX.col(i * n + j), theta);
          gradR.tube(i, j) = grad_ij;
          gradR.tube(j, i) = grad_ij;
        }
      }
      total_time += timer.elapsed_ms();
    }
    
    std::cout << "Method 3 (column-wise):            " << total_time / n_runs << " ms" << std::endl;
  }
}

// Main covariance matrix benchmark
void benchmark_cov_matrix(arma::uword n, arma::uword d, int n_runs = 5) {
  std::cout << "\n=== Covariance Matrix Computation Benchmark (n=" << n << ", d=" << d << ") ===" << std::endl;
  
  // Generate random data
  arma::mat X = arma::randn<arma::mat>(n, d);
  arma::vec theta = arma::vec(d).fill(0.5);
  
  // Simple Gaussian covariance for testing
  auto Cov = [](const arma::vec& dx, const arma::vec& theta) {
    double sum = 0.0;
    for (arma::uword i = 0; i < dx.n_elem; i++) {
      double r = dx[i] / theta[i];
      sum += r * r;
    }
    return std::exp(-0.5 * sum);
  };
  
  std::vector<std::pair<std::string, std::function<void(arma::mat&, const arma::mat&, const arma::vec&, const std::function<double(const arma::vec&, const arma::vec&)>&)>>> methods = {
    {"Method 1 (current nested loops)", method1_nested_loops},
    {"Method 2 (vectorized)", method2_vectorized},
    {"Method 3 (precompute diff)", method3_precompute_diff},
    {"Method 4 (OpenMP)", method4_openmp},
    {"Method 5 (cache-friendly)", method5_cache_friendly},
    {"Method 6 (blocked, bs=64)", [](arma::mat& R, const arma::mat& X, const arma::vec& theta, const auto& Cov) { method6_blocked(R, X, theta, Cov, 64); }},
  };
  
  // Reference result
  arma::mat R_ref(n, n);
  method1_nested_loops(R_ref, X, theta, Cov);
  
  for (const auto& [name, method] : methods) {
    Timer timer;
    double total_time = 0.0;
    arma::mat R(n, n);
    
    for (int run = 0; run < n_runs; run++) {
      R.zeros();
      timer.reset();
      method(R, X, theta, Cov);
      total_time += timer.elapsed_ms();
    }
    
    double max_diff = arma::max(arma::max(arma::abs(R - R_ref)));
    std::cout << name << ": " << total_time / n_runs << " ms"
              << " (max diff: " << max_diff << ")" << std::endl;
  }
}

int main(int argc, char* argv[]) {
  std::cout << "Covariance Computation Optimization Benchmark" << std::endl;
  std::cout << "=============================================" << std::endl;
  
  // Parse arguments
  arma::uword n = 100;  // number of points
  arma::uword d = 2;    // dimension
  int n_runs = 5;
  
  if (argc > 1) n = std::stoi(argv[1]);
  if (argc > 2) d = std::stoi(argv[2]);
  if (argc > 3) n_runs = std::stoi(argv[3]);
  
  std::cout << "Parameters: n=" << n << ", d=" << d << ", runs=" << n_runs << std::endl;
  
  // Run benchmarks for different sizes
  std::vector<arma::uword> sizes = {50, 100, 200, 500};
  
  for (arma::uword size : sizes) {
    if (size <= n) {
      benchmark_cov_matrix(size, d, n_runs);
      benchmark_gradR_computation(size, d, n_runs);
    }
  }
  
  return 0;
}
