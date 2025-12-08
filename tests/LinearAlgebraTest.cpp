// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio
#include <cmath>
// clang-format on

#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include "libKriging/LinearAlgebra.hpp"

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

// Helper function to create a positive definite matrix
arma::mat make_pd_matrix(arma::uword n, unsigned int seed) {
  arma::arma_rng::set_seed(seed);
  arma::mat A = arma::randn<arma::mat>(n, n);
  arma::mat PD = A * A.t() + n * arma::eye<arma::mat>(n, n);
  return PD;
}

// Test safe_chol_lower with threading
TEST_CASE("LinearAlgebra::safe_chol_lower - single thread", "[LinearAlgebra]") {
  arma::mat M = make_pd_matrix(10, 42);
  arma::mat L = LinearAlgebra::safe_chol_lower(M);
  
  REQUIRE(L.n_rows == 10);
  REQUIRE(L.n_cols == 10);
  
  // Check that L*L' = M
  arma::mat reconstructed = L * L.t();
  REQUIRE(arma::approx_equal(reconstructed, M, "absdiff", 1e-10));
}

TEST_CASE("LinearAlgebra::safe_chol_lower - concurrent access", "[LinearAlgebra][threading]") {
  const int n_threads = 20;
  const int n_size = 50;
  
  std::vector<std::thread> threads;
  std::vector<int> results(n_threads, 0);
  std::vector<arma::mat> inputs(n_threads);
  
  // Create different matrices for each thread
  for (int i = 0; i < n_threads; i++) {
    inputs[i] = make_pd_matrix(n_size, 100 + i);
  }
  
  // Launch threads
  for (int i = 0; i < n_threads; i++) {
    threads.emplace_back([i, &inputs, &results]() {
      try {
        arma::mat L = LinearAlgebra::safe_chol_lower(inputs[i]);
        arma::mat reconstructed = L * L.t();
        results[i] = arma::approx_equal(reconstructed, inputs[i], "absdiff", 1e-8);
      } catch (...) {
        results[i] = 0;
      }
    });
  }
  
  // Wait for all threads
  for (auto& t : threads) {
    t.join();
  }
  
  // Check all succeeded
  for (int i = 0; i < n_threads; i++) {
    REQUIRE(results[i]);
  }
}

TEST_CASE("LinearAlgebra::safe_chol_lower - varying sizes", "[LinearAlgebra][threading]") {
  const int n_threads = 20;
  std::vector<int> sizes = {5, 10, 20, 30, 50, 100};
  
  for (auto size : sizes) {
    std::vector<std::thread> threads;
    std::vector<int> results(n_threads, 0);
    std::vector<arma::mat> inputs(n_threads);
    
    for (int i = 0; i < n_threads; i++) {
      inputs[i] = make_pd_matrix(size, 200 + i * n_threads + size);
    }
    
    for (int i = 0; i < n_threads; i++) {
      threads.emplace_back([i, &inputs, &results]() {
        try {
          arma::mat L = LinearAlgebra::safe_chol_lower(inputs[i]);
          arma::mat reconstructed = L * L.t();
          results[i] = arma::approx_equal(reconstructed, inputs[i], "absdiff", 1e-8);
        } catch (...) {
          results[i] = 0;
        }
      });
    }
    
    for (auto& t : threads) {
      t.join();
    }
    
    for (int i = 0; i < n_threads; i++) {
      REQUIRE(results[i]);
    }
  }
}

TEST_CASE("LinearAlgebra::solve - concurrent access", "[LinearAlgebra][threading]") {
  const int n_threads = 20;
  const int n_size = 50;
  
  std::vector<std::thread> threads;
  std::vector<int> results(n_threads, 0);
  std::vector<arma::mat> A_matrices(n_threads);
  std::vector<arma::mat> B_matrices(n_threads);
  
  for (int i = 0; i < n_threads; i++) {
    A_matrices[i] = make_pd_matrix(n_size, 300 + i);
    arma::arma_rng::set_seed(400 + i);
    B_matrices[i] = arma::randn<arma::mat>(n_size, 5);
  }
  
  for (int i = 0; i < n_threads; i++) {
    threads.emplace_back([i, &A_matrices, &B_matrices, &results]() {
      try {
        arma::mat X = LinearAlgebra::solve(A_matrices[i], B_matrices[i]);
        arma::mat reconstructed = A_matrices[i] * X;
        results[i] = arma::approx_equal(reconstructed, B_matrices[i], "absdiff", 1e-8);
      } catch (...) {
        results[i] = 0;
      }
    });
  }
  
  for (auto& t : threads) {
    t.join();
  }
  
  for (int i = 0; i < n_threads; i++) {
    REQUIRE(results[i]);
  }
}

TEST_CASE("LinearAlgebra::rsolve - concurrent access", "[LinearAlgebra][threading]") {
  const int n_threads = 20;
  const int n_size = 50;
  
  std::vector<std::thread> threads;
  std::vector<int> results(n_threads, 0);
  std::vector<arma::mat> A_matrices(n_threads);
  std::vector<arma::mat> B_matrices(n_threads);
  
  for (int i = 0; i < n_threads; i++) {
    A_matrices[i] = make_pd_matrix(n_size, 500 + i);
    arma::arma_rng::set_seed(600 + i);
    B_matrices[i] = arma::randn<arma::mat>(5, n_size);
  }
  
  for (int i = 0; i < n_threads; i++) {
    threads.emplace_back([i, &A_matrices, &B_matrices, &results]() {
      try {
        arma::mat X = LinearAlgebra::rsolve(A_matrices[i], B_matrices[i]);
        arma::mat reconstructed = X * A_matrices[i];
        results[i] = arma::approx_equal(reconstructed, B_matrices[i], "absdiff", 1e-8);
      } catch (...) {
        results[i] = 0;
      }
    });
  }
  
  for (auto& t : threads) {
    t.join();
  }
  
  for (int i = 0; i < n_threads; i++) {
    REQUIRE(results[i]);
  }
}

TEST_CASE("LinearAlgebra::crossprod - concurrent access", "[LinearAlgebra][threading]") {
  const int n_threads = 20;
  const int n_rows = 100;
  const int n_cols = 50;
  
  std::vector<std::thread> threads;
  std::vector<int> results(n_threads, 0);
  std::vector<arma::mat> matrices(n_threads);
  
  for (int i = 0; i < n_threads; i++) {
    arma::arma_rng::set_seed(700 + i);
    matrices[i] = arma::randn<arma::mat>(n_rows, n_cols);
  }
  
  for (int i = 0; i < n_threads; i++) {
    threads.emplace_back([i, &matrices, &results]() {
      try {
        arma::mat result = LinearAlgebra::crossprod(matrices[i]);
        arma::mat expected = matrices[i].t() * matrices[i];
        results[i] = arma::approx_equal(result, expected, "absdiff", 1e-10);
      } catch (...) {
        results[i] = 0;
      }
    });
  }
  
  for (auto& t : threads) {
    t.join();
  }
  
  for (int i = 0; i < n_threads; i++) {
    REQUIRE(results[i]);
  }
}

TEST_CASE("LinearAlgebra::tcrossprod - concurrent access", "[LinearAlgebra][threading]") {
  const int n_threads = 20;
  const int n_rows = 50;
  const int n_cols = 100;
  
  std::vector<std::thread> threads;
  std::vector<int> results(n_threads, 0);
  std::vector<arma::mat> matrices(n_threads);
  
  for (int i = 0; i < n_threads; i++) {
    arma::arma_rng::set_seed(800 + i);
    matrices[i] = arma::randn<arma::mat>(n_rows, n_cols);
  }
  
  for (int i = 0; i < n_threads; i++) {
    threads.emplace_back([i, &matrices, &results]() {
      try {
        arma::mat result = LinearAlgebra::tcrossprod(matrices[i]);
        arma::mat expected = matrices[i] * matrices[i].t();
        results[i] = arma::approx_equal(result, expected, "absdiff", 1e-10);
      } catch (...) {
        results[i] = 0;
      }
    });
  }
  
  for (auto& t : threads) {
    t.join();
  }
  
  for (int i = 0; i < n_threads; i++) {
    REQUIRE(results[i]);
  }
}

TEST_CASE("LinearAlgebra::diagABA - concurrent access", "[LinearAlgebra][threading]") {
  const int n_threads = 20;
  const int n_size = 50;
  
  std::vector<std::thread> threads;
  std::vector<int> results(n_threads, 0);
  std::vector<arma::mat> A_matrices(n_threads);
  std::vector<arma::mat> B_matrices(n_threads);
  
  for (int i = 0; i < n_threads; i++) {
    arma::arma_rng::set_seed(900 + i);
    A_matrices[i] = arma::randn<arma::mat>(n_size, n_size);
    B_matrices[i] = make_pd_matrix(n_size, 1000 + i);
  }
  
  for (int i = 0; i < n_threads; i++) {
    threads.emplace_back([i, &A_matrices, &B_matrices, &results]() {
      try {
        arma::colvec result = LinearAlgebra::diagABA(A_matrices[i], B_matrices[i]);
        arma::mat ABA = A_matrices[i] * B_matrices[i] * A_matrices[i].t();
        arma::colvec expected = ABA.diag();
        results[i] = arma::approx_equal(result, expected, "absdiff", 1e-10);
      } catch (...) {
        results[i] = 0;
      }
    });
  }
  
  for (auto& t : threads) {
    t.join();
  }
  
  for (int i = 0; i < n_threads; i++) {
    REQUIRE(results[i]);
  }
}

TEST_CASE("LinearAlgebra::chol_block - concurrent access", "[LinearAlgebra][threading]") {
  const int n_threads = 20;
  const int n_old = 30;
  const int n_new = 20;
  const int n_total = n_old + n_new;
  
  std::vector<std::thread> threads;
  std::vector<int> results(n_threads, 0);
  std::vector<arma::mat> C_matrices(n_threads);
  std::vector<arma::mat> Loo_matrices(n_threads);
  
  for (int i = 0; i < n_threads; i++) {
    C_matrices[i] = make_pd_matrix(n_total, 1100 + i);
    arma::mat C_old = C_matrices[i].submat(0, 0, n_old - 1, n_old - 1);
    Loo_matrices[i] = arma::chol(C_old, "lower");
  }
  
  for (int i = 0; i < n_threads; i++) {
    threads.emplace_back([i, n_old, &C_matrices, &Loo_matrices, &results]() {
      try {
        arma::mat L = LinearAlgebra::chol_block(C_matrices[i], Loo_matrices[i]);
        arma::mat reconstructed = L * L.t();
        results[i] = arma::approx_equal(reconstructed, C_matrices[i], "absdiff", 1e-8);
      } catch (...) {
        results[i] = 0;
      }
    });
  }
  
  for (auto& t : threads) {
    t.join();
  }
  
  for (int i = 0; i < n_threads; i++) {
    REQUIRE(results[i]);
  }
}

TEST_CASE("LinearAlgebra - stress test with mixed operations", "[LinearAlgebra][threading][stress]") {
  const int n_threads = 40;
  const int n_iterations = 10;
  
  std::vector<std::thread> threads;
  std::vector<int> results(n_threads, 0);
  
  for (int i = 0; i < n_threads; i++) {
    threads.emplace_back([i, n_iterations, &results]() {
      try {
        bool all_ok = true;
        for (int iter = 0; iter < n_iterations; iter++) {
          int size = 20 + (i * n_iterations + iter) % 30;
          
          // Test safe_chol_lower
          arma::mat M = make_pd_matrix(size, 1200 + i * n_iterations + iter);
          arma::mat L = LinearAlgebra::safe_chol_lower(M);
          arma::mat reconstructed = L * L.t();
          if (!arma::approx_equal(reconstructed, M, "absdiff", 1e-8)) {
            all_ok = false;
            break;
          }
          
          // Test solve
          arma::arma_rng::set_seed(1300 + i * n_iterations + iter);
          arma::mat B = arma::randn<arma::mat>(size, 5);
          arma::mat X = LinearAlgebra::solve(M, B);
          arma::mat check = M * X;
          if (!arma::approx_equal(check, B, "absdiff", 1e-8)) {
            all_ok = false;
            break;
          }
          
          // Test crossprod
          arma::arma_rng::set_seed(1400 + i * n_iterations + iter);
          arma::mat A = arma::randn<arma::mat>(size * 2, size);
          arma::mat cp = LinearAlgebra::crossprod(A);
          arma::mat expected_cp = A.t() * A;
          if (!arma::approx_equal(cp, expected_cp, "absdiff", 1e-10)) {
            all_ok = false;
            break;
          }
        }
        results[i] = all_ok;
      } catch (...) {
        results[i] = 0;
      }
    });
  }
  
  for (auto& t : threads) {
    t.join();
  }
  
  for (int i = 0; i < n_threads; i++) {
    REQUIRE(results[i]);
  }
}

TEST_CASE("LinearAlgebra - rapid fire varying sizes", "[LinearAlgebra][threading][stress]") {
  const int n_threads = 50;
  std::vector<int> sizes = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
  
  std::vector<std::thread> threads;
  std::vector<int> results(n_threads, 0);
  
  for (int i = 0; i < n_threads; i++) {
    threads.emplace_back([i, &sizes, &results]() {
      try {
        int size = sizes[i % sizes.size()];
        arma::mat M = make_pd_matrix(size, 1500 + i);
        arma::mat L = LinearAlgebra::safe_chol_lower(M);
        
        arma::arma_rng::set_seed(1600 + i);
        arma::mat A = arma::randn<arma::mat>(size * 2, size);
        arma::mat cp = LinearAlgebra::crossprod(A);
        arma::mat tcp = LinearAlgebra::tcrossprod(A.t());
        
        arma::mat B = arma::randn<arma::mat>(size, 3);
        arma::mat X = LinearAlgebra::solve(M, B);
        
        arma::mat reconstructed = L * L.t();
        arma::mat check_solve = M * X;
        arma::mat expected_cp = A.t() * A;
        arma::mat expected_tcp = A.t() * A;
        
        results[i] = arma::approx_equal(reconstructed, M, "absdiff", 1e-8) &&
                     arma::approx_equal(check_solve, B, "absdiff", 1e-8) &&
                     arma::approx_equal(cp, expected_cp, "absdiff", 1e-10) &&
                     arma::approx_equal(tcp, expected_tcp, "absdiff", 1e-10);
      } catch (...) {
        results[i] = 0;
      }
    });
  }
  
  for (auto& t : threads) {
    t.join();
  }
  
  for (int i = 0; i < n_threads; i++) {
    REQUIRE(results[i]);
  }
}

TEST_CASE("LinearAlgebra::safe_chol_lower - nearly singular matrix", "[LinearAlgebra][chol][edge]") {
  // Create a matrix that is nearly singular (very small eigenvalue)
  arma::mat M = make_pd_matrix(10, 42);
  
  // Make it nearly singular by reducing one eigenvalue
  arma::vec eigval;
  arma::mat eigvec;
  arma::eig_sym(eigval, eigvec, M);
  
  // Set the smallest eigenvalue to a very small value
  eigval(0) = 1e-12;
  
  // Reconstruct the matrix
  M = eigvec * arma::diagmat(eigval) * eigvec.t();
  
  // This should trigger the numerical nugget addition
  arma::mat L = LinearAlgebra::safe_chol_lower(M);
  
  REQUIRE(L.n_rows == 10);
  REQUIRE(L.n_cols == 10);
  
  // The Cholesky decomposition should succeed (with added nugget)
  arma::mat reconstructed = L * L.t();
  // Due to nugget addition, this won't be exact but should be close
  REQUIRE(arma::approx_equal(reconstructed.diag(), M.diag(), "reldiff", 1e-6));
}

TEST_CASE("LinearAlgebra::safe_chol_lower - matrix with small negative eigenvalue", "[LinearAlgebra][chol][edge]") {
  // Create a matrix that should be PD but has a tiny negative eigenvalue due to numerical errors
  arma::mat M = make_pd_matrix(8, 123);
  
  arma::vec eigval;
  arma::mat eigvec;
  arma::eig_sym(eigval, eigvec, M);
  
  // Make one eigenvalue slightly negative (simulating numerical error)
  eigval(0) = -1e-14;
  
  // Reconstruct
  M = eigvec * arma::diagmat(eigval) * eigvec.t();
  
  // Should handle this gracefully with numerical nugget
  REQUIRE_NOTHROW([&]() {
    arma::mat L = LinearAlgebra::safe_chol_lower(M);
    REQUIRE(L.n_rows == 8);
  }());
}

TEST_CASE("LinearAlgebra::safe_chol_lower - ill-conditioned matrix", "[LinearAlgebra][chol][edge]") {
  // Create an ill-conditioned matrix (large condition number)
  arma::mat M = make_pd_matrix(10, 456);
  
  arma::vec eigval;
  arma::mat eigvec;
  arma::eig_sym(eigval, eigvec, M);
  
  // Create large condition number: max/min eigenvalue ratio ~ 1e15
  eigval = arma::linspace<arma::vec>(1e-14, 1.0, 10);
  
  // Reconstruct
  M = eigvec * arma::diagmat(eigval) * eigvec.t();
  
  // Should handle with numerical nugget
  arma::mat L = LinearAlgebra::safe_chol_lower(M);
  
  REQUIRE(L.n_rows == 10);
  REQUIRE(L.n_cols == 10);
}

TEST_CASE("LinearAlgebra::safe_chol_lower - rank deficient approximation", "[LinearAlgebra][chol][edge]") {
  // Create a nearly rank-deficient matrix
  arma::mat A = arma::randn<arma::mat>(10, 3);
  arma::mat M = A * A.t();  // Rank 3 matrix
  
  // Add small diagonal to make it "almost" full rank
  M.diag() += 1e-13;
  
  // This should require multiple nugget additions
  arma::mat L = LinearAlgebra::safe_chol_lower(M);
  
  REQUIRE(L.n_rows == 10);
  REQUIRE(L.n_cols == 10);
}

TEST_CASE("LinearAlgebra::safe_chol_lower - correlation-like matrix near singular", "[LinearAlgebra][chol][edge]") {
  // Create a correlation-like matrix that's nearly singular (almost collinear data)
  const int n = 5;
  arma::mat M = arma::mat(n, n, arma::fill::ones);
  
  // Set diagonal to 1 (correlation matrix)
  M.diag().ones();
  
  // Make off-diagonal elements very close to 1 (almost collinear)
  M.fill(1.0-1e-20);
  M.diag().ones();

  // Enable warnings for nugget addition
  bool prev_warn_chol = LinearAlgebra::warn_chol;
  LinearAlgebra::warn_chol = true;
  // print rcond
  double rcond = LinearAlgebra::rcond_chol(M);
  arma::cout << "[INFO] rcond: " << rcond << arma::endl;

  // This should require nugget addition
  LinearAlgebra::chol_rcond_check = true;
  arma::mat L = LinearAlgebra::safe_chol_lower(M);

  // reset warnings for nugget addition
  LinearAlgebra::warn_chol = prev_warn_chol;

  REQUIRE(L.n_rows == n);
  REQUIRE(L.n_cols == n);
}

TEST_CASE("LinearAlgebra::safe_chol_lower - near-singular concurrent", "[LinearAlgebra][chol][edge][threading]") {
  const int n_threads = 10;
  const int n_size = 15;
  
  std::vector<std::thread> threads;
  std::vector<int> results(n_threads, 0);
  std::vector<arma::mat> inputs(n_threads);
  
  // Create different near-singular matrices for each thread
  for (int i = 0; i < n_threads; i++) {
    arma::mat M = make_pd_matrix(n_size, 2000 + i);
    
    // Make it near-singular
    arma::vec eigval;
    arma::mat eigvec;
    arma::eig_sym(eigval, eigvec, M);
    
    // Set smallest eigenvalues to very small values
    eigval(0) = 1e-13;
    eigval(1) = 1e-12;
    
    inputs[i] = eigvec * arma::diagmat(eigval) * eigvec.t();
  }
  
  // Launch threads
  for (int i = 0; i < n_threads; i++) {
    threads.emplace_back([i, &inputs, &results]() {
      try {
        arma::mat L = LinearAlgebra::safe_chol_lower(inputs[i]);
        results[i] = (L.n_rows == inputs[i].n_rows);
      } catch (...) {
        results[i] = 0;
      }
    });
  }
  
  // Wait for all threads
  for (auto& t : threads) {
    t.join();
  }
  
  // Check all succeeded
  for (int i = 0; i < n_threads; i++) {
    REQUIRE(results[i]);
  }
}

TEST_CASE("LinearAlgebra::safe_chol_lower - extremely small eigenvalues", "[LinearAlgebra][chol][edge]") {
  // Test with eigenvalues spanning many orders of magnitude
  const int n = 10;
  arma::arma_rng::set_seed(789);
  arma::mat eigvec = arma::orth(arma::randn<arma::mat>(n, n));
  
  // Eigenvalues from 1e-16 to 1
  arma::vec eigval = arma::vec(n);
  for (int i = 0; i < n; i++) {
    eigval(i) = std::pow(10.0, -16.0 + i * 1.6);  // 1e-16 to 1
  }
  
  arma::mat M = eigvec * arma::diagmat(eigval) * eigvec.t();
  
  // Should handle with multiple nugget additions
  arma::mat L;
  REQUIRE_NOTHROW([&]() {
    L = LinearAlgebra::safe_chol_lower(M);
  }());
  
  REQUIRE(L.n_rows == n);
}

TEST_CASE("LinearAlgebra::safe_chol_lower - asymmetric numerical errors", "[LinearAlgebra][chol][edge]") {
  // Create a matrix that should be symmetric but has small asymmetric errors
  arma::mat M = make_pd_matrix(8, 999);
  
  // Add small asymmetric perturbation
  arma::arma_rng::set_seed(1111);
  arma::mat noise = arma::randn<arma::mat>(8, 8) * 1e-14;
  M += noise;
  
  // Symmetrize it (as would happen in practice)
  M = (M + M.t()) / 2.0;
  
  // Should still work
  arma::mat L = LinearAlgebra::safe_chol_lower(M);
  
  REQUIRE(L.n_rows == 8);
  REQUIRE(L.n_cols == 8);
}
