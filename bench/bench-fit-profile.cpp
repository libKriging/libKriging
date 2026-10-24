// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES
#include <cmath>
// clang-format on

#include "libKriging/Kriging.hpp"
#include "libKriging/utils/lk_armadillo.hpp"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

// Synthetic test function: sum_i sin(2*pi*x_i)
double test_function(const arma::rowvec& x) {
  double sum = 0.0;
  for (arma::uword i = 0; i < x.n_elem; i++) {
    sum += std::sin(2.0 * M_PI * x(i));
  }
  return sum;
}

// LK_BENCH_WARMUP_MS (default 300): busy-wait before measuring. OpenBLAS
// pthreads threads spin for ~100 ms after the library is loaded
// (OPENBLAS_THREAD_TIMEOUT); measured inside that window, a short run sees
// them as contention (bench/thread-budget/RESULTS.md, section 9).
static void bench_warmup() {
  const char* v = std::getenv("LK_BENCH_WARMUP_MS");
  const int ms = v ? std::atoi(v) : 300;
  const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
  volatile double x = 0;
  while (std::chrono::steady_clock::now() < end)
    x = x + 1;
}

int main(int argc, char* argv[]) {
  std::cout << "Kriging Fit Profile for LinearAlgebra Analysis\n\n";

  // Parse arguments
  arma::uword n_train = argc > 1 ? std::atoi(argv[1]) : 400;
  arma::uword d = argc > 2 ? std::atoi(argv[2]) : 4;
  std::string optim = argc > 3 ? argv[3] : "BFGS";

  std::cout << "n=" << n_train << " d=" << d << " optim=" << optim << "\n\n";

  // Set random seed
  arma::arma_rng::set_seed(123);

  // Prepare test data
  arma::mat X_train(n_train, d, arma::fill::randu);
  arma::colvec y_train(n_train);
  for (arma::uword i = 0; i < n_train; ++i) {
    y_train(i) = test_function(X_train.row(i));
  }

  // Run fit
  std::cout << "Running fit()...\n";

  Kriging kr("gauss");
  Kriging::Parameters params{std::nullopt, true, std::nullopt, true};

  bench_warmup();
  auto start = std::chrono::high_resolution_clock::now();
  kr.fit(y_train, X_train, Trend::RegressionModel::Constant, false, optim, "LL", params);
  auto end = std::chrono::high_resolution_clock::now();

  double total_ms = std::chrono::duration<double, std::milli>(end - start).count();

  std::cout << "\n=== TOTAL FIT TIME ===\n";
  std::cout << "Total: " << std::fixed << std::setprecision(2) << total_ms << " ms\n";
  std::cout << "\nTo profile LinearAlgebra calls, use:\n";
  std::cout << "  valgrind --tool=callgrind ./bench-fit-profile " << n_train << " " << d << "\n";
  std::cout << "  callgrind_annotate callgrind.out.<pid> | grep 'LinearAlgebra::'\n";

  return 0;
}
