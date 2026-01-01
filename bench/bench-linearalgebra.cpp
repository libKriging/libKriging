// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES
#include <cmath>
// clang-format on

#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/Covariance.hpp"
#include "libKriging/utils/lk_armadillo.hpp"

#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <functional>

#ifdef _OPENMP
#include <omp.h>
#endif

// Statistics computation
struct Stats {
  double mean;
  double std;
  double min;
  double max;
  double median;
};

Stats compute_stats(const std::vector<double>& values) {
  if (values.empty()) {
    return {0.0, 0.0, 0.0, 0.0, 0.0};
  }

  double sum = std::accumulate(values.begin(), values.end(), 0.0);
  double mean = sum / values.size();

  double sq_sum = 0.0;
  for (double v : values) {
    sq_sum += (v - mean) * (v - mean);
  }
  double std = std::sqrt(sq_sum / values.size());

  std::vector<double> sorted = values;
  std::sort(sorted.begin(), sorted.end());

  double min = sorted.front();
  double max = sorted.back();
  double median = sorted[sorted.size() / 2];

  return {mean, std, min, max, median};
}

void print_stats(const std::string& operation, const Stats& stats) {
  std::cout << std::setw(25) << std::left << operation << " | ";
  std::cout << std::setw(10) << std::right << std::fixed << std::setprecision(3) << stats.mean << " | ";
  std::cout << std::setw(10) << std::right << std::fixed << std::setprecision(3) << stats.std << " | ";
  std::cout << std::setw(10) << std::right << std::fixed << std::setprecision(3) << stats.min << " | ";
  std::cout << std::setw(10) << std::right << std::fixed << std::setprecision(3) << stats.max << " | ";
  std::cout << std::setw(10) << std::right << std::fixed << std::setprecision(3) << stats.median << std::endl;
}

void print_header() {
  std::cout << std::setw(25) << std::left << "Operation" << " | ";
  std::cout << std::setw(10) << std::right << "Mean (ms)" << " | ";
  std::cout << std::setw(10) << std::right << "Std (ms)" << " | ";
  std::cout << std::setw(10) << std::right << "Min (ms)" << " | ";
  std::cout << std::setw(10) << std::right << "Max (ms)" << " | ";
  std::cout << std::setw(10) << std::right << "Median (ms)" << std::endl;
}

void benchmark_configuration(arma::uword n, arma::uword d, int n_iterations) {
  std::cout << "\n";
  std::cout << "n=" << n << " d=" << d << " iterations=" << n_iterations << std::endl;

  // Prepare test data similar to Kriging usage
  arma::mat X = arma::randu(d, n);  // Column-major: d × n (transposed from observations)
  arma::vec theta = arma::ones(d);

  // Gaussian covariance function (matching Kriging usage)
  auto gauss_cov = [](const arma::vec& diff, const arma::vec& theta) -> double {
    double sum_sq = 0.0;
    for (arma::uword i = 0; i < diff.n_elem; i++) {
      double val = diff[i] / theta[i];
      sum_sq += val * val;
    }
    return std::exp(-0.5 * sum_sq);
  };

  std::vector<double> times_covMat_sym_X;
  std::vector<double> times_covMat_rect;
  std::vector<double> times_solve;
  std::vector<double> times_rsolve;

  for (int iter = 0; iter < n_iterations; iter++) {
    // Benchmark covMat_sym_X (symmetric covariance matrix - used in fit())
    {
      arma::mat R(n, n);
      arma::vec diag_vec;  // empty = use factor

      auto t1 = std::chrono::high_resolution_clock::now();
      LinearAlgebra::covMat_sym_X(&R, X, theta, gauss_cov, 1.0, diag_vec);
      auto t2 = std::chrono::high_resolution_clock::now();

      double elapsed = std::chrono::duration<double, std::milli>(t2 - t1).count();
      times_covMat_sym_X.push_back(elapsed);
    }

    // Benchmark covMat_rect (rectangular covariance - used in predict())
    {
      arma::uword n_pred = std::min(n, arma::uword(100));  // Typical prediction size
      arma::mat X_pred = arma::randu(d, n_pred);
      arma::mat R_rect(n, n_pred);

      auto t1 = std::chrono::high_resolution_clock::now();
      LinearAlgebra::covMat_rect(&R_rect, X, X_pred, theta, gauss_cov, 1.0);
      auto t2 = std::chrono::high_resolution_clock::now();

      double elapsed = std::chrono::duration<double, std::milli>(t2 - t1).count();
      times_covMat_rect.push_back(elapsed);
    }

    // Benchmark solve (used extensively in Kriging)
    {
      arma::mat A = arma::randu(n, n);
      A = A * A.t() + arma::eye(n, n) * 0.1;  // Make positive definite
      arma::mat B = arma::randu(n, 10);

      auto t1 = std::chrono::high_resolution_clock::now();
      arma::mat X_sol = LinearAlgebra::solve(A, B);
      auto t2 = std::chrono::high_resolution_clock::now();

      double elapsed = std::chrono::duration<double, std::milli>(t2 - t1).count();
      times_solve.push_back(elapsed);
    }

    // Benchmark rsolve (right-solve, used in Kriging)
    {
      arma::mat A = arma::randu(n, n);
      A = A * A.t() + arma::eye(n, n) * 0.1;  // Make positive definite
      arma::mat B = arma::randu(10, n);

      auto t1 = std::chrono::high_resolution_clock::now();
      arma::mat X_sol = LinearAlgebra::rsolve(A, B);
      auto t2 = std::chrono::high_resolution_clock::now();

      double elapsed = std::chrono::duration<double, std::milli>(t2 - t1).count();
      times_rsolve.push_back(elapsed);
    }
  }

  // Print results
  print_header();
  print_stats("covMat_sym_X", compute_stats(times_covMat_sym_X));
  print_stats("covMat_rect", compute_stats(times_covMat_rect));
  print_stats("solve", compute_stats(times_solve));
  print_stats("rsolve", compute_stats(times_rsolve));
  std::cout << std::endl;
}

int main(int argc, char* argv[]) {
  int n_iterations = 10;
  if (argc > 1) {
    n_iterations = std::atoi(argv[1]);
  }

  std::cout << "=============================================================================" << std::endl;
  std::cout << "              LinearAlgebra Performance Benchmark (OpenMP)                   " << std::endl;
  std::cout << "=============================================================================" << std::endl;

#ifdef _OPENMP
  int max_threads = omp_get_max_threads();
  int optimal_threads = (max_threads > 2) ? 2 : max_threads;
  std::cout << "OpenMP enabled: max_threads=" << max_threads
            << ", using optimal=" << optimal_threads << std::endl;
#else
  std::cout << "OpenMP disabled: serial execution" << std::endl;
#endif

  std::cout << "Testing covariance matrix operations with Kriging-realistic matrices" << std::endl;
  std::cout << "=============================================================================" << std::endl;

  // Test various sizes relevant to Kriging
  benchmark_configuration(100, 2, n_iterations);
  benchmark_configuration(100, 4, n_iterations);
  benchmark_configuration(200, 2, n_iterations);
  benchmark_configuration(200, 4, n_iterations);
  benchmark_configuration(500, 2, n_iterations);
  benchmark_configuration(500, 4, n_iterations);
  benchmark_configuration(1000, 2, n_iterations);
  benchmark_configuration(1000, 4, n_iterations);

  std::cout << "=============================================================================" << std::endl;
  std::cout << "Benchmark complete" << std::endl;
  std::cout << "=============================================================================" << std::endl;

  return 0;
}
