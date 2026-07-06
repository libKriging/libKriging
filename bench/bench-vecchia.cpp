// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES
#include <cmath>
// clang-format on

#include "libKriging/Bench.hpp"
#include "libKriging/Kriging.hpp"
#include "libKriging/utils/lk_armadillo.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

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
  std::cout << std::setw(25) << std::left << "Operation"
            << " | ";
  std::cout << std::setw(10) << std::right << "Mean (ms)"
            << " | ";
  std::cout << std::setw(10) << std::right << "Std (ms)"
            << " | ";
  std::cout << std::setw(10) << std::right << "Min (ms)"
            << " | ";
  std::cout << std::setw(10) << std::right << "Max (ms)"
            << " | ";
  std::cout << std::setw(10) << std::right << "Median (ms)" << std::endl;
}

// Synthetic test function: sum_i sin(2*pi*x_i)
double test_function(const arma::rowvec& x) {
  double sum = 0.0;
  for (arma::uword i = 0; i < x.n_elem; i++) {
    sum += std::sin(2.0 * M_PI * x(i));
  }
  return sum;
}

void benchmark_configuration(arma::uword n_train, arma::uword d, int n_iterations) {
  std::cout << "\n";
  std::cout << "n=" << n_train << " d=" << d << " iterations=" << n_iterations << " (VLL(30))" << std::endl;

  arma::arma_rng::set_seed(123);
  arma::mat X_train(n_train, d, arma::fill::randu);
  arma::colvec y_train(n_train);
  for (arma::uword i = 0; i < n_train; ++i)
    y_train(i) = test_function(X_train.row(i));

  // exact-LL comparison rows become prohibitive beyond a few thousand points
  const bool with_exact = (n_train <= 1000);

  std::vector<double> fit_times, fit_ll_times, vll_eval_times, vll_grad_times, ll_eval_times, ll_grad_times;

  // fitted VLL model reused for the objective-evaluation rows
  Kriging k_eval(y_train, X_train, "gauss", Trend::RegressionModel::Constant, false, "BFGS", "VLL(30)");
  const arma::vec theta_eval(d, arma::fill::value(0.3));

  for (int iter = 0; iter < n_iterations; ++iter) {
    // --- fit with the Vecchia objective ---
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      Kriging kr(y_train, X_train, "gauss", Trend::RegressionModel::Constant, false, "BFGS", "VLL(30)");
      auto t1 = std::chrono::high_resolution_clock::now();
      fit_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    // --- fit with the exact LL (comparison baseline) ---
    if (with_exact) {
      auto t0 = std::chrono::high_resolution_clock::now();
      Kriging kr(y_train, X_train, "gauss", Trend::RegressionModel::Constant, false, "BFGS", "LL");
      auto t1 = std::chrono::high_resolution_clock::now();
      fit_ll_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    // --- single objective evaluations at fixed theta ---
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      k_eval.logLikelihoodVecchiaFun(theta_eval, false);
      auto t1 = std::chrono::high_resolution_clock::now();
      vll_eval_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

      t0 = std::chrono::high_resolution_clock::now();
      k_eval.logLikelihoodVecchiaFun(theta_eval, true);
      t1 = std::chrono::high_resolution_clock::now();
      vll_grad_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    if (with_exact) {
      auto t0 = std::chrono::high_resolution_clock::now();
      k_eval.logLikelihoodFun(theta_eval, false, false);
      auto t1 = std::chrono::high_resolution_clock::now();
      ll_eval_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

      t0 = std::chrono::high_resolution_clock::now();
      k_eval.logLikelihoodFun(theta_eval, true, false);
      t1 = std::chrono::high_resolution_clock::now();
      ll_grad_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
  }

  print_header();
  print_stats("fit", compute_stats(fit_times));
  if (with_exact)
    print_stats("fit_exact_ll", compute_stats(fit_ll_times));
  print_stats("vll_eval", compute_stats(vll_eval_times));
  print_stats("vll_grad", compute_stats(vll_grad_times));
  if (with_exact) {
    print_stats("ll_eval", compute_stats(ll_eval_times));
    print_stats("ll_grad", compute_stats(ll_grad_times));
  }
}

int main(int argc, char* argv[]) {
  std::cout << "Vecchia Benchmark (objective=VLL(30) vs exact LL)" << std::endl;
  int n_iterations = 10;
  arma::uword n_points = 100;
  arma::uword d_dims = 4;
  if (argc > 1)
    n_iterations = std::atoi(argv[1]);
  if (argc > 2)
    n_points = std::atoi(argv[2]);
  if (argc > 3)
    d_dims = std::atoi(argv[3]);
  benchmark_configuration(n_points, d_dims, n_iterations);
  return 0;
}
