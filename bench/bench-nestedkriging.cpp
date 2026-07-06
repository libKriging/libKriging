// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES
#include <cmath>
// clang-format on

#include "libKriging/Bench.hpp"
#include "libKriging/Kriging.hpp"
#include "libKriging/NestedKriging.hpp"
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
  // groups of ~100 points: the typical divide-and-conquer regime
  const arma::uword p = std::max<arma::uword>(2, (n_train + 99) / 100);

  std::cout << "\n";
  std::cout << "n=" << n_train << " d=" << d << " iterations=" << n_iterations << " (nb_groups=" << p << ")"
            << std::endl;

  arma::arma_rng::set_seed(123);
  arma::mat X_train(n_train, d, arma::fill::randu);
  arma::colvec y_train(n_train);
  for (arma::uword i = 0; i < n_train; ++i)
    y_train(i) = test_function(X_train.row(i));

  const arma::uword n_pred = 100;
  arma::mat X_pred(n_pred, d, arma::fill::randu);

  std::vector<double> fit_times, predict_times, fit_gpoe_times, predict_gpoe_times;

  for (int iter = 0; iter < n_iterations; ++iter) {
    // --- NK (optimal aggregation, default) ---
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      NestedKriging nk(y_train, X_train, "gauss", p, NestedKriging::Aggregation::NK);
      auto t1 = std::chrono::high_resolution_clock::now();
      fit_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

      t0 = std::chrono::high_resolution_clock::now();
      auto [mean, stdev] = nk.predict(X_pred, true);
      t1 = std::chrono::high_resolution_clock::now();
      predict_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    // --- gPoE (cheap product-of-experts aggregation) ---
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      NestedKriging nk(y_train, X_train, "gauss", p, NestedKriging::Aggregation::gPoE);
      auto t1 = std::chrono::high_resolution_clock::now();
      fit_gpoe_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

      t0 = std::chrono::high_resolution_clock::now();
      auto [mean, stdev] = nk.predict(X_pred, true);
      t1 = std::chrono::high_resolution_clock::now();
      predict_gpoe_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
  }

  print_header();
  print_stats("fit", compute_stats(fit_times));
  print_stats("predict", compute_stats(predict_times));
  print_stats("fit_gpoe", compute_stats(fit_gpoe_times));
  print_stats("predict_gpoe", compute_stats(predict_gpoe_times));
}

int main(int argc, char* argv[]) {
  std::cout << "NestedKriging Benchmark (BFGS, LL objective, NK & gPoE aggregations)" << std::endl;
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
