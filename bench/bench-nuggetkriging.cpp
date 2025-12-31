// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES
#include <cmath>
// clang-format on

#include "libKriging/NuggetKriging.hpp"
#include "libKriging/Bench.hpp"
#include "libKriging/utils/lk_armadillo.hpp"

#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

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
  std::cout << "n=" << n_train << " d=" << d << " iterations=" << n_iterations << std::endl;

  // Set random seed for reproducibility
  arma::arma_rng::set_seed(123);

  // Prepare test data
  arma::mat X_train(n_train, d, arma::fill::randu);
  arma::colvec y_train(n_train);
  for (arma::uword i = 0; i < n_train; ++i) {
    y_train(i) = test_function(X_train.row(i));
  }

  // Prepare prediction data (100 points)
  arma::uword n_pred = 100;
  arma::mat X_pred(n_pred, d, arma::fill::randu);

  // Prepare update data (10% of training data)
  arma::uword n_update = std::max(static_cast<arma::uword>(1), n_train / 10);
  arma::mat X_update(n_update, d, arma::fill::randu);
  arma::colvec y_update(n_update);
  for (arma::uword i = 0; i < n_update; ++i) {
    y_update(i) = test_function(X_update.row(i));
  }

  // Storage for timing results
  std::vector<double> fit_times;
  std::vector<double> predict_times;
  std::vector<double> simulate_times;
  std::vector<double> update_times;
  std::vector<double> update_simulate_times;

  // Run benchmarks
  for (int iter = 0; iter < n_iterations; ++iter) {
    // Benchmark FIT
    {
      auto t0 = std::chrono::high_resolution_clock::now();

      NuggetKriging kr("gauss");
      NuggetKriging::Parameters params{std::nullopt, true, std::nullopt, true, std::nullopt, true, std::nullopt, true};
      // BFGS, no parallelization (multistart=1)
      kr.fit(y_train, X_train, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);

      auto t1 = std::chrono::high_resolution_clock::now();
      double duration = std::chrono::duration<double, std::milli>(t1 - t0).count();
      fit_times.push_back(duration);
    }

    // Create a fitted model for subsequent operations
    NuggetKriging kr("gauss");
    NuggetKriging::Parameters params{std::nullopt, true, std::nullopt, true, std::nullopt, true, std::nullopt, true};
    kr.fit(y_train, X_train, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);

    // Benchmark PREDICT
    {
      auto t0 = std::chrono::high_resolution_clock::now();

      auto [mean, stdev, cov, mean_deriv, stdev_deriv] = kr.predict(X_pred, true, true, true);

      auto t1 = std::chrono::high_resolution_clock::now();
      double duration = std::chrono::duration<double, std::milli>(t1 - t0).count();
      predict_times.push_back(duration);
    }

    // Benchmark SIMULATE
    {
      auto t0 = std::chrono::high_resolution_clock::now();

      int nsim = 100;
      int seed = 123 + iter;
      arma::mat y_sim = kr.simulate(nsim, seed, X_pred, true, false);  // with_nugget=true, will_update=false

      auto t1 = std::chrono::high_resolution_clock::now();
      double duration = std::chrono::duration<double, std::milli>(t1 - t0).count();
      simulate_times.push_back(duration);
    }

    // Benchmark UPDATE
    {
      auto t0 = std::chrono::high_resolution_clock::now();

      kr.update(y_update, X_update);

      auto t1 = std::chrono::high_resolution_clock::now();
      double duration = std::chrono::duration<double, std::milli>(t1 - t0).count();
      update_times.push_back(duration);
    }

    // Benchmark UPDATE + SIMULATE
    // Create fresh model for this test
    NuggetKriging kr2("gauss");
    kr2.fit(y_train, X_train, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);

    // First simulate to prepare the model
    int nsim = 100;
    int seed = 123 + iter;
    kr2.simulate(nsim, seed, X_pred, true, true);  // with_nugget=true, will_update=true

    {
      auto t0 = std::chrono::high_resolution_clock::now();

      arma::mat result = kr2.update_simulate(y_update, X_update);

      auto t1 = std::chrono::high_resolution_clock::now();
      double duration = std::chrono::duration<double, std::milli>(t1 - t0).count();
      update_simulate_times.push_back(duration);
    }
  }

  // Print results
  print_header();
  print_stats("fit", compute_stats(fit_times));
  print_stats("predict", compute_stats(predict_times));
  print_stats("simulate", compute_stats(simulate_times));
  print_stats("update", compute_stats(update_times));
  print_stats("update_simulate", compute_stats(update_simulate_times));
}

int main(int argc, char* argv[]) {
  std::cout << "NuggetKriging Benchmark (BFGS, LL objective)" << std::endl;

  // Parse command line arguments
  // Usage: bench-nuggetkriging [iterations] [n] [d]
  int n_iterations = 10;
  arma::uword n_points = 100;
  arma::uword d_dims = 4;

  if (argc > 1) {
    n_iterations = std::atoi(argv[1]);
  }
  if (argc > 2) {
    n_points = std::atoi(argv[2]);
  }
  if (argc > 3) {
    d_dims = std::atoi(argv[3]);
  }

  // Run benchmark with specified configuration
  benchmark_configuration(n_points, d_dims, n_iterations);

  return 0;
}
