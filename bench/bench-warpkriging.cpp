// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES
#include <cmath>
// clang-format on

#include "libKriging/WarpKriging.hpp"
#include "libKriging/utils/lk_armadillo.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

// Statistics computation (same as bench-kriging.cpp)
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

// Branin test function on [0,1]^2
double branin_scalar(const arma::rowvec& x) {
  double x1 = x(0) * 15.0 - 5.0;
  double x2 = x(1) * 15.0;
  return std::pow(x2 - 5.0 / (4.0 * M_PI * M_PI) * x1 * x1 + 5.0 / M_PI * x1 - 6.0, 2)
         + 10.0 * (1.0 - 1.0 / (8.0 * M_PI)) * std::cos(x1) + 10.0;
}

struct WarpConfig {
  std::string name;
  std::vector<std::string> warping;
  std::string optim;
  std::map<std::string, std::string> parameters;
};

void benchmark_warpkriging(const WarpConfig& config, arma::uword n_train, int n_iterations) {
  const arma::uword d = 2;  // Branin is 2D

  std::cout << "\n";
  std::cout << "warpkriging:" << config.name << " optim=" << config.optim << " n=" << n_train
            << " iterations=" << n_iterations << std::endl;

  arma::arma_rng::set_seed(123);

  // Prepare training data (LHS)
  arma::mat X_train(n_train, d, arma::fill::randu);
  arma::colvec y_train(n_train);
  for (arma::uword i = 0; i < n_train; ++i) {
    y_train(i) = branin_scalar(X_train.row(i));
  }

  // Prepare prediction data
  arma::uword n_pred = 100;
  arma::mat X_pred(n_pred, d, arma::fill::randu);

  // Storage for timing results
  std::vector<double> fit_times;
  std::vector<double> predict_times;
  std::vector<double> simulate_times;
  std::vector<double> ll_values;

  for (int iter = 0; iter < n_iterations; ++iter) {
    // Benchmark FIT
    libKriging::WarpKriging wk(config.warping, "matern5_2");
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      wk.fit(y_train, X_train, "constant", false, config.optim, "LL", config.parameters);
      auto t1 = std::chrono::high_resolution_clock::now();
      fit_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    ll_values.push_back(wk.logLikelihood());

    // Benchmark PREDICT
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      auto [mean, stdev, cov] = wk.predict(X_pred, true, false);
      auto t1 = std::chrono::high_resolution_clock::now();
      predict_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    // Benchmark SIMULATE
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      arma::mat sims = wk.simulate(10, 42 + iter, X_pred);
      auto t1 = std::chrono::high_resolution_clock::now();
      simulate_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
  }

  // Print results
  print_header();
  print_stats("fit", compute_stats(fit_times));
  print_stats("predict", compute_stats(predict_times));
  print_stats("simulate", compute_stats(simulate_times));

  // Print LL summary
  auto ll_stats = compute_stats(ll_values);
  std::cout << std::setw(25) << std::left << "logLikelihood"
            << " | " << std::fixed << std::setprecision(4) << ll_stats.mean << std::endl;
}

int main(int argc, char* argv[]) {
  std::cout << "WarpKriging Benchmark" << std::endl;

  // Parse command line arguments
  // Usage: bench-warpkriging [iterations] [n] [warping_filter]
  int n_iterations = 5;
  arma::uword n_points = 20;
  std::string warping_filter = "";  // empty = all

  if (argc > 1)
    n_iterations = std::atoi(argv[1]);
  if (argc > 2)
    n_points = std::atoi(argv[2]);
  if (argc > 3)
    warping_filter = argv[3];

  // Define warp configurations to benchmark
  // Parse optional optim filter (4th arg) and adam_iters override (5th arg)
  std::string optim_filter = "";
  std::string adam_iters_override = "";
  if (argc > 4)
    optim_filter = argv[4];
  if (argc > 5)
    adam_iters_override = argv[5];

  std::vector<WarpConfig> configs = {
      {"none", {"none", "none"}, "BFGS", {}},

      {"affine", {"affine", "affine"}, "BFGS+Adam", {}},

      {"boxcox", {"boxcox", "boxcox"}, "BFGS+Adam", {}},

      {"kumaraswamy", {"kumaraswamy", "kumaraswamy"}, "BFGS+Adam", {}},

      {"mlp", {"mlp", "mlp"}, "BFGS+Adam", {}},

      {"neural_mono", {"neural_mono", "neural_mono"}, "BFGS+Adam", {}},
  };

  for (auto& config : configs) {
    if (!warping_filter.empty() && config.name != warping_filter)
      continue;
    // Override optim if specified
    if (!optim_filter.empty())
      config.optim = optim_filter;
    // Override adam iters if specified
    if (!adam_iters_override.empty())
      config.parameters["max_iter_adam"] = adam_iters_override;
    benchmark_warpkriging(config, n_points, n_iterations);
  }

  return 0;
}
