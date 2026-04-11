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

// Synthetic test function: sum_i sin(2*pi*x_i)
// For categorical/ordinal: x(0) is an integer level; we map it to a center value.
static const arma::uword N_LEVELS = 5;

double test_function_continuous(const arma::rowvec& x) {
  double sum = 0.0;
  for (arma::uword i = 0; i < x.n_elem; i++) {
    sum += std::sin(2.0 * M_PI * x(i));
  }
  return sum;
}

double test_function_mixed(const arma::rowvec& x) {
  // x(0) is an integer level 0..N_LEVELS-1, map to center in [0,1]
  arma::vec centers = arma::linspace(0.1, 0.9, N_LEVELS);
  double sum = std::sin(2.0 * M_PI * centers(static_cast<arma::uword>(x(0))));
  for (arma::uword i = 1; i < x.n_elem; i++) {
    sum += std::sin(2.0 * M_PI * x(i));
  }
  return sum;
}

struct WarpConfig {
  std::string name;
  std::vector<std::string> warping;
  std::string optim;
  std::map<std::string, std::string> parameters;
  bool mixed_input;  // true if x(0) is categorical/ordinal (integer levels)
};

// Build warping spec for d dimensions
std::vector<std::string> repeat_warp(const std::string& warp, arma::uword d) {
  return std::vector<std::string>(d, warp);
}

std::vector<WarpConfig> build_configs(arma::uword d) {
  std::vector<WarpConfig> configs;

  // Per-dimension warpings (all continuous)
  configs.push_back({"none", repeat_warp("none", d), "BFGS", {}, false});
  configs.push_back({"affine", repeat_warp("affine", d), "BFGS+Adam", {}, false});
  configs.push_back({"boxcox", repeat_warp("boxcox", d), "BFGS+Adam", {}, false});
  configs.push_back({"kumaraswamy", repeat_warp("kumaraswamy", d), "BFGS+Adam", {}, false});
  configs.push_back({"mlp", repeat_warp("mlp", d), "BFGS+Adam", {}, false});
  configs.push_back({"neural_mono", repeat_warp("neural_mono", d), "BFGS+Adam", {}, false});

  // Joint MLP: single warp over all dimensions, output dim = d
  {
    std::string spec = "mlp_joint(16:8," + std::to_string(d) + ",selu)";
    configs.push_back({"mlp_joint", {spec}, "Adam", {}, false});
  }

  // Categorical: first dim = categorical(5,2), rest = kumaraswamy
  {
    std::vector<std::string> w;
    w.push_back("categorical(5,2)");
    for (arma::uword i = 1; i < d; ++i)
      w.push_back("kumaraswamy");
    configs.push_back({"categorical", w, "Adam", {}, true});
  }

  // Ordinal: first dim = ordinal(5), rest = kumaraswamy
  {
    std::vector<std::string> w;
    w.push_back("ordinal(5)");
    for (arma::uword i = 1; i < d; ++i)
      w.push_back("kumaraswamy");
    configs.push_back({"ordinal", w, "Adam", {}, true});
  }

  return configs;
}

void benchmark_warpkriging(const WarpConfig& config, arma::uword n_train, arma::uword d, int n_iterations) {
  std::cout << "\n";
  std::cout << "warpkriging:" << config.name << " optim=" << config.optim << " n=" << n_train << " d=" << d
            << " iterations=" << n_iterations << std::endl;

  arma::arma_rng::set_seed(123);

  // Prepare training data
  arma::mat X_train(n_train, d, arma::fill::randu);
  if (config.mixed_input) {
    // First column: integer levels 0..N_LEVELS-1
    for (arma::uword i = 0; i < n_train; ++i) {
      X_train(i, 0) = std::floor(X_train(i, 0) * N_LEVELS);
      if (X_train(i, 0) >= N_LEVELS)
        X_train(i, 0) = N_LEVELS - 1;
    }
  }

  arma::colvec y_train(n_train);
  auto test_fn = config.mixed_input ? test_function_mixed : test_function_continuous;
  for (arma::uword i = 0; i < n_train; ++i) {
    y_train(i) = test_fn(X_train.row(i));
  }

  // Prepare prediction data
  arma::uword n_pred = 100;
  arma::mat X_pred(n_pred, d, arma::fill::randu);
  if (config.mixed_input) {
    for (arma::uword i = 0; i < n_pred; ++i) {
      X_pred(i, 0) = std::floor(X_pred(i, 0) * N_LEVELS);
      if (X_pred(i, 0) >= N_LEVELS)
        X_pred(i, 0) = N_LEVELS - 1;
    }
  }

  // Storage for timing results
  std::vector<double> fit_times;
  std::vector<double> predict_times;
  std::vector<double> simulate_times;
  std::vector<double> ll_values;

  for (int iter = 0; iter < n_iterations; ++iter) {
    // Benchmark FIT
    libKriging::WarpKriging wk(config.warping, "gauss");
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
      auto [mean, stdev, cov, mean_deriv, stdev_deriv] = wk.predict(X_pred, true, false);
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
  // Usage: bench-warpkriging [iterations] [n] [d] [warping_filter] [optim_filter] [adam_iters]
  int n_iterations = 5;
  arma::uword n_points = 20;
  arma::uword d_dims = 2;
  std::string warping_filter = "";  // empty = all

  if (argc > 1)
    n_iterations = std::atoi(argv[1]);
  if (argc > 2)
    n_points = std::atoi(argv[2]);
  if (argc > 3)
    d_dims = std::atoi(argv[3]);
  if (argc > 4)
    warping_filter = argv[4];

  // Parse optional optim filter and adam_iters override
  std::string optim_filter = "";
  std::string adam_iters_override = "";
  if (argc > 5)
    optim_filter = argv[5];
  if (argc > 6)
    adam_iters_override = argv[6];

  // Build configs for the requested dimensionality
  std::vector<WarpConfig> configs = build_configs(d_dims);

  for (auto& config : configs) {
    if (!warping_filter.empty() && config.name != warping_filter)
      continue;
    // Override optim if specified
    if (!optim_filter.empty())
      config.optim = optim_filter;
    // Override adam iters if specified
    if (!adam_iters_override.empty())
      config.parameters["max_iter_adam"] = adam_iters_override;
    benchmark_warpkriging(config, n_points, d_dims, n_iterations);
  }

  return 0;
}
