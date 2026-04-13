// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES
#include <cmath>
// clang-format on

#include "libKriging/MLPKriging.hpp"
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
double test_function(const arma::rowvec& x) {
  double sum = 0.0;
  for (arma::uword i = 0; i < x.n_elem; i++) {
    sum += std::sin(2.0 * M_PI * x(i));
  }
  return sum;
}

struct MLPConfig {
  std::string name;
  std::vector<arma::uword> hidden_dims;
  arma::uword d_out;
  std::string activation;
  std::string optim;
  std::map<std::string, std::string> parameters;
};

std::vector<MLPConfig> build_configs() {
  std::vector<MLPConfig> configs;

  configs.push_back({"small_selu", {16, 8}, 2, "selu", "BFGS+Adam", {}});
  configs.push_back({"medium_selu", {32, 16}, 3, "selu", "BFGS+Adam", {}});
  configs.push_back({"large_selu", {64, 32, 16}, 4, "selu", "BFGS+Adam", {}});
  configs.push_back({"small_tanh", {16, 8}, 2, "tanh", "BFGS+Adam", {}});
  configs.push_back({"adam_only", {16, 8}, 2, "selu", "Adam", {}});

  return configs;
}

void benchmark_mlpkriging(const MLPConfig& config,
                          arma::uword n_train,
                          arma::uword d,
                          int n_iterations) {
  // Build hidden_dims string for display
  std::string hdims_str;
  for (size_t i = 0; i < config.hidden_dims.size(); ++i) {
    if (i > 0)
      hdims_str += ":";
    hdims_str += std::to_string(config.hidden_dims[i]);
  }

  std::cout << "\n";
  std::cout << "mlpkriging:" << config.name << " hidden=[" << hdims_str << "] d_out=" << config.d_out
            << " activation=" << config.activation << " optim=" << config.optim << " n=" << n_train << " d=" << d
            << " iterations=" << n_iterations << std::endl;

  arma::arma_rng::set_seed(123);

  // Prepare training data
  arma::mat X_train(n_train, d, arma::fill::randu);
  arma::colvec y_train(n_train);
  for (arma::uword i = 0; i < n_train; ++i) {
    y_train(i) = test_function(X_train.row(i));
  }

  // Prepare prediction data
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
  std::vector<double> ll_values;

  for (int iter = 0; iter < n_iterations; ++iter) {
    // Benchmark FIT
    libKriging::MLPKriging mk(config.hidden_dims, config.d_out, config.activation, "gauss");
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      mk.fit(y_train, X_train, "constant", false, config.optim, "LL", config.parameters);
      auto t1 = std::chrono::high_resolution_clock::now();
      fit_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    ll_values.push_back(mk.logLikelihood());

    // Benchmark PREDICT
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      auto [mean, stdev, cov, mean_deriv, stdev_deriv] = mk.predict(X_pred, true, false);
      auto t1 = std::chrono::high_resolution_clock::now();
      predict_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    // Benchmark SIMULATE
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      arma::mat sims = mk.simulate(10, 42 + iter, X_pred);
      auto t1 = std::chrono::high_resolution_clock::now();
      simulate_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    // Benchmark UPDATE
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      mk.update(y_update, X_update);
      auto t1 = std::chrono::high_resolution_clock::now();
      update_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
  }

  // Print results
  print_header();
  print_stats("fit", compute_stats(fit_times));
  print_stats("predict", compute_stats(predict_times));
  print_stats("simulate", compute_stats(simulate_times));
  print_stats("update", compute_stats(update_times));

  // Print LL summary
  auto ll_stats = compute_stats(ll_values);
  std::cout << std::setw(25) << std::left << "logLikelihood"
            << " | " << std::fixed << std::setprecision(4) << ll_stats.mean << std::endl;
}

int main(int argc, char* argv[]) {
  std::cout << "MLPKriging Benchmark" << std::endl;

  // Usage: bench-mlpkriging [iterations] [n] [d] [config_filter]
  int n_iterations = 5;
  arma::uword n_points = 20;
  arma::uword d_dims = 2;
  std::string config_filter = "";  // empty = all

  if (argc > 1)
    n_iterations = std::atoi(argv[1]);
  if (argc > 2)
    n_points = std::atoi(argv[2]);
  if (argc > 3)
    d_dims = std::atoi(argv[3]);
  if (argc > 4)
    config_filter = argv[4];

  // Parse optional adam_iters override
  std::string adam_iters_override = "";
  if (argc > 5)
    adam_iters_override = argv[5];

  std::vector<MLPConfig> configs = build_configs();

  for (auto& config : configs) {
    if (!config_filter.empty() && config.name != config_filter)
      continue;
    if (!adam_iters_override.empty())
      config.parameters["max_iter_adam"] = adam_iters_override;
    benchmark_mlpkriging(config, n_points, d_dims, n_iterations);
  }

  return 0;
}
