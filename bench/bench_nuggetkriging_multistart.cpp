#include "libKriging/NuggetKriging.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>

void generate_branin_data(arma::mat& X, arma::vec& y, int n) {
  arma::arma_rng::set_seed(123);
  X = arma::randu(n, 2);
  X.col(0) = X.col(0) * 15.0 - 5.0;
  X.col(1) = X.col(1) * 15.0;
  y.set_size(n);
  for (int i = 0; i < n; i++) {
    double x1 = X(i, 0), x2 = X(i, 1);
    double a = 1.0, b = 5.1 / (4.0 * M_PI * M_PI), c = 5.0 / M_PI;
    double r = 6.0, s = 10.0, t = 1.0 / (8.0 * M_PI);
    y(i) = a * std::pow(x2 - b * x1 * x1 + c * x1 - r, 2.0) + s * (1.0 - t) * std::cos(x1) + s;
  }
  // Add small noise to make nugget relevant
  y += 0.1 * arma::randn(n);
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <dataset_size> <n_runs>" << std::endl;
    return 1;
  }
  
  int n = std::stoi(argv[1]);
  int n_runs = std::stoi(argv[2]);
  
  arma::mat X;
  arma::vec y;
  generate_branin_data(X, y, n);
  
  std::cout << "NuggetKriging Multistart Benchmark" << std::endl;
  std::cout << "Dataset: n=" << n << " points, d=2 dimensions (with noise)" << std::endl;
  std::cout << "Running " << n_runs << " iterations per method" << std::endl;
  std::cout << std::endl;
  
  std::vector<std::string> methods = {"BFGS", "BFGS2", "BFGS3", "BFGS4", "BFGS5", "BFGS6", "BFGS7", "BFGS8", "BFGS9", "BFGS10", "BFGS20"};
  std::vector<int> n_starts = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20};
  
  std::cout << "┌──────────┬─────────┬──────────────┬──────────────┬──────────────────┬─────────────────┐" << std::endl;
  std::cout << "│  Method  │ Starts  │  Avg Time(s) │   Std(ms)    │  Log-Likelihood  │  Parallel Eff.  │" << std::endl;
  std::cout << "├──────────┼─────────┼──────────────┼──────────────┼──────────────────┼─────────────────┤" << std::endl;
  
  double base_time = 0;
  
  for (size_t m = 0; m < methods.size(); m++) {
    std::vector<double> times;
    double ll = 0;
    
    for (int r = 0; r < n_runs; r++) {
      NuggetKriging km("gauss");
      auto start = std::chrono::high_resolution_clock::now();
      km.fit(y, X, Trend::RegressionModel::Constant, false, methods[m], "LL");
      auto end = std::chrono::high_resolution_clock::now();
      times.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0);
      if (r == n_runs - 1) ll = km.logLikelihood();
    }
    
    double avg = 0;
    for (double t : times) avg += t;
    avg /= times.size();
    
    double std_dev = 0;
    for (double t : times) std_dev += (t - avg) * (t - avg);
    std_dev = std::sqrt(std_dev / times.size()) * 1000.0;
    
    if (m == 0) base_time = avg;
    
    double speedup = (base_time * n_starts[m]) / avg;
    double efficiency = speedup / n_starts[m] * 100.0;
    
    std::cout << "│ " << std::setw(8) << std::left << methods[m]
              << " │ " << std::setw(7) << std::right << n_starts[m]
              << " │ " << std::setw(12) << std::fixed << std::setprecision(3) << avg
              << " │ " << std::setw(12) << std::fixed << std::setprecision(1) << std_dev
              << " │ " << std::setw(16) << std::fixed << std::setprecision(4) << ll
              << " │ " << std::setw(15) << std::fixed << std::setprecision(1) << efficiency << "%"
              << " │" << std::endl;
    
    // Small delay to allow complete thread cleanup between method iterations
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  
  std::cout << "└──────────┴─────────┴──────────────┴──────────────┴──────────────────┴─────────────────┘" << std::endl;
  
  return 0;
}
