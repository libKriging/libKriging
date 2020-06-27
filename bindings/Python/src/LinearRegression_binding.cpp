//
// Created by Pascal Havé on 27/06/2020.
//

#include "LinearRegression_binding.hpp"

#include <armadillo>
#include <libKriging/LinearRegression.hpp>
#include <random>

#include <armadillo>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

void load_test() {
  std::mt19937 engine;    // the Mersenne Twister with a popular choice of parameters
  uint32_t seed_val = 0;  // populate somehow (fixed value => reproducible)
  engine.seed(seed_val);

  const int n = 40;
  const int m = 3;

  arma::vec sol(m, arma::fill::randn);
  arma::mat X(n, m);
  std::normal_distribution<double> dist(1, 10);
  X.col(0).fill(1);
  X.cols(1, m - 1).imbue([&]() { return dist(engine); });

  arma::vec y = X * sol;

  LinearRegression rl;  // linear regression object
  rl.fit(y, X);

  std::tuple<arma::colvec, arma::colvec> ans = rl.predict(X);
  const double eps = 1e-5;  // 1000 * std::numeric_limits<double>::epsilon();
  std::cout << "diff=" << arma::norm(y - std::get<0>(ans), "inf")
            << " eps=" << eps << "\n";
  std::cout << "OK loading" << std::endl;
}