// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
#include <cmath>
// clang-format on

/**
 * @file LongRangeOptimumTest.cpp
 * @brief Long-range local optima of a single start, and the multistart that
 *        replaces the long-range restart removed in 2.0 (#466).
 *
 * On noise-free data a single BFGS start often converges to a sharp local
 * optimum with a theta component far above the data scale, far below the
 * multistart optimum (docs/dev/CholJitter.md). Up to 1.x a single start was
 * restarted on that criterion (Optim::set_long_range_restart, removed): on the
 * T7 benchmark base "BFGS2" has the same miss rate as the restarted single
 * start at 1/1.4 its wall time and 1/1.7 its CPU time (#466). These data sets
 * are the ones of the former LongRangeRestartTest, where the restart was
 * measured to leave the long-range optimum.
 */

namespace {

// bench/thread-budget/jitter_n_study.cpp data: sum over the first `active`
// inputs of sin(2 pi f_j x_j) on n random points of [0,1]^d
void sin_data(int seed, arma::uword n, arma::uword d, arma::uword active, arma::mat& X, arma::vec& y) {
  arma::arma_rng::set_seed(seed);
  X = arma::randu(n, d);
  const arma::rowvec f = 0.5 + 2.5 * arma::randu<arma::rowvec>(d);
  y.zeros(n);
  for (arma::uword j = 0; j < active; ++j)
    y += arma::sin(2 * arma::datum::pi * f(j) * X.col(j));
}

double fit_ll(const arma::mat& X, const arma::vec& y, const std::string& optim, arma::vec* theta = nullptr) {
  Kriging k("gauss");
  k.fit(y, X, Trend::RegressionModel::Constant, false, optim, "LL", Kriging::Parameters{});
  if (theta != nullptr)
    *theta = k.theta();
  return k.logLikelihood();
}

}  // namespace

TEST_CASE("long-range local optimum: BFGS2 leaves it, BFGS (single start) does not", "[longrange]") {
  // sin d = 4, n = 200, data sets 0, 3, 12 of the study: the single start
  // ends with a long-range theta, 100-250 LL units below the optimum
  // (measured, OpenBLAS: -174 / -99 / -292 against 15 / 83 / -67 for BFGS2,
  // the optima the removed restart reached). Rounding
  // may change the paths on another BLAS: at least two of them must improve
  // by more than 100 with two starts, and none may get worse.
  int improved = 0;
  for (int s : {0, 3, 12}) {
    arma::mat X;
    arma::vec y;
    sin_data(9000 + s, 200, 4, 4, X, y);
    arma::vec theta1, theta2;
    const double ll1 = fit_ll(X, y, "BFGS", &theta1);
    const double ll2 = fit_ll(X, y, "BFGS2", &theta2);
    INFO("set " << s << ": LL BFGS " << ll1 << " theta " << theta1.t() << "      BFGS2 " << ll2 << " theta "
                << theta2.t());
    CHECK(ll2 >= ll1 - 1e-6 * std::abs(ll1));
    improved += ll2 > ll1 + 100;
  }
  CHECK(improved >= 2);
}

TEST_CASE("long-range local optimum: an inactive input keeps its optimum", "[longrange]") {
  // half of the inputs inactive: their theta is genuinely long range, the
  // single start already finds the optimum and two starts must not lose it
  for (int s : {0, 1}) {
    arma::mat X;
    arma::vec y;
    sin_data(9000 + s, 200, 4, 2, X, y);
    const double ll1 = fit_ll(X, y, "BFGS");
    const double ll2 = fit_ll(X, y, "BFGS2");
    INFO("set " << s << ": LL BFGS " << ll1 << ", BFGS2 " << ll2);
    CHECK(ll2 >= ll1 - 1e-6 * std::abs(ll1));
  }
}
