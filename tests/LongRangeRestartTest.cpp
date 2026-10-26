// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
#include "libKriging/Optim.hpp"
#include <cmath>
// clang-format on

/**
 * @file LongRangeRestartTest.cpp
 * @brief Long-range restart of the single-start BFGS (Optim::long_range_restart,
 *        docs/dev/CholJitter.md, long-range restart): a noise-free LL fit that
 *        converges with max_j theta_j / maxdX_j > c is restarted, the best kept.
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

double fit_ll(const arma::mat& X, const arma::vec& y, double c, arma::vec* theta = nullptr) {
  Optim::set_long_range_restart(c);
  Kriging k("gauss");
  k.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", Kriging::Parameters{});
  Optim::set_long_range_restart(2.0);
  if (theta != nullptr)
    *theta = k.theta();
  return k.logLikelihood();
}

}  // namespace

TEST_CASE("long-range restart: defaults", "[longrange]") {
  CHECK(Optim::get_long_range_restart() == 2.0);
  CHECK(Optim::get_max_long_range_restart() == 2);
}

TEST_CASE("long-range restart: leaves the long-range local optimum", "[longrange]") {
  // sin d = 4, n = 200, data sets 0, 3, 12 of the study: without the restart
  // the single start ends with a long-range theta, 180-225 LL units below
  // (measured: -174 / -99 / -292 against 16 / 83 / -68). Rounding may change
  // the path on another BLAS: at least two of them must improve by more than
  // 100. Small enough for the CI under valgrind (the n = 500 case below is
  // [intensive]).
  int improved = 0;
  for (int s : {0, 3, 12}) {
    arma::mat X;
    arma::vec y;
    sin_data(9000 + s, 200, 4, 4, X, y);
    arma::vec theta;
    const double ll_off = fit_ll(X, y, 0.0);
    const double ll_on = fit_ll(X, y, 2.0, &theta);
    INFO("set " << s << ": LL off " << ll_off << ", on " << ll_on << ", theta " << theta.t());
    CHECK(ll_on >= ll_off - 1e-6 * std::abs(ll_off));
    improved += ll_on > ll_off + 100;
  }
  CHECK(improved >= 2);
}

TEST_CASE("long-range restart: leaves the long-range local optimum at n = 500", "[longrange][intensive]") {
  // sin d = 4, n = 500, data sets 2, 6, 7, 9 of the study: without the
  // restart the single start ends with a long-range theta, 500-1000 LL units
  // below the optimum (measured: -241 / -495 / -473 / -780 against
  // 466 / 44 / 70 / -268). Rounding may change the path on another BLAS:
  // at least half of them must improve by more than 100.
  int improved = 0;
  for (int s : {2, 6, 7, 9}) {
    arma::mat X;
    arma::vec y;
    sin_data(9000 + s, 500, 4, 4, X, y);
    arma::vec theta;
    const double ll_off = fit_ll(X, y, 0.0);
    const double ll_on = fit_ll(X, y, 2.0, &theta);
    INFO("set " << s << ": LL off " << ll_off << ", on " << ll_on << ", theta " << theta.t());
    CHECK(ll_on >= ll_off - 1e-6 * std::abs(ll_off));
    improved += ll_on > ll_off + 100;
  }
  CHECK(improved >= 2);
}

TEST_CASE("long-range restart: an inactive input keeps its optimum", "[longrange]") {
  // half of the inputs inactive: their theta is genuinely long range, the
  // restart fires and must not lose LL (the best solution is kept)
  for (int s : {0, 1}) {
    arma::mat X;
    arma::vec y;
    sin_data(9000 + s, 200, 4, 2, X, y);
    const double ll_off = fit_ll(X, y, 0.0);
    arma::vec theta;
    const double ll_on = fit_ll(X, y, 2.0, &theta);
    INFO("set " << s << ": LL off " << ll_off << ", on " << ll_on << ", theta " << theta.t());
    CHECK(ll_on >= ll_off - 1e-6 * std::abs(ll_off));
  }
}
