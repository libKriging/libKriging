// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
// clang-format on

static double f2d(double x1, double x2) {
  return std::sin(3.0 * x1) + std::cos(5.0 * x2) + x1 * x2;
}

static void make_data(arma::uword n, arma::mat& X, arma::vec& y, unsigned seed = 123) {
  arma::arma_rng::set_seed(seed);
  X = arma::mat(n, 2, arma::fill::randu);
  y = arma::vec(n);
  for (arma::uword i = 0; i < n; ++i)
    y(i) = f2d(X(i, 0), X(i, 1));
}

// Builds a Kriging model with a FIXED, moderate theta (via optim="none"),
// rather than letting BFGS free-fit it. This test function is deterministic
// (noise-free), and its exact MLE is known to drift toward a very large
// theta (a well-documented GP-MLE degeneracy on noise-free data, unrelated
// to predictCG -- see KrigingNystromTest.cpp's history for the same issue).
// A large theta makes the correlation matrix extremely ill-conditioned,
// which is a genuinely hard regime for ANY iterative solver (not a
// predictCG-specific weakness); fixing theta at a moderate, well-conditioned
// value isolates predictCG's own correctness from that unrelated fit issue.
static Kriging make_fixed_theta_model(const arma::vec& y,
                                      const arma::mat& X,
                                      const std::string& cov,
                                      Trend::RegressionModel regmodel,
                                      double theta_val) {
  Kriging::Parameters params;
  params.theta = arma::mat(1, X.n_cols, arma::fill::value(theta_val));
  params.is_theta_estim = false;
  return Kriging(y, X, cov, regmodel, false, "none", "LL", params);
}

// -----------------------------------------------------------------------------

TEST_CASE("predictCG mean/stdev match exact predict at a moderate theta", "[predictcg][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(100, X, y);
  Kriging k = make_fixed_theta_model(y, X, "matern5_2", Trend::RegressionModel::Constant, 0.1);

  arma::mat Xt;
  arma::vec yt;
  make_data(50, Xt, yt, 456);

  auto [m_ex, s_ex, c, dm, ds] = k.predict(Xt, true, false, false);
  auto [m_cg, s_cg] = k.predictCG(Xt, true);

  INFO("max |mean diff| = " << arma::abs(m_cg - m_ex).max());
  INFO("max |stdev diff| = " << arma::abs(s_cg - s_ex).max());
  CHECK(arma::abs(m_cg - m_ex).max() < 0.05 * arma::stddev(y));
  CHECK(arma::abs(s_cg - s_ex).max() < 0.05 * arma::stddev(y));
}

TEST_CASE("predictCG defaults to mean only (stdev empty)", "[predictcg][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(60, X, y);
  Kriging k = make_fixed_theta_model(y, X, "matern5_2", Trend::RegressionModel::Constant, 0.1);

  arma::mat Xt(10, 2, arma::fill::randu);
  auto [mean, stdev] = k.predictCG(Xt);
  CHECK(mean.n_elem == 10);
  CHECK(stdev.n_elem == 0);
}

TEST_CASE("predictCG interpolates the training data", "[predictcg][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(80, X, y);
  Kriging k = make_fixed_theta_model(y, X, "matern5_2", Trend::RegressionModel::Constant, 0.1);

  auto [mean, stdev] = k.predictCG(X, true);
  CHECK(arma::abs(mean - y).max() < 0.05 * arma::stddev(y));
  CHECK(stdev.max() < 0.05 * arma::stddev(y));
}

TEST_CASE("predictCG matches predict for a different kernel/trend", "[predictcg][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(120, X, y);
  Kriging k = make_fixed_theta_model(y, X, "gauss", Trend::RegressionModel::Linear, 0.15);

  arma::mat Xt;
  arma::vec yt;
  make_data(30, Xt, yt, 789);

  auto [m_ex, s_ex, c, dm, ds] = k.predict(Xt, true, false, false);
  auto [m_cg, s_cg] = k.predictCG(Xt, true);

  CHECK(arma::abs(m_cg - m_ex).max() < 0.05 * arma::stddev(y));
  CHECK(arma::abs(s_cg - s_ex).max() < 0.05 * arma::stddev(y));
}

TEST_CASE("predictCG rejects wrong dimension and Nugget models", "[predictcg][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(50, X, y);
  Kriging k = make_fixed_theta_model(y, X, "matern5_2", Trend::RegressionModel::Constant, 0.1);

  arma::mat Xbad(5, 3, arma::fill::randu);
  CHECK_THROWS_AS(k.predictCG(Xbad), std::invalid_argument);

  Kriging knug("matern5_2", Kriging::NoiseModel::Nugget);
  knug.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", {});
  arma::mat Xt(5, 2, arma::fill::randu);
  CHECK_THROWS_AS(knug.predictCG(Xt), std::runtime_error);
}

TEST_CASE("predictCG accuracy improves with a larger iteration budget", "[predictcg][kriging]") {
  // Sanity check that the CG loop is doing meaningful work: an artificially
  // tiny iteration budget should be measurably less accurate than the
  // default (max_iter=0 => 2n).
  arma::mat X;
  arma::vec y;
  make_data(150, X, y);
  Kriging k = make_fixed_theta_model(y, X, "matern5_2", Trend::RegressionModel::Constant, 0.15);

  arma::mat Xt;
  arma::vec yt;
  make_data(40, Xt, yt, 456);

  auto [m_ex, s_ex, c, dm, ds] = k.predict(Xt, true, false, false);
  auto [m_default, s_default] = k.predictCG(Xt, true);
  auto [m_tiny, s_tiny] = k.predictCG(Xt, true, 2, 1e-8);  // only 2 CG iterations

  const double err_default = arma::abs(m_default - m_ex).max();
  const double err_tiny = arma::abs(m_tiny - m_ex).max();
  INFO("err with default budget (2n) = " << err_default << ", err with max_iter=2 = " << err_tiny);
  CHECK(err_default < err_tiny);
}

TEST_CASE("predictCG benchmark", "[.benchmark]") {
  arma::mat X;
  arma::vec y;
  make_data(300, X, y);
  Kriging k = make_fixed_theta_model(y, X, "matern5_2", Trend::RegressionModel::Constant, 0.15);

  arma::mat Xt(100, 2, arma::fill::randu);

  BENCHMARK("Kriging::predict (exact, stored Cholesky) n=300") {
    return k.predict(Xt, true, false, false);
  };
  BENCHMARK("Kriging::predictCG (matrix-free) n=300") {
    return k.predictCG(Xt, true);
  };
}
