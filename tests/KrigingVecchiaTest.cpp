// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
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

// -----------------------------------------------------------------------------

TEST_CASE("VLL objective spec parsing and validation", "[vecchia][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(60, X, y);

  // valid specs fit fine
  CHECK_NOTHROW(Kriging(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "VLL"));
  CHECK_NOTHROW(Kriging(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "VLL(10)"));

  // malformed specs throw
  for (const std::string bad : {"VLL()", "VLL(x)", "VLL(0)", "VLL(-3)", "VLL(10"}) {
    CHECK_THROWS_AS(Kriging(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", bad),
                    std::invalid_argument);
  }

  // VLL is not available with a nugget/noise channel
  Kriging knug("matern5_2", Kriging::NoiseModel::Nugget);
  CHECK_THROWS_AS(knug.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "VLL(10)", {}),
                  std::invalid_argument);
}

TEST_CASE("VLL(n-1) matches the exact concentrated log-likelihood", "[vecchia][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(60, X, y);

  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "VLL(59)");

  // compare at moderate theta values: at the *fitted* theta of such a smooth
  // function, R is near-singular and jitter placement dominates both sides
  for (const arma::vec theta : {arma::vec{0.2, 0.2}, arma::vec{0.4, 0.3}}) {
    auto [vll, gv] = k.logLikelihoodVecchiaFun(theta, false);
    auto [ll, gl] = k.logLikelihoodFun(theta, false, false);
    INFO("theta=" << theta.t() << ": VLL(n-1) = " << vll << " vs exact LL = " << ll);
    CHECK(std::abs(vll - ll) < 1e-3 * std::abs(ll) + 1e-3);
  }
}

TEST_CASE("VLL analytic gradient matches finite differences", "[vecchia][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(100, X, y);

  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "VLL(15)");

  const double h = 1e-6;
  for (const arma::vec theta : {arma::vec{0.2, 0.3}, arma::vec{0.5, 0.15}}) {
    auto [vll, grad] = k.logLikelihoodVecchiaFun(theta, true);
    for (arma::uword j = 0; j < theta.n_elem; ++j) {
      arma::vec tp = theta, tm = theta;
      tp(j) += h;
      tm(j) -= h;
      const double fd
          = (std::get<0>(k.logLikelihoodVecchiaFun(tp, false)) - std::get<0>(k.logLikelihoodVecchiaFun(tm, false)))
            / (2 * h);
      INFO("theta=" << theta.t() << " dim " << j << ": analytic=" << grad(j) << " fd=" << fd);
      CHECK(std::abs(grad(j) - fd) < 1e-3 * std::max(1.0, std::abs(fd)));
    }
  }
}

TEST_CASE("VLL fit commits an exact model: predict interpolates", "[vecchia][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(150, X, y);

  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "VLL(20)");

  // the final commit performs one exact factorization at theta*, so the
  // fitted model must interpolate exactly like an "LL"-fitted Kriging
  auto [mean, stdev, cov, dm, ds] = k.predict(X, true, false, false);
  CHECK(arma::abs(mean - y).max() < 1e-3);
  CHECK(stdev.max() < 1e-2);
}

TEST_CASE("VLL(20) estimation is consistent with the exact MLE", "[vecchia][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(300, X, y);

  arma::mat Xt;
  arma::vec yt;
  make_data(150, Xt, yt, 456);

  Kriging k_ll(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LL");
  Kriging k_v(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "VLL(20)");
  CHECK(k_v.vecchia_neighbors() == 20);

  auto [m_ll, s1, c1, d1, e1] = k_ll.predict(Xt, false, false, false);
  auto [m_v, s2, c2, d2, e2] = k_v.predict(Xt, false, false, false);
  const double rmse_ll = std::sqrt(arma::mean(arma::square(m_ll - yt)));
  const double rmse_v = std::sqrt(arma::mean(arma::square(m_v - yt)));
  INFO("theta LL = " << k_ll.theta().t() << " theta VLL = " << k_v.theta().t());
  INFO("rmse LL = " << rmse_ll << " rmse VLL = " << rmse_v);
  // predictive accuracy should be comparable (both use exact predict at their theta*)
  CHECK(rmse_v < 2.0 * rmse_ll + 0.05 * arma::stddev(y));
}

TEST_CASE("VLL model supports update with refit", "[vecchia][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(80, X, y);

  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "VLL(10)");

  arma::mat Xu;
  arma::vec yu;
  make_data(20, Xu, yu, 456);
  CHECK_NOTHROW(k.update(yu, Xu, true));
  CHECK(k.X().n_rows == 100);

  auto [mean, s, c, d, e] = k.predict(k.X(), false, false, false);
  CHECK(arma::abs(mean - k.y()).max() < 1e-3);  // still interpolates after refit
}

TEST_CASE("VLL large-n smoke test", "[vecchia][kriging][intensive]") {
  arma::mat X;
  arma::vec y;
  make_data(2000, X, y);

  // starting theta to keep optimizer iterations low; wall time is dominated by
  // the single exact O(n^3) factorization at commit (VLL evals are ~50ms here)
  Kriging::Parameters params;
  params.theta = arma::mat(1, 2, arma::fill::value(0.3));

  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "VLL(20)", params);

  arma::mat Xt;
  arma::vec yt;
  make_data(200, Xt, yt, 456);
  auto [mean, stdev, c, d, e] = k.predict(Xt, true, false, false);
  CHECK(mean.is_finite());
  const double rmse = std::sqrt(arma::mean(arma::square(mean - yt)));
  INFO("rmse = " << rmse << " vs sd(y) = " << arma::stddev(y));
  CHECK(rmse < 0.05 * arma::stddev(y));
}
