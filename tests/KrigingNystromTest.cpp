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

// -----------------------------------------------------------------------------

TEST_CASE("LLNys objective spec parsing and validation", "[nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(60, X, y);

  // valid specs fit fine
  CHECK_NOTHROW(Kriging(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLNys"));
  CHECK_NOTHROW(Kriging(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLNys(10)"));

  // malformed specs throw
  for (const std::string bad : {"LLNys()", "LLNys(x)", "LLNys(0)", "LLNys(-3)", "LLNys(10"}) {
    CHECK_THROWS_AS(Kriging(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", bad),
                    std::invalid_argument);
  }

  // LLNys is not available with a nugget/noise channel
  Kriging knug("matern5_2", Kriging::NoiseModel::Nugget);
  CHECK_THROWS_AS(knug.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LLNys(10)", {}),
                  std::invalid_argument);
}

TEST_CASE("LLNys(n) matches the exact concentrated log-likelihood", "[nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(60, X, y);

  // full rank (k=n): the Nystrom factorization is exact, so LLNys(n) must
  // reproduce the exact concentrated log-likelihood at any theta
  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLNys(60)");

  for (const arma::vec theta : {arma::vec{0.2, 0.2}, arma::vec{0.4, 0.3}}) {
    auto [llnys, gv] = k.logLikelihoodNystromFun(theta, false);
    auto [ll, gl] = k.logLikelihoodFun(theta, false, false);
    INFO("theta=" << theta.t() << ": LLNys(60) = " << llnys << " vs exact LL = " << ll);
    CHECK(std::abs(llnys - ll) < 1e-3 * std::abs(ll) + 1e-3);
  }
}

TEST_CASE("LLNys analytic gradient points in an ascending direction", "[nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(100, X, y);

  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLNys(25)");

  // The fixed-landmark Nystrom log-likelihood is smooth in theta (that's what
  // make_nystrom_landmarks buys us), but near small-theta / near-singular-R
  // regions it can be very steep. What actually matters for L-BFGS-B is that
  // the gradient is a genuine ascent direction for the likelihood: check that
  // a small step along it increases ll, at several step sizes -- this catches
  // a sign error or an off-by-something bug even where a tight numerical
  // comparison against finite differences (see the next test) would be
  // unreliable due to that same curvature.
  for (const arma::vec theta : {arma::vec{0.2, 0.3}, arma::vec{0.5, 0.15}, arma::vec{0.3, 0.3}}) {
    auto [ll0, grad] = k.logLikelihoodNystromFun(theta, true);
    REQUIRE(grad.n_elem == theta.n_elem);
    REQUIRE(arma::norm(grad) > 0.0);
    const arma::vec dir = arma::normalise(grad);
    for (double step : {1e-5, 1e-4}) {
      const double ll_up = std::get<0>(k.logLikelihoodNystromFun(theta + step * dir, false));
      INFO("theta=" << theta.t() << " step=" << step << ": ll0=" << ll0 << " ll_up=" << ll_up);
      CHECK(ll_up > ll0);
    }
  }
}

TEST_CASE("LLNys analytic gradient matches finite differences", "[nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(150, X, y);

  // Moderate theta, away from the near-singular-R region that makes a tight
  // FD comparison unreliable regardless of correctness (see the previous
  // test's comment) -- here both sides should agree closely.
  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLNys(30)");

  // h=1e-3, not smaller: ll's magnitude here is in the hundreds (near-fit
  // sigma2 makes n*log(sigma2) steep), so central-difference ROUNDOFF (not
  // truncation) already dominates by h=1e-5 -- e.g. at theta=(0.5,0.3),
  // dim 0: h=1e-3 gives fd=185.72 (matches analytic=185.83 to 0.1), h=1e-5
  // gives fd=146.49 (spuriously off by 39), h=1e-7 gives fd=1850 (garbage).
  // Verified by Richardson probing during development; see git history for
  // the h-sweep that motivated this choice.
  const double h = 1e-3;
  for (const arma::vec theta : {arma::vec{0.4, 0.4}, arma::vec{0.6, 0.5}, arma::vec{0.5, 0.3}}) {
    auto [ll0, grad] = k.logLikelihoodNystromFun(theta, true);
    for (arma::uword j = 0; j < theta.n_elem; ++j) {
      arma::vec tp = theta;
      arma::vec tm = theta;
      tp(j) += h;
      tm(j) -= h;
      const double fd
          = (std::get<0>(k.logLikelihoodNystromFun(tp, false)) - std::get<0>(k.logLikelihoodNystromFun(tm, false)))
            / (2 * h);
      INFO("theta=" << theta.t() << " dim " << j << ": analytic=" << grad(j) << " fd=" << fd);
      CHECK(std::abs(grad(j) - fd) < 5e-3 * std::max(1.0, std::abs(fd)));
    }
  }
}

TEST_CASE("LLNys fit is a permanent light fit: predict routes to predictNystrom", "[nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(150, X, y);

  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLNys(30)");
  CHECK(k.is_nystrom_light());
  CHECK(k.nystrom_rank() == 30);

  auto [mean, stdev, cov, dm, ds] = k.predict(X, true, false, false);
  auto [m_nys, s_nys] = k.predictNystrom(X, true);
  CHECK(arma::abs(mean - m_nys).max() == 0.0);  // same code path
  CHECK(arma::abs(stdev - s_nys).max() == 0.0);

  // return_cov/return_deriv are not supported on a Nystrom fit
  CHECK_THROWS_AS(k.predict(X, true, true, false), std::runtime_error);
  CHECK_THROWS_AS(k.predict(X, true, false, true), std::runtime_error);

  // simulate/save are not available either (update() IS -- see the dedicated
  // update tests below)
  CHECK_THROWS_AS(k.simulate(3, 123, X.rows(0, 4), false), std::runtime_error);
  CHECK_THROWS_AS(k.save("/tmp/should_not_exist.json"), std::runtime_error);
}

TEST_CASE("LLNys(k) with k close to n interpolates the training data", "[nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(80, X, y);

  // at k=n-1 the Nystrom approximation is near-exact for a smooth kernel, so
  // the fit should recover the training data closely (like the exact "LL" fit)
  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLNys(79)");

  auto [mean, stdev, cov, dm, ds] = k.predict(X, true, false, false);
  CHECK(arma::abs(mean - y).max() < 5e-2 * arma::stddev(y));
}

TEST_CASE("LLNys(k) estimation is consistent with the exact MLE", "[nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(300, X, y);

  arma::mat Xt;
  arma::vec yt;
  make_data(150, Xt, yt, 456);

  Kriging k_ll(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LL");
  Kriging k_n(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLNys(40)");
  CHECK(k_n.nystrom_rank() == 40);

  auto [m_ll, s1, c1, d1, e1] = k_ll.predict(Xt, false, false, false);
  auto [m_n, s2, c2, d2, e2] = k_n.predict(Xt, false, false, false);
  const double rmse_ll = std::sqrt(arma::mean(arma::square(m_ll - yt)));
  const double rmse_n = std::sqrt(arma::mean(arma::square(m_n - yt)));
  INFO("theta LL = " << k_ll.theta().t() << " theta LLNys = " << k_n.theta().t());
  INFO("rmse LL = " << rmse_ll << " rmse LLNys = " << rmse_n);
  CHECK(rmse_n < 2.0 * rmse_ll + 0.05 * arma::stddev(y));
}

TEST_CASE("predictNystrom matches exact predict after an LLNys fit close to full rank", "[nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(200, X, y);

  // fit exactly (theta*), then compare predictNystrom (rank close to n, so
  // near-exact) against the exact predict at the SAME theta/beta/sigma2 by
  // constructing a plain "LL" model with optim=none pinned to k's theta.
  Kriging k_n(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLNys(190)");

  Kriging::Parameters params;
  params.theta = arma::mat(1, 2, arma::fill::none);
  params.theta.value().row(0) = k_n.theta().t();
  params.is_theta_estim = false;
  Kriging k_exact(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "none", "LL", params);

  arma::mat Xt;
  arma::vec yt;
  make_data(80, Xt, yt, 456);

  auto [m_ex, s_ex, c, dm, ds] = k_exact.predict(Xt, true, false, false);
  auto [m_n, s_n] = k_n.predictNystrom(Xt, true);

  INFO("max |mean diff| = " << arma::abs(m_n - m_ex).max());
  INFO("max |stdev diff| = " << arma::abs(s_n - s_ex).max());
  CHECK(arma::abs(m_n - m_ex).max() < 5e-2 * arma::stddev(y));
  // stdev is a much more sensitive quantity than the mean here: it is a
  // (clamped) sqrt of a small residual variance, so even a good low-rank
  // approximation (k=190 of n=200) can show a larger absolute gap on a few
  // points where the true variance is already tiny -- use the median rather
  // than the max to check the approximation is good typically, and a looser
  // max bound to catch gross errors only.
  CHECK(arma::median(arma::abs(s_n - s_ex)) < 5e-2 * arma::stddev(y));
  CHECK(arma::abs(s_n - s_ex).max() < 0.5 * arma::stddev(y));

  // interpolation: predicting at training points recovers y almost exactly
  auto [m_at_X, s_at_X] = k_n.predictNystrom(X.rows(0, 49), true);
  CHECK(arma::abs(m_at_X - y.head(50)).max() < 5e-2 * arma::stddev(y));
}

TEST_CASE("LLNys update(refit=false) extends data at fixed theta/landmarks", "[nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(150, X, y);

  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLNys(30)");
  const arma::vec theta_before = k.theta();

  arma::mat Xu;
  arma::vec yu;
  make_data(40, Xu, yu, 456);
  CHECK_NOTHROW(k.update(yu, Xu, false));

  CHECK(k.X().n_rows == 190);
  CHECK(k.is_nystrom_light());
  // refit=false must not touch theta
  CHECK(arma::abs(k.theta() - theta_before).max() == 0.0);
  CHECK(k.nystrom_rank() == 30);

  // the updated model should still predict reasonably at fresh test points
  arma::mat Xt;
  arma::vec yt;
  make_data(100, Xt, yt, 789);
  auto [mean, stdev, c, d, e] = k.predict(Xt, true, false, false);
  CHECK(mean.is_finite());
  const double rmse = std::sqrt(arma::mean(arma::square(mean - yt)));
  INFO("rmse = " << rmse << " vs sd(y) = " << arma::stddev(y));
  CHECK(rmse < 0.2 * arma::stddev(y));

  // interpolation at a newly-added point
  auto [m_new, s_new] = k.predictNystrom(Xu.row(0), true);
  CHECK(std::abs(m_new(0) - yu(0)) < 0.2 * arma::stddev(y));
}

TEST_CASE("LLNys update(refit=true) warm-restarts theta over the same landmarks", "[nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(150, X, y);

  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLNys(30)");
  const arma::uword rank_before = k.nystrom_rank();

  arma::mat Xu;
  arma::vec yu;
  make_data(40, Xu, yu, 456);
  CHECK_NOTHROW(k.update(yu, Xu, true));

  CHECK(k.X().n_rows == 190);
  CHECK(k.is_nystrom_light());
  CHECK(k.nystrom_rank() == rank_before);  // landmarks/rank unchanged, only theta re-optimized

  arma::mat Xt;
  arma::vec yt;
  make_data(100, Xt, yt, 789);
  auto [mean, stdev, c, d, e] = k.predict(Xt, true, false, false);
  CHECK(mean.is_finite());
  const double rmse = std::sqrt(arma::mean(arma::square(mean - yt)));
  INFO("rmse = " << rmse << " vs sd(y) = " << arma::stddev(y));
  CHECK(rmse < 0.2 * arma::stddev(y));
}

TEST_CASE("LLNys update(refit=false) rejects mismatched dimensions", "[nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(100, X, y);

  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLNys(20)");

  arma::vec yu(5, arma::fill::randu);
  arma::mat Xu_wrong_dim(5, 3, arma::fill::randu);
  CHECK_THROWS_AS(k.update(yu, Xu_wrong_dim, false), std::runtime_error);

  arma::vec yu_wrong_len(4, arma::fill::randu);
  arma::mat Xu(5, 2, arma::fill::randu);
  CHECK_THROWS_AS(k.update(yu_wrong_len, Xu, false), std::runtime_error);
}

TEST_CASE("LLNys large-n smoke test", "[nystrom][kriging][intensive]") {
  arma::mat X;
  arma::vec y;
  make_data(3000, X, y);

  Kriging::Parameters params;
  params.theta = arma::mat(1, 2, arma::fill::value(0.3));

  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLNys(80)", params);

  arma::mat Xt;
  arma::vec yt;
  make_data(200, Xt, yt, 456);
  auto [mean, stdev, c, d, e] = k.predict(Xt, true, false, false);
  CHECK(mean.is_finite());
  const double rmse = std::sqrt(arma::mean(arma::square(mean - yt)));
  INFO("rmse = " << rmse << " vs sd(y) = " << arma::stddev(y));
  // A global rank-80 approximation of n=3000 points is a much coarser
  // compression than Vecchia's local m=20 conditioning at similar n (see
  // KrigingVecchiaTest's large-n test), so this is a looser bound -- the
  // point of this smoke test is scalability (finite/no-throw on n=3000), not
  // matching Vecchia's per-point accuracy.
  CHECK(rmse < 0.3 * arma::stddev(y));
}

TEST_CASE("LLNys benchmark", "[.benchmark]") {
  arma::mat X;
  arma::vec y;
  make_data(400, X, y);
  const arma::vec theta{0.3, 0.3};

  Kriging k_eval(y, X, "gauss", Trend::RegressionModel::Constant, false, "BFGS", "LLNys(30)");

  BENCHMARK("Kriging::fit LLNys(30) n=400") {
    return Kriging(y, X, "gauss", Trend::RegressionModel::Constant, false, "BFGS", "LLNys(30)");
  };
  BENCHMARK("Kriging::fit LL n=400") {
    return Kriging(y, X, "gauss", Trend::RegressionModel::Constant, false, "BFGS", "LL");
  };
  BENCHMARK("Kriging::logLikelihoodNystromFun n=400") {
    return std::get<0>(k_eval.logLikelihoodNystromFun(theta, false));
  };
  BENCHMARK("Kriging::logLikelihoodFun n=400") {
    return std::get<0>(k_eval.logLikelihoodFun(theta, false, false));
  };
  BENCHMARK("Kriging::predictNystrom 100pts n=400") {
    return k_eval.predictNystrom(arma::mat(100, 2, arma::fill::randu), true);
  };
}
