// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
#include <cmath>
#include <utility>
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

TEST_CASE("Nystrom method spec parsing and validation", "[nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(60, X, y);

  // valid specs fit fine
  CHECK(Kriging(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LL", {}, "nystrom").method()
        == "nystrom(50)");
  CHECK(Kriging(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LL", {}, "nystrom(10)").method()
        == "nystrom(10)");
  for (const std::string bad : {"nystrom()", "nystrom(x)", "nystrom(0)", "nystrom(-3)", "nystrom(10", "Nystrom"}) {
    CHECK_THROWS_AS(Kriging(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LL", {}, bad),
                    std::invalid_argument);
  }
  // LOO and LMP have a Nystrom version too (see KrigingApproxObjectivesTest.cpp)
  CHECK(Kriging(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LOO", {}, "nystrom(10)").method()
        == "nystrom(10)");

  // deprecated alias: objective "LLNystrom(k)" = objective "LL", method "nystrom(k)"
  Kriging k_alias(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLNystrom(10)");
  Kriging k_method(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LL", {}, "nystrom(10)");
  CHECK(k_alias.objective() == "LL");
  CHECK(k_alias.method() == "nystrom(10)");
  CHECK(arma::approx_equal(k_alias.theta(), k_method.theta(), "absdiff", 0.0));
  CHECK(Kriging(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLNystrom").method()
        == "nystrom(50)");
  CHECK_THROWS_AS(
      Kriging(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLNystrom(10)", {}, "vecchia"),
      std::invalid_argument);

  // malformed specs throw
  for (const std::string bad : {"LLNystrom()", "LLNystrom(x)", "LLNystrom(0)", "LLNystrom(-3)", "LLNystrom(10"}) {
    CHECK_THROWS_AS(Kriging(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", bad),
                    std::invalid_argument);
  }

  // the nugget/noise channel is available with method "nystrom" (and only
  // the LL objective), as with method "vecchia"
  Kriging knug("matern5_2", Kriging::NoiseModel::Nugget);
  CHECK_NOTHROW(knug.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", {}, "nystrom(10)"));
  CHECK(knug.is_nystrom_light());
  CHECK_THROWS_AS(knug.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LMP", {}, "nystrom(10)"),
                  std::invalid_argument);
  CHECK_NOTHROW(knug.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", {}, "vecchia(10)"));
  CHECK(knug.is_vecchia_light());
}

TEST_CASE("LLNystrom(n) matches the exact concentrated log-likelihood", "[nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(60, X, y);

  // full rank (k=n): the Nystrom factorization is exact, so LLNystrom(n) must
  // reproduce the exact concentrated log-likelihood at any theta
  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LL", {}, "nystrom(60)");

  for (const arma::vec theta : {arma::vec{0.2, 0.2}, arma::vec{0.4, 0.3}}) {
    auto [llnys, gv] = k.logLikelihoodNystromFun(theta, false);
    auto [ll, gl] = k.logLikelihoodFun(theta, false, false);
    INFO("theta=" << theta.t() << ": LLNystrom(60) = " << llnys << " vs exact LL = " << ll);
    CHECK(std::abs(llnys - ll) < 1e-3 * std::abs(ll) + 1e-3);
  }
}

TEST_CASE("LLNystrom analytic gradient points in an ascending direction", "[nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(100, X, y);

  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LL", {}, "nystrom(25)");

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

TEST_CASE("LLNystrom analytic gradient matches finite differences", "[nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(150, X, y);

  // Moderate theta, away from the near-singular-R region that makes a tight
  // FD comparison unreliable regardless of correctness (see the previous
  // test's comment) -- here both sides should agree closely.
  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LL", {}, "nystrom(30)");

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

TEST_CASE("Nystrom fit is a permanent light fit: method nystrom, no exact methods", "[nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(150, X, y);

  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LL", {}, "nystrom(30)");
  CHECK(k.is_nystrom_light());
  CHECK(k.nystrom_rank() == 30);

  auto [mean, stdev, cov, dm, ds] = k.predict(X, true, false, false, "nystrom");
  auto [m_nys, s_nys] = k.predictNystrom(X, true);
  CHECK(arma::abs(mean - m_nys).max() == 0.0);  // same code path
  CHECK(arma::abs(stdev - s_nys).max() == 0.0);
  // the fitted method string is accepted as is; another rank is not
  auto [mean_k, sd_k, cov_k, dm_k, ds_k] = k.predict(X, true, false, false, k.method());
  CHECK(arma::abs(mean_k - m_nys).max() == 0.0);
  CHECK_THROWS_AS(k.predict(X, true, false, false, "nystrom(7)"), std::invalid_argument);
  CHECK_THROWS_AS(k.simulate(3, 123, X.rows(0, 4), "nystrom(7)"), std::invalid_argument);
  CHECK(k.logLikelihood(k.method()) == k.logLikelihood("nystrom"));

  // no exact factorization: the exact methods raise (no implicit routing)
  CHECK_THROWS_AS(k.predict(X, true, false, false), std::runtime_error);
  CHECK_THROWS_AS(k.simulate(3, 123, X.rows(0, 4), false), std::runtime_error);
  // return_cov/return_deriv / will_update only exist for method "exact"
  CHECK_THROWS_AS(k.predict(X, true, true, false, "nystrom"), std::invalid_argument);
  CHECK_THROWS_AS(k.predict(X, true, false, true, "nystrom"), std::invalid_argument);
  CHECK(arma::approx_equal(
      k.simulate(3, 123, X.rows(0, 4), "nystrom"), k.simulateNystrom(3, 123, X.rows(0, 4)), "absdiff", 0.0));
  CHECK_THROWS_AS(k.simulate(3, 123, X.rows(0, 4), "nystrom", true), std::invalid_argument);
  CHECK_THROWS_AS(k.update_simulate(y.head(5), X.rows(0, 4)), std::runtime_error);
  // the Vecchia predictor needs only theta / beta / sigma2: usable here too
  CHECK_NOTHROW(k.predict(X, true, false, false, "vecchia(10)"));
  // the Nystrom log-likelihood is the fitted objective
  CHECK(k.logLikelihood("nystrom") == std::get<0>(k.logLikelihoodNystromFun(k.theta(), false)));
  CHECK_THROWS_AS(k.logLikelihood("vecchia"), std::runtime_error);
}

TEST_CASE("LLNystrom(k) with k close to n interpolates the training data", "[nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(80, X, y);

  // at k=n-1 the Nystrom approximation is near-exact for a smooth kernel, so
  // the fit should recover the training data closely (like the exact "LL" fit)
  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LL", {}, "nystrom(79)");

  auto [mean, stdev, cov, dm, ds] = k.predict(X, true, false, false, "nystrom");
  CHECK(arma::abs(mean - y).max() < 5e-2 * arma::stddev(y));
}

TEST_CASE("LLNystrom(k) estimation is consistent with the exact MLE", "[nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(300, X, y);

  arma::mat Xt;
  arma::vec yt;
  make_data(150, Xt, yt, 456);

  Kriging k_ll(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LL");
  Kriging k_n(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LL", {}, "nystrom(40)");
  CHECK(k_n.nystrom_rank() == 40);

  auto [m_ll, s1, c1, d1, e1] = k_ll.predict(Xt, false, false, false);
  auto [m_n, s2, c2, d2, e2] = k_n.predict(Xt, false, false, false, "nystrom");
  const double rmse_ll = std::sqrt(arma::mean(arma::square(m_ll - yt)));
  const double rmse_n = std::sqrt(arma::mean(arma::square(m_n - yt)));
  INFO("theta LL = " << k_ll.theta().t() << " theta LLNystrom = " << k_n.theta().t());
  INFO("rmse LL = " << rmse_ll << " rmse LLNystrom = " << rmse_n);
  CHECK(rmse_n < 2.0 * rmse_ll + 0.05 * arma::stddev(y));
}

TEST_CASE("predictNystrom matches exact predict after an LLNystrom fit close to full rank", "[nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(200, X, y);

  // Fixed, moderate theta for BOTH models (optim="none"): this test's point
  // is checking predictNystrom (rank close to n, so near-exact) against
  // exact predict at a SHARED theta/beta/sigma2 -- it isn't about where a
  // free BFGS fit happens to converge. A free fit here previously let theta
  // (and hence the Nystrom approximation's own error, which is
  // theta-dependent) drift with platform/compiler-specific BFGS convergence
  // differences, observed anywhere from 0.012 to 0.155 for the median-stdev-
  // diff assertion below against the same seed -- not a stable comparison.
  // Fixing theta removes that confound; this test is about the low-rank
  // approximation's accuracy at a known theta, not about free-fit
  // reproducibility (LLNystrom's own MLE consistency has its own test,
  // "LLNystrom(k) estimation is consistent with the exact MLE").
  Kriging::Parameters params;
  params.theta = arma::mat(1, 2, arma::fill::value(0.3));
  params.is_theta_estim = false;
  Kriging k_n(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "none", "LL", params, "nystrom(190)");
  Kriging k_exact(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "none", "LL", params);

  arma::mat Xt;
  arma::vec yt;
  make_data(80, Xt, yt, 456);

  auto [m_ex, s_ex, c, dm, ds] = k_exact.predict(Xt, true, false, false);
  auto [m_n, s_n] = k_n.predictNystrom(Xt, true);

  INFO("max |mean diff| = " << arma::abs(m_n - m_ex).max());
  INFO("max |stdev diff| = " << arma::abs(s_n - s_ex).max());
  CHECK(arma::abs(m_n - m_ex).max() < 1e-2 * arma::stddev(y));
  // stdev is a much more sensitive quantity than the mean here: it is a
  // (clamped) sqrt of a small residual variance, so even a good low-rank
  // approximation (k=190 of n=200) can show a larger absolute gap on a few
  // points where the true variance is already tiny -- use the median rather
  // than the max to check the approximation is good typically, and a looser
  // max bound to catch gross errors only. Both theta and the comparison
  // point are now fully fixed (no BFGS involved on either side), so the
  // remaining margin is just for ordinary cross-platform BLAS/compiler
  // rounding, not free-fit convergence variance -- measured median ~7e-4,
  // max ~3e-3 locally.
  CHECK(arma::median(arma::abs(s_n - s_ex)) < 1e-2 * arma::stddev(y));
  CHECK(arma::abs(s_n - s_ex).max() < 5e-2 * arma::stddev(y));

  // interpolation: predicting at training points recovers y almost exactly
  auto [m_at_X, s_at_X] = k_n.predictNystrom(X.rows(0, 49), true);
  CHECK(arma::abs(m_at_X - y.head(50)).max() < 5e-2 * arma::stddev(y));
}

TEST_CASE("LLNystrom update(refit=false) extends data at fixed theta/landmarks", "[nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(150, X, y);

  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LL", {}, "nystrom(30)");
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
  auto [mean, stdev, c, d, e] = k.predict(Xt, true, false, false, "nystrom");
  CHECK(mean.is_finite());
  const double rmse = std::sqrt(arma::mean(arma::square(mean - yt)));
  INFO("rmse = " << rmse << " vs sd(y) = " << arma::stddev(y));
  CHECK(rmse < 0.2 * arma::stddev(y));

  // interpolation at a newly-added point
  auto [m_new, s_new] = k.predictNystrom(Xu.row(0), true);
  CHECK(std::abs(m_new(0) - yu(0)) < 0.2 * arma::stddev(y));
}

TEST_CASE("LLNystrom update(refit=true) warm-restarts theta over the same landmarks", "[nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(150, X, y);

  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LL", {}, "nystrom(30)");
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
  auto [mean, stdev, c, d, e] = k.predict(Xt, true, false, false, "nystrom");
  CHECK(mean.is_finite());
  const double rmse = std::sqrt(arma::mean(arma::square(mean - yt)));
  INFO("rmse = " << rmse << " vs sd(y) = " << arma::stddev(y));
  CHECK(rmse < 0.2 * arma::stddev(y));
}

TEST_CASE("LLNystrom update(refit=false) rejects mismatched dimensions", "[nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(100, X, y);

  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LL", {}, "nystrom(20)");

  arma::vec yu(5, arma::fill::randu);
  arma::mat Xu_wrong_dim(5, 3, arma::fill::randu);
  CHECK_THROWS_AS(k.update(yu, Xu_wrong_dim, false), std::runtime_error);

  arma::vec yu_wrong_len(4, arma::fill::randu);
  arma::mat Xu(5, 2, arma::fill::randu);
  CHECK_THROWS_AS(k.update(yu_wrong_len, Xu, false), std::runtime_error);
}

TEST_CASE("simulateNystrom mean/marginal-variance are consistent with predictNystrom", "[nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(150, X, y);

  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LL", {}, "nystrom(30)");

  arma::mat Xt;
  arma::vec yt;
  make_data(20, Xt, yt, 456);

  auto [mean, stdev] = k.predictNystrom(Xt, true);

  const int nsim = 20000;
  arma::mat sims = k.simulate(nsim, 42, Xt, "nystrom");
  REQUIRE(sims.n_rows == 20);
  REQUIRE(sims.n_cols == static_cast<arma::uword>(nsim));

  const arma::vec sim_mean = arma::mean(sims, 1);
  const arma::vec sim_sd = arma::stddev(sims, 0, 1);

  INFO("theta=" << k.theta().t() << " sigma2=" << k.sigma2());
  INFO("max |sim_mean - predict mean| = " << arma::abs(sim_mean - mean).max());
  INFO("max |sim_sd - predict stdev| = " << arma::abs(sim_sd - stdev).max());
  CHECK(arma::abs(sim_mean - mean).max() < 0.05 * arma::stddev(y));
  // The joint (n_n x n_n) covariance behind simulateNystrom is a strictly
  // harder numerical problem than predictNystrom's per-point marginal (see
  // simulateNystrom's comment: it mixes an exact R_nn with cross-terms
  // approximated through the same rank-k landmarks, and a PSD-repair -- both
  // can lose precision, particularly when the fit's own theta lands far from
  // the data's scale, e.g. the well-known noise-free-GP-MLE degeneracy that
  // can drive theta toward its upper bound on deterministic data such as
  // this test's f2d). Compare against sqrt(sigma2), the model's own natural
  // noise scale, rather than sd(y): median for the typical case, a looser
  // max bound to catch gross errors only.
  const double noise_scale = std::sqrt(k.sigma2());
  CHECK(arma::median(arma::abs(sim_sd - stdev)) < 0.05 * noise_scale);
  CHECK(arma::abs(sim_sd - stdev).max() < 0.5 * noise_scale);

  // interpolation: simulating exactly at a training point collapses to (near) zero variance
  arma::mat sims_at_X = k.simulate(2000, 7, X.row(0), "nystrom");
  CHECK(arma::stddev(sims_at_X.row(0).t()) < 0.05 * noise_scale);
  CHECK(std::abs(arma::mean(sims_at_X.row(0).t()) - y(0)) < 0.05 * arma::stddev(y));
}

TEST_CASE("LLNystrom save/load round-trip", "[nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(120, X, y);

  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LL", {}, "nystrom(25)");
  CHECK_NOTHROW(k.save("dump_nystrom_test.json"));

  Kriging k2 = Kriging::load("dump_nystrom_test.json");
  CHECK(k2.is_nystrom_light());
  CHECK(k2.nystrom_rank() == 25);
  CHECK(k2.method() == "nystrom(25)");
  CHECK(arma::abs(k2.theta() - k.theta()).max() == 0.0);
  CHECK(k2.sigma2() == k.sigma2());

  arma::mat Xt;
  arma::vec yt;
  make_data(60, Xt, yt, 456);

  auto [m1, s1, c1, d1, e1] = k.predict(Xt, true, false, false, "nystrom");
  auto [m2, s2, c2, d2, e2] = k2.predict(Xt, true, false, false, "nystrom");
  CHECK(arma::abs(m1 - m2).max() == 0.0);
  CHECK(arma::abs(s1 - s2).max() == 0.0);

  // the reloaded model supports the same light-fit operations
  CHECK_NOTHROW(k2.simulate(2, 123, Xt.rows(0, 4), "nystrom"));
  arma::mat Xu;
  arma::vec yu;
  make_data(10, Xu, yu, 789);
  CHECK_NOTHROW(k2.update(yu, Xu, false));
  CHECK(k2.X().n_rows == 130);
}

TEST_CASE("LLNystrom large-n smoke test", "[nystrom][kriging][intensive]") {
  arma::mat X;
  arma::vec y;
  make_data(3000, X, y);

  Kriging::Parameters params;
  params.theta = arma::mat(1, 2, arma::fill::value(0.3));

  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LL", params, "nystrom(80)");

  arma::mat Xt;
  arma::vec yt;
  make_data(200, Xt, yt, 456);
  auto [mean, stdev, c, d, e] = k.predict(Xt, true, false, false, "nystrom");
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

// Same general contract as the sibling regression test in
// KrigingVecchiaTest.cpp: whatever objective is
// given at fit time is what predict()/etc actually use afterwards, for EVERY
// supported value of optim, not just the free-fit "BFGS" path exercised by
// the "permanent light fit" test above. LLNystrom already honored this
// correctly (it was the reference implementation the other was fixed to
// match); this test just makes that guarantee explicit and pins it down.
TEST_CASE("LLNystrom honors optim=none identically to optim=BFGS", "[nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(150, X, y);
  arma::mat Xt;
  arma::vec yt;
  make_data(40, Xt, yt, 456);

  const std::string optim = GENERATE(as<std::string>{}, "none", "BFGS(1)");
  CAPTURE(optim);

  Kriging::Parameters params;
  params.theta = arma::mat(1, X.n_cols, arma::fill::value(0.3));
  params.is_theta_estim = (optim != "none");  // optim="none" requires a fixed theta

  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, optim, "LL", params, "nystrom(30)");
  CHECK(k.is_nystrom_light());
  CHECK(k.nystrom_rank() == 30);

  auto [mean, stdev, cov, dm, ds] = k.predict(Xt, true, false, false, "nystrom");
  auto [m_nys, s_nys] = k.predictNystrom(Xt, true);
  CHECK(arma::abs(mean - m_nys).max() == 0.0);
  CHECK(arma::abs(stdev - s_nys).max() == 0.0);
  CHECK_THROWS_AS(k.simulate(3, 123, Xt.rows(0, 2), "nystrom", true), std::invalid_argument);
  CHECK_THROWS_AS(k.update_simulate(y.head(3), X.rows(0, 2)), std::runtime_error);
}

TEST_CASE("LLNystrom benchmark", "[.benchmark]") {
  arma::mat X;
  arma::vec y;
  make_data(400, X, y);
  const arma::vec theta{0.3, 0.3};

  Kriging k_eval(y, X, "gauss", Trend::RegressionModel::Constant, false, "BFGS", "LL", {}, "nystrom(30)");

  BENCHMARK("Kriging::fit LLNystrom(30) n=400") {
    return Kriging(y, X, "gauss", Trend::RegressionModel::Constant, false, "BFGS", "LL", {}, "nystrom(30)");
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

// -----------------------------------------------------------------------------
// Noise models (NoiseModel::Nugget / NoiseModel::Heterogeneous) with method "nystrom(k)"
// -----------------------------------------------------------------------------

static void make_noisy_data(arma::uword n, arma::mat& X, arma::vec& y, double noise_sd, unsigned seed = 123) {
  make_data(n, X, y, seed);
  y += noise_sd * arma::vec(n, arma::fill::randn);
}

static Kriging fit_nugget(const arma::vec& y,
                          const arma::mat& X,
                          const std::string& optim,
                          const Kriging::Parameters& params,
                          const std::string& method) {
  Kriging k("matern5_2", Kriging::NoiseModel::Nugget);
  k.fit(y, X, Trend::RegressionModel::Constant, false, optim, "LL", params, method);
  return k;
}

static Kriging fit_hetero(const arma::vec& y,
                          const arma::vec& noise,
                          const arma::mat& X,
                          const std::string& optim,
                          const Kriging::Parameters& params,
                          const std::string& method) {
  Kriging k("matern5_2", Kriging::NoiseModel::Heterogeneous);
  k.fit(y, noise, X, Trend::RegressionModel::Constant, false, optim, "LL", params, method);
  return k;
}

TEST_CASE("Nystrom noise models: method nystrom(k) accepted, LOO still rejected", "[nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  make_noisy_data(60, X, y, 0.1);
  const arma::vec noise(60, arma::fill::value(0.01));

  Kriging kn = fit_nugget(y, X, "BFGS", {}, "nystrom(15)");
  CHECK(kn.is_nystrom_light());
  CHECK(kn.method() == "nystrom(15)");
  CHECK(kn.nugget() > 0.0);
  Kriging kh = fit_hetero(y, noise, X, "BFGS", {}, "nystrom(15)");
  CHECK(kh.is_nystrom_light());
  CHECK(kh.sigma2() > 0.0);

  // exact-only methods are unavailable on the light fit
  CHECK_THROWS_AS(kn.predict(X.rows(0, 2), true, false, false), std::runtime_error);
  CHECK_THROWS_AS(kh.simulate(2, 1, X.rows(0, 2)), std::runtime_error);

  Kriging k2("matern5_2", Kriging::NoiseModel::Nugget);
  CHECK_THROWS_AS(k2.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LOO", {}, "nystrom(10)"),
                  std::invalid_argument);

  // simulate(..., with_nugget | with_noise, will_update, "nystrom"): the
  // latent process only, so with_nugget=true / a non-zero with_noise raise
  const arma::mat Xs = X.rows(0, 2);
  CHECK(
      arma::approx_equal(kn.simulate(4, 7, Xs, false, false, "nystrom"), kn.simulateNystrom(4, 7, Xs), "absdiff", 0.0));
  CHECK_THROWS_AS(kn.simulate(4, 7, Xs, true, false, "nystrom"), std::invalid_argument);
  CHECK(arma::approx_equal(kh.simulate(4, 7, Xs, arma::vec(3, arma::fill::zeros), false, "nystrom"),
                           kh.simulateNystrom(4, 7, Xs),
                           "absdiff",
                           0.0));
  CHECK_THROWS_AS(kh.simulate(4, 7, Xs, arma::vec(3, arma::fill::value(0.1)), false, "nystrom"), std::invalid_argument);
  CHECK_THROWS_AS(kh.simulate(4, 7, Xs, arma::vec(3, arma::fill::zeros), true, "nystrom"), std::invalid_argument);
}

TEST_CASE("LLNystrom(n) reproduces the exact Nugget / Heterogeneous log-likelihood and gradient",
          "[nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  const arma::uword n = 50;
  make_noisy_data(n, X, y, 0.1);
  arma::arma_rng::set_seed(7);
  const arma::vec noise = 0.01 * (1.0 + arma::vec(n, arma::fill::randu));

  // full rank (k = n): Ct = C (up to the num_nugget jitter), so Rhat = R and
  // the Nystrom likelihood (value AND gradient, incl. the alpha / sigma2
  // component) must match the exact one at any gamma
  SECTION("Nugget") {
    Kriging k_ex = fit_nugget(y, X, "BFGS", {}, "exact");
    Kriging k_ny = fit_nugget(y, X, "BFGS", {}, "nystrom(50)");
    for (const arma::vec gamma : {arma::vec{0.2, 0.3, 0.9}, arma::vec{0.4, 0.25, 0.6}, arma::vec{0.3, 0.3, 0.99}}) {
      auto [ll_ex, g_ex] = k_ex.logLikelihoodFun(gamma, true, false);
      auto [ll_ny, g_ny] = k_ny.logLikelihoodNystromFun(gamma, true);
      INFO("gamma=" << gamma.t() << " LL exact=" << ll_ex << " nystrom=" << ll_ny);
      INFO("grad exact=" << g_ex.t() << "grad nystrom=" << g_ny.t());
      CHECK(std::abs(ll_ny - ll_ex) < 1e-6 * std::abs(ll_ex) + 1e-6);
      REQUIRE(g_ny.n_elem == 3);
      CHECK(arma::abs(g_ny - g_ex).max() < 1e-4 * std::max(1.0, arma::abs(g_ex).max()));
    }
  }
  SECTION("Heterogeneous") {
    Kriging k_ex = fit_hetero(y, noise, X, "BFGS", {}, "exact");
    Kriging k_ny = fit_hetero(y, noise, X, "BFGS", {}, "nystrom(50)");
    for (const arma::vec gamma : {arma::vec{0.2, 0.3, 1.0}, arma::vec{0.4, 0.25, 0.5}, arma::vec{0.3, 0.3, 2.0}}) {
      auto [ll_ex, g_ex] = k_ex.logLikelihoodFun(gamma, true, false);
      auto [ll_ny, g_ny] = k_ny.logLikelihoodNystromFun(gamma, true);
      INFO("gamma=" << gamma.t() << " LL exact=" << ll_ex << " nystrom=" << ll_ny);
      INFO("grad exact=" << g_ex.t() << "grad nystrom=" << g_ny.t());
      CHECK(std::abs(ll_ny - ll_ex) < 1e-6 * std::abs(ll_ex) + 1e-6);
      REQUIRE(g_ny.n_elem == 3);
      CHECK(arma::abs(g_ny - g_ex).max() < 1e-4 * std::max(1.0, arma::abs(g_ex).max()));
    }
  }
}

TEST_CASE("LLNystrom noise-model analytic gradient matches finite differences", "[nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  const arma::uword n = 120;
  make_noisy_data(n, X, y, 0.1);
  arma::arma_rng::set_seed(11);
  const arma::vec noise = 0.01 * (1.0 + arma::vec(n, arma::fill::randu));

  auto check_fd = [](Kriging& k, const arma::vec& gamma) {
    auto [ll0, grad] = k.logLikelihoodNystromFun(gamma, true);
    REQUIRE(grad.n_elem == gamma.n_elem);
    for (arma::uword j = 0; j < gamma.n_elem; ++j) {
      const double h = 1e-5 * std::max(1.0, std::abs(gamma(j)));
      arma::vec gp = gamma;
      arma::vec gm = gamma;
      gp(j) += h;
      gm(j) -= h;
      const double fd
          = (std::get<0>(k.logLikelihoodNystromFun(gp, false)) - std::get<0>(k.logLikelihoodNystromFun(gm, false)))
            / (2 * h);
      INFO("gamma=" << gamma.t() << " dim " << j << ": analytic=" << grad(j) << " fd=" << fd);
      CHECK(std::abs(grad(j) - fd) < 1e-3 * std::max(1.0, std::abs(fd)));
    }
  };

  SECTION("Nugget, sigma2 and nugget estimated (concentrated)") {
    Kriging k = fit_nugget(y, X, "BFGS", {}, "nystrom(25)");
    for (const arma::vec gamma : {arma::vec{0.4, 0.4, 0.9}, arma::vec{0.6, 0.5, 0.7}, arma::vec{0.5, 0.3, 0.98}})
      check_fd(k, gamma);
  }
  SECTION("Nugget, nugget fixed (sigma2 estimated)") {
    Kriging::Parameters p;
    p.nugget = 0.01;
    p.is_nugget_estim = false;
    Kriging k = fit_nugget(y, X, "BFGS", p, "nystrom(25)");
    for (const arma::vec gamma : {arma::vec{0.4, 0.4, 0.9}, arma::vec{0.5, 0.3, 0.98}})
      check_fd(k, gamma);
  }
  SECTION("Nugget, sigma2 fixed (nugget estimated)") {
    Kriging::Parameters p;
    p.sigma2 = 1.0;
    p.is_sigma2_estim = false;
    Kriging k = fit_nugget(y, X, "BFGS", p, "nystrom(25)");
    for (const arma::vec gamma : {arma::vec{0.4, 0.4, 0.9}, arma::vec{0.5, 0.3, 0.98}})
      check_fd(k, gamma);
  }
  SECTION("Heterogeneous") {
    Kriging k = fit_hetero(y, noise, X, "BFGS", {}, "nystrom(25)");
    for (const arma::vec gamma : {arma::vec{0.4, 0.4, 1.0}, arma::vec{0.6, 0.5, 0.3}, arma::vec{0.5, 0.3, 3.0}})
      check_fd(k, gamma);
  }
}

TEST_CASE("Nystrom noise-model fit is consistent with the exact fit", "[nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  const arma::uword n = 150;
  make_noisy_data(n, X, y, 0.1);
  arma::mat Xt;
  arma::vec yt;
  make_data(80, Xt, yt, 456);  // noise-free truth
  const double sdy = arma::stddev(y);

  SECTION("Nugget") {
    Kriging k_ex = fit_nugget(y, X, "BFGS", {}, "exact");
    Kriging k_ny = fit_nugget(y, X, "BFGS", {}, "nystrom(100)");
    auto [m_ex, s_ex, c1, d1, e1] = k_ex.predict(Xt, true, false, false);
    auto [m_ny, s_ny, c2, d2, e2] = k_ny.predict(Xt, true, false, false, "nystrom");
    INFO("exact: theta=" << k_ex.theta().t() << " sigma2=" << k_ex.sigma2() << " nugget=" << k_ex.nugget()
                         << " LL=" << k_ex.logLikelihood());
    INFO("nystrom: theta=" << k_ny.theta().t() << " sigma2=" << k_ny.sigma2() << " nugget=" << k_ny.nugget()
                           << " LL=" << k_ny.logLikelihood("nystrom"));
    INFO("max |mean diff| = " << arma::abs(m_ny - m_ex).max()
                              << ", rmse nystrom = " << std::sqrt(arma::mean(arma::square(m_ny - yt))));
    // the noise variance estimate is the nugget (true value 0.01)
    CHECK(k_ny.nugget() > 0.002);
    CHECK(k_ny.nugget() < 0.05);
    CHECK(std::abs(k_ny.logLikelihood("nystrom") - k_ex.logLikelihood()) < 0.05 * std::abs(k_ex.logLikelihood()) + 2);
    CHECK(arma::abs(m_ny - m_ex).max() < 0.1 * sdy);
    CHECK(std::sqrt(arma::mean(arma::square(m_ny - yt))) < 0.2 * sdy);
    CHECK(arma::median(arma::abs(s_ny - s_ex)) < 0.1 * sdy);
    // theta-only evaluation uses the committed alpha
    const double ll_theta = std::get<0>(k_ny.logLikelihoodNystromFun(k_ny.theta(), false));
    CHECK(std::abs(ll_theta - k_ny.logLikelihood("nystrom")) < 1e-8 * std::abs(ll_theta));
  }
  SECTION("Heterogeneous") {
    const arma::vec noise(n, arma::fill::value(0.01));
    Kriging k_ex = fit_hetero(y, noise, X, "BFGS", {}, "exact");
    Kriging k_ny = fit_hetero(y, noise, X, "BFGS", {}, "nystrom(100)");
    auto [m_ex, s_ex, c1, d1, e1] = k_ex.predict(Xt, true, false, false);
    auto [m_ny, s_ny, c2, d2, e2] = k_ny.predict(Xt, true, false, false, "nystrom");
    INFO("exact: theta=" << k_ex.theta().t() << " sigma2=" << k_ex.sigma2() << " LL=" << k_ex.logLikelihood());
    INFO("nystrom: theta=" << k_ny.theta().t() << " sigma2=" << k_ny.sigma2()
                           << " LL=" << k_ny.logLikelihood("nystrom"));
    INFO("max |mean diff| = " << arma::abs(m_ny - m_ex).max());
    CHECK(std::abs(k_ny.logLikelihood("nystrom") - k_ex.logLikelihood()) < 0.05 * std::abs(k_ex.logLikelihood()) + 2);
    CHECK(std::abs(std::log(k_ny.sigma2() / k_ex.sigma2())) < std::log(2.0));
    CHECK(arma::abs(m_ny - m_ex).max() < 0.1 * sdy);
    CHECK(std::sqrt(arma::mean(arma::square(m_ny - yt))) < 0.2 * sdy);
    CHECK(arma::median(arma::abs(s_ny - s_ex)) < 0.1 * sdy);
  }
}

TEST_CASE("predictNystrom / simulateNystrom with noise models follow the exact semantics", "[nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  const arma::uword n = 120;
  make_noisy_data(n, X, y, 0.1);
  arma::mat Xt;
  arma::vec yt;
  make_data(40, Xt, yt, 456);
  const double sdy = arma::stddev(y);

  // shared, fixed parameters (optim="none"), full rank: the Nystrom
  // predictor must reproduce the exact one, including at the training
  // points (where the exact nugget predictor returns the observation)
  Kriging::Parameters p;
  p.theta = arma::mat(1, 2, arma::fill::value(0.3));
  p.is_theta_estim = false;
  p.sigma2 = 1.0;
  p.is_sigma2_estim = false;

  SECTION("Nugget") {
    p.nugget = 0.01;
    p.is_nugget_estim = false;
    Kriging k_ex = fit_nugget(y, X, "none", p, "exact");
    Kriging k_ny = fit_nugget(y, X, "none", p, "nystrom(120)");
    CHECK(k_ny.sigma2() == k_ex.sigma2());
    CHECK(k_ny.nugget() == k_ex.nugget());
    const arma::mat Xp = arma::join_cols(Xt, X.rows(0, 9));
    auto [m_ex, s_ex, c1, d1, e1] = k_ex.predict(Xp, true, false, false);
    auto [m_ny, s_ny] = k_ny.predictNystrom(Xp, true);
    INFO("max |mean diff| = " << arma::abs(m_ny - m_ex).max() << ", max |sd diff| = " << arma::abs(s_ny - s_ex).max());
    CHECK(arma::abs(m_ny - m_ex).max() < 1e-3 * sdy);
    // the exact stdev also carries the (small) trend-estimation term, the
    // Nystrom one treats beta as known
    CHECK(arma::abs(s_ny - s_ex).max() < 2e-2 * std::sqrt(k_ex.sigma2() + k_ex.nugget()));
    CHECK(arma::abs(m_ny.tail(10) - y.head(10)).max() < 1e-6);

    // simulate: the latent process (exact with_nugget=false): sample variance
    // + nugget = predicted variance, sample mean = predicted mean
    const arma::mat sims = k_ny.simulate(4000, 123, Xt.rows(0, 9), "nystrom");
    const arma::vec sim_mean = arma::mean(sims, 1);
    const arma::vec sim_var = arma::var(sims, 0, 1);
    const arma::vec pred_var = arma::square(s_ny.head(10));
    INFO("sim var + nugget = " << (sim_var + k_ny.nugget()).t() << "pred var = " << pred_var.t());
    CHECK(arma::abs(sim_mean - m_ny.head(10)).max() < 0.1 * std::sqrt(k_ny.sigma2()));
    CHECK(arma::abs(sim_var + k_ny.nugget() - pred_var).max() < 0.1 * arma::max(pred_var));
    // and the exact latent simulation has the same marginal variance (up to
    // the trend term and Monte-Carlo noise)
    const arma::mat sims_ex = k_ex.simulate(4000, 123, Xt.rows(0, 9));
    CHECK(arma::abs(arma::var(sims_ex, 0, 1) - sim_var).max() < 0.15 * arma::max(sim_var) + 1e-3);
  }
  SECTION("Heterogeneous") {
    arma::arma_rng::set_seed(3);
    const arma::vec noise = 0.01 * (1.0 + arma::vec(n, arma::fill::randu));
    Kriging k_ex = fit_hetero(y, noise, X, "none", p, "exact");
    Kriging k_ny = fit_hetero(y, noise, X, "none", p, "nystrom(120)");
    CHECK(k_ny.sigma2() == k_ex.sigma2());
    auto [m_ex, s_ex, c1, d1, e1] = k_ex.predict(Xt, true, false, false);
    auto [m_ny, s_ny] = k_ny.predictNystrom(Xt, true);
    INFO("max |mean diff| = " << arma::abs(m_ny - m_ex).max() << ", max |sd diff| = " << arma::abs(s_ny - s_ex).max());
    CHECK(arma::abs(m_ny - m_ex).max() < 1e-3 * sdy);
    CHECK(arma::abs(s_ny - s_ex).max() < 2e-2 * std::sqrt(k_ex.sigma2()));

    const arma::mat sims = k_ny.simulate(4000, 123, Xt.rows(0, 9), "nystrom");
    const arma::vec sim_var = arma::var(sims, 0, 1);
    const arma::vec pred_var = arma::square(s_ny.head(10));
    CHECK(arma::abs(arma::mean(sims, 1) - m_ny.head(10)).max() < 0.1 * std::sqrt(k_ny.sigma2()));
    CHECK(arma::abs(sim_var - pred_var).max() < 0.1 * arma::max(pred_var) + 1e-4);
  }
}

TEST_CASE("Nystrom noise-model save/load round-trip", "[nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  const arma::uword n = 100;
  make_noisy_data(n, X, y, 0.1);
  arma::mat Xt;
  arma::vec yt;
  make_data(30, Xt, yt, 456);
  const arma::vec noise(n, arma::fill::value(0.01));

  for (const bool nugget : {true, false}) {
    CAPTURE(nugget);
    Kriging k
        = nugget ? fit_nugget(y, X, "BFGS", {}, "nystrom(20)") : fit_hetero(y, noise, X, "BFGS", {}, "nystrom(20)");
    k.save("dump_nystrom_noise_test.json");
    Kriging k2 = Kriging::load("dump_nystrom_noise_test.json");
    CHECK(k2.noise_model() == k.noise_model());
    CHECK(k2.is_nystrom_light());
    CHECK(k2.method() == "nystrom(20)");
    CHECK(k2.sigma2() == k.sigma2());
    CHECK(k2.nugget() == k.nugget());
    CHECK(k2.logLikelihood("nystrom") == k.logLikelihood("nystrom"));
    auto [m1, s1, c1, d1, e1] = k.predict(Xt, true, false, false, "nystrom");
    auto [m2, s2, c2, d2, e2] = k2.predict(Xt, true, false, false, "nystrom");
    CHECK(arma::abs(m1 - m2).max() == 0.0);
    CHECK(arma::abs(s1 - s2).max() == 0.0);
    CHECK(arma::abs(k.simulate(3, 7, Xt.rows(0, 4), "nystrom") - k2.simulate(3, 7, Xt.rows(0, 4), "nystrom")).max()
          == 0.0);
  }
}

TEST_CASE("Nystrom noise-model update", "[nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  make_noisy_data(120, X, y, 0.1);
  arma::mat Xu;
  arma::vec yu;
  make_noisy_data(30, Xu, yu, 0.1, 456);
  arma::mat Xt;
  arma::vec yt;
  make_data(80, Xt, yt, 789);
  const double sdy = arma::stddev(y);

  SECTION("Nugget, refit=false: fixed theta / alpha, re-profiled over the extended data") {
    Kriging k = fit_nugget(y, X, "BFGS", {}, "nystrom(25)");
    const arma::vec theta0 = k.theta();
    const double ratio0 = k.sigma2() / (k.sigma2() + k.nugget());
    k.update(yu, Xu, false);
    CHECK(k.X().n_rows == 150);
    CHECK(k.is_nystrom_light());
    CHECK(arma::abs(k.theta() - theta0).max() == 0.0);
    CHECK(std::abs(k.sigma2() / (k.sigma2() + k.nugget()) - ratio0) < 1e-12);
    CHECK(std::isfinite(k.logLikelihood("nystrom")));
    auto [m, s, c, d, e] = k.predict(Xt, true, false, false, "nystrom");
    CHECK(std::sqrt(arma::mean(arma::square(m - yt))) < 0.2 * sdy);
  }
  SECTION("Nugget, refit=true: warm restart over (theta, alpha)") {
    Kriging k = fit_nugget(y, X, "BFGS", {}, "nystrom(25)");
    Kriging k_norefit(k, ExplicitCopySpecifier{});
    k_norefit.update(yu, Xu, false);
    k.update(yu, Xu, true);
    CHECK(k.X().n_rows == 150);
    CHECK(k.nystrom_rank() == 25);
    CHECK(k.nugget() > 0.0);
    // the warm restart can only improve the likelihood over the extended data
    CHECK(k.logLikelihood("nystrom") >= k_norefit.logLikelihood("nystrom") - 1e-6);
    auto [m, s, c, d, e] = k.predict(Xt, true, false, false, "nystrom");
    CHECK(std::sqrt(arma::mean(arma::square(m - yt))) < 0.2 * sdy);
  }
  SECTION("Heterogeneous: update needs the new noise") {
    const arma::vec noise(120, arma::fill::value(0.01));
    const arma::vec noise_u(30, arma::fill::value(0.01));
    Kriging k = fit_hetero(y, noise, X, "BFGS", {}, "nystrom(25)");
    CHECK_THROWS_AS(k.update(yu, Xu, false), std::invalid_argument);
    k.update(yu, noise_u, Xu, false);
    CHECK(k.X().n_rows == 150);
    CHECK(k.is_nystrom_light());
    CHECK(k.method() == "nystrom(25)");
    auto [m, s, c, d, e] = k.predict(Xt, true, false, false, "nystrom");
    CHECK(std::sqrt(arma::mean(arma::square(m - yt))) < 0.2 * sdy);
    k.update(yu, noise_u, Xu, true);
    CHECK(k.X().n_rows == 180);
    CHECK(k.is_nystrom_light());
  }
}

// #410: at the landmark rows the Nystrom diagonal residual is at the
// num_nugget level, and the former Woodbury solve weighted those rows by
// 1/num_nugget: the log-likelihood and predicted mean then depended on
// rounding (on the CPU / BLAS kernel), by up to 60% of sd(y) at an ill
// conditioned R_ss. A 1-ulp perturbation of X stands in for "another CPU".
namespace {

arma::vec hartmann3(const arma::mat& X) {
  const arma::mat a = {{3.0, 10, 30}, {0.1, 10, 35}, {3.0, 10, 30}, {0.1, 10, 35}};
  const arma::mat p
      = {{0.3689, 0.1170, 0.2673}, {0.4699, 0.4387, 0.7470}, {0.1091, 0.8732, 0.5547}, {0.0381, 0.5743, 0.8828}};
  const arma::vec c = {1.0, 1.2, 3.0, 3.2};
  arma::vec y(X.n_rows, arma::fill::zeros);
  for (arma::uword i = 0; i < 4; ++i) {
    const arma::mat dX = X.each_row() - p.row(i);
    y -= c(i) * arma::exp(-arma::square(dX) * a.row(i).t());
  }
  return y;
}

arma::vec branin01(const arma::mat& X) {
  const double pi = arma::datum::pi;
  const arma::vec x1 = 15 * X.col(0) - 5, x2 = 15 * X.col(1);
  return arma::square(x2 - 5.1 / (4 * pi * pi) * arma::square(x1) + 5 / pi * x1 - 6)
         + 10 * (1 - 1 / (8 * pi)) * arma::cos(x1) + 10;
}

arma::mat next_ulp(arma::mat X) {
  X.transform([](double v) { return std::nextafter(v, 2.0); });
  return X;
}

// log-likelihood and predicted mean at a fixed theta
std::pair<double, arma::vec> ll_and_mean(const arma::vec& y,
                                         const arma::mat& X,
                                         const std::string& kernel,
                                         const arma::rowvec& theta,
                                         const std::string& method,
                                         const arma::mat& Xt) {
  Kriging::Parameters fixed;
  fixed.theta = arma::mat(theta);
  fixed.is_theta_estim = false;
  Kriging k(y, X, kernel, Trend::RegressionModel::Constant, false, "none", "LL", fixed, method);
  if (method == "exact")
    return {k.logLikelihood(), std::get<0>(k.predict(Xt, false, false, false))};
  return {k.logLikelihood("nystrom"), std::get<0>(k.predictNystrom(Xt, false))};
}

}  // namespace

TEST_CASE("Nystrom is not sensitive to rounding at the landmark rows (#410)", "[nystrom][kriging]") {
  arma::arma_rng::set_seed(410);

  SECTION("k = n reproduces the exact model") {
    const arma::mat X = arma::randu<arma::mat>(150, 3);
    const arma::vec y = hartmann3(X);
    const arma::mat Xt = arma::randu<arma::mat>(50, 3);
    const arma::rowvec theta = {1.2, 0.5, 0.3};  // cond(R) ~ 1e7
    const auto [ll_e, m_e] = ll_and_mean(y, X, "matern5_2", theta, "exact", Xt);
    const auto [ll_n, m_n] = ll_and_mean(y, X, "matern5_2", theta, "nystrom(150)", Xt);
    const auto [ll_u, m_u] = ll_and_mean(y, next_ulp(X), "matern5_2", theta, "nystrom(150)", Xt);
    const double sd = arma::stddev(y);
    INFO("LL exact " << ll_e << " nystrom(n) " << ll_n << " (1 ulp: " << ll_u << "), mean gap "
                     << arma::abs(m_n - m_e).max() / sd << " (1 ulp: " << arma::abs(m_n - m_u).max() / sd << ")");
    CHECK(std::abs(ll_n - ll_e) < 1e-9 * std::abs(ll_e));  // was 3e-7..2e-6
    CHECK(arma::abs(m_n - m_e).max() < 1e-8 * sd);         // was 3e-4..6e-4
    CHECK(std::abs(ll_n - ll_u) < 1e-9 * std::abs(ll_e));
    CHECK(arma::abs(m_n - m_u).max() < 1e-8 * sd);
  }

  SECTION("small k with an ill-conditioned landmark block") {
    const arma::mat X = arma::randu<arma::mat>(50, 2);
    const arma::vec y = branin01(X);
    const arma::mat Xt = arma::randu<arma::mat>(50, 2);
    const arma::rowvec theta = {7.5, 9.5};  // smooth Gaussian kernel: cond(R_ss) ~ 1e10
    const auto [ll_e, m_e] = ll_and_mean(y, X, "gauss", theta, "exact", Xt);
    const auto [ll_n, m_n] = ll_and_mean(y, X, "gauss", theta, "nystrom(10)", Xt);
    const auto [ll_u, m_u] = ll_and_mean(y, next_ulp(X), "gauss", theta, "nystrom(10)", Xt);
    const double sd = arma::stddev(y);
    INFO("LL exact " << ll_e << " nystrom(10) " << ll_n << " (1 ulp: " << ll_u << "), mean gap to exact "
                     << arma::abs(m_n - m_e).max() / sd << ", 1 ulp: " << arma::abs(m_n - m_u).max() / sd);
    CHECK(std::abs(ll_n - ll_u) < 1e-7 * std::abs(ll_n));
    CHECK(arma::abs(m_n - m_u).max() < 1e-4 * sd);  // was ~0.6
    CHECK(arma::abs(m_n - m_e).max() < 0.3 * sd);   // was ~1 (0.09 now)
  }
}
