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

TEST_CASE("LLVecchia objective spec parsing and validation", "[vecchia][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(60, X, y);

  // valid specs fit fine
  CHECK_NOTHROW(Kriging(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLVecchia"));
  CHECK_NOTHROW(Kriging(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLVecchia(10)"));

  // malformed specs throw
  for (const std::string bad : {"LLVecchia()", "LLVecchia(x)", "LLVecchia(0)", "LLVecchia(-3)", "LLVecchia(10"}) {
    CHECK_THROWS_AS(Kriging(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", bad),
                    std::invalid_argument);
  }

  // LLVecchia is not available with a nugget/noise channel
  Kriging knug("matern5_2", Kriging::NoiseModel::Nugget);
  CHECK_THROWS_AS(knug.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LLVecchia(10)", {}),
                  std::invalid_argument);
}

TEST_CASE("LLVecchia(n-1) matches the exact concentrated log-likelihood", "[vecchia][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(60, X, y);

  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLVecchia(59)");

  // compare at moderate theta values: at the *fitted* theta of such a smooth
  // function, R is near-singular and jitter placement dominates both sides
  for (const arma::vec theta : {arma::vec{0.2, 0.2}, arma::vec{0.4, 0.3}}) {
    auto [vll, gv] = k.logLikelihoodVecchiaFun(theta, false);
    auto [ll, gl] = k.logLikelihoodFun(theta, false, false);
    INFO("theta=" << theta.t() << ": LLVecchia(n-1) = " << vll << " vs exact LL = " << ll);
    CHECK(std::abs(vll - ll) < 1e-3 * std::abs(ll) + 1e-3);
  }
}

TEST_CASE("LLVecchia analytic gradient matches finite differences", "[vecchia][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(100, X, y);

  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLVecchia(15)");

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

TEST_CASE("LLVecchia fit commits an exact model: predict interpolates", "[vecchia][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(150, X, y);

  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLVecchia(20)");

  // the final commit performs one exact factorization at theta*, so the
  // fitted model must interpolate exactly like an "LL"-fitted Kriging
  auto [mean, stdev, cov, dm, ds] = k.predict(X, true, false, false);
  CHECK(arma::abs(mean - y).max() < 1e-3);
  CHECK(stdev.max() < 1e-2);
}

TEST_CASE("LLVecchia(20) estimation is consistent with the exact MLE", "[vecchia][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(300, X, y);

  arma::mat Xt;
  arma::vec yt;
  make_data(150, Xt, yt, 456);

  Kriging k_ll(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LL");
  Kriging k_v(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLVecchia(20)");
  CHECK(k_v.vecchia_neighbors() == 20);

  auto [m_ll, s1, c1, d1, e1] = k_ll.predict(Xt, false, false, false);
  auto [m_v, s2, c2, d2, e2] = k_v.predict(Xt, false, false, false);
  const double rmse_ll = std::sqrt(arma::mean(arma::square(m_ll - yt)));
  const double rmse_v = std::sqrt(arma::mean(arma::square(m_v - yt)));
  INFO("theta LL = " << k_ll.theta().t() << " theta LLVecchia = " << k_v.theta().t());
  INFO("rmse LL = " << rmse_ll << " rmse LLVecchia = " << rmse_v);
  // predictive accuracy should be comparable (both use exact predict at their theta*)
  CHECK(rmse_v < 2.0 * rmse_ll + 0.05 * arma::stddev(y));
}

TEST_CASE("LLVecchia model supports update with refit", "[vecchia][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(80, X, y);

  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLVecchia(10)");

  arma::mat Xu;
  arma::vec yu;
  make_data(20, Xu, yu, 456);
  CHECK_NOTHROW(k.update(yu, Xu, true));
  CHECK(k.X().n_rows == 100);

  auto [mean, s, c, d, e] = k.predict(k.X(), false, false, false);
  CHECK(arma::abs(mean - k.y()).max() < 1e-3);  // still interpolates after refit
}

TEST_CASE("LLVecchia large-n smoke test", "[vecchia][kriging][intensive]") {
  arma::mat X;
  arma::vec y;
  make_data(2000, X, y);

  // starting theta to keep optimizer iterations low; wall time is dominated by
  // the single exact O(n^3) factorization at commit (LLVecchia evals are ~50ms here)
  Kriging::Parameters params;
  params.theta = arma::mat(1, 2, arma::fill::value(0.3));

  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLVecchia(20)", params);

  arma::mat Xt;
  arma::vec yt;
  make_data(200, Xt, yt, 456);
  auto [mean, stdev, c, d, e] = k.predict(Xt, true, false, false);
  CHECK(mean.is_finite());
  const double rmse = std::sqrt(arma::mean(arma::square(mean - yt)));
  INFO("rmse = " << rmse << " vs sd(y) = " << arma::stddev(y));
  CHECK(rmse < 0.05 * arma::stddev(y));
}

TEST_CASE("predictVecchia matches exact predict", "[vecchia][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(400, X, y);

  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLVecchia(20)");

  arma::mat Xt;
  arma::vec yt;
  make_data(100, Xt, yt, 456);

  auto [m_ex, s_ex, c, dm, ds] = k.predict(Xt, true, false, false);
  auto [m_v, s_v] = k.predictVecchia(Xt, true, 60);

  INFO("max |mean diff| = " << arma::abs(m_v - m_ex).max());
  INFO("max |stdev diff| = " << arma::abs(s_v - s_ex).max());
  // with 60 of 400 neighbors, local kriging is near-indistinguishable from exact
  CHECK(arma::abs(m_v - m_ex).max() < 1e-2 * arma::stddev(y));
  CHECK(arma::abs(s_v - s_ex).max() < 2e-2 * arma::stddev(y));

  // interpolation: a prediction point equal to an observation keeps it in its
  // own neighborhood => exact interpolation
  auto [m_at_X, s_at_X] = k.predictVecchia(X.rows(0, 49), true, 30);
  CHECK(arma::abs(m_at_X - y.head(50)).max() < 1e-3);
  CHECK(s_at_X.max() < 1e-2);
}

TEST_CASE("predictVecchia works after a plain LL fit and defaults m", "[vecchia][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(200, X, y);

  Kriging k(y, X, "matern5_2");  // objective="LL": no Vecchia sets
  CHECK(k.vecchia_neighbors() == 0);

  arma::mat Xt(50, 2, arma::fill::randu);
  auto [m_v, s_v] = k.predictVecchia(Xt, true);  // m defaults to 30
  auto [m_ex, s_ex, c, dm, ds] = k.predict(Xt, true, false, false);
  CHECK(arma::abs(m_v - m_ex).max() < 5e-2 * arma::stddev(y));
  CHECK(m_v.is_finite());
  CHECK(s_v.min() >= 0.0);

  // dimension check
  arma::mat Xbad(10, 3, arma::fill::randu);
  CHECK_THROWS_AS(k.predictVecchia(Xbad, true), std::invalid_argument);
}

TEST_CASE("light Vecchia fit skips the exact factorization", "[vecchia][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(300, X, y);

  // reference: standard LLVecchia fit (exact commit)
  Kriging k_ref(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLVecchia(20)");

  // light fit: same objective, no exact factorization at commit
  Kriging k("matern5_2");
  k.set_vecchia_exact_commit(false);
  k.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LLVecchia(20)", {});
  CHECK(k.is_vecchia_light());

  // hyperparameters consistent with the exact-commit fit
  INFO("theta light = " << k.theta().t() << " theta ref = " << k_ref.theta().t());
  CHECK(arma::abs(arma::log(k.theta()) - arma::log(k_ref.theta())).max() < 0.5);
  CHECK(k.sigma2() > 0);

  // predict auto-routes to predictVecchia (mean/stdev only)
  arma::mat Xt;
  arma::vec yt;
  make_data(100, Xt, yt, 456);
  auto [mean, stdev, cov, dm, ds] = k.predict(Xt, true, false, false);
  auto [m_v, s_v] = k.predictVecchia(Xt, true);
  CHECK(arma::abs(mean - m_v).max() == 0.0);  // same code path
  CHECK(arma::abs(stdev - s_v).max() == 0.0);
  const double rmse = std::sqrt(arma::mean(arma::square(mean - yt)));
  INFO("rmse = " << rmse);
  CHECK(rmse < 0.05 * arma::stddev(y));

  // interpolation still holds through the routed predict
  auto [m_at_X, s_at_X, c2, d2, e2] = k.predict(X.rows(0, 49), true, false, false);
  CHECK(arma::abs(m_at_X - y.head(50)).max() < 1e-3);

  // unsupported operations throw with a clear message
  CHECK_THROWS_AS(k.predict(Xt, true, true, false), std::runtime_error);  // return_cov
  CHECK_THROWS_AS(k.predict(Xt, true, false, true), std::runtime_error);  // return_deriv
  CHECK_THROWS_AS(k.simulate(3, 123, Xt, false), std::runtime_error);
  arma::vec yu(5, arma::fill::randu);
  arma::mat Xu(5, 2, arma::fill::randu);
  CHECK_THROWS_AS(k.update(yu, Xu, true), std::runtime_error);
  CHECK_THROWS_AS(k.save("/tmp/should_not_exist.json"), std::runtime_error);

  // a subsequent standard fit clears the light state
  k.set_vecchia_exact_commit(true);
  k.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LLVecchia(20)", {});
  CHECK(!k.is_vecchia_light());
  CHECK_NOTHROW(k.simulate(2, 123, Xt.rows(0, 9), false));
}

TEST_CASE("light Vecchia full pipeline at large n", "[vecchia][kriging][intensive]") {
  arma::mat X;
  arma::vec y;
  make_data(10000, X, y);

  Kriging::Parameters params;
  params.theta = arma::mat(1, 2, arma::fill::value(0.3));

  Kriging k("matern5_2");
  k.set_vecchia_exact_commit(false);
  k.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LLVecchia(20)", params);
  CHECK(k.is_vecchia_light());

  arma::mat Xt;
  arma::vec yt;
  make_data(200, Xt, yt, 456);
  auto [mean, stdev, c, d, e] = k.predict(Xt, true, false, false);
  CHECK(mean.is_finite());
  const double rmse = std::sqrt(arma::mean(arma::square(mean - yt)));
  INFO("rmse = " << rmse << " vs sd(y) = " << arma::stddev(y));
  CHECK(rmse < 0.05 * arma::stddev(y));
}

// Regression test for a real bug: optim="none" used to silently fall through
// to a plain exact factorization for LLVecchia's light mode
// (set_vecchia_exact_commit(false)), ignoring the requested objective (fixed
// alongside the identical LLIterative bug -- see "LLIterative honors
// optim=none identically to optim=BFGS" in KrigingIterativeTest.cpp). The
// contract under test: whatever objective is given at fit time is what
// predict()/etc actually use afterwards, for EVERY supported value of optim.
TEST_CASE("LLVecchia honors optim=none identically to optim=BFGS", "[vecchia][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(60, X, y);
  arma::mat Xt;
  arma::vec yt;
  make_data(20, Xt, yt, 456);

  const std::string optim = GENERATE(as<std::string>{}, "none", "BFGS(1)");
  CAPTURE(optim);

  Kriging::Parameters params;
  params.theta = arma::mat(1, X.n_cols, arma::fill::value(0.3));
  params.is_theta_estim = (optim != "none");  // optim="none" requires a fixed theta

  SECTION("set_vecchia_exact_commit(false): stays a light (non-factorized) fit") {
    Kriging k("matern5_2");
    k.set_vecchia_exact_commit(false);
    k.fit(y, X, Trend::RegressionModel::Constant, false, optim, "LLVecchia(20)", params);
    CHECK(k.is_vecchia_light());

    auto [m_pred, s_pred, cov, dm, ds] = k.predict(Xt, true, false, false);
    auto [m_v, s_v] = k.predictVecchia(Xt, true);
    CHECK(arma::abs(m_pred - m_v).max() == 0.0);
    CHECK(arma::abs(s_pred - s_v).max() == 0.0);
    CHECK_THROWS_AS(k.simulate(3, 123, Xt.rows(0, 2), false), std::runtime_error);
    CHECK_THROWS_AS(k.save("unused.json"), std::runtime_error);
  }

  SECTION("default set_vecchia_exact_commit(true): stays an exact (fully-factorized) fit") {
    // The default is documented (docs/math/Vecchia.md) to commit an exact
    // factorization at the fitted/given theta -- that must hold regardless of
    // optim too, i.e. optim="none" must not accidentally *also* switch this
    // default case into a light fit.
    Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, optim, "LLVecchia(20)", params);
    CHECK_FALSE(k.is_vecchia_light());

    auto [mean, stdev, cov, dm, ds] = k.predict(X, true, false, false);
    CHECK(arma::abs(mean - y).max() < 1e-3);  // exact commit interpolates
  }
}

TEST_CASE("LLVecchia benchmark", "[.benchmark]") {
  arma::mat X;
  arma::vec y;
  make_data(400, X, y);
  const arma::vec theta{0.3, 0.3};

  Kriging k_eval(y, X, "gauss", Trend::RegressionModel::Constant, false, "BFGS", "LLVecchia(30)");

  BENCHMARK("Kriging::fit LLVecchia(30) n=400") {
    return Kriging(y, X, "gauss", Trend::RegressionModel::Constant, false, "BFGS", "LLVecchia(30)");
  };
  BENCHMARK("Kriging::fit LL n=400") {
    return Kriging(y, X, "gauss", Trend::RegressionModel::Constant, false, "BFGS", "LL");
  };
  BENCHMARK("Kriging::logLikelihoodVecchiaFun n=400") {
    return std::get<0>(k_eval.logLikelihoodVecchiaFun(theta, false));
  };
  BENCHMARK("Kriging::logLikelihoodFun n=400") {
    return std::get<0>(k_eval.logLikelihoodFun(theta, false, false));
  };
  BENCHMARK("Kriging::predictVecchia 100pts n=400") {
    return k_eval.predictVecchia(arma::mat(100, 2, arma::fill::randu), true);
  };
}
