// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
// clang-format on

#include <fstream>
#include <sstream>
#include <vector>

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

// fit with objective "LL" and the given method
static Kriging fit_method(const arma::vec& y,
                          const arma::mat& X,
                          const std::string& method,
                          const std::string& optim = "BFGS",
                          const Kriging::Parameters& params = {}) {
  return Kriging(y, X, "matern5_2", Trend::RegressionModel::Constant, false, optim, "LL", params, method);
}

TEST_CASE("Vecchia method spec parsing and validation", "[vecchia][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(60, X, y);

  // valid specs fit fine; the method is stored with its explicit size
  Kriging k_default = fit_method(y, X, "vecchia");
  CHECK(k_default.objective() == "LL");
  CHECK(k_default.method() == "vecchia(30)");
  CHECK(k_default.vecchia_neighbors() == 30);
  Kriging k10 = fit_method(y, X, "vecchia(10)");
  CHECK(k10.method() == "vecchia(10)");
  CHECK(k10.vecchia_neighbors() == 10);
  CHECK(Kriging(y, X, "matern5_2").method() == "exact");

  // malformed specs throw
  for (const std::string bad : {"vecchia()", "vecchia(x)", "vecchia(0)", "vecchia(-3)", "vecchia(10", "Vecchia"}) {
    CHECK_THROWS_AS(fit_method(y, X, bad), std::invalid_argument);
  }
  for (const std::string bad : {"LLVecchia()", "LLVecchia(x)", "LLVecchia(0)", "LLVecchia(-3)", "LLVecchia(10"}) {
    CHECK_THROWS_AS(Kriging(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", bad),
                    std::invalid_argument);
  }

  // LOO and LMP have a Vecchia version too (see KrigingApproxObjectivesTest.cpp)
  for (const std::string objective : {"LOO", "LMP"}) {
    Kriging ko(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", objective, {}, "vecchia(10)");
    CHECK(ko.objective() == objective);
    CHECK(ko.method() == "vecchia(10)");
    CHECK(ko.is_vecchia_light());
  }

  // Vecchia is available with a nugget/noise channel (Nystrom too)
  Kriging knug("matern5_2", Kriging::NoiseModel::Nugget);
  CHECK_NOTHROW(knug.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", {}, "vecchia(10)"));
  CHECK(knug.is_vecchia_light());
  CHECK_NOTHROW(knug.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LLVecchia(10)", {}));
  CHECK_NOTHROW(knug.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", {}, "nystrom(10)"));
  CHECK(knug.is_nystrom_light());
  // LOO / LMP with an approximate method: noise model None only
  CHECK_THROWS_AS(knug.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LMP", {}, "vecchia(10)"),
                  std::invalid_argument);
}

TEST_CASE("deprecated LLVecchia objective is an alias of method vecchia", "[vecchia][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(80, X, y);

  Kriging k_alias(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLVecchia(15)");
  Kriging k = fit_method(y, X, "vecchia(15)");
  CHECK(k_alias.objective() == "LL");
  CHECK(k_alias.method() == "vecchia(15)");
  CHECK(k_alias.is_vecchia_light());
  CHECK(arma::approx_equal(k_alias.theta(), k.theta(), "absdiff", 0.0));
  CHECK(k_alias.sigma2() == k.sigma2());

  // "LLVecchia" defaults to m = 30; consistent explicit method accepted
  CHECK(Kriging(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLVecchia").method()
        == "vecchia(30)");
  CHECK_NOTHROW(
      Kriging(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLVecchia(15)", {}, "vecchia(15)"));
  // conflicting method raises
  CHECK_THROWS_AS(
      Kriging(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLVecchia(15)", {}, "vecchia(10)"),
      std::invalid_argument);
  CHECK_THROWS_AS(
      Kriging(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLVecchia(15)", {}, "nystrom"),
      std::invalid_argument);

  // the deprecation goes through the handler (bindings: native warnings)
  std::vector<std::string> messages;
  Kriging::set_deprecation_handler([&messages](const std::string& m) { messages.push_back(m); });
  Kriging(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLVecchia(15)");
  Kriging(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLNystrom");
  fit_method(y, X, "vecchia(15)");  // no deprecation
  REQUIRE(messages.size() == 2);
  CHECK(messages[0] == "objective \"LLVecchia(15)\" is deprecated; use objective=\"LL\", method=\"vecchia(15)\"");
  CHECK(messages[1] == "objective \"LLNystrom\" is deprecated; use objective=\"LL\", method=\"nystrom(50)\"");
  CHECK(Kriging::deprecated_objective_message("LL").empty());
  // a throwing handler turns the deprecation into an error
  Kriging::set_deprecation_handler([](const std::string& m) { throw std::runtime_error(m); });
  CHECK_THROWS_AS(Kriging(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLVecchia(15)"),
                  std::runtime_error);
  Kriging::set_deprecation_handler(nullptr);  // default: printed once

  // set_vecchia_exact_commit is deprecated: false is a no-op, true raises
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
  CHECK_NOTHROW(k.set_vecchia_exact_commit(false));
  CHECK_THROWS_AS(k.set_vecchia_exact_commit(true), std::invalid_argument);
  CHECK_FALSE(k.vecchia_exact_commit());
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
}

TEST_CASE("LLVecchia(n-1) matches the exact concentrated log-likelihood", "[vecchia][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(60, X, y);

  Kriging k = fit_method(y, X, "vecchia(59)");

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

  Kriging k = fit_method(y, X, "vecchia(15)");

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

TEST_CASE("Vecchia fit: the Vecchia predictor interpolates", "[vecchia][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(150, X, y);

  Kriging k = fit_method(y, X, "vecchia(20)");

  auto [mean, stdev, cov, dm, ds] = k.predict(X, true, false, false, "vecchia");
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
  Kriging k_vecchia = fit_method(y, X, "vecchia(20)");
  CHECK(k_vecchia.vecchia_neighbors() == 20);
  // exact model at the Vecchia estimate: separate fit, optim "none"
  Kriging::Parameters at_vecchia;
  at_vecchia.theta = arma::mat(k_vecchia.theta().t());
  at_vecchia.is_theta_estim = false;
  Kriging k_v = fit_method(y, X, "exact", "none", at_vecchia);

  auto [m_ll, s1, c1, d1, e1] = k_ll.predict(Xt, false, false, false);
  auto [m_v, s2, c2, d2, e2] = k_v.predict(Xt, false, false, false);
  const double rmse_ll = std::sqrt(arma::mean(arma::square(m_ll - yt)));
  const double rmse_v = std::sqrt(arma::mean(arma::square(m_v - yt)));
  INFO("theta LL = " << k_ll.theta().t() << " theta LLVecchia = " << k_v.theta().t());
  INFO("rmse LL = " << rmse_ll << " rmse LLVecchia = " << rmse_v);
  // predictive accuracy should be comparable (both use exact predict at their theta*)
  CHECK(rmse_v < 2.0 * rmse_ll + 0.05 * arma::stddev(y));
}

TEST_CASE("Vecchia model supports update with refit", "[vecchia][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(80, X, y);

  Kriging k = fit_method(y, X, "vecchia(10)");

  arma::mat Xu;
  arma::vec yu;
  make_data(20, Xu, yu, 456);
  CHECK_NOTHROW(k.update(yu, Xu, true));
  CHECK(k.X().n_rows == 100);
  CHECK(k.method() == "vecchia(10)");

  auto [mean, s, c, d, e] = k.predict(Xu, false, false, false, "vecchia");
  CHECK(arma::abs(mean - yu).max() < 1e-3);  // still interpolates after refit
}

TEST_CASE("Vecchia large-n smoke test", "[vecchia][kriging][intensive]") {
  arma::mat X;
  arma::vec y;
  make_data(2000, X, y);

  // starting theta to keep optimizer iterations low
  Kriging::Parameters params;
  params.theta = arma::mat(1, 2, arma::fill::value(0.3));

  Kriging k = fit_method(y, X, "vecchia(20)", "BFGS", params);

  arma::mat Xt;
  arma::vec yt;
  make_data(200, Xt, yt, 456);
  auto [mean, stdev, c, d, e] = k.predict(Xt, true, false, false, "vecchia");
  CHECK(mean.is_finite());
  const double rmse = std::sqrt(arma::mean(arma::square(mean - yt)));
  INFO("rmse = " << rmse << " vs sd(y) = " << arma::stddev(y));
  CHECK(rmse < 0.05 * arma::stddev(y));
}

TEST_CASE("predictVecchia matches exact predict", "[vecchia][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(400, X, y);

  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LL");

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

TEST_CASE("Vecchia fit keeps no exact factorization", "[vecchia][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(300, X, y);

  // reference: exact LL fit
  Kriging k_ref(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LL");

  Kriging k = fit_method(y, X, "vecchia(20)");
  CHECK(k.is_vecchia_light());
  CHECK(k.T().is_empty());

  // hyperparameters consistent with the exact fit
  INFO("theta Vecchia = " << k.theta().t() << " theta exact = " << k_ref.theta().t());
  CHECK(arma::abs(arma::log(k.theta()) - arma::log(k_ref.theta())).max() < 0.5);
  CHECK(k.sigma2() > 0);

  // method "vecchia" = predictVecchia (mean/stdev only); no implicit routing
  arma::mat Xt;
  arma::vec yt;
  make_data(100, Xt, yt, 456);
  auto [mean, stdev, cov, dm, ds] = k.predict(Xt, true, false, false, "vecchia");
  auto [m_v, s_v] = k.predictVecchia(Xt, true);
  CHECK(arma::abs(mean - m_v).max() == 0.0);  // same code path
  CHECK(arma::abs(stdev - s_v).max() == 0.0);
  const double rmse = std::sqrt(arma::mean(arma::square(mean - yt)));
  INFO("rmse = " << rmse);
  CHECK(rmse < 0.05 * arma::stddev(y));

  // interpolation holds with the Vecchia predictor
  auto [m_at_X, s_at_X, c2, d2, e2] = k.predict(X.rows(0, 49), true, false, false, "vecchia");
  CHECK(arma::abs(m_at_X - y.head(50)).max() < 1e-3);

  // the exact methods need the exact factorization: explicit error, never a
  // silent switch to the approximation
  CHECK_THROWS_AS(k.predict(Xt, true, false, false), std::runtime_error);
  CHECK_THROWS_AS(k.predict(Xt, true, false, false, "exact"), std::runtime_error);
  CHECK_THROWS_AS(k.simulate(3, 123, Xt, false), std::runtime_error);
  CHECK_THROWS_AS(k.predict(Xt, true, true, false, "vecchia"), std::invalid_argument);   // return_cov
  CHECK_THROWS_AS(k.predict(Xt, true, false, true, "vecchia"), std::invalid_argument);   // return_deriv
  CHECK_THROWS_AS(k.simulate(3, 123, Xt, "vecchia", true), std::invalid_argument);       // will_update
  CHECK_THROWS_AS(k.predict(Xt, true, false, false, "nystrom"), std::runtime_error);     // no Nystrom fit
  CHECK_THROWS_AS(k.predict(Xt, true, false, false, "Vecchia"), std::invalid_argument);  // unknown method
  CHECK(arma::approx_equal(k.simulate(3, 123, Xt, "vecchia"), k.simulateVecchia(3, 123, Xt), "absdiff", 0.0));
  CHECK(arma::approx_equal(k.simulate(3, 123, Xt, "vecchia(5)"), k.simulateVecchia(3, 123, Xt, 5), "absdiff", 0.0));

  // the Vecchia log-likelihood is the fitted objective
  CHECK(k.logLikelihood("vecchia") == std::get<0>(k.logLikelihoodVecchiaFun(k.theta(), false)));
  CHECK(k.logLikelihood(k.method()) == k.logLikelihood("vecchia"));  // fitted method string accepted
  CHECK_THROWS_AS(k.logLikelihood("nystrom"), std::runtime_error);

  // update (Vecchia refit) and save / load work on a light fit
  arma::mat Xu;
  arma::vec yu;
  make_data(20, Xu, yu, 789);
  CHECK_NOTHROW(k.update(yu, Xu, false));
  CHECK(k.is_vecchia_light());
  CHECK(k.X().n_rows == 320);
  CHECK_NOTHROW(k.update(yu, Xu, true));
  CHECK(k.X().n_rows == 340);
  auto [m_u, s_u, cu, du, eu] = k.predict(Xu, true, false, false, "vecchia");
  CHECK(arma::abs(m_u - yu).max() < 1e-3);  // interpolates the added data
  const std::string fname = "light_vecchia_saved.json";
  k.save(fname);
  Kriging kl = Kriging::load(fname);
  CHECK(kl.is_vecchia_light());
  CHECK(kl.vecchia_neighbors() == 20);
  auto [m_l, s_l, cl, dl, el] = kl.predict(Xt, true, false, false, "vecchia");
  auto [m_k, s_k, ck, dk, ek] = k.predict(Xt, true, false, false, "vecchia");
  CHECK(arma::abs(m_l - m_k).max() < 1e-12);
  CHECK(std::abs(kl.logLikelihood("vecchia") - k.logLikelihood("vecchia")) < 1e-9);
  CHECK_THROWS_AS(kl.predict(Xt, true, false, false), std::runtime_error);

  // a subsequent exact fit clears the Vecchia state
  k.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", {}, "exact");
  CHECK(!k.is_vecchia_light());
  CHECK(k.method() == "exact");
  CHECK(k.vecchia_neighbors() == 0);
  CHECK_NOTHROW(k.simulate(2, 123, Xt.rows(0, 9), false));
  // exact and Vecchia predictors both available on an exact fit
  CHECK_NOTHROW(k.predict(Xt, true, true, false, "exact"));
  CHECK_NOTHROW(k.predict(Xt, true, false, false, "vecchia"));
  CHECK(k.logLikelihood("exact") == k.logLikelihood());
  CHECK_THROWS_AS(k.logLikelihood("vecchia"), std::runtime_error);  // no Vecchia fit

  // exact model at the Vecchia estimate: exact refit with optim "none"
  Kriging kv = fit_method(y, X, "vecchia(20)");
  Kriging::Parameters at_vecchia;
  at_vecchia.theta = arma::mat(kv.theta().t());
  at_vecchia.is_theta_estim = false;
  Kriging ke = fit_method(y, X, "exact", "none", at_vecchia);
  CHECK(arma::approx_equal(ke.theta(), kv.theta(), "absdiff", 0.0));
  CHECK_NOTHROW(ke.predict(Xt, true, true, false));
}

TEST_CASE("save files written before the fit method load as Vecchia fits", "[vecchia][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(80, X, y);
  arma::mat Xt;
  arma::vec yt;
  make_data(10, Xt, yt, 456);

  // libKriging <= 1.2 wrote an LLVecchia fit as objective "LLVecchia(m)" with
  // its exact factorization: emulate it with an exact fit renamed in the file
  Kriging k_exact(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LL");
  const std::string fname = "legacy_llvecchia_saved.json";
  k_exact.save(fname);
  {
    std::ifstream in(fname);
    std::stringstream ss;
    ss << in.rdbuf();
    std::string txt = ss.str();
    const std::string key = "\"objective\": \"LL\"";
    const auto pos = txt.find(key);
    REQUIRE(pos != std::string::npos);
    txt.replace(pos, key.size(), "\"objective\": \"LLVecchia(12)\"");
    std::ofstream out(fname);
    out << txt;
  }
  Kriging kl = Kriging::load(fname);
  CHECK(kl.objective() == "LL");
  CHECK(kl.method() == "vecchia(12)");
  CHECK(kl.vecchia_neighbors() == 12);
  // the stored factorization keeps the exact methods available
  CHECK_FALSE(kl.is_vecchia_light());
  auto [m_l, s_l, c_l, d_l, e_l] = kl.predict(Xt, true, false, false);
  auto [m_e, s_e, c_e, d_e, e_e] = k_exact.predict(Xt, true, false, false);
  CHECK(arma::abs(m_l - m_e).max() < 1e-12);
  CHECK_NOTHROW(kl.logLikelihood("vecchia"));
  // until an update, which is a Vecchia refit
  kl.update(yt, Xt, false);
  CHECK(kl.is_vecchia_light());
  CHECK_THROWS_AS(kl.predict(Xt, true, false, false), std::runtime_error);
  auto [m_u, s_u, c_u, d_u, e_u] = kl.predict(Xt, true, false, false, "vecchia");
  CHECK(arma::abs(m_u - yt).max() < 1e-3);
}

TEST_CASE("light Vecchia full pipeline at large n", "[vecchia][kriging][intensive]") {
  arma::mat X;
  arma::vec y;
  make_data(10000, X, y);

  Kriging::Parameters params;
  params.theta = arma::mat(1, 2, arma::fill::value(0.3));

  Kriging k = fit_method(y, X, "vecchia(20)", "BFGS", params);
  CHECK(k.is_vecchia_light());

  arma::mat Xt;
  arma::vec yt;
  make_data(200, Xt, yt, 456);
  auto [mean, stdev, c, d, e] = k.predict(Xt, true, false, false, "vecchia");
  CHECK(mean.is_finite());
  const double rmse = std::sqrt(arma::mean(arma::square(mean - yt)));
  INFO("rmse = " << rmse << " vs sd(y) = " << arma::stddev(y));
  CHECK(rmse < 0.05 * arma::stddev(y));
}

// Regression test for a real bug: optim="none" used to silently fall through
// to a plain exact factorization for a light Vecchia fit, ignoring the
// requested approximation. The contract under test: a method="vecchia" fit
// holds no exact factorization afterwards, for EVERY supported value of optim.
TEST_CASE("Vecchia method honors optim=none identically to optim=BFGS", "[vecchia][kriging]") {
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

  {
    Kriging k = fit_method(y, X, "vecchia(20)", optim, params);
    CHECK(k.is_vecchia_light());

    auto [m_pred, s_pred, cov, dm, ds] = k.predict(Xt, true, false, false, "vecchia");
    auto [m_v, s_v] = k.predictVecchia(Xt, true);
    CHECK(arma::abs(m_pred - m_v).max() == 0.0);
    CHECK(arma::abs(s_pred - s_v).max() == 0.0);
    // no exact factorization: the exact methods raise
    CHECK_THROWS_AS(k.predict(Xt, true, false, false), std::runtime_error);
    CHECK_THROWS_AS(k.simulate(3, 123, Xt.rows(0, 2), false), std::runtime_error);
  }
}

TEST_CASE("simulateVecchia with all predecessors matches the exact conditional law", "[vecchia][kriging]") {
  // No trend: the exact predict() covariance is then the simple-kriging one,
  // which is what sequential conditioning on ALL predecessors (m >= n+q-1)
  // reproduces exactly (chain rule).
  arma::mat X;
  arma::vec y;
  make_data(30, X, y);
  arma::mat Xt;
  arma::vec yt;
  make_data(5, Xt, yt, 456);
  Kriging::Parameters params;
  params.theta = arma::mat(1, X.n_cols, arma::fill::value(0.3));
  params.is_theta_estim = false;
  Kriging k(y, X, "matern5_2", Trend::RegressionModel::None, false, "none", "LL", params);
  auto [mean, stdev, cov, dm, ds] = k.predict(Xt, true, true, false);

  const int nsim = 20000;
  arma::mat sims = k.simulateVecchia(nsim, 42, Xt, X.n_rows + Xt.n_rows);
  REQUIRE(sims.n_rows == Xt.n_rows);
  REQUIRE(sims.n_cols == static_cast<arma::uword>(nsim));
  const arma::vec emp_mean = arma::mean(sims, 1);
  const arma::mat emp_cov = arma::cov(sims.t());
  INFO("mean exact " << mean.t() << "mean emp " << emp_mean.t());
  INFO("cov exact\n" << cov << "cov emp\n" << emp_cov);
  // Monte-Carlo tolerances: ~5 standard errors
  CHECK(arma::all(arma::abs(emp_mean - mean) < 5.0 * stdev / std::sqrt(nsim) + 1e-12));
  const double smax = stdev.max();
  CHECK(arma::abs(emp_cov - cov).max() < 5.0 * std::sqrt(2.0 / nsim) * smax * smax + 1e-12);

  // reproducible with the seed
  CHECK(arma::approx_equal(k.simulateVecchia(10, 7, Xt, 8), k.simulateVecchia(10, 7, Xt, 8), "absdiff", 0.0));
}

TEST_CASE("simulateVecchia first point matches predictVecchia; trajectories are coherent", "[vecchia][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(200, X, y);
  Kriging::Parameters params;
  params.theta = arma::mat(1, X.n_cols, arma::fill::value(0.3));
  params.is_theta_estim = false;
  Kriging k = fit_method(y, X, "vecchia(15)", "none", params);
  REQUIRE(k.is_vecchia_light());

  // the first simulation point conditions on observations only: its marginal
  // law is exactly the predictVecchia one, with the same m
  arma::mat Xt = {{0.31, 0.62}, {0.315, 0.62}, {0.9, 0.1}};
  const int nsim = 20000;
  arma::mat sims = k.simulate(nsim, 3, Xt, "vecchia");
  auto [m_v, s_v] = k.predictVecchia(Xt, true);
  const double emp_m0 = arma::mean(sims.row(0));
  const double emp_s0 = arma::stddev(sims.row(0));
  INFO("predictVecchia " << m_v(0) << " +/- " << s_v(0) << ", emp " << emp_m0 << " +/- " << emp_s0);
  CHECK(std::abs(emp_m0 - m_v(0)) < 5.0 * s_v(0) / std::sqrt(nsim));
  CHECK(std::abs(emp_s0 - s_v(0)) < 5.0 * s_v(0) * std::sqrt(0.5 / nsim));

  // second point is 0.005 away from the first one and conditions on it:
  // strongly correlated trajectories (independent marginals would not be)
  const double corr01 = arma::as_scalar(arma::cor(sims.row(0).t(), sims.row(1).t()));
  CHECK(corr01 > 0.9);

  // at an observed point, every trajectory interpolates
  arma::mat sims_at_X = k.simulate(50, 9, X.row(0), "vecchia");
  CHECK(arma::abs(sims_at_X - y(0)).max() < 1e-3);

  // the exact simulate overloads (incl. nugget / noise) need the exact
  // factorization: explicit error, never a silent switch to simulateVecchia
  CHECK_THROWS_AS(k.simulate(4, 5, Xt, false, false), std::runtime_error);
  CHECK_THROWS_AS(k.simulate(4, 5, Xt, arma::vec{0.0}, false), std::runtime_error);
  CHECK_THROWS_AS(k.update_simulate(y.head(2), X.rows(0, 1)), std::runtime_error);
}

TEST_CASE("simulateVecchia argument checks", "[vecchia][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(40, X, y);
  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LL");
  CHECK_THROWS_AS(k.simulateVecchia(3, 1, arma::mat(2, 3, arma::fill::randu)), std::invalid_argument);
  CHECK_THROWS_AS(k.simulateVecchia(0, 1, X.rows(0, 1)), std::invalid_argument);
  CHECK_NOTHROW(k.simulateVecchia(3, 1, X.rows(0, 1)));  // usable after any (noise-free) fit

  // noise models: usable too (see the noise-model test cases below)
  Kriging knug("matern5_2", Kriging::NoiseModel::Nugget);
  knug.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", {});
  CHECK_NOTHROW(knug.simulateVecchia(3, 1, X.rows(0, 1)));
  CHECK_THROWS_AS(knug.simulateVecchia(3, 1, X.rows(0, 1), 0, false, arma::vec{0.1, 0.2, 0.3}), std::invalid_argument);
}

// =============================================================================
// Noise models (response Vecchia): Nugget and Heterogeneous
// =============================================================================

// noisy observations of f2d: homogeneous (sd 0.1) or heteroscedastic
static void make_noisy_data(arma::uword n, arma::mat& X, arma::vec& y, arma::vec& noise, unsigned seed = 123) {
  make_data(n, X, y, seed);
  noise = 0.005 + 0.02 * X.col(0);  // variances
  arma::arma_rng::set_seed(seed + 1);
  y += arma::sqrt(noise) % arma::randn<arma::vec>(n);
}

static Kriging fit_nugget(const arma::vec& y,
                          const arma::mat& X,
                          const std::string& method,
                          const std::string& optim = "BFGS",
                          const Kriging::Parameters& params = {},
                          Trend::RegressionModel trend = Trend::RegressionModel::Constant) {
  Kriging k("matern5_2", Kriging::NoiseModel::Nugget);
  k.fit(y, X, trend, false, optim, "LL", params, method);
  return k;
}

static Kriging fit_hetero(const arma::vec& y,
                          const arma::vec& noise,
                          const arma::mat& X,
                          const std::string& method,
                          const std::string& optim = "BFGS",
                          const Kriging::Parameters& params = {},
                          Trend::RegressionModel trend = Trend::RegressionModel::Constant) {
  Kriging k("matern5_2", Kriging::NoiseModel::Heterogeneous);
  k.fit(y, noise, X, trend, false, optim, "LL", params, method);
  return k;
}

TEST_CASE("Vecchia(n-1) noise-model likelihood equals the exact one", "[vecchia][kriging][noise]") {
  arma::mat X;
  arma::vec y, noise;
  make_noisy_data(60, X, y, noise);

  SECTION("Nugget, sigma2 and nugget estimated (concentrated)") {
    Kriging k = fit_nugget(y, X, "vecchia(59)");
    for (const arma::vec gamma : {arma::vec{0.2, 0.2, 0.9}, arma::vec{0.4, 0.3, 0.6}}) {
      const double vll = std::get<0>(k.logLikelihoodVecchiaFun(gamma, false));
      const double ll = std::get<0>(k.logLikelihoodFun(gamma, false, false));
      INFO("gamma=" << gamma.t() << ": Vecchia(n-1) = " << vll << " vs exact = " << ll);
      CHECK(std::abs(vll - ll) < 1e-6 * std::abs(ll) + 1e-6);
    }
    // d-vector theta: the committed alpha
    const arma::vec g = arma::join_cols(k.theta(), arma::vec{k.alpha()});
    CHECK(std::get<0>(k.logLikelihoodVecchiaFun(k.theta(), false))
          == Approx(std::get<0>(k.logLikelihoodVecchiaFun(g, false))));
    CHECK(k.logLikelihood("vecchia") == Approx(std::get<0>(k.logLikelihoodFun(g, false, false))).epsilon(1e-6));
  }
  SECTION("Nugget, nugget fixed") {
    Kriging::Parameters params;
    params.nugget = 0.01;
    params.is_nugget_estim = false;
    Kriging k = fit_nugget(y, X, "vecchia(59)", "BFGS", params);
    const arma::vec gamma{0.3, 0.25, 0.8};
    CHECK(std::get<0>(k.logLikelihoodVecchiaFun(gamma, false))
          == Approx(std::get<0>(k.logLikelihoodFun(gamma, false, false))).epsilon(1e-6));
  }
  SECTION("Heterogeneous") {
    Kriging k = fit_hetero(y, noise, X, "vecchia(59)");
    for (const arma::vec gamma : {arma::vec{0.2, 0.2, 0.5}, arma::vec{0.4, 0.3, 1.5}}) {
      const double vll = std::get<0>(k.logLikelihoodVecchiaFun(gamma, false));
      const double ll = std::get<0>(k.logLikelihoodFun(gamma, false, false));
      INFO("gamma=" << gamma.t() << ": Vecchia(n-1) = " << vll << " vs exact = " << ll);
      CHECK(std::abs(vll - ll) < 1e-6 * std::abs(ll) + 1e-6);
    }
  }
}

TEST_CASE("Vecchia noise-model gradient matches finite differences", "[vecchia][kriging][noise]") {
  arma::mat X;
  arma::vec y, noise;
  make_noisy_data(100, X, y, noise);

  auto check_fd = [](Kriging& k, const arma::vec& gamma) {
    auto [vll, grad] = k.logLikelihoodVecchiaFun(gamma, true);
    REQUIRE(grad.n_elem == gamma.n_elem);
    for (arma::uword j = 0; j < gamma.n_elem; ++j) {
      const double h = 1e-6 * std::max(1.0, std::abs(gamma(j)));
      arma::vec gp = gamma, gm = gamma;
      gp(j) += h;
      gm(j) -= h;
      const double fd
          = (std::get<0>(k.logLikelihoodVecchiaFun(gp, false)) - std::get<0>(k.logLikelihoodVecchiaFun(gm, false)))
            / (2 * h);
      INFO("gamma=" << gamma.t() << " dim " << j << ": analytic=" << grad(j) << " fd=" << fd);
      CHECK(std::abs(grad(j) - fd) < 1e-4 * std::max(1.0, std::abs(fd)));
    }
  };

  SECTION("Nugget, concentrated") {
    Kriging k = fit_nugget(y, X, "vecchia(15)");
    check_fd(k, arma::vec{0.2, 0.3, 0.9});
    check_fd(k, arma::vec{0.5, 0.15, 0.5});
  }
  SECTION("Nugget, nugget fixed") {
    Kriging::Parameters params;
    params.nugget = 0.01;
    params.is_nugget_estim = false;
    Kriging k = fit_nugget(y, X, "vecchia(15)", "BFGS", params);
    check_fd(k, arma::vec{0.3, 0.2, 0.8});
  }
  SECTION("Nugget, sigma2 fixed") {
    Kriging::Parameters params;
    params.sigma2 = 1.0;
    params.is_sigma2_estim = false;
    Kriging k = fit_nugget(y, X, "vecchia(15)", "BFGS", params);
    check_fd(k, arma::vec{0.3, 0.2, 0.8});
  }
  SECTION("Heterogeneous") {
    Kriging k = fit_hetero(y, noise, X, "vecchia(15)");
    check_fd(k, arma::vec{0.2, 0.3, 0.7});
    check_fd(k, arma::vec{0.5, 0.15, 2.0});
  }
}

TEST_CASE("Vecchia noise-model fit is consistent with the exact fit", "[vecchia][kriging][noise]") {
  arma::mat X;
  arma::vec y, noise;
  make_noisy_data(200, X, y, noise);
  arma::mat Xt;
  arma::vec yt;
  make_data(100, Xt, yt, 456);  // noise-free truth
  // absolute rmse slack: a third of the smallest noise standard deviation
  const double noise_tol = std::sqrt(noise.min()) / 3.0;

  SECTION("Nugget") {
    Kriging ke = fit_nugget(y, X, "exact");
    Kriging kv = fit_nugget(y, X, "vecchia(20)");
    CHECK(kv.is_vecchia_light());
    CHECK(kv.T().is_empty());
    CHECK(kv.method() == "vecchia(20)");
    INFO("theta exact " << ke.theta().t() << " vecchia " << kv.theta().t());
    INFO("sigma2 exact " << ke.sigma2() << " vecchia " << kv.sigma2());
    INFO("nugget exact " << ke.nugget() << " vecchia " << kv.nugget());
    CHECK(arma::abs(arma::log(kv.theta()) - arma::log(ke.theta())).max() < 0.5);
    CHECK(std::abs(std::log(kv.nugget() / ke.nugget())) < 0.5);
    // large ranges on this smooth function: sigma2 and theta are only weakly
    // identified separately (sigma2 / theta^(2 nu) is), hence the loose check
    // (the data, drawn by Armadillo's RNG, differ between standard libraries)
    CHECK(std::abs(std::log(kv.sigma2() / ke.sigma2())) < 1.5);
    // committed parameters are consistent: alpha = sigma2 / (sigma2 + nugget)
    CHECK(kv.alpha() == Approx(kv.sigma2() / (kv.sigma2() + kv.nugget())));

    auto [m_e, s_e, c_e, d_e, e_e] = ke.predict(Xt, true, false, false);
    auto [m_v, s_v, c_v, d_v, e_v] = kv.predict(Xt, true, false, false, "vecchia");
    const double rmse_e = std::sqrt(arma::mean(arma::square(m_e - yt)));
    const double rmse_v = std::sqrt(arma::mean(arma::square(m_v - yt)));
    INFO("rmse exact " << rmse_e << " vecchia " << rmse_v);
    CHECK(rmse_v < 1.5 * rmse_e + noise_tol);
    CHECK(arma::abs(s_v - s_e).max() < 0.3 * s_e.max());
    // exact methods unavailable, Vecchia ones available
    CHECK_THROWS_AS(kv.predict(Xt, true, false, false), std::runtime_error);
    CHECK_THROWS_AS(kv.simulate(2, 1, Xt, true, false), std::runtime_error);
    CHECK(kv.simulateVecchia(2, 1, Xt).is_finite());
  }
  SECTION("Heterogeneous") {
    Kriging ke = fit_hetero(y, noise, X, "exact");
    Kriging kv = fit_hetero(y, noise, X, "vecchia(20)");
    CHECK(kv.is_vecchia_light());
    INFO("theta exact " << ke.theta().t() << " vecchia " << kv.theta().t());
    INFO("sigma2 exact " << ke.sigma2() << " vecchia " << kv.sigma2());
    CHECK(arma::abs(arma::log(kv.theta()) - arma::log(ke.theta())).max() < 0.5);
    CHECK(std::abs(std::log(kv.sigma2() / ke.sigma2())) < 0.5);

    auto [m_e, s_e, c_e, d_e, e_e] = ke.predict(Xt, true, false, false);
    auto [m_v, s_v, c_v, d_v, e_v] = kv.predict(Xt, true, false, false, "vecchia");
    const double rmse_e = std::sqrt(arma::mean(arma::square(m_e - yt)));
    const double rmse_v = std::sqrt(arma::mean(arma::square(m_v - yt)));
    INFO("rmse exact " << rmse_e << " vecchia " << rmse_v);
    CHECK(rmse_v < 1.5 * rmse_e + noise_tol);
  }
  SECTION("optim none") {
    Kriging::Parameters params;
    params.theta = arma::mat{{0.3, 0.3}};
    params.is_theta_estim = false;
    params.sigma2 = 1.0;
    params.nugget = 0.01;
    Kriging ke = fit_nugget(y, X, "exact", "none", params);
    Kriging kv = fit_nugget(y, X, "vecchia(20)", "none", params);
    CHECK(kv.is_vecchia_light());
    CHECK(kv.alpha() == Approx(ke.alpha()));
    CHECK(kv.sigma2() == Approx(ke.sigma2()).epsilon(0.2));
    Kriging::Parameters hparams;
    hparams.theta = arma::mat{{0.3, 0.3}};
    hparams.is_theta_estim = false;
    hparams.sigma2 = 1.0;
    hparams.is_sigma2_estim = false;
    Kriging kh = fit_hetero(y, noise, X, "vecchia(20)", "none", hparams);
    CHECK(kh.is_vecchia_light());
    CHECK(kh.sigma2() == 1.0);
  }
}

TEST_CASE("predictVecchia with all neighbors equals exact noise-model predict", "[vecchia][kriging][noise]") {
  // no trend: the exact predictor variance is then the simple-kriging one,
  // which local kriging on ALL observations reproduces exactly
  arma::mat X;
  arma::vec y, noise;
  make_noisy_data(50, X, y, noise);
  arma::mat Xt;
  arma::vec yt;
  make_data(20, Xt, yt, 456);
  Xt = arma::join_cols(Xt, X.rows(0, 4));  // includes observed points

  Kriging::Parameters params;
  params.theta = arma::mat{{0.3, 0.25}};
  params.is_theta_estim = false;
  params.sigma2 = 1.2;
  params.nugget = 0.02;

  SECTION("Nugget") {
    Kriging k = fit_nugget(y, X, "exact", "none", params, Trend::RegressionModel::None);
    auto [m_e, s_e, c_e, d_e, e_e] = k.predict(Xt, true, false, false);
    auto [m_v, s_v] = k.predictVecchia(Xt, true, X.n_rows);
    INFO("max |mean diff| = " << arma::abs(m_v - m_e).max() << ", max |sd diff| = " << arma::abs(s_v - s_e).max());
    CHECK(arma::abs(m_v - m_e).max() < 1e-6);
    CHECK(arma::abs(s_v - s_e).max() < 1e-4);
    // the nugget convention: at an observed point, the observation is returned
    CHECK(arma::abs(m_v.tail(5) - y.head(5)).max() < 1e-6);
  }
  SECTION("Heterogeneous") {
    // sigma2 fixed: with optim="none" an estimated sigma2 is re-profiled after
    // the factorization is built at the given one (exact fit convention)
    params.is_sigma2_estim = false;
    Kriging k = fit_hetero(y, noise, X, "exact", "none", params, Trend::RegressionModel::None);
    auto [m_e, s_e, c_e, d_e, e_e] = k.predict(Xt, true, false, false);
    auto [m_v, s_v] = k.predictVecchia(Xt, true, X.n_rows);
    INFO("max |mean diff| = " << arma::abs(m_v - m_e).max() << ", max |sd diff| = " << arma::abs(s_v - s_e).max());
    CHECK(arma::abs(m_v - m_e).max() < 1e-6);
    CHECK(arma::abs(s_v - s_e).max() < 1e-4);
  }
}

TEST_CASE("simulateVecchia with all predecessors matches the exact noise-model law", "[vecchia][kriging][noise]") {
  arma::mat X;
  arma::vec y, noise;
  make_noisy_data(30, X, y, noise);
  arma::mat Xt;
  arma::vec yt;
  make_data(5, Xt, yt, 456);
  Kriging::Parameters params;
  params.theta = arma::mat{{0.3, 0.3}};
  params.is_theta_estim = false;
  params.sigma2 = 1.0;
  params.nugget = 0.05;
  const int nsim = 20000;
  const arma::uword m_all = X.n_rows + Xt.n_rows;

  auto check_law = [&](const arma::mat& sims, const arma::vec& mean, const arma::mat& cov) {
    const arma::vec emp_mean = arma::mean(sims, 1);
    const arma::mat emp_cov = arma::cov(sims.t());
    const arma::vec sd = arma::sqrt(cov.diag());
    INFO("mean exact " << mean.t() << "mean emp " << emp_mean.t());
    INFO("cov exact\n" << cov << "cov emp\n" << emp_cov);
    CHECK(arma::all(arma::abs(emp_mean - mean) < 5.0 * sd / std::sqrt(nsim) + 1e-12));
    const double smax = sd.max();
    CHECK(arma::abs(emp_cov - cov).max() < 5.0 * std::sqrt(2.0 / nsim) * smax * smax + 1e-12);
  };

  SECTION("Nugget") {
    Kriging k = fit_nugget(y, X, "exact", "none", params, Trend::RegressionModel::None);
    // predict's covariance is the one of the noisy response (nugget on the diagonal)
    auto [mean, stdev, cov, dm, ds] = k.predict(Xt, true, true, false);
    check_law(k.simulateVecchia(nsim, 42, Xt, m_all, /*with_nugget=*/true), mean, cov);
    // latent process: without the nugget variance
    const arma::mat cov_latent = cov - k.nugget() * arma::eye(Xt.n_rows, Xt.n_rows);
    check_law(k.simulateVecchia(nsim, 43, Xt, m_all), mean, cov_latent);
    // method overloads
    CHECK(arma::approx_equal(
        k.simulate(5, 7, Xt, true, false, "vecchia"), k.simulateVecchia(5, 7, Xt, 0, true), "absdiff", 0.0));
    CHECK(arma::approx_equal(k.simulate(5, 7, Xt, "vecchia"), k.simulateVecchia(5, 7, Xt), "absdiff", 0.0));
    CHECK_THROWS_AS(k.simulate(5, 7, Xt, true, true, "vecchia"), std::invalid_argument);
    // with the nugget, a simulation at an observed point reproduces the observation
    CHECK(arma::abs(k.simulateVecchia(20, 9, X.row(0), 0, true) - y(0)).max() < 1e-4);
  }
  SECTION("Heterogeneous") {
    params.is_sigma2_estim = false;  // see the predictVecchia test above
    Kriging k = fit_hetero(y, noise, X, "exact", "none", params, Trend::RegressionModel::None);
    auto [mean, stdev, cov, dm, ds] = k.predict(Xt, true, true, false);
    check_law(k.simulateVecchia(nsim, 42, Xt, m_all), mean, cov);
    // with_noise (standard deviations) adds independent noise
    const arma::vec sdn = arma::linspace(0.1, 0.3, Xt.n_rows);
    check_law(k.simulateVecchia(nsim, 44, Xt, m_all, false, sdn), mean, cov + arma::diagmat(arma::square(sdn)));
    CHECK(arma::approx_equal(
        k.simulate(5, 7, Xt, sdn, false, "vecchia"), k.simulateVecchia(5, 7, Xt, 0, false, sdn), "absdiff", 0.0));
    CHECK_THROWS_AS(k.simulate(5, 7, Xt, sdn, false, "nystrom"), std::invalid_argument);
  }
}

TEST_CASE("Vecchia noise-model fits support save/load and update", "[vecchia][kriging][noise]") {
  arma::mat X;
  arma::vec y, noise;
  make_noisy_data(120, X, y, noise);
  arma::mat Xu;
  arma::vec yu, noise_u;
  make_noisy_data(20, Xu, yu, noise_u, 789);
  arma::mat Xt;
  arma::vec yt;
  make_data(30, Xt, yt, 456);

  SECTION("Nugget") {
    Kriging k = fit_nugget(y, X, "vecchia(15)");
    const std::string fname = "vecchia_nugget_saved.json";
    k.save(fname);
    Kriging kl = Kriging::load(fname);
    CHECK(kl.is_vecchia_light());
    CHECK(kl.noise_model() == Kriging::NoiseModel::Nugget);
    CHECK(kl.vecchia_neighbors() == 15);
    CHECK(kl.nugget() == k.nugget());
    CHECK(kl.alpha() == k.alpha());
    auto [m_l, s_l] = kl.predictVecchia(Xt, true);
    auto [m_k, s_k] = k.predictVecchia(Xt, true);
    CHECK(arma::abs(m_l - m_k).max() < 1e-12);
    CHECK(arma::abs(s_l - s_k).max() < 1e-12);
    CHECK(std::abs(kl.logLikelihood("vecchia") - k.logLikelihood("vecchia")) < 1e-9);

    // update without refit: same theta / alpha, re-profiled sigma2 and nugget
    const arma::vec theta0 = k.theta();
    const double alpha0 = k.alpha();
    k.update(yu, Xu, false);
    CHECK(k.is_vecchia_light());
    CHECK(k.X().n_rows == 140);
    CHECK(arma::approx_equal(k.theta(), theta0, "absdiff", 0.0));
    CHECK(k.alpha() == alpha0);
    CHECK(k.alpha() == Approx(k.sigma2() / (k.sigma2() + k.nugget())));
    // update with refit: equals a fresh Vecchia fit's likelihood level
    k.update(yu, Xu, true);
    CHECK(k.X().n_rows == 160);
    CHECK(k.is_vecchia_light());
    CHECK(k.alpha() == Approx(k.sigma2() / (k.sigma2() + k.nugget())));
    CHECK(k.logLikelihood("vecchia")
          >= std::get<0>(k.logLikelihoodVecchiaFun(arma::join_cols(theta0, arma::vec{alpha0}), false)) - 1e-6);
    auto [m_u, s_u] = k.predictVecchia(Xt, true);
    CHECK(m_u.is_finite());
  }
  SECTION("Heterogeneous") {
    Kriging k = fit_hetero(y, noise, X, "vecchia(15)");
    const std::string fname = "vecchia_hetero_saved.json";
    k.save(fname);
    Kriging kl = Kriging::load(fname);
    CHECK(kl.is_vecchia_light());
    CHECK(kl.noise_model() == Kriging::NoiseModel::Heterogeneous);
    CHECK(arma::approx_equal(kl.noise(), k.noise(), "absdiff", 0.0));
    auto [m_l, s_l] = kl.predictVecchia(Xt, true);
    auto [m_k, s_k] = k.predictVecchia(Xt, true);
    CHECK(arma::abs(m_l - m_k).max() < 1e-12);
    CHECK(std::abs(kl.logLikelihood("vecchia") - k.logLikelihood("vecchia")) < 1e-9);

    // update needs the noise of the new data
    CHECK_THROWS_AS(k.update(yu, Xu, false), std::runtime_error);
    k.update(yu, noise_u, Xu, false);
    CHECK(k.is_vecchia_light());
    CHECK(k.X().n_rows == 140);
    CHECK(k.noise().n_elem == 140);
    k.update(yu, noise_u, Xu, true);
    CHECK(k.is_vecchia_light());
    CHECK(k.X().n_rows == 160);
    auto [m_u, s_u] = k.predictVecchia(Xt, true);
    CHECK(m_u.is_finite());
  }
}
TEST_CASE("Vecchia benchmark", "[.benchmark]") {
  arma::mat X;
  arma::vec y;
  make_data(400, X, y);
  const arma::vec theta{0.3, 0.3};

  Kriging k_eval(y, X, "gauss", Trend::RegressionModel::Constant, false, "BFGS", "LL", {}, "vecchia(30)");

  BENCHMARK("Kriging::fit LL vecchia(30) n=400") {
    return Kriging(y, X, "gauss", Trend::RegressionModel::Constant, false, "BFGS", "LL", {}, "vecchia(30)");
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
