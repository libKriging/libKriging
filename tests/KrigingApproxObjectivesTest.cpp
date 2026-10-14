// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
// clang-format on

#include <cstdio>
#include <functional>

// LOO and LMP objectives with the approximate methods "vecchia(m)" and
// "nystrom(k)" (docs/math/LOO.md, docs/math/LMP.md).

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

// model fitted at a fixed theta (no optimization): only its data, trend and
// approximate structure (Vecchia sets / Nystrom landmarks) are used below
static Kriging fit_at(const arma::vec& y,
                      const arma::mat& X,
                      const std::string& objective,
                      const std::string& method,
                      Trend::RegressionModel trend = Trend::RegressionModel::Constant) {
  Kriging::Parameters params;
  params.theta = arma::mat{{0.3, 0.3}};
  params.is_theta_estim = false;
  return Kriging(y, X, "matern5_2", trend, false, "none", objective, params, method);
}

using Fun = std::function<std::tuple<double, arma::vec>(const arma::vec&, bool)>;

static void check_gradient(const Fun& f, const arma::vec& theta, double h, double rtol) {
  auto [f0, grad] = f(theta, true);
  REQUIRE(grad.n_elem == theta.n_elem);
  for (arma::uword j = 0; j < theta.n_elem; ++j) {
    arma::vec tp = theta;
    arma::vec tm = theta;
    tp(j) += h;
    tm(j) -= h;
    const double fd = (std::get<0>(f(tp, false)) - std::get<0>(f(tm, false))) / (2 * h);
    INFO("theta=" << theta.t() << " dim " << j << ": analytic=" << grad(j) << " fd=" << fd << " f=" << f0);
    // relative to the gradient scale: a near-zero component is compared in absolute terms
    CHECK(std::abs(grad(j) - fd) < rtol * std::max(std::abs(fd), 0.1 * arma::abs(grad).max()));
  }
}

TEST_CASE("every objective accepts the approximate methods", "[approx][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(60, X, y);
  for (const std::string objective : {"LOO", "LMP"}) {
    for (const std::string method : {"vecchia(10)", "nystrom(15)"}) {
      const auto om = Kriging::fit_objective_method(objective, method);
      CHECK(om.first == objective);
      CHECK(om.second == method);
      Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", objective, {}, method);
      CHECK(k.objective() == objective);
      CHECK(k.method() == method);
      CHECK(k.is_vecchia_light() == (method[0] == 'v'));
      CHECK(k.is_nystrom_light() == (method[0] == 'n'));
      CHECK(k.T().is_empty());
    }
  }
  // still not implemented with a nugget / noise channel
  Kriging knug("matern5_2", Kriging::NoiseModel::Nugget);
  CHECK_THROWS_AS(knug.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LMP", {}, "vecchia(10)"),
                  std::invalid_argument);
  CHECK_THROWS_AS(knug.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LMP", {}, "nystrom(10)"),
                  std::invalid_argument);
}

TEST_CASE("Vecchia(n-1) LOO and LMP reproduce the exact ones", "[approx][vecchia][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(60, X, y);
  for (const auto trend : {Trend::RegressionModel::Constant, Trend::RegressionModel::Linear}) {
    Kriging k = fit_at(y, X, "LOO", "vecchia(59)", trend);
    for (const arma::vec theta : {arma::vec{0.2, 0.2}, arma::vec{0.4, 0.3}}) {
      const double loo_v = std::get<0>(k.leaveOneOutVecchiaFun(theta, false));
      const double loo = std::get<0>(k.leaveOneOutFun(theta, false, false));
      INFO("theta=" << theta.t() << ": LOO vecchia(n-1) = " << loo_v << " vs exact " << loo);
      CHECK(std::abs(loo_v - loo) < 1e-4 * std::abs(loo) + 1e-10);
      const double lmp_v = std::get<0>(k.logMargPostVecchiaFun(theta, false));
      const double lmp = std::get<0>(k.logMargPostFun(theta, false, false));
      INFO("LMP vecchia(n-1) = " << lmp_v << " vs exact " << lmp);
      CHECK(std::abs(lmp_v - lmp) < 1e-4 * std::abs(lmp) + 1e-6);

      auto [yhat_v, sd_v] = k.leaveOneOutVec(theta, "vecchia");
      auto [yhat, sd] = k.leaveOneOutVec(theta);
      CHECK(arma::abs(yhat_v - yhat).max() < 1e-4 * arma::abs(y).max());
      CHECK(arma::abs(sd_v - sd).max() < 1e-3 * sd.max());
    }
  }
}

TEST_CASE("Nystrom(n) LOO and LMP reproduce the exact ones", "[approx][nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(60, X, y);
  for (const auto trend : {Trend::RegressionModel::Constant, Trend::RegressionModel::Linear}) {
    Kriging k = fit_at(y, X, "LMP", "nystrom(60)", trend);
    for (const arma::vec theta : {arma::vec{0.2, 0.2}, arma::vec{0.4, 0.3}}) {
      const double loo_n = std::get<0>(k.leaveOneOutNystromFun(theta, false));
      const double loo = std::get<0>(k.leaveOneOutFun(theta, false, false));
      INFO("theta=" << theta.t() << ": LOO nystrom(n) = " << loo_n << " vs exact " << loo);
      CHECK(std::abs(loo_n - loo) < 1e-3 * std::abs(loo) + 1e-10);
      const double lmp_n = std::get<0>(k.logMargPostNystromFun(theta, false));
      const double lmp = std::get<0>(k.logMargPostFun(theta, false, false));
      INFO("LMP nystrom(n) = " << lmp_n << " vs exact " << lmp);
      CHECK(std::abs(lmp_n - lmp) < 1e-3 * std::abs(lmp) + 1e-3);
    }
  }
}

TEST_CASE("predictNystrom applies the LMP variance factor like predict", "[approx][nystrom][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(60, X, y);
  arma::mat Xn;
  arma::vec yn;
  make_data(10, Xn, yn, 7);
  // same theta and sigma2: only the objective differs
  Kriging::Parameters params;
  params.theta = arma::mat{{0.3, 0.3}};
  params.is_theta_estim = false;
  params.sigma2 = 0.5;
  params.is_sigma2_estim = false;
  Kriging k_ll(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "none", "LL", params, "nystrom(30)");
  Kriging k_lmp(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "none", "LMP", params, "nystrom(30)");
  REQUIRE(k_ll.sigma2() == Approx(k_lmp.sigma2()));
  const auto [m_ll, s_ll] = k_ll.predictNystrom(Xn, true);
  const auto [m_lmp, s_lmp] = k_lmp.predictNystrom(Xn, true);
  const double n = 60.0, p = 1.0;
  CHECK(arma::abs(m_lmp - m_ll).max() < 1e-12);
  CHECK(arma::abs(s_lmp / s_ll - std::sqrt((n - p) / (n - p - 2.0))).max() < 1e-10);
}

TEST_CASE("approximate LOO / LMP analytic gradients match finite differences", "[approx][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(100, X, y);
  for (const auto trend : {Trend::RegressionModel::Constant, Trend::RegressionModel::Linear}) {
    Kriging kv = fit_at(y, X, "LOO", "vecchia(15)", trend);
    Kriging kn = fit_at(y, X, "LMP", "nystrom(30)", trend);
    for (const arma::vec theta : {arma::vec{0.3, 0.4}, arma::vec{0.5, 0.25}}) {
      INFO("LOO vecchia");
      check_gradient([&](const arma::vec& t, bool g) { return kv.leaveOneOutVecchiaFun(t, g); }, theta, 1e-6, 1e-3);
      INFO("LMP vecchia");
      check_gradient([&](const arma::vec& t, bool g) { return kv.logMargPostVecchiaFun(t, g); }, theta, 1e-6, 1e-3);
      INFO("LOO nystrom");
      check_gradient([&](const arma::vec& t, bool g) { return kn.leaveOneOutNystromFun(t, g); }, theta, 1e-3, 5e-3);
      INFO("LMP nystrom");
      check_gradient([&](const arma::vec& t, bool g) { return kn.logMargPostNystromFun(t, g); }, theta, 1e-3, 5e-3);
    }
  }

  // known beta: LOO under the approximate law without the trend projection
  Kriging::Parameters known;
  known.theta = arma::mat{{0.3, 0.3}};
  known.is_theta_estim = false;
  known.beta = arma::vec{0.5};
  known.is_beta_estim = false;
  Kriging kvb(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "none", "LOO", known, "vecchia(15)");
  Kriging knb(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "none", "LOO", known, "nystrom(30)");
  CHECK(kvb.beta()(0) == 0.5);
  const arma::vec theta{0.4, 0.3};
  check_gradient([&](const arma::vec& t, bool g) { return kvb.leaveOneOutVecchiaFun(t, g); }, theta, 1e-6, 1e-3);
  check_gradient([&](const arma::vec& t, bool g) { return knb.leaveOneOutNystromFun(t, g); }, theta, 1e-3, 5e-3);
}

TEST_CASE("approximate LOO / LMP fits are consistent with the exact fits", "[approx][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(150, X, y);
  for (const std::string objective : {"LOO", "LMP"}) {
    Kriging ke(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", objective);
    for (const std::string method : {"vecchia(30)", "nystrom(80)"}) {
      Kriging ka(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", objective, {}, method);
      INFO(objective << " / " << method << ": theta = " << ka.theta().t() << " exact theta = " << ke.theta().t());
      // on this smooth function the LOO / LMP surfaces are flat in the
      // large-range direction (exact LOO theta ~ (3.6, 1.5)): compare the
      // fitted models by their predictive accuracy rather than theta
      CHECK(arma::abs(arma::log(ka.theta()) - arma::log(ke.theta())).max() < 1.5);
      // the optimum of the approximate criterion is no worse than the exact optimum, for that criterion
      const std::string am = method.substr(0, 7);
      Kriging::Parameters at_exact;
      at_exact.theta = arma::mat(ke.theta().t());
      at_exact.is_theta_estim = false;
      Kriging kae(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "none", objective, at_exact, method);
      if (objective == "LOO")
        CHECK(ka.leaveOneOut(am) <= kae.leaveOneOut(am) * (1 + 1e-6));
      else
        CHECK(ka.logMargPost(am) >= kae.logMargPost(am) - 1e-6 * std::abs(kae.logMargPost(am)));
      arma::mat Xt;
      arma::vec yt;
      make_data(200, Xt, yt, 456);
      auto [mt, st, ct, dmt, dst] = ka.predict(Xt, false, false, false, am);
      auto [me, se, ce, dme, dse] = ke.predict(Xt, false, false, false);
      const double rmse_a = std::sqrt(arma::mean(arma::square(mt - yt)));
      const double rmse_e = std::sqrt(arma::mean(arma::square(me - yt)));
      INFO("test RMSE approximate = " << rmse_a << " exact = " << rmse_e << " sd(y) = " << arma::stddev(yt));
      CHECK(rmse_a < 0.02 * arma::stddev(yt));
      // the committed criterion is the optimized one
      if (objective == "LOO") {
        CHECK(ka.leaveOneOut(method) == ka.leaveOneOut(method.substr(0, 7)));  // fitted method string accepted
        CHECK(ka.leaveOneOut(method.substr(0, 7)) < 2.0 * ke.leaveOneOut() + 1e-6);
      } else {
        CHECK(std::isfinite(ka.logMargPost(method.substr(0, 7))));
      }
      // sigma2 / beta conventions: LL-profiled under the approximation, and
      // SSE / (n - p) for LMP (like the exact LMP commit)
      const double sigma2_ll = [&] {
        Kriging::Parameters at;
        at.theta = arma::mat(ka.theta().t());
        at.is_theta_estim = false;
        return Kriging(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "none", "LL", at, method).sigma2();
      }();
      const double n = 150.0;
      const double expected = (objective == "LMP") ? sigma2_ll * n / (n - 1) : sigma2_ll;
      CHECK(std::abs(ka.sigma2() - expected) < 1e-10 * expected);
      // the approximate predictor interpolates the data
      auto [m, s, c, dm, ds] = ka.predict(X.rows(0, 19), true, false, false, method.substr(0, 7));
      CHECK(arma::abs(m - y.head(20)).max() < 5e-2);
    }
  }
}

TEST_CASE("approximate LOO / LMP fits survive save/load and update", "[approx][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(120, X, y);
  arma::mat Xu;
  arma::vec yu;
  make_data(10, Xu, yu, 789);
  for (const std::string objective : {"LOO", "LMP"}) {
    for (const std::string method : {"vecchia(20)", "nystrom(40)"}) {
      const std::string m = method.substr(0, 7);
      Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", objective, {}, method);
      const std::string fname = "approx_objective_saved.json";
      k.save(fname);
      Kriging kl = Kriging::load(fname);
      std::remove(fname.c_str());
      CHECK(kl.objective() == objective);
      CHECK(kl.method() == method);
      CHECK(std::abs(kl.leaveOneOut(m) - k.leaveOneOut(m)) < 1e-10 * std::abs(k.leaveOneOut(m)) + 1e-14);
      CHECK(std::abs(kl.logMargPost(m) - k.logMargPost(m)) < 1e-8 * std::abs(k.logMargPost(m)));

      kl.update(yu, Xu, true);
      CHECK(kl.objective() == objective);
      CHECK(kl.method() == method);
      CHECK(kl.X().n_rows == 130);
      CHECK(kl.T().is_empty());
      auto [mu, su, cu, du, eu] = kl.predict(Xu, true, false, false, m);
      CHECK(arma::abs(mu - yu).max() < 5e-2);  // (near-)interpolates the added data
      kl.update(yu + 0.0, Xu + 1e-3, false);
      CHECK(kl.X().n_rows == 140);
      CHECK(std::isfinite(kl.leaveOneOut(m)));
    }
  }
}

TEST_CASE("leaveOneOut / logMargPost method argument checks", "[approx][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(60, X, y);
  Kriging kv = fit_at(y, X, "LOO", "vecchia(10)");
  Kriging ke = fit_at(y, X, "LOO", "exact");
  CHECK(ke.leaveOneOut("exact") == ke.leaveOneOut());
  CHECK(ke.logMargPost("exact") == ke.logMargPost());
  CHECK(kv.leaveOneOut("vecchia") == std::get<0>(kv.leaveOneOutVecchiaFun(kv.theta(), false)));
  CHECK(kv.logMargPost("vecchia(10)") == std::get<0>(kv.logMargPostVecchiaFun(kv.theta(), false)));
  CHECK_THROWS_AS(kv.leaveOneOut("vecchia(7)"), std::invalid_argument);  // not the fitted m
  CHECK_THROWS_AS(kv.leaveOneOut("nystrom"), std::runtime_error);        // no Nystrom fit
  CHECK_THROWS_AS(ke.logMargPost("vecchia"), std::runtime_error);        // no Vecchia fit
  CHECK_THROWS_AS(kv.leaveOneOut("Vecchia"), std::invalid_argument);     // unknown method
  CHECK_THROWS_AS(kv.leaveOneOutVec(kv.theta(), "nystrom"), std::runtime_error);
  // the exact criteria remain available on an approximate fit (dense, O(n^3))
  CHECK(std::isfinite(kv.leaveOneOut("exact")));
}

TEST_CASE("approximate LOO / LMP evaluations reject noise models", "[approx][kriging][noise]") {
  // fit raises for objective LOO / LMP with an approximate method and a noise
  // model; the public approximate evaluations must not silently return the
  // noise-free criterion on such a model (fitted with objective "LL")
  arma::mat X;
  arma::vec y;
  make_data(60, X, y);
  const arma::vec noise(y.n_elem, arma::fill::value(0.01));
  Kriging::Parameters params;
  params.theta = arma::mat{{0.3, 0.3}};
  params.is_theta_estim = false;
  for (const std::string method : {"vecchia(10)", "nystrom(15)"}) {
    const bool vecchia = method[0] == 'v';
    const std::string name = vecchia ? "vecchia" : "nystrom";
    for (const auto nm : {Kriging::NoiseModel::Nugget, Kriging::NoiseModel::Heterogeneous}) {
      Kriging k("matern5_2", nm);
      if (nm == Kriging::NoiseModel::Nugget)
        k.fit(y, X, Trend::RegressionModel::Constant, false, "none", "LL", params, method);
      else
        k.fit(y, noise, X, Trend::RegressionModel::Constant, false, "none", "LL", params, method);
      CAPTURE(method, static_cast<int>(nm));
      const arma::vec theta = k.theta();
      CHECK_THROWS_AS(k.leaveOneOut(name), std::invalid_argument);
      CHECK_THROWS_AS(k.logMargPost(name), std::invalid_argument);
      CHECK_THROWS_AS(k.leaveOneOutVec(theta, name), std::invalid_argument);
      if (vecchia) {
        CHECK_THROWS_AS(k.leaveOneOutVecchiaFun(theta, false), std::invalid_argument);
        CHECK_THROWS_AS(k.logMargPostVecchiaFun(theta, true), std::invalid_argument);
      } else {
        CHECK_THROWS_AS(k.leaveOneOutNystromFun(theta, false), std::invalid_argument);
        CHECK_THROWS_AS(k.logMargPostNystromFun(theta, true), std::invalid_argument);
      }
      try {
        k.leaveOneOut(name);
      } catch (const std::invalid_argument& e) {
        CHECK(std::string(e.what()).find("not implemented for Nugget/Heterogeneous noise models") != std::string::npos);
      }
      // the approximate LL stays available with a noise model
      CHECK(std::isfinite(k.logLikelihood(name)));
    }
  }
}
