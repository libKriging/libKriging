// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
#include "libKriging/WarpKriging.hpp"
#include "libKriging/MLPKriging.hpp"
// clang-format on

// The state stored by simulate(..., will_update=true) (and the update-points
// cache of update_simulate) is built for the model of that simulate: fit() and
// update() invalidate it, so a later update_simulate() raises (as without any
// simulate) instead of mixing matrices built for the old model / old n with
// the new one. A new simulate(..., will_update=true) works again.

using namespace libKriging;
using Catch::Matchers::Contains;

namespace {

const char* const no_sim = "No previous simulation data available";

struct Data {
  arma::mat X, X_u, X_n, X2, X_v;
  arma::vec y, y_u, y2, y_v;
};

Data make_data(arma::uword d) {
  arma::arma_rng::set_seed(4242);
  auto f = [](const arma::mat& X) {
    arma::vec y = arma::sin(3.0 * X.col(0));
    for (arma::uword j = 1; j < X.n_cols; ++j)
      y += arma::cos(5.0 * X.col(j));
    return y;
  };
  Data D;
  D.X = arma::mat(12, d, arma::fill::randu);
  D.y = f(D.X);
  D.X_u = arma::mat(3, d, arma::fill::randu);
  D.y_u = f(D.X_u);
  D.X_n = arma::mat(5, d, arma::fill::randu);
  D.X2 = arma::mat(12, d, arma::fill::randu);  // same n as X: a refit on other data
  D.y2 = f(D.X2) + 1.0;
  // update points for an update_simulate after update(y_u, X_u): X_u is then part
  // of the observations (conditioning on it again is degenerate, platform-dependent)
  D.X_v = arma::mat(3, d, arma::fill::randu);
  D.y_v = f(D.X_v);
  return D;
}

Kriging::Parameters fixed_params(arma::uword d) {
  Kriging::Parameters p;
  p.sigma2 = 1.0;
  p.is_sigma2_estim = false;
  p.theta = arma::mat(arma::vec(d).fill(0.4));
  p.is_theta_estim = false;
  return p;
}

}  // namespace

TEST_CASE("SimulateStateResetTest - Kriging update() invalidates the last simulate", "[update_simulate][kriging]") {
  const arma::uword d = 2;
  const Data D = make_data(d);
  const auto params = fixed_params(d);

  for (bool refit : {false, true}) {
    for (bool warm_cache : {false, true}) {
      INFO("refit=" << refit << " warm_cache=" << warm_cache);
      Kriging k("gauss");
      k.fit(D.y, D.X, Trend::RegressionModel::Constant, false, refit ? "BFGS" : "none", "LL", params);
      k.simulate(20, 1, D.X_n, true);
      if (warm_cache)  // fills the update-points cache for X_u
        k.update_simulate(D.y_u, D.X_u);
      k.update(D.y_u, D.X_u, refit);
      REQUIRE_THROWS_WITH(k.update_simulate(D.y_u, D.X_u), Contains(no_sim));
      // a new simulate on the updated model works again
      k.simulate(20, 1, D.X_n, true);
      REQUIRE(k.update_simulate(D.y_v, D.X_v).n_rows == D.X_n.n_rows);
    }
  }
}

TEST_CASE("SimulateStateResetTest - Kriging fit() invalidates the last simulate", "[update_simulate][kriging]") {
  const arma::uword d = 2;
  const Data D = make_data(d);
  const auto params = fixed_params(d);

  Kriging k("gauss");
  k.fit(D.y, D.X, Trend::RegressionModel::Constant, false, "none", "LL", params);
  k.simulate(20, 1, D.X_n, true);
  k.update_simulate(D.y_u, D.X_u);
  k.fit(D.y2, D.X2, Trend::RegressionModel::Constant, false, "none", "LL", params);
  REQUIRE_THROWS_WITH(k.update_simulate(D.y_u, D.X_u), Contains(no_sim));

  // and the result after a new simulate equals the one of a fresh model
  k.simulate(20, 1, D.X_n, true);
  Kriging fresh("gauss");
  fresh.fit(D.y2, D.X2, Trend::RegressionModel::Constant, false, "none", "LL", params);
  fresh.simulate(20, 1, D.X_n, true);
  CHECK(arma::approx_equal(k.update_simulate(D.y_u, D.X_u), fresh.update_simulate(D.y_u, D.X_u), "absdiff", 1e-10));
}

TEST_CASE("SimulateStateResetTest - Kriging noise models and approximate methods", "[update_simulate][kriging]") {
  const arma::uword d = 2;
  const Data D = make_data(d);
  const auto params = fixed_params(d);

  SECTION("Nugget: update, refit or not") {
    for (bool refit : {false, true}) {
      Kriging::Parameters p = params;
      p.nugget = 0.01;
      p.is_nugget_estim = false;
      Kriging k("gauss", Kriging::NoiseModel::Nugget);
      k.fit(D.y, D.X, Trend::RegressionModel::Constant, false, "BFGS", "LL", p);
      k.simulate(20, 1, D.X_n, true, true);
      k.update_simulate(D.y_u, D.X_u);
      k.update(D.y_u, D.X_u, refit);
      REQUIRE_THROWS_WITH(k.update_simulate(D.y_u, D.X_u), Contains(no_sim));
    }
  }

  SECTION("Heterogeneous: update(y_u, noise_u, X_u), refit or not") {
    for (bool refit : {false, true}) {
      Kriging k("gauss", Kriging::NoiseModel::Heterogeneous);
      k.fit(D.y, arma::vec(D.y.n_elem).fill(0.01), D.X, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);
      const arma::vec noise_u(D.y_u.n_elem, arma::fill::value(0.02));
      k.simulate(20, 1, D.X_n, arma::vec{0.01}, true);
      k.update_simulate(D.y_u, noise_u, D.X_u);
      k.update(D.y_u, noise_u, D.X_u, refit);
      REQUIRE_THROWS_WITH(k.update_simulate(D.y_u, noise_u, D.X_u), Contains(no_sim));
    }
  }

  SECTION("a rejected update leaves the simulate state unchanged") {
    Kriging k("gauss", Kriging::NoiseModel::Heterogeneous);
    k.fit(D.y, arma::vec(D.y.n_elem).fill(0.01), D.X, Trend::RegressionModel::Constant, false, "none", "LL", params);
    const arma::vec noise_u(D.y_u.n_elem, arma::fill::value(0.02));
    k.simulate(20, 1, D.X_n, arma::vec(), true);
    const arma::mat before = k.update_simulate(D.y_u, noise_u, D.X_u);
    REQUIRE_THROWS_AS(k.update(D.y_u, D.X_u), std::invalid_argument);                       // noise_u missing
    REQUIRE_THROWS(k.update(D.y_u, arma::vec(D.y_u.n_elem + 1, arma::fill::ones), D.X_u));  // wrong length
    CHECK(arma::approx_equal(k.update_simulate(D.y_u, noise_u, D.X_u), before, "absdiff", 1e-12));
  }

  SECTION("vecchia / nystrom fits and updates clear an earlier exact simulate") {
    for (const char* method : {"vecchia(5)", "nystrom(6)"}) {
      INFO(method);
      Kriging k("gauss");
      k.fit(D.y, D.X, Trend::RegressionModel::Constant, false, "none", "LL", params);
      k.simulate(20, 1, D.X_n, true);
      k.fit(D.y, D.X, Trend::RegressionModel::Constant, false, "none", "LL", params, method);
      k.fit(D.y, D.X, Trend::RegressionModel::Constant, false, "none", "LL", params);  // back to exact
      REQUIRE_THROWS_WITH(k.update_simulate(D.y_u, D.X_u), Contains(no_sim));
    }
  }

  SECTION("simulate without with_noise drops the with_noise of an earlier simulate") {
    const arma::vec noise(D.y.n_elem, arma::fill::value(0.01));
    const arma::vec noise_u(D.y_u.n_elem, arma::fill::value(0.02));
    Kriging k("gauss", Kriging::NoiseModel::Heterogeneous);
    k.fit(D.y, noise, D.X, Trend::RegressionModel::Constant, false, "none", "LL", params);
    k.simulate(20, 1, D.X_n, arma::vec{0.5}, true);
    k.simulate(20, 1, D.X_n, true);  // latent: no with_noise to re-add
    Kriging fresh("gauss", Kriging::NoiseModel::Heterogeneous);
    fresh.fit(D.y, noise, D.X, Trend::RegressionModel::Constant, false, "none", "LL", params);
    fresh.simulate(20, 1, D.X_n, true);
    CHECK(arma::approx_equal(
        k.update_simulate(D.y_u, noise_u, D.X_u), fresh.update_simulate(D.y_u, noise_u, D.X_u), "absdiff", 1e-10));
  }
}

TEST_CASE("SimulateStateResetTest - WarpKriging / MLPKriging fit() and update() invalidate the last simulate",
          "[update_simulate][warpkriging]") {
  const arma::uword d = 1;
  const Data D = make_data(d);
  const std::map<std::string, std::string> tuning{{"max_iter_adam", "20"}, {"max_iter_bfgs", "20"}};

  SECTION("WarpKriging update, refit or not") {
    for (bool refit : {false, true}) {
      for (bool warm_cache : {false, true}) {
        INFO("refit=" << refit << " warm_cache=" << warm_cache);
        WarpKriging wk({"affine"}, "gauss");
        wk.fit(D.y, D.X, Trend::RegressionModel::Constant, false, "BFGS+Adam", "LL", tuning);
        wk.simulate(20, 1, D.X_n, true);
        if (warm_cache)
          wk.update_simulate(D.y_u, D.X_u);
        wk.update(D.y_u, D.X_u, refit);
        REQUIRE_THROWS_WITH(wk.update_simulate(D.y_u, D.X_u), Contains(no_sim));
        wk.simulate(20, 1, D.X_n, true);
        REQUIRE(wk.update_simulate(D.y_v, D.X_v).n_rows == D.X_n.n_rows);
      }
    }
  }

  SECTION("WarpKriging fit") {
    WarpKriging wk({"affine"}, "gauss");
    wk.fit(D.y, D.X, Trend::RegressionModel::Constant, false, "BFGS+Adam", "LL", tuning);
    wk.simulate(20, 1, D.X_n, true);
    wk.update_simulate(D.y_u, D.X_u);
    wk.fit(D.y2, D.X2, Trend::RegressionModel::Constant, false, "BFGS+Adam", "LL", tuning);
    REQUIRE_THROWS_WITH(wk.update_simulate(D.y_u, D.X_u), Contains(no_sim));
  }

  SECTION("WarpKriging with noise: a rejected update leaves the simulate state unchanged") {
    WarpKriging::Parameters p;
    p.noise = arma::vec(D.y.n_elem).fill(1e-3);
    WarpKriging wk({"affine"}, "gauss");
    wk.fit(D.y, D.X, Trend::RegressionModel::Constant, false, "BFGS+Adam", "LL", p, tuning);
    const arma::vec noise_u(D.y_u.n_elem, arma::fill::value(1e-3));
    wk.simulate(20, 1, D.X_n, true);
    const arma::mat before = wk.update_simulate(D.y_u, D.X_u, noise_u);
    REQUIRE_THROWS_AS(wk.update_simulate(D.y_u, D.X_u), std::invalid_argument);  // noise_u missing
    REQUIRE_THROWS_AS(wk.update(D.y_u, D.X_u, false), std::invalid_argument);    // noise_u missing
    CHECK(arma::approx_equal(wk.update_simulate(D.y_u, D.X_u, noise_u), before, "absdiff", 1e-12));
    wk.update(D.y_u, D.X_u, false, noise_u);
    REQUIRE_THROWS_WITH(wk.update_simulate(D.y_u, D.X_u, noise_u), Contains(no_sim));
  }

  SECTION("MLPKriging update and fit") {
    MLPKriging mk(std::vector<arma::uword>{4}, 1, "tanh", "gauss");
    mk.fit(D.y, D.X, Trend::RegressionModel::Constant, false, "BFGS+Adam", "LL", tuning);
    mk.simulate(20, 1, D.X_n, true);
    mk.update_simulate(D.y_u, D.X_u);
    mk.update(D.y_u, D.X_u, false);
    REQUIRE_THROWS_WITH(mk.update_simulate(D.y_u, D.X_u), Contains(no_sim));
    mk.simulate(20, 1, D.X_n, true);
    mk.fit(D.y2, D.X2, Trend::RegressionModel::Constant, false, "BFGS+Adam", "LL", tuning);
    REQUIRE_THROWS_WITH(mk.update_simulate(D.y_u, D.X_u), Contains(no_sim));
  }
}
