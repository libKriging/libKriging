// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/MLPKriging.hpp"
#include "libKriging/WarpKriging.hpp"
// clang-format on

using namespace libKriging;

// with_noise of WarpKriging / MLPKriging simulate holds noise VARIANCES (raw y
// scale, like fit's noise), as Kriging::simulate: with the same seed the latent
// draws are identical, so the difference is the added observation noise only;
// its empirical variance must match the given variances.
namespace {

const arma::uword n_test = 6;
const int n_sim = 20000;

void check_noise_var(const arma::mat& noisy, const arma::mat& latent, const arma::vec& expected) {
  REQUIRE(noisy.n_rows == expected.n_elem);
  REQUIRE(latent.n_rows == expected.n_elem);
  const arma::mat diff = noisy - latent;
  const arma::vec emp_var = arma::var(diff, 0, 1);
  for (arma::uword i = 0; i < expected.n_elem; ++i) {
    INFO("Point " << i << ": empirical noise variance " << emp_var(i) << " vs " << expected(i));
    CHECK(std::abs(emp_var(i) / expected(i) - 1.0) < 0.05);  // ~ 5 sd of the variance estimator
  }
}

template <typename Model, typename UpdateSim>
void check_model(Model& model, const arma::mat& X_test, const arma::vec& var_n, UpdateSim update_sim) {
  const arma::vec zero(n_test, arma::fill::zeros);
  SECTION("simulate, per-point variances") {
    check_noise_var(
        model.simulate(n_sim, 7, X_test, false, var_n), model.simulate(n_sim, 7, X_test, false, zero), var_n);
  }
  SECTION("simulate, scalar variance") {
    const arma::vec v1{0.3};
    check_noise_var(model.simulate(n_sim, 8, X_test, false, v1),
                    model.simulate(n_sim, 8, X_test, false, arma::vec{}),
                    arma::vec(n_test).fill(0.3));
  }
  SECTION("update_simulate keeps the with_noise variances of simulate") {
    model.simulate(n_sim, 10, X_test, true, var_n);
    const arma::mat noisy = update_sim(model);
    model.simulate(n_sim, 10, X_test, true, zero);
    const arma::mat latent = update_sim(model);
    check_noise_var(noisy, latent, var_n);
  }
  SECTION("negative / NaN variances are rejected") {
    CHECK_THROWS_AS(model.simulate(10, 1, X_test, false, arma::vec(-var_n)), std::invalid_argument);
    CHECK_THROWS_AS(model.simulate(10, 1, X_test, false, arma::vec{-0.1}), std::invalid_argument);
    arma::vec nan_var = var_n;
    nan_var(0) = arma::datum::nan;
    CHECK_THROWS_AS(model.simulate(10, 1, X_test, false, nan_var), std::invalid_argument);
  }
}

}  // namespace

TEST_CASE("WarpKrigingSimulateNoiseTest - with_noise holds noise variances (heterogeneous fit)",
          "[simulate][warpkriging]") {
  arma::arma_rng::set_seed(321);
  const arma::uword n = 20;
  arma::mat X(n, 1, arma::fill::randu);
  const arma::vec noise = 0.01 + 0.02 * arma::randu<arma::vec>(n);
  const arma::vec y = arma::sin(6.0 * X.col(0)) + arma::sqrt(noise) % arma::randn<arma::vec>(n);

  WarpKriging wk({"none"}, "matern5_2");
  WarpKriging::Parameters params;
  params.noise = noise;
  wk.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);

  arma::mat X_test(n_test, 1, arma::fill::randu);
  const arma::vec var_n = arma::linspace(0.05, 0.5, n_test);
  arma::mat X_u(3, 1, arma::fill::randu);
  const arma::vec y_u = arma::sin(6.0 * X_u.col(0));
  const arma::vec noise_u(3, arma::fill::value(0.02));
  check_model(wk, X_test, var_n, [&](WarpKriging& m) { return m.update_simulate(y_u, X_u, noise_u); });
}

TEST_CASE("WarpKrigingSimulateNoiseTest - with_noise holds noise variances (kumaraswamy, no fit noise)",
          "[simulate][warpkriging]") {
  arma::arma_rng::set_seed(123);
  const arma::uword n = 15;
  arma::mat X(n, 1, arma::fill::randu);
  const arma::vec y = arma::sin(6.0 * X.col(0)) + 0.5 * X.col(0);

  WarpKriging wk({"kumaraswamy"}, "gauss");
  wk.fit(y, X, Trend::RegressionModel::Constant, false, "Adam", "LL", {{"max_iter_adam", "100"}});

  arma::mat X_test(n_test, 1, arma::fill::randu);
  const arma::vec var_n = arma::linspace(0.02, 0.2, n_test);
  arma::mat X_u(2, 1, arma::fill::randu);
  const arma::vec y_u = arma::sin(6.0 * X_u.col(0)) + 0.5 * X_u.col(0);
  check_model(wk, X_test, var_n, [&](WarpKriging& m) { return m.update_simulate(y_u, X_u); });
}

TEST_CASE("MLPKrigingSimulateNoiseTest - with_noise holds noise variances", "[simulate][mlpkriging]") {
  arma::arma_rng::set_seed(123);
  const arma::uword n = 15;
  arma::mat X(n, 1, arma::fill::randu);
  const arma::vec y = arma::sin(6.0 * X.col(0)) + 0.5 * X.col(0);

  MLPKriging mk({8}, 2, "tanh", "gauss");
  mk.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS+Adam", "LL", {{"max_iter_adam", "100"}});

  arma::mat X_test(n_test, 1, arma::fill::randu);
  const arma::vec var_n = arma::linspace(0.02, 0.2, n_test);
  arma::mat X_u(2, 1, arma::fill::randu);
  const arma::vec y_u = arma::sin(6.0 * X_u.col(0)) + 0.5 * X_u.col(0);
  check_model(mk, X_test, var_n, [&](MLPKriging& m) { return m.update_simulate(y_u, X_u); });
}
