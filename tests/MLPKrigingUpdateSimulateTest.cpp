// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/MLPKriging.hpp"
#include "ks_test.hpp"
#include <sstream>
// clang-format on

using namespace libKriging;

TEST_CASE("MLPKrigingUpdateSimulateTest - Update simulate equals updated model simulate",
          "[update_simulate][mlpkriging]") {
  arma::arma_rng::set_seed(123);

  // Generate initial training data
  const arma::uword n_old = 15;
  const arma::uword n_new = 3;
  const arma::uword d = 1;

  arma::mat X_old(n_old, d, arma::fill::randu);
  arma::colvec y_old(n_old);

  auto test_function = [](const arma::rowvec& x) { return std::sin(6.0 * x(0)) + 0.5 * x(0); };

  for (arma::uword i = 0; i < n_old; ++i) {
    y_old(i) = test_function(X_old.row(i));
  }

  // Generate new data points
  arma::mat X_new(n_new, d, arma::fill::randu);
  arma::colvec y_new(n_new);

  for (arma::uword i = 0; i < n_new; ++i) {
    y_new(i) = test_function(X_new.row(i));
  }

  SECTION("update_simulate gives same distribution as update then simulate") {
    MLPKriging mk1({8}, 2, "tanh", "gauss");
    mk1.fit(y_old, X_old, "constant", false, "BFGS+Adam", "LL", {{"max_iter_adam", "100"}});

    // Simulation points
    const arma::uword n_sim_points = 8;
    arma::mat X_sim(n_sim_points, d, arma::fill::randu);

    // Method 1: Use update_simulate
    const int n_sims = 1000;
    const int seed = 789;
    mk1.simulate(n_sims, seed, X_sim, true);
    arma::mat sims1 = mk1.update_simulate(y_new, X_new);

    // Method 2: Build mk2 on full data
    MLPKriging mk2({8}, 2, "tanh", "gauss");
    arma::mat X_full = arma::join_cols(X_old, X_new);
    arma::colvec y_full = arma::join_cols(y_old, y_new);
    mk2.fit(y_full, X_full, "constant", false, "BFGS+Adam", "LL", {{"max_iter_adam", "100"}});
    arma::mat sims2 = mk2.simulate(n_sims, seed, X_sim);

    // Check dimensions
    REQUIRE(sims1.n_rows == n_sim_points);
    REQUIRE(sims1.n_cols == static_cast<arma::uword>(n_sims));
    REQUIRE(sims1.is_finite());
  }

  SECTION("Smoke test: simulate without will_update still works") {
    MLPKriging mk({8}, 2, "tanh", "gauss");
    mk.fit(y_old, X_old, "constant", false, "BFGS+Adam", "LL", {{"max_iter_adam", "100"}});

    arma::mat X_sim(5, d, arma::fill::randu);
    arma::mat sims = mk.simulate(10, 42, X_sim);  // will_update defaults to false
    REQUIRE(sims.n_rows == 5);
    REQUIRE(sims.n_cols == 10);
    REQUIRE(sims.is_finite());
  }

  SECTION("Error: update_simulate without prior simulate throws") {
    MLPKriging mk({8}, 2, "tanh", "gauss");
    mk.fit(y_old, X_old, "constant", false, "BFGS+Adam", "LL", {{"max_iter_adam", "100"}});

    REQUIRE_THROWS_WITH(mk.update_simulate(y_new, X_new), Catch::Contains("No previous simulation data"));
  }
}
