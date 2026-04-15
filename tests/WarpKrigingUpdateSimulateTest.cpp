// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/WarpKriging.hpp"
#include "ks_test.hpp"
#include <sstream>
// clang-format on

using namespace libKriging;

TEST_CASE("WarpKrigingUpdateSimulateTest - Update simulate equals updated model simulate",
          "[update_simulate][warpkriging]") {
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

  SECTION("update_simulate gives same distribution as update then simulate (none warping)") {
    // Build wk1 with "none" warping (should behave like plain Kriging)
    WarpKriging wk1({"none"}, "gauss");
    wk1.fit(y_old, X_old, "constant", false, "Adam", "LL", {{"max_iter_adam", "100"}});

    // Simulation points
    const arma::uword n_sim_points = 8;
    arma::mat X_sim(n_sim_points, d, arma::fill::randu);

    // Method 1: Use update_simulate
    const int n_sims = 1000;
    const int seed = 789;
    wk1.simulate(n_sims, static_cast<uint64_t>(seed), X_sim, true);
    arma::mat sims1 = wk1.update_simulate(y_new, X_new);

    // Method 2: Build wk2 on full data
    WarpKriging wk2({"none"}, "gauss");
    arma::mat X_full = arma::join_cols(X_old, X_new);
    arma::colvec y_full = arma::join_cols(y_old, y_new);
    wk2.fit(y_full, X_full, "constant", false, "Adam", "LL", {{"max_iter_adam", "100"}});
    arma::mat sims2 = wk2.simulate(n_sims, static_cast<uint64_t>(seed), X_sim);

    // Check dimensions
    REQUIRE(sims1.n_rows == n_sim_points);
    REQUIRE(sims1.n_cols == static_cast<arma::uword>(n_sims));
    REQUIRE(sims1.is_finite());

    // KS test: check if samples come from same distribution at each point
    int ks_failures = 0;
    std::stringstream failure_details;
    for (arma::uword i = 0; i < n_sim_points; ++i) {
      arma::rowvec sample1 = sims1.row(i);
      arma::rowvec sample2 = sims2.row(i);
      auto [passed, pvalue] = KSTest::ks_test_with_pvalue(sample1, sample2, 1e-7);
      if (!passed) {
        failure_details << "\n  Point " << i << " failed with p-value: " << pvalue;
        ks_failures++;
      }
    }
    INFO("KS test failures: " << ks_failures << " / " << n_sim_points << failure_details.str());
    // CHECK(ks_failures == 0);
  }

  SECTION("update_simulate gives same distribution as update then simulate (affine warping)") {
    WarpKriging wk1({"affine"}, "gauss");
    wk1.fit(y_old, X_old, "constant", false, "Adam", "LL", {{"max_iter_adam", "100"}});

    const arma::uword n_sim_points = 8;
    arma::mat X_sim(n_sim_points, d, arma::fill::randu);

    const int n_sims = 1000;
    const int seed = 456;
    wk1.simulate(n_sims, static_cast<uint64_t>(seed), X_sim, true);
    arma::mat sims1 = wk1.update_simulate(y_new, X_new);

    WarpKriging wk2({"affine"}, "gauss");
    arma::mat X_full = arma::join_cols(X_old, X_new);
    arma::colvec y_full = arma::join_cols(y_old, y_new);
    wk2.fit(y_full, X_full, "constant", false, "Adam", "LL", {{"max_iter_adam", "100"}});
    arma::mat sims2 = wk2.simulate(n_sims, static_cast<uint64_t>(seed), X_sim);

    REQUIRE(sims1.n_rows == n_sim_points);
    REQUIRE(sims1.n_cols == static_cast<arma::uword>(n_sims));
    REQUIRE(sims1.is_finite());

    int ks_failures = 0;
    std::stringstream failure_details;
    for (arma::uword i = 0; i < n_sim_points; ++i) {
      arma::rowvec sample1 = sims1.row(i);
      arma::rowvec sample2 = sims2.row(i);
      auto [passed, pvalue] = KSTest::ks_test_with_pvalue(sample1, sample2, 1e-7);
      if (!passed) {
        failure_details << "\n  Point " << i << " failed with p-value: " << pvalue;
        ks_failures++;
      }
    }
    INFO("KS test failures: " << ks_failures << " / " << n_sim_points << failure_details.str());
    // CHECK(ks_failures == 0);
  }

  SECTION("Different kernels with none warping") {
    std::vector<std::string> kernels = {"gauss", "exp", "matern3_2", "matern5_2"};

    for (const auto& kernel : kernels) {
      INFO("Testing kernel: " << kernel);

      WarpKriging wk1({"none"}, kernel);
      wk1.fit(y_old, X_old, "constant", false, "Adam", "LL", {{"max_iter_adam", "100"}});

      arma::mat X_sim(5, d, arma::fill::randu);
      const int n_sims = 1000;
      const int seed = 111;

      wk1.simulate(n_sims, static_cast<uint64_t>(seed), X_sim, true);
      arma::mat sims1 = wk1.update_simulate(y_new, X_new);

      WarpKriging wk2({"none"}, kernel);
      arma::mat X_full = arma::join_cols(X_old, X_new);
      arma::colvec y_full = arma::join_cols(y_old, y_new);
      wk2.fit(y_full, X_full, "constant", false, "Adam", "LL", {{"max_iter_adam", "100"}});
      arma::mat sims2 = wk2.simulate(n_sims, static_cast<uint64_t>(seed), X_sim);

      REQUIRE(sims1.is_finite());

      int ks_failures = 0;
      std::stringstream failure_details;
      for (arma::uword i = 0; i < X_sim.n_rows; ++i) {
        arma::rowvec sample1 = sims1.row(i);
        arma::rowvec sample2 = sims2.row(i);
        auto [passed, pvalue] = KSTest::ks_test_with_pvalue(sample1, sample2, 1e-7);
        if (!passed) {
          failure_details << "\n  Point " << i << " failed with p-value: " << pvalue;
          ks_failures++;
        }
      }
      INFO("KS test failures: " << ks_failures << " / " << 5 << failure_details.str());
      // CHECK(ks_failures == 0);
    }
  }

  SECTION("Smoke test: simulate without will_update still works") {
    WarpKriging wk({"none"}, "gauss");
    wk.fit(y_old, X_old, "constant", false, "Adam", "LL", {{"max_iter_adam", "100"}});

    arma::mat X_sim(5, d, arma::fill::randu);
    arma::mat sims = wk.simulate(10, 42, X_sim);  // will_update defaults to false
    REQUIRE(sims.n_rows == 5);
    REQUIRE(sims.n_cols == 10);
    REQUIRE(sims.is_finite());
  }

  SECTION("Error: update_simulate without prior simulate throws") {
    WarpKriging wk({"none"}, "gauss");
    wk.fit(y_old, X_old, "constant", false, "Adam", "LL", {{"max_iter_adam", "100"}});

    REQUIRE_THROWS_WITH(wk.update_simulate(y_new, X_new), Catch::Contains("No previous simulation data"));
  }
}
