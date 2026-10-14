// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
#include "ks_test.hpp"
#include <sstream>
// clang-format on

TEST_CASE("NoiseKrigingUpdateSimulateTest - Update simulate equals updated model simulate",
          "[update_simulate][noisekriging]") {
  arma::arma_rng::set_seed(123);

  // Generate initial training data
  const arma::uword n_old = 15;
  const arma::uword n_new = 5;
  const arma::uword d = 2;

  arma::mat X_old(n_old, d, arma::fill::randu);
  arma::colvec y_old(n_old);
  arma::colvec noise_old = arma::vec(n_old).fill(0.01);  // Fixed noise variance

  auto test_function
      = [](const arma::rowvec& x) { return std::sin(3.0 * x(0)) + std::cos(5.0 * x(1)) + 10.0 + 10 * x(0); };

  for (arma::uword i = 0; i < n_old; ++i) {
    y_old(i) = test_function(X_old.row(i));
  }

  // Generate new data points
  arma::mat X_new(n_new, d, arma::fill::randu);
  arma::colvec y_new(n_new);
  arma::colvec noise_new = arma::vec(n_new).fill(0.01);  // Fixed noise variance

  for (arma::uword i = 0; i < n_new; ++i) {
    y_new(i) = test_function(X_new.row(i));
  }

  SECTION("update_simulate gives same distribution as update then simulate") {
    // Use fixed hardcoded parameters for both models
    // sigma2 = 100, theta = [1, 1], beta = [10] (constant trend)
    double sigma2_fixed = 100.0;
    arma::vec theta_fixed = arma::vec(d).fill(1.0);
    arma::vec beta_fixed = {10.0};
    Kriging::Parameters params_fixed;
    params_fixed.sigma2 = sigma2_fixed;
    params_fixed.is_sigma2_estim = false;
    params_fixed.theta = arma::mat(theta_fixed);
    params_fixed.is_theta_estim = false;
    params_fixed.beta = beta_fixed;
    params_fixed.is_beta_estim = false;

    // Build nk1 with fixed parameters
    Kriging nk1("gauss", Kriging::NoiseModel::Heterogeneous);
    nk1.fit(y_old, noise_old, X_old, Trend::RegressionModel::Constant, false, "none", "LL", params_fixed);

    // Simulation points
    const arma::uword n_sim_points = 10;
    arma::mat X_sim(n_sim_points, d, arma::fill::randu);

    // Method 1: Use update_simulate
    const int n_sims = 1000;
    const int seed = 789;
    nk1.simulate(n_sims, seed, X_sim, arma::vec(X_sim.n_rows).fill(1e-4), true);  // Store for update_simulate
    arma::mat sims1 = nk1.update_simulate(y_new, noise_new, X_new);

    // Method 2: Build nk2 with same fixed parameters on full data
    Kriging nk2("gauss", Kriging::NoiseModel::Heterogeneous);
    arma::mat X_full = arma::join_cols(X_old, X_new);
    arma::colvec y_full = arma::join_cols(y_old, y_new);
    arma::colvec noise_full = arma::join_cols(noise_old, noise_new);
    nk2.fit(y_full, noise_full, X_full, Trend::RegressionModel::Constant, false, "none", "LL", params_fixed);
    arma::mat sims2 = nk2.simulate(n_sims, seed, X_sim, arma::vec(X_sim.n_rows).fill(1e-4), false);

    // Save simulations to CSV for debugging
    sims1.save("sims1_noise_update_simulate.csv", arma::csv_ascii);
    sims2.save("sims2_noise_full_model.csv", arma::csv_ascii);
    X_sim.save("X_sim_noise.csv", arma::csv_ascii);

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
    CHECK(ks_failures == 0);
  }

  SECTION("Multiple points update_simulate") {
    // Use fixed hardcoded parameters for both models
    // sigma2 = 100, theta = [1, 1], beta = [10] (constant trend)
    double sigma2_fixed = 100.0;
    arma::vec theta_fixed = arma::vec(d).fill(1.0);
    arma::vec beta_fixed = {10.0};
    Kriging::Parameters params_fixed;
    params_fixed.sigma2 = sigma2_fixed;
    params_fixed.is_sigma2_estim = false;
    params_fixed.theta = arma::mat(theta_fixed);
    params_fixed.is_theta_estim = false;
    params_fixed.beta = beta_fixed;
    params_fixed.is_beta_estim = false;

    // Build nk1 with fixed parameters
    Kriging nk1("gauss", Kriging::NoiseModel::Heterogeneous);
    nk1.fit(y_old, noise_old, X_old, Trend::RegressionModel::Constant, false, "none", "LL", params_fixed);

    // Simulation points
    const arma::uword n_sim_points = 5;
    arma::mat X_sim(n_sim_points, d, arma::fill::randu);

    const int n_sims = 1000;
    const int seed = 999;

    // Use update_simulate with multiple new points
    nk1.simulate(n_sims, seed, X_sim, arma::vec(X_sim.n_rows).fill(1e-4), true);
    arma::mat sims1 = nk1.update_simulate(y_new, noise_new, X_new);

    // Build nk2 with same fixed parameters on full data
    Kriging nk2("gauss", Kriging::NoiseModel::Heterogeneous);
    arma::mat X_full = arma::join_cols(X_old, X_new);
    arma::colvec y_full = arma::join_cols(y_old, y_new);
    arma::colvec noise_full = arma::join_cols(noise_old, noise_new);
    nk2.fit(y_full, noise_full, X_full, Trend::RegressionModel::Constant, false, "none", "LL", params_fixed);
    arma::mat sims2 = nk2.simulate(n_sims, seed, X_sim, arma::vec(X_sim.n_rows).fill(1e-4), false);

    // KS test
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
    INFO("KS test failures: " << ks_failures << " / " << n_sim_points << failure_details.str());
    CHECK(ks_failures == 0);
  }

  SECTION("Different kernels") {
    std::vector<std::string> kernels = {"gauss", "exp", "matern3_2", "matern5_2"};

    for (const auto& kernel : kernels) {
      INFO("Testing kernel: " << kernel);

      // Use fixed hardcoded parameters for both models
      // sigma2 = 100, theta = [1, 1], beta = [10] (constant trend)
      double sigma2_fixed = 100.0;
      arma::vec theta_fixed = arma::vec(d).fill(1.0);
      arma::vec beta_fixed = {10.0};
      Kriging::Parameters params_fixed;
      params_fixed.sigma2 = sigma2_fixed;
      params_fixed.is_sigma2_estim = false;
      params_fixed.theta = arma::mat(theta_fixed);
      params_fixed.is_theta_estim = false;
      params_fixed.beta = beta_fixed;
      params_fixed.is_beta_estim = false;

      // Build nk1 with fixed parameters
      Kriging nk1(kernel, Kriging::NoiseModel::Heterogeneous);
      nk1.fit(y_old, noise_old, X_old, Trend::RegressionModel::Constant, false, "none", "LL", params_fixed);

      arma::mat X_sim(5, d, arma::fill::randu);
      const int n_sims = 1000;
      const int seed = 111;

      nk1.simulate(n_sims, seed, X_sim, arma::vec(X_sim.n_rows).fill(1e-4), true);
      arma::mat sims1 = nk1.update_simulate(y_new, noise_new, X_new);

      // Build nk2 with same fixed parameters on full data
      Kriging nk2(kernel, Kriging::NoiseModel::Heterogeneous);
      arma::mat X_full = arma::join_cols(X_old, X_new);
      arma::colvec y_full = arma::join_cols(y_old, y_new);
      arma::colvec noise_full = arma::join_cols(noise_old, noise_new);
      nk2.fit(y_full, noise_full, X_full, Trend::RegressionModel::Constant, false, "none", "LL", params_fixed);
      arma::mat sims2 = nk2.simulate(n_sims, seed, X_sim, arma::vec(X_sim.n_rows).fill(1e-4), false);

      // KS test
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
      CHECK(ks_failures == 0);
    }
  }

  SECTION("Different trend models") {
    std::vector<Trend::RegressionModel> trends
        = {Trend::RegressionModel::Constant, Trend::RegressionModel::Linear, Trend::RegressionModel::Quadratic};

    for (const auto& trend : trends) {
      INFO("Testing trend model: " << Trend::toString(trend));

      // Use fixed hardcoded parameters: sigma2 = 100, theta = [1, 1]
      // beta depends on trend: Constant: [10], Linear: [10, 10, 10], Quadratic: [10, 10, 10, 1, 1, 1]
      double sigma2_fixed = 100.0;
      arma::vec theta_fixed = arma::vec(d).fill(1.0);
      arma::vec beta_fixed;
      if (trend == Trend::RegressionModel::Constant) {
        beta_fixed = {10.0};
      } else if (trend == Trend::RegressionModel::Linear) {
        beta_fixed = {10.0, 10.0, 10.0};                 // intercept + 2 slopes for d=2
      } else {                                           // Quadratic
        beta_fixed = {10.0, 10.0, 10.0, 1.0, 1.0, 1.0};  // intercept + 2 linear + 3 quadratic terms for d=2
      }
      Kriging::Parameters params_fixed;
      params_fixed.sigma2 = sigma2_fixed;
      params_fixed.is_sigma2_estim = false;
      params_fixed.theta = arma::mat(theta_fixed);
      params_fixed.is_theta_estim = false;
      params_fixed.beta = beta_fixed;
      params_fixed.is_beta_estim = false;

      // Build nk1 with fixed parameters
      Kriging nk1("gauss", Kriging::NoiseModel::Heterogeneous);
      nk1.fit(y_old, noise_old, X_old, trend, false, "none", "LL", params_fixed);

      arma::mat X_sim(5, d, arma::fill::randu);
      const int n_sims = 1000;
      const int seed = 222;

      nk1.simulate(n_sims, seed, X_sim, arma::vec(X_sim.n_rows).fill(1e-4), true);
      arma::mat sims1 = nk1.update_simulate(y_new, noise_new, X_new);

      // Build nk2 with same fixed parameters on full data
      Kriging nk2("gauss", Kriging::NoiseModel::Heterogeneous);
      arma::mat X_full = arma::join_cols(X_old, X_new);
      arma::colvec y_full = arma::join_cols(y_old, y_new);
      arma::colvec noise_full = arma::join_cols(noise_old, noise_new);
      nk2.fit(y_full, noise_full, X_full, trend, false, "none", "LL", params_fixed);
      arma::mat sims2 = nk2.simulate(n_sims, seed, X_sim, arma::vec(X_sim.n_rows).fill(1e-4), false);

      // KS test
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
      CHECK(ks_failures == 0);
    }
  }
}

TEST_CASE("NoiseKrigingUpdateSimulateTest - noisy update_simulate does not leak into the noise-free form",
          "[update_simulate][noisekriging]") {
  arma::arma_rng::set_seed(321);
  const arma::uword d = 2;
  arma::mat X_old(15, d, arma::fill::randu);
  arma::vec y_old = arma::sin(3.0 * X_old.col(0)) + arma::cos(5.0 * X_old.col(1));
  arma::mat X_new(4, d, arma::fill::randu);
  arma::vec y_new = arma::sin(3.0 * X_new.col(0)) + arma::cos(5.0 * X_new.col(1));
  arma::mat X_sim(6, d, arma::fill::randu);

  Kriging::Parameters params;
  params.sigma2 = 1.0;
  params.is_sigma2_estim = false;
  params.theta = arma::mat(arma::vec(d).fill(0.5));
  params.is_theta_estim = false;
  Kriging k("gauss", Kriging::NoiseModel::Heterogeneous);
  k.fit(y_old, arma::vec(15).fill(0.01), X_old, Trend::RegressionModel::Constant, false, "none", "LL", params);

  k.simulate(50, 7, X_sim, arma::vec(), true);
  const arma::mat free_first = k.update_simulate(y_new, X_new);
  const arma::mat noisy = k.update_simulate(y_new, arma::vec(4).fill(0.5), X_new);
  REQUIRE(arma::abs(noisy - free_first).max() > 1e-6);  // the noise_u does change the result
  // Same X_u as the noisy call: must not reuse its (noisy) cached factorisation
  const arma::mat free_after = k.update_simulate(y_new, X_new);
  CHECK(arma::approx_equal(free_after, free_first, "absdiff", 1e-10));
}
