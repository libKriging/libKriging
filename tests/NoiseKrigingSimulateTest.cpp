// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
// clang-format on

TEST_CASE("NoiseKrigingSimulateTest - Check predict is consistent with simulate", "[simulate][noisekriging]") {
  arma::arma_rng::set_seed(123);

  // Generate test data with heteroscedastic noise
  const arma::uword n = 25;
  const arma::uword d = 2;
  arma::mat X(n, d, arma::fill::randu);
  arma::colvec y(n);
  arma::colvec noise(n);

  // Use a simple function with varying noise levels
  for (arma::uword i = 0; i < n; ++i) {
    double x1 = X(i, 0);
    double x2 = X(i, 1);
    double y_true = std::sin(3.0 * x1) + std::cos(5.0 * x2);
    noise(i) = 0.01 + 0.02 * arma::randu();
    y(i) = y_true + std::sqrt(noise(i)) * arma::randn();
  }

  // Fit NoiseKriging model
  Kriging nk("gauss", Kriging::NoiseModel::Heterogeneous);
  Kriging::Parameters params{std::nullopt, true, std::nullopt, true, std::nullopt, true};
  nk.fit(y, noise, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);

  // Test points for prediction/simulation
  const arma::uword n_test = 10;
  arma::mat X_test(n_test, d, arma::fill::randu);

  SECTION("Empirical mean from simulations matches prediction mean") {
    // Get analytical prediction
    auto [mean_pred, stdev_pred, cov, md, sd] = nk.predict(X_test, true, false, false);

    // Run many simulations
    const int n_sim = 1000;
    arma::vec no_noise = arma::zeros(n_test);
    arma::mat sims = nk.simulate(n_sim, 456, X_test, no_noise);  // without noise for comparison

    // Compute empirical mean
    arma::vec mean_sim = arma::mean(sims, 1);

    const double tol = 0.2;

    for (arma::uword i = 0; i < n_test; ++i) {
      double diff = std::abs(mean_pred(i) - mean_sim(i));

      INFO("Point " << i);
      INFO("Predicted mean: " << mean_pred(i));
      INFO("Simulated mean: " << mean_sim(i));
      INFO("Absolute error: " << diff);

      CHECK(diff < tol);
    }
  }

  SECTION("Empirical stdev from simulations matches prediction stdev") {
    // Get analytical prediction
    auto [mean_pred, stdev_pred, cov, md, sd] = nk.predict(X_test, true, false, false);

    // Run many simulations WITHOUT noise
    const int n_sim = 1000;
    arma::vec no_noise = arma::zeros(n_test);
    arma::mat sims = nk.simulate(n_sim, 789, X_test, no_noise);  // without noise

    // Compute empirical standard deviation
    arma::vec stdev_sim = arma::stddev(sims, 0, 1);

    // Note: For NoiseKriging at new points (not training points),
    // the prediction variance is the process variance only
    // So stdev_sim should match stdev_pred closely
    const double tol = 0.15;

    for (arma::uword i = 0; i < n_test; ++i) {
      double diff = std::abs(stdev_pred(i) - stdev_sim(i));

      INFO("Point " << i);
      INFO("Predicted stdev: " << stdev_pred(i));
      INFO("Simulated stdev (no noise): " << stdev_sim(i));
      INFO("Absolute error: " << diff);

      CHECK(diff < tol);
    }
  }

  SECTION("Simulations do NOT reproduce training data exactly (noise effect)") {
    // NoiseKriging simulations at training points should not exactly match
    // training data (due to heteroscedastic noise)
    const int n_sim = 500;
    arma::vec no_noise_train = arma::zeros(n);
    arma::mat sims = nk.simulate(n_sim, 111, X, no_noise_train);  // without noise

    // Mean should be reasonably close but not exact
    arma::vec mean_sim = arma::mean(sims, 1);

    // Check that at least some points have non-trivial difference
    int n_different = 0;
    for (arma::uword i = 0; i < n; ++i) {
      double diff = std::abs(y(i) - mean_sim(i));
      if (diff > 0.05) {
        n_different++;
      }
    }

    INFO("Number of training points with |y - sim_mean| > 0.05: " << n_different);

    // At least some points should show the noise effect
    CHECK(n_different > 0);
  }
}

TEST_CASE("NoiseKrigingSimulateTest - with_noise holds noise variances", "[simulate][noisekriging]") {
  arma::arma_rng::set_seed(321);
  const arma::uword n = 20;
  const arma::uword d = 2;
  arma::mat X(n, d, arma::fill::randu);
  arma::colvec noise = 0.01 + 0.02 * arma::randu<arma::vec>(n);
  arma::colvec y
      = arma::sin(3.0 * X.col(0)) + arma::cos(5.0 * X.col(1)) + arma::sqrt(noise) % arma::randn<arma::vec>(n);

  Kriging nk("matern5_2", Kriging::NoiseModel::Heterogeneous);
  nk.fit(y, noise, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", Kriging::Parameters{});

  const arma::uword n_test = 6;
  arma::mat X_test(n_test, d, arma::fill::randu);
  const arma::vec var_n = arma::linspace(0.05, 0.5, n_test);  // noise VARIANCES, like fit's noise
  const int n_sim = 20000;
  const arma::vec zero(n_test, arma::fill::zeros);

  // Same seed: the latent draws are identical, so the difference is the added
  // observation noise only; its empirical variance must match var_n.
  auto check_noise_var = [&](const arma::mat& noisy, const arma::mat& latent, const arma::vec& expected) {
    const arma::mat diff = noisy - latent;
    const arma::vec emp_var = arma::var(diff, 0, 1);
    for (arma::uword i = 0; i < n_test; ++i) {
      INFO("Point " << i << ": empirical noise variance " << emp_var(i) << " vs " << expected(i));
      CHECK(std::abs(emp_var(i) / expected(i) - 1.0) < 0.05);  // ~ 5 sd of the variance estimator
    }
  };

  SECTION("exact simulate, per-point variances") {
    check_noise_var(nk.simulate(n_sim, 7, X_test, var_n), nk.simulate(n_sim, 7, X_test, zero), var_n);
  }
  SECTION("exact simulate, scalar variance") {
    const arma::vec v1{0.3};
    check_noise_var(
        nk.simulate(n_sim, 8, X_test, v1), nk.simulate(n_sim, 8, X_test, zero), arma::vec(n_test).fill(0.3));
  }
  SECTION("simulate method vecchia, per-point variances") {
    check_noise_var(nk.simulate(n_sim, 9, X_test, var_n, false, "vecchia"),
                    nk.simulate(n_sim, 9, X_test, zero, false, "vecchia"),
                    var_n);
  }
  SECTION("update_simulate keeps the with_noise variances of simulate") {
    arma::mat X_u(3, d, arma::fill::randu);
    const arma::vec y_u = arma::sin(3.0 * X_u.col(0)) + arma::cos(5.0 * X_u.col(1));
    const arma::vec noise_u(3, arma::fill::value(0.02));
    Kriging nk_noisy(nk, ExplicitCopySpecifier{});
    Kriging nk_latent(nk, ExplicitCopySpecifier{});
    nk_noisy.simulate(n_sim, 10, X_test, var_n, true);
    nk_latent.simulate(n_sim, 10, X_test, zero, true);
    // same seed and data: the latent updates are identical, the difference is
    // the observation noise drawn with the with_noise variances of simulate
    check_noise_var(nk_noisy.update_simulate(y_u, noise_u, X_u), nk_latent.update_simulate(y_u, noise_u, X_u), var_n);
  }
  SECTION("negative variances are rejected") {
    CHECK_THROWS_AS(nk.simulate(10, 1, X_test, arma::vec(-var_n)), std::invalid_argument);
    CHECK_THROWS_AS(nk.simulate(10, 1, X_test, arma::vec(-var_n), false, "vecchia"), std::invalid_argument);
  }
}
