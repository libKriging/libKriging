// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/NoiseKriging.hpp"
// clang-format on

TEST_CASE("NoiseKrigingPredictTest - Check gradient vs finite differences", "[predict][noisekriging]") {
  arma::arma_rng::set_seed(123);

  // Generate test data with noise
  const arma::uword n = 25;
  const arma::uword d = 2;
  arma::mat X(n, d, arma::fill::randu);
  arma::colvec y(n);
  arma::colvec noise(n);
  
  // Use a simple function with heteroscedastic noise
  for (arma::uword i = 0; i < n; ++i) {
    double x1 = X(i, 0);
    double x2 = X(i, 1);
    double y_true = std::sin(3.0 * x1) + std::cos(5.0 * x2);
    noise(i) = 0.01 + 0.02 * arma::randu();
    y(i) = y_true + std::sqrt(noise(i)) * arma::randn();
  }

  // Fit NoiseKriging model
  NoiseKriging nk("gauss");
  NoiseKriging::Parameters params{std::nullopt, true, std::nullopt, true, std::nullopt, true};
  nk.fit(y, noise, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);

  // Test points for prediction
  arma::mat X_new(5, d, arma::fill::randu);

  SECTION("Mean gradient vs finite differences") {
    // Get prediction with analytical gradient
    auto [mean, stdev, cov, mean_deriv, stdev_deriv] = nk.predict(X_new, true, false, true);
    
    // Verify dimensions
    CHECK(mean_deriv.n_rows == X_new.n_rows);
    CHECK(mean_deriv.n_cols == d);
    
    // Check gradient using finite differences
    const double h = 1e-6;
    const double tol = 1e-4;
    
    for (arma::uword i = 0; i < X_new.n_rows; ++i) {
      for (arma::uword j = 0; j < d; ++j) {
        // Compute finite difference
        arma::mat X_plus = X_new;
        arma::mat X_minus = X_new;
        X_plus(i, j) += h;
        X_minus(i, j) -= h;
        
        auto [mean_plus, s1, c1, d1, sd1] = nk.predict(X_plus, false, false, false);
        auto [mean_minus, s2, c2, d2, sd2] = nk.predict(X_minus, false, false, false);
        
        double finite_diff = (mean_plus(i) - mean_minus(i)) / (2.0 * h);
        double analytical = mean_deriv(i, j);
        
        INFO("Point " << i << ", dimension " << j);
        INFO("Analytical gradient: " << analytical);
        INFO("Finite difference: " << finite_diff);
        INFO("Absolute error: " << std::abs(analytical - finite_diff));
        
        CHECK(std::abs(analytical - finite_diff) < tol);
      }
    }
  }

  SECTION("Standard deviation gradient vs finite differences") {
    // Get prediction with analytical gradient
    auto [mean, stdev, cov, mean_deriv, stdev_deriv] = nk.predict(X_new, true, false, true);
    
    // Verify dimensions
    CHECK(stdev_deriv.n_rows == X_new.n_rows);
    CHECK(stdev_deriv.n_cols == d);
    
    // Check gradient using finite differences
    const double h = 1e-6;
    const double tol = 1e-4;
    
    for (arma::uword i = 0; i < X_new.n_rows; ++i) {
      for (arma::uword j = 0; j < d; ++j) {
        // Compute finite difference
        arma::mat X_plus = X_new;
        arma::mat X_minus = X_new;
        X_plus(i, j) += h;
        X_minus(i, j) -= h;
        
        auto [m1, stdev_plus, c1, d1, sd1] = nk.predict(X_plus, true, false, false);
        auto [m2, stdev_minus, c2, d2, sd2] = nk.predict(X_minus, true, false, false);
        
        double finite_diff = (stdev_plus(i) - stdev_minus(i)) / (2.0 * h);
        double analytical = stdev_deriv(i, j);
        
        INFO("Point " << i << ", dimension " << j);
        INFO("Analytical stdev gradient: " << analytical);
        INFO("Finite difference: " << finite_diff);
        INFO("Absolute error: " << std::abs(analytical - finite_diff));
        
        CHECK(std::abs(analytical - finite_diff) < tol);
      }
    }
  }
}
