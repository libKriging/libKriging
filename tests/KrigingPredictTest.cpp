// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
// clang-format on

TEST_CASE("KrigingPredictTest - Check gradient vs finite differences", "[predict][kriging]") {
  arma::arma_rng::set_seed(123);

  // Generate test data: simple 2D problem
  const arma::uword n = 20;
  const arma::uword d = 2;
  arma::mat X(n, d, arma::fill::randu);
  arma::colvec y(n);
  
  // Use a simple function
  for (arma::uword i = 0; i < n; ++i) {
    double x1 = X(i, 0);
    double x2 = X(i, 1);
    y(i) = std::sin(3.0 * x1) + std::cos(5.0 * x2);
  }

  // Fit Kriging model
  Kriging kr("gauss");
  Kriging::Parameters params{std::nullopt, true, std::nullopt, true, std::nullopt, true};
  kr.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);

  // Test points for prediction
  arma::mat X_new(5, d, arma::fill::randu);

  SECTION("Mean gradient vs finite differences") {
    // Get prediction with analytical gradient
    auto [mean, stdev, cov, mean_deriv, stdev_deriv] = kr.predict(X_new, true, false, true);
    
    // Verify dimensions
    CHECK(mean_deriv.n_rows == X_new.n_rows);
    CHECK(mean_deriv.n_cols == d);
    
    // Check gradient using finite differences
    const double h = 1e-6;
    const double tol = 1e-4;  // Tolerance for finite difference comparison
    
    for (arma::uword i = 0; i < X_new.n_rows; ++i) {
      for (arma::uword j = 0; j < d; ++j) {
        // Compute finite difference
        arma::mat X_plus = X_new;
        arma::mat X_minus = X_new;
        X_plus(i, j) += h;
        X_minus(i, j) -= h;
        
        auto [mean_plus, s1, c1, d1, sd1] = kr.predict(X_plus, false, false, false);
        auto [mean_minus, s2, c2, d2, sd2] = kr.predict(X_minus, false, false, false);
        
        double finite_diff = (mean_plus(i) - mean_minus(i)) / (2.0 * h);
        double analytical = mean_deriv(i, j);
        
        INFO("Point " << i << ", dimension " << j);
        INFO("Analytical gradient: " << analytical);
        INFO("Finite difference: " << finite_diff);
        INFO("Absolute error: " << std::abs(analytical - finite_diff));
        INFO("Relative error: " << std::abs((analytical - finite_diff) / finite_diff));
        
        CHECK(std::abs(analytical - finite_diff) < tol);
      }
    }
  }

  SECTION("Standard deviation gradient vs finite differences") {
    // Get prediction with analytical gradient
    auto [mean, stdev, cov, mean_deriv, stdev_deriv] = kr.predict(X_new, true, false, true);
    
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
        
        auto [m1, stdev_plus, c1, d1, sd1] = kr.predict(X_plus, true, false, false);
        auto [m2, stdev_minus, c2, d2, sd2] = kr.predict(X_minus, true, false, false);
        
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

TEST_CASE("KrigingPredictTest - Check gradient vs finite differences with normalize=true",
          "[predict][kriging][normalize]") {
  // Regression test: predict(..., return_deriv=true) with normalize=true used to
  // return derivatives off by a factor of scaleX_j (missing chain-rule division
  // by scaleX_j when un-normalizing dyhat/dx and dysd2/dx). Use a wide, anisotropic
  // input range so the bug (ratio ~ scaleX_j, here very different from 1) is
  // clearly distinguishable from a correct derivative (ratio ~ 1).
  arma::arma_rng::set_seed(11);

  const arma::uword n = 30;
  arma::mat X = 100 * arma::randu<arma::mat>(n, 2) - 50;
  arma::colvec y(n);
  for (arma::uword i = 0; i < n; ++i)
    y(i) = 1000 * (0.01 * X(i, 0) * X(i, 0) - 0.02 * X(i, 1));

  Kriging kr("gauss");
  kr.fit(y, X, Trend::RegressionModel::Constant, true, "BFGS", "LL");

  // Off-design points: the stdev derivative is singular (ysd2_n == 0) exactly
  // at training points, so predict away from them.
  arma::mat X_new = X.head_rows(3) + 1.0;

  auto [mean, stdev, cov, mean_deriv, stdev_deriv] = kr.predict(X_new, true, false, true);

  const double h = 1e-3;
  const double tol = 1.0;  // absolute tolerance; the pre-fix bug is off by ~30-100x

  SECTION("Mean gradient vs finite differences") {
    for (arma::uword i = 0; i < X_new.n_rows; ++i) {
      for (arma::uword j = 0; j < X_new.n_cols; ++j) {
        arma::mat X_plus = X_new.row(i);
        arma::mat X_minus = X_new.row(i);
        X_plus(0, j) += h;
        X_minus(0, j) -= h;

        auto [mean_plus, s1, c1, d1, sd1] = kr.predict(X_plus, false, false, false);
        auto [mean_minus, s2, c2, d2, sd2] = kr.predict(X_minus, false, false, false);

        double finite_diff = (mean_plus(0) - mean_minus(0)) / (2.0 * h);
        double analytical = mean_deriv(i, j);

        INFO("Point " << i << ", dimension " << j);
        INFO("Analytical gradient: " << analytical);
        INFO("Finite difference: " << finite_diff);

        CHECK(analytical == Approx(finite_diff).margin(tol));
      }
    }
  }

  SECTION("Standard deviation gradient vs finite differences") {
    for (arma::uword i = 0; i < X_new.n_rows; ++i) {
      // Skip points where the predicted stdev is (near) zero: the analytical
      // gradient formula divides by stdev and is singular there (d/dx sqrt(v)
      // blows up as v -> 0). Whether a given off-design point falls in this
      // near-interpolation regime depends on the BFGS-optimized range
      // parameter, which can differ slightly across platforms/BLAS backends.
      if (stdev(i) < 1e-6)
        continue;
      for (arma::uword j = 0; j < X_new.n_cols; ++j) {
        arma::mat X_plus = X_new.row(i);
        arma::mat X_minus = X_new.row(i);
        X_plus(0, j) += h;
        X_minus(0, j) -= h;

        auto [m1, stdev_plus, c1, d1, sd1] = kr.predict(X_plus, true, false, false);
        auto [m2, stdev_minus, c2, d2, sd2] = kr.predict(X_minus, true, false, false);

        double finite_diff = (stdev_plus(0) - stdev_minus(0)) / (2.0 * h);
        double analytical = stdev_deriv(i, j);

        INFO("Point " << i << ", dimension " << j);
        INFO("Analytical stdev gradient: " << analytical);
        INFO("Finite difference: " << finite_diff);

        CHECK(analytical == Approx(finite_diff).margin(tol));
      }
    }
  }
}
