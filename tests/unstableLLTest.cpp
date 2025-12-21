// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
#include "libKriging/LinearAlgebra.hpp"
// clang-format on

TEST_CASE("Fit: unstable LL (long range with 1D Gauss kernel)", "[fit][unstable][loglik]") {
  // 1D function, small design, but not stationary
  arma::mat X(7, 1);
  X << 0.0 << arma::endr
    << 0.25 << arma::endr
    << 0.33 << arma::endr
    << 0.45 << arma::endr
    << 0.5 << arma::endr
    << 0.75 << arma::endr
    << 1.0 << arma::endr;

  // Function: f(x) = 1 - 1/2 * (sin(4*x) / (1+x) + 2*cos(12*x)*x^6 + 0.7)
  arma::vec y(7);
  for (arma::uword i = 0; i < X.n_rows; ++i) {
    double x = X(i, 0);
    y(i) = 1.0 - 0.5 * (std::sin(4.0 * x) / (1.0 + x) + 2.0 * std::cos(12.0 * x) * std::pow(x, 6.0) + 0.7);
  }

  INFO("X = " << X.t());
  INFO("y = " << y.t());

  // Save default settings
  bool default_rcond_checked = LinearAlgebra::chol_rcond_checked();
  double default_num_nugget = LinearAlgebra::get_num_nugget();

  SECTION("Test with bad initial theta and no optimization") {
    // Set linalg settings
    LinearAlgebra::check_chol_rcond(false);
    LinearAlgebra::set_num_nugget(1e-15);
    LinearAlgebra::set_chol_warning(true);

    // Build Kriging with bad init theta value (no optimization)
    Kriging k("gauss");
    Kriging::Parameters params;
    arma::mat theta_init(1, 1);
    theta_init(0, 0) = 9.0;
    params.theta = theta_init;
    params.sigma2 = 1e10;
    params.is_theta_estim = false;
    params.is_sigma2_estim = false;

    k.fit(y, X, Trend::RegressionModel::Constant, false, "none", "LL", params);

    arma::vec theta_fixed = k.theta();
    double sigma2_fixed = k.sigma2();

    INFO("Fixed theta: " << theta_fixed.t());
    INFO("Fixed sigma2: " << sigma2_fixed);

    CHECK(arma::approx_equal(theta_fixed, arma::vec{9.0}, "absdiff", 1e-10));
    CHECK(std::abs(sigma2_fixed - 1e10) < 1.0);

    // Restore defaults
    LinearAlgebra::check_chol_rcond(default_rcond_checked);
    LinearAlgebra::set_num_nugget(default_num_nugget);
  }

  SECTION("Test BFGS optimization with rcond checking") {
    LinearAlgebra::check_chol_rcond(true);
    LinearAlgebra::set_num_nugget(1e-15);

    Kriging k("gauss");
    Kriging::Parameters params;
    params.is_theta_estim = true;
    params.is_sigma2_estim = true;

    k.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);

    arma::vec theta_bfgs = k.theta();
    double ll_bfgs = k.logLikelihood();

    INFO("BFGS theta: " << theta_bfgs.t());
    INFO("BFGS log-likelihood: " << ll_bfgs);

    CHECK(theta_bfgs(0) > 0.0);
    CHECK(std::isfinite(ll_bfgs));

    // Restore defaults
    LinearAlgebra::check_chol_rcond(default_rcond_checked);
    LinearAlgebra::set_num_nugget(default_num_nugget);
  }

  SECTION("Test logLikelihood with and without rcond checking") {
    // Test at a specific theta value
    arma::vec theta_test = {5.0};

    // First, fit a model to get the structure
    Kriging k("gauss");
    Kriging::Parameters params;
    params.is_theta_estim = true;
    params.is_sigma2_estim = true;

    LinearAlgebra::check_chol_rcond(true);
    k.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);

    // Test LL evaluation with rcond checking
    LinearAlgebra::check_chol_rcond(true);
    auto result_with_check = k.logLikelihoodFun(theta_test, false, false, false);
    double ll_with_check = std::get<0>(result_with_check);

    // Test LL evaluation without rcond checking
    LinearAlgebra::check_chol_rcond(false);
    auto result_no_check = k.logLikelihoodFun(theta_test, false, false, false);
    double ll_no_check = std::get<0>(result_no_check);

    INFO("LL with rcond check: " << ll_with_check);
    INFO("LL without rcond check: " << ll_no_check);

    // Both should be finite
    CHECK(std::isfinite(ll_with_check));
    CHECK(std::isfinite(ll_no_check));

    // Restore defaults
    LinearAlgebra::check_chol_rcond(default_rcond_checked);
    LinearAlgebra::set_num_nugget(default_num_nugget);
  }

  SECTION("Test rcond computation at different theta values") {
    // Fit a model first
    Kriging k("gauss");
    Kriging::Parameters params;
    params.is_theta_estim = false;

    LinearAlgebra::check_chol_rcond(false);

    std::vector<double> theta_vals = {1.0, 3.0, 5.0, 7.0, 9.0};

    for (double theta_val : theta_vals) {
      arma::mat theta_tmp(1, 1);
      theta_tmp(0, 0) = theta_val;
      params.theta = theta_tmp;

      try {
        Kriging k_tmp("gauss");
        k_tmp.fit(y, X, Trend::RegressionModel::Constant, false, "none", "LL", params);

        arma::mat T = k_tmp.T();
        double rcond_val = LinearAlgebra::rcond_chol(T);
        double rcond_approx_val = LinearAlgebra::rcond_approx_chol(T);

        INFO("theta=" << theta_val << ": rcond=" << rcond_val << ", rcond_approx=" << rcond_approx_val);

        CHECK(rcond_val >= 0.0);
        CHECK(rcond_approx_val >= 0.0);
      } catch (const std::exception& e) {
        INFO("theta=" << theta_val << ": failed to fit - " << e.what());
      }
    }

    // Restore defaults
    LinearAlgebra::check_chol_rcond(default_rcond_checked);
    LinearAlgebra::set_num_nugget(default_num_nugget);
  }

  SECTION("LL / Fit: unstable LL fixed using rcond failover - BFGS vs BFGS10") {
    LinearAlgebra::check_chol_rcond(true);
    LinearAlgebra::set_num_nugget(1e-15);
    LinearAlgebra::set_chol_warning(true);

    // Fit with BFGS
    Kriging k_bfgs("gauss");
    Kriging::Parameters params_bfgs;
    params_bfgs.is_theta_estim = true;
    params_bfgs.is_sigma2_estim = true;

    k_bfgs.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", params_bfgs);

    arma::vec theta_bfgs = k_bfgs.theta();
    double ll_bfgs = k_bfgs.logLikelihood();

    // Fit with BFGS10
    Kriging k_bfgs10("gauss");
    Kriging::Parameters params_bfgs10;
    params_bfgs10.is_theta_estim = true;
    params_bfgs10.is_sigma2_estim = true;

    k_bfgs10.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS10", "LL", params_bfgs10);

    arma::vec theta_bfgs10 = k_bfgs10.theta();
    double ll_bfgs10 = k_bfgs10.logLikelihood();

    INFO("BFGS theta: " << theta_bfgs.t() << ", LL: " << ll_bfgs);
    INFO("BFGS10 theta: " << theta_bfgs10.t() << ", LL: " << ll_bfgs10);

    // Check that both optimizers find similar theta values
    // This is the main test from the R code
    CHECK(arma::approx_equal(theta_bfgs, theta_bfgs10, "absdiff", 1e-4));

    // Both should produce finite log-likelihoods
    CHECK(std::isfinite(ll_bfgs));
    CHECK(std::isfinite(ll_bfgs10));

    // The log-likelihoods should be similar
    CHECK(std::abs(ll_bfgs - ll_bfgs10) < 1e-2);

    // Restore defaults
    LinearAlgebra::check_chol_rcond(default_rcond_checked);
    LinearAlgebra::set_num_nugget(default_num_nugget);
  }

  // Ensure defaults are restored at the end
  LinearAlgebra::check_chol_rcond(default_rcond_checked);
  LinearAlgebra::set_num_nugget(default_num_nugget);
}
