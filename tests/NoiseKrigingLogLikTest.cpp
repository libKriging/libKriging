// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/NoiseKriging.hpp"

// clang-format on

TEST_CASE("NoiseKriging gradient verification with fixed parameters", "[gradient][loglik][noise]") {
  // Create a simple test case
  arma::mat X = {{0.0, 0.0},
                 {1.0, 0.0},
                 {0.0, 1.0},
                 {1.0, 1.0},
                 {0.5, 0.5},
                 {0.3, 0.7}};
  
  arma::vec y = {0.0, 1.0, 1.0, 2.0, 1.0, 0.8};
  arma::vec noise = {0.01, 0.01, 0.01, 0.01, 0.01, 0.01};  // Small noise
  
  // Fit the model WITHOUT optimization - use fixed parameters
  NoiseKriging nk("gauss");
  NoiseKriging::Parameters parameters;
  parameters.theta = arma::mat{{0.5, 0.5}};  // Fixed theta
  parameters.is_theta_estim = false;         // Don't optimize theta
  parameters.is_beta_estim = true;
  parameters.sigma2 = arma::vec{0.5};        // Fixed sigma2
  parameters.is_sigma2_estim = false;
  
  nk.fit(y, noise, X, Trend::RegressionModel::Constant, false, "none", "LL", parameters);
  
  arma::vec theta_fitted = nk.theta();
  INFO("Fitted (fixed) theta: " << theta_fitted.t());
  INFO("NoiseKriging summary:\n" << nk.summary());
  CHECK(arma::approx_equal(theta_fitted, arma::vec{0.5, 0.5}, "absdiff", 1e-10));
  
  SECTION("Gradient at a different theta value") {
    // Test gradient at a different theta value
    // logLikelihoodFun expects [theta(1), ..., theta(d), sigma2]
    arma::vec theta_sigma2_test = {0.4, 0.6, nk.sigma2()};
    
    // Get analytical gradient
    auto result = nk.logLikelihoodFun(theta_sigma2_test, true, false);
    double ll = std::get<0>(result);
    arma::vec grad_analytical = std::get<1>(result);
    
    INFO("Test theta_sigma2: " << theta_sigma2_test.t());
    INFO("Log-likelihood: " << ll);
    INFO("Analytical gradient: " << grad_analytical.t());
    
    // Compute numerical gradient using finite differences
    double eps = 1e-7;
    arma::vec grad_numerical(theta_sigma2_test.n_elem);
    
    for (arma::uword i = 0; i < theta_sigma2_test.n_elem; i++) {
      arma::vec theta_sigma2_plus = theta_sigma2_test;
      arma::vec theta_sigma2_minus = theta_sigma2_test;
      
      theta_sigma2_plus(i) += eps;
      theta_sigma2_minus(i) -= eps;
      
      double ll_plus = std::get<0>(nk.logLikelihoodFun(theta_sigma2_plus, false, false));
      double ll_minus = std::get<0>(nk.logLikelihoodFun(theta_sigma2_minus, false, false));
      
      grad_numerical(i) = (ll_plus - ll_minus) / (2.0 * eps);
    }
    
    INFO("Numerical gradient: " << grad_numerical.t());
    
    // Check each component
    for (arma::uword i = 0; i < theta_sigma2_test.n_elem; i++) {
      double abs_diff = std::abs(grad_analytical(i) - grad_numerical(i));
      double rel_error = abs_diff / (std::abs(grad_analytical(i)) + 1e-10);
      
      INFO("Component " << i << ": analytical=" << grad_analytical(i) 
           << ", numerical=" << grad_numerical(i)
           << ", abs_diff=" << abs_diff 
           << ", rel_error=" << rel_error);
      
      // Gradient should match within 0.1% relative error
      CHECK(rel_error < 1e-3);
    }
  }
  
  SECTION("Gradient at fitted parameters") {
    // logLikelihoodFun expects [theta(1), ..., theta(d), sigma2]
    arma::vec theta_sigma2_fitted = arma::join_cols(theta_fitted, arma::vec{nk.sigma2()});
    
    auto result = nk.logLikelihoodFun(theta_sigma2_fitted, true, false);
    double ll = std::get<0>(result);
    arma::vec grad_analytical = std::get<1>(result);
    
    INFO("Fitted theta_sigma2: " << theta_sigma2_fitted.t());
    INFO("Log-likelihood at fitted parameters: " << ll);
    INFO("Analytical gradient: " << grad_analytical.t());
    
    // Numerical gradient
    double eps = 1e-7;
    arma::vec grad_numerical(theta_sigma2_fitted.n_elem);
    
    for (arma::uword i = 0; i < theta_sigma2_fitted.n_elem; i++) {
      arma::vec theta_sigma2_plus = theta_sigma2_fitted;
      arma::vec theta_sigma2_minus = theta_sigma2_fitted;
      
      theta_sigma2_plus(i) += eps;
      theta_sigma2_minus(i) -= eps;
      
      double ll_plus = std::get<0>(nk.logLikelihoodFun(theta_sigma2_plus, false, false));
      double ll_minus = std::get<0>(nk.logLikelihoodFun(theta_sigma2_minus, false, false));
      
      grad_numerical(i) = (ll_plus - ll_minus) / (2.0 * eps);
    }
    
    INFO("Numerical gradient: " << grad_numerical.t());
    
    for (arma::uword i = 0; i < theta_sigma2_fitted.n_elem; i++) {
      double abs_diff = std::abs(grad_analytical(i) - grad_numerical(i));
      double rel_error = abs_diff / (std::abs(grad_analytical(i)) + 1e-10);
      
      INFO("Component " << i << ": rel_error=" << rel_error);
      CHECK(rel_error < 1e-3);
    }
  }
}

TEST_CASE("NoiseKriging gradient verification 1D case", "[gradient][loglik][noise][1d]") {
  // Simple 1D case for easier debugging
  arma::mat X(5, 1);
  X(0, 0) = 0.0;
  X(1, 0) = 0.25;
  X(2, 0) = 0.5;
  X(3, 0) = 0.75;
  X(4, 0) = 1.0;
  
  arma::vec y = {0.0, 0.5, 1.0, 0.5, 0.0};
  arma::vec noise = {0.01, 0.01, 0.01, 0.01, 0.01};
  
  NoiseKriging nk("gauss");
  NoiseKriging::Parameters parameters;
  arma::mat theta_init(1, 1);
  theta_init(0, 0) = 0.5;
  parameters.theta = theta_init;            // Fixed theta
  parameters.is_theta_estim = false;
  parameters.is_beta_estim = true;
  parameters.sigma2 = arma::vec{0.3};       // Fixed sigma2
  parameters.is_sigma2_estim = false;
  
  nk.fit(y, noise, X, Trend::RegressionModel::Constant, false, "none", "LL", parameters);
  
  // Test at a specific theta
  // logLikelihoodFun expects [theta(1), ..., theta(d), sigma2]
  arma::vec theta_sigma2_test = {0.6, nk.sigma2()};
  
  auto result = nk.logLikelihoodFun(theta_sigma2_test, true, false);
  double ll = std::get<0>(result);
  arma::vec grad_analytical = std::get<1>(result);
  
  INFO("Test theta_sigma2 (1D): " << theta_sigma2_test.t());
  INFO("Log-likelihood: " << ll);
  INFO("Analytical gradient: " << grad_analytical.t());
  
  // Numerical gradient
  double eps = 1e-7;
  arma::vec grad_numerical(theta_sigma2_test.n_elem);
  
  for (arma::uword i = 0; i < theta_sigma2_test.n_elem; i++) {
    arma::vec theta_sigma2_plus = theta_sigma2_test;
    arma::vec theta_sigma2_minus = theta_sigma2_test;
    
    theta_sigma2_plus(i) += eps;
    theta_sigma2_minus(i) -= eps;
    
    double ll_plus = std::get<0>(nk.logLikelihoodFun(theta_sigma2_plus, false, false));
    double ll_minus = std::get<0>(nk.logLikelihoodFun(theta_sigma2_minus, false, false));
    
    grad_numerical(i) = (ll_plus - ll_minus) / (2.0 * eps);
  }
  
  INFO("Numerical gradient: " << grad_numerical.t());
  
  // Check all components
  for (arma::uword i = 0; i < theta_sigma2_test.n_elem; i++) {
    double abs_diff = std::abs(grad_analytical(i) - grad_numerical(i));
    double rel_error = abs_diff / (std::abs(grad_analytical(i)) + 1e-10);
    
    INFO("Component " << i << ": analytical=" << grad_analytical(i) 
         << ", numerical=" << grad_numerical(i)
         << ", abs_diff=" << abs_diff 
         << ", rel_error=" << rel_error);
    
    CHECK(rel_error < 1e-3);
  }
}
