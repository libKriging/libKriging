// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"

// clang-format on

TEST_CASE("Gradient verification with fixed theta (no optimization)", "[gradient][loglik]") {
  // Create a simple test case
  arma::mat X = {{0.0, 0.0},
                 {1.0, 0.0},
                 {0.0, 1.0},
                 {1.0, 1.0},
                 {0.5, 0.5},
                 {0.3, 0.7}};
  
  arma::vec y = {0.0, 1.0, 1.0, 2.0, 1.0, 0.8};
  
  // Fit the model WITHOUT optimization - use fixed theta
  Kriging kr("gauss");
  Kriging::Parameters parameters;
  parameters.theta = arma::mat{{0.5, 0.5}};  // Fixed theta
  parameters.is_theta_estim = false;         // Don't optimize theta
  parameters.is_beta_estim = true;
  parameters.is_sigma2_estim = true;
  
  kr.fit(y, X, Trend::RegressionModel::Constant, false, "none", "LL", parameters);
  
  arma::vec theta_fitted = kr.theta();
  INFO("Fitted (fixed) theta: " << theta_fitted.t());
  INFO("Kriging summary:\n" << kr.summary());
  CHECK(arma::approx_equal(theta_fitted, arma::vec{0.5, 0.5}, "absdiff", 1e-10));
  
  SECTION("Gradient at a different theta value") {
    // Test gradient at a different theta value
    arma::vec theta_test = {0.4, 0.6};
    
    // Get analytical gradient
    auto result = kr.logLikelihoodFun(theta_test, true, false, false);
    double ll = std::get<0>(result);
    arma::vec grad_analytical = std::get<1>(result);
    
    INFO("Test theta: " << theta_test.t());
    INFO("Log-likelihood: " << ll);
    INFO("Analytical gradient: " << grad_analytical.t());
    
    // Compute numerical gradient using finite differences
    double eps = 1e-7;
    arma::vec grad_numerical(theta_test.n_elem);
    
    for (arma::uword i = 0; i < theta_test.n_elem; i++) {
      arma::vec theta_plus = theta_test;
      arma::vec theta_minus = theta_test;
      
      theta_plus(i) += eps;
      theta_minus(i) -= eps;
      
      double ll_plus = std::get<0>(kr.logLikelihoodFun(theta_plus, false, false, false));
      double ll_minus = std::get<0>(kr.logLikelihoodFun(theta_minus, false, false, false));
      
      grad_numerical(i) = (ll_plus - ll_minus) / (2.0 * eps);
    }
    
    INFO("Numerical gradient: " << grad_numerical.t());
    
    // Check each component
    for (arma::uword i = 0; i < theta_test.n_elem; i++) {
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
  
  SECTION("Gradient at fitted theta") {
    arma::vec theta_test = theta_fitted;
    
    auto result = kr.logLikelihoodFun(theta_test, true, false, false);
    double ll = std::get<0>(result);
    arma::vec grad_analytical = std::get<1>(result);
    
    INFO("Fitted theta: " << theta_test.t());
    INFO("Log-likelihood at fitted theta: " << ll);
    INFO("Analytical gradient: " << grad_analytical.t());
    
    // Numerical gradient
    double eps = 1e-7;
    arma::vec grad_numerical(theta_test.n_elem);
    
    for (arma::uword i = 0; i < theta_test.n_elem; i++) {
      arma::vec theta_plus = theta_test;
      arma::vec theta_minus = theta_test;
      
      theta_plus(i) += eps;
      theta_minus(i) -= eps;
      
      double ll_plus = std::get<0>(kr.logLikelihoodFun(theta_plus, false, false, false));
      double ll_minus = std::get<0>(kr.logLikelihoodFun(theta_minus, false, false, false));
      
      grad_numerical(i) = (ll_plus - ll_minus) / (2.0 * eps);
    }
    
    INFO("Numerical gradient: " << grad_numerical.t());
    
    for (arma::uword i = 0; i < theta_test.n_elem; i++) {
      double abs_diff = std::abs(grad_analytical(i) - grad_numerical(i));
      double rel_error = abs_diff / (std::abs(grad_analytical(i)) + 1e-10);
      
      INFO("Component " << i << ": rel_error=" << rel_error);
      CHECK(rel_error < 1e-3);
    }
  }
}

TEST_CASE("Hessian verification with fixed theta (no optimization)", "[hessian][loglik]") {
  arma::mat X = {{0.0, 0.0},
                 {1.0, 0.0},
                 {0.0, 1.0},
                 {1.0, 1.0},
                 {0.5, 0.5},
                 {0.3, 0.7}};
  
  arma::vec y = {0.0, 1.0, 1.0, 2.0, 1.0, 0.8};
  
  Kriging kr("gauss");
  Kriging::Parameters parameters;
  parameters.theta = arma::mat{{0.5, 0.5}};
  parameters.is_theta_estim = false;
  parameters.is_beta_estim = true;
  parameters.is_sigma2_estim = true;
  
  kr.fit(y, X, Trend::RegressionModel::Constant, false, "none", "LL", parameters);
  
  INFO("Kriging summary:\n" << kr.summary());
  
  SECTION("Hessian at test theta") {
    arma::vec theta_test = {0.4, 0.6};
    
    // Get analytical Hessian
    auto result = kr.logLikelihoodFun(theta_test, true, true, false);
    double ll = std::get<0>(result);
    arma::vec grad = std::get<1>(result);
    arma::mat hess_analytical = std::get<2>(result);
    
    INFO("Test theta: " << theta_test.t());
    INFO("Log-likelihood: " << ll);
    INFO("Analytical Hessian:\n" << hess_analytical);
    
    // Compute numerical Hessian using finite differences of gradient
    double eps = 1e-6;
    arma::mat hess_numerical(theta_test.n_elem, theta_test.n_elem);
    
    for (arma::uword i = 0; i < theta_test.n_elem; i++) {
      arma::vec theta_plus = theta_test;
      arma::vec theta_minus = theta_test;
      
      theta_plus(i) += eps;
      theta_minus(i) -= eps;
      
      arma::vec grad_plus = std::get<1>(kr.logLikelihoodFun(theta_plus, true, false, false));
      arma::vec grad_minus = std::get<1>(kr.logLikelihoodFun(theta_minus, true, false, false));
      
      hess_numerical.col(i) = (grad_plus - grad_minus) / (2.0 * eps);
    }
    
    INFO("Numerical Hessian:\n" << hess_numerical);
    
    // Check each element
    for (arma::uword i = 0; i < theta_test.n_elem; i++) {
      for (arma::uword j = 0; j < theta_test.n_elem; j++) {
        double abs_diff = std::abs(hess_analytical(i, j) - hess_numerical(i, j));
        double denominator = std::max(std::abs(hess_analytical(i, j)), std::abs(hess_numerical(i, j)));
        double rel_error = (denominator > 1e-10) ? (abs_diff / denominator) : abs_diff;
        
        INFO("Hessian(" << i << "," << j << "): analytical=" << hess_analytical(i, j) 
             << ", numerical=" << hess_numerical(i, j)
             << ", abs_diff=" << abs_diff 
             << ", rel_error=" << rel_error);
        
        // Hessian should match within 1% relative error
        CHECK(rel_error < 1e-2);
      }
    }
  }
  
  SECTION("Hessian symmetry check") {
    arma::vec theta_test = {0.4, 0.6};
    auto result = kr.logLikelihoodFun(theta_test, true, true, false);
    arma::mat hess = std::get<2>(result);
    
    // Hessian should be symmetric
    for (arma::uword i = 0; i < theta_test.n_elem; i++) {
      for (arma::uword j = i + 1; j < theta_test.n_elem; j++) {
        double diff = std::abs(hess(i, j) - hess(j, i));
        INFO("Hessian(" << i << "," << j << ") vs Hessian(" << j << "," << i << "): diff=" << diff);
        CHECK(diff < 1e-10);
      }
    }
  }
}

TEST_CASE("Simple 1D case for gradient verification", "[gradient][loglik][1d]") {
  // Very simple 1D case for easier debugging
  arma::mat X(4, 1);
  X << 0.0 << arma::endr
    << 0.3 << arma::endr
    << 0.7 << arma::endr
    << 1.0 << arma::endr;
  arma::vec y = {0.0, 0.5, 0.8, 1.0};
  
  Kriging kr("gauss");
  Kriging::Parameters parameters;
  arma::mat theta_mat(1, 1);
  theta_mat(0, 0) = 0.5;
  parameters.theta = theta_mat;
  parameters.is_theta_estim = false;
  parameters.is_beta_estim = true;
  parameters.is_sigma2_estim = true;
  
  kr.fit(y, X, Trend::RegressionModel::Constant, false, "none", "LL", parameters);
  
  INFO("Kriging summary:\n" << kr.summary());
  
  SECTION("Gradient at test theta") {
    arma::vec theta_test = {0.6};
    
    auto result = kr.logLikelihoodFun(theta_test, true, false, false);
    double ll = std::get<0>(result);
    arma::vec grad_analytical = std::get<1>(result);
    
    INFO("Test theta: " << theta_test.t());
    INFO("Log-likelihood: " << ll);
    INFO("Analytical gradient: " << grad_analytical.t());
    
    // Numerical gradient
    double eps = 1e-7;
    double ll_plus = std::get<0>(kr.logLikelihoodFun(arma::vec{theta_test(0) + eps}, false, false, false));
    double ll_minus = std::get<0>(kr.logLikelihoodFun(arma::vec{theta_test(0) - eps}, false, false, false));
    double grad_numerical = (ll_plus - ll_minus) / (2.0 * eps);
    
    INFO("Numerical gradient: " << grad_numerical);
    
    double abs_diff = std::abs(grad_analytical(0) - grad_numerical);
    double rel_error = abs_diff / (std::abs(grad_analytical(0)) + 1e-10);
    
    INFO("abs_diff=" << abs_diff << ", rel_error=" << rel_error);
    
    CHECK(rel_error < 1e-3);
  }
}
