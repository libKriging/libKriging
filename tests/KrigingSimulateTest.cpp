// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
// clang-format on

TEST_CASE("KrigingSimulateTest - Check predict is consistent with simulate", "[simulate][kriging]") {
  arma::arma_rng::set_seed(123);

  // Generate test data
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

  // Test points for prediction/simulation
  const arma::uword n_test = 10;
  arma::mat X_test(n_test, d, arma::fill::randu);

  SECTION("Empirical mean from simulations matches prediction mean") {
    // Get analytical prediction
    auto [mean_pred, stdev_pred, cov, md, sd] = kr.predict(X_test, true, false, false);
    
    // Run many simulations
    const int n_sim = 1000;
    arma::mat sims = kr.simulate(n_sim, 456, X_test);
    
    // Compute empirical mean
    arma::vec mean_sim = arma::mean(sims, 1);  // Mean across simulations (rows)
    
    // Check consistency (allow some statistical variation)
    const double tol = 0.2;  // Tolerance for mean comparison
    
    for (arma::uword i = 0; i < n_test; ++i) {
      double diff = std::abs(mean_pred(i) - mean_sim(i));
      double relative_error = diff / (std::abs(mean_pred(i)) + 1e-10);
      
      INFO("Point " << i);
      INFO("Predicted mean: " << mean_pred(i));
      INFO("Simulated mean: " << mean_sim(i));
      INFO("Absolute error: " << diff);
      INFO("Relative error: " << relative_error);
      
      CHECK(diff < tol);
    }
  }

  SECTION("Empirical stdev from simulations matches prediction stdev") {
    // Get analytical prediction
    auto [mean_pred, stdev_pred, cov, md, sd] = kr.predict(X_test, true, false, false);
    
    // Run many simulations
    const int n_sim = 1000;
    arma::mat sims = kr.simulate(n_sim, 789, X_test);
    
    // Compute empirical standard deviation
    arma::vec stdev_sim = arma::stddev(sims, 0, 1);  // Stddev across simulations (dim=1)
    
    // Check consistency (allow statistical variation, ~1/sqrt(n_sim))
    const double tol = 0.15;  // Tolerance for stdev comparison
    
    for (arma::uword i = 0; i < n_test; ++i) {
      double diff = std::abs(stdev_pred(i) - stdev_sim(i));
      double relative_error = diff / (stdev_pred(i) + 1e-10);
      
      INFO("Point " << i);
      INFO("Predicted stdev: " << stdev_pred(i));
      INFO("Simulated stdev: " << stdev_sim(i));
      INFO("Absolute error: " << diff);
      INFO("Relative error: " << relative_error);
      
      CHECK(diff < tol);
    }
  }

  SECTION("Simulations at training points match training data") {
    // Simulate at training points - should reproduce training values
    const int n_sim = 500;
    arma::mat sims = kr.simulate(n_sim, 111, X);
    
    // Mean of simulations should be close to training y
    arma::vec mean_sim = arma::mean(sims, 1);
    
    const double tol = 0.1;
    
    for (arma::uword i = 0; i < n; ++i) {
      double diff = std::abs(y(i) - mean_sim(i));
      
      INFO("Training point " << i);
      INFO("Training value: " << y(i));
      INFO("Simulated mean: " << mean_sim(i));
      INFO("Absolute error: " << diff);
      
      CHECK(diff < tol);
    }
  }

  SECTION("Covariance structure in simulations") {
    // Get two close test points
    arma::mat X_close(2, d);
    X_close.row(0) = arma::vec{0.5, 0.5}.t();
    X_close.row(1) = arma::vec{0.51, 0.5}.t();  // Very close to first point
    
    // Get prediction with covariance
    auto [mean_pred, stdev_pred, cov_pred, md, sd] = kr.predict(X_close, true, true, false);
    
    // Run simulations
    const int n_sim = 1000;
    arma::mat sims = kr.simulate(n_sim, 222, X_close);
    
    // Compute empirical covariance
    arma::mat cov_sim = arma::cov(sims.t());
    
    INFO("Predicted covariance matrix:");
    INFO(cov_pred);
    INFO("Simulated covariance matrix:");
    INFO(cov_sim);
    
    // Check diagonal elements (variances)
    CHECK(std::abs(cov_pred(0,0) - cov_sim(0,0)) < 0.2);
    CHECK(std::abs(cov_pred(1,1) - cov_sim(1,1)) < 0.2);
    
    // Check off-diagonal element (covariance) - should be positive and large
    INFO("Predicted correlation: " << cov_pred(0,1) / std::sqrt(cov_pred(0,0) * cov_pred(1,1)));
    INFO("Simulated correlation: " << cov_sim(0,1) / std::sqrt(cov_sim(0,0) * cov_sim(1,1)));
    CHECK(cov_sim(0,1) > 0);  // Close points should be positively correlated
  }
}
