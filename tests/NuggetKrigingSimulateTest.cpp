// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/NuggetKriging.hpp"
// clang-format on

TEST_CASE("NuggetKrigingSimulateTest - Check predict is consistent with simulate", "[simulate][nuggetkriging]") {
  arma::arma_rng::set_seed(123);

  // Generate test data with nugget
  const arma::uword n = 25;
  const arma::uword d = 2;
  arma::mat X(n, d, arma::fill::randu);
  arma::colvec y(n);
  
  // Use a simple function with noise
  for (arma::uword i = 0; i < n; ++i) {
    double x1 = X(i, 0);
    double x2 = X(i, 1);
    y(i) = std::sin(3.0 * x1) + std::cos(5.0 * x2) + 0.1 * arma::randn();
  }

  // Fit NuggetKriging model
  NuggetKriging nk("gauss");
  NuggetKriging::Parameters params{std::nullopt, true, std::nullopt, true, std::nullopt, true, std::nullopt, true};
  nk.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);

  // Test points for prediction/simulation
  const arma::uword n_test = 10;
  arma::mat X_test(n_test, d, arma::fill::randu);

  SECTION("Empirical mean from simulations matches prediction mean") {
    // Get analytical prediction
    auto [mean_pred, stdev_pred, cov, md, sd] = nk.predict(X_test, true, false, false);
    
    // Run many simulations
    const int n_sim = 1000;
    arma::mat sims = nk.simulate(n_sim, 456, X_test, false);  // without nugget for comparison
    
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
    
    // Run many simulations WITHOUT nugget to compare with process stdev
    const int n_sim = 1000;
    arma::mat sims = nk.simulate(n_sim, 789, X_test, false);  // without nugget
    
    // Compute empirical standard deviation
    arma::vec stdev_sim = arma::stddev(sims, 0, 1);
    
    // Note: stdev_pred includes nugget, but simulations are without nugget
    // So we should compare with process stdev (excluding nugget)
    // The nugget adds variance: stdev_total^2 = stdev_process^2 + nugget
    double nugget_val = nk.nugget();
    arma::vec stdev_process = arma::sqrt(arma::square(stdev_pred) - nugget_val);
    
    const double tol = 0.15;
    
    for (arma::uword i = 0; i < n_test; ++i) {
      double diff = std::abs(stdev_process(i) - stdev_sim(i));
      
      INFO("Point " << i);
      INFO("Predicted total stdev: " << stdev_pred(i));
      INFO("Predicted process stdev: " << stdev_process(i));
      INFO("Simulated stdev (no nugget): " << stdev_sim(i));
      INFO("Nugget: " << nugget_val);
      INFO("Absolute error: " << diff);
      
      CHECK(diff < tol);
    }
  }

  SECTION("Simulations do NOT reproduce training data exactly (nugget effect)") {
    // Unlike standard Kriging, nugget kriging simulations at training points
    // should not exactly match training data (due to nugget/noise)
    const int n_sim = 500;
    arma::mat sims = nk.simulate(n_sim, 111, X, false);  // without nugget
    
    // Mean should be reasonably close but not exact
    arma::vec mean_sim = arma::mean(sims, 1);
    
    // Check that at least some points have non-trivial difference
    int n_different = 0;
    for (arma::uword i = 0; i < n; ++i) {
      double diff = std::abs(y(i) - mean_sim(i));
      if (diff > 0.05) {  // Threshold for "different"
        n_different++;
      }
    }
    
    INFO("Number of training points with |y - sim_mean| > 0.05: " << n_different);
    INFO("Nugget value: " << nk.nugget());
    
    // At least some points should show the nugget effect
    CHECK(n_different > 0);
  }
}
