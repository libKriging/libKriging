// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/NoiseKriging.hpp"
// clang-format on

TEST_CASE("NoiseKrigingFitTest - BFGS finds better LL than grid search", "[fit][noisekriging]") {
  arma::arma_rng::set_seed(123);

  // Generate test data with noise
  const arma::uword n = 25;
  const arma::uword d = 2;
  arma::mat X(n, d, arma::fill::randu);
  arma::colvec y(n);
  arma::colvec noise(n);
  
  // Use Branin-like function with added noise
  for (arma::uword i = 0; i < n; ++i) {
    double x1 = X(i, 0);
    double x2 = X(i, 1);
    double y_true = (x2 - 0.5 * x1 * x1 + 5 * x1 - 6) * (x2 - 0.5 * x1 * x1 + 5 * x1 - 6)
                    + 10 * (1 - 0.1) * std::cos(x1) + 10;
    noise(i) = 0.01 + 0.02 * arma::randu();
    y(i) = y_true + std::sqrt(noise(i)) * arma::randn();
  }

  SECTION("LL - BFGS should find better or equal LL than grid search") {
    // First fit a NoiseKriging model
    NoiseKriging nk_grid("gauss");
    NoiseKriging::Parameters params{std::nullopt, true, std::nullopt, false, std::nullopt, true};
    nk_grid.fit(y, noise, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);
    
    // Grid search over theta using logLikelihoodFun
    const arma::uword grid_size = 8;
    double theta_min = 1e-6;
    double theta_max = 10.0;
    
    double best_ll_grid = -arma::datum::inf;
    arma::rowvec best_theta_grid(d);
    
    INFO("Performing dense grid search for LL (evaluating " << grid_size * grid_size << " points)...");
    for (arma::uword i1 = 0; i1 < grid_size; ++i1) {
      for (arma::uword i2 = 0; i2 < grid_size; ++i2) {
        arma::vec theta(d);
        theta(0) = theta_min + i1 * (theta_max - theta_min) / (grid_size - 1);
        theta(1) = theta_min + i2 * (theta_max - theta_min) / (grid_size - 1);
        
        // Evaluate LL at this theta
        auto [ll, grad] = nk_grid.logLikelihoodFun(theta, false, false);
        if (ll > best_ll_grid) {
          best_ll_grid = ll;
          best_theta_grid = theta.t();
        }
      }
    }
    
    INFO("Best grid LL: " << best_ll_grid << " at theta = " << best_theta_grid);
    
    // BFGS optimization
    NoiseKriging nk_bfgs("gauss");
    NoiseKriging::Parameters params_bfgs{std::nullopt, true, std::nullopt, true, std::nullopt, true};
    nk_bfgs.fit(y, noise, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", params_bfgs);
    
    double ll_bfgs = nk_bfgs.logLikelihood();
    arma::vec theta_bfgs = nk_bfgs.theta();
    
    INFO("BFGS LL: " << ll_bfgs << " at theta = " << theta_bfgs.t());
    
    // BFGS should find at least as good solution as grid search
    CHECK(ll_bfgs >= best_ll_grid - 1e-3);
  }
}
