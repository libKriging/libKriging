// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/NuggetKriging.hpp"
// clang-format on

TEST_CASE("NuggetKrigingFitTest - BFGS finds better LL/LMP than grid search", "[fit][nuggetkriging]") {
  arma::arma_rng::set_seed(123);

  // Generate test data with nugget effect
  const arma::uword n = 25;
  const arma::uword d = 2;
  arma::mat X(n, d, arma::fill::randu);
  arma::colvec y(n);
  
  // Use Branin-like function with added homoscedastic noise
  double true_nugget = 0.1;
  for (arma::uword i = 0; i < n; ++i) {
    double x1 = X(i, 0);
    double x2 = X(i, 1);
    double y_true = (x2 - 0.5 * x1 * x1 + 5 * x1 - 6) * (x2 - 0.5 * x1 * x1 + 5 * x1 - 6)
                    + 10 * (1 - 0.1) * std::cos(x1) + 10;
    y(i) = y_true + std::sqrt(true_nugget) * arma::randn();
  }

  SECTION("LL - BFGS should find better or equal LL than grid search") {
    // First fit a NuggetKriging model
    NuggetKriging nk_grid("gauss");
    NuggetKriging::Parameters params{std::nullopt, true, std::nullopt, true, std::nullopt, true, std::nullopt, true};
    nk_grid.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);
    
    // Grid search over theta_alpha = [theta(1), theta(2), alpha]
    const arma::uword grid_size_theta = 6;
    const arma::uword grid_size_alpha = 5;
    double theta_min = 1e-6;
    double theta_max = 10.0;
    double alpha_min = 0.1;  // alpha in (0, 1), avoid extremes
    double alpha_max = 0.9;
    
    double best_ll_grid = -arma::datum::inf;
    arma::vec best_theta_alpha_grid(d + 1);
    
    INFO("Performing dense grid search for LL (evaluating " << grid_size_theta * grid_size_theta * grid_size_alpha << " points)...");
    for (arma::uword i1 = 0; i1 < grid_size_theta; ++i1) {
      for (arma::uword i2 = 0; i2 < grid_size_theta; ++i2) {
        for (arma::uword ia = 0; ia < grid_size_alpha; ++ia) {
          // logLikelihoodFun expects [theta(1), ..., theta(d), alpha]
          arma::vec theta_alpha(d + 1);
          theta_alpha(0) = theta_min + i1 * (theta_max - theta_min) / (grid_size_theta - 1);
          theta_alpha(1) = theta_min + i2 * (theta_max - theta_min) / (grid_size_theta - 1);
          theta_alpha(d) = alpha_min + ia * (alpha_max - alpha_min) / (grid_size_alpha - 1);
          
          // Evaluate LL at this theta_alpha
          auto [ll, grad] = nk_grid.logLikelihoodFun(theta_alpha, false, false);
          if (ll > best_ll_grid) {
            best_ll_grid = ll;
            best_theta_alpha_grid = theta_alpha;
          }
        }
      }
    }
    
    INFO("Best grid LL: " << best_ll_grid << " at theta_alpha = " << best_theta_alpha_grid.t());
    
    // BFGS optimization
    NuggetKriging nk_bfgs("gauss");
    NuggetKriging::Parameters params_bfgs{std::nullopt, true, std::nullopt, true, std::nullopt, true, std::nullopt, true};
    nk_bfgs.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", params_bfgs);
    
    double ll_bfgs = nk_bfgs.logLikelihood();
    arma::vec theta_bfgs = nk_bfgs.theta();
    
    INFO("BFGS LL: " << ll_bfgs << " at theta = " << theta_bfgs.t());
    INFO("BFGS nugget: " << nk_bfgs.nugget());
    
    // BFGS should find at least as good solution as grid search
    CHECK(ll_bfgs >= best_ll_grid - 1e-3);
  }

  SECTION("LMP - BFGS should find better or equal LMP than grid search") {
    // First fit a NuggetKriging model
    NuggetKriging nk_grid("gauss");
    NuggetKriging::Parameters params{std::nullopt, true, std::nullopt, false, std::nullopt, true, std::nullopt, true};
    nk_grid.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LMP", params);
    
    // Grid search over theta_alpha = [theta(1), theta(2), alpha]
    const arma::uword grid_size_theta = 5;
    const arma::uword grid_size_alpha = 4;
    double theta_min = 1e-6;
    double theta_max = 10.0;
    double alpha_min = 0.1;  // alpha in (0, 1), avoid extremes
    double alpha_max = 0.9;
    
    double best_lmp_grid = -arma::datum::inf;  // LMP is maximized
    arma::vec best_theta_alpha_grid(d + 1);
    
    INFO("Performing dense grid search for LMP (evaluating " << grid_size_theta * grid_size_theta * grid_size_alpha << " points)...");
    for (arma::uword i1 = 0; i1 < grid_size_theta; ++i1) {
      for (arma::uword i2 = 0; i2 < grid_size_theta; ++i2) {
        for (arma::uword ia = 0; ia < grid_size_alpha; ++ia) {
          arma::vec theta_alpha(d + 1);
          theta_alpha(0) = theta_min + i1 * (theta_max - theta_min) / (grid_size_theta - 1);
          theta_alpha(1) = theta_min + i2 * (theta_max - theta_min) / (grid_size_theta - 1);
          theta_alpha(d) = alpha_min + ia * (alpha_max - alpha_min) / (grid_size_alpha - 1);
          
          // Evaluate LMP at this theta_alpha
          auto [lmp, grad] = nk_grid.logMargPostFun(theta_alpha, false, false);
          if (lmp > best_lmp_grid) {
            best_lmp_grid = lmp;
            best_theta_alpha_grid = theta_alpha;
          }
        }
      }
    }
    
    INFO("Best grid LMP: " << best_lmp_grid << " at theta_alpha = " << best_theta_alpha_grid.t());
    
    // BFGS optimization
    NuggetKriging nk_bfgs("gauss");
    NuggetKriging::Parameters params_bfgs{std::nullopt, true, std::nullopt, true, std::nullopt, true, std::nullopt, true};
    nk_bfgs.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LMP", params_bfgs);
    
    double lmp_bfgs = nk_bfgs.logMargPost();
    arma::vec theta_bfgs = nk_bfgs.theta();
    
    INFO("BFGS LMP: " << lmp_bfgs << " at theta = " << theta_bfgs.t());
    INFO("BFGS nugget: " << nk_bfgs.nugget());
    
    // BFGS should find at least as good solution
    CHECK(lmp_bfgs >= best_lmp_grid - 1e-3);
  }
}
