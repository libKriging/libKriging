// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
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
    double y_true
        = (x2 - 0.5 * x1 * x1 + 5 * x1 - 6) * (x2 - 0.5 * x1 * x1 + 5 * x1 - 6) + 10 * (1 - 0.1) * std::cos(x1) + 10;
    y(i) = y_true + std::sqrt(true_nugget) * arma::randn();
  }

  SECTION("LL - BFGS should find better or equal LL than grid search") {
    // First fit a NuggetKriging model
    Kriging nk_grid("gauss", Kriging::NoiseModel::Nugget);
    Kriging::Parameters params{std::nullopt, true, std::nullopt, true, std::nullopt, true, std::nullopt, true};
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

    INFO("Performing dense grid search for LL (evaluating " << grid_size_theta * grid_size_theta * grid_size_alpha
                                                            << " points)...");
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
    Kriging nk_bfgs("gauss", Kriging::NoiseModel::Nugget);
    Kriging::Parameters params_bfgs{std::nullopt, true, std::nullopt, true, std::nullopt, true, std::nullopt, true};
    nk_bfgs.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", params_bfgs);

    double ll_bfgs = nk_bfgs.logLikelihood();
    arma::vec theta_bfgs = nk_bfgs.theta();

    INFO("BFGS LL: " << ll_bfgs << " at theta = " << theta_bfgs.t());
    INFO("BFGS nugget: " << nk_bfgs.nugget());

    // BFGS should find at least as good solution as grid search
    CHECK(ll_bfgs >= best_ll_grid - std::abs(best_ll_grid) * 1e-3);
  }

  SECTION("LMP - BFGS should find better or equal LMP than grid search") {
    // First fit a NuggetKriging model
    Kriging nk_grid("gauss", Kriging::NoiseModel::Nugget);
    Kriging::Parameters params{std::nullopt, true, std::nullopt, false, std::nullopt, true, std::nullopt, true};
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

    INFO("Performing dense grid search for LMP (evaluating " << grid_size_theta * grid_size_theta * grid_size_alpha
                                                             << " points)...");
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
    Kriging nk_bfgs("gauss", Kriging::NoiseModel::Nugget);
    Kriging::Parameters params_bfgs{std::nullopt, true, std::nullopt, true, std::nullopt, true, std::nullopt, true};
    nk_bfgs.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LMP", params_bfgs);

    double lmp_bfgs = nk_bfgs.logMargPost();
    arma::vec theta_bfgs = nk_bfgs.theta();

    INFO("BFGS LMP: " << lmp_bfgs << " at theta = " << theta_bfgs.t());
    INFO("BFGS nugget: " << nk_bfgs.nugget());

    // BFGS should find at least as good solution
    CHECK(lmp_bfgs >= best_lmp_grid - std::abs(best_lmp_grid) * 1e-3);
  }
}

TEST_CASE("Nugget single-start BFGS is close to the multistart optimum (#383)", "[nugget][fit]") {
  // Branin, n = 30, noise sd = 5% of sd(y): a random single start used to
  // end in the all-noise region (small theta, nugget >> sigma2)
  auto branin = [](const arma::mat& U) -> arma::vec {
    const arma::vec x1 = 15 * U.col(0) - 5;
    const arma::vec x2 = 15 * U.col(1);
    const double pi = arma::datum::pi;  // (M_PI is not standard: undefined with MSVC)
    return arma::square(x2 - 5.1 / (4 * pi * pi) * arma::square(x1) + 5 / pi * x1 - 6)
           + 10 * (1 - 1 / (8 * pi)) * arma::cos(x1) + 10;
  };
  const arma::uword n = 30;
  const int n_rep = 10;
  double gap_sum = 0;
  for (int rep = 0; rep < n_rep; rep++) {
    arma::arma_rng::set_seed(1000 + rep);
    arma::mat X(n, 2);
    for (arma::uword j = 0; j < 2; j++)
      X.col(j) = (arma::conv_to<arma::vec>::from(arma::randperm(n)) + arma::randu<arma::vec>(n)) / n;
    arma::vec y = branin(X);
    y += 0.05 * arma::stddev(y) * arma::randn<arma::vec>(n);

    Kriging k1("matern5_2", Kriging::NoiseModel::Nugget);
    k1.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", {});
    Kriging k10("matern5_2", Kriging::NoiseModel::Nugget);
    k10.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS10", "LL", {});
    const double gap = k10.logLikelihood() - k1.logLikelihood();
    INFO("rep " << rep << ": LL(BFGS) = " << k1.logLikelihood() << ", LL(BFGS10) = " << k10.logLikelihood()
                << ", nugget = " << k1.nugget() << ", sigma2 = " << k1.sigma2());
    CHECK(gap < 5);                    // never the all-noise region (gaps of 10 to 40 before)
    CHECK(k1.nugget() < k1.sigma2());  // signal-dominated model
    gap_sum += std::max(gap, 0.0);
  }
  CHECK(gap_sum / n_rep < 1.5);  // mean gap was ~8 with a random single start
}
