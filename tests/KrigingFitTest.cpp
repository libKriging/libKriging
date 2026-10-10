// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
#include "libKriging/Optim.hpp"
// clang-format on

TEST_CASE("KrigingFitTest - BFGS finds better LL/LOO/LMP than grid search", "[fit][kriging]") {
  arma::arma_rng::set_seed(123);

  // Generate test data: simple 2D problem
  const arma::uword n = 20;
  const arma::uword d = 2;
  arma::mat X(n, d, arma::fill::randu);
  arma::colvec y(n);

  // Use Branin-like function
  for (arma::uword i = 0; i < n; ++i) {
    double x1 = X(i, 0);
    double x2 = X(i, 1);
    y(i) = (x2 - 0.5 * x1 * x1 + 5 * x1 - 6) * (x2 - 0.5 * x1 * x1 + 5 * x1 - 6) + 10 * (1 - 0.1) * std::cos(x1) + 10;
  }

  SECTION("LL - BFGS should find better or equal LL than grid search") {
    // First fit a Kriging model with arbitrary parameters to get the model structure
    Kriging kr_grid("gauss");
    Kriging::Parameters params{std::nullopt, true, std::nullopt, false, std::nullopt, true};
    kr_grid.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);

    // Grid search over theta using logLikelihoodFun
    const arma::uword grid_size = 10;
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
        auto [ll, grad] = kr_grid.logLikelihoodFun(theta, false, false);
        if (ll > best_ll_grid) {
          best_ll_grid = ll;
          best_theta_grid = theta.t();
        }
      }
    }

    INFO("Best grid LL: " << best_ll_grid << " at theta = " << best_theta_grid);

    // BFGS optimization with multiple starts for robustness across BLAS implementations (MKL vs OpenBLAS)
    Kriging kr_bfgs("gauss");
    Kriging::Parameters params_bfgs{std::nullopt, true, std::nullopt, true, std::nullopt, true};
    kr_bfgs.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS10", "LL", params_bfgs);

    double ll_bfgs = kr_bfgs.logLikelihood();
    arma::vec theta_bfgs = kr_bfgs.theta();

    INFO("BFGS LL: " << ll_bfgs << " at theta = " << theta_bfgs.t());

    // BFGS should find at least as good solution as grid search
    CHECK(ll_bfgs >= best_ll_grid - std::abs(best_ll_grid) * 1e-3);
  }

  SECTION("LOO - BFGS should find better or equal LOO than grid search") {
    // First fit a Kriging model
    Kriging kr_grid("gauss");
    Kriging::Parameters params{std::nullopt, true, std::nullopt, false, std::nullopt, true};
    kr_grid.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LOO", params);

    // Grid search over theta using leaveOneOutFun
    const arma::uword grid_size = 10;  // Smaller grid as LOO is more expensive
    double theta_min = 1e-6;
    double theta_max = 10.0;

    double best_loo_grid = arma::datum::inf;  // LOO is minimized
    arma::rowvec best_theta_grid(d);

    INFO("Performing dense grid search for LOO (evaluating " << grid_size * grid_size << " points)...");
    for (arma::uword i1 = 0; i1 < grid_size; ++i1) {
      for (arma::uword i2 = 0; i2 < grid_size; ++i2) {
        arma::vec theta(d);
        theta(0) = theta_min + i1 * (theta_max - theta_min) / (grid_size - 1);
        theta(1) = theta_min + i2 * (theta_max - theta_min) / (grid_size - 1);

        // Evaluate LOO at this theta
        auto [loo, grad] = kr_grid.leaveOneOutFun(theta, false, false);
        if (loo < best_loo_grid) {
          best_loo_grid = loo;
          best_theta_grid = theta.t();
        }
      }
    }

    INFO("Best grid LOO: " << best_loo_grid << " at theta = " << best_theta_grid);

    Optim::set_log_level(10);

    // BFGS optimization
    Kriging kr_bfgs("gauss");
    Kriging::Parameters params_bfgs{std::nullopt, true, std::nullopt, true, std::nullopt, true};
    kr_bfgs.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS10", "LOO", params_bfgs);

    Optim::set_log_level(0);

    double loo_bfgs = kr_bfgs.leaveOneOut();
    arma::vec theta_bfgs = kr_bfgs.theta();

    INFO("BFGS LOO: " << loo_bfgs << " at theta = " << theta_bfgs.t());

    // BFGS should find at least as good solution (lower LOO is better)
    CHECK(loo_bfgs <= best_loo_grid + 1e-3);
  }

  SECTION("LMP - BFGS should find better or equal LMP than grid search") {
    // First fit a Kriging model
    Kriging kr_grid("gauss");
    Kriging::Parameters params{std::nullopt, true, std::nullopt, false, std::nullopt, true};
    kr_grid.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LMP", params);

    // Grid search over theta using logMargPostFun
    const arma::uword grid_size = 10;
    double theta_min = 1e-6;
    double theta_max = 10.0;

    double best_lmp_grid = -arma::datum::inf;  // LMP is maximized
    arma::rowvec best_theta_grid(d);

    INFO("Performing dense grid search for LMP (evaluating " << grid_size * grid_size << " points)...");
    for (arma::uword i1 = 0; i1 < grid_size; ++i1) {
      for (arma::uword i2 = 0; i2 < grid_size; ++i2) {
        arma::vec theta(d);
        theta(0) = theta_min + i1 * (theta_max - theta_min) / (grid_size - 1);
        theta(1) = theta_min + i2 * (theta_max - theta_min) / (grid_size - 1);

        // Evaluate LMP at this theta
        auto [lmp, grad] = kr_grid.logMargPostFun(theta, false, false);
        if (lmp > best_lmp_grid) {
          best_lmp_grid = lmp;
          best_theta_grid = theta.t();
        }
      }
    }

    INFO("Best grid LMP: " << best_lmp_grid << " at theta = " << best_theta_grid);

    // BFGS optimization
    Kriging kr_bfgs("gauss");
    Kriging::Parameters params_bfgs{std::nullopt, true, std::nullopt, true, std::nullopt, true};
    kr_bfgs.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS10", "LMP", params_bfgs);

    double lmp_bfgs = kr_bfgs.logMargPost();
    arma::vec theta_bfgs = kr_bfgs.theta();

    INFO("BFGS LMP: " << lmp_bfgs << " at theta = " << theta_bfgs.t());

    // BFGS should find a competitive solution (within reasonable range of best grid)
    // Note: For some objectives/data, dense grid might find better local optima
    INFO("Difference: " << (best_lmp_grid - lmp_bfgs) << " (positive means grid is better)");
    CHECK(lmp_bfgs >= best_lmp_grid - std::abs(best_lmp_grid) * 1e-3);  // Relax tolerance for wider theta range
  }
}

// y in the span of the trend (constant y with a constant trend, affine y with a
// linear one): the residual is zero for every theta, sigma2 = 0 and theta is
// not identifiable. The fit used to drive theta to a bound where R is singular
// and fail ("All 1 optimization attempts failed"); it now commits the model
// without optimization: trend predictions, zero variance.
TEST_CASE("KrigingFitTest - y in the span of the trend", "[fit][kriging]") {
  const arma::vec x = arma::linspace(0.01, 0.99, 10);
  const arma::mat X = x;
  const arma::mat Xt = arma::linspace(0.05, 0.95, 4);

  auto check = [&](Kriging& k, const arma::vec& expected_mean) {
    CHECK_FALSE(k.is_theta_estim());
    CHECK(arma::all(k.theta() > 0));
    CHECK(std::abs(k.sigma2()) < 1e-20);
    auto [mean, stdev, cov, dm, ds] = k.predict(Xt, true, false, false);
    CHECK(arma::abs(mean - expected_mean).max() < 1e-8);
    CHECK(arma::abs(stdev).max() < 1e-8);
    const arma::mat sims = k.simulate(5, 1, Xt);
    CHECK(sims.is_finite());
    CHECK(arma::abs(sims.each_col() - expected_mean).max() < 1e-6);
  };

  const std::string objective = GENERATE(as<std::string>{}, "LL", "LOO", "LMP");
  const bool normalize = GENERATE(false, true);
  CAPTURE(objective, normalize);

  SECTION("constant y, constant trend") {
    const arma::vec y(10, arma::fill::value(3.0));
    Kriging k(y, X, "gauss", Trend::RegressionModel::Constant, normalize, "BFGS", objective);
    check(k, arma::vec(4, arma::fill::value(3.0)));
  }
  SECTION("affine y, linear trend") {
    Kriging k(1 + 2 * x, X, "matern5_2", Trend::RegressionModel::Linear, normalize, "BFGS", objective);
    check(k, 1 + 2 * arma::vec(Xt));
  }
  SECTION("given theta is kept") {
    Kriging::Parameters params;
    params.theta = arma::mat(1, 1, arma::fill::value(0.3));
    Kriging k(arma::vec(10, arma::fill::ones),
              X,
              "gauss",
              Trend::RegressionModel::Constant,
              normalize,
              "BFGS",
              objective,
              params);
    check(k, arma::vec(4, arma::fill::ones));
    if (!normalize)
      CHECK(k.theta()(0) == 0.3);
  }
}

TEST_CASE("KrigingFitTest - y in the span of the trend, noise models", "[fit][kriging]") {
  const arma::mat X = arma::linspace(0.01, 0.99, 10);
  const arma::vec y(10, arma::fill::ones);

  Kriging knug("gauss", Kriging::NoiseModel::Nugget);
  knug.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", {});
  CHECK(knug.sigma2() == 0.0);
  CHECK(knug.nugget() == 0.0);
  auto [mean, stdev, cov, dm, ds] = knug.predict(X.rows(0, 2), true, false, false);
  CHECK(arma::abs(mean - 1.0).max() < 1e-8);
  CHECK(stdev.is_finite());

  // known heterogeneous noise: sigma2 = 0 is not representable -> explicit error
  Kriging khet("gauss", Kriging::NoiseModel::Heterogeneous);
  CHECK_THROWS_AS(
      khet.fit(y, arma::vec(10, arma::fill::value(0.01)), X, Trend::RegressionModel::Constant, false, "BFGS", "LL", {}),
      std::invalid_argument);
}

TEST_CASE("KrigingFitTest - constant input column", "[fit][kriging]") {
  arma::arma_rng::set_seed(7);
  arma::mat X(20, 2, arma::fill::randu);
  X.col(1).fill(0.5);  // zero range: [0, 0] theta bounds, and a division by zero with normalize
  const arma::vec y = arma::sin(3 * X.col(0));
  const bool normalize = GENERATE(false, true);
  CAPTURE(normalize);
  Kriging k(y, X, "gauss", Trend::RegressionModel::Constant, normalize, "BFGS", "LL");
  CHECK(k.theta().is_finite());
  CHECK(k.theta()(1) == 1.0);  // not identifiable: pinned, not driven out of its bounds
  auto [mean, stdev, cov, dm, ds] = k.predict(X.rows(0, 4), true, false, false);
  CHECK(arma::abs(mean - y.head(5)).max() < 1e-4);
}
