// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/WarpKriging.hpp"
#include "ks_test.hpp"
#include <cmath>
#include <sstream>
#include <string>
#include <vector>
// clang-format on

/**
 * @file WarpKrigingPerWarpTest.cpp
 * @brief Per-warping regression tests on 1D data for WarpKriging.
 *
 * Three test kinds are exercised for each continuous warping type
 * (none, affine, boxcox, kumaraswamy, knots(3), neural_mono, mlp):
 *
 *  1. predict vs simulate
 *     Empirical mean and standard deviation from N=1000 conditional
 *     simulations must be close to the analytical predict() values.
 *
 *  2. predict derivative vs finite differences
 *     mean_deriv and stdev_deriv returned by predict(..., return_deriv=true)
 *     must match central finite differences to within a mixed
 *     absolute/relative tolerance.
 *
 *  3. update+simulate vs update_simulate
 *     simulate(will_update=true) followed by update_simulate(y_u, X_u)
 *     must produce a distribution statistically indistinguishable (KS test)
 *     from update(y_u, X_u) followed by simulate() on the same seed.
 */

using namespace libKriging;

// ---------------------------------------------------------------------------
//  Shared 1-D test function
// ---------------------------------------------------------------------------
static double f1d(double x) {
  return std::sin(6.0 * x) + 0.5 * x;
}

// ---------------------------------------------------------------------------
//  Warp types under test (continuous only)
// ---------------------------------------------------------------------------
static const std::vector<std::string> WARP_SPECS = {
    "none",
    "affine",
    "boxcox",
    "kumaraswamy",
    "knots(3)",
    "neural_mono",
    "mlp(8,1,selu)",
};

// ---------------------------------------------------------------------------
//  Helper: build a fitted 1-D WarpKriging model
// ---------------------------------------------------------------------------
static WarpKriging make_model(const std::string& warp_spec, arma::uword n = 15, int seed = 42) {
  arma::arma_rng::set_seed(static_cast<arma::arma_rng::seed_type>(seed));
  arma::vec x = arma::linspace(0.05, 0.95, n);
  arma::mat X(n, 1);
  X.col(0) = x;
  arma::vec y(n);
  for (arma::uword i = 0; i < n; ++i)
    y(i) = f1d(x(i));

  WarpKriging wk({warp_spec}, "matern5_2");
  wk.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS+Adam", "LL",
         {{"max_iter_adam", "200"}});
  return wk;
}

// ==========================================================================
//  Test 1: predict vs simulate
// ==========================================================================

/// At each of m test points, empirical {mean, stdev} from `nsim` simulations
/// must lie within `n_sigma` standard-error units of the analytical predict.
static void check_predict_vs_simulate(const std::string& warp_spec) {
  auto wk = make_model(warp_spec);

  // Test grid inside training range
  arma::uword m = 6;
  arma::mat X_test(m, 1);
  X_test.col(0) = arma::linspace(0.1, 0.9, m);

  auto [pred_mean, pred_stdev, _cov, _md, _sd] = wk.predict(X_test, true, false);

  const int nsim = 2000;
  arma::mat sims = wk.simulate(nsim, /*seed=*/7, X_test);

  REQUIRE(sims.n_rows == m);
  REQUIRE(sims.n_cols == static_cast<arma::uword>(nsim));
  REQUIRE(sims.is_finite());

  // Tolerance: 4 standard errors for mean, 25 % relative for stdev
  const double n_sigma_mean = 4.0;
  const double rel_stdev_tol = 0.25;

  int mean_failures = 0;
  int stdev_failures = 0;
  std::stringstream details;

  for (arma::uword i = 0; i < m; ++i) {
    arma::rowvec s = sims.row(i);
    double emp_mean = arma::mean(s);
    double emp_stdev = arma::stddev(s);

    double se_mean = pred_stdev(i) / std::sqrt(static_cast<double>(nsim));
    double mean_tol = n_sigma_mean * se_mean;
    double stdev_tol = rel_stdev_tol * pred_stdev(i);

    double mean_err = std::abs(emp_mean - pred_mean(i));
    double stdev_err = std::abs(emp_stdev - pred_stdev(i));

    if (mean_err > mean_tol) {
      details << "\n  point " << i << ": emp_mean=" << emp_mean << " pred_mean=" << pred_mean(i)
              << " err=" << mean_err << " tol=" << mean_tol;
      ++mean_failures;
    }
    if (stdev_err > stdev_tol) {
      details << "\n  point " << i << ": emp_stdev=" << emp_stdev
              << " pred_stdev=" << pred_stdev(i) << " err=" << stdev_err << " tol=" << stdev_tol;
      ++stdev_failures;
    }
  }

  INFO("warp=" << warp_spec << " mean_failures=" << mean_failures << details.str());
  CHECK(mean_failures == 0);

  INFO("warp=" << warp_spec << " stdev_failures=" << stdev_failures << details.str());
  CHECK(stdev_failures == 0);
}

TEST_CASE("WarpKrigingPerWarpTest - predict vs simulate (1D)", "[predict][simulate][warpkriging]") {
  const int idx = GENERATE_COPY(range(0, static_cast<int>(WARP_SPECS.size())));
  const std::string& warp = WARP_SPECS[static_cast<std::size_t>(idx)];
  INFO("warp=" << warp);
  check_predict_vs_simulate(warp);
}

// ==========================================================================
//  Test 2: predict derivative vs finite differences
// ==========================================================================

static void check_predict_deriv_vs_fd(const std::string& warp_spec) {
  auto wk = make_model(warp_spec);

  arma::uword m = 5;
  arma::mat X_test(m, 1);
  X_test.col(0) = arma::linspace(0.15, 0.85, m);

  auto [mean, stdev, _cov, mean_deriv, stdev_deriv] = wk.predict(X_test, true, false, true);

  REQUIRE(mean_deriv.n_rows == m);
  REQUIRE(mean_deriv.n_cols == 1u);
  REQUIRE(stdev_deriv.n_rows == m);
  REQUIRE(stdev_deriv.n_cols == 1u);
  REQUIRE(mean_deriv.is_finite());
  REQUIRE(stdev_deriv.is_finite());

  const double h = 1e-5;
  // Mixed tolerance: max(abs_floor, rel * |value|)
  const double abs_floor = 0.02;
  const double rel_tol = 0.10;

  int mean_failures = 0, stdev_failures = 0;
  std::stringstream details;

  for (arma::uword i = 0; i < m; ++i) {
    arma::mat Xp = X_test, Xm = X_test;
    Xp(i, 0) += h;
    Xm(i, 0) -= h;

    auto [mp, sp, cp, dp, dsp] = wk.predict(Xp, true, false, false);
    auto [mm, sm, cm, dm, dsm] = wk.predict(Xm, true, false, false);

    double fd_mean = (mp(i) - mm(i)) / (2.0 * h);
    double fd_stdev = (sp(i) - sm(i)) / (2.0 * h);

    double tol_mean = std::max(abs_floor, rel_tol * std::max(std::abs(fd_mean), std::abs(mean_deriv(i, 0))));
    double tol_stdev = std::max(abs_floor, rel_tol * std::max(std::abs(fd_stdev), std::abs(stdev_deriv(i, 0))));

    double err_mean = std::abs(mean_deriv(i, 0) - fd_mean);
    double err_stdev = std::abs(stdev_deriv(i, 0) - fd_stdev);

    if (err_mean > tol_mean) {
      details << "\n  point " << i << " mean_deriv=" << mean_deriv(i, 0) << " fd=" << fd_mean
              << " err=" << err_mean << " tol=" << tol_mean;
      ++mean_failures;
    }
    if (err_stdev > tol_stdev) {
      details << "\n  point " << i << " stdev_deriv=" << stdev_deriv(i, 0) << " fd=" << fd_stdev
              << " err=" << err_stdev << " tol=" << tol_stdev;
      ++stdev_failures;
    }
  }

  INFO("warp=" << warp_spec << " mean_deriv failures=" << mean_failures << details.str());
  CHECK(mean_failures == 0);

  INFO("warp=" << warp_spec << " stdev_deriv failures=" << stdev_failures << details.str());
  CHECK(stdev_failures == 0);
}

TEST_CASE("WarpKrigingPerWarpTest - predict derivative vs FD (1D)",
          "[predict][derivative][fd][warpkriging]") {
  const int idx = GENERATE_COPY(range(0, static_cast<int>(WARP_SPECS.size())));
  const std::string& warp = WARP_SPECS[static_cast<std::size_t>(idx)];
  INFO("warp=" << warp);
  check_predict_deriv_vs_fd(warp);
}

// ==========================================================================
//  Test 3: update+simulate vs update_simulate
// ==========================================================================

static void check_update_simulate(const std::string& warp_spec) {
  arma::arma_rng::set_seed(99);

  const arma::uword n_old = 12;
  const arma::uword n_new = 3;

  arma::vec x_old = arma::linspace(0.05, 0.95, n_old);
  arma::mat X_old(n_old, 1);
  X_old.col(0) = x_old;
  arma::vec y_old(n_old);
  for (arma::uword i = 0; i < n_old; ++i)
    y_old(i) = f1d(x_old(i));

  arma::vec x_new = {0.2, 0.5, 0.8};
  arma::mat X_new(n_new, 1);
  X_new.col(0) = x_new;
  arma::vec y_new(n_new);
  for (arma::uword i = 0; i < n_new; ++i)
    y_new(i) = f1d(x_new(i));

  // Simulation points (different from training)
  arma::uword m = 8;
  arma::mat X_sim(m, 1);
  X_sim.col(0) = arma::linspace(0.1, 0.9, m);

  const int nsim = 1000;
  const int seed = 314;

  // --- Method A: simulate with will_update=true, then update_simulate ---
  WarpKriging wk_a({warp_spec}, "matern5_2");
  wk_a.fit(y_old, X_old, Trend::RegressionModel::Constant, false, "BFGS+Adam", "LL",
            {{"max_iter_adam", "200"}});
  wk_a.simulate(nsim, seed, X_sim, /*will_update=*/true);
  arma::mat sims_a = wk_a.update_simulate(y_new, X_new);

  // --- Method B: update WITHOUT refit (same hyperparams as A), then simulate ---
  WarpKriging wk_b({warp_spec}, "matern5_2");
  wk_b.fit(y_old, X_old, Trend::RegressionModel::Constant, false, "BFGS+Adam", "LL",
            {{"max_iter_adam", "200"}});
  wk_b.update(y_new, X_new, /*refit=*/false);
  arma::mat sims_b = wk_b.simulate(nsim, seed, X_sim);

  REQUIRE(sims_a.n_rows == m);
  REQUIRE(sims_a.n_cols == static_cast<arma::uword>(nsim));
  REQUIRE(sims_a.is_finite());

  // KS test at each simulation point
  int ks_failures = 0;
  std::stringstream details;
  for (arma::uword i = 0; i < m; ++i) {
    arma::rowvec sa = sims_a.row(i);
    arma::rowvec sb = sims_b.row(i);
    auto [passed, pvalue] = KSTest::ks_test_with_pvalue(sa, sb, /*alpha=*/1e-6);
    if (!passed) {
      details << "\n  point " << i << " p-value=" << pvalue;
      ++ks_failures;
    }
  }

  INFO("warp=" << warp_spec << " KS failures=" << ks_failures << "/" << m << details.str());
  // Allow at most 1 marginal failure out of m (random seed sensitivity)
  CHECK(ks_failures <= 1);
}

TEST_CASE("WarpKrigingPerWarpTest - update+simulate vs update_simulate (1D)",
          "[update][simulate][update_simulate][warpkriging]") {
  const int idx = GENERATE_COPY(range(0, static_cast<int>(WARP_SPECS.size())));
  const std::string& warp = WARP_SPECS[static_cast<std::size_t>(idx)];
  INFO("warp=" << warp);
  check_update_simulate(warp);
}

// ==========================================================================
//  Test 4: log-likelihood gradient vs finite differences
// ==========================================================================

/// After fitting, evaluate logLikelihoodFun at a perturbed theta and verify
/// that the analytical gradient matches central finite differences for every
/// component of log(theta).
static void check_loglik_grad_vs_fd(const std::string& warp_spec) {
  auto wk = make_model(warp_spec);

  // Use the fitted theta as the evaluation point (gradient is non-trivial there)
  arma::vec theta0 = wk.theta();
  REQUIRE(theta0.n_elem >= 1);
  REQUIRE(theta0.is_finite());

  // Compute analytical gradient
  auto [ll0, grad, hess] = wk.logLikelihoodFun(theta0, /*return_grad=*/true, false);
  REQUIRE(grad.n_elem == theta0.n_elem);
  REQUIRE(grad.is_finite());

  const double h = 1e-5;
  // Mixed tolerance: max(abs_floor, rel * |value|)
  const double abs_floor = 1e-3;
  const double rel_tol = 0.01;  // 1 % relative

  int failures = 0;
  std::stringstream details;

  for (arma::uword k = 0; k < theta0.n_elem; ++k) {
    arma::vec tp = theta0, tm = theta0;
    tp(k) += h;
    tm(k) -= h;

    double llp = std::get<0>(wk.logLikelihoodFun(tp, false, false));
    double llm = std::get<0>(wk.logLikelihoodFun(tm, false, false));
    double fd = (llp - llm) / (2.0 * h);

    double tol = std::max(abs_floor, rel_tol * std::max(std::abs(fd), std::abs(grad(k))));
    double err = std::abs(grad(k) - fd);

    if (err > tol) {
      details << "\n  theta[" << k << "]=" << theta0(k) << " analytic=" << grad(k) << " fd=" << fd
              << " err=" << err << " tol=" << tol;
      ++failures;
    }
  }

  INFO("warp=" << warp_spec << " loglik_grad failures=" << failures << details.str());
  CHECK(failures == 0);
}

TEST_CASE("WarpKrigingPerWarpTest - loglik gradient vs FD (1D)",
          "[loglik][gradient][fd][warpkriging]") {
  const int idx = GENERATE_COPY(range(0, static_cast<int>(WARP_SPECS.size())));
  const std::string& warp = WARP_SPECS[static_cast<std::size_t>(idx)];
  INFO("warp=" << warp);
  check_loglik_grad_vs_fd(warp);
}
