#define _USE_MATH_DEFINES  // required for Visual Studio
/**
 * @file MLPKrigingTest.cpp
 * @brief Tests for the MLPKriging class (joint-MLP Deep Kernel Learning).
 */

#include "libKriging/MLPKriging.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace libKriging;

static double f1d(double x) {
  return 1.0 - 0.5 * (std::sin(12.0 * x) / (1.0 + x) + 2.0 * std::cos(7.0 * x) * std::pow(x, 5) + 0.7);
}

static double f2(double x1, double x2) {
  return std::sin(3.0 * x1) + std::cos(4.0 * x2) + 0.5 * x1 * x2;
}

// ==========================================================================
//  Test 1: 1D + 2D fit/predict/simulate/update
// ==========================================================================
static void test_mlp_kriging_basic() {
  std::cout << "=== Test 1: MLPKriging (Deep Kernel Learning) ===" << std::endl;

  // --- 1a: 1D function ---
  {
    arma::vec X_train = arma::linspace(0.01, 0.99, 10);
    arma::mat X_mat(X_train.n_elem, 1);
    X_mat.col(0) = X_train;
    arma::vec y(X_train.n_elem);
    for (arma::uword i = 0; i < X_train.n_elem; ++i)
      y(i) = f1d(X_train(i));

    MLPKriging model({16, 8}, 2, "selu", "gauss");
    model.fit(y, X_mat, "constant", true, "Adam", "LL", {{"max_iter_adam", "300"}});

    std::cout << model.summary();

    auto [mean_train, stdev_train, _c1b, _md1b, _sd1b] = model.predict(X_mat, true, false);
    double max_err = arma::max(arma::abs(mean_train - y));
    std::cout << "  [1D] Max train interp error:  " << max_err << std::endl;

    arma::vec x_pred = arma::linspace(0.01, 0.99, 50);
    arma::mat xp(50, 1);
    xp.col(0) = x_pred;
    auto [mean, stdev, _c2b, _md2b, _sd2b] = model.predict(xp, true, false);
    arma::vec ytrue(50);
    for (arma::uword i = 0; i < 50; ++i)
      ytrue(i) = f1d(x_pred(i));
    double rmse = std::sqrt(arma::mean(arma::square(mean - ytrue)));
    std::cout << "  [1D] RMSE on dense grid:      " << rmse << std::endl;
  }

  // --- 1b: Branin 2D ---
  {
    auto branin = [](double x1, double x2) -> double {
      double a = 1.0, b = 5.1 / (4.0 * M_PI * M_PI);
      double c = 5.0 / M_PI, r = 6.0, s = 10.0, t = 1.0 / (8.0 * M_PI);
      return a * std::pow(x2 - b * x1 * x1 + c * x1 - r, 2) + s * (1.0 - t) * std::cos(x1) + s;
    };

    arma::uword n = 20;
    arma::arma_rng::set_seed(77);
    arma::mat X = arma::randu<arma::mat>(n, 2);
    arma::vec y(n);
    for (arma::uword i = 0; i < n; ++i)
      y(i) = branin(X(i, 0) * 15.0 - 5.0, X(i, 1) * 15.0);

    MLPKriging model({32, 16}, 3, "selu", "matern5_2");
    model.fit(y, X, "constant", true, "Adam", "LL", {{"max_iter_adam", "300"}});

    std::cout << model.summary();

    arma::mat X_test = arma::randu<arma::mat>(10, 2);
    auto [mean, stdev, _c3b, _md3b, _sd3b] = model.predict(X_test, true, false);
    assert(mean.n_elem == 10);
    assert(stdev.n_elem == 10);
    assert(mean.is_finite());
    assert(stdev.is_finite());

    std::cout << "  [2D] Predicted mean range:  [" << arma::min(mean) << ", " << arma::max(mean) << "]" << std::endl;
    std::cout << "  [2D] Predicted stdev range: [" << arma::min(stdev) << ", " << arma::max(stdev) << "]" << std::endl;

    arma::mat sims = model.simulate(20, 42, X_test);
    assert(sims.n_rows == 10 && sims.n_cols == 20);
    assert(sims.is_finite());

    arma::mat X_new = arma::randu<arma::mat>(3, 2);
    arma::vec y_new(3);
    for (arma::uword i = 0; i < 3; ++i)
      y_new(i) = branin(X_new(i, 0) * 15.0 - 5.0, X_new(i, 1) * 15.0);
    model.update(y_new, X_new);
    std::cout << "  [2D] n after update: " << model.y().n_elem << std::endl;
    assert(model.y().n_elem == n + 3);
  }

  std::cout << "  PASSED\n" << std::endl;
}

// ==========================================================================
//  Test 2: Derivative check vs finite differences
// ==========================================================================
static void check_deriv_vs_fd(const MLPKriging& model,
                              const arma::mat& X_new,
                              const std::vector<arma::uword>& check_dims,
                              const std::string& label,
                              double h = 1e-6,
                              double rel_tol = 0.2,
                              double abs_tol = 0.1) {
  auto [mean, stdev, cov, mean_deriv, stdev_deriv] = model.predict(X_new, true, false, true);

  const arma::uword n_n = X_new.n_rows;
  for (arma::uword i = 0; i < n_n; ++i) {
    for (arma::uword dim : check_dims) {
      arma::mat Xp = X_new;
      arma::mat Xm = X_new;
      Xp(i, dim) += h;
      Xm(i, dim) -= h;
      auto [mp, sp, _cp, _mdp, _sdp] = model.predict(Xp, true, false);
      auto [mm, sm, _cm, _mdm, _sdm] = model.predict(Xm, true, false);

      double dmean_num = (mp(i) - mm(i)) / (2.0 * h);
      double dstdev_num = (sp(i) - sm(i)) / (2.0 * h);
      double dmean_ana = mean_deriv(i, dim);
      double dstdev_ana = stdev_deriv(i, dim);

      double mean_err = std::abs(dmean_num - dmean_ana) / (std::abs(dmean_num) + abs_tol);
      double stdev_err = std::abs(dstdev_num - dstdev_ana) / (std::abs(dstdev_num) + abs_tol);

      if (mean_err > rel_tol) {
        std::cerr << "  [" << label << "] mean deriv mismatch at i=" << i << " dim=" << dim << ": num=" << dmean_num
                  << " ana=" << dmean_ana << std::endl;
        assert(false);
      }
      if (stdev_err > rel_tol) {
        std::cerr << "  [" << label << "] stdev deriv mismatch at i=" << i << " dim=" << dim << ": num=" << dstdev_num
                  << " ana=" << dstdev_ana << std::endl;
        assert(false);
      }
    }
  }
  std::cout << "  [" << label << "] derivative check PASSED" << std::endl;
}

static void test_predict_derivative() {
  std::cout << "=== Test 2: MLPKriging predict derivative vs finite differences ===" << std::endl;

  arma::arma_rng::set_seed(88);
  const arma::uword n = 20;
  arma::mat X = arma::randu<arma::mat>(n, 2);
  arma::vec y(n);
  for (arma::uword i = 0; i < n; ++i)
    y(i) = f2(X(i, 0), X(i, 1));

  MLPKriging model({16, 8}, 2, "selu", "gauss");
  model.fit(y, X, "constant", false, "Adam", "LL", {{"max_iter_adam", "100"}});

  arma::mat X_new = arma::randu<arma::mat>(5, 2);
  check_deriv_vs_fd(model, X_new, {0, 1}, "MLPKriging 2D gauss");

  std::cout << "  PASSED\n" << std::endl;
}

// ==========================================================================
//  main
// ==========================================================================
int main() {
  try {
    test_mlp_kriging_basic();
    test_predict_derivative();

    std::cout << "============================================\n"
              << "  ALL MLPKriging TESTS PASSED\n"
              << "============================================" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "\n*** TEST FAILED: " << e.what() << " ***" << std::endl;
    return 1;
  }
  return 0;
}
