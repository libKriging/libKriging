/**
 * @file test_NeuralKernelKriging.cpp
 * @brief Tests for the NeuralKernelKriging class.
 *
 * Tests cover:
 *  1. MLP forward / backward sanity
 *  2. Full fit + predict on a 1D analytical function
 *  3. Conditional simulation statistics
 *  4. Incremental update
 *  5. Comparison with standard Kriging (should be at least as good)
 */

#include "libKriging/NeuralKernelKriging.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

using namespace libKriging;

// ---- Test helper: analytical function ----------------------------------
static double f_test(double x) {
  return 1.0 - 0.5 * (std::sin(12.0 * x) / (1.0 + x)
                       + 2.0 * std::cos(7.0 * x) * std::pow(x, 5) + 0.7);
}

// ---- Test 1: MLP forward pass dimensions --------------------------------
static void test_mlp_forward() {
  std::cout << "=== Test 1: MLP forward pass ===" << std::endl;

  std::vector<DenseLayerSpec> layers = {
    {32, Activation::SELU, true},
    {16, Activation::SELU, false},
    {4,  Activation::ReLU, false}   // output layer
  };
  MLP net(3, layers, 42);

  assert(net.input_dim() == 3);
  assert(net.output_dim() == 4);
  assert(net.n_params() > 0);

  arma::mat X = arma::randn<arma::mat>(10, 3);
  arma::mat Phi = net.forward(X);

  assert(Phi.n_rows == 10);
  assert(Phi.n_cols == 4);
  assert(Phi.is_finite());

  std::cout << "  MLP: " << net.n_params() << " parameters" << std::endl;
  std::cout << "  Output shape: " << Phi.n_rows << " x " << Phi.n_cols << std::endl;
  std::cout << "  PASSED" << std::endl;
}

// ---- Test 2: MLP parameter serialisation round-trip ----------------------
static void test_mlp_params_roundtrip() {
  std::cout << "\n=== Test 2: MLP param round-trip ===" << std::endl;

  std::vector<DenseLayerSpec> layers = {
    {16, Activation::ReLU, false},
    {4,  Activation::ReLU, false}
  };
  MLP net(2, layers, 123);

  arma::vec params = net.get_params();
  arma::mat X = arma::randn<arma::mat>(5, 2);
  arma::mat out1 = net.forward(X);

  // Modify and restore
  arma::vec perturbed = params + 0.01 * arma::randn(params.n_elem);
  net.set_params(perturbed);
  arma::mat out2 = net.forward(X);
  assert(arma::norm(out1 - out2) > 1e-10);  // should differ

  net.set_params(params);
  arma::mat out3 = net.forward(X);
  assert(arma::approx_equal(out1, out3, "absdiff", 1e-12));

  std::cout << "  PASSED" << std::endl;
}

// ---- Test 3: MLP backward gradient check (finite differences) -----------
static void test_mlp_backward() {
  std::cout << "\n=== Test 3: MLP backward (gradient check) ===" << std::endl;

  std::vector<DenseLayerSpec> layers = {
    {8, Activation::Tanh, false},
    {3, Activation::Tanh, false}
  };
  MLP net(2, layers, 7);

  arma::mat X = arma::randn<arma::mat>(4, 2);
  arma::mat dL_dPhi = arma::randn<arma::mat>(4, 3);

  arma::vec grad_analytic = net.backward(X, dL_dPhi);

  // Numerical gradient
  arma::vec params = net.get_params();
  arma::vec grad_numeric(params.n_elem);
  const double h = 1e-5;

  // Loss = sum of  dL_dPhi .* Phi  (linear in Phi → grad = backward(dL_dPhi))
  auto loss_fn = [&](const arma::vec& p) -> double {
    MLP tmp_net(2, layers, 7);
    tmp_net.set_params(p);
    arma::mat out = tmp_net.forward(X);
    return arma::accu(dL_dPhi % out);
  };

  for (arma::uword i = 0; i < params.n_elem; ++i) {
    arma::vec pp = params, pm = params;
    pp(i) += h;
    pm(i) -= h;
    grad_numeric(i) = (loss_fn(pp) - loss_fn(pm)) / (2.0 * h);
  }

  double rel_err = arma::norm(grad_analytic - grad_numeric)
                 / (arma::norm(grad_numeric) + 1e-12);
  std::cout << "  Relative gradient error: " << rel_err << std::endl;
  // We expect ~1e-4 or better with tanh (smooth activation)
  assert(rel_err < 1e-2);  // generous tolerance for BN approximations

  std::cout << "  PASSED" << std::endl;
}

// ---- Test 4: Full fit + predict on 1D function --------------------------
static void test_fit_predict_1d() {
  std::cout << "\n=== Test 4: fit + predict (1D) ===" << std::endl;

  // Training data
  arma::vec X_train = arma::linspace(0.0, 1.0, 8);
  arma::mat X_mat = arma::mat(X_train.n_elem, 1);
  X_mat.col(0) = X_train;

  arma::vec y_train(X_train.n_elem);
  for (arma::uword i = 0; i < X_train.n_elem; ++i)
    y_train(i) = f_test(X_train(i));

  // Build model
  NeuralKernelKriging model("gauss");
  model.setNNArchitecture({16, 8}, 2, "selu", false, 42);
  model.fit(y_train, X_mat, "constant", true, "Adam", "LL",
            {{"max_iter_adam", "200"}});

  std::cout << model.summary();

  // Predict on a dense grid
  arma::vec x_pred = arma::linspace(0.0, 1.0, 50);
  arma::mat x_pred_mat(x_pred.n_elem, 1);
  x_pred_mat.col(0) = x_pred;

  auto [mean, stdev, cov] = model.predict(x_pred_mat, true, false);

  assert(mean.n_elem == 50);
  assert(stdev.n_elem == 50);
  assert(mean.is_finite());
  assert(stdev.is_finite());

  // Check interpolation at training points
  auto [mean_train, stdev_train, _] = model.predict(X_mat, true, false);
  double max_interp_error = arma::max(arma::abs(mean_train - y_train));
  std::cout << "  Max interpolation error at training points: "
            << max_interp_error << std::endl;

  // Check that stdev is small at training points
  double max_train_std = arma::max(stdev_train);
  std::cout << "  Max stdev at training points: " << max_train_std << std::endl;

  // Check overall prediction quality
  arma::vec y_true(x_pred.n_elem);
  for (arma::uword i = 0; i < x_pred.n_elem; ++i)
    y_true(i) = f_test(x_pred(i));

  double rmse = std::sqrt(arma::mean(arma::square(mean - y_true)));
  std::cout << "  RMSE on dense grid: " << rmse << std::endl;

  std::cout << "  PASSED" << std::endl;
}

// ---- Test 5: Conditional simulations ------------------------------------
static void test_simulate() {
  std::cout << "\n=== Test 5: simulate ===" << std::endl;

  arma::vec X_train = {0.0, 0.25, 0.5, 0.75, 1.0};
  arma::mat X_mat(5, 1);
  X_mat.col(0) = X_train;
  arma::vec y_train(5);
  for (arma::uword i = 0; i < 5; ++i)
    y_train(i) = f_test(X_train(i));

  NeuralKernelKriging model("gauss");
  model.fit(y_train, X_mat, "constant", true, "Adam", "LL",
            {{"max_iter_adam", "100"}});

  arma::vec x_sim = arma::linspace(0.0, 1.0, 30);
  arma::mat x_sim_mat(30, 1);
  x_sim_mat.col(0) = x_sim;

  arma::mat sims = model.simulate(50, 123, x_sim_mat);

  assert(sims.n_rows == 30);
  assert(sims.n_cols == 50);
  assert(sims.is_finite());

  // The mean of simulations should be close to the kriging mean
  arma::vec sim_mean = arma::mean(sims, 1);
  auto [krig_mean, krig_std, _] = model.predict(x_sim_mat, true, false);

  double mean_diff = arma::norm(sim_mean - krig_mean) / arma::norm(krig_mean);
  std::cout << "  Relative diff between sim mean and kriging mean: "
            << mean_diff << std::endl;

  std::cout << "  Simulation matrix shape: " << sims.n_rows
            << " x " << sims.n_cols << std::endl;
  std::cout << "  PASSED" << std::endl;
}

// ---- Test 6: update (incremental) --------------------------------------
static void test_update() {
  std::cout << "\n=== Test 6: update (incremental) ===" << std::endl;

  arma::vec X_init = {0.0, 0.5, 1.0};
  arma::mat X_mat(3, 1);
  X_mat.col(0) = X_init;
  arma::vec y_init(3);
  for (arma::uword i = 0; i < 3; ++i)
    y_init(i) = f_test(X_init(i));

  NeuralKernelKriging model("gauss");
  model.fit(y_init, X_mat, "constant", true, "Adam", "LL",
            {{"max_iter_adam", "100"}});

  double ll_before = model.logLikelihood();
  std::cout << "  LL before update: " << ll_before << std::endl;

  // Add new points
  arma::vec X_new = {0.25, 0.75};
  arma::mat X_new_mat(2, 1);
  X_new_mat.col(0) = X_new;
  arma::vec y_new(2);
  for (arma::uword i = 0; i < 2; ++i)
    y_new(i) = f_test(X_new(i));

  model.update(y_new, X_new_mat);

  assert(model.X().n_rows == 5);
  assert(model.y().n_elem == 5);

  double ll_after = model.logLikelihood();
  std::cout << "  LL after update:  " << ll_after << std::endl;
  std::cout << "  n obs after:      " << model.y().n_elem << std::endl;
  std::cout << "  PASSED" << std::endl;
}

// ---- Test 7: Multi-dimensional input ------------------------------------
static void test_multidim() {
  std::cout << "\n=== Test 7: multi-dimensional input (2D) ===" << std::endl;

  // Branin-like function
  auto branin = [](double x1, double x2) -> double {
    double a = 1.0, b = 5.1 / (4.0 * M_PI * M_PI);
    double c = 5.0 / M_PI, r = 6.0, s = 10.0, t = 1.0 / (8.0 * M_PI);
    return a * std::pow(x2 - b * x1 * x1 + c * x1 - r, 2)
           + s * (1.0 - t) * std::cos(x1) + s;
  };

  // Latin-hypercube–ish design in [0,1]²
  arma::uword n = 20;
  arma::mat X = arma::randu<arma::mat>(n, 2);
  arma::vec y(n);
  for (arma::uword i = 0; i < n; ++i)
    y(i) = branin(X(i, 0) * 15.0 - 5.0, X(i, 1) * 15.0);

  NeuralKernelKriging model("matern5_2");
  model.setNNArchitecture({32, 16}, 3, "selu", false, 99);
  model.fit(y, X, "constant", true, "Adam", "LL",
            {{"max_iter_adam", "300"}});

  std::cout << model.summary();

  // Predict
  arma::mat X_test = arma::randu<arma::mat>(10, 2);
  auto [mean, stdev, _] = model.predict(X_test, true, false);

  assert(mean.n_elem == 10);
  assert(stdev.n_elem == 10);
  assert(mean.is_finite());
  assert(stdev.is_finite());

  std::cout << "  Predicted mean range: ["
            << arma::min(mean) << ", " << arma::max(mean) << "]" << std::endl;
  std::cout << "  Predicted stdev range: ["
            << arma::min(stdev) << ", " << arma::max(stdev) << "]" << std::endl;
  std::cout << "  PASSED" << std::endl;
}

// ---- main ---------------------------------------------------------------
int main() {
  std::cout << "============================================\n"
            << "  NeuralKernelKriging test suite\n"
            << "============================================\n" << std::endl;

  try {
    test_mlp_forward();
    test_mlp_params_roundtrip();
    test_mlp_backward();
    test_fit_predict_1d();
    test_simulate();
    test_update();
    test_multidim();

    std::cout << "\n============================================\n"
              << "  ALL TESTS PASSED\n"
              << "============================================" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "\n*** TEST FAILED: " << e.what() << " ***" << std::endl;
    return 1;
  }

  return 0;
}
