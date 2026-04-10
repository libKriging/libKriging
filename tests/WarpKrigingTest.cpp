#define _USE_MATH_DEFINES  // required for Visual Studio
/**
 * @file test_WarpKriging.cpp
 * @brief Tests for the WarpKriging class.
 *
 * Tests cover:
 *  1. Individual warp functions (forward + backward)
 *  2. Continuous-only: Kumaraswamy warping on 1D function
 *  3. Categorical-only: embedding on a purely discrete problem
 *  4. Mixed continuous + categorical
 *  5. Ordinal variable
 *  6. NeuralMono warping
 *  7. Conditional simulations with mixed variables
 *  8. Incremental update
 */

#include "libKriging/WarpKriging.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace libKriging;

// -- helpers ---------------------------------------------------------------

static double f1d(double x) {
  return 1.0 - 0.5 * (std::sin(12.0 * x) / (1.0 + x) + 2.0 * std::cos(7.0 * x) * std::pow(x, 5) + 0.7);
}

/// Check that the numerical and analytical gradients of a warp agree.
static double check_warp_grad(IWarp& w, const arma::vec& x) {
  arma::vec params = w.get_params();
  if (params.empty())
    return 0.0;  // no params

  arma::mat Phi = w.forward(x);
  arma::mat dL_dPhi = arma::randn(Phi.n_rows, Phi.n_cols);

  arma::vec grad_analytic = w.backward(x, dL_dPhi);

  // loss = sum(dL_dPhi .* Phi)  -- linear in Phi
  auto loss_fn = [&](const arma::vec& p) -> double {
    w.set_params(p);
    arma::mat out = w.forward(x);
    return arma::accu(dL_dPhi % out);
  };

  const double h = 1e-5;
  arma::vec grad_numeric(params.n_elem);
  for (arma::uword i = 0; i < params.n_elem; ++i) {
    arma::vec pp = params, pm = params;
    pp(i) += h;
    pm(i) -= h;
    grad_numeric(i) = (loss_fn(pp) - loss_fn(pm)) / (2.0 * h);
  }
  w.set_params(params);  // restore

  double rel = arma::norm(grad_analytic - grad_numeric) / (arma::norm(grad_numeric) + 1e-12);
  return rel;
}

// ==========================================================================
//  Test 1: individual warp functions
// ==========================================================================
static void test_warp_functions() {
  std::cout << "=== Test 1: individual warp functions ===" << std::endl;

  arma::vec x_cont = arma::linspace(0.01, 0.99, 20);
  arma::vec x_disc = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0};

  // -- Affine --
  {
    WarpAffine w;
    w.set_params({2.0, -0.5});
    arma::mat out = w.forward(x_cont);
    assert(out.n_rows == 20 && out.n_cols == 1);
    // w(0.5) = 2*0.5 - 0.5 = 0.5
    double rel = check_warp_grad(w, x_cont);
    std::cout << "  Affine grad error:      " << rel << std::endl;
    assert(rel < 1e-6);
  }

  // -- BoxCox --
  {
    WarpBoxCox w;
    w.set_params({0.5});
    arma::mat out = w.forward(x_cont);
    assert(out.is_finite());
    double rel = check_warp_grad(w, x_cont);
    std::cout << "  BoxCox grad error:      " << rel << std::endl;
    assert(rel < 1e-4);
  }

  // -- Kumaraswamy --
  {
    WarpKumaraswamy w;
    arma::mat out = w.forward(x_cont);
    assert(out.is_finite());
    double rel = check_warp_grad(w, x_cont);
    std::cout << "  Kumaraswamy grad error: " << rel << std::endl;
    assert(rel < 1e-4);
  }

  // -- NeuralMono --
  {
    WarpNeuralMono w(8, 42);
    arma::mat out = w.forward(x_cont);
    assert(out.is_finite());
    // Check monotonicity
    bool mono = true;
    for (arma::uword i = 1; i < out.n_rows; ++i)
      if (out(i, 0) < out(i - 1, 0) - 1e-10)
        mono = false;
    std::cout << "  NeuralMono monotone:    " << (mono ? "yes" : "NO") << std::endl;
    double rel = check_warp_grad(w, x_cont);
    std::cout << "  NeuralMono grad error:  " << rel << std::endl;
    assert(rel < 1e-4);
  }

  // -- MLP (unconstrained, multi-dim output) --
  {
    WarpMLP w({16, 8}, 3, WarpMLP::Act::SELU, 42);
    arma::mat out = w.forward(x_cont);
    assert(out.n_rows == 20 && out.n_cols == 3);
    assert(out.is_finite());
    std::cout << "  MLP output shape:       " << out.n_rows << " x " << out.n_cols << std::endl;
    std::cout << "  MLP n_params:           " << w.n_params() << std::endl;
    double rel = check_warp_grad(w, x_cont);
    std::cout << "  MLP grad error:         " << rel << std::endl;
    assert(rel < 1e-4);

    // Also test with Tanh activation
    WarpMLP w2({8}, 2, WarpMLP::Act::Tanh, 99);
    double rel2 = check_warp_grad(w2, x_cont);
    std::cout << "  MLP(Tanh) grad error:   " << rel2 << std::endl;
    assert(rel2 < 1e-4);
  }

  // -- Embedding --
  {
    WarpEmbedding w(3, 2, 42);
    arma::mat out = w.forward(x_disc);
    assert(out.n_rows == 10 && out.n_cols == 2);
    // Same level → same embedding
    assert(arma::approx_equal(out.row(0), out.row(3), "absdiff", 1e-12));
    double rel = check_warp_grad(w, x_disc);
    std::cout << "  Embedding grad error:   " << rel << std::endl;
    assert(rel < 1e-6);
  }

  // -- Ordinal --
  {
    WarpOrdinal w(4, 42);
    arma::vec x_ord = {0, 1, 2, 3, 0, 1, 2, 3};
    arma::mat out = w.forward(x_ord);
    assert(out.is_finite());
    // Check ordering
    bool ordered = true;
    for (arma::uword l = 1; l < 4; ++l) {
      // find first occurrence of level l and l-1
      double z_l = 0, z_lm1 = 0;
      for (arma::uword i = 0; i < x_ord.n_elem; ++i) {
        if (std::round(x_ord(i)) == l)
          z_l = out(i, 0);
        if (std::round(x_ord(i)) == l - 1)
          z_lm1 = out(i, 0);
      }
      if (z_l <= z_lm1)
        ordered = false;
    }
    std::cout << "  Ordinal ordered:        " << (ordered ? "yes" : "NO") << std::endl;
    double rel = check_warp_grad(w, x_ord);
    std::cout << "  Ordinal grad error:     " << rel << std::endl;
    assert(rel < 1e-5);
  }

  std::cout << "  PASSED\n" << std::endl;
}

// ==========================================================================
//  Test 2: continuous-only with Kumaraswamy
// ==========================================================================
static void test_continuous_kumaraswamy() {
  std::cout << "=== Test 2: continuous + Kumaraswamy warping ===" << std::endl;

  arma::vec X_train = arma::linspace(0.01, 0.99, 8);
  arma::mat X_mat(X_train.n_elem, 1);
  X_mat.col(0) = X_train;
  arma::vec y(X_train.n_elem);
  for (arma::uword i = 0; i < X_train.n_elem; ++i)
    y(i) = f1d(X_train(i));

  WarpKriging model({"kumaraswamy"}, "gauss");
  model.fit(y, X_mat, "constant", false, "Adam", "LL", {{"max_iter_adam", "200"}});

  std::cout << model.summary();

  auto [mean_train, stdev_train, _] = model.predict(X_mat, true, false);
  double max_err = arma::max(arma::abs(mean_train - y));
  std::cout << "  Max train interp error: " << max_err << std::endl;

  arma::vec x_pred = arma::linspace(0.01, 0.99, 50);
  arma::mat xp(50, 1);
  xp.col(0) = x_pred;
  auto [mean, stdev, __] = model.predict(xp, true, false);
  arma::vec ytrue(50);
  for (arma::uword i = 0; i < 50; ++i)
    ytrue(i) = f1d(x_pred(i));
  double rmse = std::sqrt(arma::mean(arma::square(mean - ytrue)));
  std::cout << "  RMSE on dense grid:     " << rmse << std::endl;
  std::cout << "  PASSED\n" << std::endl;
}

// ==========================================================================
//  Test 3: categorical-only
// ==========================================================================
static void test_categorical_only() {
  std::cout << "=== Test 3: categorical-only (embedding) ===" << std::endl;

  // 3 categories with different means
  double mu[] = {1.0, 5.0, 3.0};
  arma::uword n = 15;
  arma::mat X(n, 1);
  arma::vec y(n);
  arma::arma_rng::set_seed(42);
  for (arma::uword i = 0; i < n; ++i) {
    arma::uword level = i % 3;
    X(i, 0) = static_cast<double>(level);
    y(i) = mu[level] + 0.1 * arma::randn<arma::vec>(1)(0);
  }

  WarpKriging model({"categorical(3,2)"}, "gauss");
  model.fit(y, X, "constant", false, "Adam", "LL", {{"max_iter_adam", "200"}});

  std::cout << model.summary();

  // Predict at each category
  arma::mat X_test(3, 1);
  X_test(0, 0) = 0.0;
  X_test(1, 0) = 1.0;
  X_test(2, 0) = 2.0;
  auto [mean, stdev, _] = model.predict(X_test, true, false);

  std::cout << "  Predictions per level:\n";
  for (int l = 0; l < 3; ++l)
    std::cout << "    level " << l << ": mean=" << mean(l) << ", stdev=" << stdev(l) << ", true_mu=" << mu[l]
              << std::endl;

  // Check that predictions are reasonable
  for (int l = 0; l < 3; ++l)
    assert(std::abs(mean(l) - mu[l]) < 2.0);

  // Inspect learned embeddings
  const auto& emb = dynamic_cast<const WarpEmbedding&>(model.warp(0));
  std::cout << "  Learned embeddings:\n";
  arma::vec lev = {0, 1, 2};
  arma::mat E = emb.forward(lev);
  for (int l = 0; l < 3; ++l)
    std::cout << "    level " << l << ": " << E.row(l);

  std::cout << "  PASSED\n" << std::endl;
}

// ==========================================================================
//  Test 3b: categorical prediction at training points (interpolation check)
//
//  A well-fitted GP must interpolate its training data exactly (up to the
//  tiny jitter nugget).  When predict() accidentally scales the cross-
//  correlation by σ², the predicted mean at training points is
//    F*β + σ² * r * R⁻¹(y - F*β)  instead of  F*β + r * R⁻¹(y - F*β),
//  which only equals y when σ² == 1.
// ==========================================================================
static void test_categorical_predict_at_train() {
  std::cout << "=== Test 3b: categorical predict at training points ===" << std::endl;

  // 3 categories with well-separated means → σ² >> 1
  double mu[] = {1.0, 10.0, 50.0};
  arma::uword n = 15;
  arma::mat X(n, 1);
  arma::vec y(n);
  arma::arma_rng::set_seed(42);
  for (arma::uword i = 0; i < n; ++i) {
    arma::uword level = i % 3;
    X(i, 0) = static_cast<double>(level);
    y(i) = mu[level] + 0.01 * arma::randn<arma::vec>(1)(0);
  }

  WarpKriging model({"categorical(3,2)"}, "gauss");
  model.fit(y, X, "constant", false, "Adam", "LL", {{"max_iter_adam", "300"}});

  std::cout << model.summary();

  // Predict at training points — must recover training values
  auto [mean, stdev, _] = model.predict(X, true, false);

  double max_mean_err = arma::max(arma::abs(mean - y));
  double max_stdev = arma::max(stdev);
  std::cout << "  sigma2 = " << model.sigma2() << std::endl;
  std::cout << "  Max |pred_mean - y_train| = " << max_mean_err << std::endl;
  std::cout << "  Max pred_stdev            = " << max_stdev << std::endl;

  // With the nugget 1e-8 on diagonal, interpolation error should be tiny
  assert(max_mean_err < 0.5 && "Prediction at training points must interpolate y");
  assert(max_stdev < 1.0 && "Stdev at training points must be near zero");

  std::cout << "  PASSED\n" << std::endl;
}

// ==========================================================================
//  Test 4: mixed continuous + categorical
// ==========================================================================
static void test_mixed() {
  std::cout << "=== Test 4: mixed (1 continuous + 1 categorical) ===" << std::endl;

  // f(x, cat) = sin(x) * offset[cat]
  double offset[] = {1.0, 2.0, 0.5};
  arma::uword n = 30;
  arma::mat X(n, 2);
  arma::vec y(n);
  arma::arma_rng::set_seed(99);

  for (arma::uword i = 0; i < n; ++i) {
    double xi = arma::randu<arma::vec>(1)(0);
    arma::uword cat = i % 3;
    X(i, 0) = xi;
    X(i, 1) = static_cast<double>(cat);
    y(i) = std::sin(2.0 * M_PI * xi) * offset[cat];
  }

  WarpKriging model(
      {
          "kumaraswamy",      // x0: continuous
          "categorical(3,2)"  // x1: 3-level categorical
      },
      "matern5_2");

  model.fit(y, X, "constant", false, "Adam", "LL", {{"max_iter_adam", "300"}});

  std::cout << model.summary();

  // Predict on a grid per category
  arma::vec xc = arma::linspace(0.01, 0.99, 20);
  for (int cat = 0; cat < 3; ++cat) {
    arma::mat X_test(20, 2);
    X_test.col(0) = xc;
    X_test.col(1).fill(static_cast<double>(cat));

    auto [mean, stdev, _] = model.predict(X_test, true, false);

    arma::vec ytrue(20);
    for (arma::uword i = 0; i < 20; ++i)
      ytrue(i) = std::sin(2.0 * M_PI * xc(i)) * offset[cat];

    double rmse = std::sqrt(arma::mean(arma::square(mean - ytrue)));
    std::cout << "  cat=" << cat << "  RMSE=" << rmse << std::endl;
  }

  std::cout << "  PASSED\n" << std::endl;
}

// ==========================================================================
//  Test 5: ordinal variable
// ==========================================================================
static void test_ordinal() {
  std::cout << "=== Test 5: ordinal variable ===" << std::endl;

  // f(level) = level^2  (monotone in level)
  arma::uword L = 5;
  arma::uword n = 20;
  arma::mat X(n, 1);
  arma::vec y(n);
  arma::arma_rng::set_seed(7);

  for (arma::uword i = 0; i < n; ++i) {
    arma::uword level = i % L;
    X(i, 0) = static_cast<double>(level);
    y(i) = static_cast<double>(level * level) + 0.1 * arma::randn<arma::vec>(1)(0);
  }

  WarpKriging model({"ordinal(5)"}, "gauss");
  model.fit(y, X, "constant", false, "Adam", "LL", {{"max_iter_adam", "200"}});

  std::cout << model.summary();

  arma::mat X_test(L, 1);
  for (arma::uword l = 0; l < L; ++l)
    X_test(l, 0) = static_cast<double>(l);
  auto [mean, stdev, _] = model.predict(X_test, true, false);

  std::cout << "  Predictions:\n";
  for (arma::uword l = 0; l < L; ++l)
    std::cout << "    level " << l << ": pred=" << mean(l) << ", true=" << l * l << std::endl;

  // Inspect learned positions
  std::cout << "  Warp: " << model.warp(0).describe() << std::endl;
  std::cout << "  PASSED\n" << std::endl;
}

// ==========================================================================
//  Test 6: NeuralMono warping
// ==========================================================================
static void test_neural_mono() {
  std::cout << "=== Test 6: NeuralMono warping ===" << std::endl;

  arma::vec X_train = arma::linspace(0.01, 0.99, 10);
  arma::mat X_mat(X_train.n_elem, 1);
  X_mat.col(0) = X_train;
  arma::vec y(X_train.n_elem);
  for (arma::uword i = 0; i < X_train.n_elem; ++i)
    y(i) = f1d(X_train(i));

  WarpKriging model({"neural_mono(8)"}, "gauss");
  model.fit(y, X_mat, "constant", false, "Adam", "LL", {{"max_iter_adam", "200"}});

  std::cout << model.summary();

  auto [mean_train, stdev_train, _] = model.predict(X_mat, true, false);
  double max_err = arma::max(arma::abs(mean_train - y));
  std::cout << "  Max train interp error: " << max_err << std::endl;

  std::cout << "  PASSED\n" << std::endl;
}

// ==========================================================================
//  Test 7: MLP warping (unconstrained, multi-dim output)
// ==========================================================================
static void test_mlp_warp() {
  std::cout << "=== Test 7: MLP warping ===" << std::endl;

  // --- 7a: 1D function with MLP warp ---
  arma::vec X_train = arma::linspace(0.01, 0.99, 10);
  arma::mat X_mat(X_train.n_elem, 1);
  X_mat.col(0) = X_train;
  arma::vec y(X_train.n_elem);
  for (arma::uword i = 0; i < X_train.n_elem; ++i)
    y(i) = f1d(X_train(i));

  WarpKriging model1d({"mlp(16:8,3,selu)"}, "gauss");
  model1d.fit(y, X_mat, "constant", false, "Adam", "LL", {{"max_iter_adam", "300"}});

  std::cout << model1d.summary();

  auto [mean_train, stdev_train, _1] = model1d.predict(X_mat, true, false);
  double max_err = arma::max(arma::abs(mean_train - y));
  std::cout << "  [1D] Max train interp error:  " << max_err << std::endl;

  arma::vec x_pred = arma::linspace(0.01, 0.99, 50);
  arma::mat xp(50, 1);
  xp.col(0) = x_pred;
  auto [mean, stdev, _2] = model1d.predict(xp, true, false);
  arma::vec ytrue(50);
  for (arma::uword i = 0; i < 50; ++i)
    ytrue(i) = f1d(x_pred(i));
  double rmse = std::sqrt(arma::mean(arma::square(mean - ytrue)));
  std::cout << "  [1D] RMSE on dense grid:      " << rmse << std::endl;

  // --- 7b: mixed MLP + categorical ---
  double offset[] = {1.0, 2.0, 0.5};
  arma::uword n = 30;
  arma::mat X_mix(n, 2);
  arma::vec y_mix(n);
  arma::arma_rng::set_seed(99);
  for (arma::uword i = 0; i < n; ++i) {
    double xi = arma::randu<arma::vec>(1)(0);
    arma::uword cat = i % 3;
    X_mix(i, 0) = xi;
    X_mix(i, 1) = static_cast<double>(cat);
    y_mix(i) = std::sin(2.0 * M_PI * xi) * offset[cat];
  }

  WarpKriging model_mix(
      {
          "mlp(16:8,2,tanh)",  // x0: MLP → ℝ²
          "categorical(3,2)"   // x1: 3 cats → ℝ²
      },
      "matern5_2");

  model_mix.fit(y_mix, X_mix, "constant", false, "Adam", "LL", {{"max_iter_adam", "300"}});

  std::cout << "  [Mixed MLP+cat] summary:\n" << model_mix.summary();

  arma::vec xc = arma::linspace(0.01, 0.99, 20);
  for (int cat = 0; cat < 3; ++cat) {
    arma::mat X_test(20, 2);
    X_test.col(0) = xc;
    X_test.col(1).fill(static_cast<double>(cat));
    auto [m, s, _] = model_mix.predict(X_test, true, false);
    arma::vec yt(20);
    for (arma::uword i = 0; i < 20; ++i)
      yt(i) = std::sin(2.0 * M_PI * xc(i)) * offset[cat];
    double cat_rmse = std::sqrt(arma::mean(arma::square(m - yt)));
    std::cout << "  [Mixed] cat=" << cat << "  RMSE=" << cat_rmse << std::endl;
  }

  std::cout << "  PASSED\n" << std::endl;
}

// ==========================================================================
//  Test 8: simulate with mixed variables
// ==========================================================================
static void test_simulate_mixed() {
  std::cout << "=== Test 8: simulate (mixed) ===" << std::endl;

  double offset[] = {1.0, 3.0};
  arma::uword n = 20;
  arma::mat X(n, 2);
  arma::vec y(n);
  arma::arma_rng::set_seed(42);
  for (arma::uword i = 0; i < n; ++i) {
    X(i, 0) = arma::randu<arma::vec>(1)(0);
    X(i, 1) = static_cast<double>(i % 2);
    y(i) = std::sin(2.0 * M_PI * X(i, 0)) * offset[i % 2];
  }

  WarpKriging model({"affine", "categorical(2,2)"}, "gauss");
  model.fit(y, X, "constant", false, "Adam", "LL", {{"max_iter_adam", "200"}});

  arma::mat X_sim(10, 2);
  X_sim.col(0) = arma::linspace(0.1, 0.9, 10);
  X_sim.col(1).fill(0.0);

  arma::mat sims = model.simulate(30, 123, X_sim);
  assert(sims.n_rows == 10 && sims.n_cols == 30);
  assert(sims.is_finite());

  auto [mean, stdev, _] = model.predict(X_sim, true, false);
  arma::vec sim_mean = arma::mean(sims, 1);
  double rel_diff = arma::norm(sim_mean - mean) / arma::norm(mean);
  std::cout << "  Sim mean vs kriging mean rel diff: " << rel_diff << std::endl;
  std::cout << "  PASSED\n" << std::endl;
}

// ==========================================================================
//  Test 9: update
// ==========================================================================
static void test_update() {
  std::cout << "=== Test 9: update ===" << std::endl;

  arma::mat X0 = {{0.1, 0.0}, {0.5, 1.0}, {0.9, 0.0}};
  arma::vec y0 = {1.0, 3.0, 0.5};

  WarpKriging model({"none", "categorical(2,1)"}, "gauss");
  model.fit(y0, X0, "constant", false, "Adam", "LL", {{"max_iter_adam", "100"}});

  std::cout << "  n before update: " << model.y().n_elem << std::endl;

  arma::mat X_new = {{0.3, 1.0}, {0.7, 0.0}};
  arma::vec y_new = {2.0, 1.5};
  model.update(y_new, X_new);

  assert(model.y().n_elem == 5);
  std::cout << "  n after update:  " << model.y().n_elem << std::endl;
  std::cout << "  PASSED\n" << std::endl;
}

// ==========================================================================
//  Test 10: WarpMLP param round-trip
// ==========================================================================
static void test_mlp_params_roundtrip() {
  std::cout << "=== Test 10: WarpMLP param round-trip ===" << std::endl;

  WarpMLP w({16, 8}, 3, WarpMLP::Act::SELU, 42);
  arma::vec params = w.get_params();
  arma::vec x = arma::linspace(0.0, 1.0, 10);
  arma::mat out1 = w.forward(x);

  // Perturb weights
  arma::vec perturbed = params + 0.01 * arma::randn(params.n_elem);
  w.set_params(perturbed);
  arma::mat out2 = w.forward(x);
  assert(arma::norm(out1 - out2, "fro") > 1e-10);  // output must differ

  // Restore → must match original
  w.set_params(params);
  arma::mat out3 = w.forward(x);
  assert(arma::approx_equal(out1, out3, "absdiff", 1e-12));

  std::cout << "  n_params = " << w.n_params() << std::endl;
  std::cout << "  PASSED\n" << std::endl;
}

// ==========================================================================
//  Test 11: Branin 2D (multi-dimensional continuous with MLP warp)
// ==========================================================================
static void test_branin_2d_mlp() {
  std::cout << "=== Test 11: Branin 2D with MLP warping ===" << std::endl;

  auto branin = [](double x1, double x2) -> double {
    double a = 1.0, b = 5.1 / (4.0 * M_PI * M_PI);
    double c = 5.0 / M_PI, r = 6.0, s = 10.0, t = 1.0 / (8.0 * M_PI);
    return a * std::pow(x2 - b * x1 * x1 + c * x1 - r, 2) + s * (1.0 - t) * std::cos(x1) + s;
  };

  arma::uword n = 25;
  arma::arma_rng::set_seed(77);
  arma::mat X = arma::randu<arma::mat>(n, 2);
  arma::vec y(n);
  for (arma::uword i = 0; i < n; ++i)
    y(i) = branin(X(i, 0) * 15.0 - 5.0, X(i, 1) * 15.0);

  // Each continuous input gets its own MLP warp → ℝ²
  WarpKriging model({"mlp(16:8,2,selu)", "mlp(16:8,2,selu)"}, "matern5_2");

  model.fit(y, X, "constant", true, "Adam", "LL", {{"max_iter_adam", "300"}});

  std::cout << model.summary();

  // Predict on training data (interpolation check)
  auto [mean_train, stdev_train, _1] = model.predict(X, true, false);
  double max_interp = arma::max(arma::abs(mean_train - y));
  std::cout << "  Max train interpolation error: " << max_interp << std::endl;
  std::cout << "  Max train stdev:               " << arma::max(stdev_train) << std::endl;

  // Predict on new test points
  arma::mat X_test = arma::randu<arma::mat>(15, 2);
  auto [mean_test, stdev_test, _2] = model.predict(X_test, true, false);
  assert(mean_test.n_elem == 15);
  assert(stdev_test.n_elem == 15);
  assert(mean_test.is_finite());
  assert(stdev_test.is_finite());

  std::cout << "  Predicted mean range:  [" << arma::min(mean_test) << ", " << arma::max(mean_test) << "]" << std::endl;
  std::cout << "  Predicted stdev range: [" << arma::min(stdev_test) << ", " << arma::max(stdev_test) << "]"
            << std::endl;

  // Simulate
  arma::mat sims = model.simulate(20, 42, X_test);
  assert(sims.n_rows == 15 && sims.n_cols == 20);
  assert(sims.is_finite());

  std::cout << "  PASSED\n" << std::endl;
}

// ==========================================================================
//  Test 12: Comparison None (baseline) vs MLP warping
// ==========================================================================
static void test_none_vs_mlp() {
  std::cout << "=== Test 12: None vs MLP comparison ===" << std::endl;

  arma::vec X_train = arma::linspace(0.01, 0.99, 12);
  arma::mat X_mat(X_train.n_elem, 1);
  X_mat.col(0) = X_train;
  arma::vec y(X_train.n_elem);
  for (arma::uword i = 0; i < X_train.n_elem; ++i)
    y(i) = f1d(X_train(i));

  arma::vec x_pred = arma::linspace(0.01, 0.99, 50);
  arma::mat xp(50, 1);
  xp.col(0) = x_pred;
  arma::vec ytrue(50);
  for (arma::uword i = 0; i < 50; ++i)
    ytrue(i) = f1d(x_pred(i));

  // Model A: no warping (baseline, like standard Kriging)
  WarpKriging model_none({"none"}, "gauss");
  model_none.fit(y, X_mat, "constant", false, "Adam", "LL", {{"max_iter_adam", "200"}});
  auto [mean_none, sd_none, _1] = model_none.predict(xp, true, false);
  double rmse_none = std::sqrt(arma::mean(arma::square(mean_none - ytrue)));

  // Model B: MLP warping
  WarpKriging model_mlp({"mlp(16:8,2,selu)"}, "gauss");
  model_mlp.fit(y, X_mat, "constant", false, "Adam", "LL", {{"max_iter_adam", "300"}});
  auto [mean_mlp, sd_mlp, _2] = model_mlp.predict(xp, true, false);
  double rmse_mlp = std::sqrt(arma::mean(arma::square(mean_mlp - ytrue)));

  // Model C: Kumaraswamy warping
  WarpKriging model_kuma({"kumaraswamy"}, "gauss");
  model_kuma.fit(y, X_mat, "constant", false, "Adam", "LL", {{"max_iter_adam", "200"}});
  auto [mean_kuma, sd_kuma, _3] = model_kuma.predict(xp, true, false);
  double rmse_kuma = std::sqrt(arma::mean(arma::square(mean_kuma - ytrue)));

  std::cout << "  RMSE None (baseline):    " << rmse_none << std::endl;
  std::cout << "  RMSE Kumaraswamy:        " << rmse_kuma << std::endl;
  std::cout << "  RMSE MLP:                " << rmse_mlp << std::endl;
  std::cout << "  LL None:   " << model_none.logLikelihood() << std::endl;
  std::cout << "  LL Kuma:   " << model_kuma.logLikelihood() << std::endl;
  std::cout << "  LL MLP:    " << model_mlp.logLikelihood() << std::endl;
  std::cout << "  PASSED\n" << std::endl;
}

// ==========================================================================
//  Test 13: logLikelihoodFun (GP hyper-params evaluation)
// ==========================================================================
static void test_loglikelihood_fun() {
  std::cout << "=== Test 13: logLikelihoodFun ===" << std::endl;

  arma::vec X_train = arma::linspace(0.01, 0.99, 8);
  arma::mat X_mat(X_train.n_elem, 1);
  X_mat.col(0) = X_train;
  arma::vec y(X_train.n_elem);
  for (arma::uword i = 0; i < X_train.n_elem; ++i)
    y(i) = f1d(X_train(i));

  WarpKriging model({"affine"}, "gauss");
  model.fit(y, X_mat, "constant", false, "Adam", "LL", {{"max_iter_adam", "100"}});

  arma::vec theta = model.theta();
  auto [ll, grad, hess] = model.logLikelihoodFun(theta, true, false);

  std::cout << "  LL at optimum theta:    " << ll << std::endl;
  std::cout << "  Gradient norm at opt:   " << arma::norm(grad) << std::endl;
  assert(std::isfinite(ll));
  assert(grad.is_finite());

  // Check gradient by finite differences
  const double h = 1e-5;
  arma::vec grad_num(theta.n_elem);
  for (arma::uword k = 0; k < theta.n_elem; ++k) {
    arma::vec tp = theta, tm = theta;
    tp(k) += h;
    tm(k) -= h;
    auto [llp, _a, _b] = model.logLikelihoodFun(tp, false, false);
    auto [llm, _c, _d] = model.logLikelihoodFun(tm, false, false);
    grad_num(k) = (llp - llm) / (2.0 * h);
  }
  double rel = arma::norm(grad - grad_num) / (arma::norm(grad_num) + 1e-12);
  std::cout << "  Gradient FD check err:  " << rel << std::endl;
  std::cout << "  PASSED\n" << std::endl;
}

// ==========================================================================
//  Test 14: from_string / to_string round-trip
// ==========================================================================
static void test_from_string_roundtrip() {
  std::cout << "=== Test 14: from_string / to_string round-trip ===" << std::endl;

  // All canonical forms
  std::vector<std::string> inputs = {
      "none",
      "affine",
      "boxcox",
      "kumaraswamy",
      "neural_mono(8)",
      "neural_mono(16)",
      "mlp(16:8,3,selu)",
      "mlp(32:16:8,4,tanh)",
      "categorical(5,2)",
      "categorical(3,1)",
      "ordinal(4)",
      "ordinal(7)",
      "mlp_joint(32:16,3,selu)",
      "mlp_joint(64:32:16,4,tanh)",
  };

  for (const auto& s : inputs) {
    WarpSpec spec = WarpSpec::from_string(s);
    std::string back = spec.to_string();
    std::cout << "  \"" << s << "\" -> parse -> \"" << back << "\"";
    assert(back == s);
    std::cout << "  OK" << std::endl;
  }

  // Default-argument forms
  {
    WarpSpec s1 = WarpSpec::from_string("neural_mono");
    assert(s1.n_hidden == 8);
    assert(s1.to_string() == "neural_mono(8)");
    std::cout << "  \"neural_mono\" -> \"" << s1.to_string() << "\"  OK" << std::endl;
  }
  {
    WarpSpec s2 = WarpSpec::from_string("mlp(16:8)");
    assert(s2.hidden_dims.size() == 2);
    assert(s2.d_out == 2);
    assert(s2.activation == "selu");
    std::cout << "  \"mlp(16:8)\" -> \"" << s2.to_string() << "\"  OK" << std::endl;
  }
  {
    WarpSpec s3 = WarpSpec::from_string("categorical(5)");
    assert(s3.n_levels == 5);
    assert(s3.embed_dim == 2);
    std::cout << "  \"categorical(5)\" -> \"" << s3.to_string() << "\"  OK" << std::endl;
  }
  {
    WarpSpec s4 = WarpSpec::from_string("mlp_joint");
    assert(s4.type == WarpType::MLPJoint);
    assert(s4.hidden_dims.size() == 2);
    assert(s4.d_out == 2);
    std::cout << "  \"mlp_joint\" -> \"" << s4.to_string() << "\"  OK" << std::endl;
  }
  {
    WarpSpec s5 = WarpSpec::from_string("mlp_joint(16:8)");
    assert(s5.type == WarpType::MLPJoint);
    assert(s5.d_out == 2);
    assert(s5.activation == "selu");
    std::cout << "  \"mlp_joint(16:8)\" -> \"" << s5.to_string() << "\"  OK" << std::endl;
  }

  // Whitespace tolerance
  {
    WarpSpec s4 = WarpSpec::from_string("  kumaraswamy  ");
    assert(s4.type == WarpType::Kumaraswamy);
    std::cout << "  \"  kumaraswamy  \" -> \"" << s4.to_string() << "\"  OK" << std::endl;
  }

  // Invalid strings
  bool caught = false;
  try {
    WarpSpec::from_string("foobar");
  } catch (...) {
    caught = true;
  }
  assert(caught);
  std::cout << "  \"foobar\" -> exception  OK" << std::endl;

  caught = false;
  try {
    WarpSpec::from_string("categorical");
  } catch (...) {
    caught = true;
  }
  assert(caught);
  std::cout << "  \"categorical\" (no args) -> exception  OK" << std::endl;

  std::cout << "  PASSED\n" << std::endl;
}

// ==========================================================================
//  Test 15: warping_strings() accessor
// ==========================================================================
static void test_warping_strings_accessor() {
  std::cout << "=== Test 15: warping_strings() accessor ===" << std::endl;

  arma::mat X(10, 3);
  X.col(0) = arma::linspace(0.01, 0.99, 10);
  X.col(1).fill(0.0);  // categorical levels
  for (arma::uword i = 0; i < 10; ++i)
    X(i, 1) = i % 3;
  X.col(2).fill(0.0);  // ordinal levels
  for (arma::uword i = 0; i < 10; ++i)
    X(i, 2) = i % 4;

  arma::vec y = arma::randn<arma::vec>(10);

  WarpKriging model({"kumaraswamy", "categorical(3,2)", "ordinal(4)"}, "gauss");
  model.fit(y, X, "constant", false, "Adam", "LL", {{"max_iter_adam", "50"}});

  auto ws = model.warping_strings();
  assert(ws.size() == 3);
  assert(ws[0] == "kumaraswamy");
  assert(ws[1] == "categorical(3,2)");
  assert(ws[2] == "ordinal(4)");

  std::cout << "  warping_strings = {";
  for (arma::uword i = 0; i < ws.size(); ++i)
    std::cout << (i ? ", " : "") << "\"" << ws[i] << "\"";
  std::cout << "}" << std::endl;

  std::cout << "  PASSED\n" << std::endl;
}

// ==========================================================================
//  Test 16: mlp_joint — full-input MLP (≡ NeuralKernelKriging)
// ==========================================================================
static void test_mlp_joint() {
  std::cout << "=== Test 16: mlp_joint (Deep Kernel Learning) ===" << std::endl;

  // --- 16a: 1D function ---
  {
    arma::vec X_train = arma::linspace(0.01, 0.99, 10);
    arma::mat X_mat(X_train.n_elem, 1);
    X_mat.col(0) = X_train;
    arma::vec y(X_train.n_elem);
    for (arma::uword i = 0; i < X_train.n_elem; ++i)
      y(i) = f1d(X_train(i));

    WarpKriging model({"mlp_joint(16:8,2,selu)"}, "gauss");
    model.fit(y, X_mat, "constant", true, "Adam", "LL", {{"max_iter_adam", "300"}});

    std::cout << model.summary();

    auto [mean_train, stdev_train, _1] = model.predict(X_mat, true, false);
    double max_err = arma::max(arma::abs(mean_train - y));
    std::cout << "  [1D] Max train interp error:  " << max_err << std::endl;

    arma::vec x_pred = arma::linspace(0.01, 0.99, 50);
    arma::mat xp(50, 1);
    xp.col(0) = x_pred;
    auto [mean, stdev, _2] = model.predict(xp, true, false);
    arma::vec ytrue(50);
    for (arma::uword i = 0; i < 50; ++i)
      ytrue(i) = f1d(x_pred(i));
    double rmse = std::sqrt(arma::mean(arma::square(mean - ytrue)));
    std::cout << "  [1D] RMSE on dense grid:      " << rmse << std::endl;

    // Verify warping_strings
    auto ws = model.warping_strings();
    assert(ws.size() == 1);
    assert(ws[0] == "mlp_joint(16:8,2,selu)");
  }

  // --- 16b: Branin 2D (exact equivalent of old NeuralKernelKriging test) ---
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

    WarpKriging model({"mlp_joint(32:16,3,selu)"}, "matern5_2");
    model.fit(y, X, "constant", true, "Adam", "LL", {{"max_iter_adam", "300"}});

    std::cout << model.summary();

    // Predict
    arma::mat X_test = arma::randu<arma::mat>(10, 2);
    auto [mean, stdev, _3] = model.predict(X_test, true, false);
    assert(mean.n_elem == 10);
    assert(stdev.n_elem == 10);
    assert(mean.is_finite());
    assert(stdev.is_finite());

    std::cout << "  [2D] Predicted mean range:  [" << arma::min(mean) << ", " << arma::max(mean) << "]" << std::endl;
    std::cout << "  [2D] Predicted stdev range: [" << arma::min(stdev) << ", " << arma::max(stdev) << "]" << std::endl;

    // Simulate
    arma::mat sims = model.simulate(20, 42, X_test);
    assert(sims.n_rows == 10 && sims.n_cols == 20);
    assert(sims.is_finite());

    // Update
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
int main() {
  std::cout << "============================================\n"
            << "  WarpKriging test suite\n"
            << "============================================\n"
            << std::endl;

  try {
    test_warp_functions();
    test_continuous_kumaraswamy();
    test_categorical_only();
    test_categorical_predict_at_train();
    test_mixed();
    test_ordinal();
    test_neural_mono();
    test_mlp_warp();
    test_simulate_mixed();
    test_update();
    test_mlp_params_roundtrip();
    test_branin_2d_mlp();
    test_none_vs_mlp();
    test_loglikelihood_fun();
    test_from_string_roundtrip();
    test_warping_strings_accessor();
    test_mlp_joint();

    std::cout << "============================================\n"
              << "  ALL TESTS PASSED\n"
              << "============================================" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "\n*** TEST FAILED: " << e.what() << " ***" << std::endl;
    return 1;
  }
  return 0;
}
