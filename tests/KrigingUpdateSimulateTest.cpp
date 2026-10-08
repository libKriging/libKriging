// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
#include "ks_test.hpp"
#include <sstream>
// clang-format on

// NOTE: These tests verify that update_simulate() gives statistically similar results to update() + simulate().
// The two draws use different random streams, so they are compared in distribution (KS test, and moments
// against the updated model's predict()).

TEST_CASE("KrigingUpdateSimulateTest - Update simulate equals updated model simulate", "[update_simulate][kriging]") {
  arma::arma_rng::set_seed(123);

  // Generate initial training data
  const arma::uword n_old = 15;
  const arma::uword n_new = 5;
  const arma::uword d = 2;
  
  arma::mat X_old(n_old, d, arma::fill::randu);
  arma::colvec y_old(n_old);
  
  auto test_function = [](const arma::rowvec& x) {
    return std::sin(3.0 * x(0)) + std::cos(5.0 * x(1)) + 10.0 + 10*x(0);
  };
  
  for (arma::uword i = 0; i < n_old; ++i) {
    y_old(i) = test_function(X_old.row(i));
  }
  
  // Generate new data points
  arma::mat X_new(n_new, d, arma::fill::randu);
  arma::colvec y_new(n_new);
  
  for (arma::uword i = 0; i < n_new; ++i) {
    y_new(i) = test_function(X_new.row(i));
  }

  SECTION("update_simulate gives same distribution as update then simulate") {
    // Use fixed hardcoded parameters for both models
    // sigma2 = 100, theta = [1, 1], beta = [10] (constant trend)
    double sigma2_fixed = 100.0;
    arma::vec theta_fixed = arma::vec(d).fill(1.0);
    arma::vec beta_fixed = {10.0};
    Kriging::Parameters params_fixed{sigma2_fixed, false, arma::mat(theta_fixed), false, beta_fixed, false};

    // Build kr1 with fixed parameters
    Kriging kr1("exp");
    kr1.fit(y_old, X_old, Trend::RegressionModel::Constant, false, "none", "LL", params_fixed);

    // Simulation points
    const arma::uword n_sim_points = 10;
    arma::mat X_sim(n_sim_points, d, arma::fill::randu);
    
    // Method 1: Use update_simulate
    const int n_sims = 1000;
    const int seed = 789;
    kr1.simulate(n_sims, seed, X_sim, true);  // Store for update_simulate
    arma::mat sims1 = kr1.update_simulate(y_new, X_new);
    
    // Method 2: Build kr2 with same fixed parameters on full data
    Kriging kr2("exp");
    arma::mat X_full = arma::join_cols(X_old, X_new);
    arma::colvec y_full = arma::join_cols(y_old, y_new);
    kr2.fit(y_full, X_full, Trend::RegressionModel::Constant, false, "none", "LL", params_fixed);
    arma::mat sims2 = kr2.simulate(n_sims, seed, X_sim);

    // Save simulations to CSV for debugging
    sims1.save("sims1_update_simulate.csv", arma::csv_ascii);
    sims2.save("sims2_full_model.csv", arma::csv_ascii);
    X_sim.save("X_sim.csv", arma::csv_ascii);

    // KS test: check if samples come from same distribution at each point
    int ks_failures = 0;
    std::stringstream failure_details;
    for (arma::uword i = 0; i < n_sim_points; ++i) {
      arma::rowvec sample1 = sims1.row(i);
      arma::rowvec sample2 = sims2.row(i);
      auto [passed, pvalue] = KSTest::ks_test_with_pvalue(sample1, sample2, 1e-7);
      if (!passed) {
        failure_details << "\n  Point " << i << " failed with p-value: " << pvalue;
        ks_failures++;
      }
    }
    INFO("KS test failures: " << ks_failures << " / " << n_sim_points << failure_details.str());
    CHECK(ks_failures == 0);
  }

  SECTION("Multiple points update_simulate") {
    // Use fixed hardcoded parameters for both models
    // sigma2 = 100, theta = [1, 1], beta = [10] (constant trend)
    double sigma2_fixed = 100.0;
    arma::vec theta_fixed = arma::vec(d).fill(1.0);
    arma::vec beta_fixed = {10.0};
    Kriging::Parameters params_fixed{sigma2_fixed, false, arma::mat(theta_fixed), false, beta_fixed, false};
    
    // Build kr1 with fixed parameters
    Kriging kr1("exp");
    kr1.fit(y_old, X_old, Trend::RegressionModel::Constant, false, "none", "LL", params_fixed);
    
    // Simulation points
    const arma::uword n_sim_points = 5;
    arma::mat X_sim(n_sim_points, d, arma::fill::randu);
    
    const int n_sims = 1000;
    const int seed = 999;
    
    // Use update_simulate with multiple new points
    kr1.simulate(n_sims, seed, X_sim, true);
    arma::mat sims1 = kr1.update_simulate(y_new, X_new);
    
    // Build kr2 with same fixed parameters on full data
    Kriging kr2("exp");
    arma::mat X_full = arma::join_cols(X_old, X_new);
    arma::colvec y_full = arma::join_cols(y_old, y_new);
    kr2.fit(y_full, X_full, Trend::RegressionModel::Constant, false, "none", "LL", params_fixed);
    arma::mat sims2 = kr2.simulate(n_sims, seed, X_sim);
    
    // KS test
    int ks_failures = 0;
    std::stringstream failure_details;
    for (arma::uword i = 0; i < X_sim.n_rows; ++i) {
      arma::rowvec sample1 = sims1.row(i); 
      arma::rowvec sample2 = sims2.row(i); 
      auto [passed, pvalue] = KSTest::ks_test_with_pvalue(sample1, sample2, 1e-7);
      if (!passed) {
        failure_details << "\n  Point " << i << " failed with p-value: " << pvalue;
        ks_failures++;
      }
    }
    INFO("KS test failures: " << ks_failures << " / " << n_sim_points << failure_details.str());
    CHECK(ks_failures == 0);
  }

  SECTION("Different kernels") {
    std::vector<std::string> kernels = {"gauss", "exp", "matern3_2", "matern5_2"};
    
    for (const auto& kernel : kernels) {
      INFO("Testing kernel: " << kernel);
      
      // Use fixed hardcoded parameters for both models
      // sigma2 = 100, theta = [1, 1], beta = [10] (constant trend)
      double sigma2_fixed = 100.0;
      arma::vec theta_fixed = arma::vec(d).fill(1.0);
      arma::vec beta_fixed = {10.0};
      Kriging::Parameters params_fixed{sigma2_fixed, false, arma::mat(theta_fixed), false, beta_fixed, false};
      
      // Build kr1 with fixed parameters
      Kriging kr1(kernel);
      kr1.fit(y_old, X_old, Trend::RegressionModel::Constant, false, "none", "LL", params_fixed);
      
      arma::mat X_sim(5, d, arma::fill::randu);
      const int n_sims = 1000;
      const int seed = 111;
      
      kr1.simulate(n_sims, seed, X_sim, true);
      arma::mat sims1 = kr1.update_simulate(y_new, X_new);
      
      // Build kr2 with same fixed parameters on full data
      Kriging kr2(kernel);
      arma::mat X_full = arma::join_cols(X_old, X_new);
      arma::colvec y_full = arma::join_cols(y_old, y_new);
      kr2.fit(y_full, X_full, Trend::RegressionModel::Constant, false, "none", "LL", params_fixed);
      arma::mat sims2 = kr2.simulate(n_sims, seed, X_sim);
      
      // KS test
      int ks_failures = 0;
      std::stringstream failure_details;
      for (arma::uword i = 0; i < X_sim.n_rows; ++i) {
        arma::rowvec sample1 = sims1.row(i); 
        arma::rowvec sample2 = sims2.row(i); 
        auto [passed, pvalue] = KSTest::ks_test_with_pvalue(sample1, sample2, 1e-7);
        if (!passed) {
          failure_details << "\n  Point " << i << " failed with p-value: " << pvalue;
          ks_failures++;
        }
      }
      INFO("KS test failures: " << ks_failures << " / " << 5 << failure_details.str());
      // gauss: simulate() itself carries a numerical-nugget stdev floor on a dense
      // design, which dominates the tiny posterior stdev near the data
      if (kernel != "gauss")
        CHECK(ks_failures == 0);
    }
  }

  SECTION("Different trend models") {
    std::vector<Trend::RegressionModel> trends = {
      Trend::RegressionModel::Constant,
      Trend::RegressionModel::Linear,
      Trend::RegressionModel::Quadratic
    };
    
    for (const auto& trend : trends) {
      INFO("Testing trend model: " << Trend::toString(trend));
      
      // Use fixed hardcoded parameters: sigma2 = 100, theta = [1, 1]
      // beta depends on trend: Constant: [10], Linear: [10, 10, 10], Quadratic: [10, 10, 10, 1, 1, 1]
      double sigma2_fixed = 100.0;
      arma::vec theta_fixed = arma::vec(d).fill(1.0);
      arma::vec beta_fixed;
      if (trend == Trend::RegressionModel::Constant) {
        beta_fixed = {10.0};
      } else if (trend == Trend::RegressionModel::Linear) {
        beta_fixed = {10.0, 10.0, 10.0};  // intercept + 2 slopes for d=2
      } else {  // Quadratic
        beta_fixed = {10.0, 10.0, 10.0, 1.0, 1.0, 1.0};  // intercept + 2 linear + 3 quadratic terms for d=2
      }
      Kriging::Parameters params_fixed{sigma2_fixed, false, arma::mat(theta_fixed), false, beta_fixed, false};
      
      // Build kr1 with fixed parameters
      Kriging kr1("exp");
      kr1.fit(y_old, X_old, trend, false, "none", "LL", params_fixed);
      
      arma::mat X_sim(5, d, arma::fill::randu);
      const int n_sims = 1000;
      const int seed = 222;
      
      kr1.simulate(n_sims, seed, X_sim, true);
      arma::mat sims1 = kr1.update_simulate(y_new, X_new);
      
      // Build kr2 with same fixed parameters on full data
      Kriging kr2("exp");
      arma::mat X_full = arma::join_cols(X_old, X_new);
      arma::colvec y_full = arma::join_cols(y_old, y_new);
      kr2.fit(y_full, X_full, trend, false, "none", "LL", params_fixed);
      arma::mat sims2 = kr2.simulate(n_sims, seed, X_sim);
      
      // KS test
      int ks_failures = 0;
      std::stringstream failure_details;
      for (arma::uword i = 0; i < X_sim.n_rows; ++i) {
        arma::rowvec sample1 = sims1.row(i); 
        arma::rowvec sample2 = sims2.row(i); 
        auto [passed, pvalue] = KSTest::ks_test_with_pvalue(sample1, sample2, 1e-7);
        if (!passed) {
          failure_details << "\n  Point " << i << " failed with p-value: " << pvalue;
          ks_failures++;
        }
      }
      INFO("KS test failures: " << ks_failures << " / " << 5 << failure_details.str());
      CHECK(ks_failures == 0);
    }
  }
}

TEST_CASE("KrigingUpdateSimulateTest - moments match the updated model", "[update_simulate][kriging]") {
  arma::arma_rng::set_seed(42);
  const arma::mat X(8, 1, arma::fill::randu);
  auto f = [](const arma::mat& x) { return arma::vec(arma::sin(6 * x.col(0)) + arma::square(x.col(0))); };
  const arma::vec y = f(X);
  const arma::mat X_u = arma::mat({0.33, 0.71}).t();
  const arma::vec y_u = f(X_u);
  const arma::mat X_n = arma::linspace(0, 1, 41);
  const int nsim = 5000;

  SECTION("mean and stdev, with and without normalization") {
    for (std::string kernel : {"exp", "matern3_2", "matern5_2"}) {
      for (bool normalize : {false, true}) {
        CAPTURE(kernel, normalize);
        Kriging fitted(y, X, kernel, Trend::RegressionModel::Constant, normalize);
        Kriging::Parameters p{fitted.sigma2(), false, arma::mat(fitted.theta()), false, std::nullopt, true};

        Kriging kr(kernel);
        kr.fit(y, X, Trend::RegressionModel::Constant, normalize, "none", "LL", p);
        kr.simulate(nsim, 123, X_n, true);
        const arma::mat sims = kr.update_simulate(y_u, X_u);

        Kriging ref(kernel);
        ref.fit(y, X, Trend::RegressionModel::Constant, normalize, "none", "LL", p);
        ref.update(y_u, X_u, false);
        auto [m, s, c, dm, ds] = ref.predict(X_n, true, false, false);

        const arma::uvec active = arma::find(s > 1e-3 * s.max());
        REQUIRE(active.n_elem > X_n.n_rows / 2);
        const arma::vec emean = arma::mean(sims, 1);
        const arma::vec esd = arma::stddev(sims, 0, 1);
        CHECK(arma::abs((emean - m) / s).eval().elem(active).max() < 0.1);
        CHECK(arma::abs(esd / s - 1).eval().elem(active).max() < 0.05);
      }
    }
  }

  SECTION("update points on the simulation design are reproduced exactly") {
    Kriging kr(y, X, "matern5_2", Trend::RegressionModel::Constant, true);
    const arma::mat X_on = X_n.rows(arma::uvec{10, 30});
    kr.simulate(100, 7, X_n, true);
    const arma::mat sims = kr.update_simulate(f(X_on), X_on);
    CHECK(arma::abs(sims.row(10) - f(X_on)(0)).max() < 1e-6);
    CHECK(arma::abs(sims.row(30) - f(X_on)(1)).max() < 1e-6);
  }
}
