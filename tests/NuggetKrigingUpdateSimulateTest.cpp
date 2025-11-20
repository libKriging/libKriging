// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/NuggetKriging.hpp"
#include "ks_test.hpp"
#include <sstream>
// clang-format on

TEST_CASE("NuggetKrigingUpdateSimulateTest - Update simulate equals updated model simulate", "[update_simulate][nuggetkriging]") {
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
    // nugget = 0.01, sigma2 = 100, theta = [1, 1], beta = [10] (constant trend)
    double nugget_fixed = 0.01;
    double sigma2_fixed = 100.0;
    arma::vec theta_fixed = arma::vec(d).fill(1.0);
    arma::vec beta_fixed = {10.0};
    NuggetKriging::Parameters params_fixed{arma::vec(1).fill(nugget_fixed), false, arma::vec(1).fill(sigma2_fixed), false, arma::mat(theta_fixed), false, beta_fixed, false};

    // Build nk1 with fixed parameters
    NuggetKriging nk1("gauss");
    nk1.fit(y_old, X_old, Trend::RegressionModel::Constant, false, "none", "LL", params_fixed);

    // Simulation points
    const arma::uword n_sim_points = 10;
    arma::mat X_sim(n_sim_points, d, arma::fill::randu);
    
    // Method 1: Use update_simulate
    const int n_sims = 1000;
    const int seed = 789;
    nk1.simulate(n_sims, seed, X_sim, false, true);  // Store for update_simulate
    arma::mat sims1 = nk1.update_simulate(y_new, X_new);
    
    // Method 2: Build nk2 with same fixed parameters on full data
    NuggetKriging nk2("gauss");
    arma::mat X_full = arma::join_cols(X_old, X_new);
    arma::colvec y_full = arma::join_cols(y_old, y_new);
    nk2.fit(y_full, X_full, Trend::RegressionModel::Constant, false, "none", "LL", params_fixed);
    arma::mat sims2 = nk2.simulate(n_sims, seed, X_sim, false, false);

    // Save simulations to CSV for debugging
    sims1.save("sims1_nugget_update_simulate.csv", arma::csv_ascii);
    sims2.save("sims2_nugget_full_model.csv", arma::csv_ascii);
    X_sim.save("X_sim_nugget.csv", arma::csv_ascii);

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
    // nugget = 0.01, sigma2 = 100, theta = [1, 1], beta = [10] (constant trend)
    double nugget_fixed = 0.01;
    double sigma2_fixed = 100.0;
    arma::vec theta_fixed = arma::vec(d).fill(1.0);
    arma::vec beta_fixed = {10.0};
    NuggetKriging::Parameters params_fixed{arma::vec(1).fill(nugget_fixed), false, arma::vec(1).fill(sigma2_fixed), false, arma::mat(theta_fixed), false, beta_fixed, false};
    
    // Build nk1 with fixed parameters
    NuggetKriging nk1("gauss");
    nk1.fit(y_old, X_old, Trend::RegressionModel::Constant, false, "none", "LL", params_fixed);
    
    // Simulation points
    const arma::uword n_sim_points = 5;
    arma::mat X_sim(n_sim_points, d, arma::fill::randu);
    
    const int n_sims = 1000;
    const int seed = 999;
    
    // Use update_simulate with multiple new points
    nk1.simulate(n_sims, seed, X_sim, false, true);
    arma::mat sims1 = nk1.update_simulate(y_new, X_new);
    
    // Build nk2 with same fixed parameters on full data
    NuggetKriging nk2("gauss");
    arma::mat X_full = arma::join_cols(X_old, X_new);
    arma::colvec y_full = arma::join_cols(y_old, y_new);
    nk2.fit(y_full, X_full, Trend::RegressionModel::Constant, false, "none", "LL", params_fixed);
    arma::mat sims2 = nk2.simulate(n_sims, seed, X_sim, false, false);
    
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
      // nugget = 0.01, sigma2 = 100, theta = [1, 1], beta = [10] (constant trend)
      double nugget_fixed = 0.01;
      double sigma2_fixed = 100.0;
      arma::vec theta_fixed = arma::vec(d).fill(1.0);
      arma::vec beta_fixed = {10.0};
      NuggetKriging::Parameters params_fixed{arma::vec(1).fill(nugget_fixed), false, arma::vec(1).fill(sigma2_fixed), false, arma::mat(theta_fixed), false, beta_fixed, false};
      
      // Build nk1 with fixed parameters
      NuggetKriging nk1(kernel);
      nk1.fit(y_old, X_old, Trend::RegressionModel::Constant, false, "none", "LL", params_fixed);
      
      arma::mat X_sim(5, d, arma::fill::randu);
      const int n_sims = 1000;
      const int seed = 111;
      
      nk1.simulate(n_sims, seed, X_sim, false, true);
      arma::mat sims1 = nk1.update_simulate(y_new, X_new);
      
      // Build nk2 with same fixed parameters on full data
      NuggetKriging nk2(kernel);
      arma::mat X_full = arma::join_cols(X_old, X_new);
      arma::colvec y_full = arma::join_cols(y_old, y_new);
      nk2.fit(y_full, X_full, Trend::RegressionModel::Constant, false, "none", "LL", params_fixed);
      arma::mat sims2 = nk2.simulate(n_sims, seed, X_sim, false, false);
      
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

  SECTION("Different trend models") {
    std::vector<Trend::RegressionModel> trends = {
      Trend::RegressionModel::Constant,
      Trend::RegressionModel::Linear,
      Trend::RegressionModel::Quadratic
    };
    
    for (const auto& trend : trends) {
      INFO("Testing trend model: " << Trend::toString(trend));
      
      // Use fixed hardcoded parameters: nugget = 0.01, sigma2 = 100, theta = [1, 1]
      // beta depends on trend: Constant: [10], Linear: [10, 10, 10], Quadratic: [10, 10, 10, 1, 1, 1]
      double nugget_fixed = 0.01;
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
      NuggetKriging::Parameters params_fixed{arma::vec(1).fill(nugget_fixed), false, arma::vec(1).fill(sigma2_fixed), false, arma::mat(theta_fixed), false, beta_fixed, false};
      
      // Build nk1 with fixed parameters
      NuggetKriging nk1("gauss");
      nk1.fit(y_old, X_old, trend, false, "none", "LL", params_fixed);
      
      arma::mat X_sim(5, d, arma::fill::randu);
      const int n_sims = 1000;
      const int seed = 222;
      
      nk1.simulate(n_sims, seed, X_sim, false, true);
      arma::mat sims1 = nk1.update_simulate(y_new, X_new);
      
      // Build nk2 with same fixed parameters on full data
      NuggetKriging nk2("gauss");
      arma::mat X_full = arma::join_cols(X_old, X_new);
      arma::colvec y_full = arma::join_cols(y_old, y_new);
      nk2.fit(y_full, X_full, trend, false, "none", "LL", params_fixed);
      arma::mat sims2 = nk2.simulate(n_sims, seed, X_sim, false, false);
      
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
