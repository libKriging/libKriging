// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/NuggetKriging.hpp"
// clang-format on

TEST_CASE("NuggetKrigingUpdateTest - Updated model equals combined fit", "[update][nuggetkriging]") {
  arma::arma_rng::set_seed(123);

  // Generate initial training data
  const arma::uword n_old = 10;
  const arma::uword n_new = 1;
  const arma::uword d = 2;
  
  arma::mat X_old(n_old, d, arma::fill::randu);
  arma::colvec y_old(n_old);
  
  // Use a simple test function with noise
  auto test_function = [](const arma::rowvec& x) {
    return x(0) * x(0) + x(1) * x(1);
  };
  
  for (arma::uword i = 0; i < n_old; ++i) {
    y_old(i) = test_function(X_old.row(i)) + 0.01 * arma::randn();
  }
  
  // Generate new data points
  arma::mat X_new(n_new, d, arma::fill::randu);
  arma::colvec y_new(n_new);
  
  for (arma::uword i = 0; i < n_new; ++i) {
    y_new(i) = test_function(X_new.row(i)) + 0.01 * arma::randn();
  }
  
  // Combine old and new data
  arma::mat X_combined = arma::join_cols(X_old, X_new);
  arma::colvec y_combined = arma::join_cols(y_old, y_new);

  SECTION("Update with refit equals combined fit") {
    // Fit model on old data
    NuggetKriging nk_updated("gauss");
    NuggetKriging::Parameters params{std::nullopt, true, std::nullopt, true, std::nullopt, true, std::nullopt, true};
    nk_updated.fit(y_old, X_old, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);
    
    // Update with new data
    nk_updated.update(y_new, X_new, true);
    
    // Fit model on combined data
    NuggetKriging nk_combined("gauss");
    nk_combined.fit(y_combined, X_combined, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);
    
    // Compare parameters
    INFO("Updated theta: " << nk_updated.theta().t());
    INFO("Combined theta: " << nk_combined.theta().t());
    INFO("Updated sigma2: " << nk_updated.sigma2());
    INFO("Combined sigma2: " << nk_combined.sigma2());
    INFO("Updated nugget: " << nk_updated.nugget());
    INFO("Combined nugget: " << nk_combined.nugget());
    INFO("Updated beta: " << nk_updated.beta().t());
    INFO("Combined beta: " << nk_combined.beta().t());
    
    // Check theta (should be very close) - using relative precision 0.05 (5%)
    // Use relative comparison, which handles zero values by falling back to absolute comparison
    CHECK(arma::approx_equal(nk_updated.theta(), nk_combined.theta(), "reldiff", 0.05));
    
    // Check sigma2 - using relative precision 0.05 (5%)
    if (nk_combined.sigma2() == 0.0) {
      CHECK(nk_updated.sigma2() == 0.0);  // If combined is zero, updated should also be zero
    } else {
      CHECK(std::abs(nk_updated.sigma2() - nk_combined.sigma2()) / std::abs(nk_combined.sigma2()) < 0.05);
    }
    
    // Check nugget - using relative precision 0.05 (5%)
    if (nk_combined.nugget() == 0.0) {
      CHECK(nk_updated.nugget() == 0.0);  // If combined is zero, updated should also be zero
    } else {
      CHECK(std::abs(nk_updated.nugget() - nk_combined.nugget()) / std::abs(nk_combined.nugget()) < 0.05);
    }
    
    // Check beta - using relative precision 0.05 (5%)
    // Use relative comparison, which handles zero values by falling back to absolute comparison
    CHECK(arma::approx_equal(nk_updated.beta(), nk_combined.beta(), "reldiff", 0.05));
    
    // Test predictions on new points (main validation)
    arma::mat X_test(5, d, arma::fill::randu);
    
    auto pred_updated = nk_updated.predict(X_test, true, false, false);
    auto pred_combined = nk_combined.predict(X_test, true, false, false);
    
    INFO("Max prediction difference: " << arma::max(arma::abs(std::get<0>(pred_updated) - std::get<0>(pred_combined))));
    
    CHECK(arma::approx_equal(std::get<0>(pred_updated), std::get<0>(pred_combined), "absdiff", 1e-2));
  }

  SECTION("Multiple updates equal combined fit") {
    // Fit model on old data
    NuggetKriging nk_updated("gauss");
    NuggetKriging::Parameters params{std::nullopt, true, std::nullopt, true, std::nullopt, true, std::nullopt, true};
    nk_updated.fit(y_old, X_old, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);
    
    // Update one point at a time
    for (arma::uword i = 0; i < n_new; ++i) {
      arma::vec y_single = y_new.row(i);
      arma::mat X_single = X_new.row(i);
      nk_updated.update(y_single, X_single, true);
    }
    
    // Fit model on combined data
    NuggetKriging nk_combined("gauss");
    nk_combined.fit(y_combined, X_combined, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);
    
    // Test predictions
    arma::mat X_test(5, d, arma::fill::randu);
    
    auto pred_updated = nk_updated.predict(X_test, true, false, false);
    auto pred_combined = nk_combined.predict(X_test, true, false, false);
    
    INFO("Max prediction difference: " << arma::max(arma::abs(std::get<0>(pred_updated) - std::get<0>(pred_combined))));
    
    CHECK(arma::approx_equal(std::get<0>(pred_updated), std::get<0>(pred_combined), "absdiff", 1e-2));
  }

  SECTION("Different kernels") {
    std::vector<std::string> kernels = {"gauss", "exp", "matern3_2", "matern5_2"};
    
    for (const auto& kernel : kernels) {
      INFO("Testing kernel: " << kernel);
      
      // Fit and update
      NuggetKriging nk_updated(kernel);
      NuggetKriging::Parameters params{std::nullopt, true, std::nullopt, true, std::nullopt, true, std::nullopt, true};
      nk_updated.fit(y_old, X_old, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);
      nk_updated.update(y_new, X_new, true);
      
      // Combined fit
      NuggetKriging nk_combined(kernel);
      nk_combined.fit(y_combined, X_combined, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);
      
      // Test predictions
      arma::mat X_test(3, d, arma::fill::randu);
      auto pred_updated = nk_updated.predict(X_test, true, false, false);
      auto pred_combined = nk_combined.predict(X_test, true, false, false);
      
      CHECK(arma::approx_equal(std::get<0>(pred_updated), std::get<0>(pred_combined), "absdiff", 1e-2));
    }
  }

  SECTION("Different trend models") {
    std::vector<Trend::RegressionModel> trends = {
      Trend::RegressionModel::Constant,
      Trend::RegressionModel::Linear,
      Trend::RegressionModel::Quadratic
    };
    
    for (const auto& trend : trends) {
      INFO("Testing trend model");
      
      // Fit and update
      NuggetKriging nk_updated("gauss");
      NuggetKriging::Parameters params{std::nullopt, true, std::nullopt, true, std::nullopt, true, std::nullopt, true};
      nk_updated.fit(y_old, X_old, trend, false, "BFGS", "LL", params);
      nk_updated.update(y_new, X_new, true);
      
      // Combined fit
      NuggetKriging nk_combined("gauss");
      nk_combined.fit(y_combined, X_combined, trend, false, "BFGS", "LL", params);
      
      // Test predictions
      arma::mat X_test(3, d, arma::fill::randu);
      auto pred_updated = nk_updated.predict(X_test, true, false, false);
      auto pred_combined = nk_combined.predict(X_test, true, false, false);
      
      CHECK(arma::approx_equal(std::get<0>(pred_updated), std::get<0>(pred_combined), "absdiff", 1e-2));
    }
  }
}
