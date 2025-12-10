// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
// clang-format on

TEST_CASE("KrigingUpdateTest - Updated model equals combined fit", "[update][kriging]") {
  arma::arma_rng::set_seed(123);

  // Generate initial training data
  const arma::uword n_old = 10;
  const arma::uword n_new = 1;
  const arma::uword d = 2;
  
  arma::mat X_old(n_old, d, arma::fill::randu);
  arma::colvec y_old(n_old);
  
  // Use a simple test function
  auto test_function = [](const arma::rowvec& x) {
    return x(0) * x(0) + x(1) * x(1);
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
  
  // Combine old and new data
  arma::mat X_combined = arma::join_cols(X_old, X_new);
  arma::colvec y_combined = arma::join_cols(y_old, y_new);

  SECTION("Update with refit equals combined fit") {
    // Fit model on old data
    Kriging kr_updated("gauss");
    Kriging::Parameters params{std::nullopt, true, std::nullopt, true, std::nullopt, true};
    kr_updated.fit(y_old, X_old, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);
    
    // Update with new data
    kr_updated.update(y_new, X_new, true);
    
    // Fit model on combined data
    Kriging kr_combined("gauss");
    kr_combined.fit(y_combined, X_combined, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);
    
    // Compare parameters
    INFO("Updated theta: " << kr_updated.theta().t());
    INFO("Combined theta: " << kr_combined.theta().t());
    INFO("Updated sigma2: " << kr_updated.sigma2());
    INFO("Combined sigma2: " << kr_combined.sigma2());
    INFO("Updated beta: " << kr_updated.beta().t());
    INFO("Combined beta: " << kr_combined.beta().t());
    
    // Check theta (should be very close) - using relative precision 0.05 (5%)
    REQUIRE(arma::all(kr_combined.theta() != 0.0));  // Ensure no zero values for relative comparison
    CHECK(arma::approx_equal(kr_updated.theta(), kr_combined.theta(), "reldiff", 0.05));
    
    // Check sigma2 - using relative precision 0.05 (5%)
    REQUIRE(kr_combined.sigma2() != 0.0);  // Ensure non-zero for relative comparison
    CHECK(std::abs(kr_updated.sigma2() - kr_combined.sigma2()) / std::abs(kr_combined.sigma2()) < 0.05);
    
    // Check beta - using relative precision 0.05 (5%)
    REQUIRE(arma::all(kr_combined.beta() != 0.0));  // Ensure no zero values for relative comparison
    CHECK(arma::approx_equal(kr_updated.beta(), kr_combined.beta(), "reldiff", 0.05));
    
    // Test predictions on new points
    arma::mat X_test(5, d, arma::fill::randu);
    
    auto pred_updated = kr_updated.predict(X_test, true, false, false);
    auto pred_combined = kr_combined.predict(X_test, true, false, false);
    
    INFO("Max prediction difference: " << arma::max(arma::abs(std::get<0>(pred_updated) - std::get<0>(pred_combined))));
    
    CHECK(arma::approx_equal(std::get<0>(pred_updated), std::get<0>(pred_combined), "absdiff", 1e-2));
  }

  SECTION("Multiple updates equal combined fit") {
    // Fit model on old data
    Kriging kr_updated("gauss");
    Kriging::Parameters params{std::nullopt, true, std::nullopt, true, std::nullopt, true};
    kr_updated.fit(y_old, X_old, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);
    
    // Update one point at a time
    for (arma::uword i = 0; i < n_new; ++i) {
      arma::vec y_single = y_new.row(i);
      arma::mat X_single = X_new.row(i);
      kr_updated.update(y_single, X_single, true);
    }
    
    // Fit model on combined data
    Kriging kr_combined("gauss");
    kr_combined.fit(y_combined, X_combined, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);
    
    // Test predictions
    arma::mat X_test(5, d, arma::fill::randu);
    
    auto pred_updated = kr_updated.predict(X_test, true, false, false);
    auto pred_combined = kr_combined.predict(X_test, true, false, false);
    
    INFO("Max prediction difference: " << arma::max(arma::abs(std::get<0>(pred_updated) - std::get<0>(pred_combined))));
    
    CHECK(arma::approx_equal(std::get<0>(pred_updated), std::get<0>(pred_combined), "absdiff", 1e-2));
  }

  SECTION("Different kernels") {
    std::vector<std::string> kernels = {"gauss", "exp", "matern3_2", "matern5_2"};
    
    for (const auto& kernel : kernels) {
      INFO("Testing kernel: " << kernel);
      
      // Fit and update
      Kriging kr_updated(kernel);
      Kriging::Parameters params{std::nullopt, true, std::nullopt, true, std::nullopt, true};
      kr_updated.fit(y_old, X_old, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);
      kr_updated.update(y_new, X_new, true);
      
      // Combined fit
      Kriging kr_combined(kernel);
      kr_combined.fit(y_combined, X_combined, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);
      
      // Test predictions
      arma::mat X_test(3, d, arma::fill::randu);
      auto pred_updated = kr_updated.predict(X_test, true, false, false);
      auto pred_combined = kr_combined.predict(X_test, true, false, false);
      
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
      Kriging kr_updated("gauss");
      Kriging::Parameters params{std::nullopt, true, std::nullopt, true, std::nullopt, true};
      kr_updated.fit(y_old, X_old, trend, false, "BFGS", "LL", params);
      kr_updated.update(y_new, X_new, true);
      
      // Combined fit
      Kriging kr_combined("gauss");
      kr_combined.fit(y_combined, X_combined, trend, false, "BFGS", "LL", params);
      
      // Test predictions
      arma::mat X_test(3, d, arma::fill::randu);
      auto pred_updated = kr_updated.predict(X_test, true, false, false);
      auto pred_combined = kr_combined.predict(X_test, true, false, false);
      
      CHECK(arma::approx_equal(std::get<0>(pred_updated), std::get<0>(pred_combined), "absdiff", 1e-2));
    }
  }
}
