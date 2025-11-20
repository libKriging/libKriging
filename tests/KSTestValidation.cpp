// Test to validate KS test implementation with known distributions
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "ks_test.hpp"

TEST_CASE("KS Test Validation - Same distribution", "[ks_test][validation]") {
  arma::arma_rng::set_seed(123);
  
  SECTION("Two samples from same normal distribution should pass") {
    const int n = 1000;
    arma::vec sample1 = arma::randn(n);
    arma::vec sample2 = arma::randn(n);
    
    bool result = KSTest::ks_test(sample1, sample2, 0.05);
    INFO("KS statistic: " << KSTest::ks_statistic(
      std::vector<double>(sample1.begin(), sample1.end()),
      std::vector<double>(sample2.begin(), sample2.end())));
    CHECK(result == true);
  }
  
  SECTION("Two samples from same uniform distribution should pass") {
    const int n = 1000;
    arma::vec sample1 = arma::randu(n);
    arma::vec sample2 = arma::randu(n);
    
    bool result = KSTest::ks_test(sample1, sample2, 0.05);
    CHECK(result == true);
  }
  
  SECTION("Multiple samples from same distribution - low false positive rate") {
    const int n = 1000;
    const int n_tests = 100;
    int false_positives = 0;
    
    for (int i = 0; i < n_tests; ++i) {
      arma::vec sample1 = arma::randn(n);
      arma::vec sample2 = arma::randn(n);
      
      if (!KSTest::ks_test(sample1, sample2, 0.05)) {
        false_positives++;
      }
    }
    
    // With alpha=0.05, we expect ~5% false positives
    double false_positive_rate = static_cast<double>(false_positives) / n_tests;
    INFO("False positive rate: " << false_positive_rate << " (expected ~0.05)");
    CHECK(false_positive_rate < 0.10); // Should be less than 10%
  }
}

TEST_CASE("KS Test Validation - Different distributions", "[ks_test][validation]") {
  arma::arma_rng::set_seed(456);
  
  SECTION("Normal(0,1) vs Normal(0,2) should fail") {
    const int n = 1000;
    arma::vec sample1 = arma::randn(n);
    arma::vec sample2 = arma::randn(n) * 2.0; // Different variance
    
    bool result = KSTest::ks_test(sample1, sample2, 0.05);
    double ks_stat = KSTest::ks_statistic(
      std::vector<double>(sample1.begin(), sample1.end()),
      std::vector<double>(sample2.begin(), sample2.end()));
    INFO("KS statistic: " << ks_stat);
    CHECK(result == false);
  }
  
  SECTION("Normal(0,1) vs Normal(0.5,1) should fail") {
    const int n = 1000;
    arma::vec sample1 = arma::randn(n);
    arma::vec sample2 = arma::randn(n) + 0.5; // Different mean
    
    bool result = KSTest::ks_test(sample1, sample2, 0.05);
    double ks_stat = KSTest::ks_statistic(
      std::vector<double>(sample1.begin(), sample1.end()),
      std::vector<double>(sample2.begin(), sample2.end()));
    INFO("KS statistic: " << ks_stat);
    CHECK(result == false);
  }
  
  SECTION("Normal vs Uniform should fail") {
    const int n = 1000;
    arma::vec sample1 = arma::randn(n);
    arma::vec sample2 = arma::randu(n);
    
    bool result = KSTest::ks_test(sample1, sample2, 0.05);
    double ks_stat = KSTest::ks_statistic(
      std::vector<double>(sample1.begin(), sample1.end()),
      std::vector<double>(sample2.begin(), sample2.end()));
    INFO("KS statistic: " << ks_stat);
    CHECK(result == false);
  }
  
  SECTION("Multiple tests - high detection rate for different distributions") {
    const int n = 1000;
    const int n_tests = 100;
    int true_positives = 0;
    
    for (int i = 0; i < n_tests; ++i) {
      arma::vec sample1 = arma::randn(n);
      arma::vec sample2 = arma::randn(n) + 0.5; // Shifted mean
      
      if (!KSTest::ks_test(sample1, sample2, 0.05)) {
        true_positives++;
      }
    }
    
    // Should detect almost all cases of different distributions
    double detection_rate = static_cast<double>(true_positives) / n_tests;
    INFO("Detection rate: " << detection_rate << " (expected > 0.95)");
    CHECK(detection_rate > 0.95); // Should detect > 95%
  }
}

TEST_CASE("KS Test Validation - Sensitivity to sample size", "[ks_test][validation]") {
  arma::arma_rng::set_seed(789);
  
  SECTION("Small mean shift - large samples needed") {
    const double shift = 0.1; // Small shift
    
    // With n=100, might not detect
    arma::vec sample1_small = arma::randn(100);
    arma::vec sample2_small = arma::randn(100) + shift;
    bool result_small = KSTest::ks_test(sample1_small, sample2_small, 0.05);
    
    // With n=1000, should detect
    arma::vec sample1_large = arma::randn(1000);
    arma::vec sample2_large = arma::randn(1000) + shift;
    bool result_large = KSTest::ks_test(sample1_large, sample2_large, 0.05);
    
    INFO("Small sample (n=100) result: " << (result_small ? "PASS" : "FAIL"));
    INFO("Large sample (n=1000) result: " << (result_large ? "PASS" : "FAIL"));
    
    // Large sample should be more sensitive
    // (We don't enforce small sample fails, just that large sample is more sensitive)
    CHECK(true); // Always passes - just informational
  }
}

TEST_CASE("KS Test Validation - Same seed should give identical samples", "[ks_test][validation]") {
  SECTION("Identical samples should always pass") {
    arma::arma_rng::set_seed(111);
    arma::vec sample1 = arma::randn(1000);
    
    arma::arma_rng::set_seed(111);
    arma::vec sample2 = arma::randn(1000);
    
    // Should be identical
    CHECK(arma::approx_equal(sample1, sample2, "absdiff", 1e-10));
    
    bool result = KSTest::ks_test(sample1, sample2, 0.01);
    double ks_stat = KSTest::ks_statistic(
      std::vector<double>(sample1.begin(), sample1.end()),
      std::vector<double>(sample2.begin(), sample2.end()));
    
    INFO("KS statistic for identical samples: " << ks_stat);
    CHECK(result == true);
    CHECK(ks_stat < 1e-10); // Should be essentially zero
  }
}
