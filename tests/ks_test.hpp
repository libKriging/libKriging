#ifndef LIBKRIGING_KS_TEST_HPP
#define LIBKRIGING_KS_TEST_HPP

#include <algorithm>
#include <cmath>
#include <vector>

namespace KSTest {

// Compute the Kolmogorov-Smirnov statistic for two samples
// Returns the maximum absolute difference between ECDFs
inline double ks_statistic(const std::vector<double>& sample1, const std::vector<double>& sample2) {
  std::vector<double> s1 = sample1;
  std::vector<double> s2 = sample2;
  
  // Sort both samples
  std::sort(s1.begin(), s1.end());
  std::sort(s2.begin(), s2.end());
  
  const size_t n1 = s1.size();
  const size_t n2 = s2.size();
  
  double max_diff = 0.0;
  size_t i1 = 0, i2 = 0;
  
  // Walk through both sorted samples
  while (i1 < n1 && i2 < n2) {
    double ecdf1 = static_cast<double>(i1) / n1;
    double ecdf2 = static_cast<double>(i2) / n2;
    
    double diff = std::abs(ecdf1 - ecdf2);
    max_diff = std::max(max_diff, diff);
    
    if (s1[i1] < s2[i2]) {
      i1++;
    } else if (s1[i1] > s2[i2]) {
      i2++;
    } else {
      i1++;
      i2++;
    }
  }
  
  // Check remaining elements
  if (i1 < n1) {
    double ecdf1 = static_cast<double>(i1) / n1;
    double ecdf2 = 1.0;
    max_diff = std::max(max_diff, std::abs(ecdf1 - ecdf2));
  }
  if (i2 < n2) {
    double ecdf1 = 1.0;
    double ecdf2 = static_cast<double>(i2) / n2;
    max_diff = std::max(max_diff, std::abs(ecdf1 - ecdf2));
  }
  
  return max_diff;
}

// Compute critical value for KS test at given significance level
// Using asymptotic approximation for large samples
inline double ks_critical_value(size_t n1, size_t n2, double alpha = 0.05) {
  double n_eff = std::sqrt((n1 * n2) / static_cast<double>(n1 + n2));
  
  // Asymptotic critical values for common alpha levels
  // c_alpha values based on Kolmogorov-Smirnov distribution
  double c_alpha;
  if (alpha <= 1e-9) {
    c_alpha = 2.81;
  } else if (alpha <= 1e-7) {
    c_alpha = 2.58;
  } else if (alpha <= 1e-5) {
    c_alpha = 2.33;
  } else if (alpha <= 1e-3) {
    c_alpha = 2.05;
  } else if (alpha <= 0.001) {
    c_alpha = 1.95;
  } else if (alpha <= 0.01) {
    c_alpha = 1.63;
  } else if (alpha <= 0.05) {
    c_alpha = 1.36;
  } else if (alpha <= 0.10) {
    c_alpha = 1.22;
  } else {
    c_alpha = 1.07;
  }
  
  return c_alpha / n_eff;
}

// Compute approximate p-value for KS test using asymptotic distribution
inline double ks_pvalue(double statistic, size_t n1, size_t n2) {
  double n_eff = std::sqrt((n1 * n2) / static_cast<double>(n1 + n2));
  double lambda = n_eff * statistic;
  
  // Asymptotic p-value approximation (Kolmogorov distribution)
  // P(D > d) ≈ 2 * sum_{k=1}^∞ (-1)^(k-1) * exp(-2k^2 * lambda^2)
  // Using first few terms for approximation
  double pval = 0.0;
  for (int k = 1; k <= 10; ++k) {
    double term = std::pow(-1.0, k - 1) * std::exp(-2.0 * k * k * lambda * lambda);
    pval += term;
    if (std::abs(term) < 1e-10) break;
  }
  pval *= 2.0;
  
  return std::min(1.0, std::max(0.0, pval));
}

// Perform KS test and return true if samples are from same distribution
// (i.e., fail to reject null hypothesis)
inline bool ks_test(const std::vector<double>& sample1, 
                    const std::vector<double>& sample2,
                    double alpha = 0.05) {
  if (sample1.empty() || sample2.empty()) {
    return false;
  }
  
  double stat = ks_statistic(sample1, sample2);
  double critical = ks_critical_value(sample1.size(), sample2.size(), alpha);
  
  return stat <= critical;
}

// Perform KS test and return both result and p-value
inline std::pair<bool, double> ks_test_with_pvalue(const std::vector<double>& sample1, 
                                                     const std::vector<double>& sample2,
                                                     double alpha = 0.05) {
  if (sample1.empty() || sample2.empty()) {
    return {false, 1.0};
  }
  
  double stat = ks_statistic(sample1, sample2);
  double critical = ks_critical_value(sample1.size(), sample2.size(), alpha);
  double pval = ks_pvalue(stat, sample1.size(), sample2.size());
  
  return {stat <= critical, pval};
}

// Convenience function for Armadillo vectors
inline bool ks_test(const arma::vec& sample1, const arma::vec& sample2, double alpha = 0.05) {
  std::vector<double> s1(sample1.begin(), sample1.end());
  std::vector<double> s2(sample2.begin(), sample2.end());
  return ks_test(s1, s2, alpha);
}

// Convenience function for Armadillo row vectors
inline bool ks_test(const arma::rowvec& sample1, const arma::rowvec& sample2, double alpha = 0.05) {
  std::vector<double> s1(sample1.begin(), sample1.end());
  std::vector<double> s2(sample2.begin(), sample2.end());
  return ks_test(s1, s2, alpha);
}

// Convenience function for Armadillo row vectors with p-value
inline std::pair<bool, double> ks_test_with_pvalue(const arma::rowvec& sample1, const arma::rowvec& sample2, double alpha = 0.05) {
  std::vector<double> s1(sample1.begin(), sample1.end());
  std::vector<double> s2(sample2.begin(), sample2.end());
  return ks_test_with_pvalue(s1, s2, alpha);
}

}  // namespace KSTest

#endif  // LIBKRIGING_KS_TEST_HPP
