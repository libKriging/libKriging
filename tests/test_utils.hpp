#ifndef LIBKRIGING_TEST_UTILS_HPP
#define LIBKRIGING_TEST_UTILS_HPP

#include <armadillo>

double relative_error(const arma::vec& x, const arma::vec& y) {
  double x_norm = arma::norm(x, "inf");
  double y_norm = arma::norm(y, "inf");

  if (x_norm > 0 || y_norm > 0) {
    double diff_norm = arma::norm(x-y, "inf");
    return diff_norm / std::max(x_norm, y_norm);
  } else {
    return 0.;
  }
}

#endif  // LIBKRIGING_TEST_UTILS_HPP
