#ifndef LIBKRIGING_SRC_LIB_SEPARABLEDLN_HPP
#define LIBKRIGING_SRC_LIB_SEPARABLEDLN_HPP

// Private (not installed) helper: one-dimension factors of the kernels'
// d ln k / d theta (Covariance::DlnCovDtheta_*, Covariance.cpp). The kernels
// are separable: component k of DlnCovDtheta_*(dX, theta) only depends on
// (dX[k], theta[k]). Each factor is built once per dimension (per gradient
// evaluation) with its theta-only constants, so that evaluating it on a pair
// needs at most one division, against one to three in Covariance.cpp. Same
// formulas, rounded differently (a few ulps per value).

#include <cmath>

namespace libKriging {
namespace detail {

struct DlnGauss {  // dx^2 / theta^3
  double c;
  explicit DlnGauss(double theta) : c(1.0 / (theta * theta * theta)) {}
  double operator()(double dx) const { return dx * dx * c; }
};

struct DlnExp {  // |dx| / theta^2
  double c;
  explicit DlnExp(double theta) : c(1.0 / (theta * theta)) {}
  double operator()(double dx) const { return std::abs(dx) * c; }
};

struct DlnMatern32 {  // u^2 / (1 + u) / theta, u = sqrt(3) |dx| / theta
  double s, inv_theta;
  explicit DlnMatern32(double theta) : s(std::sqrt(3.0) / theta), inv_theta(1.0 / theta) {}
  double operator()(double dx) const {
    const double u = s * std::abs(dx);
    return (u * u) / (1.0 + u) * inv_theta;
  }
};

struct DlnMatern52 {  // a b / (a + b) / theta, a = 1 + u, b = u^2 / 3, u = sqrt(5) |dx| / theta
  double s, inv_theta;
  explicit DlnMatern52(double theta) : s(std::sqrt(5.0) / theta), inv_theta(1.0 / theta) {}
  double operator()(double dx) const {
    const double u = s * std::abs(dx);
    const double a = 1.0 + u;
    const double b = (u * u) / 3.0;
    return (a * b) / (a + b) * inv_theta;
  }
};

}  // namespace detail
}  // namespace libKriging

#endif  // LIBKRIGING_SRC_LIB_SEPARABLEDLN_HPP
