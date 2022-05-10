// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio
#include <cmath>
// clang-format on

#include "libKriging/Covariance.hpp"

#include "libKriging/utils/lk_armadillo.hpp"

#include <cassert>
#include <tuple>
#include <vector>

//' @ref: https://github.com/psbiomech/dace-toolbox-source/blob/master/dace.pdf
//'  (where CovMatrix<-R, Ft<-M, C<-T, rho<-z)
//' @ref: https://github.com/cran/DiceKriging/blob/master/R/kmEstimate.R (same variables names)

//' @ref https://github.com/cran/DiceKriging/blob/master/src/CovFuns.c
// Covariance function on normalized data

std::function<double(const arma::vec&, const arma::vec&)> Covariance::Cov_gauss = [](const arma::vec& _dX, const arma::vec& _theta) {
  const arma::vec& dXnorm = _dX / _theta;
  return exp(-0.5 * arma::dot(dXnorm, dXnorm));
};

std::function<arma::vec(const arma::vec&, const arma::vec&)> Covariance::DlnCovDtheta_gauss = [](const arma::vec& _dX, const arma::vec& _theta) { 
  return _dX % _dX / arma::pow(_theta, 3);
};

//std::function<arma::vec(const arma::vec&, const arma::vec&)> Covariance::DlnCovDx_gauss = [](const arma::vec& _dX, const arma::vec& _theta) {
//  return -2.0 * _dX / arma::square(_theta);
//};

std::function<double(const arma::vec&, const arma::vec&)> Covariance::Cov_exp = [](const arma::vec& _dX, const arma::vec& _theta) {
  return exp(-arma::sum(arma::abs(_dX / _theta))); 
};

std::function<arma::vec(const arma::vec&, const arma::vec&)> Covariance::DlnCovDtheta_exp = [](const arma::vec& _dX, const arma::vec& _theta) {
  arma::vec dlnC = arma::abs(_dX / arma::square(_theta));
  return dlnC; // WARN: cannot return arma::abs(_dX / arma::square(_theta)) because gen. segfault(!!!)
};

const double SQRT_3 = std::sqrt(3.0);

std::function<double(const arma::vec&, const arma::vec&)> Covariance::Cov_matern32 = [](const arma::vec& _dX, const arma::vec& _theta) {
  arma::vec d = SQRT_3 * arma::abs(_dX / _theta);
  return exp(-arma::sum(d - arma::log1p(d)));
};

std::function<arma::vec(const arma::vec&, const arma::vec&)> Covariance::DlnCovDtheta_matern32 = [](const arma::vec& _dX, const arma::vec& _theta) {
  arma::vec d = SQRT_3 * arma::abs(_dX / _theta);
  return arma::conv_to<arma::vec>::from((d % d) / (1 + d)) / _theta;
};

const double SQRT_5 = std::sqrt(5.0);

std::function<double(const arma::vec&, const arma::vec&)> Covariance::Cov_matern52 = [](const arma::vec& _dX, const arma::vec& _theta) {
  arma::vec d = SQRT_5 * arma::abs(_dX / _theta);
  return exp(-arma::sum(d - arma::log1p(d + (d % d) / 3)));
};

std::function<arma::vec(const arma::vec&, const arma::vec&)> Covariance::DlnCovDtheta_matern52 = [](const arma::vec& _dX, const arma::vec& _theta) {
  arma::vec d = SQRT_5 * arma::abs(_dX / _theta);
  arma::vec a = 1 + d;
  arma::vec b = (d % d) / 3;
  return arma::conv_to<arma::vec>::from((a % b) / (a + b)) / _theta;
};

const double EPSILON = 1E-13;

std::function<double(const arma::vec&, const arma::vec&)> Covariance::Cov_whitenoise = [](const arma::vec& _dX, const arma::vec& _theta) {
  if (arma::sum(arma::abs(_dX / _theta)) < EPSILON)
    return 1.0;
  return 0.0;
};

std::function<arma::vec(const arma::vec&, const arma::vec&)> Covariance::DlnCovDtheta_whitenoise = [](const arma::vec& _dX, const arma::vec& _theta) { 
  return arma::vec(_dX.n_elem); // TBD
};
