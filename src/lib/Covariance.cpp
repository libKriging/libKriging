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

std::function<double(const arma::vec&)> Covariance::CovNorm_fun_gauss = [](const arma::vec& _dist_norm) {
  const double temp = arma::dot(_dist_norm, _dist_norm);
  return exp(-0.5 * temp);
};

std::function<arma::vec(const arma::vec&)> Covariance::Dln_CovNorm_gauss
    = [](const arma::vec& _dist_norm) { return _dist_norm % _dist_norm; };

std::function<double(const arma::vec&)> Covariance::CovNorm_fun_exp
    = [](const arma::vec& _dist_norm) { return exp(-arma::sum(arma::abs(_dist_norm))); };

std::function<arma::vec(const arma::vec&)> Covariance::Dln_CovNorm_exp
    = [](const arma::vec& _dist_norm) { return arma::abs(_dist_norm); };

const double SQRT_3 = std::sqrt(3.0);

std::function<double(const arma::vec&)> Covariance::CovNorm_fun_matern32 = [](const arma::vec& _dist_norm) {
  arma::vec d = SQRT_3 * arma::abs(_dist_norm);
  return exp(-arma::sum(d - arma::log1p(d)));
};

std::function<arma::vec(const arma::vec&)> Covariance::Dln_CovNorm_matern32 = [](const arma::vec& _dist_norm) {
  arma::vec d = SQRT_3 * arma::abs(_dist_norm);
  return arma::conv_to<arma::vec>::from((d % d) / (1 + d));
};

const double SQRT_5 = std::sqrt(5.0);

std::function<double(const arma::vec&)> Covariance::CovNorm_fun_matern52 = [](const arma::vec& _dist_norm) {
  arma::vec d = SQRT_5 * arma::abs(_dist_norm);
  return exp(-arma::sum(d - arma::log1p(d + (d % d) / 3)));
};

std::function<arma::vec(const arma::vec&)> Covariance::Dln_CovNorm_matern52 = [](const arma::vec& _dist_norm) {
  arma::vec d = SQRT_5 * arma::abs(_dist_norm);
  arma::vec a = 1 + d;
  arma::vec b = (d % d) / 3;
  return arma::conv_to<arma::vec>::from((a % b) / (a + b));
};

const double EPSILON = 1E-13;

std::function<double(const arma::vec&)> Covariance::CovNorm_fun_whitenoise = [](const arma::vec& _dist_norm) {
  if (arma::sum(arma::abs(_dist_norm)) < EPSILON)
    return 1.0;
  return 0.0;
};

std::function<arma::vec(const arma::vec&)> Covariance::Dln_CovNorm_whitenoise
    = [](const arma::vec& _dist_norm) { return arma::vec(_dist_norm.n_elem); };
