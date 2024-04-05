#ifndef LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_COVARIANCE_HPP
#define LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_COVARIANCE_HPP

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/libKriging_exports.h"

class Covariance {
 public:
  static std::function<float(const arma::fvec&, const arma::fvec&)> Cov_gauss;

  static std::function<arma::fvec(const arma::fvec&, const arma::fvec&)> DlnCovDtheta_gauss;

  static std::function<arma::fvec(const arma::fvec&, const arma::fvec&)> DlnCovDx_gauss;

  static std::function<float(const arma::fvec&, const arma::fvec&)> Cov_exp;

  static std::function<arma::fvec(const arma::fvec&, const arma::fvec&)> DlnCovDtheta_exp;

  static std::function<arma::fvec(const arma::fvec&, const arma::fvec&)> DlnCovDx_exp;

  static std::function<float(const arma::fvec&, const arma::fvec&)> Cov_matern32;

  static std::function<arma::fvec(const arma::fvec&, const arma::fvec&)> DlnCovDtheta_matern32;

  static std::function<arma::fvec(const arma::fvec&, const arma::fvec&)> DlnCovDx_matern32;

  static std::function<float(const arma::fvec&, const arma::fvec&)> Cov_matern52;

  static std::function<arma::fvec(const arma::fvec&, const arma::fvec&)> DlnCovDtheta_matern52;

  static std::function<arma::fvec(const arma::fvec&, const arma::fvec&)> DlnCovDx_matern52;

  static std::function<float(const arma::fvec&, const arma::fvec&)> Cov_whitenoise;

  static std::function<arma::fvec(const arma::fvec&, const arma::fvec&)> DlnCovDtheta_whitenoise;

  static std::function<arma::fvec(const arma::fvec&, const arma::fvec&)> DlnCovDx_whitenoise;
};

#endif  // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_COVARIANCE_HPP
