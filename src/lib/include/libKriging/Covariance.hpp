#ifndef LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_COVARIANCE_HPP
#define LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_COVARIANCE_HPP

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/libKriging_exports.h"

class Covariance {
 public:
  static std::function<double(const arma::vec&, const arma::vec&)> Cov_gauss;

  static std::function<arma::vec(const arma::vec&, const arma::vec&)> DlnCovDtheta_gauss;

  //static std::function<arma::vec(const arma::vec&, const arma::vec&)> DlnCovDx_gauss;

  static std::function<double(const arma::vec&, const arma::vec&)> Cov_exp;

  static std::function<arma::vec(const arma::vec&, const arma::vec&)> DlnCovDtheta_exp;

  //static std::function<arma::vec(const arma::vec&, const arma::vec&)> DlnCovDx_exp;

  static std::function<double(const arma::vec&, const arma::vec&)> Cov_matern32;

  static std::function<arma::vec(const arma::vec&, const arma::vec&)> DlnCovDtheta_matern32;

  //static std::function<arma::vec(const arma::vec&, const arma::vec&)> DlnCovDx_matern32;

  static std::function<double(const arma::vec&, const arma::vec&)> Cov_matern52;

  static std::function<arma::vec(const arma::vec&, const arma::vec&)> DlnCovDtheta_matern52;

  //static std::function<arma::vec(const arma::vec&, const arma::vec&)> DlnCovDx_matern52;

  static std::function<double(const arma::vec&, const arma::vec&)> Cov_whitenoise;

  static std::function<arma::vec(const arma::vec&, const arma::vec&)> DlnCovDtheta_whitenoise;

  //static std::function<arma::vec(const arma::vec&, const arma::vec&)> DlnCovDx_whitenoise;
};

#endif  // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_COVARIANCE_HPP