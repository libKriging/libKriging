#ifndef LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_COVARIANCE_HPP
#define LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_COVARIANCE_HPP

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/libKriging_exports.h"

class Covariance {
 public:
  static std::function<double(const arma::vec&)> CovNorm_fun_gauss;

  static std::function<arma::vec(const arma::vec&)> Dln_CovNorm_gauss;

  static std::function<double(const arma::vec&)> CovNorm_fun_exp;

  static std::function<arma::vec(const arma::vec&)> Dln_CovNorm_exp;

  static std::function<double(const arma::vec&)> CovNorm_fun_matern32;

  static std::function<arma::vec(const arma::vec&)> Dln_CovNorm_matern32;

  static std::function<double(const arma::vec&)> CovNorm_fun_matern52;

  static std::function<arma::vec(const arma::vec&)> Dln_CovNorm_matern52;

  static std::function<double(const arma::vec&)> CovNorm_fun_whitenoise;

  static std::function<arma::vec(const arma::vec&)> Dln_CovNorm_whitenoise;
};

#endif  // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_COVARIANCE_HPP