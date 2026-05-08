#ifndef LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_COVARIANCE_HPP
#define LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_COVARIANCE_HPP

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/libKriging_exports.h"

class Covariance {
 public:
  using CovFunc = std::function<double(const arma::vec&, const arma::vec&)>;
  using GradFunc = std::function<arma::vec(const arma::vec&, const arma::vec&)>;

  struct CovFunctions {
    CovFunc Cov;
    GradFunc DlnCovDtheta;
    GradFunc DlnCovDx;
  };

  /// Resolve kernel name to function triplet.
  /// Accepted names: "gauss", "exp", "matern3_2", "matern5_2", "whitenoise".
  LIBKRIGING_EXPORT static CovFunctions resolve(const std::string& covType);

  static CovFunc Cov_gauss;
  static GradFunc DlnCovDtheta_gauss;
  static GradFunc DlnCovDx_gauss;

  static CovFunc Cov_exp;
  static GradFunc DlnCovDtheta_exp;
  static GradFunc DlnCovDx_exp;

  static CovFunc Cov_matern32;
  static GradFunc DlnCovDtheta_matern32;
  static GradFunc DlnCovDx_matern32;

  static CovFunc Cov_matern52;
  static GradFunc DlnCovDtheta_matern52;
  static GradFunc DlnCovDx_matern52;

  static CovFunc Cov_whitenoise;
  static GradFunc DlnCovDtheta_whitenoise;
  static GradFunc DlnCovDx_whitenoise;
};

#endif  // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_COVARIANCE_HPP
