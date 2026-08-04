#ifndef LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_COVARIANCE_HPP
#define LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_COVARIANCE_HPP

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/libKriging_exports.h"

class Covariance {
 public:
  using CovFunc = std::function<double(const arma::vec&, const arma::vec&)>;
  using GradFunc = std::function<arma::vec(const arma::vec&, const arma::vec&)>;
  using HessFunc = std::function<arma::mat(const arma::vec&, const arma::vec&)>;

  struct CovFunctions {
    CovFunc Cov;
    GradFunc DlnCovDtheta;
    GradFunc DlnCovDx;
    /// ∂k/∂x (length d), with `_dX = x - x'`. Empty when the kernel is not
    /// differentiable (see `supportsDerivativeObservations`).
    GradFunc DCovDx;
    /// ∂²k/∂x_i∂x'_j (d×d), with `_dX = x - x'`. Empty when the kernel is not
    /// twice differentiable (see `supportsDerivativeObservations`).
    HessFunc D2CovDxDxp;
  };

  /// Resolve kernel name to function bundle.
  /// Accepted names: "gauss", "exp", "matern3_2", "matern5_2", "whitenoise".
  /// See docs/math/Kernels.md for the formulas, smoothness properties and
  /// choice guidance.
  LIBKRIGING_EXPORT static CovFunctions resolve(const std::string& covType);

  /// True when the kernel is mean-square differentiable, i.e. `DCovDx` and
  /// `D2CovDxDxp` are available and gradient observations (gradient-enhanced
  /// kriging) can be assimilated. False for "exp" (kink at 0) and
  /// "whitenoise" (nowhere continuous).
  LIBKRIGING_EXPORT static bool supportsDerivativeObservations(const std::string& covType);

  static CovFunc Cov_gauss;
  static GradFunc DlnCovDtheta_gauss;
  static GradFunc DlnCovDx_gauss;
  static GradFunc DCovDx_gauss;
  static HessFunc D2CovDxDxp_gauss;

  static CovFunc Cov_exp;
  static GradFunc DlnCovDtheta_exp;
  static GradFunc DlnCovDx_exp;

  static CovFunc Cov_matern32;
  static GradFunc DlnCovDtheta_matern32;
  static GradFunc DlnCovDx_matern32;
  static GradFunc DCovDx_matern32;
  static HessFunc D2CovDxDxp_matern32;

  static CovFunc Cov_matern52;
  static GradFunc DlnCovDtheta_matern52;
  static GradFunc DlnCovDx_matern52;
  static GradFunc DCovDx_matern52;
  static HessFunc D2CovDxDxp_matern52;

  static CovFunc Cov_whitenoise;
  static GradFunc DlnCovDtheta_whitenoise;
  static GradFunc DlnCovDx_whitenoise;
};

#endif  // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_COVARIANCE_HPP
