#ifndef LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_TREND_HPP
#define LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_TREND_HPP

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/libKriging_exports.h"

struct Trend {
  enum class LIBKRIGING_EXPORT RegressionModel { None, Constant, Linear, Interactive, Quadratic };
  LIBKRIGING_EXPORT static const char* const enum_RegressionModel_strings[];

  LIBKRIGING_EXPORT static RegressionModel fromString(const std::string& s);
  LIBKRIGING_EXPORT static std::string toString(const RegressionModel& m);

  LIBKRIGING_EXPORT static arma::mat regressionModelMatrix(const RegressionModel& regmodel, const arma::mat& newXt);

  /// Analytical derivative of the trend basis w.r.t. input coordinates.
  /// Given a single point x (d-vector), returns (d × p) matrix where entry (k,j) = ∂F_j/∂x_k.
  LIBKRIGING_EXPORT static arma::mat regressionModelDerivative(const RegressionModel& regmodel, const arma::vec& x);

  /// Stacked trend derivatives for a batch of points, laid out for the
  /// gradient-enhanced (GEK) augmented trend matrix.
  ///
  /// Given X (n × d), returns a (n·d × p) matrix whose row `a*d + k` holds
  /// ∂F_j/∂x_k evaluated at X.row(a) — i.e. the per-observation (d × p) blocks
  /// of `regressionModelDerivative` stacked in observation-major order. The
  /// augmented trend matrix is then `F_aug = [regressionModelMatrix(X) ;
  /// regressionModelDerivativeMatrix(X)]`, matching the ordering of the
  /// augmented observation vector `[y ; vec(dy/dx)]`.
  LIBKRIGING_EXPORT static arma::mat regressionModelDerivativeMatrix(const RegressionModel& regmodel,
                                                                     const arma::mat& X);
};

#endif  // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_TREND_HPP
