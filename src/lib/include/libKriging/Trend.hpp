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
};

#endif  // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_TREND_HPP
