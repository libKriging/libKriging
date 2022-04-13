#ifndef LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_TREND_HPP
#define LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_TREND_HPP

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/libKriging_exports.h"

class Trend {
 public:
  LIBKRIGING_EXPORT enum class RegressionModel { Constant, Linear, Interactive, Quadratic };
  LIBKRIGING_EXPORT static const char* enum_RegressionModel_strings[];

 public:
  LIBKRIGING_EXPORT static RegressionModel fromString(const std::string& s);
  LIBKRIGING_EXPORT static std::string toString(const RegressionModel& m);

  LIBKRIGING_EXPORT static arma::mat regressionModelMatrix(const RegressionModel& regmodel,
                                                           const arma::mat& newX,
                                                           arma::uword n,
                                                           arma::uword d);
};

#endif  // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_TREND_HPP
