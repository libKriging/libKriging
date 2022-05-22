// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio
#include <cmath>
// clang-format on

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/KrigingException.hpp"
#include "libKriging/Trend.hpp"

#include <cassert>

const char* Trend::enum_RegressionModel_strings[] = {"constant", "linear", "interactive", "quadratic"};

Trend::RegressionModel Trend::fromString(const std::string& value) {
  static auto begin = std::begin(Trend::enum_RegressionModel_strings);
  static auto end = std::end(Trend::enum_RegressionModel_strings);

  auto find = std::find(begin, end, value);
  if (find != end) {
    return static_cast<Trend::RegressionModel>(std::distance(begin, find));
  } else {
    // FIXME use std::optional as returned type
    throw KrigingException("Cannot convert '" + value + "' as a regression model");
  }
}

std::string Trend::toString(const Trend::RegressionModel& e) {
  assert(static_cast<std::size_t>(e) < sizeof(Trend::enum_RegressionModel_strings));
  return Trend::enum_RegressionModel_strings[static_cast<int>(e)];
}

arma::mat Trend::regressionModelMatrix(const Trend::RegressionModel& regmodel, const arma::mat& newX) {
  arma::uword n = newX.n_rows;
  arma::uword d = newX.n_cols;
  arma::mat F;  // uses modern RTO to avoid returned object copy
  switch (regmodel) {
    case Trend::RegressionModel::Constant: {
      F.set_size(n, 1);
      F = arma::ones(n, 1);
      return F;
    } break;

    case Trend::RegressionModel::Linear: {
      F.set_size(n, 1 + d);
      F.col(0) = arma::ones(n, 1);
      for (arma::uword i = 0; i < d; i++) {
        F.col(i + 1) = newX.col(i);
      }
      return F;
    } break;

    case Trend::RegressionModel::Interactive: {
      F.set_size(n, 1 + d + d * (d - 1) / 2);
      F.col(0) = arma::ones(n, 1);
      arma::uword count = 1;
      for (arma::uword i = 0; i < d; i++) {
        F.col(count) = newX.col(i);
        count += 1;
        for (arma::uword j = 0; j < i; j++) {
          F.col(count) = newX.col(i) % newX.col(j);
          count += 1;
        }
      }
      return F;
    } break;

    case Trend::RegressionModel::Quadratic: {
      F.set_size(n, 1 + 2 * d + d * (d - 1) / 2);
      F.col(0) = arma::ones(n, 1);
      arma::uword count = 1;
      for (arma::uword i = 0; i < d; i++) {
        F.col(count) = newX.col(i);
        count += 1;
        for (arma::uword j = 0; j <= i; j++) {
          F.col(count) = newX.col(i) % newX.col(j);
          count += 1;
        }
      }
      return F;
    } break;

    default:
      throw std::runtime_error("Unreachable code");
  }
}