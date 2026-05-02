// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio
#include <cmath>
// clang-format on

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/KrigingException.hpp"
#include "libKriging/Trend.hpp"

#include <cassert>

const char* const Trend::enum_RegressionModel_strings[] = {"none", "constant", "linear", "interactive", "quadratic"};

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
    case Trend::RegressionModel::None: {
      F.set_size(n, 0);
      F = arma::mat(n, 0, arma::fill::ones);
      return F;
    } break;

    case Trend::RegressionModel::Constant: {
      F.set_size(n, 1);
      F = arma::mat(n, 1, arma::fill::ones);
      return F;
    } break;

    case Trend::RegressionModel::Linear: {
      F.set_size(n, 1 + d);
      F.col(0) = arma::mat(n, 1, arma::fill::ones);
      for (arma::uword i = 0; i < d; i++) {
        F.col(i + 1) = newX.col(i);
      }
      return F;
    } break;

    case Trend::RegressionModel::Interactive: {
      F.set_size(n, 1 + d + d * (d - 1) / 2);
      F.col(0) = arma::mat(n, 1, arma::fill::ones);
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
      F.col(0) = arma::mat(n, 1, arma::fill::ones);
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

arma::mat Trend::regressionModelDerivative(const Trend::RegressionModel& regmodel, const arma::vec& x) {
  arma::uword d = x.n_elem;
  switch (regmodel) {
    case Trend::RegressionModel::None:
      return arma::mat(d, 0);

    case Trend::RegressionModel::Constant:
      return arma::mat(d, 1, arma::fill::zeros);

    case Trend::RegressionModel::Linear: {
      // F = [1, x_1, ..., x_d] → ∂F/∂x_k = [0, δ_{1k}, ..., δ_{dk}]
      arma::mat DF(d, 1 + d, arma::fill::zeros);
      DF.submat(0, 1, d - 1, d) = arma::mat(d, d, arma::fill::eye);
      return DF;
    }

    case Trend::RegressionModel::Interactive: {
      // F = [1, x_1, x_1*x_0, x_2, x_2*x_0, x_2*x_1, ..., x_{d-1}, x_{d-1}*x_0, ...]
      arma::uword p = 1 + d + d * (d - 1) / 2;
      arma::mat DF(d, p, arma::fill::zeros);
      arma::uword count = 1;
      for (arma::uword i = 0; i < d; i++) {
        // ∂x_i/∂x_k = δ_{ik}
        DF(i, count) = 1.0;
        count += 1;
        for (arma::uword j = 0; j < i; j++) {
          // ∂(x_i*x_j)/∂x_k = x_j if k==i, x_i if k==j
          DF(i, count) = x(j);
          DF(j, count) = x(i);
          count += 1;
        }
      }
      return DF;
    }

    case Trend::RegressionModel::Quadratic: {
      // F = [1, x_1, x_1*x_0, x_1*x_1, x_2, x_2*x_0, x_2*x_1, x_2*x_2, ...]
      arma::uword p = 1 + 2 * d + d * (d - 1) / 2;
      arma::mat DF(d, p, arma::fill::zeros);
      arma::uword count = 1;
      for (arma::uword i = 0; i < d; i++) {
        // ∂x_i/∂x_k = δ_{ik}
        DF(i, count) = 1.0;
        count += 1;
        for (arma::uword j = 0; j <= i; j++) {
          // ∂(x_i*x_j)/∂x_k: if i==j → 2*x_i for k==i; else x_j for k==i, x_i for k==j
          if (i == j) {
            DF(i, count) = 2.0 * x(i);
          } else {
            DF(i, count) = x(j);
            DF(j, count) = x(i);
          }
          count += 1;
        }
      }
      return DF;
    }

    default:
      throw std::runtime_error("Unreachable code");
  }
}