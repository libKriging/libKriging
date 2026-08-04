// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio
#include <cmath>
// clang-format on

#include <vector>

#include "libKriging/Trend.hpp"

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

namespace {

const std::vector<Trend::RegressionModel> all_models = {Trend::RegressionModel::None,
                                                        Trend::RegressionModel::Constant,
                                                        Trend::RegressionModel::Linear,
                                                        Trend::RegressionModel::Interactive,
                                                        Trend::RegressionModel::Quadratic};

// Central finite difference of F_j(x) with respect to x_k.
double fd_trend(const Trend::RegressionModel& m, const arma::vec& x, arma::uword j, arma::uword k, double h) {
  arma::vec p = x, mm = x;
  p[k] += h;
  mm[k] -= h;
  double Fp = Trend::regressionModelMatrix(m, p.t())(0, j);
  double Fm = Trend::regressionModelMatrix(m, mm.t())(0, j);
  return (Fp - Fm) / (2 * h);
}

}  // namespace

TEST_CASE("Trend::regressionModelDerivative matches finite differences", "[Trend]") {
  const double h = 1e-6;

  arma::arma_rng::set_seed(51);
  for (arma::uword d = 1; d <= 4; d++)
    for (const auto& model : all_models)
      for (int rep = 0; rep < 5; rep++) {
        arma::vec x = 2 * arma::randu<arma::vec>(d) - 1;
        arma::mat DF = Trend::regressionModelDerivative(model, x);
        arma::mat F = Trend::regressionModelMatrix(model, x.t());

        INFO("model=" << Trend::toString(model) << " d=" << d);
        REQUIRE(DF.n_rows == d);
        REQUIRE(DF.n_cols == F.n_cols);

        for (arma::uword j = 0; j < DF.n_cols; j++)
          for (arma::uword k = 0; k < d; k++) {
            INFO("j=" << j << " k=" << k << " x=" << x.t());
            REQUIRE(DF(k, j) == Approx(fd_trend(model, x, j, k, h)).margin(1e-6));
          }
      }
}

TEST_CASE("Trend::regressionModelDerivativeMatrix stacks per-point blocks", "[Trend]") {
  arma::arma_rng::set_seed(52);
  const arma::uword n = 6;

  for (arma::uword d = 1; d <= 4; d++) {
    arma::mat X = 2 * arma::randu<arma::mat>(n, d) - 1;
    for (const auto& model : all_models) {
      arma::mat DF = Trend::regressionModelDerivativeMatrix(model, X);
      arma::mat F = Trend::regressionModelMatrix(model, X);

      INFO("model=" << Trend::toString(model) << " d=" << d);
      REQUIRE(DF.n_rows == n * d);
      REQUIRE(DF.n_cols == F.n_cols);

      // Row a*d + k must equal the single-point derivative block of X.row(a).
      for (arma::uword a = 0; a < n; a++) {
        arma::mat block = Trend::regressionModelDerivative(model, X.row(a).t());
        REQUIRE(arma::approx_equal(DF.rows(a * d, a * d + d - 1), block, "absdiff", 1e-14));
      }
    }
  }
}

TEST_CASE("Trend::regressionModelDerivativeMatrix on a linear function", "[Trend]") {
  // With a linear trend, F_aug * beta must reproduce both the values and the
  // gradients of the underlying linear function -- the consistency property the
  // gradient-enhanced fit relies on.
  const arma::uword n = 5;
  const arma::uword d = 3;

  arma::arma_rng::set_seed(53);
  arma::mat X = arma::randu<arma::mat>(n, d);
  arma::vec beta = {0.4, -1.2, 2.0, 0.7};  // [intercept, slope_1..slope_d]

  arma::mat F = Trend::regressionModelMatrix(Trend::RegressionModel::Linear, X);
  arma::mat DF = Trend::regressionModelDerivativeMatrix(Trend::RegressionModel::Linear, X);

  arma::vec y = F * beta;
  arma::vec dy = DF * beta;

  for (arma::uword a = 0; a < n; a++) {
    REQUIRE(y[a] == Approx(beta[0] + arma::dot(X.row(a).t(), beta.subvec(1, d))));
    for (arma::uword k = 0; k < d; k++)
      REQUIRE(dy[a * d + k] == Approx(beta[1 + k]));
  }
}

TEST_CASE("Trend::regressionModelDerivativeMatrix edge cases", "[Trend]") {
  const arma::uword d = 2;

  // None: zero trend columns, but the row count must still be n*d.
  arma::mat X = arma::randu<arma::mat>(4, d);
  arma::mat DF_none = Trend::regressionModelDerivativeMatrix(Trend::RegressionModel::None, X);
  REQUIRE(DF_none.n_rows == 4 * d);
  REQUIRE(DF_none.n_cols == 0);

  // Constant: the trend derivative is identically zero.
  arma::mat DF_const = Trend::regressionModelDerivativeMatrix(Trend::RegressionModel::Constant, X);
  REQUIRE(DF_const.n_rows == 4 * d);
  REQUIRE(DF_const.n_cols == 1);
  REQUIRE(arma::norm(DF_const, "inf") == Approx(0.0).margin(1e-14));

  // Empty design: sized consistently with the trend matrix.
  arma::mat X0(0, d);
  arma::mat DF0 = Trend::regressionModelDerivativeMatrix(Trend::RegressionModel::Quadratic, X0);
  REQUIRE(DF0.n_rows == 0);
  REQUIRE(DF0.n_cols == Trend::regressionModelMatrix(Trend::RegressionModel::Quadratic, X0).n_cols);
}
