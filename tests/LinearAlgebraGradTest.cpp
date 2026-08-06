// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio
#include <cmath>
// clang-format on

#include <string>
#include <vector>

#include "libKriging/Covariance.hpp"
#include "libKriging/LinearAlgebra.hpp"

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

namespace {

const std::vector<std::string> differentiable_kernels = {"gauss", "matern3_2", "matern5_2"};

// Straightforward (unoptimized) reference assembly of the augmented GEK
// covariance, written directly from the covariance identities:
//   Cov(Z(a), Z(b))           = k(dX)
//   Cov(dZ(a)/dx_i, Z(b))     = dk/dx_i (dX)
//   Cov(Z(a), dZ(b)/dx_j)     = -dk/dx_j (dX)
//   Cov(dZ(a)/dx_i, dZ(b)/dx_j) = d2k/dx_i dx'_j (dX)
// with dX = x_a - x_b, X given as (d x n).
arma::mat reference_R_grad(const arma::mat& X, const arma::vec& theta, const Covariance::CovFunctions& cov) {
  const arma::uword d = X.n_rows;
  const arma::uword n = X.n_cols;
  arma::mat R(n * (1 + d), n * (1 + d), arma::fill::zeros);

  for (arma::uword a = 0; a < n; a++)
    for (arma::uword b = 0; b < n; b++) {
      arma::vec dX = X.col(a) - X.col(b);
      R(a, b) = cov.Cov(dX, theta);
      arma::vec g = cov.DCovDx(dX, theta);
      arma::mat H = cov.D2CovDxDxp(dX, theta);
      for (arma::uword i = 0; i < d; i++) {
        R(n + a * d + i, b) = g[i];
        R(a, n + b * d + i) = -g[i];
        for (arma::uword j = 0; j < d; j++)
          R(n + a * d + i, n + b * d + j) = H(i, j);
      }
    }
  return R;
}

}  // namespace

TEST_CASE("LinearAlgebra::covMat_sym_X_grad matches the reference assembly", "[LinearAlgebra][grad]") {
  arma::arma_rng::set_seed(61);

  for (arma::uword d = 1; d <= 3; d++)
    for (const auto& kernel : differentiable_kernels) {
      auto cov = Covariance::resolve(kernel);
      const arma::uword n = 6;
      arma::mat X = arma::randu<arma::mat>(d, n);
      arma::vec theta = 0.3 + arma::randu<arma::vec>(d);

      arma::mat R(n * (1 + d), n * (1 + d), arma::fill::none);
      LinearAlgebra::covMat_sym_X_grad(&R, X, theta, cov.Cov, cov.DCovDx, cov.D2CovDxDxp);

      INFO("kernel=" << kernel << " d=" << d);
      REQUIRE(arma::approx_equal(R, reference_R_grad(X, theta, cov), "absdiff", 1e-12));
    }
}

TEST_CASE("LinearAlgebra::covMat_sym_X_grad value block matches covMat_sym_X", "[LinearAlgebra][grad]") {
  arma::arma_rng::set_seed(62);
  const arma::uword d = 2;
  const arma::uword n = 7;
  arma::mat X = arma::randu<arma::mat>(d, n);
  arma::vec theta = {0.6, 0.9};

  for (const auto& kernel : differentiable_kernels) {
    auto cov = Covariance::resolve(kernel);

    arma::mat R_aug(n * (1 + d), n * (1 + d), arma::fill::none);
    LinearAlgebra::covMat_sym_X_grad(&R_aug, X, theta, cov.Cov, cov.DCovDx, cov.D2CovDxDxp);

    arma::mat R_plain(n, n, arma::fill::none);
    LinearAlgebra::covMat_sym_X(&R_plain, X, theta, cov.Cov, 1.0, arma::ones<arma::vec>(n));

    INFO("kernel=" << kernel);
    REQUIRE(arma::approx_equal(R_aug.submat(0, 0, n - 1, n - 1), R_plain, "absdiff", 1e-12));
  }
}

TEST_CASE("LinearAlgebra::covMat_sym_X_grad is symmetric positive definite", "[LinearAlgebra][grad]") {
  arma::arma_rng::set_seed(63);
  const arma::uword d = 2;
  const arma::uword n = 8;
  arma::mat X = arma::randu<arma::mat>(d, n);
  arma::vec theta = {0.8, 0.5};
  const arma::uword N = n * (1 + d);

  for (const auto& kernel : differentiable_kernels) {
    auto cov = Covariance::resolve(kernel);
    arma::mat R(N, N, arma::fill::none);
    LinearAlgebra::covMat_sym_X_grad(&R, X, theta, cov.Cov, cov.DCovDx, cov.D2CovDxDxp);

    INFO("kernel=" << kernel);
    REQUIRE(arma::approx_equal(R, R.t(), "absdiff", 1e-14));
    arma::mat L;
    REQUIRE(arma::chol(L, R + 1e-8 * arma::eye(N, N), "lower"));
  }
}

TEST_CASE("LinearAlgebra::covMat_sym_X_grad honours factor and diag", "[LinearAlgebra][grad]") {
  arma::arma_rng::set_seed(64);
  const arma::uword d = 2;
  const arma::uword n = 4;
  const arma::uword N = n * (1 + d);
  arma::mat X = arma::randu<arma::mat>(d, n);
  arma::vec theta = {0.7, 0.7};
  auto cov = Covariance::resolve("gauss");

  arma::mat R1(N, N, arma::fill::none);
  LinearAlgebra::covMat_sym_X_grad(&R1, X, theta, cov.Cov, cov.DCovDx, cov.D2CovDxDxp);

  const double factor = 0.75;
  arma::mat R2(N, N, arma::fill::none);
  LinearAlgebra::covMat_sym_X_grad(&R2, X, theta, cov.Cov, cov.DCovDx, cov.D2CovDxDxp, factor);
  REQUIRE(arma::approx_equal(R2, factor * R1, "absdiff", 1e-14));

  // A per-observation diagonal (nugget on values, none on gradients) is applied
  // after the factor.
  arma::vec diag = factor * R1.diag();
  diag.subvec(0, n - 1) += 0.1;
  arma::mat R3(N, N, arma::fill::none);
  LinearAlgebra::covMat_sym_X_grad(&R3, X, theta, cov.Cov, cov.DCovDx, cov.D2CovDxDxp, factor, diag);
  REQUIRE(arma::approx_equal(R3.diag(), diag, "absdiff", 1e-14));
  R3.diag() = R2.diag();
  REQUIRE(arma::approx_equal(R3, R2, "absdiff", 1e-14));
}

TEST_CASE("LinearAlgebra::covMat_rect_X_grad matches the sym assembly", "[LinearAlgebra][grad]") {
  // The cross-covariance between augmented observations at X1 and value-only
  // locations at X2 must coincide with the corresponding sub-block of the
  // symmetric assembly built on the concatenation [X1, X2].
  arma::arma_rng::set_seed(65);
  const arma::uword d = 2;
  const arma::uword n1 = 5;
  const arma::uword n2 = 3;
  arma::mat X1 = arma::randu<arma::mat>(d, n1);
  arma::mat X2 = arma::randu<arma::mat>(d, n2);
  arma::vec theta = {0.6, 1.1};

  for (const auto& kernel : differentiable_kernels) {
    auto cov = Covariance::resolve(kernel);

    arma::mat R_rect(n1 * (1 + d), n2, arma::fill::none);
    LinearAlgebra::covMat_rect_X_grad(&R_rect, X1, X2, theta, cov.Cov, cov.DCovDx);

    arma::mat Xall = arma::join_rows(X1, X2);
    arma::mat R_all = reference_R_grad(Xall, theta, cov);
    const arma::uword n = n1 + n2;

    INFO("kernel=" << kernel);
    for (arma::uword a = 0; a < n1; a++)
      for (arma::uword j = 0; j < n2; j++) {
        REQUIRE(R_rect(a, j) == Approx(R_all(a, n1 + j)));
        for (arma::uword i = 0; i < d; i++)
          REQUIRE(R_rect(n1 + a * d + i, j) == Approx(R_all(n + a * d + i, n1 + j)));
      }
  }
}

TEST_CASE("LinearAlgebra GEK assembly rejects non-differentiable kernels", "[LinearAlgebra][grad]") {
  const arma::uword d = 2;
  const arma::uword n = 3;
  arma::mat X = arma::randu<arma::mat>(d, n);
  arma::vec theta = {0.5, 0.5};

  for (const auto& kernel : {"exp", "whitenoise"}) {
    auto cov = Covariance::resolve(kernel);
    arma::mat R(n * (1 + d), n * (1 + d), arma::fill::none);
    INFO("kernel=" << kernel);
    REQUIRE_THROWS_AS(
        LinearAlgebra::covMat_sym_X_grad(&R, X, theta, cov.Cov, cov.DCovDx, cov.D2CovDxDxp), std::invalid_argument);
    arma::mat R2(n * (1 + d), n, arma::fill::none);
    REQUIRE_THROWS_AS(LinearAlgebra::covMat_rect_X_grad(&R2, X, X, theta, cov.Cov, cov.DCovDx), std::invalid_argument);
  }
}

TEST_CASE("LinearAlgebra GEK assembly rejects badly sized output", "[LinearAlgebra][grad]") {
  const arma::uword d = 2;
  const arma::uword n = 3;
  arma::mat X = arma::randu<arma::mat>(d, n);
  arma::vec theta = {0.5, 0.5};
  auto cov = Covariance::resolve("gauss");

  arma::mat R_bad(n, n, arma::fill::none);
  REQUIRE_THROWS_AS(LinearAlgebra::covMat_sym_X_grad(&R_bad, X, theta, cov.Cov, cov.DCovDx, cov.D2CovDxDxp),
                    std::invalid_argument);

  arma::mat R(n * (1 + d), n * (1 + d), arma::fill::none);
  arma::vec diag_bad(n, arma::fill::ones);
  REQUIRE_THROWS_AS(
      LinearAlgebra::covMat_sym_X_grad(&R, X, theta, cov.Cov, cov.DCovDx, cov.D2CovDxDxp, 1.0, diag_bad),
      std::invalid_argument);
}

TEST_CASE("GEK conditioning interpolates values and gradients", "[LinearAlgebra][grad]") {
  // End-to-end sanity check of the assembly: a zero-mean GP conditioned on
  // [y ; dy/dx] must reproduce both the values and the gradients at the design
  // points. This is the property the gradient-enhanced fit is built on, and it
  // exercises the sign conventions of every block at once.
  const arma::uword d = 2;
  const arma::uword n = 6;
  arma::arma_rng::set_seed(66);
  arma::mat X = arma::randu<arma::mat>(d, n);
  arma::vec theta = {0.9, 0.9};
  const arma::uword N = n * (1 + d);

  // Arbitrary smooth test function and its gradient.
  auto f = [](const arma::vec& x) { return std::sin(2 * x[0]) + 0.5 * x[1] * x[1]; };
  auto df = [](const arma::vec& x) { return arma::vec{2 * std::cos(2 * x[0]), x[1]}; };

  for (const auto& kernel : differentiable_kernels) {
    auto cov = Covariance::resolve(kernel);

    arma::vec y_aug(N);
    for (arma::uword a = 0; a < n; a++) {
      y_aug[a] = f(X.col(a));
      arma::vec g = df(X.col(a));
      for (arma::uword i = 0; i < d; i++)
        y_aug[n + a * d + i] = g[i];
    }

    arma::mat R(N, N, arma::fill::none);
    LinearAlgebra::covMat_sym_X_grad(&R, X, theta, cov.Cov, cov.DCovDx, cov.D2CovDxDxp);
    R += 1e-10 * arma::eye(N, N);
    arma::vec w = arma::solve(R, y_aug);

    INFO("kernel=" << kernel);
    for (arma::uword b = 0; b < n; b++) {
      // Prediction at a design point: k_aug(x_b)' * R^-1 * y_aug.
      arma::mat k_aug(N, 1, arma::fill::none);
      LinearAlgebra::covMat_rect_X_grad(&k_aug, X, X.col(b), theta, cov.Cov, cov.DCovDx);
      REQUIRE(arma::dot(k_aug, w) == Approx(f(X.col(b))).margin(1e-6));

      // Gradient of the prediction at a design point: d/dx_b of the above.
      // d/dx_j Cov(Z(x_a), Z(x_b)) = -dk/dx_j, and
      // d/dx_j Cov(dZ(x_a)/dx_i, Z(x_b)) = d2k/dx_i dx'_j.
      arma::vec gpred(d, arma::fill::zeros);
      for (arma::uword a = 0; a < n; a++) {
        arma::vec dX = X.col(a) - X.col(b);
        arma::vec g = cov.DCovDx(dX, theta);
        arma::mat H = cov.D2CovDxDxp(dX, theta);
        for (arma::uword j = 0; j < d; j++) {
          gpred[j] += -g[j] * w[a];
          for (arma::uword i = 0; i < d; i++)
            gpred[j] += H(i, j) * w[n + a * d + i];
        }
      }
      arma::vec gtrue = df(X.col(b));
      for (arma::uword j = 0; j < d; j++)
        REQUIRE(gpred[j] == Approx(gtrue[j]).margin(1e-5));
    }
  }
}
