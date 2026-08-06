// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio
#include <cmath>
// clang-format on

#include <string>
#include <vector>

#include "libKriging/Covariance.hpp"

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

namespace {

const std::vector<std::string> differentiable_kernels = {"gauss", "matern3_2", "matern5_2"};

// Central finite difference of k(dX) with respect to dX_i.
// Since dX = x - x', ∂k/∂x_i == ∂k/∂dX_i.
double fd_DCovDx(const Covariance::CovFunc& Cov,
                 const arma::vec& dX,
                 const arma::vec& theta,
                 arma::uword i,
                 double h) {
  arma::vec p = dX, m = dX;
  p[i] += h;
  m[i] -= h;
  return (Cov(p, theta) - Cov(m, theta)) / (2 * h);
}

// Central finite difference of ∂k/∂x_i with respect to x'_j.
// ∂/∂x'_j = -∂/∂dX_j, hence the leading minus sign.
double fd_D2CovDxDxp(const Covariance::GradFunc& DCovDx,
                     const arma::vec& dX,
                     const arma::vec& theta,
                     arma::uword i,
                     arma::uword j,
                     double h) {
  arma::vec p = dX, m = dX;
  p[j] += h;
  m[j] -= h;
  return -(DCovDx(p, theta)[i] - DCovDx(m, theta)[i]) / (2 * h);
}

}  // namespace

TEST_CASE("Covariance::supportsDerivativeObservations", "[Covariance]") {
  for (const auto& k : differentiable_kernels) {
    REQUIRE(Covariance::supportsDerivativeObservations(k));
    auto cov = Covariance::resolve(k);
    REQUIRE(static_cast<bool>(cov.DCovDx));
    REQUIRE(static_cast<bool>(cov.D2CovDxDxp));
  }

  // "exp" has a kink at the origin and "whitenoise" is nowhere continuous:
  // neither admits gradient observations, and both expose empty functors.
  for (const auto& k : {"exp", "whitenoise"}) {
    REQUIRE_FALSE(Covariance::supportsDerivativeObservations(k));
    auto cov = Covariance::resolve(k);
    REQUIRE_FALSE(static_cast<bool>(cov.DCovDx));
    REQUIRE_FALSE(static_cast<bool>(cov.D2CovDxDxp));
  }
}

TEST_CASE("Covariance::DCovDx matches finite differences", "[Covariance]") {
  const arma::vec theta = {0.7, 1.3, 0.4};
  const double h = 1e-6;

  arma::arma_rng::set_seed(42);
  for (const auto& kernel : differentiable_kernels) {
    auto cov = Covariance::resolve(kernel);
    for (int rep = 0; rep < 20; rep++) {
      arma::vec dX = 2 * arma::randu<arma::vec>(3) - 1;
      arma::vec analytic = cov.DCovDx(dX, theta);
      REQUIRE(analytic.n_elem == 3);
      for (arma::uword i = 0; i < 3; i++) {
        INFO("kernel=" << kernel << " i=" << i << " dX=" << dX.t());
        REQUIRE(analytic[i] == Approx(fd_DCovDx(cov.Cov, dX, theta, i, h)).margin(1e-6));
      }
    }
  }
}

TEST_CASE("Covariance::D2CovDxDxp matches finite differences", "[Covariance]") {
  const arma::vec theta = {0.7, 1.3, 0.4};
  const double h = 1e-5;

  arma::arma_rng::set_seed(43);
  for (const auto& kernel : differentiable_kernels) {
    auto cov = Covariance::resolve(kernel);
    for (int rep = 0; rep < 20; rep++) {
      // Stay away from the origin: Matérn kernels have a |dX| kink there, so the
      // finite-difference reference is only valid off-diagonal-free of the kink.
      arma::vec dX = 2 * arma::randu<arma::vec>(3) - 1;
      for (arma::uword i = 0; i < 3; i++)
        dX[i] += (dX[i] >= 0 ? 0.1 : -0.1);

      arma::mat analytic = cov.D2CovDxDxp(dX, theta);
      REQUIRE(analytic.n_rows == 3);
      REQUIRE(analytic.n_cols == 3);
      for (arma::uword i = 0; i < 3; i++)
        for (arma::uword j = 0; j < 3; j++) {
          INFO("kernel=" << kernel << " i=" << i << " j=" << j << " dX=" << dX.t());
          REQUIRE(analytic(i, j) == Approx(fd_D2CovDxDxp(cov.DCovDx, dX, theta, i, j, h)).margin(1e-5));
        }
    }
  }
}

TEST_CASE("Covariance::D2CovDxDxp is symmetric", "[Covariance]") {
  const arma::vec theta = {0.7, 1.3, 0.4};

  arma::arma_rng::set_seed(44);
  for (const auto& kernel : differentiable_kernels) {
    auto cov = Covariance::resolve(kernel);
    for (int rep = 0; rep < 20; rep++) {
      arma::vec dX = 2 * arma::randu<arma::vec>(3) - 1;
      arma::mat H = cov.D2CovDxDxp(dX, theta);
      INFO("kernel=" << kernel);
      REQUIRE(arma::approx_equal(H, H.t(), "absdiff", 1e-12));
    }
  }
}

TEST_CASE("Covariance derivative variance at the origin", "[Covariance]") {
  const arma::vec theta = {0.7, 1.3, 0.4};
  const arma::vec zero = arma::zeros<arma::vec>(3);

  // ∂²k/∂x_i∂x'_i at dX=0 is Var(∂Z/∂x_i)/σ²: 1/θ² (gauss), 3/θ² (Matérn 3/2),
  // 5/(3θ²) (Matérn 5/2). Off-diagonal terms vanish for a separable kernel.
  const std::vector<std::pair<std::string, double>> expected
      = {{"gauss", 1.0}, {"matern3_2", 3.0}, {"matern5_2", 5.0 / 3.0}};

  for (const auto& [kernel, factor] : expected) {
    auto cov = Covariance::resolve(kernel);
    arma::mat H = cov.D2CovDxDxp(zero, theta);
    for (arma::uword i = 0; i < 3; i++) {
      INFO("kernel=" << kernel << " i=" << i);
      REQUIRE(H(i, i) == Approx(factor / (theta[i] * theta[i])));
      for (arma::uword j = 0; j < 3; j++)
        if (i != j)
          REQUIRE(H(i, j) == Approx(0.0).margin(1e-14));
    }
    // The gradient of the kernel vanishes at its maximum.
    REQUIRE(arma::norm(cov.DCovDx(zero, theta), "inf") == Approx(0.0).margin(1e-14));
  }
}

TEST_CASE("Covariance gradient-enhanced matrix is positive definite", "[Covariance]") {
  // The full GEK covariance built from [k, ∂k/∂x', ∂k/∂x, ∂²k/∂x∂x'] must be
  // positive definite -- this is the property the gradient-enhanced fit relies on.
  const arma::uword d = 2;
  const arma::uword n = 5;
  const arma::vec theta = {0.8, 0.5};

  arma::arma_rng::set_seed(45);
  arma::mat X = arma::randu<arma::mat>(n, d);

  for (const auto& kernel : differentiable_kernels) {
    auto cov = Covariance::resolve(kernel);
    arma::mat R(n * (1 + d), n * (1 + d), arma::fill::zeros);

    for (arma::uword a = 0; a < n; a++)
      for (arma::uword b = 0; b < n; b++) {
        arma::vec dX = (X.row(a) - X.row(b)).t();
        R(a, b) = cov.Cov(dX, theta);
        arma::vec g = cov.DCovDx(dX, theta);
        arma::mat H = cov.D2CovDxDxp(dX, theta);
        for (arma::uword i = 0; i < d; i++) {
          R(n + a * d + i, b) = g[i];       // Cov(∂Z(x_a)/∂x_i, Z(x_b))
          R(a, n + b * d + i) = -g[i];      // Cov(Z(x_a), ∂Z(x_b)/∂x_i)
          for (arma::uword j = 0; j < d; j++)
            R(n + a * d + i, n + b * d + j) = H(i, j);
        }
      }

    INFO("kernel=" << kernel);
    REQUIRE(arma::approx_equal(R, R.t(), "absdiff", 1e-10));
    arma::mat L;
    REQUIRE(arma::chol(L, R + 1e-8 * arma::eye(n * (1 + d), n * (1 + d)), "lower"));
  }
}
