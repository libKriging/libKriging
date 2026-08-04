// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio
#include <cmath>
// clang-format on

#include "libKriging/Covariance.hpp"

#include "libKriging/utils/lk_armadillo.hpp"

#include <cassert>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

//' @ref: https://github.com/psbiomech/dace-toolbox-source/blob/master/dace.pdf
//'  (where CovMatrix<-R, Ft<-M, C<-T, rho<-z)
//' @ref: https://github.com/cran/DiceKriging/blob/master/R/kmEstimate.R (same variables names)

//' @ref https://github.com/cran/DiceKriging/blob/master/src/covMats.c
// Covariance function on normalized data

namespace {

/// Assemble ∂²k/∂x_i∂x'_j for a separable stationary kernel k(dX) = Π_i g(dX_i/θ_i),
/// with `dX = x - x'`.
///
/// Since ∂/∂x'_j = -∂/∂dX_j, we have ∂²k/∂x_i∂x'_j = -∂²k/∂dX_i∂dX_j, and
/// separability gives, for i≠j, ∂²k/∂dX_i∂dX_j = k·v_i·v_j where
/// v = ∂ln k/∂dX (i.e. `DlnCovDx`).  The diagonal is not v_i² but the genuine
/// per-coordinate curvature, passed in as `hdiag_i = (∂²k/∂dX_i²)/k`.
arma::mat assemble_D2CovDxDxp(double k, const arma::vec& v, const arma::vec& hdiag) {
  arma::mat H = -k * (v * v.t());
  for (arma::uword i = 0; i < v.n_elem; i++)
    H(i, i) = -k * hdiag[i];
  return H;
}

}  // namespace

std::function<double(const arma::vec&, const arma::vec&)> Covariance::Cov_gauss
    = [](const arma::vec& _dX, const arma::vec& _theta) {
        // Optimized version: compute directly without temporary vector allocation
        // Previous version (creates temporary vector):
        // const arma::vec& dXnorm = _dX / _theta;
        // return exp(-0.5 * arma::dot(dXnorm, dXnorm));
        double sum_sq = 0.0;
        for (arma::uword i = 0; i < _dX.n_elem; i++) {
          double val = _dX[i] / _theta[i];
          sum_sq += val * val;
        }
        return exp(-0.5 * sum_sq);
      };

std::function<arma::vec(const arma::vec&, const arma::vec&)> Covariance::DlnCovDtheta_gauss
    = [](const arma::vec& _dX, const arma::vec& _theta) {
        // Optimized version: compute directly without temporary vector allocations
        // Previous version (creates multiple temporary vectors):
        // return arma::conv_to<arma::colvec>::from(_dX % _dX / arma::pow(_theta, 3));
        arma::vec result(_dX.n_elem);
        for (arma::uword i = 0; i < _dX.n_elem; i++) {
          double dX_i = _dX[i];
          double theta_i = _theta[i];
          result[i] = (dX_i * dX_i) / (theta_i * theta_i * theta_i);
        }
        return result;
      };

std::function<arma::vec(const arma::vec&, const arma::vec&)> Covariance::DlnCovDx_gauss
    = [](const arma::vec& _dX, const arma::vec& _theta) {
        // Optimized version: compute directly without temporary vector allocations
        // Previous version (creates temporary vectors):
        // return arma::conv_to<arma::colvec>::from(-_dX / arma::square(_theta));
        arma::vec result(_dX.n_elem);
        for (arma::uword i = 0; i < _dX.n_elem; i++) {
          double theta_i = _theta[i];
          result[i] = -_dX[i] / (theta_i * theta_i);
        }
        return result;
      };

// ∂k/∂x for the gaussian kernel: k · ∂ln k/∂x = k · (-dX_i/θ_i²).
std::function<arma::vec(const arma::vec&, const arma::vec&)> Covariance::DCovDx_gauss
    = [](const arma::vec& _dX, const arma::vec& _theta) {
        return Covariance::Cov_gauss(_dX, _theta) * Covariance::DlnCovDx_gauss(_dX, _theta);
      };

// ∂²k/∂x_i∂x'_j for the gaussian kernel.  With t_i = dX_i/θ_i, the per-coordinate
// curvature is (∂²k/∂dX_i²)/k = (t_i² - 1)/θ_i², so the diagonal at dX=0 gives
// the derivative variance 1/θ_i².
std::function<arma::mat(const arma::vec&, const arma::vec&)> Covariance::D2CovDxDxp_gauss
    = [](const arma::vec& _dX, const arma::vec& _theta) {
        arma::vec hdiag(_dX.n_elem);
        for (arma::uword i = 0; i < _dX.n_elem; i++) {
          double theta_i = _theta[i];
          double t = _dX[i] / theta_i;
          hdiag[i] = (t * t - 1.0) / (theta_i * theta_i);
        }
        return assemble_D2CovDxDxp(
            Covariance::Cov_gauss(_dX, _theta), Covariance::DlnCovDx_gauss(_dX, _theta), hdiag);
      };

std::function<double(const arma::vec&, const arma::vec&)> Covariance::Cov_exp
    = [](const arma::vec& _dX, const arma::vec& _theta) {
        // Optimized version: compute directly without temporary vector allocation
        // Previous version (creates temporary vector):
        // return exp(-arma::sum(arma::abs(_dX / _theta)));
        double sum = 0.0;
        for (arma::uword i = 0; i < _dX.n_elem; i++) {
          sum += std::abs(_dX[i] / _theta[i]);
        }
        return exp(-sum);
      };

std::function<arma::vec(const arma::vec&, const arma::vec&)> Covariance::DlnCovDtheta_exp
    = [](const arma::vec& _dX, const arma::vec& _theta) {
        // Optimized version: compute directly without temporary vector allocations
        // Previous version (creates temporary vectors):
        // return arma::conv_to<arma::colvec>::from(arma::abs(_dX / arma::square(_theta)));
        arma::vec result(_dX.n_elem);
        for (arma::uword i = 0; i < _dX.n_elem; i++) {
          double theta_i = _theta[i];
          result[i] = std::abs(_dX[i]) / (theta_i * theta_i);
        }
        return result;
      };

std::function<arma::vec(const arma::vec&, const arma::vec&)> Covariance::DlnCovDx_exp
    = [](const arma::vec& _dX, const arma::vec& _theta) {
        // Optimized version: compute directly without temporary vector allocations
        // Previous version (creates temporary vectors):
        // return arma::conv_to<arma::colvec>::from(-arma::sign(_dX) / _theta);
        arma::vec result(_dX.n_elem);
        for (arma::uword i = 0; i < _dX.n_elem; i++) {
          result[i] = -(_dX[i] > 0 ? 1.0 : (_dX[i] < 0 ? -1.0 : 0.0)) / _theta[i];
        }
        return result;
      };

const double SQRT_3 = std::sqrt(3.0);

std::function<double(const arma::vec&, const arma::vec&)> Covariance::Cov_matern32
    = [](const arma::vec& _dX, const arma::vec& _theta) {
        // Optimized version: compute directly without temporary vector allocation
        // Previous version (creates temporary vector):
        // arma::vec d = SQRT_3 * arma::abs(_dX / _theta);
        // return exp(-arma::sum(d - arma::log1p(d)));
        double sum = 0.0;
        for (arma::uword i = 0; i < _dX.n_elem; i++) {
          double d = SQRT_3 * std::abs(_dX[i] / _theta[i]);
          sum += d - std::log1p(d);
        }
        return exp(-sum);
      };

std::function<arma::vec(const arma::vec&, const arma::vec&)> Covariance::DlnCovDtheta_matern32
    = [](const arma::vec& _dX, const arma::vec& _theta) {
        // Optimized version: compute directly without temporary vector allocations
        // Previous version (creates multiple temporary vectors):
        // arma::vec d = SQRT_3 * arma::abs(_dX / _theta);
        // return arma::conv_to<arma::colvec>::from((d % d) / (1 + d) / _theta);
        arma::vec result(_dX.n_elem);
        for (arma::uword i = 0; i < _dX.n_elem; i++) {
          double d = SQRT_3 * std::abs(_dX[i] / _theta[i]);
          result[i] = (d * d) / (1.0 + d) / _theta[i];
        }
        return result;
      };

std::function<arma::vec(const arma::vec&, const arma::vec&)> Covariance::DlnCovDx_matern32
    = [](const arma::vec& _dX, const arma::vec& _theta) {
        // Optimized version: compute directly without temporary vector allocations
        // Previous version (creates multiple temporary vectors):
        // arma::vec d = SQRT_3 * arma::abs(_dX / _theta);
        // return arma::conv_to<arma::colvec>::from(-SQRT_3 * arma::sign(_dX) % d / (1 + d) / _theta);
        arma::vec result(_dX.n_elem);
        for (arma::uword i = 0; i < _dX.n_elem; i++) {
          double d = SQRT_3 * std::abs(_dX[i] / _theta[i]);
          double sign_dX = _dX[i] > 0 ? 1.0 : (_dX[i] < 0 ? -1.0 : 0.0);
          result[i] = -SQRT_3 * sign_dX * d / (1.0 + d) / _theta[i];
        }
        return result;
      };

// ∂k/∂x for the Matérn 3/2 kernel.
std::function<arma::vec(const arma::vec&, const arma::vec&)> Covariance::DCovDx_matern32
    = [](const arma::vec& _dX, const arma::vec& _theta) {
        return Covariance::Cov_matern32(_dX, _theta) * Covariance::DlnCovDx_matern32(_dX, _theta);
      };

// ∂²k/∂x_i∂x'_j for the Matérn 3/2 kernel.  With d_i = √3·|dX_i|/θ_i, the
// per-coordinate curvature is (∂²k/∂dX_i²)/k = 3(d_i² - 1)/(θ_i²(1+d_i)²), whose
// value at dX=0 yields the derivative variance 3/θ_i².  The kernel is only C¹ at
// the origin, but the process is mean-square differentiable, which is what the
// gradient-enhanced formulation requires.
std::function<arma::mat(const arma::vec&, const arma::vec&)> Covariance::D2CovDxDxp_matern32
    = [](const arma::vec& _dX, const arma::vec& _theta) {
        arma::vec hdiag(_dX.n_elem);
        for (arma::uword i = 0; i < _dX.n_elem; i++) {
          double theta_i = _theta[i];
          double d = SQRT_3 * std::abs(_dX[i] / theta_i);
          double a = 1.0 + d;
          hdiag[i] = 3.0 * (d * d - 1.0) / (theta_i * theta_i * a * a);
        }
        return assemble_D2CovDxDxp(
            Covariance::Cov_matern32(_dX, _theta), Covariance::DlnCovDx_matern32(_dX, _theta), hdiag);
      };

const double SQRT_5 = std::sqrt(5.0);

std::function<double(const arma::vec&, const arma::vec&)> Covariance::Cov_matern52
    = [](const arma::vec& _dX, const arma::vec& _theta) {
        // Optimized version: compute directly without temporary vector allocation
        // Previous version (creates multiple temporary vectors):
        // arma::vec d = SQRT_5 * arma::abs(_dX / _theta);
        // return exp(-arma::sum(d - arma::log1p(d + (d % d) / 3)));
        double sum = 0.0;
        for (arma::uword i = 0; i < _dX.n_elem; i++) {
          double d = SQRT_5 * std::abs(_dX[i] / _theta[i]);
          sum += d - std::log1p(d + (d * d) / 3.0);
        }
        return exp(-sum);
      };

std::function<arma::vec(const arma::vec&, const arma::vec&)> Covariance::DlnCovDtheta_matern52
    = [](const arma::vec& _dX, const arma::vec& _theta) {
        // Optimized version: compute directly without temporary vector allocations
        // Previous version (creates multiple temporary vectors):
        // arma::vec d = SQRT_5 * arma::abs(_dX / _theta);
        // arma::vec a = 1 + d;
        // arma::vec b = (d % d) / 3;
        // return arma::conv_to<arma::colvec>::from((a % b) / (a + b) / _theta);
        arma::vec result(_dX.n_elem);
        for (arma::uword i = 0; i < _dX.n_elem; i++) {
          double d = SQRT_5 * std::abs(_dX[i] / _theta[i]);
          double a = 1.0 + d;
          double b = (d * d) / 3.0;
          result[i] = (a * b) / (a + b) / _theta[i];
        }
        return result;
      };

std::function<arma::vec(const arma::vec&, const arma::vec&)> Covariance::DlnCovDx_matern52
    = [](const arma::vec& _dX, const arma::vec& _theta) {
        // Optimized version: compute directly without temporary vector allocations
        // Previous version (creates multiple temporary vectors):
        // arma::vec d = SQRT_5 * arma::abs(_dX / _theta);
        // arma::vec a = 1 + d;
        // arma::vec b = d / 3;
        // return arma::conv_to<arma::colvec>::from(-SQRT_5 * arma::sign(_dX) % (a % b) / (a + d % b) / _theta);
        arma::vec result(_dX.n_elem);
        for (arma::uword i = 0; i < _dX.n_elem; i++) {
          double d = SQRT_5 * std::abs(_dX[i] / _theta[i]);
          double a = 1.0 + d;
          double b = d / 3.0;
          double sign_dX = _dX[i] > 0 ? 1.0 : (_dX[i] < 0 ? -1.0 : 0.0);
          result[i] = -SQRT_5 * sign_dX * (a * b) / (a + d * b) / _theta[i];
        }
        return result;
      };

// ∂k/∂x for the Matérn 5/2 kernel.
std::function<arma::vec(const arma::vec&, const arma::vec&)> Covariance::DCovDx_matern52
    = [](const arma::vec& _dX, const arma::vec& _theta) {
        return Covariance::Cov_matern52(_dX, _theta) * Covariance::DlnCovDx_matern52(_dX, _theta);
      };

// ∂²k/∂x_i∂x'_j for the Matérn 5/2 kernel.  With d_i = √5·|dX_i|/θ_i and
// P = 1 + d + d²/3, the per-coordinate curvature is
// (∂²k/∂dX_i²)/k = 5(d⁴ + 2d³ - d² - 6d - 3)/(9 θ_i² P²), whose value at dX=0
// yields the derivative variance 5/(3θ_i²).
std::function<arma::mat(const arma::vec&, const arma::vec&)> Covariance::D2CovDxDxp_matern52
    = [](const arma::vec& _dX, const arma::vec& _theta) {
        arma::vec hdiag(_dX.n_elem);
        for (arma::uword i = 0; i < _dX.n_elem; i++) {
          double theta_i = _theta[i];
          double d = SQRT_5 * std::abs(_dX[i] / theta_i);
          double P = 1.0 + d + (d * d) / 3.0;
          double num = d * d * d * d + 2.0 * d * d * d - d * d - 6.0 * d - 3.0;
          hdiag[i] = 5.0 * num / (9.0 * theta_i * theta_i * P * P);
        }
        return assemble_D2CovDxDxp(
            Covariance::Cov_matern52(_dX, _theta), Covariance::DlnCovDx_matern52(_dX, _theta), hdiag);
      };

const double EPSILON = 1E-13;
std::function<double(const arma::vec&, const arma::vec&)> Covariance::Cov_whitenoise
    = [](const arma::vec& _dX, const arma::vec& _theta) {
        if (arma::sum(arma::abs(_dX / _theta)) < EPSILON)
          return 1.0;
        return 0.0;
      };

std::function<arma::vec(const arma::vec&, const arma::vec&)> Covariance::DlnCovDtheta_whitenoise
    = [](const arma::vec& _dX, const arma::vec& /*_theta*/) {
        return arma::vec(_dX.n_elem);  // TBD
      };

std::function<arma::vec(const arma::vec&, const arma::vec&)> Covariance::DlnCovDx_whitenoise
    = [](const arma::vec& _dX, const arma::vec& /*_theta*/) {
        return arma::vec(_dX.n_elem);  // TBD
      };

LIBKRIGING_EXPORT Covariance::CovFunctions Covariance::resolve(const std::string& covType) {
  if (covType == "gauss")
    return {Cov_gauss, DlnCovDtheta_gauss, DlnCovDx_gauss, DCovDx_gauss, D2CovDxDxp_gauss};
  if (covType == "exp")
    return {Cov_exp, DlnCovDtheta_exp, DlnCovDx_exp, {}, {}};  // not differentiable at 0
  if (covType == "matern3_2")
    return {Cov_matern32, DlnCovDtheta_matern32, DlnCovDx_matern32, DCovDx_matern32, D2CovDxDxp_matern32};
  if (covType == "matern5_2")
    return {Cov_matern52, DlnCovDtheta_matern52, DlnCovDx_matern52, DCovDx_matern52, D2CovDxDxp_matern52};
  if (covType == "whitenoise")
    return {Cov_whitenoise, DlnCovDtheta_whitenoise, DlnCovDx_whitenoise, {}, {}};  // nowhere continuous
  throw std::invalid_argument("Unsupported covariance kernel: " + covType);
}

LIBKRIGING_EXPORT bool Covariance::supportsDerivativeObservations(const std::string& covType) {
  return covType == "gauss" || covType == "matern3_2" || covType == "matern5_2";
}
