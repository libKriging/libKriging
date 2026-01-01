// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio
#include <cmath>
// clang-format on

#include "libKriging/Covariance.hpp"

#include "libKriging/utils/lk_armadillo.hpp"

#include <cassert>
#include <tuple>
#include <vector>

//' @ref: https://github.com/psbiomech/dace-toolbox-source/blob/master/dace.pdf
//'  (where CovMatrix<-R, Ft<-M, C<-T, rho<-z)
//' @ref: https://github.com/cran/DiceKriging/blob/master/R/kmEstimate.R (same variables names)

//' @ref https://github.com/cran/DiceKriging/blob/master/src/covMats.c
// Covariance function on normalized data

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
