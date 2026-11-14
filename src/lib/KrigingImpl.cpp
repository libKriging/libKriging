// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES  // required for Visual Studio

#include <cmath>
// clang-format on

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/Bench.hpp"
#include "libKriging/Covariance.hpp"
#include "libKriging/KrigingImpl.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/Random.hpp"
#include "libKriging/Trend.hpp"
#include "libKriging/utils/jsonutils.hpp"
#include "libKriging/utils/nlohmann/json.hpp"

#include "BlasThreads.hpp"
#include "IterativeMatvec.hpp"
#include "OmpLoops.hpp"
#include "ParallelFor.hpp"
#include "SeparableDln.hpp"
#include "ThreadBudget.hpp"

#include "cuda/CudaLinearAlgebra.cuh"    // no-op unless built with -DENABLE_CUDA_ITERATIVE=ON
#include "hip/HipLinearAlgebra.hpp"      // no-op unless built with -DENABLE_HIP_ITERATIVE=ON
#include "metal/MetalLinearAlgebra.hpp"  // no-op unless built with -DENABLE_METAL_ITERATIVE=ON
#include "sycl/SyclLinearAlgebra.hpp"    // no-op unless built with -DENABLE_SYCL_ITERATIVE=ON

#include <array>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
// Basis change of a polynomial trend under an affine change of its input:
// returns the p x p matrix A such that, for every x (row vector),
//   F(a + b % x) = F(x) * A
// where F is Trend::regressionModelMatrix(regmodel, .) (same column layout).
// Exact (no solve): constant, linear, interactive (x_i x_j, j < i) and
// quadratic (x_i x_j, j <= i) bases are all closed under affine maps.
arma::mat trend_affine_map(const Trend::RegressionModel& regmodel,
                           const arma::uword d,
                           const arma::rowvec& a,
                           const arma::rowvec& b) {
  using RM = Trend::RegressionModel;
  if (regmodel == RM::None)
    return arma::mat(0, 0);
  if (regmodel == RM::Constant)
    return arma::mat(1, 1, arma::fill::ones);
  // column index of each basis function, replicating regressionModelMatrix's layout
  const bool interactive = (regmodel == RM::Interactive);
  const bool quadratic = (regmodel == RM::Quadratic);
  arma::uvec lin_idx(d);
  arma::umat quad_idx(d, d, arma::fill::zeros);  // quad_idx(i, j), j <= i (j < i for interactive)
  arma::uword count = 1;
  for (arma::uword i = 0; i < d; i++) {
    lin_idx(i) = count++;
    if (interactive || quadratic) {
      const arma::uword jmax = quadratic ? i + 1 : i;
      for (arma::uword j = 0; j < jmax; j++)
        quad_idx(i, j) = count++;
    }
  }
  const arma::uword p = count;
  arma::mat A(p, p, arma::fill::zeros);
  A(0, 0) = 1.0;
  for (arma::uword i = 0; i < d; i++) {
    // z_i = a_i + b_i x_i
    const arma::uword k = lin_idx(i);
    A(0, k) += a(i);
    A(lin_idx(i), k) += b(i);
    if (interactive || quadratic) {
      const arma::uword jmax = quadratic ? i + 1 : i;
      for (arma::uword j = 0; j < jmax; j++) {
        // z_i z_j = a_i a_j + a_i b_j x_j + b_i a_j x_i + b_i b_j x_i x_j
        const arma::uword kk = quad_idx(i, j);
        A(0, kk) += a(i) * a(j);
        A(lin_idx(j), kk) += a(i) * b(j);
        A(lin_idx(i), kk) += b(i) * a(j);
        A(quad_idx(i, j), kk) += b(i) * b(j);
      }
    }
  }
  return A;
}
}  // namespace

arma::vec KrigingImpl::beta_to_original(const arma::vec& beta_n) const {
  if (!m_normalize || beta_n.is_empty())
    return beta_n;
  // F(x_n) = F(x) * A with x_n = (x - centerX) / scaleX, hence
  // F(x) * beta_o = scaleY * F(x_n) * beta_n + centerY  <=>  beta_o = scaleY * A * beta_n + centerY * e_0
  const arma::uword d = m_scaleX.n_elem;
  const arma::mat A = trend_affine_map(m_regmodel, d, -m_centerX / m_scaleX, 1.0 / m_scaleX);
  if (A.n_cols != beta_n.n_elem)
    throw std::invalid_argument("beta has " + std::to_string(beta_n.n_elem) + " coefficients, the "
                                + Trend::toString(m_regmodel) + " trend needs " + std::to_string(A.n_cols));
  arma::vec beta_o = m_scaleY * (A * beta_n);
  beta_o(0) += m_centerY;
  return beta_o;
}

arma::vec KrigingImpl::beta_from_original(const arma::vec& beta_o) const {
  if (!m_normalize || beta_o.is_empty())
    return beta_o;
  // F(x) = F(x_n) * B with x = centerX + scaleX % x_n, hence
  // beta_n = (B * beta_o - centerY * e_0) / scaleY
  const arma::uword d = m_scaleX.n_elem;
  const arma::mat B = trend_affine_map(m_regmodel, d, m_centerX, m_scaleX);
  if (B.n_cols != beta_o.n_elem)
    throw std::invalid_argument("beta has " + std::to_string(beta_o.n_elem) + " coefficients, the "
                                + Trend::toString(m_regmodel) + " trend needs " + std::to_string(B.n_cols));
  arma::vec beta_n = B * beta_o;
  beta_n(0) -= m_centerY;
  return beta_n / m_scaleY;
}

double KrigingImpl::trend_log_abs_det_to_original() const {
  if (!m_normalize || m_regmodel == Trend::RegressionModel::None)
    return 0.0;
  const arma::uword d = m_scaleX.n_elem;
  const arma::mat A = trend_affine_map(m_regmodel, d, -m_centerX / m_scaleX, 1.0 / m_scaleX);
  double val = 0.0, sign = 0.0;
  arma::log_det(val, sign, A);
  return val;
}

// Sentinel used with LinearAlgebra::cholCov / update_cholCov when no
// per-point diagonal override is required (plain Kriging, or Nugget/Noise
// when the diagonal is uniform and baked into the cov factor).
arma::vec KrigingImpl::ones = arma::ones<arma::vec>(0);

void KrigingImpl::make_Cov(const std::string& covType) {
  m_covType = covType;

  auto cov = Covariance::resolve(covType);
  _Cov = std::move(cov.Cov);
  _DlnCovDtheta = std::move(cov.DlnCovDtheta);
  _DlnCovDx = std::move(cov.DlnCovDx);

  if (covType == "gauss")
    _Cov_pow = 2;
  else if (covType == "exp")
    _Cov_pow = 1;
  else if (covType == "matern3_2")
    _Cov_pow = 1.5;
  else if (covType == "matern5_2")
    _Cov_pow = 2.5;
  else
    _Cov_pow = 2;  // default
}

LIBKRIGING_EXPORT arma::mat KrigingImpl::covMat(const arma::mat& X1, const arma::mat& X2) {
  arma::mat Xn1 = X1;
  arma::mat Xn2 = X2;
  Xn1.each_row() -= m_centerX;
  Xn1.each_row() /= m_scaleX;
  Xn2.each_row() -= m_centerX;
  Xn2.each_row() /= m_scaleX;

  arma::mat R = arma::mat(X1.n_rows, X2.n_rows, arma::fill::none);
  LinearAlgebra::covMat_rect(&R, Xn1.t(), Xn2.t(), m_theta, _Cov, m_sigma2);
  return R;
}

void KrigingImpl::populate_Model(KModel& m,
                                 const arma::vec& theta,
                                 const double alpha,
                                 const arma::vec& diag_norm,
                                 const bool update_eligible,
                                 std::map<std::string, double>* bench) const {
  auto t0 = Bench::tic();
  // m_dX may have been skipped at fit time for objectives that never call
  // populate_Model (Nystrom, light Vecchia) -- if a caller reaches here
  // anyway (e.g. the exact logLikelihoodFun/leaveOneOutFun/logMargPostFun API
  // called explicitly on such a model), lazily rebuild it locally: this whole
  // call is already O(n^3), so the extra one-time O(n^2) build here doesn't
  // change its complexity class.
  arma::mat dX_local;
  const arma::mat& dX = m_dX.is_empty() ? (dX_local = LinearAlgebra::compute_dX(m_X)) : m_dX;

  // Invalidate cached Linv/Rinv so gradient code recomputes them for the new L
  m.Linv = arma::mat();
  m.Rinv = arma::mat();
  if (update_eligible) {
    m.L = LinearAlgebra::update_cholCov(&(m.R), dX, theta, _Cov, alpha, diag_norm, m_T, m_R);
  } else {
    m.L = LinearAlgebra::cholCov(&(m.R), dX, theta, _Cov, alpha, diag_norm);
  }
  t0 = Bench::toc(bench, "R = _Cov(dX) & L = Chol(R)", t0);

  // Rinv (R^-1 = L^-T * L^-1) is NOT computed here anymore: it is only ever
  // consumed by an analytic THETA GRADIENT (Kriging::_logLikelihood's
  // grad_out != nullptr branch, WarpKriging::concentrated_ll_and_grad_theta /
  // warp_gradient) -- never by predict(), and never by a refit=false
  // update() (which only re-profiles beta/sigma2 at the CURRENT theta, no
  // gradient). It used to be built unconditionally right here regardless,
  // an O(n^3) dense triangular solve against the identity that swamped the
  // (genuinely O(n_old^2*n_u), already-incremental) Cholesky block update
  // above by ~2 orders of magnitude at n=8000/n_u=20 -- silently undoing
  // most of update()'s own speedup. Each gradient consumer now computes it
  // lazily on first actual need (same size/null-check pattern
  // Kriging::_leaveOneOut already uses for Linv, a few lines below).

  // Direct GLS: compute whitened matrices using triangular solves
  m.Fstar = LinearAlgebra::solve_lower(m.L, m_F);
  m.ystar = LinearAlgebra::solve_lower(m.L, m_y);
  t0 = Bench::toc(bench, "F* = L \\ F, y* = L \\ y", t0);

  // Gram matrix and its upper Cholesky
  arma::mat FtRinvF = m.Fstar.t() * m.Fstar;
  m.Rstar = LinearAlgebra::chol_upper(FtRinvF);
  t0 = Bench::toc(bench, "R* = chol(F*'F*)", t0);

  // Qstar is computed on demand in gradient/LOO functions when needed
  m.Qstar = arma::mat();

  // Always compute GLS β̂ for residual and SSE (even when beta is fixed
  // externally). Derived wrappers overwrite m.betahat afterwards if needed.
  arma::vec betahat_gls
      = LinearAlgebra::solve_upper(m.Rstar, LinearAlgebra::solve_lower(m.Rstar.t(), m.Fstar.t() * m.ystar));
  t0 = Bench::toc(bench, "^b = R*^-1 R*'^-1 F*'y*", t0);
  m.betahat = betahat_gls;

  arma::vec residual = m_y - m_F * betahat_gls;
  m.Estar = LinearAlgebra::solve_lower(m.L, residual);
  m.SSEstar = arma::dot(m.Estar, m.Estar);
  t0 = Bench::toc(bench, "z = L \\ (y - F*b), SSE = z'z", t0);
}

KrigingImpl::KModel KrigingImpl::allocate_KModel() const {
  const arma::uword n = m_X.n_rows;
  const arma::uword p = m_F.n_cols;

  KModel m{};
  m.R = arma::mat(n, n, arma::fill::none);
  m.L = arma::mat(n, n, arma::fill::none);
  m.Linv = arma::mat();  // filled on demand in gradient computation
  m.Rinv = arma::mat();  // computed lazily, on demand, by whichever gradient consumer first needs it
  m.Fstar = arma::mat(n, p, arma::fill::none);
  m.ystar = arma::vec(n, arma::fill::none);
  m.Rstar = arma::mat(p, p, arma::fill::none);
  m.Qstar = arma::mat(n, p, arma::fill::none);
  m.Estar = arma::vec(n, arma::fill::none);
  m.betahat = arma::vec(p, arma::fill::none);
  return m;
}

std::tuple<arma::vec, arma::vec, arma::mat, arma::mat, arma::mat> KrigingImpl::predict_impl(
    const arma::mat& X_n,
    bool return_stdev,
    bool return_cov,
    bool return_deriv,
    double R_on_factor,
    double R_nn_factor,
    const arma::vec& R_nn_diag,
    double var_scale,
    const FeatureMap& phi,
    const FeatureJacobian& jac) const {
  arma::uword n_n = X_n.n_rows;
  arma::uword n_o = m_X.n_rows;
  arma::uword d = m_X.n_cols;  // kernel/feature space dim
  arma::uword d_input = jac ? X_n.n_cols : d;
  if (!phi && X_n.n_cols != d)
    throw std::runtime_error("Predict locations have wrong dimension: " + std::to_string(X_n.n_cols) + " instead of "
                             + std::to_string(d));

  arma::vec yhat_n = arma::vec(n_n, arma::fill::none);
  arma::vec ysd2_n = arma::vec(n_n, arma::fill::zeros);
  arma::mat Sigma_n = arma::mat(n_n, n_n, arma::fill::zeros);
  arma::mat Dyhat_n = arma::mat(n_n, d_input, arma::fill::zeros);
  arma::mat Dysd2_n = arma::mat(n_n, d_input, arma::fill::zeros);
  // Floor of the predicted variance (units of sigma2): the variance is 1 - r'R^-1 r + ..., computed from terms
  // of order 1, so its rounding is a few eps
  const double predict_var_floor = 4.0 * arma::datum::eps;
  arma::uvec ysd2_at_floor(n_n, arma::fill::zeros);

  arma::mat Xn_o = trans(m_X);  // already in feature space (normalized)
  arma::mat Xn_n = X_n;
  // Normalize X_n (row-vector form; equivalent to Warp's per-dim mask when centerX/scaleX
  // are 0/1 for non-continuous dims — see W-2 key insight in refactor_table.md)
  Xn_n.each_row() -= m_centerX;
  Xn_n.each_row() /= m_scaleX;

  // Apply feature map phi if provided; trend is in feature space.
  // Keep input-space copy (d_input × n_n, transposed) for jac_fn calls in deriv loop.
  arma::mat Xn_n_input;  // d_input × n_n; only used when jac is set
  arma::mat F_n;
  if (phi) {
    if (jac)
      Xn_n_input = trans(Xn_n);       // save input-space before overwriting with phi-space
    arma::mat Xn_n_feat = phi(Xn_n);  // n_n × d
    F_n = Trend::regressionModelMatrix(m_regmodel, Xn_n_feat);
    Xn_n = trans(Xn_n_feat);  // d × n_n  (phi-space from here on)
  } else {
    F_n = Trend::regressionModelMatrix(m_regmodel, Xn_n);
    Xn_n = trans(Xn_n);  // d × n_n
  }

  auto t0 = Bench::tic();
  arma::mat R_on = arma::mat(n_o, n_n, arma::fill::none);
  libKriging::detail::parallel_for(
      0,
      static_cast<long>(n_o * n_n),
      512,
      libKriging::detail::omp_threads(),
      n_o * n_n >= 40000,
      [&](long lo, long hi) {
        for (arma::uword idx = static_cast<arma::uword>(lo); idx < static_cast<arma::uword>(hi); idx++) {
          const arma::uword i = idx / n_n, j = idx % n_n;
          arma::vec dij = Xn_o.col(i) - Xn_n.col(j);
          if (dij.is_zero(arma::datum::eps))
            R_on.at(i, j) = 1.0;
          else
            R_on.at(i, j) = _Cov(dij, m_theta) * R_on_factor;
        }
      });
  t0 = Bench::toc(nullptr, "R_on       ", t0);

  arma::mat Rstar_on = LinearAlgebra::solve_lower(m_T, R_on);
  t0 = Bench::toc(nullptr, "Rstar_on   ", t0);

  yhat_n = F_n * m_beta + trans(Rstar_on) * m_z;
  t0 = Bench::toc(nullptr, "yhat_n     ", t0);

  // Un-normalize predictor
  yhat_n = m_centerY + m_scaleY * yhat_n;

  arma::mat Fhat_n = trans(Rstar_on) * m_M;
  arma::mat E_n = F_n - Fhat_n;
  arma::mat Ecirc_n = LinearAlgebra::rsolve_upper(m_circ, E_n);
  t0 = Bench::toc(nullptr, "Ecirc_n    ", t0);

  if (return_stdev) {
    ysd2_n = 1.0 - sum(Rstar_on % Rstar_on, 0).as_col() + sum(Ecirc_n % Ecirc_n, 1).as_col();
    // Floor instead of 0: at (or, for warped inputs, near) a training point the variance is a difference of
    // nearly equal terms, i.e. rounding of order eps and often negative. A zero stdev would give NaN
    // NLPD / stdev derivative, so it is floored at the rounding level (in units of sigma2).
    ysd2_n.transform(
        [&](double val) { return (std::isnan(val) || val < predict_var_floor ? predict_var_floor : val); });
    ysd2_at_floor = ysd2_n <= predict_var_floor;
    ysd2_n *= var_scale * m_scaleY * m_scaleY;
    t0 = Bench::toc(nullptr, "ysd2_n     ", t0);
  }

  if (return_cov) {
    arma::mat R_nn = arma::mat(n_n, n_n, arma::fill::none);
    LinearAlgebra::covMat_sym_X(&R_nn, Xn_n, m_theta, _Cov, R_nn_factor, R_nn_diag);
    t0 = Bench::toc(nullptr, "R_nn       ", t0);

    Sigma_n = R_nn - trans(Rstar_on) * Rstar_on + Ecirc_n * trans(Ecirc_n);
    Sigma_n *= var_scale * m_scaleY * m_scaleY;
    t0 = Bench::toc(nullptr, "Sigma_n    ", t0);
  }

  if (return_deriv) {
    for (arma::uword i = 0; i < n_n; i++) {
      // dR_on / dPhi_k — (n_o × d) in feature space
      arma::mat DR_on_i = arma::mat(n_o, d, arma::fill::none);
      for (arma::uword j = 0; j < n_o; j++) {
        DR_on_i.row(j) = R_on.at(j, i) * trans(_DlnCovDx(Xn_n.col(i) - Xn_o.col(j), m_theta));
      }
      t0 = Bench::toc(nullptr, "DR_on_i    ", t0);

      // dF / dPhi_k — (d × p) in feature space (analytical)
      arma::mat DF_n_i = Trend::regressionModelDerivative(m_regmodel, Xn_n.col(i));
      t0 = Bench::toc(nullptr, "DF_n_i     ", t0);

      // Chain rule: dR_on/dx = dR_on/dPhi * J,  dF/dx = J.t() * dF/dPhi   (d_input cols)
      // jac receives the normalized INPUT-space point (before phi), not phi-space.
      if (jac) {
        arma::mat J_i = jac(Xn_n_input.col(i));  // d_phi × d_input
        DR_on_i = DR_on_i * J_i;                 // n_o × d_input
        DF_n_i = J_i.t() * DF_n_i;               // d_input × p
      }

      arma::mat W_i = LinearAlgebra::solve_lower(m_T, DR_on_i);
      t0 = Bench::toc(nullptr, "W_i        ", t0);
      Dyhat_n.row(i) = trans(DF_n_i * m_beta + trans(W_i) * m_z);
      t0 = Bench::toc(nullptr, "Dyhat_n    ", t0);

      if (return_stdev) {
        arma::mat DEcirc_n_i = LinearAlgebra::solve_lower(m_circ.t(), trans(DF_n_i - W_i.t() * m_M));
        // the floored variance is rounding: its derivative is defined as 0
        if (!ysd2_at_floor.at(i))
          Dysd2_n.row(i) = -2 * Rstar_on.col(i).t() * W_i + 2 * Ecirc_n.row(i) * DEcirc_n_i;
        t0 = Bench::toc(nullptr, "Dysd2_n    ", t0);
      }
    }
    // Chain rule for the normalization Xn = (X - centerX)/scaleX, yn = (y - centerY)/scaleY:
    //   dyhat/dX_j = scaleY · dyhat_n/dXn_j / scaleX_j  (and likewise, once, for dysd2/dX_j)
    Dyhat_n *= m_scaleY;
    Dyhat_n.each_row() /= m_scaleX;
    Dysd2_n *= var_scale * m_scaleY * m_scaleY;
    Dysd2_n.each_row() /= m_scaleX;
  }

  return std::make_tuple(std::move(yhat_n),
                         std::move(arma::sqrt(ysd2_n)),
                         std::move(Sigma_n),
                         std::move(Dyhat_n),
                         std::move(Dysd2_n / (2 * arma::sqrt(ysd2_n) * arma::mat(1, d_input, arma::fill::ones))));
}

void KrigingImpl::check_noise_variances(const arma::vec& noise_var, const char* what) {
  if (noise_var.has_nan() || arma::any(noise_var < 0.0))
    throw std::invalid_argument(std::string(what) + ": with_noise must hold non-negative noise variances");
}

void KrigingImpl::add_noise_draws(arma::mat& y_n, const arma::vec& noise_var, const char* what) {
  if (noise_var.n_elem == 0)
    return;
  check_noise_variances(noise_var, what);
  arma::mat eps = Random::randn_mat(y_n.n_rows, y_n.n_cols);
  if (noise_var.n_elem == 1)
    eps *= std::sqrt(noise_var.at(0));
  else
    eps.each_col() %= arma::sqrt(noise_var);
  y_n += eps;
}

bool KrigingImpl::build_separable_cov(const std::string& covType,
                                      const arma::mat& Xt,
                                      const arma::vec& theta,
                                      arma::mat& R,
                                      std::vector<arma::mat>* dR) {
  const bool is_gauss = (covType == "gauss");
  const bool is_exp = (covType == "exp");
  const bool is_m32 = (covType == "matern3_2");
  const bool is_m52 = (covType == "matern5_2");
  if (!(is_gauss || is_exp || is_m32 || is_m52))
    return false;

  constexpr arma::uword kMaxD = 64;  // per-pair stack scratch; ARD dims never approach this
  const arma::uword d = Xt.n_rows;
  const arma::uword n = Xt.n_cols;
  if (d == 0 || d > kMaxD)
    return false;

  const double SQ3 = std::sqrt(3.0);
  const double SQ5 = std::sqrt(5.0);
  const double* Xtm = Xt.memptr();
  const double* th = theta.memptr();

  R.set_size(n, n);
  if (dR != nullptr)
    dR->assign(d, arma::mat(n, n, arma::fill::none));

  auto s_of = [&](double uk) -> double {
    if (is_gauss)
      return 0.5 * uk * uk;
    if (is_exp)
      return uk;
    if (is_m32) {
      const double dd = SQ3 * uk;
      return dd - std::log1p(dd);
    }
    const double dd = SQ5 * uk;  // matern5_2
    return dd - std::log1p(dd + dd * dd / 3.0);
  };
  auto h_of = [&](double uk, double thk) -> double {  // d(ln Cov)/d(theta_k), >= 0
    if (is_gauss)
      return uk * uk / thk;
    if (is_exp)
      return uk / thk;
    if (is_m32) {
      const double dd = SQ3 * uk;
      return (dd * dd / (1.0 + dd)) / thk;
    }
    const double dd = SQ5 * uk;  // matern5_2
    const double a = 1.0 + dd;
    const double b = dd * dd / 3.0;
    return (a * b / (a + b)) / thk;
  };

  // Column j is written entirely by the thread that owns it (both R(i,j) for
  // i>j and the mirrored R(j,i)); disjoint threads touch disjoint pairs, so
  // the symmetric double-write is race-free without per-thread accumulators.
  // column j writes R(i, j) and R(j, i) for i > j, and R(j, j): disjoint cells
  libKriging::detail::parallel_for(
      0,
      static_cast<long>(n),
      16,
      libKriging::detail::omp_threads(),
      n >= static_cast<arma::uword>(libKriging::detail::omp_min_rows()),
      [&](long lo, long hi) {
        for (arma::uword j = static_cast<arma::uword>(lo); j < static_cast<arma::uword>(hi); ++j) {
          double u[kMaxD];
          for (arma::uword i = j + 1; i < n; ++i) {
            double S = 0.0;
            for (arma::uword k = 0; k < d; ++k) {
              const double uk = std::abs(Xtm[k + i * d] - Xtm[k + j * d]) / th[k];
              u[k] = uk;
              S += s_of(uk);
            }
            const double rij = std::exp(-S);
            R.at(i, j) = rij;
            R.at(j, i) = rij;
            if (dR != nullptr) {
              for (arma::uword k = 0; k < d; ++k) {
                const double v = rij * h_of(u[k], th[k]);
                (*dR)[k].at(i, j) = v;
                (*dR)[k].at(j, i) = v;
              }
            }
          }
          R.at(j, j) = 1.0;
          if (dR != nullptr)
            for (arma::uword k = 0; k < d; ++k)
              (*dR)[k].at(j, j) = 0.0;  // dR/dtheta_k vanishes at dx = 0
        }
      });
  return true;
}

std::tuple<arma::vec, arma::vec> KrigingImpl::predictIterative_impl(const arma::mat& X_n,
                                                                    bool return_stdev,
                                                                    arma::uword max_iter,
                                                                    double tol,
                                                                    bool use_nystrom_precond,
                                                                    arma::uword precond_rank,
                                                                    const FeatureMap& phi,
                                                                    const arma::mat* RinvFY_cache) const {
  const arma::uword n = m_X.n_rows;
  const arma::uword d = m_X.n_cols;
  if (!phi && X_n.n_cols != d)
    throw std::invalid_argument("predict (method iterative): X_n has wrong dimension: " + std::to_string(X_n.n_cols)
                                + " instead of " + std::to_string(d));

  if (max_iter == 0)
    // n is CG's exact-arithmetic convergence bound, but GP covariance
    // matrices are commonly ill-conditioned enough (smooth kernels, many
    // points) that round-off keeps the true error shrinking well past that
    // point in practice (see LinearAlgebra::conjugateGradient's periodic
    // residual-recompute comment) -- 2n is a more realistic default budget.
    max_iter = 2 * n;

  // RmulBatched applies R to a WHOLE block V (n x k) at once: R(i,j) =
  // _Cov(X_i - X_j, theta) for i != j, 1 on the diagonal (correlation
  // matrix, no noise channel). m_X is already in feature space when phi is
  // set (same convention as predict_impl), so this needs no phi of its own.
  // Matrix-free fallback: O(n) memory (R never materialized), O(n^2) time
  // per call, batched the same way as Kriging::_logLikelihoodIterative's
  // RmulBatched (every transcendental Cov() evaluated once, applied to all
  // k columns -- shared across the n_n prediction points' stdev solve
  // instead of paying k separate covariance sweeps). Superseded by the
  // dense fast path below when it applies.
  const arma::mat Xt = m_X.t();  // d x n, contiguous columns for cache-friendly access
  const arma::vec& theta = m_theta;
  const auto& cov = _Cov;
  const double* Xt_mem = Xt.memptr();
  arma::mat R_dense;  // populated below, after the GPU-dispatch block, iff `dense`
  bool dense = false;
  auto RmulBatched = [Xt_mem, &theta, &cov, n, d, &dense, &R_dense](const arma::mat& V) -> arma::mat {
    if (dense)
      return R_dense * V;  // BLAS-3, R materialized once below
    // diag of R is 1: V plus the off-diagonal terms, each R_ij evaluated once
    arma::mat out = libKriging::detail::symmetric_pair_sum(
        n, d, V.n_cols, [&](arma::uword i, arma::uword j, arma::vec& dx, arma::mat& acc) {
          libKriging::detail::pair_delta(Xt_mem, d, i, j, dx.memptr());
          const double c = cov(dx, theta);
          for (arma::uword col = 0; col < V.n_cols; ++col) {
            acc.at(i, col) += c * V.at(j, col);
            acc.at(j, col) += c * V.at(i, col);
          }
        });
    out += V;
    return out;
  };

  // Optional Nystrom preconditioner: a rank-precond_rank factor of R at the
  // model's own (fixed, already-fitted) theta -- unlike the LLNystrom
  // objective's landmarks, no cross-theta smoothness constraint applies
  // here, so this can legitimately be built exactly at the theta being used
  // for prediction rather than at a generic reference. PinvBatched is left
  // empty (== plain CG) when disabled, matching the prior behavior exactly.
  std::unique_ptr<LinearAlgebra::WoodburyFactorization> woodbury_pc;
  std::function<arma::mat(const arma::mat&)> PinvBatched;
  // Automatic mode (see LinearAlgebra::cg_auto_precond): when not requested
  // explicitly, build the factor anyway and keep it only if it captures a
  // large enough share of trace(R) = n.
  const arma::uword pc_rank = std::min(precond_rank, n);
  const bool auto_pc = !use_nystrom_precond && LinearAlgebra::cg_auto_precond && pc_rank > 0 && n >= 2 * pc_rank;
  if (use_nystrom_precond || auto_pc) {
    arma::vec diag_resid;
    arma::mat U_pc
        = LinearAlgebra::nystromFactor(&diag_resid, m_X, m_theta, _Cov, /*factor=*/1.0, ones, pc_rank, 1e-12);
    const double captured = 1.0 - arma::sum(diag_resid) / static_cast<double>(n);
    const bool keep = use_nystrom_precond || captured >= LinearAlgebra::cg_auto_precond_min_captured;
    if (keep) {
      arma::vec D_pc = arma::clamp(diag_resid, LinearAlgebra::num_nugget, arma::datum::inf);
      // Factor the Woodbury correction ONCE here (O(n*k^2 + k^3)) and reuse it
      // for every CG iteration's PinvBatched(V) call (O(n*k + k^2) per column)
      // -- calling LinearAlgebra::woodbury_solve fresh per apply would redo
      // that factorization every iteration, making the "cheap" preconditioner
      // apply as expensive as the O(n^2) matvec it's meant to help avoid.
      woodbury_pc = std::make_unique<LinearAlgebra::WoodburyFactorization>(U_pc, D_pc);
      PinvBatched = [&woodbury_pc](const arma::mat& V) -> arma::mat { return woodbury_pc->solve(V); };
    }
  }

  // GPU-accelerated CG solve when built with -DENABLE_CUDA_ITERATIVE=ON or
  // -DENABLE_HIP_ITERATIVE=ON, a device is available at runtime, no Nystrom
  // preconditioner is in play (neither GPU path implements it yet), and the
  // kernel has a device-side implementation (see CudaLinearAlgebraKernel.cu
  // / HipLinearAlgebraKernel.hip.cpp). CUDA is tried first when both are
  // compiled in. Falls back to the CPU RmulBatched-based
  // LinearAlgebra::conjugateGradientBatched otherwise.
  // GPU dispatch as a backend-agnostic std::function (see the same pattern
  // in Kriging::_logLikelihoodIterative). Empty -> CPU fallback.
  std::function<arma::mat(const arma::mat&, double, const arma::mat*)> gpuCgSolve;
#define LK_PRED_GPU_BIND(NS)                                                                                          \
  gpuCgSolve = [&](const arma::mat& B, double solve_tol, const arma::mat* x0) {                                       \
    return woodbury_pc                                                                                                \
               ? NS::conjugateGradient(Xt,                                                                            \
                                       theta,                                                                         \
                                       m_covType,                                                                     \
                                       B,                                                                             \
                                       max_iter,                                                                      \
                                       solve_tol,                                                                     \
                                       woodbury_pc->U(),                                                              \
                                       woodbury_pc->Dinv(),                                                           \
                                       woodbury_pc->McholLower(),                                                     \
                                       nullptr,                                                                       \
                                       x0)                                                                            \
               : NS::conjugateGradient(                                                                               \
                   Xt, theta, m_covType, B, max_iter, solve_tol, arma::mat(), arma::vec(), arma::mat(), nullptr, x0); \
  }
#ifdef LIBKRIGING_USE_CUDA_ITERATIVE
  if (LinearAlgebraCuda::enabled() && LinearAlgebraCuda::supports(m_covType))
    LK_PRED_GPU_BIND(LinearAlgebraCuda);
#endif
#ifdef LIBKRIGING_USE_HIP_ITERATIVE
  if (!gpuCgSolve && LinearAlgebraHip::enabled() && LinearAlgebraHip::supports(m_covType))
    LK_PRED_GPU_BIND(LinearAlgebraHip);
#endif
#ifdef LIBKRIGING_USE_SYCL_ITERATIVE
  if (!gpuCgSolve && LinearAlgebraSycl::enabled() && LinearAlgebraSycl::supports(m_covType))
    LK_PRED_GPU_BIND(LinearAlgebraSycl);
#endif
#ifdef LIBKRIGING_USE_METAL_ITERATIVE
  if (!gpuCgSolve && LinearAlgebraMetal::enabled() && LinearAlgebraMetal::supports(m_covType))
    LK_PRED_GPU_BIND(LinearAlgebraMetal);
#endif
#undef LK_PRED_GPU_BIND

  // Materialize the dense fast-path R (see the R_dense/dense declarations
  // above) when the CPU path will actually run the solves -- skipped when a
  // GPU backend is bound (it has its own device matvecs). No gradient is
  // ever needed here, so unlike _logLikelihoodIterative this never
  // materializes the dR/dtheta_k blocks.
  if (!gpuCgSolve) {
    std::size_t budget_mb = 6144;
    if (const char* e = std::getenv("LK_ITERATIVE_DENSE_MAX_MB")) {
      try {
        budget_mb = static_cast<std::size_t>(std::stoull(e));
      } catch (...) { /* keep default */
      }
    }
    const double need_mb = static_cast<double>(n) * static_cast<double>(n) * 8.0 / (1024.0 * 1024.0);
    if (budget_mb > 0 && need_mb <= static_cast<double>(budget_mb))
      dense = build_separable_cov(m_covType, Xt, theta, R_dense, nullptr);
  }

  auto cgSolve = [&](const arma::mat& B, double solve_tol, const arma::mat* x0 = nullptr) -> arma::mat {
    if (gpuCgSolve)
      return gpuCgSolve(B, solve_tol, x0);
    return LinearAlgebra::conjugateGradientBatched(RmulBatched, B, max_iter, solve_tol, PinvBatched, nullptr, x0);
  };

  // RinvFY_cache = [R^-1*F | R^-1*y] from the last fit/updateIterative
  // commit (see this function's doc comment). Two of the CG right-hand
  // sides below don't depend on X_n at all -- reusing the cache as their
  // warm start turns an always-cold O(n^2 * iters) solve into, typically,
  // (close to) zero further CG iterations, without changing what either
  // solve converges to (a bad/stale x0 just costs iterations, never
  // correctness -- see LinearAlgebra::conjugateGradientBatched's X0).
  const arma::uword p = m_F.n_cols;
  const bool have_cache = (RinvFY_cache != nullptr) && (RinvFY_cache->n_rows == n) && (RinvFY_cache->n_cols == p + 1);
  arma::vec x0_mean;
  arma::mat RinvFY_cache_F;  // R^-1*F, read straight off the cache -- exact, not approximate
  if (have_cache) {
    x0_mean = RinvFY_cache->col(p);  // R^-1*y
    if (p > 0) {
      RinvFY_cache_F = RinvFY_cache->cols(0, p - 1);
      x0_mean -= RinvFY_cache_F * m_beta;  // R^-1*y - (R^-1*F)*beta ~= R^-1*resid
    }
  }

  // The posterior MEAN solve gets the caller's full `tol`. The posterior
  // VARIANCE only needs quad = sum(R_on % R^-1 R_on) to a few digits (its
  // error in the returned stdev is ~ sqrt of the solve's relative
  // residual), so its solves use sqrt(tol) -- roughly halves their CG
  // iteration count, which is the whole cost of return_stdev=True (one
  // solve per prediction point). Never tighter than `tol` itself.
  const double stdev_tol = std::max(std::sqrt(std::max(tol, 0.0)), tol);

  arma::mat Xn_n = X_n;
  Xn_n.each_row() -= m_centerX;
  Xn_n.each_row() /= m_scaleX;
  arma::mat F_n;
  if (phi) {
    arma::mat Xn_n_feat = phi(Xn_n);  // n_n × d
    F_n = Trend::regressionModelMatrix(m_regmodel, Xn_n_feat);
    Xn_n = trans(Xn_n_feat);  // d × n_n (phi-space from here on)
  } else {
    F_n = Trend::regressionModelMatrix(m_regmodel, Xn_n);
    Xn_n = trans(Xn_n);  // d × n_n
  }
  const arma::uword n_n = X_n.n_rows;

  // One CG solve, reused for every prediction point's mean.
  const arma::vec resid = m_y - m_F * m_beta;
  const arma::mat w = have_cache ? cgSolve(resid, tol, &x0_mean) : cgSolve(resid, tol);

  arma::mat R_on = arma::mat(n, n_n, arma::fill::none);
  LinearAlgebra::covMat_rect(&R_on, Xt, Xn_n, m_theta, _Cov, 1.0);

  arma::vec mean = F_n * m_beta + R_on.t() * w.col(0);
  mean = mean * m_scaleY + m_centerY;

  arma::vec stdev;
  if (return_stdev) {
    // One CG solve PER prediction point (R_on's columns don't share a
    // Krylov subspace): O(n^2 * iters * n_n) total, hence opt-in. Solved to
    // the looser stdev_tol (see above); re-solved at the tighter `tol` below
    // for any column the cancellation guard flags as unreliable.
    arma::mat V = cgSolve(R_on, stdev_tol);
    arma::vec quad = arma::sum(R_on % V, 0).t();

    arma::vec gls_correction(n_n, arma::fill::zeros);
    arma::mat W_F;
    if (m_F.n_cols > 0) {
      // GLS correction for the trend coefficients' own estimation
      // uncertainty -- same u^T(F^T R^-1 F)^-1 u term predict_impl computes
      // via m_circ/Ecirc_n, here via one extra shared CG solve (R^-1 F,
      // m_F.n_cols right-hand sides) plus O(n_n * p^2) small dense linear
      // algebra (p = m_F.n_cols). Without it, stdev reduces to the
      // simple-kriging formula (correct only when the trend is known
      // rather than estimated, i.e. regmodel == "none") and understates
      // the prediction uncertainty otherwise. Solved at the full `tol`, not
      // stdev_tol: only m_F.n_cols right-hand sides (typically 1-2), so
      // tightening it is essentially free, and its error feeds straight
      // into the 1-quad+correction cancellation below.
      W_F = have_cache ? cgSolve(m_F, tol, &RinvFY_cache_F) : cgSolve(m_F, tol);
      const arma::mat FtRinvF = m_F.t() * W_F;
      const arma::mat E = F_n - R_on.t() * W_F;
      const arma::mat correction_rhs = arma::solve(FtRinvF, E.t());
      gls_correction = arma::sum(E.t() % correction_rhs, 0).t();
    }

    arma::vec var_raw = 1.0 - quad + gls_correction;

    // Catastrophic-cancellation guard: `1 - quad` subtracts two O(1)
    // quantities, so the stdev_tol solve's absolute error in quad (~
    // stdev_tol times a column's norm) can dominate var_raw wherever it's
    // small -- which happens whenever a prediction point coincides with, or
    // sits extremely close to, a training point, where the true posterior
    // variance is tiny (down to the nugget floor). Left alone this silently
    // returns a wildly-inflated stdev there, or even a hard 0 once var_raw
    // goes slightly negative and gets clamped below. Re-solve just the
    // flagged columns (rare in practice: normal predictions land away from
    // the training set) at the caller's full `tol` instead -- but ONLY when
    // the original solve actually reached stdev_tol: a column CG couldn't
    // converge (large n, no preconditioner, ill-conditioned R) already hit
    // its max_iter budget, so re-solving at a tighter tol with that same
    // budget repeats the identical failure, silently doubling this
    // function's cost for no accuracy gain (found via a genuine slowdown in
    // docs/math/predictiterative_vs_cholesky.ipynb's large-n sweep). Checked
    // directly against RmulBatched (backend-agnostic: the same CPU matvec
    // regardless of which cgSolve dispatched to) rather than trusted from
    // the solver's own bookkeeping, so this works whether cgSolve ran on
    // CPU or GPU.
    const double cancel_guard = 16.0 * stdev_tol;
    const arma::uvec candidates = arma::find(var_raw < cancel_guard);
    if (!candidates.is_empty()) {
      const arma::mat R_on_cand = R_on.cols(candidates);
      const arma::mat resid_cand = R_on_cand - RmulBatched(V.cols(candidates));
      const arma::vec cand_norm
          = arma::clamp(arma::sqrt(arma::sum(arma::square(R_on_cand), 0)).t(), 1e-300, arma::datum::inf);
      const arma::vec relres = arma::sqrt(arma::sum(arma::square(resid_cand), 0)).t() / cand_norm;
      const arma::uvec bad = candidates.elem(arma::find(relres <= 2.0 * stdev_tol));
      if (!bad.is_empty()) {
        const arma::mat R_on_bad = R_on.cols(bad);
        const arma::mat V_bad = cgSolve(R_on_bad, tol);
        quad(bad) = arma::sum(R_on_bad % V_bad, 0).t();
        if (m_F.n_cols > 0) {
          const arma::mat FtRinvF = m_F.t() * W_F;
          const arma::mat E_bad = F_n.rows(bad) - R_on_bad.t() * W_F;
          const arma::mat correction_rhs_bad = arma::solve(FtRinvF, E_bad.t());
          gls_correction(bad) = arma::sum(E_bad.t() % correction_rhs_bad, 0).t();
        }
        var_raw = 1.0 - quad + gls_correction;
      }
    }

    // floored at the rounding level (in units of sigma2), as predict_impl
    const double predict_var_floor = 4.0 * arma::datum::eps;
    var_raw.transform(
        [&](double val) { return (std::isnan(val) || val < predict_var_floor) ? predict_var_floor : val; });
    stdev = arma::sqrt(m_sigma2 * var_raw);
    stdev *= m_scaleY;
  }
  return {mean, stdev};
}

arma::mat KrigingImpl::simulate_impl(int nsim,
                                     int seed,
                                     const arma::mat& X_n,
                                     bool will_update,
                                     double R_on_factor,
                                     bool R_on_coincident_to_one,
                                     double R_nn_factor,
                                     const arma::vec& R_nn_diag,
                                     double Sigma_divisor,
                                     const FeatureMap& phi) {
  arma::uword n_n = X_n.n_rows;
  arma::uword n_o = m_X.n_rows;
  // Use m_centerX.n_elem for dimension check: works for both plain Kriging
  // (input_dim == feature_dim) and WarpKriging/MLPKriging (input_dim != feature_dim).
  if (X_n.n_cols != m_centerX.n_elem)
    throw std::runtime_error("Simulate locations have wrong dimension: " + std::to_string(X_n.n_cols) + " instead of "
                             + std::to_string(m_centerX.n_elem));

  arma::mat Xn_o = trans(m_X);  // already in kernel-input space (phi-space for Warp/MLP)
  arma::mat Xn_n = X_n;
  Xn_n.each_row() -= m_centerX;
  Xn_n.each_row() /= m_scaleX;
  // Apply feature map if provided (WarpKriging/MLPKriging): Xn_n → Φ(Xn_n)
  if (phi)
    Xn_n = phi(Xn_n);

  arma::mat F_n = Trend::regressionModelMatrix(m_regmodel, Xn_n);
  Xn_n = trans(Xn_n);

  auto t0 = Bench::tic();
  arma::mat R_nn = arma::mat(n_n, n_n, arma::fill::none);
  LinearAlgebra::covMat_sym_X(&R_nn, Xn_n, m_theta, _Cov, R_nn_factor, R_nn_diag);
  t0 = Bench::toc(nullptr, "R_nn          ", t0);

  arma::mat R_on = arma::mat(n_o, n_n, arma::fill::none);
  libKriging::detail::parallel_for(
      0,
      static_cast<long>(n_o * n_n),
      512,
      libKriging::detail::omp_threads(),
      n_o * n_n >= 40000,
      [&](long lo, long hi) {
        for (arma::uword idx = static_cast<arma::uword>(lo); idx < static_cast<arma::uword>(hi); idx++) {
          const arma::uword i = idx / n_n, j = idx % n_n;
          arma::vec dij = Xn_o.col(i) - Xn_n.col(j);
          if (R_on_coincident_to_one && dij.is_zero(arma::datum::eps))
            R_on.at(i, j) = 1.0;
          else
            R_on.at(i, j) = _Cov(dij, m_theta) * R_on_factor;
        }
      });
  t0 = Bench::toc(nullptr, "R_on        ", t0);

  arma::mat Rstar_on = LinearAlgebra::solve_lower(m_T, R_on);
  t0 = Bench::toc(nullptr, "Rstar_on   ", t0);

  arma::vec yhat_n = F_n * m_beta + trans(Rstar_on) * m_z;
  t0 = Bench::toc(nullptr, "yhat_n        ", t0);

  arma::mat Fhat_n = trans(Rstar_on) * m_M;
  arma::mat E_n = F_n - Fhat_n;
  arma::mat Ecirc_n = LinearAlgebra::rsolve_upper(m_circ, E_n);
  t0 = Bench::toc(nullptr, "Ecirc_n       ", t0);

  arma::mat SigmaNoTrend_nKo = R_nn - trans(Rstar_on) * Rstar_on;
  arma::mat Sigma_nKo = SigmaNoTrend_nKo + Ecirc_n * trans(Ecirc_n);
  t0 = Bench::toc(nullptr, "Sigma_nKo     ", t0);

  arma::mat LSigma_nKo = LinearAlgebra::safe_chol_lower(Sigma_nKo / Sigma_divisor);
  t0 = Bench::toc(nullptr, "LSigma_nKo     ", t0);

  arma::mat y_n = arma::mat(n_n, nsim, arma::fill::none);
  y_n.each_col() = yhat_n;
  Random::reset_seed(seed);
  y_n += LSigma_nKo * Random::randn_mat(n_n, nsim) * std::sqrt(m_sigma2);

  y_n = m_centerY + m_scaleY * y_n;

  if (will_update) {
    lastsimup_Xn_u.clear();
    lastsim_y_n = y_n;

    lastsim_Xn_n = Xn_n;
    lastsim_seed = seed;
    lastsim_nsim = nsim;

    lastsim_L_oCn = Rstar_on;
    lastsim_Ecirc_nKo = Ecirc_n;
    lastsim_LSigma_nKo = LSigma_nKo;
  }

  return y_n;
}

void KrigingImpl::clear_simulate_state() {
  lastsim_Xn_n.reset();
  lastsim_y_n.reset();
  lastsim_nsim = 0;
  lastsim_seed = 0;
  lastsim_L_oCn.reset();
  lastsim_Ecirc_nKo.reset();
  lastsim_LSigma_nKo.reset();
  lastsimup_Xn_u.reset();
  lastsimup_y_u.reset();
  lastsimup_Wtild_nKu.reset();
  lastsimup_R_uo.reset();
  lastsimup_R_un.reset();
  lastsimup_R_uu.reset();
}

arma::mat KrigingImpl::update_simulate_impl(const arma::vec& y_u,
                                            const arma::mat& X_u,
                                            bool allow_cache,
                                            double R_uu_factor,
                                            const arma::vec& R_uu_diag,
                                            double R_uo_factor,
                                            double R_un_factor,
                                            bool R_un_coincident_to_one,
                                            double Sigma_divisor,
                                            const FeatureMap& phi) {
  if (y_u.n_elem != X_u.n_rows)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (" + std::to_string(X_u.n_rows) + "x"
                             + std::to_string(X_u.n_cols) + "), y: (" + std::to_string(y_u.n_elem) + ")");

  // Use m_centerX.n_elem: input_dim (invariant regardless of feature map presence)
  if (X_u.n_cols != m_centerX.n_elem)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (...x" + std::to_string(m_centerX.n_elem)
                             + "), new X: (...x" + std::to_string(X_u.n_cols) + ")");

  if (lastsim_y_n.is_empty() || lastsim_y_n.n_rows == 0)
    throw std::runtime_error("No previous simulation data available");

  const arma::uword n_n = lastsim_Xn_n.n_cols;
  const arma::uword n_o = m_X.n_rows;
  // y_u is raw; the conditioning below works on the normalized output scale (as m_y, m_sigma2)
  const arma::vec yn_u = (y_u - m_centerY) / m_scaleY;
  const arma::mat Xn_o = trans(m_X);    // already in kernel-input space
  const arma::mat Xn_n = lastsim_Xn_n;  // already in kernel-input space (phi-space when phi was used)

  const arma::uword n_u = X_u.n_rows;
  // Normalize X_u then apply feature map if provided
  arma::mat Xn_u = X_u;
  Xn_u.each_row() -= m_centerX;
  Xn_u.each_row() /= m_scaleX;
  // Apply feature map if provided (WarpKriging/MLPKriging): Xn_u → Φ(Xn_u)
  if (phi)
    Xn_u = phi(Xn_u);

  // Define regression matrix on kernel-input space
  const arma::mat F_u = Trend::regressionModelMatrix(m_regmodel, Xn_u);

  auto t0 = Bench::tic();
  Xn_u = trans(Xn_u);
  t0 = Bench::toc(nullptr, "Xn_u.t()      ", t0);

  const bool use_lastsimup = allow_cache && (!lastsimup_Xn_u.is_empty())
                             && arma::approx_equal(lastsimup_Xn_u, Xn_u, "absdiff", arma::datum::eps);
  if (!use_lastsimup) {
    lastsimup_Xn_u = Xn_u;

    // Compute covariance between updated data
    lastsimup_R_uu = arma::mat(n_u, n_u, arma::fill::none);
    LinearAlgebra::covMat_sym_X(&lastsimup_R_uu, Xn_u, m_theta, _Cov, R_uu_factor, R_uu_diag);
    t0 = Bench::toc(nullptr, "R_uu          ", t0);

    // Compute covariance between updated/old data
    lastsimup_R_uo = arma::mat(n_u, n_o, arma::fill::none);
    LinearAlgebra::covMat_rect(&lastsimup_R_uo, Xn_u, Xn_o, m_theta, _Cov, R_uo_factor);
    t0 = Bench::toc(nullptr, "R_uo          ", t0);

    // Compute covariance between updated/new data
    lastsimup_R_un = arma::mat(n_u, n_n, arma::fill::none);
    if (R_un_coincident_to_one) {
      for (arma::uword i = 0; i < n_u; i++) {
        for (arma::uword j = 0; j < n_n; j++) {
          arma::vec dij = Xn_u.col(i) - Xn_n.col(j);
          if (dij.is_zero(arma::datum::eps))
            lastsimup_R_un.at(i, j) = 1.0;
          else
            lastsimup_R_un.at(i, j) = _Cov(dij, m_theta) * R_un_factor;
        }
      }
    } else {
      LinearAlgebra::covMat_rect(&lastsimup_R_un, Xn_u, Xn_n, m_theta, _Cov, R_un_factor);
    }
    t0 = Bench::toc(nullptr, "R_un          ", t0);
  }

  if (!use_lastsimup) {
    // Covariances given the observations (universal kriging, trend included)
    arma::mat Rstar_ou = LinearAlgebra::solve_lower(m_T, lastsimup_R_uo.t());
    arma::mat Ecirc_uKo = LinearAlgebra::rsolve_upper(m_circ, F_u - Rstar_ou.t() * m_M);
    arma::mat Sigma_uKo = lastsimup_R_uu - Rstar_ou.t() * Rstar_ou + Ecirc_uKo * Ecirc_uKo.t();
    arma::mat Sigma_unKo = lastsimup_R_un - Rstar_ou.t() * lastsim_L_oCn + Ecirc_uKo * lastsim_Ecirc_nKo.t();
    t0 = Bench::toc(nullptr, "Sigma_uKo          ", t0);

    // ====================================================================
    // FOXY step #1  Extend the simulation to 'X_u', jointly with the paths
    // on 'X_n': block-Cholesky extension of the covariance given the
    // observations, driven by the same normals as simulate().  Unlike
    // conditioning y_u on the simulated y_n, this never inverts a matrix
    // built on the (dense, possibly near-singular) simulation design.
    // ====================================================================
    const arma::mat A = LinearAlgebra::solve_lower(lastsim_LSigma_nKo, Sigma_unKo.t() / Sigma_divisor);
    arma::vec S_eigval;
    arma::mat S_eigvec;
    LinearAlgebra::eig_sym(S_eigval, S_eigvec, arma::symmatu(Sigma_uKo / Sigma_divisor - A.t() * A));
    const arma::mat LS_u = S_eigvec * arma::diagmat(arma::sqrt(arma::clamp(S_eigval, 0.0, arma::datum::inf)));

    arma::vec yhat_u = F_u * m_beta + Rstar_ou.t() * m_z;
    Random::reset_seed(lastsim_seed);
    const arma::mat Z_n = Random::randn_mat(n_n, lastsim_nsim);  // the normals behind lastsim_y_n
    const arma::mat Z_u = Random::randn_mat(n_u, lastsim_nsim);  // fresh ones, independent of Z_n
    lastsimup_y_u = A.t() * Z_n + LS_u * Z_u;
    lastsimup_y_u *= std::sqrt(m_sigma2);
    lastsimup_y_u.each_col() += yhat_u;
    t0 = Bench::toc(nullptr, "y_u          ", t0);

    // ====================================================================
    // FOXY step #2  Kriging weights of the residuals at 'X_u' onto 'X_n'
    // ====================================================================
    lastsimup_Wtild_nKu = LinearAlgebra::solve(Sigma_uKo, Sigma_unKo).t();
    t0 = Bench::toc(nullptr, "Wtild_nKu          ", t0);
  }

  return lastsim_y_n + m_scaleY * lastsimup_Wtild_nKu * (arma::repmat(yn_u, 1, lastsim_nsim) - lastsimup_y_u);
}

void KrigingImpl::update_no_refit_impl(const arma::vec& y_u,
                                       const arma::mat& X_u,
                                       const std::function<void()>& extend_class_data,
                                       const std::function<KModel()>& build_model) {
  // Normalize new data using existing normalization parameters
  arma::mat Xn_u = X_u;
  Xn_u.each_row() -= m_centerX;
  Xn_u.each_row() /= m_scaleX;

  arma::vec yn_u = (y_u - m_centerY) / m_scaleY;

  const arma::uword n_total = m_X.n_rows + X_u.n_rows;

  // Append to m_X only when input and feature dimensions match.
  // When a feature map changes the dimension (e.g. MLPKriging with d_out != d_in),
  // extend_class_data() below is responsible for replacing m_X with the
  // correctly feature-mapped version, so skip the premature join here.
  if (m_X.n_cols == Xn_u.n_cols) {
    m_X = arma::join_cols(m_X, Xn_u);
  }
  m_y = arma::join_cols(m_y, yn_u);

  // Class-specific extension hook (Noise extends m_noise here)
  extend_class_data();

  // Recompute distance matrix (full recomputation for now)
  m_dX = LinearAlgebra::compute_dX(m_X);
  m_maxdX = arma::max(arma::abs(m_dX), 1);

  // Extend trend matrix
  m_F = Trend::regressionModelMatrix(m_regmodel, m_X);

  // Call the class-specific make_Model (uses incremental Cholesky update when
  // theta — and per-class alpha/sigma2 — match the cached state).
  KModel m = build_model();

  // Update member variables from the extended model
  m_T = std::move(m.L);
  m_R = std::move(m.R);
  m_M = std::move(m.Fstar);
  m_circ = std::move(m.Rstar);
  m_star = std::move(m.Qstar);
  m_Rinv = std::move(m.Rinv);

  if (m_est_beta) {
    m_beta = std::move(m.betahat);
    m_z = std::move(m.Estar);
  } else {
    // m_beta remains unchanged (fixed parameter)
    m_z = std::move(m.ystar) - m_M * m_beta;
  }

  if (m_est_sigma2) {
    m_sigma2 = m.SSEstar / n_total;
  }
  // else m_sigma2 remains unchanged (fixed parameter)
}

void KrigingImpl::commit_model(KModel& m) {
  m_R = std::move(m.R);
  m_T = std::move(m.L);
  m_Rinv = std::move(m.Rinv);
  m_M = std::move(m.Fstar);
  m_circ = std::move(m.Rstar);
  m_beta = std::move(m.betahat);
  m_z = std::move(m.Estar);
}

void KrigingImpl::dump_common_to_json(nlohmann::json& j) const {
  // _Cov_pow & std::function are embedded by make_Cov on load — caller stores
  // covType.
  j["covType"] = m_covType;
  j["X"] = to_json(m_X);
  j["centerX"] = to_json(m_centerX);
  j["scaleX"] = to_json(m_scaleX);
  j["y"] = to_json(m_y);
  j["centerY"] = m_centerY;
  j["scaleY"] = m_scaleY;
  j["normalize"] = m_normalize;
  j["regmodel"] = Trend::toString(m_regmodel);
  j["optim"] = m_optim;
  j["objective"] = m_objective;
  // Only emit `method` when approximate so exact save files stay byte-identical
  if (m_method != "exact")
    j["method"] = m_method;
  // Auxiliary data
  j["dX"] = to_json(m_dX);
  j["maxdX"] = to_json(m_maxdX);
  j["F"] = to_json(m_F);
  j["T"] = to_json(m_T);
  j["R"] = to_json(m_R);
  j["M"] = to_json(m_M);
  j["star"] = to_json(m_star);
  j["circ"] = to_json(m_circ);
  j["z"] = to_json(m_z);
  j["Rinv"] = to_json(m_Rinv);
  j["beta"] = to_json(m_beta);
  j["est_beta"] = m_est_beta;
  j["theta"] = to_json(m_theta);
  j["est_theta"] = m_est_theta;
  j["sigma2"] = m_sigma2;
  j["est_sigma2"] = m_est_sigma2;
  // Only emit `noise` when non-empty so K/Nug saved files stay byte-identical
  // to the pre-hoist format.
  if (!m_noise.is_empty())
    j["noise"] = to_json(m_noise);
}

void KrigingImpl::load_common_from_json(const nlohmann::json& j) {
  m_X = mat_from_json(j["X"]);
  m_centerX = rowvec_from_json(j["centerX"]);
  m_scaleX = rowvec_from_json(j["scaleX"]);
  m_y = colvec_from_json(j["y"]);
  m_centerY = j["centerY"].template get<decltype(m_centerY)>();
  m_scaleY = j["scaleY"].template get<decltype(m_scaleY)>();
  m_normalize = j["normalize"].template get<decltype(m_normalize)>();
  m_regmodel = Trend::fromString(j["regmodel"].template get<std::string>());
  m_optim = j["optim"].template get<decltype(m_optim)>();
  m_objective = j["objective"].template get<decltype(m_objective)>();
  m_method = j.contains("method") ? j["method"].template get<std::string>() : std::string("exact");
  m_dX = mat_from_json(j["dX"]);
  m_maxdX = colvec_from_json(j["maxdX"]);
  m_F = mat_from_json(j["F"]);
  m_T = mat_from_json(j["T"]);
  m_R = mat_from_json(j["R"]);
  m_M = mat_from_json(j["M"]);
  m_star = mat_from_json(j["star"]);
  m_circ = mat_from_json(j["circ"]);
  m_z = colvec_from_json(j["z"]);
  m_Rinv = mat_from_json(j["Rinv"]);
  m_beta = colvec_from_json(j["beta"]);
  m_est_beta = j["est_beta"].template get<decltype(m_est_beta)>();
  m_theta = colvec_from_json(j["theta"]);
  m_est_theta = j["est_theta"].template get<decltype(m_est_theta)>();
  m_sigma2 = j["sigma2"].template get<decltype(m_sigma2)>();
  m_est_sigma2 = j["est_sigma2"].template get<decltype(m_est_sigma2)>();
  if (j.contains("noise"))
    m_noise = colvec_from_json(j["noise"]);
  m_is_empty = false;
}

bool KrigingImpl::summary_top(std::ostringstream& oss,
                              const arma::mat* X_display_override,
                              const SummaryValues* values) const {
  const arma::vec& beta_disp = values ? values->beta : m_beta;
  const double sigma2_disp = values ? values->sigma2 : m_sigma2;
  const arma::vec& theta_disp = values ? values->theta : m_theta;
  const arma::vec& noise_disp = values ? values->noise : m_noise;
  const arma::mat& X_disp = X_display_override ? *X_display_override : m_X;
  if (X_disp.is_empty() || X_disp.n_rows == 0) {
    oss << "* covariance:\n";
    oss << "  * kernel: " << m_covType << "\n";
    return false;
  }

  auto vec_printer = [&oss](const arma::vec& v) {
    v.for_each([&oss, i = 0](const arma::vec::elem_type& val) mutable {
      if (i++ > 0)
        oss << ", ";
      oss << val;
    });
  };

  oss << "* data";
  oss << ((m_normalize) ? " (normalized): " : ": ") << X_disp.n_rows << "x";
  arma::rowvec Xmins = arma::min(X_disp, 0);
  arma::rowvec Xmaxs = arma::max(X_disp, 0);
  for (arma::uword i = 0; i < X_disp.n_cols; i++) {
    oss << "[" << Xmins[i] << "," << Xmaxs[i] << "]";
    if (i < X_disp.n_cols - 1)
      oss << ",";
  }
  oss << " -> " << m_y.n_elem << "x[" << arma::min(m_y) << "," << arma::max(m_y) << "]\n";
  if (!noise_disp.is_empty())
    oss << "  * noise: " << noise_disp.n_elem << "x[" << arma::min(noise_disp) << "," << arma::max(noise_disp) << "]\n";
  oss << "* trend " << Trend::toString(m_regmodel);
  oss << ((m_est_beta) ? " (est.): " : ": ");
  vec_printer(beta_disp);
  oss << "\n";
  oss << "* variance";
  oss << ((m_est_sigma2) ? " (est.): " : ": ");
  oss << sigma2_disp;
  oss << "\n";
  oss << "* covariance:\n";
  oss << "  * kernel: " << m_covType << "\n";
  oss << "  * range";
  oss << ((m_est_theta) ? " (est.): " : ": ");
  vec_printer(theta_disp);
  oss << "\n";
  return true;
}

void KrigingImpl::summary_bottom(std::ostringstream& oss) const {
  oss << "  * fit:\n";
  oss << "    * objective: " << m_objective << "\n";
  if (m_method != "exact")
    oss << "    * method: " << m_method << "\n";
  oss << "    * optim: " << m_optim << "\n";
}

arma::mat KrigingImpl::fit_setup_impl(const arma::vec& y,
                                      const arma::mat& X,
                                      const Trend::RegressionModel& regmodel,
                                      bool normalize,
                                      bool is_beta_estim,
                                      const std::optional<arma::vec>& beta,
                                      const std::optional<arma::mat>& theta,
                                      bool build_dX) {
  const arma::uword d = X.n_cols;

  // Normalization of inputs and output
  arma::rowvec centerX;
  arma::rowvec scaleX;
  double centerY;
  double scaleY;
  m_normalize = normalize;
  if (m_normalize) {
    centerX = min(X, 0);
    scaleX = max(X, 0) - min(X, 0);
    centerY = min(y);
    scaleY = max(y) - min(y);
    // a constant column / output has a zero range: keep it unscaled (only
    // centered) instead of dividing by zero
    scaleX.replace(0.0, 1.0);
    if (scaleY == 0.0)
      scaleY = 1.0;
  } else {
    centerX = arma::rowvec(d, arma::fill::zeros);
    scaleX = arma::rowvec(d, arma::fill::ones);
    centerY = 0;
    scaleY = 1;
  }
  m_centerX = centerX;
  m_scaleX = scaleX;
  m_centerY = centerY;
  m_scaleY = scaleY;
  {
    arma::mat newX = X;
    newX.each_row() -= centerX;
    newX.each_row() /= scaleX;
    arma::vec newy = (y - centerY) / scaleY;
    m_X = newX;
    m_y = newy;
  }

  // Distance matrix between points (transposed compared to m_X)
  if (build_dX) {
    m_dX = LinearAlgebra::compute_dX(m_X);
    m_maxdX = arma::max(arma::abs(m_dX), 1);
  } else {
    m_dX.reset();
    // max(abs(m_dX), 1) column-wise reduces, per dimension, to max(X) - min(X)
    // over all pairs (achieved at the extremes) -- same value, O(n*d) instead
    // of the O(n^2) compute_dX + max(abs(.)) it would otherwise take.
    m_maxdX = arma::trans(arma::max(m_X, 0) - arma::min(m_X, 0));
  }

  // Regression matrix
  m_regmodel = regmodel;
  m_F = Trend::regressionModelMatrix(regmodel, m_X);
  m_est_beta = is_beta_estim && (m_regmodel != Trend::RegressionModel::None);
  if (!m_est_beta && beta.has_value() && beta.value().n_elem > 0) {  // Force beta fixed (not estimated, no variance)
    m_est_beta = false;
    // given in the original scale: exact affine transform to the normalized one
    // (needs m_regmodel and the normalization constants, both set above)
    m_beta = beta_from_original(beta.value());
  } else
    m_est_beta = true;

  // Normalize theta0 if provided
  arma::mat theta0;
  if (theta.has_value()) {
    theta0 = theta.value();
    if ((theta0.n_cols != d) && (theta0.n_rows == d))
      theta0 = theta0.t();
    if (theta0.n_cols != d)
      throw std::runtime_error("Dimension of theta should be kx" + std::to_string(d)
                               + " (one starting point per row, one column per input) instead of "
                               + std::to_string(theta0.n_rows) + "x" + std::to_string(theta0.n_cols));
    if (m_normalize)
      theta0.each_row() /= scaleX;
  }
  return theta0;
}

// -------------------------------------------------------------------------
//  Objective-function evaluation helpers
// -------------------------------------------------------------------------

void KrigingImpl::print_bench(const std::map<std::string, double>& bench) {
  size_t num = 0;
  for (auto& kv : bench)
    num = std::max(kv.first.size(), num);
  for (auto& kv : bench)
    arma::cout << "| " << Bench::pad(kv.first, num, ' ') << " | " << kv.second << " |" << arma::endl;
}

namespace {
// Threads of the gradient pair loops: the budget from n = omp_min_rows() on, 1
// below. A performance setting only: the sums do not depend on it
// (SeparableDln.hpp, sum_pairs_dln).
int pair_loop_threads(arma::uword n) {
#ifdef LK_OMP_LOOPS
  if (n >= libKriging::detail::omp_min_rows())
    return libKriging::detail::omp_threads();
#endif
  (void)n;
  return 1;
}
}  // namespace

void KrigingImpl::compute_ll_grad_theta_vecs(const arma::mat& R,
                                             const arma::mat& Rinv,
                                             const arma::mat& x,
                                             const arma::vec& theta,
                                             arma::vec& term1_vec,
                                             arma::vec& term2_vec) const {
  const arma::uword n = m_X.n_rows;
  const arma::uword d = theta.n_elem;
  // m_dX may have been skipped at fit time for objectives that never reach
  // this exact-gradient path (Nystrom, light Vecchia) -- if a caller reaches
  // it anyway (e.g. logLikelihoodFun(..., return_grad=true) called explicitly
  // on such a model), lazily rebuild it locally: this whole function is
  // already O(n^2), so the extra one-time O(n^2) build here doesn't change
  // its complexity class.
  arma::mat dX_local;
  const arma::mat& dX = m_dX.is_empty() ? (dX_local = LinearAlgebra::compute_dX(m_X)) : m_dX;
  // dR/dtheta_k = R % D_k (zero diagonal), so over the pairs j < i:
  //   term1_k = x' dR/dtheta_k x  = sum 2 x_i x_j R_ij D_k(i, j)
  //   term2_k = -tr(R^-1 dR/dtheta_k) = sum -2 Rinv_ij R_ij D_k(i, j)
  arma::vec out(2 * d);
  libKriging::detail::with_dln(m_covType, theta, [&](const auto& dln) {
    libKriging::detail::sum_pairs_dln<2>(
        dX,
        n,
        theta,
        dln,
        pair_loop_threads(n),
        [&](arma::uword i, arma::uword j) {
          const double R_ij = R.at(j, i);  // symmetric: column i, contiguous in j
          return std::array<double, 2>{2.0 * x.at(i) * x.at(j) * R_ij, -2.0 * Rinv.at(j, i) * R_ij};
        },
        out.memptr());
  });
  term1_vec += out.head(d);
  term2_vec += out.tail(d);
}

arma::vec KrigingImpl::compute_lmp_theta_ans(const KModel& m,
                                             const arma::vec& theta,
                                             double sigma2,
                                             const arma::mat& Rinv_X_Xt_Rinv_X_inv_Xt_Rinv,
                                             const arma::mat& Q_output,
                                             std::map<std::string, double>* bench) const {
  const arma::uword n = m_X.n_rows;
  const arma::uword d = theta.n_elem;
  // See compute_ll_grad_theta_vecs's comment: lazily rebuild m_dX locally if
  // it was skipped at fit time (Nystrom/light Vecchia) but this exact
  // (LMP-gradient) path is reached anyway.
  arma::mat dX_local;
  const arma::mat& dX = m_dX.is_empty() ? (dX_local = LinearAlgebra::compute_dX(m_X)) : m_dX;
  auto t0 = Bench::tic();
  // With G_k = dR/dtheta_k = R % D_k (symmetric, zero diagonal), P =
  // Rinv_X_Xt_Rinv_X_inv_Xt_Rinv and M = R^-1 - P, the former
  // Wb_k = (L^-T L^-1 G_k)' - G_k P is G_k M, so
  //   ans_k = -tr(G_k M) / 2 + (M y)' G_k Q / (2 sigma2)
  //         = sum_{j < i} G_k(i, j) [-(M_ij + M_ji) / 2 + (u_i q_j + u_j q_i) / (2 sigma2)]
  // with u = M y, q = Q: one R^-1 (O(n^3)) then O(d n^2), instead of d
  // triangular solves with n right-hand sides (O(d n^3)).
  const arma::mat& P = Rinv_X_Xt_Rinv_X_inv_Xt_Rinv;
  arma::vec u;
  arma::mat S;  // M + M' = 2 R^-1 - P - P'
  {
    arma::mat Rinv_local;
    const bool has_Rinv = (m.Rinv.memptr() != nullptr) && (arma::size(m.Rinv) == arma::size(m.L));
    const arma::mat& Rinv = has_Rinv ? m.Rinv : (Rinv_local = LinearAlgebra::inv_sympd(m.L));
    t0 = Bench::toc(bench, "R^-1 = L^-T * L^-1", t0);
    u = Rinv * m_y - P * m_y;
    S = 2.0 * Rinv - P - P.t();
  }
  const double* q = Q_output.memptr();
  const double c = 1.0 / (2.0 * sigma2);
  t0 = Bench::toc(bench, "u = M y, S = M + M'", t0);
  arma::vec ans(d);
  libKriging::detail::with_dln(m_covType, theta, [&](const auto& dln) {
    libKriging::detail::sum_pairs_dln<1>(
        dX,
        n,
        theta,
        dln,
        pair_loop_threads(n),
        [&](arma::uword i, arma::uword j) {
          // R and S symmetric: column i, contiguous in j
          return std::array<double, 1>{m.R.at(j, i) * (-0.5 * S.at(j, i) + c * (u.at(i) * q[j] + u.at(j) * q[i]))};
        },
        ans.memptr());
  });
  t0 = Bench::toc(bench, "ans[k] = Sum(G_k % W) [vectorized]", t0);
  return ans;
}
