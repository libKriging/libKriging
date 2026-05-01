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

#include <tuple>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
namespace {
inline int get_optimal_threads(int max_default = 2) {
  int max_threads = omp_get_max_threads();
  if (max_threads <= 0) {
    return 1;
  }
  return (max_threads > max_default) ? max_default : max_threads;
}
}  // namespace
#endif

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
  // Invalidate cached Linv so gradient code recomputes it for the new L
  m.Linv = arma::mat();
  if (update_eligible) {
    m.L = LinearAlgebra::update_cholCov(&(m.R), m_dX, theta, _Cov, alpha, diag_norm, m_T, m_R);
  } else {
    m.L = LinearAlgebra::cholCov(&(m.R), m_dX, theta, _Cov, alpha, diag_norm);
  }
  t0 = Bench::toc(bench, "R = _Cov(dX) & L = Chol(R)", t0);

  m.Rinv = LinearAlgebra::inv_sympd(m.L);
  t0 = Bench::toc(bench, "R^-1 = L^-T * L^-1", t0);

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
  m.Rinv = arma::mat();  // computed in populate_Model
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
#ifdef _OPENMP
  arma::uword total_work = n_o * n_n;
  if (total_work >= 40000) {
    int optimal_threads = get_optimal_threads(2);
#pragma omp parallel for schedule(static) collapse(2) num_threads(optimal_threads) if (total_work >= 40000)
    for (arma::sword i = 0; i < static_cast<arma::sword>(n_o); i++) {
      for (arma::sword j = 0; j < static_cast<arma::sword>(n_n); j++) {
        arma::vec dij = Xn_o.col(i) - Xn_n.col(j);
        if (dij.is_zero(arma::datum::eps))
          R_on.at(i, j) = 1.0;
        else
          R_on.at(i, j) = _Cov(dij, m_theta) * R_on_factor;
      }
    }
  } else {
#endif
    for (arma::uword i = 0; i < n_o; i++) {
      for (arma::uword j = 0; j < n_n; j++) {
        arma::vec dij = Xn_o.col(i) - Xn_n.col(j);
        if (dij.is_zero(arma::datum::eps))
          R_on.at(i, j) = 1.0;
        else
          R_on.at(i, j) = _Cov(dij, m_theta) * R_on_factor;
      }
    }
#ifdef _OPENMP
  }
#endif
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
    ysd2_n.transform([](double val) { return (std::isnan(val) || val < 0 ? 0.0 : val); });
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
    const double h = 1.0E-5;
    // Perturbations in feature space (d × d); when jac is set we chain-rule to input space.
    arma::mat h_eye_d = h * arma::mat(d, d, arma::fill::eye);

    for (arma::uword i = 0; i < n_n; i++) {
      // dR_on / dPhi_k — (n_o × d) in feature space
      arma::mat DR_on_i = arma::mat(n_o, d, arma::fill::none);
      for (arma::uword j = 0; j < n_o; j++) {
        DR_on_i.row(j) = R_on.at(j, i) * trans(_DlnCovDx(Xn_n.col(i) - Xn_o.col(j), m_theta));
      }
      t0 = Bench::toc(nullptr, "DR_on_i    ", t0);

      // dF / dPhi_k — (d × p) in feature space
      arma::mat tXn_n_repd_i = arma::trans(Xn_n.col(i) * arma::mat(1, d, arma::fill::ones));
      arma::mat DF_n_i = (Trend::regressionModelMatrix(m_regmodel, tXn_n_repd_i + h_eye_d)
                          - Trend::regressionModelMatrix(m_regmodel, tXn_n_repd_i - h_eye_d))
                         / (2 * h);
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
        Dysd2_n.row(i) = -2 * Rstar_on.col(i).t() * W_i + 2 * Ecirc_n.row(i) * DEcirc_n_i;
        t0 = Bench::toc(nullptr, "Dysd2_n    ", t0);
      }
    }
    Dyhat_n *= m_scaleY;
    Dysd2_n *= var_scale * m_scaleY * m_scaleY;
  }

  return std::make_tuple(std::move(yhat_n),
                         std::move(arma::sqrt(ysd2_n)),
                         std::move(Sigma_n),
                         std::move(Dyhat_n),
                         std::move(Dysd2_n / (2 * arma::sqrt(ysd2_n) * arma::mat(1, d_input, arma::fill::ones))));
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
                                     bool use_qr_for_circ,
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
#ifdef _OPENMP
  arma::uword total_work = n_o * n_n;
  if (total_work >= 40000) {
    int optimal_threads = get_optimal_threads(2);
#pragma omp parallel for schedule(static) collapse(2) num_threads(optimal_threads) if (total_work >= 40000)
    for (arma::sword i = 0; i < static_cast<arma::sword>(n_o); i++) {
      for (arma::sword j = 0; j < static_cast<arma::sword>(n_n); j++) {
        arma::vec dij = Xn_o.col(i) - Xn_n.col(j);
        if (R_on_coincident_to_one && dij.is_zero(arma::datum::eps))
          R_on.at(i, j) = 1.0;
        else
          R_on.at(i, j) = _Cov(dij, m_theta) * R_on_factor;
      }
    }
  } else {
#endif
    for (arma::uword i = 0; i < n_o; i++) {
      for (arma::uword j = 0; j < n_n; j++) {
        arma::vec dij = Xn_o.col(i) - Xn_n.col(j);
        if (R_on_coincident_to_one && dij.is_zero(arma::datum::eps))
          R_on.at(i, j) = 1.0;
        else
          R_on.at(i, j) = _Cov(dij, m_theta) * R_on_factor;
      }
    }
#ifdef _OPENMP
  }
#endif
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

    lastsim_R_nn = R_nn;
    lastsim_F_n = F_n;

    lastsim_L_oCn = Rstar_on;
    lastsim_L_nCn = LinearAlgebra::safe_chol_lower(SigmaNoTrend_nKo);
    t0 = Bench::toc(nullptr, "L_nCn     ", t0);

    lastsim_L_on = arma::join_rows(arma::join_cols(m_T, lastsim_L_oCn.t()),
                                   arma::join_cols(arma::zeros(n_o, n_n), lastsim_L_nCn));

    lastsim_Rinv_on = LinearAlgebra::inv_sympd(lastsim_L_on);
    t0 = Bench::toc(nullptr, "Rinv_on     ", t0);

    lastsim_F_on = arma::join_cols(m_F, lastsim_F_n);
    lastsim_Fstar_on = LinearAlgebra::solve_lower(lastsim_L_on, lastsim_F_on);
    t0 = Bench::toc(nullptr, "Fstar_on     ", t0);

    if (use_qr_for_circ) {
      arma::mat Q_Fstar_on;
      LinearAlgebra::qr_econ(Q_Fstar_on, lastsim_circ_on, lastsim_Fstar_on);
    } else {
      arma::mat FtRinvF_on = lastsim_Fstar_on.t() * lastsim_Fstar_on;
      lastsim_circ_on = LinearAlgebra::chol_upper(FtRinvF_on);
    }
    lastsim_Fcirc_on = LinearAlgebra::rsolve_upper(lastsim_circ_on, lastsim_F_on);
    t0 = Bench::toc(nullptr, "Fcirc_on     ", t0);

    lastsim_Fhat_nKo = lastsim_L_oCn.t() * m_M;
    t0 = Bench::toc(nullptr, "Fhat_nKo     ", t0);
    lastsim_Ecirc_nKo = LinearAlgebra::rsolve_upper(m_circ, F_n - lastsim_Fhat_nKo);
    t0 = Bench::toc(nullptr, "Ecirc_nKo     ", t0);
  }

  return y_n;
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

  // ======================================================================
  // FOXY step #1 Extend the simulation to the design 'X_u' IF NECESSARY.
  // ======================================================================

  if (!use_lastsimup) {
    arma::mat R_onCu = arma::join_rows(lastsimup_R_uo, lastsimup_R_un).t();
    arma::mat Rstar_onCu = LinearAlgebra::solve_lower(lastsim_L_on, R_onCu);
    t0 = Bench::toc(nullptr, "Rstar_onCu          ", t0);

    arma::mat Ecirc_uKon = LinearAlgebra::rsolve_upper(lastsim_circ_on, F_u - Rstar_onCu.t() * lastsim_Fstar_on);
    t0 = Bench::toc(nullptr, "Ecirc_uKon          ", t0);

    arma::mat Sigma_uKon = lastsimup_R_uu - Rstar_onCu.t() * Rstar_onCu + Ecirc_uKon * Ecirc_uKon.t();
    t0 = Bench::toc(nullptr, "Sigma_uKon          ", t0);

    arma::mat LSigma_uKon = LinearAlgebra::safe_chol_lower(Sigma_uKon / Sigma_divisor);
    t0 = Bench::toc(nullptr, "LSigma_uKon          ", t0);

    arma::mat W_uCon = (R_onCu.t() + Ecirc_uKon * lastsim_Fcirc_on.t()) * lastsim_Rinv_on;
    t0 = Bench::toc(nullptr, "W_uCon          ", t0);

    arma::mat m_u = W_uCon.head_cols(n_o) * m_y;
    arma::mat M_u = arma::repmat(m_u, 1, lastsim_nsim) + W_uCon.tail_cols(n_n) * lastsim_y_n;

    Random::reset_seed(lastsim_seed);
    lastsimup_y_u = M_u + LSigma_uKon * Random::randn_mat(n_u, lastsim_nsim) * std::sqrt(m_sigma2);
    t0 = Bench::toc(nullptr, "y_u          ", t0);
  }

  // ======================================================================
  // FOXY step #2   Update the simulated paths on 'X_n'
  // ======================================================================

  if (!use_lastsimup) {
    arma::mat Rstar_ou = LinearAlgebra::solve_lower(m_T, lastsimup_R_uo.t());
    t0 = Bench::toc(nullptr, "Rstar_ou          ", t0);

    arma::mat Fhat_uKo = Rstar_ou.t() * m_M;
    arma::mat Ecirc_uKo = LinearAlgebra::rsolve_upper(m_circ, F_u - Fhat_uKo);
    t0 = Bench::toc(nullptr, "Ecirc_uKo          ", t0);

    arma::mat Rtild_uCu = lastsimup_R_uu - Rstar_ou.t() * Rstar_ou + Ecirc_uKo * Ecirc_uKo.t();
    t0 = Bench::toc(nullptr, "Rtild_uCu          ", t0);

    arma::mat Rtild_nCu = lastsimup_R_un - Rstar_ou.t() * lastsim_L_oCn + Ecirc_uKo * lastsim_Ecirc_nKo.t();
    t0 = Bench::toc(nullptr, "Rtild_nCu          ", t0);

    lastsimup_Wtild_nKu = LinearAlgebra::solve(Rtild_uCu, Rtild_nCu).t();
    t0 = Bench::toc(nullptr, "Wtild_nKu          ", t0);
  }

  return lastsim_y_n + lastsimup_Wtild_nKu * (arma::repmat(y_u, 1, lastsim_nsim) - lastsimup_y_u);
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

  m_X = arma::join_cols(m_X, Xn_u);
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

bool KrigingImpl::summary_top(std::ostringstream& oss, const arma::mat* X_display_override) const {
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
  if (!m_noise.is_empty())
    oss << "  * noise: " << m_noise.n_elem << "x[" << arma::min(m_noise) << "," << arma::max(m_noise) << "]\n";
  oss << "* trend " << Trend::toString(m_regmodel);
  oss << ((m_est_beta) ? " (est.): " : ": ");
  vec_printer(m_beta);
  oss << "\n";
  oss << "* variance";
  oss << ((m_est_sigma2) ? " (est.): " : ": ");
  oss << m_sigma2;
  oss << "\n";
  oss << "* covariance:\n";
  oss << "  * kernel: " << m_covType << "\n";
  oss << "  * range";
  oss << ((m_est_theta) ? " (est.): " : ": ");
  vec_printer(m_theta);
  oss << "\n";
  return true;
}

void KrigingImpl::summary_bottom(std::ostringstream& oss) const {
  oss << "  * fit:\n";
  oss << "    * objective: " << m_objective << "\n";
  oss << "    * optim: " << m_optim << "\n";
}

arma::mat KrigingImpl::fit_setup_impl(const arma::vec& y,
                                      const arma::mat& X,
                                      const Trend::RegressionModel& regmodel,
                                      bool normalize,
                                      bool is_beta_estim,
                                      const std::optional<arma::vec>& beta,
                                      const std::optional<arma::mat>& theta) {
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
  m_dX = LinearAlgebra::compute_dX(m_X);
  m_maxdX = arma::max(arma::abs(m_dX), 1);

  // Regression matrix
  m_regmodel = regmodel;
  m_F = Trend::regressionModelMatrix(regmodel, m_X);
  m_est_beta = is_beta_estim && (m_regmodel != Trend::RegressionModel::None);
  if (!m_est_beta && beta.has_value() && beta.value().n_elem > 0) {  // Force beta fixed (not estimated, no variance)
    m_est_beta = false;
    m_beta = beta.value();
    if (m_normalize)
      m_beta /= scaleY;
  } else
    m_est_beta = true;

  // Normalize theta0 if provided
  arma::mat theta0;
  if (theta.has_value()) {
    theta0 = theta.value();
    if ((theta0.n_cols != d) && (theta0.n_rows == d))
      theta0 = theta0.t();
    if (m_normalize)
      theta0.each_row() /= scaleX;
    if (theta0.n_cols != d)
      throw std::runtime_error("Dimension of theta should be nx" + std::to_string(d) + " instead of "
                               + std::to_string(theta0.n_rows) + "x" + std::to_string(theta0.n_cols));
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

void KrigingImpl::compute_ll_grad_theta_vecs(const arma::mat& R,
                                             const arma::mat& Rinv,
                                             const arma::mat& x,
                                             const arma::vec& theta,
                                             arma::vec& term1_vec,
                                             arma::vec& term2_vec) const {
  const arma::uword n = m_X.n_rows;
  const arma::uword d = theta.n_elem;
  for (arma::uword i = 0; i < n; i++) {
    for (arma::uword j = 0; j < i; j++) {
      arma::vec dlnCov = _DlnCovDtheta(m_dX.col(i * n + j), theta);
      double R_ij = R.at(i, j);
      double x_i = x.at(i);
      double x_j = x.at(j);
      double Rinv_ij = Rinv.at(i, j);
      for (arma::uword k = 0; k < d; k++) {
        double gradR_k_ij = R_ij * dlnCov.at(k);
        term1_vec.at(k) += 2.0 * x_i * gradR_k_ij * x_j;
        term2_vec.at(k) -= 2.0 * Rinv_ij * gradR_k_ij;
      }
    }
  }
}

arma::vec KrigingImpl::compute_lmp_theta_ans(const KModel& m,
                                             const arma::vec& theta,
                                             double sigma2,
                                             const arma::mat& Rinv_X_Xt_Rinv_X_inv_Xt_Rinv,
                                             const arma::mat& Q_output,
                                             std::map<std::string, double>* bench) const {
  const arma::uword n = m_X.n_rows;
  const arma::uword d = theta.n_elem;
  arma::vec ans(d, arma::fill::none);
  arma::mat Wb_k;
  for (arma::uword k = 0; k < d; k++) {
    auto t0 = Bench::tic();
    arma::mat gradR_k(n, n, arma::fill::zeros);
    for (arma::uword i = 0; i < n; i++) {
      for (arma::uword j = 0; j < i; j++) {
        arma::vec dlnCov = _DlnCovDtheta(m_dX.col(i * n + j), theta);
        double gradR_k_ij = m.R.at(i, j) * dlnCov.at(k);
        gradR_k.at(i, j) = gradR_k_ij;
        gradR_k.at(j, i) = gradR_k_ij;
      }
    }
    t0 = Bench::toc(bench, "gradR_k [optimized]", t0);
    Wb_k = LinearAlgebra::solve_upper(m.L.t(), LinearAlgebra::solve_lower(m.L, gradR_k)).t()
           - gradR_k * Rinv_X_Xt_Rinv_X_inv_Xt_Rinv;
    t0 = Bench::toc(bench, "Wb_k = gradR_k \\ L \\ Tt - gradR_k * RiFFtRiFiFtRi", t0);
    ans[k] = -sum(Wb_k.diag()) / 2.0 + as_scalar(trans(m_y) * trans(Wb_k) * Q_output) / (2.0 * sigma2);
    t0 = Bench::toc(bench, "ans[k] = Sum(diag(Wb_k)) + yt * Wb_kt * Qo / S2...", t0);
  }
  return ans;
}
