/**
 * @file MLPKriging.cpp
 * @brief Kriging with a joint MLP feature map (Deep Kernel Learning).
 * See MLPKriging.hpp for documentation.
 */

#include "libKriging/MLPKriging.hpp"
#include "libKriging/AdamBFGS.hpp"
#include "libKriging/Covariance.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/Optim.hpp"
#include "libKriging/Random.hpp"

#include "libKriging/utils/jsonutils.hpp"
#include "libKriging/utils/nlohmann/json.hpp"
#include "libKriging/utils/utils.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace libKriging {

// -------------------------------------------------------------------------
//  parse_kernel / make_Cov  (identical to WarpKriging)
// -------------------------------------------------------------------------
WarpBaseKernel MLPKriging::parse_kernel(const std::string& name) {
  std::string s = name;
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
  if (s == "gauss" || s == "rbf")
    return WarpBaseKernel::Gauss;
  if (s == "matern3_2" || s == "matern32")
    return WarpBaseKernel::Matern32;
  if (s == "matern5_2" || s == "matern52")
    return WarpBaseKernel::Matern52;
  if (s == "exp" || s == "exponential")
    return WarpBaseKernel::Exp;
  throw std::invalid_argument("Unknown kernel: " + name);
}

void MLPKriging::make_Cov(const std::string& kernel) {
  std::string s = kernel;
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
  if (s == "rbf")
    s = "gauss";
  else if (s == "exponential")
    s = "exp";
  else if (s == "matern32")
    s = "matern3_2";
  else if (s == "matern52")
    s = "matern5_2";

  auto cov = Covariance::resolve(s);
  _Cov = std::move(cov.Cov);
  _DlnCovDtheta = std::move(cov.DlnCovDtheta);
  _DlnCovDx = std::move(cov.DlnCovDx);
}

// -------------------------------------------------------------------------
//  Constructors
// -------------------------------------------------------------------------
MLPKriging::MLPKriging(const std::vector<arma::uword>& hidden_dims,
                       arma::uword d_out,
                       const std::string& activation,
                       const std::string& kernel)
    : m_hidden_dims(hidden_dims), m_d_out(d_out), m_activation(activation) {
  if (m_hidden_dims.empty())
    m_hidden_dims = {32, 16};
  if (m_d_out == 0)
    throw std::invalid_argument("MLPKriging: d_out must be >= 1");
  m_kernel_name = kernel;
  m_base_kernel = parse_kernel(kernel);
  make_Cov(kernel);
}

MLPKriging::MLPKriging(const arma::vec& y,
                       const arma::mat& X,
                       const std::vector<arma::uword>& hidden_dims,
                       arma::uword d_out,
                       const std::string& activation,
                       const std::string& kernel,
                       const std::string& regmodel,
                       bool normalize,
                       const std::string& optim,
                       const std::string& objective,
                       const std::map<std::string, std::string>& parameters)
    : MLPKriging(hidden_dims, d_out, activation, kernel) {
  fit(y, X, regmodel, normalize, optim, objective, parameters);
}

// -------------------------------------------------------------------------
//  ensure_joint_warp  — instantiate WarpMLPJoint now that d_in is known
// -------------------------------------------------------------------------
void MLPKriging::ensure_joint_warp(arma::uword d_in) {
  if (!m_joint_warp) {
    m_joint_warp = std::make_unique<WarpMLPJoint>(d_in, m_hidden_dims, m_d_out, WarpMLP::parse_act(m_activation), 42);
  }
}

// -------------------------------------------------------------------------
//  apply_warping:  X → Φ
// -------------------------------------------------------------------------
arma::mat MLPKriging::apply_warping(const arma::mat& X) const {
  return m_joint_warp->forward(X);  // (n × d_in) → (n × d_out)
}

// -------------------------------------------------------------------------
//  Trend matrix  (identical structure to WarpKriging)
// -------------------------------------------------------------------------
arma::mat MLPKriging::build_trend_matrix(const arma::mat& X) const {
  const arma::uword n = X.n_rows;
  if (m_regmodel == "constant")
    return arma::ones<arma::mat>(n, 1);

  arma::mat Phi = apply_warping(X);
  arma::uword d = Phi.n_cols;

  if (m_regmodel == "linear") {
    arma::mat F(n, 1 + d);
    F.col(0) = arma::ones<arma::vec>(n);
    F.cols(1, d) = Phi;
    return F;
  }
  if (m_regmodel == "quadratic") {
    arma::uword p = 1 + d + d * (d + 1) / 2;
    arma::mat F(n, p);
    F.col(0) = arma::ones<arma::vec>(n);
    arma::uword c = 1;
    for (arma::uword j = 0; j < d; ++j)
      F.col(c++) = Phi.col(j);
    for (arma::uword j = 0; j < d; ++j)
      for (arma::uword k = j; k < d; ++k)
        F.col(c++) = Phi.col(j) % Phi.col(k);
    return F;
  }
  throw std::invalid_argument("Unknown regmodel: " + m_regmodel);
}

// -------------------------------------------------------------------------
//  Data normalisation  (all inputs are continuous → normalise all columns)
// -------------------------------------------------------------------------
void MLPKriging::normalise_data() {
  arma::uword d = m_X.n_cols;
  m_centerX = arma::zeros<arma::rowvec>(d);
  m_scaleX = arma::ones<arma::rowvec>(d);

  if (m_normalize) {
    for (arma::uword j = 0; j < d; ++j) {
      m_centerX(j) = arma::mean(m_X.col(j));
      m_scaleX(j) = arma::stddev(m_X.col(j));
      if (m_scaleX(j) < 1e-12)
        m_scaleX(j) = 1.0;
      m_X.col(j) = (m_X.col(j) - m_centerX(j)) / m_scaleX(j);
    }
    m_centerY = arma::mean(m_y);
    m_scaleY = arma::stddev(m_y);
    if (m_scaleY < 1e-12)
      m_scaleY = 1.0;
    m_y = (m_y - m_centerY) / m_scaleY;
  } else {
    m_centerY = 0.0;
    m_scaleY = 1.0;
  }
}

// -------------------------------------------------------------------------
//  compute_dPhi  (identical)
// -------------------------------------------------------------------------
void MLPKriging::compute_dPhi() {
  const arma::uword n = m_Phi.n_rows;
  const arma::uword d = m_Phi.n_cols;
  m_dPhi.set_size(d, n * n);
  m_dPhi.zeros();

  const double* Phi_mem = m_Phi.memptr();
  double* dPhi_mem = m_dPhi.memptr();

  for (arma::uword i = 0; i < n; ++i) {
    for (arma::uword j = i + 1; j < n; ++j) {
      arma::uword ij = i * n + j;
      arma::uword ji = j * n + i;
      for (arma::uword k = 0; k < d; ++k) {
        double diff = Phi_mem[i + k * n] - Phi_mem[j + k * n];
        dPhi_mem[k + ij * d] = diff;
        dPhi_mem[k + ji * d] = diff;
      }
    }
  }
}

arma::mat MLPKriging::build_Rcross(const arma::mat& Phi_new, const arma::mat& Phi_train) const {
  const arma::uword m = Phi_new.n_rows;
  const arma::uword n = Phi_train.n_rows;
  arma::mat Rc(m, n);
  LinearAlgebra::covMat_rect(&Rc, Phi_new.t(), Phi_train.t(), m_theta, _Cov, 1.0);
  return Rc;
}

// -------------------------------------------------------------------------
//  refresh_cache
// -------------------------------------------------------------------------
void MLPKriging::refresh_cache() {
  m_Phi = apply_warping(m_X);
  compute_dPhi();
  refresh_cache_theta_only();
}

void MLPKriging::refresh_cache_theta_only() {
  const arma::uword n = m_y.n_elem;

  m_R.set_size(n, n);
  arma::vec diag_with_nugget(n, arma::fill::value(1.0 + 1e-8));
  m_T = LinearAlgebra::cholCov(&m_R, m_dPhi, m_theta, _Cov, 1, diag_with_nugget);
  m_Rinv = LinearAlgebra::inv_sympd(m_T);
  m_logdet = 2.0 * arma::sum(arma::log(m_T.diag()));

  m_F = build_trend_matrix(m_X);
  m_M = LinearAlgebra::solve_lower(m_T, m_F);              // Fstar = C⁻¹ F
  arma::vec ystar = LinearAlgebra::solve_lower(m_T, m_y);   // C⁻¹ y

  arma::mat FtRinvF = m_M.t() * m_M;
  m_circ = LinearAlgebra::chol_upper(FtRinvF);              // chol(F'R⁻¹F)

  arma::vec FtRinvy = m_M.t() * ystar;
  m_beta = LinearAlgebra::solve_upper(m_circ, LinearAlgebra::solve_lower(m_circ.t(), FtRinvy));

  arma::vec residual = m_y - m_F * m_beta;
  m_z = LinearAlgebra::solve_lower(m_T, residual);
  m_sigma2 = std::max(arma::dot(m_z, m_z) / n, 1e-20);
}

// -------------------------------------------------------------------------
//  Concentrated LL
// -------------------------------------------------------------------------
double MLPKriging::concentrated_ll() const {
  const double n = static_cast<double>(m_y.n_elem);
  return -0.5 * n * (1.0 + std::log(2.0 * arma::datum::pi) + std::log(m_sigma2)) - 0.5 * m_logdet;
}

double MLPKriging::logLikelihood() const {
  if (!m_fitted)
    throw std::runtime_error("MLPKriging: model not fitted");
  return concentrated_ll();
}

std::tuple<double, arma::vec, arma::mat> MLPKriging::logLikelihoodFun(const arma::vec& theta_gp,
                                                                      bool withGrad,
                                                                      bool /*withHess*/) const {
  auto* self = const_cast<MLPKriging*>(this);
  arma::vec old_theta = m_theta;
  self->m_theta = theta_gp;
  self->refresh_cache();

  double ll = concentrated_ll();
  arma::vec grad;
  if (withGrad) {
    auto [ll2, g] = concentrated_ll_and_grad_theta();
    grad = g;
    (void)ll2;
  }

  self->m_theta = old_theta;
  self->refresh_cache();
  return {ll, grad, arma::mat()};
}

// -------------------------------------------------------------------------
//  ∂R/∂θ_k
// -------------------------------------------------------------------------
arma::mat MLPKriging::build_dR_dtheta_k(arma::uword k) const {
  const arma::uword n = m_Phi.n_rows;
  arma::mat dR(n, n, arma::fill::zeros);

  for (arma::uword i = 0; i < n; ++i) {
    for (arma::uword j = i + 1; j < n; ++j) {
      arma::vec dlnCov = _DlnCovDtheta(m_dPhi.col(i * n + j), m_theta);
      double dR_ij = m_R(i, j) * dlnCov(k);
      dR(i, j) = dR_ij;
      dR(j, i) = dR_ij;
    }
  }
  return dR;
}

// -------------------------------------------------------------------------
//  Concentrated LL + analytical gradient in θ-space
// -------------------------------------------------------------------------
std::pair<double, arma::vec> MLPKriging::concentrated_ll_and_grad_theta() const {
  double ll = concentrated_ll();

  const arma::uword n = m_y.n_elem;
  const arma::uword d = m_theta.n_elem;

  arma::vec alpha = LinearAlgebra::solve_upper(m_T.t(), m_z);  // R⁻¹(y - Fβ)
  const arma::mat& Rinv = m_Rinv;  // Use cached
  arma::mat dLL_dR = 0.5 * (alpha * alpha.t() / m_sigma2 - Rinv);

  arma::vec grad_theta(d, arma::fill::zeros);
  for (arma::uword i = 0; i < n; ++i) {
    for (arma::uword j = i + 1; j < n; ++j) {
      arma::vec dlnCov = _DlnCovDtheta(m_dPhi.col(i * n + j), m_theta);
      double R_ij = m_R(i, j);
      double w = 2.0 * dLL_dR(i, j);
      for (arma::uword k = 0; k < d; ++k) {
        grad_theta(k) += w * R_ij * dlnCov(k);
      }
    }
  }
  return {ll, grad_theta};
}

// -------------------------------------------------------------------------
//  Warp gradient (backprop through the joint MLP)
// -------------------------------------------------------------------------
arma::mat MLPKriging::dK_dPhi(const arma::mat& dL_dK) const {
  const arma::uword n = m_Phi.n_rows;
  const arma::uword d = m_Phi.n_cols;
  arma::mat dL_dPhi(n, d, arma::fill::zeros);

  for (arma::uword i = 0; i < n; ++i) {
    for (arma::uword j = 0; j < n; ++j) {
      if (i == j)
        continue;
      double coeff = dL_dK(i, j);
      if (std::abs(coeff) < 1e-15)
        continue;

      double K_ij = m_sigma2 * m_R(i, j);
      arma::vec dlnCdx = _DlnCovDx(m_dPhi.col(i * n + j), m_theta);
      dL_dPhi.row(i) += coeff * K_ij * dlnCdx.t();
    }
  }
  return dL_dPhi;
}

arma::vec MLPKriging::warp_gradient() const {
  arma::mat Kinv = (1.0 / m_sigma2) * m_Rinv;  // Use cached
  arma::vec alpha = LinearAlgebra::solve_upper(m_T.t(), m_z);
  arma::mat dLL_dK = 0.5 * (alpha * alpha.t() - Kinv);
  arma::mat dLL_dPhi = dK_dPhi(dLL_dK);

  arma::uword n_warp = total_warp_params();
  arma::vec grad(n_warp, arma::fill::zeros);
  if (m_joint_warp && n_warp > 0) {
    arma::vec gw = m_joint_warp->backward(m_X, dLL_dPhi);
    if (gw.n_elem > 0)
      grad.head(gw.n_elem) = gw;
  }
  return grad;
}

// -------------------------------------------------------------------------
//  Warp param pack/unpack
// -------------------------------------------------------------------------
arma::uword MLPKriging::total_warp_params() const {
  return m_joint_warp ? m_joint_warp->n_params() : 0;
}

arma::vec MLPKriging::pack_warp_params() const {
  if (!m_joint_warp || m_joint_warp->n_params() == 0)
    return {};
  return m_joint_warp->get_params();
}

void MLPKriging::unpack_warp_params(const arma::vec& wp) {
  if (m_joint_warp && wp.n_elem > 0)
    m_joint_warp->set_params(wp);
}

// -------------------------------------------------------------------------
//  clone_for_thread
// -------------------------------------------------------------------------
MLPKriging MLPKriging::clone_for_thread() const {
  MLPKriging c(m_hidden_dims, m_d_out, m_activation, m_kernel_name);
  c.m_y = m_y;
  c.m_X = m_X;
  c.m_F = m_F;
  c.m_normalize = m_normalize;
  c.m_centerX = m_centerX;
  c.m_scaleX = m_scaleX;
  c.m_centerY = m_centerY;
  c.m_scaleY = m_scaleY;
  c.m_regmodel = m_regmodel;
  c.m_max_iter_bfgs = m_max_iter_bfgs;
  c.m_max_iter_adam = m_max_iter_adam;
  c.m_adam_lr = m_adam_lr;
  c.m_fitted = true;

  c.m_theta = m_theta;
  c.m_sigma2 = m_sigma2;
  c.m_beta = m_beta;
  c.m_z = m_z;
  c.m_M = m_M;
  c.m_circ = m_circ;
  c.m_R = m_R;
  c.m_T = m_T;
  c.m_Rinv = m_Rinv;
  c.m_logdet = m_logdet;
  c.m_Phi = m_Phi;
  c.m_dPhi = m_dPhi;

  if (m_joint_warp)
    c.m_joint_warp = m_joint_warp->clone();

  return c;
}

// -------------------------------------------------------------------------
//  Joint optimisation (bi-level or joint L-BFGS-B) with multistart
// -------------------------------------------------------------------------
void MLPKriging::optimise_joint(const std::string& method) {
  arma::uword n_warp = total_warp_params();
  arma::uword d_theta = m_theta.n_elem;

  arma::vec maxdPhi = arma::max(arma::abs(m_dPhi), 1);
  auto theta_bounds_pair = Optim::theta_bounds(maxdPhi, m_dPhi, m_y, m_y.n_elem);
  arma::vec theta_lower = theta_bounds_pair.first;
  arma::vec theta_upper = theta_bounds_pair.second;
  theta_lower = arma::clamp(theta_lower, 1e-10, arma::datum::inf);
  theta_upper = arma::max(theta_lower * 2.0, theta_upper);

  auto to_gamma = [](const arma::vec& t) { return Optim::reparametrize ? Optim::reparam_to(t) : t; };
  auto from_gamma = [](const arma::vec& g) { return Optim::reparametrize ? Optim::reparam_from(g) : g; };
  auto grad_theta_to_gamma = [](const arma::vec& theta, const arma::vec& g_theta) {
    return Optim::reparametrize ? Optim::reparam_from_deriv(theta, g_theta) : g_theta;
  };

  arma::vec gamma_lower = to_gamma(theta_lower);
  arma::vec gamma_upper = to_gamma(theta_upper);

  auto parsed = Optim::parse_method(method, "BFGS");
  const std::string base_method = parsed.first;
  const int multistart = parsed.second;

  auto run_joint_bfgs = [&](MLPKriging& wk) -> double {
    arma::uword nw = wk.total_warp_params();
    arma::uword dt = wk.m_theta.n_elem;
    arma::uword n_total = nw + dt;
    lbfgsb::Optimizer optimizer{static_cast<unsigned int>(n_total)};
    optimizer.iprint = -1;
    optimizer.max_iter = Optim::max_iteration;
    optimizer.pgtol = Optim::gradient_tolerance;
    optimizer.factr = Optim::objective_rel_tolerance / 1E-13;

    arma::vec theta0 = wk.m_theta;
    arma::vec wp_init = (nw > 0) ? wk.pack_warp_params() : arma::vec();

    arma::vec x0(n_total);
    if (nw > 0)
      x0.head(nw) = wp_init;
    x0.tail(dt) = to_gamma(theta0);

    arma::vec lb(n_total), ub(n_total);
    arma::ivec btype(n_total);
    if (nw > 0) {
      lb.head(nw).fill(-1e20);
      ub.head(nw).fill(1e20);
      btype.head(nw).fill(0);
    }
    lb.tail(dt) = gamma_lower;
    ub.tail(dt) = gamma_upper;
    btype.tail(dt).fill(2);

    auto obj_fn = [&](const arma::vec& x, arma::vec& grad) -> double {
      if (nw > 0)
        wk.unpack_warp_params(x.head(nw));
      wk.m_theta = from_gamma(x.tail(dt));
      wk.refresh_cache();

      auto [ll, g_theta] = wk.concentrated_ll_and_grad_theta();
      grad.tail(dt) = -grad_theta_to_gamma(wk.m_theta, g_theta);
      if (nw > 0)
        grad.head(nw) = -wk.warp_gradient();
      return -ll;
    };

    int retry = 0;
    double best_f = std::numeric_limits<double>::infinity();
    arma::vec best_x = x0;
    arma::vec x = x0;
    while (retry <= Optim::max_restart) {
      auto res = optimizer.minimize(obj_fn, x, lb.memptr(), ub.memptr(), btype.memptr());
      if (res.f_opt < best_f) {
        best_f = res.f_opt;
        best_x = x;
      }
      arma::vec theta_cur = from_gamma(x.tail(dt));
      double sol_to_lb = arma::min(arma::abs(theta_cur - theta_lower));
      if ((retry < Optim::max_restart)
          && ((res.task.rfind("ABNORMAL_TERMINATION_IN_LNSRCH", 0) == 0) || (res.num_iters <= 2)
              || (sol_to_lb < arma::datum::eps) || (res.f_opt > best_f))) {
        arma::vec theta_restart = (theta0 + theta_lower) / std::pow(2.0, retry + 1);
        x.tail(dt) = to_gamma(theta_restart);
        if (nw > 0)
          x.head(nw) = wp_init;
        retry++;
      } else {
        break;
      }
    }

    if (nw > 0)
      wk.unpack_warp_params(best_x.head(nw));
    wk.m_theta = from_gamma(best_x.tail(dt));
    wk.refresh_cache();
    return best_f;
  };

  auto run_adam_bfgs = [&](MLPKriging& wk) -> double {
    arma::uword nw = wk.total_warp_params();
    arma::uword dt = wk.m_theta.n_elem;
    AdamBFGS opt(nw, dt);
    opt.max_iter_adam = wk.m_max_iter_adam;
    opt.adam_lr = wk.m_adam_lr;
    opt.max_iter_bfgs = Optim::max_iteration;
    opt.bfgs_pgtol = Optim::gradient_tolerance;
    opt.bfgs_factr = Optim::objective_rel_tolerance / 1E-13;
    opt.maximize = true;

    arma::vec current_wp = wk.pack_warp_params();
    arma::vec current_gamma = to_gamma(wk.m_theta);

    auto obj_fn = [&wk, &current_wp, &from_gamma, &grad_theta_to_gamma](const arma::vec& x_outer,
                                                                        const arma::vec& x_inner,
                                                                        arma::vec* grad_outer,
                                                                        arma::vec* grad_inner) -> double {
      bool warp_changed = false;
      if (x_outer.n_elem > 0) {
        if (current_wp.n_elem != x_outer.n_elem || arma::any(current_wp != x_outer)) {
          wk.unpack_warp_params(x_outer);
          current_wp = x_outer;
          warp_changed = true;
        }
      }

      wk.m_theta = from_gamma(x_inner);

      if (warp_changed)
        wk.refresh_cache();
      else
        wk.refresh_cache_theta_only();

      double ll = wk.concentrated_ll();

      if (grad_inner) {
        auto [ll2, g_theta] = wk.concentrated_ll_and_grad_theta();
        *grad_inner = grad_theta_to_gamma(wk.m_theta, g_theta);
        (void)ll2;
      }
      if (grad_outer && x_outer.n_elem > 0) {
        *grad_outer = wk.warp_gradient();
      }
      return ll;
    };

    auto result = opt.optimize(current_wp, current_gamma, gamma_lower, gamma_upper, obj_fn);

    if (nw > 0)
      wk.unpack_warp_params(result.x_outer);
    wk.m_theta = from_gamma(result.x_inner);
    wk.refresh_cache();
    return -wk.concentrated_ll();
  };

  struct OptimizationResult {
    double neg_ll = std::numeric_limits<double>::infinity();
    arma::vec theta;
    arma::vec warp_params;
    double sigma2 = 0;
    arma::vec beta;
    arma::vec z;
    arma::mat M;
    arma::mat circ;
    arma::mat C;
    arma::mat Rinv;
    arma::mat R;
    arma::mat Phi;
    arma::mat dPhi;
    double logdet = 0;
    bool success = false;
  };

  auto extract_result = [](MLPKriging& wk, double neg_ll) -> OptimizationResult {
    OptimizationResult r;
    r.neg_ll = neg_ll;
    r.theta = wk.m_theta;
    r.warp_params = wk.pack_warp_params();
    r.sigma2 = wk.m_sigma2;
    r.beta = wk.m_beta;
    r.z = wk.m_z;
    r.M = wk.m_M;
    r.circ = wk.m_circ;
    r.C = wk.m_T;
    r.Rinv = wk.m_Rinv;
    r.R = wk.m_R;
    r.Phi = wk.m_Phi;
    r.dPhi = wk.m_dPhi;
    r.logdet = wk.m_logdet;
    r.success = true;
    return r;
  };

  auto restore_best = [&](const OptimizationResult& r) {
    if (n_warp > 0)
      unpack_warp_params(r.warp_params);
    m_theta = r.theta;
    m_sigma2 = r.sigma2;
    m_beta = r.beta;
    m_z = r.z;
    m_M = r.M;
    m_circ = r.circ;
    m_T = r.C;
    m_Rinv = r.Rinv;
    m_R = r.R;
    m_Phi = r.Phi;
    m_dPhi = r.dPhi;
    m_logdet = r.logdet;
  };

  auto run_parallel_multistart = [&](auto& optimizer_fn) {
    arma::mat theta0_rand(multistart, d_theta);
    for (int i = 0; i < multistart; ++i)
      theta0_rand.row(i) = arma::trans(theta_lower + arma::randu<arma::vec>(d_theta) % (theta_upper - theta_lower));
    theta0_rand.row(0) = m_theta.t();

    arma::vec wp_init = pack_warp_params();

    std::vector<OptimizationResult> results(multistart);
    std::mutex results_mutex;

    std::vector<MLPKriging> clones;
    clones.reserve(multistart);
    for (int i = 0; i < multistart; ++i)
      clones.push_back(clone_for_thread());

    auto worker = [&](int task_id) {
      try {
        MLPKriging& wk = clones[task_id];
        if (n_warp > 0)
          wk.unpack_warp_params(wp_init);
        wk.m_theta = theta0_rand.row(task_id).t();
        wk.refresh_cache();

        double neg_ll = optimizer_fn(wk);

        OptimizationResult r = extract_result(wk, neg_ll);
        std::lock_guard<std::mutex> lock(results_mutex);
        results[task_id] = std::move(r);
      } catch (const std::exception& e) {
        if (Optim::log_level > Optim::log_none) {
          std::lock_guard<std::mutex> lock(results_mutex);
          arma::cout << "Warning: MLPKriging multistart " << (task_id + 1) << " failed: " << e.what() << arma::endl;
        }
      }
    };

    if (multistart == 1) {
      worker(0);
    } else {
      unsigned int n_cpu = std::thread::hardware_concurrency();
      int pool_size = Optim::thread_pool_size;
      if (pool_size <= 0)
        pool_size = std::max(1u, n_cpu);
      pool_size = std::min(pool_size, multistart);

      if (Optim::log_level > Optim::log_none) {
        arma::cout << "MLPKriging thread pool: " << pool_size << " workers (ncpu=" << n_cpu
                   << ", multistart=" << multistart << ")" << arma::endl;
      }

      std::atomic<int> next_task(0);
      std::vector<std::thread> threads;
      threads.reserve(pool_size);

      struct ThreadJoiner {
        std::vector<std::thread>& threads_ref;
        explicit ThreadJoiner(std::vector<std::thread>& t) : threads_ref(t) {}
        ~ThreadJoiner() {
          for (auto& t : threads_ref)
            if (t.joinable())
              t.join();
        }
      };

      for (int worker_id = 0; worker_id < pool_size; worker_id++) {
        threads.emplace_back([&]() {
          while (true) {
            int task_id = next_task.fetch_add(1);
            if (task_id >= multistart)
              break;

            int delay_ms = task_id * Optim::thread_start_delay_ms;
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));

            worker(task_id);
          }
        });
      }

      ThreadJoiner joiner(threads);
    }

    int best_idx = -1;
    double best_neg_ll = std::numeric_limits<double>::infinity();
    for (int i = 0; i < multistart; ++i) {
      if (results[i].success && results[i].neg_ll < best_neg_ll) {
        best_neg_ll = results[i].neg_ll;
        best_idx = i;
      }
    }
    if (best_idx >= 0)
      restore_best(results[best_idx]);
  };

  if (base_method == "BFGS" || n_warp == 0) {
    if (multistart <= 1) {
      run_joint_bfgs(*this);
    } else {
      run_parallel_multistart(run_joint_bfgs);
    }
  } else {
    if (multistart <= 1) {
      run_adam_bfgs(*this);
    } else {
      run_parallel_multistart(run_adam_bfgs);
    }

    if (base_method == "BFGS+Adam+BFGS" && n_warp > 0) {
      run_joint_bfgs(*this);
    }
  }
}

// -------------------------------------------------------------------------
//  fit() — map-based overload
// -------------------------------------------------------------------------
void MLPKriging::fit(const arma::vec& y,
                     const arma::mat& X,
                     const std::string& regmodel,
                     bool normalize,
                     const std::string& optim,
                     const std::string& objective,
                     const std::map<std::string, std::string>& parameters) {
  for (const auto& [key, val] : parameters) {
    if (key == "adam_lr")
      m_adam_lr = std::stod(val);
    if (key == "max_iter_adam")
      m_max_iter_adam = std::stoul(val);
    if (key == "max_iter_bfgs")
      m_max_iter_bfgs = std::stoul(val);
  }
  fit(y, X, regmodel, normalize, optim, objective, Parameters{});
}

// -------------------------------------------------------------------------
//  fit() — typed-parameters overload
// -------------------------------------------------------------------------
void MLPKriging::fit(const arma::vec& y,
                     const arma::mat& X,
                     const std::string& regmodel,
                     bool normalize,
                     const std::string& optim,
                     const std::string& /*objective*/,
                     const Parameters& parameters) {
  if (y.n_elem != X.n_rows)
    throw std::invalid_argument("fit: y/X size mismatch");

  m_y = y;
  m_X = X;
  m_regmodel = regmodel;
  m_normalize = normalize;

  normalise_data();

  // Instantiate joint warp now that d_in is known
  ensure_joint_warp(X.n_cols);

  if (parameters.theta.has_value()) {
    const arma::vec& t0 = *parameters.theta;
    if (t0.n_elem != m_d_out)
      throw std::invalid_argument("fit: parameters.theta has " + std::to_string(t0.n_elem) + " elements but d_out is "
                                  + std::to_string(m_d_out));
    m_theta = t0;
  } else {
    m_theta = arma::ones<arma::vec>(m_d_out);
  }

  if (parameters.warp_params.has_value() && parameters.warp_params->n_elem > 0) {
    const arma::vec& wp = *parameters.warp_params;
    arma::uword nw = total_warp_params();
    if (wp.n_elem != nw)
      throw std::invalid_argument("fit: parameters.warp_params has " + std::to_string(wp.n_elem)
                                  + " elements but total_warp_params is " + std::to_string(nw));
    unpack_warp_params(wp);
  }

  refresh_cache();
  optimise_joint(optim);
  m_fitted = true;
}

// -------------------------------------------------------------------------
//  predict()
// -------------------------------------------------------------------------
std::tuple<arma::vec, arma::vec, arma::mat, arma::mat, arma::mat> MLPKriging::predict(const arma::mat& x_new,
                                                                                      bool withStd,
                                                                                      bool withCov,
                                                                                      bool withDeriv) const {
  if (!m_fitted)
    throw std::runtime_error("predict: model not fitted");

  const arma::uword d = x_new.n_cols;
  arma::mat x_n = x_new;
  if (m_normalize) {
    for (arma::uword j = 0; j < x_n.n_cols; ++j)
      x_n.col(j) = (x_n.col(j) - m_centerX(j)) / m_scaleX(j);
  }

  const arma::uword n_n = x_n.n_rows;
  const arma::uword n_o = m_Phi.n_rows;
  arma::mat Phi_new = apply_warping(x_n);
  arma::mat F_new = build_trend_matrix(x_n);
  arma::mat Rcross = build_Rcross(Phi_new, m_Phi);

  arma::mat Rstar_on = LinearAlgebra::solve_lower(m_T, Rcross.t());
  arma::vec mean = F_new * m_beta + Rstar_on.t() * m_z;
  mean = mean * m_scaleY + m_centerY;

  arma::vec stdev;
  arma::mat cov;
  arma::mat Dyhat_n;
  arma::mat Dystdev_n;

  arma::mat v;
  arma::vec ysd2_n;

  if (withStd || withCov || withDeriv) {
    v = Rstar_on;

    if (withCov) {
      arma::mat R_new(n_n, n_n);
      LinearAlgebra::covMat_sym_X(&R_new, Phi_new.t(), m_theta, _Cov);
      cov = R_new - v.t() * v;
      arma::mat H = F_new.t() - m_M.t() * v;
      arma::mat Hcirc = LinearAlgebra::rsolve_upper(m_circ, H.t());
      cov += Hcirc * Hcirc.t();
      cov *= (m_sigma2 * m_scaleY * m_scaleY);
      cov = 0.5 * (cov + cov.t());
      cov.diag() = arma::clamp(cov.diag(), 0.0, arma::datum::inf);
      stdev = arma::sqrt(cov.diag());
      ysd2_n = cov.diag();
    } else {
      ysd2_n.set_size(n_n);
      for (arma::uword i = 0; i < n_n; ++i) {
        ysd2_n(i) = std::max(0.0, 1.0 - arma::dot(v.col(i), v.col(i)));
        arma::vec r_i = F_new.row(i).t() - m_M.t() * v.col(i);
        arma::mat r_circ = LinearAlgebra::solve_lower(m_circ.t(), r_i);
        ysd2_n(i) += arma::dot(r_circ, r_circ);
      }
      ysd2_n *= (m_sigma2 * m_scaleY * m_scaleY);
      ysd2_n = arma::clamp(ysd2_n, 0.0, arma::datum::inf);
      if (withStd)
        stdev = arma::sqrt(ysd2_n);
    }
  }

  if (withDeriv) {
    const double h = 1.0E-5;
    const double sigma2 = m_sigma2;

    Dyhat_n.set_size(n_n, d);
    Dyhat_n.zeros();
    arma::mat Dysd2_n(n_n, d, arma::fill::zeros);

    arma::mat Ecirc_n;
    if (withStd) {
      arma::mat Fhat_n = v.t() * m_M;
      arma::mat E_n = F_new - Fhat_n;
      Ecirc_n = LinearAlgebra::rsolve_upper(m_circ, E_n);
    }

    for (arma::uword i = 0; i < n_n; i++) {
      arma::mat x_perturbed(2 * d, d);
      for (arma::uword k = 0; k < d; k++) {
        x_perturbed.row(k) = x_n.row(i);
        x_perturbed.row(d + k) = x_n.row(i);
        x_perturbed(k, k) += h;
        x_perturbed(d + k, k) -= h;
      }

      arma::mat Phi_perturbed = apply_warping(x_perturbed);
      arma::mat J_warp(m_d_out, d);
      for (arma::uword k = 0; k < d; k++) {
        J_warp.col(k) = (Phi_perturbed.row(k) - Phi_perturbed.row(d + k)).t() / (2.0 * h);
      }

      arma::mat F_perturbed = build_trend_matrix(x_perturbed);
      arma::mat DF_n_i(d, F_new.n_cols);
      for (arma::uword k = 0; k < d; k++) {
        DF_n_i.row(k) = (F_perturbed.row(k) - F_perturbed.row(d + k)) / (2.0 * h);
      }

      arma::mat DR_on_i(n_o, d);
      for (arma::uword j = 0; j < n_o; j++) {
        arma::vec dPhi = Phi_new.row(i).t() - m_Phi.row(j).t();
        arma::vec dlnCovDx = _DlnCovDx(dPhi, m_theta);
        DR_on_i.row(j) = Rcross(i, j) * (dlnCovDx.t() * J_warp);
      }

      arma::mat W_i = LinearAlgebra::solve_lower(m_T, DR_on_i);

      arma::vec alpha = LinearAlgebra::solve_upper(m_T.t(), m_z);
      Dyhat_n.row(i) = (DF_n_i * m_beta + DR_on_i.t() * alpha).t();

      if (withStd) {
        arma::mat DEcirc_n_i = LinearAlgebra::solve_lower(m_circ.t(), (DF_n_i - W_i.t() * m_M).t());
        Dysd2_n.row(i) = -2.0 * v.col(i).t() * W_i + 2.0 * Ecirc_n.row(i) * DEcirc_n_i;
      }
    }
    Dyhat_n *= m_scaleY;
    Dysd2_n *= sigma2 * m_scaleY * m_scaleY;

    if (withStd) {
      Dystdev_n.set_size(n_n, d);
      for (arma::uword i = 0; i < n_n; i++) {
        double sd_i = std::sqrt(ysd2_n(i));
        if (sd_i > 0)
          Dystdev_n.row(i) = Dysd2_n.row(i) / (2.0 * sd_i);
        else
          Dystdev_n.row(i).zeros();
      }
    }
  }

  return {mean, stdev, cov, Dyhat_n, Dystdev_n};
}

// -------------------------------------------------------------------------
//  simulate()
// -------------------------------------------------------------------------
arma::mat MLPKriging::simulate(int nsim, uint64_t seed, const arma::mat& x_new) const {
  if (!m_fitted)
    throw std::runtime_error("simulate: model not fitted");

  const arma::uword n_n = x_new.n_rows;

  arma::mat x_n = x_new;
  if (m_normalize) {
    for (arma::uword j = 0; j < x_n.n_cols; ++j)
      x_n.col(j) = (x_n.col(j) - m_centerX(j)) / m_scaleX(j);
  }

  arma::mat Phi_new = apply_warping(x_n);
  arma::mat F_new = build_trend_matrix(x_n);

  arma::mat R_nn(n_n, n_n);
  LinearAlgebra::covMat_sym_X(&R_nn, Phi_new.t(), m_theta, _Cov);

  arma::mat Rcross = build_Rcross(Phi_new, m_Phi);

  arma::mat v = LinearAlgebra::solve_lower(m_T, Rcross.t());

  arma::vec yhat_n = F_new * m_beta + v.t() * m_z;

  arma::mat Ecirc_n = LinearAlgebra::rsolve_upper(m_circ, F_new - v.t() * m_M);

  arma::mat Sigma_nKo = R_nn - v.t() * v + Ecirc_n * Ecirc_n.t();

  arma::mat LSigma = LinearAlgebra::safe_chol_lower(Sigma_nKo);

  arma::mat y_n(n_n, nsim);
  y_n.each_col() = yhat_n;
  Random::reset_seed(seed);
  y_n += LSigma * Random::randn_mat(n_n, nsim) * std::sqrt(m_sigma2);

  y_n = m_centerY + m_scaleY * y_n;

  return y_n;
}

// -------------------------------------------------------------------------
//  update()
// -------------------------------------------------------------------------
void MLPKriging::update(const arma::vec& y_new, const arma::mat& X_new) {
  if (!m_fitted)
    throw std::runtime_error("update: model not fitted");

  arma::vec y_all = arma::join_vert(m_y * m_scaleY + m_centerY, y_new);
  arma::mat X_all = m_X;
  for (arma::uword j = 0; j < X_all.n_cols; ++j)
    X_all.col(j) = X_all.col(j) * m_scaleX(j) + m_centerX(j);
  X_all = arma::join_vert(X_all, X_new);

  m_y = y_all;
  m_X = X_all;
  normalise_data();
  refresh_cache();

  arma::uword saved_adam = m_max_iter_adam;
  m_max_iter_adam /= 5;
  optimise_joint("BFGS+Adam");
  m_max_iter_adam = saved_adam;
}

// -------------------------------------------------------------------------
//  summary()
// -------------------------------------------------------------------------
std::string MLPKriging::summary() const {
  std::ostringstream oss;
  oss << "* MLPKriging\n"
      << "  - kernel:      " << m_kernel_name << "\n"
      << "  - regmodel:    " << m_regmodel << "\n"
      << "  - normalize:   " << (m_normalize ? "true" : "false") << "\n"
      << "  - n obs:       " << m_y.n_elem << "\n"
      << "  - d input:     " << m_X.n_cols << "\n"
      << "  - d features:  " << m_d_out << "\n";

  if (m_joint_warp) {
    oss << "  - warping:     " << m_joint_warp->describe() << "\n";
  }

  if (m_fitted) {
    oss << "  - sigma2:      " << m_sigma2 << "\n"
        << "  - theta:       " << m_theta.t() << "  - beta:        " << m_beta.t()
        << "  - LL:          " << logLikelihood() << "\n"
        << "  - total warp params: " << total_warp_params() << "\n";
  } else {
    oss << "  ** model not yet fitted **\n";
  }
  return oss.str();
}

// -------------------------------------------------------------------------
//  save / load
// -------------------------------------------------------------------------
void MLPKriging::save(const std::string filename) const {
  nlohmann::json j;

  j["version"] = 2;
  j["content"] = "MLPKriging";

  // Architecture
  j["kernel"] = m_kernel_name;
  j["hidden_dims"] = std::vector<arma::uword>(m_hidden_dims.begin(), m_hidden_dims.end());
  j["d_out"] = m_d_out;
  j["activation"] = m_activation;

  // Training data
  j["X"] = to_json(m_X);
  j["y"] = to_json(m_y);

  // Normalization
  j["normalize"] = m_normalize;
  j["X_mean"] = to_json(m_centerX);
  j["X_std"] = to_json(m_scaleX);
  j["y_mean"] = m_centerY;
  j["y_std"] = m_scaleY;

  // Settings
  j["regmodel"] = m_regmodel;
  j["max_iter_bfgs"] = m_max_iter_bfgs;
  j["max_iter_adam"] = m_max_iter_adam;
  j["adam_lr"] = m_adam_lr;

  // State
  j["fitted"] = m_fitted;

  // GP hyperparameters
  j["theta"] = to_json(m_theta);
  j["sigma2"] = m_sigma2;

  // GP cached data
  j["beta"] = to_json(m_beta);
  j["z"] = to_json(m_z);
  j["M"] = to_json(m_M);
  j["circ"] = to_json(m_circ);
  j["F"] = to_json(m_F);
  j["R"] = to_json(m_R);
  j["C"] = to_json(m_T);
  j["Rinv"] = to_json(m_Rinv);
  j["logdet"] = m_logdet;
  j["Phi"] = to_json(m_Phi);
  j["dPhi"] = to_json(m_dPhi);

  // MLP warp parameters
  j["warp_params"] = to_json(pack_warp_params());

  std::ofstream f(filename);
  f << std::setw(4) << j;
}

MLPKriging MLPKriging::load(const std::string filename) {
  std::ifstream f(filename);
  nlohmann::json j = nlohmann::json::parse(f);

  uint32_t version = j["version"].template get<uint32_t>();
  if (version != 2) {
    throw std::runtime_error(asString("Bad version to load from '", filename, "'; found ", version, ", requires 2"));
  }
  std::string content = j["content"].template get<std::string>();
  if (content != "MLPKriging") {
    throw std::runtime_error(
        asString("Bad content to load from '", filename, "'; found '", content, "', requires 'MLPKriging'"));
  }

  // Reconstruct from architecture
  auto hd = j["hidden_dims"].template get<std::vector<arma::uword>>();
  arma::uword d_out_val = j["d_out"].template get<arma::uword>();
  std::string activation = j["activation"].template get<std::string>();
  std::string kernel = j["kernel"].template get<std::string>();
  MLPKriging mk(hd, d_out_val, activation, kernel);

  // Training data
  mk.m_X = mat_from_json(j["X"]);
  mk.m_y = colvec_from_json(j["y"]);

  // Normalization
  mk.m_normalize = j["normalize"].template get<bool>();
  mk.m_centerX = rowvec_from_json(j["X_mean"]);
  mk.m_scaleX = rowvec_from_json(j["X_std"]);
  mk.m_centerY = j["y_mean"].template get<double>();
  mk.m_scaleY = j["y_std"].template get<double>();

  // Settings
  mk.m_regmodel = j["regmodel"].template get<std::string>();
  mk.m_max_iter_bfgs = j["max_iter_bfgs"].template get<arma::uword>();
  mk.m_max_iter_adam = j["max_iter_adam"].template get<arma::uword>();
  mk.m_adam_lr = j["adam_lr"].template get<double>();

  // State
  mk.m_fitted = j["fitted"].template get<bool>();

  // GP hyperparameters
  mk.m_theta = colvec_from_json(j["theta"]);
  mk.m_sigma2 = j["sigma2"].template get<double>();

  // GP cached data
  mk.m_beta = colvec_from_json(j["beta"]);
  mk.m_z = colvec_from_json(j["z"]);
  mk.m_M = mat_from_json(j["M"]);
  mk.m_circ = mat_from_json(j["circ"]);
  mk.m_F = mat_from_json(j["F"]);
  mk.m_R = mat_from_json(j["R"]);
  mk.m_T = mat_from_json(j["C"]);
  mk.m_Rinv = mat_from_json(j["Rinv"]);
  mk.m_logdet = j["logdet"].template get<double>();
  mk.m_Phi = mat_from_json(j["Phi"]);
  mk.m_dPhi = mat_from_json(j["dPhi"]);

  // Restore MLP warp parameters — need to instantiate the joint warp first
  mk.ensure_joint_warp(mk.m_X.n_cols);
  arma::vec wp = colvec_from_json(j["warp_params"]);
  mk.unpack_warp_params(wp);

  return mk;
}

}  // namespace libKriging
