// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#include <cmath>
#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/MultiOutputKriging.hpp"

#include "libKriging/KrigingImpl.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/Optim.hpp"
#include "libKriging/Random.hpp"
#include "libKriging/utils/data_from_arma_vec.hpp"

#include <lbfgsb_cpp/lbfgsb.hpp>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>

// =============================================================================
// "shared" output model: one θ for all outputs, β_j and σ_j² per output
// =============================================================================

class MultiOutputKriging::SharedModel : public KrigingImpl {
 public:
  explicit SharedModel(const std::string& covType) {
    m_covType = covType;
    make_Cov(covType);
  }

  // normalized outputs (n × q), whitened residuals (n × q), trend (p × q),
  // variances (q), output centering / scaling (1 × q)
  arma::mat Y;
  arma::mat Z;
  arma::mat B;
  arma::vec s2;
  arma::rowvec centerY;
  arma::rowvec scaleY;
  /// outputs entering the likelihood: an output that is exactly constant
  /// (e.g. a curve's initial condition) is reproduced exactly by a trend with
  /// an intercept, has σ_j² = 0 and is left out (it would make LL infinite)
  arma::uvec active;

  [[nodiscard]] bool fitted() const { return !m_is_empty; }
  [[nodiscard]] const arma::vec& theta_() const { return m_theta; }

  void normalize_outputs(const arma::mat& Yraw, bool normalize) {
    const arma::uword q = Yraw.n_cols;
    centerY = arma::rowvec(q, arma::fill::zeros);
    scaleY = arma::rowvec(q, arma::fill::ones);
    if (normalize) {
      // same as Kriging: min and range of each output
      centerY = arma::min(Yraw, 0);
      const arma::rowvec range = arma::max(Yraw, 0) - centerY;
      for (arma::uword j = 0; j < q; ++j)
        if (range(j) > 0)
          scaleY(j) = range(j);
    }
    Y = Yraw.each_row() - centerY;
    Y.each_row() /= scaleY;
  }

  void set_active(const arma::mat& Yraw) {
    const arma::rowvec range = arma::max(Yraw, 0) - arma::min(Yraw, 0);
    active = (m_regmodel == Trend::RegressionModel::None) ? arma::regspace<arma::uvec>(0, Yraw.n_cols - 1)
                                                          : arma::uvec(arma::find(range > 0));
    if (active.is_empty())
      throw std::invalid_argument("MultiOutputKriging: all outputs are constant");
  }

  KModel make_model(const arma::vec& theta) const {
    KModel m = allocate_KModel();
    populate_Model(m, theta, 1.0, KrigingImpl::ones, false, nullptr, &Y);
    return m;
  }

  // Summed concentrated log-likelihood: Σ_j −½ (n log(2π σ̂_j²) + log|R| + n),
  // σ̂_j² = SSE_j / n. Gradient in θ.
  double logLikelihood(const arma::vec& theta, arma::vec* grad_out, KModel* model) const {
    const double n = static_cast<double>(m_X.n_rows);
    const arma::uword d = m_X.n_cols;
    const double q = static_cast<double>(active.n_elem);

    KModel m_local;
    if (model != nullptr)
      populate_Model(*model, theta, 1.0, KrigingImpl::ones, false, nullptr, &Y);
    else
      m_local = make_model(theta);
    KModel& m = (model != nullptr) ? *model : m_local;

    const arma::mat E = (active.n_elem == Y.n_cols) ? m.Estar : arma::mat(m.Estar.cols(active));
    const arma::rowvec s2_j = arma::sum(arma::square(E), 0) / n;
    const double logdetL = arma::sum(arma::log(m.L.diag()));
    const double ll = -0.5 * arma::accu(n * arma::log(2 * M_PI * s2_j) + 2 * logdetL + n);

    if (grad_out != nullptr) {
      if ((m.Rinv.memptr() == nullptr) || (arma::size(m.Rinv) != arma::size(m.L)))
        m.Rinv = LinearAlgebra::inv_sympd(m.L);
      arma::mat x = LinearAlgebra::solve_upper(m.L.t(), E);
      x.each_row() /= arma::sqrt(s2_j);
      arma::vec term1(d, arma::fill::zeros);
      arma::vec term2(d, arma::fill::zeros);
      compute_ll_grad_theta_vecs(m.R, m.Rinv, x, theta, term1, term2);
      *grad_out = (term1 + q * term2) / 2.0;
    }
    return ll;
  }

  void commit(KModel& m) {
    const double n = static_cast<double>(m_X.n_rows);
    m_R = std::move(m.R);
    m_T = std::move(m.L);
    m_Rinv = std::move(m.Rinv);
    m_M = std::move(m.Fstar);
    m_circ = std::move(m.Rstar);
    B = std::move(m.betahat);
    Z = std::move(m.Estar);
    s2 = arma::vec(Y.n_cols, arma::fill::zeros);
    s2.elem(active) = arma::trans(arma::sum(arma::square(Z.cols(active)), 0)) / n;
    m_is_empty = false;
  }

  void fit(const arma::mat& Yraw,
           const arma::mat& X,
           const Trend::RegressionModel& regmodel,
           bool normalize,
           const std::string& optim,
           const std::string& objective,
           const Parameters& parameters) {
    if (objective != "LL")
      throw std::invalid_argument("MultiOutputKriging: objective '" + objective
                                  + "' is not supported by the \"shared\" output model (only \"LL\")");
    const arma::uword n = X.n_rows;
    const arma::uword d = X.n_cols;
    m_optim = optim;
    m_objective = objective;
    m_is_empty = true;

    normalize_outputs(Yraw, normalize);
    arma::mat theta0 = fit_setup_X_impl(X, regmodel, normalize, parameters.theta);
    set_active(Yraw);
    m_est_beta = true;
    m_est_sigma2 = true;

    if (optim == "none" || !parameters.is_theta_estim) {
      if (!parameters.theta.has_value())
        throw std::invalid_argument("MultiOutputKriging: theta must be given (1 x " + std::to_string(d)
                                    + ") when optim = \"none\" or is_theta_estim = false");
      m_theta = trans(theta0.row(0));
      m_est_theta = false;
      KModel m = make_model(m_theta);
      commit(m);
      return;
    }
    if (optim.rfind("BFGS", 0) != 0)
      throw std::invalid_argument("MultiOutputKriging: unsupported optim '" + optim + "' (none, BFGS[#])");

    // θ bounds: union of the per-output bounds of Kriging
    arma::vec theta_lower, theta_upper;
    for (arma::uword a = 0; a < active.n_elem; ++a) {
      auto [lo, up] = Optim::theta_bounds(m_maxdX, m_dX, Y.col(active(a)), n);
      theta_lower = (a == 0) ? lo : arma::min(theta_lower, lo);
      theta_upper = (a == 0) ? up : arma::max(theta_upper, up);
    }

    // starting points: same draw as Kriging (so q = 1 reproduces it)
    Random::init();
    int multistart = Optim::parse_method(optim, "BFGS").second;
    arma::mat theta0_rand
        = arma::repmat(trans(theta_lower), multistart, 1)
          + Random::randu_mat(multistart, d) % arma::repmat(trans(theta_upper - theta_lower), multistart, 1);
    if (parameters.theta.has_value()) {
      multistart = std::max(multistart, static_cast<int>(theta0.n_rows));
      theta0 = arma::join_cols(theta0, theta0_rand);
      theta0.resize(multistart, theta0.n_cols);
    } else {
      theta0 = theta0_rand;
    }

    auto to_gamma = [](const arma::vec& t) { return Optim::reparametrize ? Optim::reparam_to(t) : t; };
    auto from_gamma = [](const arma::vec& g) { return Optim::reparametrize ? Optim::reparam_from(g) : g; };
    // minimized: −LL in γ
    auto ofn = [&](const arma::vec& gamma, arma::vec* grad_out, KModel* m) -> double {
      const arma::vec theta = from_gamma(gamma);
      double ll = logLikelihood(theta, grad_out, m);
      if (grad_out != nullptr)
        *grad_out = Optim::reparametrize ? arma::vec(-Optim::reparam_from_deriv(theta, *grad_out))
                                         : arma::vec(-*grad_out);
      return -ll;
    };

    const arma::vec gamma_lower = to_gamma(theta_lower);
    const arma::vec gamma_upper = to_gamma(theta_upper);

    double best_ofn = std::numeric_limits<double>::infinity();
    arma::vec best_theta;
    std::string last_error;
    for (int s = 0; s < multistart; ++s) {
      try {
        const arma::vec theta_start = theta0.row(s).t();
        arma::vec gamma = to_gamma(theta_start);
        arma::vec lower = arma::min(gamma, gamma_lower);
        arma::vec upper = arma::max(gamma, gamma_upper);

        KModel m = allocate_KModel();
        lbfgsb::Optimizer optimizer{static_cast<unsigned int>(d)};
        optimizer.iprint = -1;
        optimizer.max_iter = Optim::max_iteration;
        optimizer.pgtol = Optim::gradient_tolerance;
        optimizer.factr = Optim::objective_rel_tolerance / 1E-13;
        arma::ivec bounds_type{d, arma::fill::value(2)};

        // same restart policy as Kriging
        double start_best = std::numeric_limits<double>::infinity();
        arma::vec start_best_gamma = gamma;
        for (int retry = 0; retry <= Optim::max_restart; ++retry) {
          auto res = optimizer.minimize(
              [&](const arma::vec& g, arma::vec& grad) -> double { return ofn(g, &grad, &m); },
              gamma,
              lower.memptr(),
              upper.memptr(),
              bounds_type.memptr());
          if (res.f_opt < start_best) {
            start_best = res.f_opt;
            start_best_gamma = gamma;
          }
          const double sol_to_lb = arma::min(arma::abs(from_gamma(gamma) - theta_lower));
          if ((retry < Optim::max_restart)
              && ((res.task.rfind("ABNORMAL_TERMINATION_IN_LNSRCH", 0) == 0) || (res.num_iters <= 2)
                  || (sol_to_lb < arma::datum::eps) || (res.f_opt > start_best))) {
            const arma::vec restart_theta = (theta_start + theta_lower) / std::pow(2.0, retry + 1);
            gamma = to_gamma(restart_theta);
            lower = arma::min(gamma, lower);
            upper = arma::max(gamma, upper);
          } else {
            break;
          }
        }
        const double f = ofn(start_best_gamma, nullptr, &m);
        if (f < best_ofn) {
          best_ofn = f;
          best_theta = from_gamma(start_best_gamma);
        }
      } catch (const std::exception& e) {
        last_error = e.what();
      }
    }
    if (best_theta.is_empty())
      throw std::runtime_error("MultiOutputKriging: all " + std::to_string(multistart)
                               + " optimization starts failed (" + last_error + ")");

    m_theta = best_theta;
    m_est_theta = true;
    KModel m = make_model(m_theta);
    commit(m);
  }

  // Normalized new points (d × m) and trend (m × p)
  std::pair<arma::mat, arma::mat> normalize_X(const arma::mat& X_n) const {
    arma::mat Xn = X_n;
    Xn.each_row() -= m_centerX;
    Xn.each_row() /= m_scaleX;
    arma::mat F_n = Trend::regressionModelMatrix(m_regmodel, Xn);
    return {arma::trans(Xn), F_n};
  }

  std::tuple<arma::mat, arma::mat, arma::mat, arma::cube> predict(const arma::mat& X_n,
                                                                  bool return_stdev,
                                                                  bool return_cov,
                                                                  bool return_deriv) const {
    const arma::uword m = X_n.n_rows;
    const arma::uword n_o = m_X.n_rows;
    const arma::uword d = m_X.n_cols;
    const arma::uword q = Y.n_cols;
    auto [Xn, F_n] = normalize_X(X_n);
    const arma::mat Xo = arma::trans(m_X);

    const arma::mat R_on = cross_corr(Xo, Xn, 1.0, true);
    const arma::mat Rstar_on = LinearAlgebra::solve_lower(m_T, R_on);

    arma::mat mean = F_n * B + Rstar_on.t() * Z;
    mean.each_row() %= scaleY;
    mean.each_row() += centerY;

    const arma::mat Ecirc_n = LinearAlgebra::rsolve_upper(m_circ, F_n - Rstar_on.t() * m_M);
    // σ_j² s_j²: variance scale of output j
    const arma::rowvec vscale = arma::trans(s2) % arma::square(scaleY);

    arma::mat stdev;
    if (return_stdev) {
      arma::vec base = 1.0 - arma::sum(Rstar_on % Rstar_on, 0).as_col() + arma::sum(Ecirc_n % Ecirc_n, 1);
      base.transform([](double v) { return (std::isnan(v) || v < 0) ? 0.0 : v; });
      stdev = arma::sqrt(base * vscale);
    }

    arma::mat cov;
    if (return_cov) {
      arma::mat R_nn(m, m, arma::fill::none);
      LinearAlgebra::covMat_sym_X(&R_nn, Xn, m_theta, _Cov, 1.0, KrigingImpl::ones);
      const arma::mat Sigma = R_nn - Rstar_on.t() * Rstar_on + Ecirc_n * Ecirc_n.t();
      cov = arma::kron(arma::diagmat(vscale), Sigma);  // block-diagonal over vec(Y_n)
    }

    arma::cube deriv;
    if (return_deriv) {
      deriv.zeros(m, d, q);
      for (arma::uword i = 0; i < m; ++i) {
        arma::mat DR_on_i(n_o, d, arma::fill::none);
        for (arma::uword j = 0; j < n_o; ++j)
          DR_on_i.row(j) = R_on.at(j, i) * arma::trans(_DlnCovDx(Xn.col(i) - Xo.col(j), m_theta));
        const arma::mat DF_n_i = Trend::regressionModelDerivative(m_regmodel, Xn.col(i));  // d × p
        const arma::mat W_i = LinearAlgebra::solve_lower(m_T, DR_on_i);
        const arma::mat D = DF_n_i * B + W_i.t() * Z;  // d × q
        for (arma::uword k = 0; k < q; ++k)
          deriv.slice(k).row(i) = arma::trans(D.col(k)) * scaleY(k) / m_scaleX;
      }
    }
    return std::make_tuple(std::move(mean), std::move(stdev), std::move(cov), std::move(deriv));
  }

  arma::cube simulate(int nsim, int seed, const arma::mat& X_n) const {
    const arma::uword m = X_n.n_rows;
    const arma::uword q = Y.n_cols;
    auto [Xn, F_n] = normalize_X(X_n);
    const arma::mat R_on = cross_corr(arma::trans(m_X), Xn, 1.0, false);
    const arma::mat Rstar_on = LinearAlgebra::solve_lower(m_T, R_on);
    const arma::mat yhat = F_n * B + Rstar_on.t() * Z;
    const arma::mat Ecirc_n = LinearAlgebra::rsolve_upper(m_circ, F_n - Rstar_on.t() * m_M);
    arma::mat R_nn(m, m, arma::fill::none);
    LinearAlgebra::covMat_sym_X(&R_nn, Xn, m_theta, _Cov, 1.0, KrigingImpl::ones);
    const arma::mat L = LinearAlgebra::safe_chol_lower(R_nn - Rstar_on.t() * Rstar_on + Ecirc_n * Ecirc_n.t());

    // outputs drawn in turn from one seeded stream (q = 1 matches Kriging)
    arma::cube out(m, q, nsim);
    Random::reset_seed(seed);
    for (arma::uword j = 0; j < q; ++j) {
      arma::mat y_n = L * Random::randn_mat(m, nsim) * std::sqrt(s2(j));
      y_n.each_col() += yhat.col(j);
      y_n = centerY(j) + scaleY(j) * y_n;
      for (int s = 0; s < nsim; ++s)
        out.slice(s).col(j) = y_n.col(s);
    }
    return out;
  }

  // Append data, keep θ, re-estimate β_j and σ_j²
  void update_no_refit(const arma::mat& Y_u, const arma::mat& X_u) {
    arma::mat Xn_u = X_u;
    Xn_u.each_row() -= m_centerX;
    Xn_u.each_row() /= m_scaleX;
    arma::mat Yn_u = Y_u.each_row() - centerY;
    Yn_u.each_row() /= scaleY;
    m_X = arma::join_cols(m_X, Xn_u);
    Y = arma::join_cols(Y, Yn_u);
    m_dX = LinearAlgebra::compute_dX(m_X);
    m_maxdX = arma::max(arma::abs(m_dX), 1);
    m_F = Trend::regressionModelMatrix(m_regmodel, m_X);
    KModel m = make_model(m_theta);
    commit(m);
  }

  // Closed-form LOO (Dubrule 1983), shared R: (mean, stdev) n × q
  std::tuple<arma::mat, arma::mat> leaveOneOut() const {
    const arma::uword n = m_X.n_rows;
    const arma::mat Linv = LinearAlgebra::solve_lower(m_T, arma::mat(n, n, arma::fill::eye));
    arma::mat Q, Rq;
    LinearAlgebra::qr_econ(Q, Rq, m_M);
    const arma::mat A = Q.t() * Linv;
    const arma::mat H = LinearAlgebra::crossprod(Linv) - LinearAlgebra::crossprod(A);
    const arma::vec s2loo = 1.0 / H.diag();
    arma::mat err = Linv.t() * Z;
    err.each_col() %= s2loo;
    arma::mat mean = Y - err;
    mean.each_row() %= scaleY;
    mean.each_row() += centerY;
    arma::mat sd = arma::sqrt(s2loo * arma::trans(s2));
    sd.each_row() %= scaleY;
    return std::make_tuple(std::move(mean), std::move(sd));
  }

  std::string summary() const {
    std::ostringstream os;
    os << "  * theta: " << arma::trans(m_theta);
    os << "  * sigma2: " << arma::trans(s2);
    return os.str();
  }
};


// =============================================================================
// construction / configuration
// =============================================================================

MultiOutputKriging::MultiOutputKriging(const std::string& covType, const std::string& outputModel)
    : m_covType(covType) {
  parse_output_model(outputModel);
}

MultiOutputKriging::MultiOutputKriging(const arma::mat& Y,
                                       const arma::mat& X,
                                       const std::string& covType,
                                       const std::string& outputModel,
                                       const Trend::RegressionModel& regmodel,
                                       bool normalize,
                                       const std::string& optim,
                                       const std::string& objective,
                                       const Parameters& parameters)
    : MultiOutputKriging(covType, outputModel) {
  fit(Y, X, regmodel, normalize, optim, objective, parameters);
}

MultiOutputKriging::~MultiOutputKriging() = default;
MultiOutputKriging::MultiOutputKriging(MultiOutputKriging&&) noexcept = default;
MultiOutputKriging& MultiOutputKriging::operator=(MultiOutputKriging&&) noexcept = default;

void MultiOutputKriging::parse_output_model(const std::string& s) {
  const auto open = s.find('(');
  const std::string head = s.substr(0, open);
  std::string arg;
  if (open != std::string::npos) {
    if (s.back() != ')' || s.size() <= open + 2)
      throw std::invalid_argument("MultiOutputKriging: malformed output model '" + s + "'");
    arg = s.substr(open + 1, s.size() - open - 2);
  }

  if (head == "pca") {
    m_output_model = OutputModel::PCA;
    if (arg.empty()) {
      m_pca_spec = 0.99;
      return;
    }
    double v = 0;
    try {
      std::size_t pos = 0;
      v = std::stod(arg, &pos);
      if (pos != arg.size())
        throw std::invalid_argument("");
    } catch (const std::exception&) {
      throw std::invalid_argument("MultiOutputKriging: malformed pca argument in '" + s + "'");
    }
    const bool fraction = v > 0 && v < 1;
    const bool count = v >= 1 && std::abs(v - std::round(v)) < 1e-12;
    if (!fraction && !count)
      throw std::invalid_argument(
          "MultiOutputKriging: pca argument must be an integer >= 1 or a fraction in (0,1), got '" + arg + "'");
    m_pca_spec = v;
  } else if (head == "shared" && arg.empty()) {
    m_output_model = OutputModel::Shared;
  } else if (head == "separable") {
    m_output_model = arg.empty() ? OutputModel::Separable : OutputModel::SeparableKernel;
    m_output_covType = arg;
  } else {
    throw std::invalid_argument("MultiOutputKriging: unknown output model '" + s
                                + "' (expected pca, pca(K), pca(v), shared, separable or separable(<kernel>))");
  }
}

std::string MultiOutputKriging::output_model_string() const {
  switch (m_output_model) {
    case OutputModel::Shared:
      return "shared";
    case OutputModel::Separable:
      return "separable";
    case OutputModel::SeparableKernel:
      return "separable(" + m_output_covType + ")";
    case OutputModel::PCA: {
      std::ostringstream os;
      os << "pca(" << m_pca_spec << ")";
      return os.str();
    }
  }
  return "?";
}

void MultiOutputKriging::set_output_coordinates(const arma::mat& t) {
  if (!t.is_finite())
    throw std::invalid_argument("MultiOutputKriging::set_output_coordinates: coordinates must be finite");
  m_t = t;
}

void MultiOutputKriging::check_fitted(const std::string& where) const {
  if (m_components.empty() && !(m_shared && m_shared->fitted()))
    throw std::runtime_error("MultiOutputKriging::" + where + ": model is not fitted");
}

const MultiOutputKriging::SharedModel& MultiOutputKriging::shared(const std::string& where) const {
  if (m_output_model != OutputModel::Shared)
    throw std::runtime_error("MultiOutputKriging::" + where + ": only available with the \"shared\" output model"
                             + (m_output_model == OutputModel::PCA ? " (use component(k) in \"pca\")" : ""));
  check_fitted(where);
  return *m_shared;
}

MultiOutputKriging::SharedModel& MultiOutputKriging::shared(const std::string& where) {
  return const_cast<SharedModel&>(static_cast<const MultiOutputKriging*>(this)->shared(where));
}

const arma::vec& MultiOutputKriging::theta() const {
  return shared("theta").theta_();
}

const arma::vec& MultiOutputKriging::sigma2() const {
  return shared("sigma2").s2;
}

const arma::mat& MultiOutputKriging::beta() const {
  return shared("beta").B;
}

double MultiOutputKriging::logLikelihood() {
  const SharedModel& sm = shared("logLikelihood");
  return sm.logLikelihood(sm.theta_(), nullptr, nullptr);
}

std::tuple<double, arma::vec> MultiOutputKriging::logLikelihoodFun(const arma::vec& theta, bool grad) {
  const SharedModel& sm = shared("logLikelihoodFun");
  if (theta.n_elem != m_X.n_cols)
    throw std::invalid_argument("MultiOutputKriging::logLikelihoodFun: theta must have " + std::to_string(m_X.n_cols)
                                + " elements");
  arma::vec g;
  const double ll = sm.logLikelihood(theta, grad ? &g : nullptr, nullptr);
  return std::make_tuple(ll, std::move(g));
}

const Kriging& MultiOutputKriging::component(arma::uword k) const {
  check_fitted("component");
  if (k >= m_components.size())
    throw std::out_of_range("MultiOutputKriging::component: index " + std::to_string(k)
                            + " >= nb_components() = " + std::to_string(m_components.size()));
  return *m_components[k];
}

// =============================================================================
// PCA helpers
// =============================================================================

arma::mat MultiOutputKriging::project(const arma::mat& Y) const {
  arma::mat Yc = Y.each_row() - m_centerY;
  Yc.each_row() /= m_scaleY;
  return Yc * m_pca_basis;
}

arma::mat MultiOutputKriging::reconstruct(const arma::mat& A) const {
  arma::mat Yr = A * m_pca_basis.t();
  Yr.each_row() %= m_scaleY;
  Yr.each_row() += m_centerY;
  return Yr;
}

arma::rowvec MultiOutputKriging::residual_variance() const {
  const double n = static_cast<double>(m_pca_residual.n_rows);
  return arma::sum(arma::square(m_pca_residual), 0) / (n - 1.0);
}

// =============================================================================
// fit
// =============================================================================

void MultiOutputKriging::fit(const arma::mat& Y,
                             const arma::mat& X,
                             const Trend::RegressionModel& regmodel,
                             bool normalize,
                             const std::string& optim,
                             const std::string& objective,
                             const Parameters& parameters) {
  if (m_output_model != OutputModel::PCA && m_output_model != OutputModel::Shared)
    throw std::runtime_error("MultiOutputKriging: output model '" + output_model_string()
                             + "' is not implemented yet (only \"pca\" and \"shared\")");

  const arma::uword n = Y.n_rows;
  const arma::uword q = Y.n_cols;
  if (X.n_rows != n)
    throw std::invalid_argument("MultiOutputKriging::fit: Y has " + std::to_string(n) + " rows but X has "
                                + std::to_string(X.n_rows) + " (rows are observations for both)");
  if (n < 2 || q < 1)
    throw std::invalid_argument("MultiOutputKriging::fit: need at least 2 observations and 1 output");
  if (!Y.is_finite() || !X.is_finite())
    throw std::invalid_argument("MultiOutputKriging::fit: Y and X must be finite (isotopic design, no missing value)");

  if (m_t.n_elem > 0 && m_t.n_rows != q)
    throw std::invalid_argument("MultiOutputKriging::fit: output_coordinates has " + std::to_string(m_t.n_rows)
                                + " rows but Y has " + std::to_string(q) + " columns (outputs)");

  m_X = X;
  m_Y = Y;
  m_regmodel = regmodel;
  m_normalize = normalize;
  m_optim = optim;
  m_objective = objective;
  m_parameters = parameters;
  m_lastsim_residual.reset();
  m_components.clear();
  m_shared.reset();

  if (m_output_model == OutputModel::Shared) {
    m_shared = std::make_unique<SharedModel>(m_covType);
    m_shared->fit(Y, X, regmodel, normalize, optim, objective, parameters);
    m_centerY = m_shared->centerY;
    m_scaleY = m_shared->scaleY;
    m_pca_basis.reset();
    m_pca_explained.reset();
    m_pca_residual.reset();
    return;
  }

  // --- centering / scaling ---------------------------------------------------
  m_centerY = arma::mean(Y, 0);
  m_scaleY = arma::rowvec(q, arma::fill::ones);
  if (normalize) {
    const arma::rowvec sd = arma::stddev(Y, 0, 0);
    for (arma::uword j = 0; j < q; ++j)
      if (sd(j) > 1e-12 * std::max(1.0, std::abs(m_centerY(j))))
        m_scaleY(j) = sd(j);
  }
  arma::mat Yc = Y.each_row() - m_centerY;
  Yc.each_row() /= m_scaleY;

  // --- PCA (SVD of the centered data) -----------------------------------------
  arma::mat U, V;
  arma::vec sv;
  if (!arma::svd_econ(U, sv, V, Yc))
    throw std::runtime_error("MultiOutputKriging::fit: SVD failed");
  const arma::vec lambda = arma::square(sv) / static_cast<double>(n - 1);
  const double total = arma::accu(lambda);
  if (!(total > 0))
    throw std::invalid_argument("MultiOutputKriging::fit: Y has no variance (all rows identical)");

  arma::uword rank = 0;
  while (rank < lambda.n_elem && lambda(rank) > 1e-12 * lambda(0))
    ++rank;
  const arma::uword Kmax = std::min<arma::uword>(rank, n - 1);
  const arma::vec cum = arma::cumsum(lambda) / total;

  arma::uword K;
  if (m_pca_spec >= 1) {
    K = std::min<arma::uword>(static_cast<arma::uword>(std::lround(m_pca_spec)), Kmax);
  } else {
    K = 1;
    while (K < Kmax && cum(K - 1) < m_pca_spec)
      ++K;
  }

  m_pca_basis = V.cols(0, K - 1);
  // deterministic sign: largest-magnitude loading of each axis is positive
  for (arma::uword k = 0; k < K; ++k) {
    const arma::uword imax = arma::index_max(arma::abs(m_pca_basis.col(k)));
    if (m_pca_basis(imax, k) < 0)
      m_pca_basis.col(k) *= -1.0;
  }
  m_pca_explained = cum.head(K);

  const arma::mat A = Yc * m_pca_basis;
  m_pca_residual = Yc - A * m_pca_basis.t();
  m_pca_residual.each_row() %= m_scaleY;

  // --- one Kriging per score ---------------------------------------------------
  Kriging::Parameters kp;
  kp.theta = parameters.theta;
  kp.is_theta_estim = parameters.is_theta_estim;

  m_components.clear();
  m_components.reserve(K);
  for (arma::uword k = 0; k < K; ++k) {
    auto km = std::make_unique<Kriging>(m_covType);
    km->fit(A.col(k), X, regmodel, normalize, optim, objective, kp);
    m_components.push_back(std::move(km));
  }
}

// =============================================================================
// predict
// =============================================================================

std::tuple<arma::mat, arma::mat, arma::mat, arma::cube> MultiOutputKriging::predict(const arma::mat& X_n,
                                                                                    bool return_stdev,
                                                                                    bool return_cov,
                                                                                    bool return_deriv) {
  check_fitted("predict");
  if (X_n.n_cols != m_X.n_cols)
    throw std::invalid_argument("MultiOutputKriging::predict: X_n has " + std::to_string(X_n.n_cols)
                                + " columns, expected " + std::to_string(m_X.n_cols));

  if (m_shared)
    return m_shared->predict(X_n, return_stdev, return_cov, return_deriv);

  const arma::uword m = X_n.n_rows;
  const arma::uword q = m_Y.n_cols;
  const arma::uword d = m_X.n_cols;
  const arma::uword K = m_components.size();

  arma::mat mu(m, K);
  arma::mat s2(return_stdev ? m : 0, K);
  std::vector<arma::mat> C(return_cov ? K : 0);
  std::vector<arma::mat> dmu(return_deriv ? K : 0);
  for (arma::uword k = 0; k < K; ++k) {
    auto [mean_k, sd_k, cov_k, dmean_k, dsd_k] = m_components[k]->predict(X_n, return_stdev, return_cov, return_deriv);
    mu.col(k) = mean_k;
    if (return_stdev)
      s2.col(k) = arma::square(sd_k);
    if (return_cov)
      C[k] = std::move(cov_k);
    if (return_deriv)
      dmu[k] = std::move(dmean_k);
  }

  arma::mat mean = reconstruct(mu);

  arma::mat stdev;
  if (return_stdev) {
    arma::mat var = s2 * arma::square(m_pca_basis).t();
    var.each_row() %= arma::square(m_scaleY);
    var.each_row() += residual_variance();
    stdev = arma::sqrt(var);
  }

  arma::mat cov;
  if (return_cov) {
    cov.zeros(m * q, m * q);
    for (arma::uword k = 0; k < K; ++k) {
      const arma::vec w = m_pca_basis.col(k) % m_scaleY.t();
      cov += arma::kron(w * w.t(), C[k]);
    }
    const double nm1 = static_cast<double>(m_pca_residual.n_rows) - 1.0;
    const arma::mat Sres = m_pca_residual.t() * m_pca_residual / nm1;
    cov += arma::kron(Sres, arma::eye(m, m));
  }

  arma::cube deriv;
  if (return_deriv) {
    deriv.zeros(m, d, q);
    for (arma::uword j = 0; j < q; ++j)
      for (arma::uword k = 0; k < K; ++k)
        deriv.slice(j) += (m_scaleY(j) * m_pca_basis(j, k)) * dmu[k];
  }

  return std::make_tuple(std::move(mean), std::move(stdev), std::move(cov), std::move(deriv));
}

// =============================================================================
// simulate / update_simulate
// =============================================================================

// Truncation residual draws Ρᵀw / sqrt(n-1), w ~ N(0, I_n), independent per
// (point, simulation): m × q × nsim.
static arma::cube draw_residual(const arma::mat& residual, arma::uword m, int nsim, int seed) {
  const arma::uword n = residual.n_rows;
  const arma::uword q = residual.n_cols;
  arma::cube out(m, q, nsim, arma::fill::zeros);
  if (!residual.is_zero(0)) {
    std::mt19937 gen(static_cast<std::mt19937::result_type>(seed));
    std::normal_distribution<double> N01(0.0, 1.0);
    const double f = 1.0 / std::sqrt(static_cast<double>(n) - 1.0);
    arma::mat W(n, m);
    for (int s = 0; s < nsim; ++s) {
      W.imbue([&]() { return N01(gen); });
      out.slice(s) = f * (W.t() * residual);
    }
  }
  return out;
}

arma::cube MultiOutputKriging::simulate(int nsim, int seed, const arma::mat& X_n, bool will_update) {
  check_fitted("simulate");
  if (X_n.n_cols != m_X.n_cols)
    throw std::invalid_argument("MultiOutputKriging::simulate: X_n has " + std::to_string(X_n.n_cols)
                                + " columns, expected " + std::to_string(m_X.n_cols));
  if (nsim < 1)
    throw std::invalid_argument("MultiOutputKriging::simulate: nsim must be >= 1");

  if (m_shared) {
    if (will_update)
      throw std::runtime_error(
          "MultiOutputKriging::simulate: will_update (update_simulate) is not implemented yet for \"shared\"");
    return m_shared->simulate(nsim, seed, X_n);
  }

  const arma::uword m = X_n.n_rows;
  const arma::uword K = m_components.size();

  // latent draws: component k uses seed + k; residual uses seed + K
  std::vector<arma::mat> Z(K);
  for (arma::uword k = 0; k < K; ++k)
    Z[k] = m_components[k]->simulate(nsim, seed + static_cast<int>(k), X_n, will_update);

  arma::cube res = draw_residual(m_pca_residual, m, nsim, seed + static_cast<int>(K));
  if (will_update)
    m_lastsim_residual = res;
  else
    m_lastsim_residual.reset();

  arma::cube out(m, m_Y.n_cols, nsim);
  arma::mat A(m, K);
  for (int s = 0; s < nsim; ++s) {
    for (arma::uword k = 0; k < K; ++k)
      A.col(k) = Z[k].col(s);
    out.slice(s) = reconstruct(A) + res.slice(s);
  }
  return out;
}

arma::cube MultiOutputKriging::update_simulate(const arma::mat& Y_u, const arma::mat& X_u) {
  check_fitted("update_simulate");
  if (m_lastsim_residual.n_elem == 0)
    throw std::runtime_error("MultiOutputKriging::update_simulate: call simulate(..., will_update=true) first");
  if (Y_u.n_cols != m_Y.n_cols || Y_u.n_rows != X_u.n_rows || X_u.n_cols != m_X.n_cols)
    throw std::invalid_argument("MultiOutputKriging::update_simulate: Y_u must be n_u × q and X_u n_u × d");

  const arma::uword K = m_components.size();
  const arma::mat A_u = project(Y_u);

  std::vector<arma::mat> Z(K);
  for (arma::uword k = 0; k < K; ++k)
    Z[k] = m_components[k]->update_simulate(A_u.col(k), X_u);

  const arma::uword m = m_lastsim_residual.n_rows;
  const arma::uword nsim = m_lastsim_residual.n_slices;
  arma::cube out(m, m_Y.n_cols, nsim);
  arma::mat A(m, K);
  for (arma::uword s = 0; s < nsim; ++s) {
    for (arma::uword k = 0; k < K; ++k)
      A.col(k) = Z[k].col(s);
    out.slice(s) = reconstruct(A) + m_lastsim_residual.slice(s);
  }
  return out;
}

// =============================================================================
// update
// =============================================================================

void MultiOutputKriging::update(const arma::mat& Y_u, const arma::mat& X_u, bool refit) {
  check_fitted("update");
  if (Y_u.n_cols != m_Y.n_cols || Y_u.n_rows != X_u.n_rows || X_u.n_cols != m_X.n_cols)
    throw std::invalid_argument("MultiOutputKriging::update: Y_u must be n_u × q and X_u n_u × d");

  if (refit) {
    const arma::mat Y_all = arma::join_cols(m_Y, Y_u);
    const arma::mat X_all = arma::join_cols(m_X, X_u);
    fit(Y_all, X_all, m_regmodel, m_normalize, m_optim, m_objective, m_parameters);
    return;
  }

  if (m_shared) {
    m_shared->update_no_refit(Y_u, X_u);
    m_X = arma::join_cols(m_X, X_u);
    m_Y = arma::join_cols(m_Y, Y_u);
    return;
  }

  const arma::mat A_u = project(Y_u);
  for (arma::uword k = 0; k < m_components.size(); ++k)
    m_components[k]->update(A_u.col(k), X_u, false);
  m_pca_residual = arma::join_cols(m_pca_residual, Y_u - reconstruct(A_u));
  m_X = arma::join_cols(m_X, X_u);
  m_Y = arma::join_cols(m_Y, Y_u);
  m_lastsim_residual.reset();
}

// =============================================================================
// leave-one-out
// =============================================================================

std::tuple<arma::mat, arma::mat> MultiOutputKriging::leaveOneOutMat() {
  check_fitted("leaveOneOutMat");
  if (m_shared)
    return m_shared->leaveOneOut();
  const arma::uword n = m_Y.n_rows;
  const arma::uword K = m_components.size();

  arma::mat A(n, K), S2(n, K);
  for (arma::uword k = 0; k < K; ++k) {
    Kriging& km = *m_components[k];
    auto [yhat, sd] = km.leaveOneOutVec(km.theta());
    // leaveOneOutVec works on the (possibly normalized) internal scale
    A.col(k) = yhat * km.scaleY() + km.centerY();
    S2.col(k) = arma::square(sd * km.scaleY());
  }

  arma::mat mean = reconstruct(A);
  arma::mat var = S2 * arma::square(m_pca_basis).t();
  var.each_row() %= arma::square(m_scaleY);
  var.each_row() += residual_variance();
  return std::make_tuple(std::move(mean), arma::mat(arma::sqrt(var)));
}

double MultiOutputKriging::leaveOneOut() {
  const arma::mat mean = std::get<0>(leaveOneOutMat());
  return arma::accu(arma::square(m_Y - mean)) / static_cast<double>(m_Y.n_elem);
}

// =============================================================================
// summary
// =============================================================================

std::string MultiOutputKriging::summary() const {
  std::ostringstream os;
  os << "* MultiOutputKriging, output model: " << output_model_string() << "\n";
  os << "  * covariance kernel: " << m_covType << "\n";
  if (m_components.empty() && !(m_shared && m_shared->fitted())) {
    os << "  (not fitted)\n";
    return os.str();
  }
  os << "  * data: " << m_X.n_rows << " x " << m_X.n_cols << " -> " << m_Y.n_rows << " x " << m_Y.n_cols << "\n";
  os << "  * trend: " << Trend::toString(m_regmodel) << (m_normalize ? " (normalized outputs)" : "") << "\n";
  if (m_shared) {
    os << m_shared->summary();
    return os.str();
  }
  os << "  * PCA components: " << m_components.size()
     << ", explained variance: " << m_pca_explained(m_pca_explained.n_elem - 1) << "\n";
  for (arma::uword k = 0; k < m_components.size(); ++k) {
    const Kriging& km = *m_components[k];
    os << "    - component " << k << ": cum. explained " << m_pca_explained(k) << ", sigma2 " << km.sigma2()
       << ", theta " << arma::trans(km.theta());
  }
  return os.str();
}
