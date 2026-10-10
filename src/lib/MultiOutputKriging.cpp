// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#include <cmath>
#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/MultiOutputKriging.hpp"

#include "libKriging/KrigingImpl.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/Optim.hpp"
#include "libKriging/Random.hpp"
#include "libKriging/utils/data_from_arma_vec.hpp"
#include "libKriging/utils/jsonutils.hpp"
#include "libKriging/utils/nlohmann/json.hpp"

#include <fstream>
#include <iomanip>
#include <lbfgsb_cpp/lbfgsb.hpp>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>

// =============================================================================
// "shared" and "separable" output models: one θ for all outputs, β_j per
// output; Cov(vec Y) = Σ ⊗ R_θ with Σ diagonal (shared, PP-GaSP), free q × q
// (separable, ICM) or σ² R_t(φ) from output coordinates (separable(<kernel>)).
// All give the same θ-profile of the trend and the same predictive mean
// (autokrigeability, isotopic design).
// =============================================================================

/// Correlation kernel on the output coordinates t (q × d_t), for
/// "separable(<kernel>)": R_t(φ) and its gradient loop, reusing the
/// KrigingImpl kernel machinery with t as the "design".
class MultiOutputKriging::OutputKernel : public KrigingImpl {
 public:
  OutputKernel(const std::string& covType, const arma::mat& t) {
    m_covType = covType;
    make_Cov(covType);
    m_X = t;
    m_dX = LinearAlgebra::compute_dX(m_X);
    m_maxdX = arma::max(arma::abs(m_dX), 1);
  }

  /// R_t(φ), q × q
  [[nodiscard]] arma::mat corr(const arma::vec& phi) const {
    arma::mat R(m_X.n_rows, m_X.n_rows, arma::fill::none);
    LinearAlgebra::covMat_sym_X(&R, arma::trans(m_X), phi, _Cov, 1.0, KrigingImpl::ones);
    return R;
  }

  /// Σ_ij ∂R_ij x_i·x_j and −tr(R⁻¹ ∂R), per φ_k (see compute_ll_grad_theta_vecs)
  void grad_terms(const arma::mat& R,
                  const arma::mat& Rinv,
                  const arma::mat& x,
                  const arma::vec& phi,
                  arma::vec& term1,
                  arma::vec& term2) const {
    term1.zeros(phi.n_elem);
    term2.zeros(phi.n_elem);
    compute_ll_grad_theta_vecs(R, Rinv, x, phi, term1, term2);
  }

  /// φ bounds: same factors of the coordinate ranges as θ in Kriging
  [[nodiscard]] std::pair<arma::vec, arma::vec> bounds() const {
    return Optim::theta_bounds(m_maxdX, m_dX, arma::vec(), 0);
  }

  [[nodiscard]] arma::uword dim() const { return m_X.n_cols; }
};

static void check_output_kernel(const std::string& covType) {
  struct Probe : KrigingImpl {
    explicit Probe(const std::string& k) { make_Cov(k); }
  };
  Probe probe(covType);
}

class MultiOutputKriging::SharedModel : public KrigingImpl {
 public:
  enum class SigmaForm {
    Diag,    ///< Σ diagonal ("shared")
    Full,    ///< Σ free q × q ("separable")
    Kernel,  ///< Σ = σ² R_t(φ) ("separable(<kernel>)")
  };

  SharedModel(const std::string& covType,
              SigmaForm form,
              const std::string& output_covType = "",
              const arma::mat& t = arma::mat())
      : sigma_form(form), full(form != SigmaForm::Diag) {
    m_covType = covType;
    make_Cov(covType);
    if (form == SigmaForm::Kernel)
      tker = std::make_unique<OutputKernel>(output_covType, t);
  }

  SigmaForm sigma_form = SigmaForm::Diag;
  /// Σ not diagonal: outputs mixed through its Cholesky factor
  bool full = false;
  /// kernel on t and its fitted φ ("separable(<kernel>)" only)
  std::unique_ptr<OutputKernel> tker;
  arma::vec phi;

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
  /// Σ (q × q, normalized scale; zero rows/columns for constant outputs) and
  /// its lower Cholesky factor; diag(Σ) = s2
  arma::mat Sig;
  arma::mat Lsig;

  // state of the last simulate(..., will_update = true): raw points (m × d),
  // normalized draws (m × q × nsim), seed
  arma::mat sim_X;
  arma::cube sim_Yn;
  int sim_seed = 0;

  [[nodiscard]] bool fitted() const { return !m_is_empty; }
  [[nodiscard]] const arma::vec& theta_() const { return m_theta; }
  [[nodiscard]] bool kernel_form() const { return sigma_form == SigmaForm::Kernel; }

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
    // Σ = σ² R_t keeps every output: a constant one only has a zero residual
    if (kernel_form())
      active = arma::regspace<arma::uvec>(0, Yraw.n_cols - 1);
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
    const double logdetL = arma::sum(arma::log(m.L.diag()));
    double ll;
    arma::rowvec s2_j;
    arma::mat Ls;
    if (!full) {
      s2_j = arma::sum(arma::square(E), 0) / n;
      ll = -0.5 * arma::accu(n * arma::log(2 * arma::datum::pi * s2_j) + 2 * logdetL + n);
    } else {
      // −2ℓ = nq log 2π + n log|Σ̂| + q log|R| + nq,  Σ̂ = E*ᵀ E* / n
      Ls = sigma_chol(E.t() * E / n);
      ll = -0.5
           * (n * q * std::log(2 * arma::datum::pi) + 2 * n * arma::sum(arma::log(Ls.diag())) + 2 * q * logdetL
              + n * q);
    }

    if (grad_out != nullptr) {
      if ((m.Rinv.memptr() == nullptr) || (arma::size(m.Rinv) != arma::size(m.L)))
        m.Rinv = LinearAlgebra::inv_sympd(m.L);
      arma::mat x = LinearAlgebra::solve_upper(m.L.t(), E);
      // whiten across outputs: x Σ̂^{-1/2}, so that Σ_ij dR_ij x_i·x_j = tr(Σ̂⁻¹ xᵀ dR x)
      if (!full)
        x.each_row() /= arma::sqrt(s2_j);
      else
        x = arma::trans(LinearAlgebra::solve_lower(Ls, x.t()));
      arma::vec term1(d, arma::fill::zeros);
      arma::vec term2(d, arma::fill::zeros);
      compute_ll_grad_theta_vecs(m.R, m.Rinv, x, theta, term1, term2);
      *grad_out = (term1 + q * term2) / 2.0;
    }
    return ll;
  }

  /// σ̂² = tr(R_t⁻¹ E*ᵀ E*) / (nq) and the Cholesky factor of R_t(φ)
  std::pair<double, arma::mat> kernel_sigma2(const arma::mat& E, const arma::vec& phi_) const {
    const arma::mat Lt = LinearAlgebra::safe_chol_lower(tker->corr(phi_));
    const arma::mat W = LinearAlgebra::solve_lower(Lt, E.t());  // q × n
    return {arma::accu(W % W) / static_cast<double>(E.n_elem), Lt};
  }

  // "separable(<kernel>)": vec(Y − F B) ~ N(0, σ² R_t(φ) ⊗ R_x(θ)), with
  //   −2ℓ = nq log 2π + q log|R_x| + n log|R_t| + nq log σ̂² + nq,
  //   σ̂² = tr(R_t⁻¹ E*ᵀ E*) / (nq)  (B̂ is the per-output GLS, as in "shared").
  // Gradient in θ (d) then φ (d_t): the rows of E* are iid N(0, σ² R_t), so
  // φ enters as θ does, with E*ᵀ as data and n in place of q.
  double logLikelihoodKernel(const arma::vec& theta, const arma::vec& phi_, arma::vec* grad_out, KModel* model) const {
    const double n = static_cast<double>(m_X.n_rows);
    const double q = static_cast<double>(Y.n_cols);
    const arma::uword d = m_X.n_cols;

    KModel m_local;
    if (model != nullptr)
      populate_Model(*model, theta, 1.0, KrigingImpl::ones, false, nullptr, &Y);
    else
      m_local = make_model(theta);
    KModel& m = (model != nullptr) ? *model : m_local;

    const arma::mat& E = m.Estar;
    auto [s2_, Lt] = kernel_sigma2(E, phi_);
    const double logdetL = arma::sum(arma::log(m.L.diag()));
    const double logdetLt = arma::sum(arma::log(Lt.diag()));
    const double ll = -0.5 * (n * q * std::log(2 * arma::datum::pi * s2_) + 2 * q * logdetL + 2 * n * logdetLt + n * q);

    if (grad_out != nullptr) {
      grad_out->set_size(d + phi_.n_elem);
      const double sd = std::sqrt(s2_);
      if ((m.Rinv.memptr() == nullptr) || (arma::size(m.Rinv) != arma::size(m.L)))
        m.Rinv = LinearAlgebra::inv_sympd(m.L);
      // θ: x = R_x⁻¹ (Y − F B) whitened across outputs by (σ̂ L_t)⁻ᵀ
      arma::mat x = LinearAlgebra::solve_upper(m.L.t(), E);
      x = arma::trans(LinearAlgebra::solve_lower(Lt, x.t())) / sd;
      arma::vec term1(d, arma::fill::zeros);
      arma::vec term2(d, arma::fill::zeros);
      compute_ll_grad_theta_vecs(m.R, m.Rinv, x, theta, term1, term2);
      grad_out->head(d) = (term1 + q * term2) / 2.0;
      // φ: x_t = R_t⁻¹ E*ᵀ / σ̂ (q × n)
      const arma::mat Rt = tker->corr(phi_);
      const arma::mat xt = LinearAlgebra::solve_upper(Lt.t(), LinearAlgebra::solve_lower(Lt, E.t())) / sd;
      arma::vec t1, t2;
      tker->grad_terms(Rt, LinearAlgebra::inv_sympd(Lt), xt, phi_, t1, t2);
      grad_out->tail(phi_.n_elem) = (t1 + n * t2) / 2.0;
    }
    return ll;
  }

  // Leave-one-out mean squared error (Dubrule 1983), summed over outputs on
  // the internal (normalized) scale: Σ_j Σ_i e_ij² / n. Gradient in θ as in
  // Kriging::_leaveOneOut, column by column.
  double leaveOneOutObj(const arma::vec& theta, arma::vec* grad_out, KModel* model) const {
    const arma::uword n = m_X.n_rows;
    const arma::uword d = m_X.n_cols;
    KModel m_local;
    if (model != nullptr)
      populate_Model(*model, theta, 1.0, KrigingImpl::ones, false, nullptr, &Y);
    else
      m_local = make_model(theta);
    KModel& m = (model != nullptr) ? *model : m_local;

    if ((m.Linv.memptr() == nullptr) || (arma::size(m.Linv) != arma::size(m.L)))
      m.Linv = LinearAlgebra::solve_lower(m.L, arma::mat(n, n, arma::fill::eye));
    const arma::mat By = m.Linv.t() * m.Estar;  // n × q
    arma::mat Qstar, Rtmp;
    LinearAlgebra::qr_econ(Qstar, Rtmp, m.Fstar);
    const arma::mat A = Qstar.t() * m.Linv;
    const arma::mat H = LinearAlgebra::crossprod(m.Linv) - LinearAlgebra::crossprod(A);
    const arma::vec s2loo = 1.0 / H.diag();
    arma::mat err = By;
    err.each_col() %= s2loo;
    const double loo = arma::accu(err % err) / n;

    if (grad_out != nullptr) {
      grad_out->set_size(d);
      arma::mat gradR_k(n, n);
      for (arma::uword k = 0; k < d; ++k) {
        gradR_k.zeros();
        for (arma::uword i = 0; i < n; ++i)
          for (arma::uword j = 0; j < i; ++j) {
            const double g = m.R.at(i, j) * _DlnCovDtheta(m_dX.col(i * n + j), theta).at(k);
            gradR_k.at(i, j) = g;
            gradR_k.at(j, i) = g;
          }
        const arma::vec diagdH = -LinearAlgebra::diagABA(H, gradR_k);
        const arma::vec ds2loo = -s2loo % s2loo % diagdH;
        arma::mat derr = By;
        derr.each_col() %= ds2loo;
        arma::mat t2 = H * (gradR_k * By);
        t2.each_col() %= s2loo;
        derr -= t2;
        (*grad_out)(k) = 2 * arma::accu(err % derr) / n;
      }
    }
    return loo;
  }

  // Lower Cholesky factor of the estimated output covariance; refuses a
  // (numerically) singular estimate
  static arma::mat sigma_chol(const arma::mat& S) {
    arma::mat Ls;
    const bool ok = arma::chol(Ls, S, "lower");
    if (!ok || Ls.diag().min() < 1e-8 * Ls.diag().max())
      throw std::runtime_error(
          "MultiOutputKriging: the estimated output covariance of \"separable\" is singular (outputs nearly "
          "linearly dependent, e.g. smooth curves); use \"pca\" or \"separable(<kernel>)\"");
    return Ls;
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
    const arma::uword q = Y.n_cols;
    s2 = arma::vec(q, arma::fill::zeros);
    Sig = arma::mat(q, q, arma::fill::zeros);
    Lsig = arma::mat(q, q, arma::fill::zeros);
    if (kernel_form()) {
      auto [s2_, Lt] = kernel_sigma2(Z, phi);
      Sig = s2_ * tker->corr(phi);
      Lsig = std::sqrt(s2_) * Lt;
      s2 = Sig.diag();
    } else if (full) {
      const arma::mat Za = Z.cols(active);
      const arma::mat S = Za.t() * Za / n;
      Sig.submat(active, active) = S;
      Lsig.submat(active, active) = sigma_chol(S);
      s2 = Sig.diag();
    } else {
      s2.elem(active) = arma::trans(arma::sum(arma::square(Z.cols(active)), 0)) / n;
      Sig.diag() = s2;
      Lsig.diag() = arma::sqrt(s2);
    }
    m_is_empty = false;
  }

  void fit(const arma::mat& Yraw,
           const arma::mat& X,
           const Trend::RegressionModel& regmodel,
           bool normalize,
           const std::string& optim,
           const std::string& objective,
           const Parameters& parameters) {
    if (objective != "LL" && objective != "LOO")
      throw std::invalid_argument("MultiOutputKriging: objective '" + objective
                                  + "' is not supported by the \"shared\" output model (LL or LOO)");
    const bool loo = objective == "LOO";
    if (loo && kernel_form())
      throw std::invalid_argument("MultiOutputKriging: objective 'LOO' does not depend on the output kernel; use 'LL'");
    const arma::uword n = X.n_rows;
    const arma::uword d = X.n_cols;
    const arma::uword dt = kernel_form() ? tker->dim() : 0;
    const arma::uword D = d + dt;  // optimized parameters: θ, then φ
    m_optim = optim;
    m_objective = objective;
    m_is_empty = true;
    clear_sim();

    normalize_outputs(Yraw, normalize);
    arma::mat theta0 = fit_setup_X_impl(X, regmodel, normalize, parameters.theta);
    set_active(Yraw);
    if (sigma_form == SigmaForm::Full && active.n_elem > n - m_F.n_cols)
      throw std::invalid_argument("MultiOutputKriging: \"separable\" estimates a free " + std::to_string(active.n_elem)
                                  + " x " + std::to_string(active.n_elem) + " output covariance, which needs n - p >= q"
                                  + " (here n - p = " + std::to_string(n - m_F.n_cols)
                                  + "); use \"pca\" or \"separable(<kernel>)\"");
    m_est_beta = true;
    m_est_sigma2 = true;

    if (parameters.output_theta.has_value()) {
      if (!kernel_form())
        throw std::invalid_argument(
            "MultiOutputKriging: output_theta is only used by the \"separable(<kernel>)\" output model");
      if (parameters.output_theta->n_cols != dt)
        throw std::invalid_argument("MultiOutputKriging: output_theta must have " + std::to_string(dt)
                                    + " columns (one per output coordinate)");
    }

    if (optim == "none" || !parameters.is_theta_estim) {
      if (!parameters.theta.has_value() || (kernel_form() && !parameters.output_theta.has_value()))
        throw std::invalid_argument("MultiOutputKriging: theta (1 x " + std::to_string(d) + ")"
                                    + (kernel_form() ? " and output_theta (1 x " + std::to_string(dt) + ")" : "")
                                    + " must be given when optim = \"none\" or is_theta_estim = false");
      m_theta = trans(theta0.row(0));
      if (kernel_form())
        phi = trans(parameters.output_theta->row(0));
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
    const int multistart_parsed = Optim::parse_method(optim, "BFGS").second;
    // given starting points first, then random ones, up to the larger count
    auto starts = [](const std::optional<arma::mat>& given, const arma::mat& rand, int count) {
      arma::mat out = given.has_value() ? arma::mat(arma::join_cols(*given, rand)) : rand;
      return arma::mat(out.rows(0, count - 1));
    };
    int multistart = multistart_parsed;
    if (parameters.theta.has_value())
      multistart = std::max(multistart, static_cast<int>(theta0.n_rows));
    if (parameters.output_theta.has_value())
      multistart = std::max(multistart, static_cast<int>(parameters.output_theta->n_rows));
    const arma::mat theta0_rand = arma::repmat(trans(theta_lower), multistart_parsed, 1)
                                  + Random::randu_mat(multistart_parsed, d)
                                        % arma::repmat(trans(theta_upper - theta_lower), multistart_parsed, 1);
    theta0 = starts(parameters.theta.has_value() ? std::optional<arma::mat>(theta0) : std::nullopt,
                    arma::join_cols(theta0_rand, arma::repmat(theta0_rand.row(0), multistart, 1)),
                    multistart);
    if (kernel_form()) {
      // φ: bounds from the ranges of t, starts drawn after those of θ
      auto [phi_lower, phi_upper] = tker->bounds();
      const arma::mat phi0_rand = arma::repmat(trans(phi_lower), multistart_parsed, 1)
                                  + Random::randu_mat(multistart_parsed, dt)
                                        % arma::repmat(trans(phi_upper - phi_lower), multistart_parsed, 1);
      theta0 = arma::join_rows(theta0,
                               starts(parameters.output_theta,
                                      arma::join_cols(phi0_rand, arma::repmat(phi0_rand.row(0), multistart, 1)),
                                      multistart));
      theta_lower = arma::join_cols(theta_lower, phi_lower);
      theta_upper = arma::join_cols(theta_upper, phi_upper);
    }

    auto to_gamma = [](const arma::vec& t) { return Optim::reparametrize ? Optim::reparam_to(t) : t; };
    auto from_gamma = [](const arma::vec& g) { return Optim::reparametrize ? Optim::reparam_from(g) : g; };
    // minimized in γ (θ, then φ): −LL, or the LOO mean squared error
    auto ofn = [&](const arma::vec& gamma, arma::vec* grad_out, KModel* m) -> double {
      const arma::vec theta = from_gamma(gamma);
      const double sign = loo ? 1.0 : -1.0;
      double f = loo             ? leaveOneOutObj(theta, grad_out, m)
                 : kernel_form() ? logLikelihoodKernel(theta.head(d), theta.tail(dt), grad_out, m)
                                 : logLikelihood(theta, grad_out, m);
      if (grad_out != nullptr)
        *grad_out = sign * (Optim::reparametrize ? arma::vec(Optim::reparam_from_deriv(theta, *grad_out)) : *grad_out);
      return sign * f;
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
        lbfgsb::Optimizer optimizer{static_cast<unsigned int>(D)};
        optimizer.iprint = -1;
        optimizer.max_iter = Optim::max_iteration;
        // tolerances scaled by n² for LOO, as in Kriging
        const double tol_scale = loo ? static_cast<double>(n * n) : 1.0;
        optimizer.pgtol = Optim::gradient_tolerance / tol_scale;
        optimizer.factr = Optim::objective_rel_tolerance / 1E-13 / tol_scale;
        arma::ivec bounds_type{D, arma::fill::value(2)};

        // same restart policy as Kriging
        double start_best = std::numeric_limits<double>::infinity();
        arma::vec start_best_gamma = gamma;
        for (int retry = 0; retry <= Optim::max_restart; ++retry) {
          auto res
              = optimizer.minimize([&](const arma::vec& g, arma::vec& grad) -> double { return ofn(g, &grad, &m); },
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
      throw std::runtime_error("MultiOutputKriging: all " + std::to_string(multistart) + " optimization starts failed ("
                               + last_error + ")");

    m_theta = best_theta.head(d);
    if (kernel_form())
      phi = best_theta.tail(dt);
    m_est_theta = true;
    KModel m = make_model(m_theta);
    commit(m);
  }

  /// S Σ S: output covariance on the original scale
  arma::mat output_cov_raw() const {
    arma::mat C = Sig;
    C.each_row() %= scaleY;
    C.each_col() %= scaleY.t();
    return C;
  }

  /// Draws Σ^{1/2}-mixed across outputs: E[j] are independent (rows × nsim)
  /// unit draws; returns output j's noise. "shared": σ_j E[j].
  arma::mat mix(const std::vector<arma::mat>& E, arma::uword j) const {
    if (!full)
      return E[j] * std::sqrt(s2(j));
    arma::mat out = E[j] * Lsig(j, j);
    for (arma::uword k = 0; k < j; ++k)
      if (Lsig(j, k) != 0)
        out += E[k] * Lsig(j, k);
    return out;
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
      cov = arma::kron(output_cov_raw(), Sigma);  // over vec(Y_n); block-diagonal for "shared"
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

  void clear_sim() {
    sim_X.reset();
    sim_Yn.reset();
  }

  // Joint conditional distribution at raw points X_p (P × d), normalized
  // scale and unit variance: mean (P × q) and correlation Σ (P × P); output j
  // has covariance σ_j² Σ.
  std::pair<arma::mat, arma::mat> conditional(const arma::mat& X_p) const {
    const arma::uword P = X_p.n_rows;
    auto [Xn, F_n] = normalize_X(X_p);
    const arma::mat R_on = cross_corr(arma::trans(m_X), Xn, 1.0, false);
    const arma::mat Rstar_on = LinearAlgebra::solve_lower(m_T, R_on);
    arma::mat yhat = F_n * B + Rstar_on.t() * Z;
    const arma::mat Ecirc_n = LinearAlgebra::rsolve_upper(m_circ, F_n - Rstar_on.t() * m_M);
    arma::mat R_nn(P, P, arma::fill::none);
    LinearAlgebra::covMat_sym_X(&R_nn, Xn, m_theta, _Cov, 1.0, KrigingImpl::ones);
    arma::mat Sigma = R_nn - Rstar_on.t() * Rstar_on + Ecirc_n * Ecirc_n.t();
    return {std::move(yhat), std::move(Sigma)};
  }

  arma::cube simulate(int nsim, int seed, const arma::mat& X_n, bool will_update) {
    const arma::uword m = X_n.n_rows;
    const arma::uword q = Y.n_cols;
    auto [yhat, Sigma] = conditional(X_n);
    const arma::mat L = LinearAlgebra::safe_chol_lower(Sigma);

    // outputs drawn in turn from one seeded stream (q = 1 matches Kriging)
    arma::cube yn(m, q, nsim);
    Random::reset_seed(seed);
    std::vector<arma::mat> E(q);
    for (arma::uword j = 0; j < q; ++j)
      E[j] = L * Random::randn_mat(m, nsim);
    for (arma::uword j = 0; j < q; ++j) {
      arma::mat y_n = mix(E, j);
      y_n.each_col() += yhat.col(j);
      for (int s = 0; s < nsim; ++s)
        yn.slice(s).col(j) = y_n.col(s);
    }
    if (will_update) {
      sim_X = X_n;
      sim_Yn = yn;
      sim_seed = seed;
    } else {
      clear_sim();
    }
    return to_raw(std::move(yn));
  }

  arma::cube to_raw(arma::cube yn) const {
    for (arma::uword s = 0; s < yn.n_slices; ++s) {
      yn.slice(s).each_row() %= scaleY;
      yn.slice(s).each_row() += centerY;
    }
    return yn;
  }

  /** Condition the last simulate() draws on new data (X_u, Y_u), model kept:
   * with Σ the joint conditional correlation of (Y(X_n), Y(X_u)) given the
   * data, for each draw y_n
   *   1. draw y_u ~ N(μ_u + Σ_un Σ_nn⁻¹ (y_n − μ_n), σ² (Σ_uu − Σ_un Σ_nn⁻¹ Σ_nu)),
   *      so (y_n, y_u) is a joint draw;
   *   2. y_n ← y_n + Σ_nu Σ_uu⁻¹ (Y_u − y_u)   (conditioning by kriging).
   * Exact for fixed θ, β integrated out as in universal kriging. */
  arma::cube update_simulate(const arma::mat& Y_u, const arma::mat& X_u) const {
    if (sim_Yn.n_elem == 0)
      throw std::runtime_error("MultiOutputKriging::update_simulate: call simulate(..., will_update=true) first");
    const arma::uword m = sim_X.n_rows;
    const arma::uword u = X_u.n_rows;
    const arma::uword q = Y.n_cols;
    const arma::uword nsim = sim_Yn.n_slices;

    auto [mu, S] = conditional(arma::join_cols(sim_X, X_u));
    const arma::mat S_nn = S.submat(0, 0, m - 1, m - 1);
    const arma::mat S_nu = S.submat(0, m, m - 1, m + u - 1);
    const arma::mat S_uu = S.submat(m, m, m + u - 1, m + u - 1);

    const arma::mat L_nn = LinearAlgebra::safe_chol_lower(S_nn);
    const arma::mat A = LinearAlgebra::solve_lower(L_nn, S_nu);              // m × u
    const arma::mat W = LinearAlgebra::solve_upper(L_nn.t(), A);             // Σ_nn⁻¹ Σ_nu
    const arma::mat L_u = LinearAlgebra::safe_chol_lower(S_uu - A.t() * A);  // Σ_u|n
    const arma::mat L_uu = LinearAlgebra::safe_chol_lower(S_uu);
    const arma::mat Lambda
        = arma::trans(LinearAlgebra::solve_upper(L_uu.t(), LinearAlgebra::solve_lower(L_uu, S_nu.t())));  // m × u

    arma::mat Yn_u = Y_u.each_row() - centerY;
    Yn_u.each_row() /= scaleY;

    arma::cube out(m, q, nsim);
    Random::reset_seed(sim_seed + 1);
    std::vector<arma::mat> Eu(q);
    for (arma::uword j = 0; j < q; ++j)
      Eu[j] = L_u * Random::randn_mat(u, nsim);
    for (arma::uword j = 0; j < q; ++j) {
      arma::mat yn(m, nsim);
      for (arma::uword s = 0; s < nsim; ++s)
        yn.col(s) = sim_Yn.slice(s).col(j);
      arma::mat dev = yn.each_col() - mu.col(j).head(m);
      arma::mat yu = W.t() * dev + mix(Eu, j);
      yu.each_col() += mu.col(j).tail(u);
      arma::mat resid = -yu;  // Y_u − y_u, per draw
      resid.each_col() += Yn_u.col(j);
      const arma::mat upd = yn + Lambda * resid;
      for (arma::uword s = 0; s < nsim; ++s)
        out.slice(s).col(j) = upd.col(s);
    }
    return to_raw(std::move(out));
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
    clear_sim();
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

  // save / load: normalized data and fitted (θ, φ); the factorization is
  // rebuilt on load exactly as at the end of fit / update
  void dump(nlohmann::json& j) const {
    j["X"] = to_json(m_X);
    j["centerX"] = to_json(m_centerX);
    j["scaleX"] = to_json(m_scaleX);
    j["Y"] = to_json(Y);
    j["centerY"] = to_json(centerY);
    j["scaleY"] = to_json(scaleY);
    j["active"] = std::vector<arma::uword>(active.begin(), active.end());
    j["normalize"] = m_normalize;
    j["regmodel"] = Trend::toString(m_regmodel);
    j["optim"] = m_optim;
    j["objective"] = m_objective;
    j["theta"] = to_json(m_theta);
    j["est_theta"] = m_est_theta;
    if (kernel_form())
      j["output_theta"] = to_json(phi);
  }

  void restore(const nlohmann::json& j) {
    m_X = mat_from_json(j["X"]);
    m_centerX = rowvec_from_json(j["centerX"]);
    m_scaleX = rowvec_from_json(j["scaleX"]);
    Y = mat_from_json(j["Y"]);
    centerY = rowvec_from_json(j["centerY"]);
    scaleY = rowvec_from_json(j["scaleY"]);
    active = arma::uvec(j["active"].template get<std::vector<arma::uword>>());
    m_normalize = j["normalize"].template get<bool>();
    m_regmodel = Trend::fromString(j["regmodel"].template get<std::string>());
    m_optim = j["optim"].template get<std::string>();
    m_objective = j["objective"].template get<std::string>();
    m_theta = colvec_from_json(j["theta"]);
    m_est_theta = j["est_theta"].template get<bool>();
    if (kernel_form())
      phi = colvec_from_json(j["output_theta"]);
    m_dX = LinearAlgebra::compute_dX(m_X);
    m_maxdX = arma::max(arma::abs(m_dX), 1);
    m_F = Trend::regressionModelMatrix(m_regmodel, m_X);
    m_est_beta = true;
    m_est_sigma2 = true;
    KModel m = make_model(m_theta);
    commit(m);
  }

  std::string summary() const {
    std::ostringstream os;
    os << "  * theta: " << arma::trans(m_theta);
    if (kernel_form())
      os << "  * sigma2: " << s2(0) << "\n";
    else
      os << "  * sigma2: " << arma::trans(s2);
    if (kernel_form()) {
      os << "  * output kernel: " << tker->kernel() << ", theta: " << arma::trans(phi);
    } else if (full) {
      const arma::vec sd = arma::sqrt(s2);
      arma::mat C = Sig;
      for (arma::uword i = 0; i < C.n_rows; ++i)
        for (arma::uword j = 0; j < C.n_cols; ++j)
          C(i, j) = (sd(i) > 0 && sd(j) > 0) ? C(i, j) / (sd(i) * sd(j)) : 0.0;
      os << "  * output correlation:\n" << C;
    }
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
    if (!arg.empty())
      check_output_kernel(arg);
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
  if (m_output_model == OutputModel::PCA)
    throw std::runtime_error("MultiOutputKriging::" + where
                             + ": only available with the \"shared\" and \"separable\" output models"
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

const arma::mat& MultiOutputKriging::output_cov() const {
  return shared("output_cov").Sig;
}

std::tuple<arma::mat, arma::mat> MultiOutputKriging::predictCovFactors(const arma::mat& X_n) {
  const SharedModel& sm = shared("predictCovFactors");
  if (X_n.n_cols != m_X.n_cols)
    throw std::invalid_argument("MultiOutputKriging::predictCovFactors: X_n has " + std::to_string(X_n.n_cols)
                                + " columns, expected " + std::to_string(m_X.n_cols));
  arma::mat C = sm.conditional(X_n).second;
  return std::make_tuple(std::move(C), sm.output_cov_raw());
}

double MultiOutputKriging::logLikelihood() {
  const SharedModel& sm = shared("logLikelihood");
  if (sm.kernel_form())
    return sm.logLikelihoodKernel(sm.theta_(), sm.phi, nullptr, nullptr);
  return sm.logLikelihood(sm.theta_(), nullptr, nullptr);
}

const arma::vec& MultiOutputKriging::output_theta() const {
  const SharedModel& sm = shared("output_theta");
  if (!sm.kernel_form())
    throw std::runtime_error("MultiOutputKriging::output_theta: only available with \"separable(<kernel>)\"");
  return sm.phi;
}

std::tuple<double, arma::vec> MultiOutputKriging::leaveOneOutFun(const arma::vec& theta, bool grad) {
  const SharedModel& sm = shared("leaveOneOutFun");
  if (theta.n_elem != m_X.n_cols)
    throw std::invalid_argument("MultiOutputKriging::leaveOneOutFun: theta must have " + std::to_string(m_X.n_cols)
                                + " elements");
  arma::vec g;
  const double loo = sm.leaveOneOutObj(theta, grad ? &g : nullptr, nullptr);
  return std::make_tuple(loo, std::move(g));
}

std::tuple<double, arma::vec> MultiOutputKriging::logLikelihoodFun(const arma::vec& theta, bool grad) {
  const SharedModel& sm = shared("logLikelihoodFun");
  const arma::uword d = m_X.n_cols;
  const arma::uword np = d + (sm.kernel_form() ? m_t.n_cols : 0);
  if (theta.n_elem != np)
    throw std::invalid_argument("MultiOutputKriging::logLikelihoodFun: theta must have " + std::to_string(np)
                                + " elements" + (sm.kernel_form() ? " (theta, then output_theta)" : ""));
  arma::vec g;
  if (sm.kernel_form()) {
    const double ll = sm.logLikelihoodKernel(theta.head(d), theta.tail(np - d), grad ? &g : nullptr, nullptr);
    return std::make_tuple(ll, std::move(g));
  }
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
  if (parameters.output_theta.has_value() && m_output_model != OutputModel::SeparableKernel)
    throw std::invalid_argument(
        "MultiOutputKriging: output_theta is only used by the \"separable(<kernel>)\" output model");
  if (m_output_model == OutputModel::SeparableKernel) {
    if (m_t.n_elem == 0)
      throw std::invalid_argument("MultiOutputKriging::fit: \"" + output_model_string()
                                  + "\" needs the output coordinates (q x d_t): call set_output_coordinates first");
    const arma::rowvec range = arma::max(m_t, 0) - arma::min(m_t, 0);
    if (q < 2 || range.min() <= 0)
      throw std::invalid_argument(
          "MultiOutputKriging::fit: output coordinates must vary along each of their columns (q >= 2)");
  }

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

  if (m_output_model != OutputModel::PCA) {
    using Form = SharedModel::SigmaForm;
    const Form form = m_output_model == OutputModel::Shared      ? Form::Diag
                      : m_output_model == OutputModel::Separable ? Form::Full
                                                                 : Form::Kernel;
    m_shared = std::make_unique<SharedModel>(m_covType, form, m_output_covType, m_t);
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

  if (m_shared)
    return m_shared->simulate(nsim, seed, X_n, will_update);

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
  if (Y_u.n_cols != m_Y.n_cols || Y_u.n_rows != X_u.n_rows || X_u.n_cols != m_X.n_cols)
    throw std::invalid_argument("MultiOutputKriging::update_simulate: Y_u must be n_u × q and X_u n_u × d");
  if (m_shared)
    return m_shared->update_simulate(Y_u, X_u);
  if (m_lastsim_residual.n_elem == 0)
    throw std::runtime_error("MultiOutputKriging::update_simulate: call simulate(..., will_update=true) first");

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

// =============================================================================
// save / load
// =============================================================================

void MultiOutputKriging::save(const std::string& filename) const {
  nlohmann::json j;
  j["version"] = 2;
  j["content"] = "MultiOutputKriging";
  j["covType"] = m_covType;
  j["output_model"] = output_model_string();
  j["pca_spec"] = m_pca_spec;
  j["regmodel"] = Trend::toString(m_regmodel);
  j["normalize"] = m_normalize;
  j["optim"] = m_optim;
  j["objective"] = m_objective;
  nlohmann::json jp;
  jp["is_theta_estim"] = m_parameters.is_theta_estim;
  if (m_parameters.theta.has_value())
    jp["theta"] = to_json(*m_parameters.theta);
  if (m_parameters.output_theta.has_value())
    jp["output_theta"] = to_json(*m_parameters.output_theta);
  j["parameters"] = jp;
  j["X"] = to_json(m_X);
  j["Y"] = to_json(m_Y);
  j["output_coordinates"] = to_json(m_t);
  j["centerY"] = to_json(m_centerY);
  j["scaleY"] = to_json(m_scaleY);
  j["fitted"] = !m_components.empty() || (m_shared && m_shared->fitted());
  if (m_shared && m_shared->fitted()) {
    nlohmann::json js;
    m_shared->dump(js);
    j["shared"] = js;
  }
  if (!m_components.empty()) {
    j["pca_basis"] = to_json(m_pca_basis);
    j["pca_explained"] = to_json(m_pca_explained);
    j["pca_residual"] = to_json(m_pca_residual);
    nlohmann::json jc = nlohmann::json::array();
    for (const auto& km : m_components) {
      nlohmann::json jk;
      km->dump_to_json(jk);
      jc.push_back(jk);
    }
    j["pca_components"] = jc;  // sorts after "content" (tools that scan for the first "content")
  }
  std::ofstream f(filename);
  if (!f)
    throw std::runtime_error("MultiOutputKriging::save: cannot write '" + filename + "'");
  f << std::setw(4) << j;
}

MultiOutputKriging MultiOutputKriging::load(const std::string& filename) {
  std::ifstream f(filename);
  if (!f)
    throw std::runtime_error("MultiOutputKriging::load: cannot read '" + filename + "'");
  nlohmann::json j = nlohmann::json::parse(f);
  const std::string content = j.contains("content") ? j["content"].template get<std::string>() : "";
  if (content != "MultiOutputKriging")
    throw std::runtime_error("MultiOutputKriging::load: bad content in '" + filename + "'; found '" + content
                             + "', requires 'MultiOutputKriging'");
  const auto version = j["version"].template get<uint32_t>();
  if (version != 2)
    throw std::runtime_error("MultiOutputKriging::load: bad version in '" + filename + "'; found "
                             + std::to_string(version) + ", requires 2");

  MultiOutputKriging mo(j["covType"].template get<std::string>(), j["output_model"].template get<std::string>());
  mo.m_pca_spec = j["pca_spec"].template get<double>();
  mo.m_regmodel = Trend::fromString(j["regmodel"].template get<std::string>());
  mo.m_normalize = j["normalize"].template get<bool>();
  mo.m_optim = j["optim"].template get<std::string>();
  mo.m_objective = j["objective"].template get<std::string>();
  const nlohmann::json& jp = j["parameters"];
  mo.m_parameters.is_theta_estim = jp["is_theta_estim"].template get<bool>();
  if (jp.contains("theta"))
    mo.m_parameters.theta = mat_from_json(jp["theta"]);
  if (jp.contains("output_theta"))
    mo.m_parameters.output_theta = mat_from_json(jp["output_theta"]);
  mo.m_X = mat_from_json(j["X"]);
  mo.m_Y = mat_from_json(j["Y"]);
  mo.m_t = mat_from_json(j["output_coordinates"]);
  mo.m_centerY = rowvec_from_json(j["centerY"]);
  mo.m_scaleY = rowvec_from_json(j["scaleY"]);

  if (j.contains("shared")) {
    using Form = SharedModel::SigmaForm;
    const Form form = mo.m_output_model == OutputModel::Shared      ? Form::Diag
                      : mo.m_output_model == OutputModel::Separable ? Form::Full
                                                                    : Form::Kernel;
    mo.m_shared = std::make_unique<SharedModel>(mo.m_covType, form, mo.m_output_covType, mo.m_t);
    mo.m_shared->restore(j["shared"]);
  }
  if (j.contains("pca_components")) {
    mo.m_pca_basis = mat_from_json(j["pca_basis"]);
    mo.m_pca_explained = colvec_from_json(j["pca_explained"]);
    mo.m_pca_residual = mat_from_json(j["pca_residual"]);
    for (const auto& jk : j["pca_components"]) {
      const Kriging::NoiseModel nm = Kriging::noise_model_from_json(jk, filename);
      auto km = std::make_unique<Kriging>(jk["covType"].template get<std::string>(), nm);
      km->load_from_json(jk);
      mo.m_components.push_back(std::move(km));
    }
  }
  return mo;
}
