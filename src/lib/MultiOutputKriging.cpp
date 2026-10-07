#include "libKriging/MultiOutputKriging.hpp"

#include <cmath>
#include <random>
#include <sstream>
#include <stdexcept>

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
  if (m_components.empty())
    throw std::runtime_error("MultiOutputKriging::" + where + ": model is not fitted");
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
  if (m_output_model != OutputModel::PCA)
    throw std::runtime_error("MultiOutputKriging: output model '" + output_model_string()
                             + "' is not implemented yet (only \"pca\")");

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
  if (m_components.empty()) {
    os << "  (not fitted)\n";
    return os.str();
  }
  os << "  * data: " << m_X.n_rows << " x " << m_X.n_cols << " -> " << m_Y.n_rows << " x " << m_Y.n_cols << "\n";
  os << "  * trend: " << Trend::toString(m_regmodel) << (m_normalize ? " (normalized outputs)" : "") << "\n";
  os << "  * PCA components: " << m_components.size()
     << ", explained variance: " << m_pca_explained(m_pca_explained.n_elem - 1) << "\n";
  for (arma::uword k = 0; k < m_components.size(); ++k) {
    const Kriging& km = *m_components[k];
    os << "    - component " << k << ": cum. explained " << m_pca_explained(k) << ", sigma2 " << km.sigma2()
       << ", theta " << arma::trans(km.theta());
  }
  return os.str();
}
