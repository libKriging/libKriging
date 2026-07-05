#include "libKriging/NestedKriging.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

// =============================================================================
// static helpers
// =============================================================================

NestedKriging::Aggregation NestedKriging::aggregationFromString(const std::string& s) {
  if (s == "PoE")
    return Aggregation::PoE;
  if (s == "gPoE")
    return Aggregation::gPoE;
  if (s == "BCM")
    return Aggregation::BCM;
  if (s == "rBCM")
    return Aggregation::rBCM;
  if (s == "NK")
    return Aggregation::NK;
  throw std::invalid_argument("Unknown aggregation: " + s + " (expected PoE, gPoE, BCM, rBCM or NK)");
}

std::string NestedKriging::aggregationToString(NestedKriging::Aggregation a) {
  switch (a) {
    case Aggregation::PoE:
      return "PoE";
    case Aggregation::gPoE:
      return "gPoE";
    case Aggregation::BCM:
      return "BCM";
    case Aggregation::rBCM:
      return "rBCM";
    case Aggregation::NK:
      return "NK";
  }
  return "?";
}

// =============================================================================
// construction
// =============================================================================

NestedKriging::NestedKriging(const std::string& covType) : m_covType(covType) {
  m_corr = Covariance::resolve(covType).Cov;
}

NestedKriging::NestedKriging(const arma::vec& y,
                             const arma::mat& X,
                             const std::string& covType,
                             arma::uword nb_groups,
                             Aggregation aggregation,
                             Partition partition,
                             int seed,
                             const Trend::RegressionModel& regmodel,
                             const std::string& optim,
                             const std::string& objective,
                             const Kriging::Parameters& parameters,
                             const std::vector<std::string>& warping)
    : NestedKriging(covType) {
  m_aggregation = aggregation;
  m_partition_method = partition;
  m_seed = seed;
  fit(y, X, nb_groups, regmodel, optim, objective, parameters, warping);
}

// =============================================================================
// submodel accessors
// =============================================================================

const Kriging& NestedKriging::submodel(arma::uword i) const {
  if (warped())
    throw std::runtime_error("submodel(): model uses WarpKriging submodels, use wsubmodel()");
  return *m_submodels.at(i);
}

const libKriging::WarpKriging& NestedKriging::wsubmodel(arma::uword i) const {
  if (!warped())
    throw std::runtime_error("wsubmodel(): model uses Kriging submodels, use submodel()");
  return *m_wsubmodels.at(i);
}

// =============================================================================
// correlation helper (common prior)
// =============================================================================

arma::mat NestedKriging::corrMat(const arma::mat& X1, const arma::mat& X2) const {
  if (warped()) {
    // warped prior: k(Phi(x), Phi(x')) via the reference submodel's public
    // covMat (all submodels share (theta, warp) after unification)
    const libKriging::WarpKriging& ref = *m_wsubmodels[m_ref];
    return ref.covMat(X1, X2) / ref.sigma2();
  }
  arma::mat R(X1.n_rows, X2.n_rows);
  for (arma::uword i = 0; i < X1.n_rows; ++i) {
    const arma::rowvec x1 = X1.row(i);
    for (arma::uword j = 0; j < X2.n_rows; ++j) {
      R(i, j) = m_corr((x1 - X2.row(j)).t(), m_theta);
    }
  }
  return R;
}

// =============================================================================
// partition
// =============================================================================

void NestedKriging::make_partition(arma::uword nb_groups) {
  const arma::uword n = m_X.n_rows;
  const arma::uword d = m_X.n_cols;
  const arma::uword min_group_size = d + 2;  // enough points to identify theta

  arma::arma_rng::set_seed(static_cast<arma::arma_rng::seed_type>(m_seed));

  arma::uvec assignment(n);
  bool kmeans_ok = false;

  if (m_partition_method == Partition::KMeans && nb_groups > 1) {
    arma::mat centroids;
    // arma::kmeans expects d x n data
    kmeans_ok = arma::kmeans(centroids, m_X.t(), nb_groups, arma::random_subset, 10, false);
    if (kmeans_ok) {
      for (arma::uword i = 0; i < n; ++i) {
        arma::vec dist2(nb_groups);
        for (arma::uword g = 0; g < nb_groups; ++g)
          dist2(g) = arma::accu(arma::square(m_X.row(i).t() - centroids.col(g)));
        assignment(i) = dist2.index_min();
      }
      // check group sizes; degenerate k-means (empty/tiny clusters) => fallback
      for (arma::uword g = 0; g < nb_groups; ++g) {
        if (arma::accu(assignment == g) < min_group_size) {
          kmeans_ok = false;
          break;
        }
      }
    }
  }

  if (!kmeans_ok) {  // Partition::Random, or k-means fallback: balanced random split
    arma::uvec perm = arma::randperm(n);
    for (arma::uword i = 0; i < n; ++i)
      assignment(perm(i)) = i % nb_groups;
  }

  m_groups.clear();
  m_groups.reserve(nb_groups);
  for (arma::uword g = 0; g < nb_groups; ++g)
    m_groups.push_back(arma::find(assignment == g));
}

// =============================================================================
// fit
// =============================================================================

void NestedKriging::fit(const arma::vec& y,
                        const arma::mat& X,
                        arma::uword nb_groups,
                        const Trend::RegressionModel& regmodel,
                        const std::string& optim,
                        const std::string& objective,
                        const Kriging::Parameters& parameters,
                        const std::vector<std::string>& warping) {
  if (y.n_elem != X.n_rows)
    throw std::invalid_argument("y and X should have the same number of rows");
  if (nb_groups < 1 || nb_groups > X.n_rows / (X.n_cols + 2))
    throw std::invalid_argument("nb_groups should be in [1, n/(d+2)]");
  if (m_aggregation == Aggregation::NK && regmodel != Trend::RegressionModel::Constant)
    throw std::invalid_argument("NK aggregation requires a Constant trend; use PoE/gPoE/BCM/rBCM otherwise");

  m_X = X;
  m_y = y;
  m_regmodel = regmodel;
  m_warping = warping;

  make_partition(nb_groups);
  const arma::uword p = m_groups.size();

  // --- 1. independent submodel fits ------------------------------------------
  // NOTE: kept sequential for now; Kriging::fit thread-safety (optimizer &
  // arma RNG state) must be audited before adding `#pragma omp parallel for`.
  m_submodels.clear();
  m_wsubmodels.clear();
  if (warped()) {
    m_wsubmodels.reserve(p);
    for (arma::uword g = 0; g < p; ++g) {
      const arma::uvec& idx = m_groups[g];
      auto wk = std::make_unique<libKriging::WarpKriging>(m_warping, m_covType);
      wk->fit(m_y(idx), m_X.rows(idx), regmodel, /*normalize=*/false, optim, objective, libKriging::WarpKriging::Parameters{});
      m_wsubmodels.push_back(std::move(wk));
    }
  } else {
    m_submodels.reserve(p);
    for (arma::uword g = 0; g < p; ++g) {
      const arma::uvec& idx = m_groups[g];
      auto km = std::make_unique<Kriging>(m_covType);
      km->fit(m_y(idx), m_X.rows(idx), regmodel, /*normalize=*/false, optim, objective, parameters);
      m_submodels.push_back(std::move(km));
    }
  }

  // --- 2. unify hyperparameters (common prior) ------------------------------
  unify_hyperparameters(objective);

  // --- 3. NK precomputations -------------------------------------------------
  if (m_aggregation == Aggregation::NK)
    precompute_nk();

  m_is_fitted = true;
}

void NestedKriging::unify_hyperparameters(const std::string& objective) {
  const arma::uword p = m_groups.size();
  const arma::uword d = m_X.n_cols;
  const double n = static_cast<double>(m_X.n_rows);

  if (warped()) {
    // --- warped path: common (theta, warp) from the largest group -------------
    // Warp parameters (embeddings, MLP weights, ...) live on non-convex
    // manifolds: unlike theta they cannot be meaningfully averaged across
    // groups, so the whole (theta, warp) pair is taken from one reference fit.
    m_ref = 0;
    for (arma::uword g = 1; g < p; ++g)
      if (m_groups[g].n_elem > m_groups[m_ref].n_elem)
        m_ref = g;

    libKriging::WarpKriging::Parameters fixed;
    fixed.theta = m_wsubmodels[m_ref]->theta();
    fixed.warp_params = m_wsubmodels[m_ref]->warp_params();

    // refit every submodel with optim="none": keeps the seeded (theta, warp)
    // as-is; sigma2 and beta stay profiled per group given the common prior
    for (arma::uword g = 0; g < p; ++g) {
      const arma::uvec& idx = m_groups[g];
      m_wsubmodels[g]->fit(m_y(idx), m_X.rows(idx), m_regmodel, false, "none", objective, fixed);
    }

    m_theta = m_wsubmodels[m_ref]->theta();
    m_sigma2 = 0;
    m_beta0 = 0;
    for (arma::uword g = 0; g < p; ++g) {
      const double w = static_cast<double>(m_groups[g].n_elem) / n;
      m_sigma2 += w * m_wsubmodels[g]->sigma2();
      if (m_regmodel == Trend::RegressionModel::Constant)
        m_beta0 += w * m_wsubmodels[g]->beta()(0);
    }
    return;
  }

  // --- plain path: weighted geometric mean of thetas --------------------------
  arma::vec log_theta(d, arma::fill::zeros);
  double sigma2 = 0;
  double beta0 = 0;
  for (arma::uword g = 0; g < p; ++g) {
    const double w = static_cast<double>(m_groups[g].n_elem) / n;
    log_theta += w * arma::log(m_submodels[g]->theta());
    sigma2 += w * m_submodels[g]->sigma2();
    if (m_regmodel == Trend::RegressionModel::Constant)
      beta0 += w * m_submodels[g]->beta()(0);
  }
  m_theta = arma::exp(log_theta);
  m_sigma2 = sigma2;
  m_beta0 = beta0;

  // refit each submodel with the common (fixed) hyperparameters
  Kriging::Parameters fixed;
  fixed.theta = arma::mat(m_theta.t());  // 1 x d
  fixed.is_theta_estim = false;
  fixed.sigma2 = m_sigma2;
  fixed.is_sigma2_estim = false;
  if (m_regmodel == Trend::RegressionModel::Constant) {
    fixed.beta = arma::vec{m_beta0};
    fixed.is_beta_estim = false;
  }
  for (arma::uword g = 0; g < p; ++g) {
    const arma::uvec& idx = m_groups[g];
    m_submodels[g]->fit(m_y(idx), m_X.rows(idx), m_regmodel, false, "none", objective, fixed);
  }
}

void NestedKriging::precompute_nk() {
  const arma::uword p = m_groups.size();
  m_L.assign(p, arma::mat());
  m_alpha.assign(p, arma::vec());
  for (arma::uword g = 0; g < p; ++g) {
    const arma::uvec& idx = m_groups[g];
    arma::mat R = corrMat(m_X.rows(idx), m_X.rows(idx));
    R.diag() += jitter;
    m_L[g] = arma::chol(R, "lower");
    const arma::vec r = m_y(idx) - m_beta0;
    m_alpha[g] = arma::solve(arma::trimatu(m_L[g].t()), arma::solve(arma::trimatl(m_L[g]), r));
  }
}

// =============================================================================
// predict
// =============================================================================

std::tuple<arma::vec, arma::vec> NestedKriging::predict(const arma::mat& X_n, bool return_stdev) {
  if (!m_is_fitted)
    throw std::runtime_error("NestedKriging is not fitted");
  if (X_n.n_cols != m_X.n_cols)
    throw std::invalid_argument("X_n should have the same number of columns as X");

  if (m_aggregation == Aggregation::NK)
    return predict_nk(X_n, return_stdev);
  return predict_poe_family(X_n, return_stdev);
}

std::tuple<arma::vec, arma::vec> NestedKriging::predict_poe_family(const arma::mat& X_n, bool return_stdev) const {
  const arma::uword q = X_n.n_rows;
  const arma::uword p = m_groups.size();
  const double prior_var = m_sigma2;
  constexpr double tiny = 1e-12;

  arma::mat mus(q, p);
  arma::mat vars(q, p);
  for (arma::uword g = 0; g < p; ++g) {
    if (warped()) {
      auto [mu, sd, cov, dmu, dsd] = m_wsubmodels[g]->predict(X_n, true, false, false);
      mus.col(g) = mu;
      vars.col(g) = arma::square(sd) + tiny;
    } else {
      auto [mu, sd, cov, dmu, dsd] = m_submodels[g]->predict(X_n, true, false, false);
      mus.col(g) = mu;
      vars.col(g) = arma::square(sd) + tiny;
    }
  }

  arma::vec mean(q);
  arma::vec var(q);
  switch (m_aggregation) {
    case Aggregation::PoE: {
      const arma::vec prec = arma::sum(1.0 / vars, 1);
      mean = arma::sum(mus / vars, 1) / prec;
      var = 1.0 / prec;
    } break;
    case Aggregation::gPoE: {  // beta_g = 1/p : calibrated PoE
      const arma::vec prec = arma::sum(1.0 / vars, 1) / static_cast<double>(p);
      mean = (arma::sum(mus / vars, 1) / static_cast<double>(p)) / prec;
      var = 1.0 / prec;
    } break;
    case Aggregation::BCM: {
      arma::vec prec = arma::sum(1.0 / vars, 1) - (static_cast<double>(p) - 1.0) / prior_var;
      prec = arma::clamp(prec, tiny, arma::datum::inf);
      mean = arma::sum(mus / vars, 1) / prec;
      var = 1.0 / prec;
    } break;
    case Aggregation::rBCM: {
      const arma::mat beta = 0.5 * (std::log(prior_var) - arma::log(vars));
      arma::vec prec = arma::sum(beta / vars, 1) + (1.0 - arma::sum(beta, 1)) / prior_var;
      prec = arma::clamp(prec, tiny, arma::datum::inf);
      mean = arma::sum(beta % mus / vars, 1) / prec;
      var = 1.0 / prec;
    } break;
    default:
      throw std::logic_error("unreachable");
  }

  if (!return_stdev)
    return {mean, arma::vec()};
  return {mean, arma::sqrt(arma::clamp(var, 0.0, arma::datum::inf))};
}

/* NK optimal aggregation (Rullière et al. 2018), simple kriging on the
 * submodel predictors, per prediction point x:
 *   M_g(x)            = beta0 + r_g(x)' R_g^{-1} (y_g - beta0)
 *   cov(M_i, M_j)/s2  = (R_i^{-1} r_i)' R_ij (R_j^{-1} r_j)  (i != j)
 *   cov(Y, M_g)/s2    = cov(M_g, M_g)/s2 = r_g' R_g^{-1} r_g = k_g
 *   =>  K_M (p x p), k_M = diag(K_M)
 *   mean = beta0 + k_M' K_M^{-1} (M - beta0)
 *   var  = s2 * (1 - k_M' K_M^{-1} k_M)
 * All correlations come from corrMat (plain or warped common prior).
 * Whitened weights per chunk: U_g = L_g \ C_g, W_g = L_g' \ U_g,
 * so k_g = colsum(U_g^2) and cross(i,j) = colsum(W_i % (R_ij W_j)). */
std::tuple<arma::vec, arma::vec> NestedKriging::predict_nk(const arma::mat& X_n, bool return_stdev) const {
  const arma::uword q = X_n.n_rows;
  const arma::uword p = m_groups.size();

  arma::vec mean(q);
  arma::vec var(return_stdev ? q : 0);

  for (arma::uword start = 0; start < q; start += m_chunk) {
    const arma::uword stop = std::min(start + m_chunk, q) - 1;
    const arma::mat Xc = X_n.rows(start, stop);
    const arma::uword qc = Xc.n_rows;

    // per-group whitened correlation vectors
    std::vector<arma::mat> C(p), W(p);
    arma::mat M(p, qc);      // submodel means
    arma::mat Kdiag(p, qc);  // k_g(x)
    for (arma::uword g = 0; g < p; ++g) {
      C[g] = corrMat(m_X.rows(m_groups[g]), Xc);  // n_g x qc
      const arma::mat U = arma::solve(arma::trimatl(m_L[g]), C[g]);
      W[g] = arma::solve(arma::trimatu(m_L[g].t()), U);
      M.row(g) = m_beta0 + (C[g].t() * m_alpha[g]).t();
      Kdiag.row(g) = arma::sum(U % U, 0);
    }

    // cross-covariances between submodel predictors, parallel over pairs
    arma::cube cross(p, p, qc, arma::fill::zeros);
    const arma::sword npairs = static_cast<arma::sword>(p * (p - 1) / 2);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (arma::sword k = 0; k < npairs; ++k) {
      // unrank pair index k -> (i, j), i < j
      arma::uword i = 0, j = 0, c = static_cast<arma::uword>(k);
      for (i = 0; i < p; ++i) {
        const arma::uword row_len = p - 1 - i;
        if (c < row_len) {
          j = i + 1 + c;
          break;
        }
        c -= row_len;
      }
      const arma::mat Rij = corrMat(m_X.rows(m_groups[i]), m_X.rows(m_groups[j]));  // n_i x n_j
      const arma::rowvec cij = arma::sum(W[i] % (Rij * W[j]), 0);                   // 1 x qc
      for (arma::uword t = 0; t < qc; ++t) {
        cross(i, j, t) = cij(t);
        cross(j, i, t) = cij(t);
      }
    }

    // aggregate per point
    for (arma::uword t = 0; t < qc; ++t) {
      arma::mat KM = cross.slice(t);
      KM.diag() = Kdiag.col(t);
      KM.diag() += jitter;
      const arma::vec kM = KM.diag();
      arma::vec w;
      if (!arma::solve(w, arma::symmatu(KM), kM, arma::solve_opts::likely_sympd))
        w = arma::pinv(arma::symmatu(KM)) * kM;  // graceful degradation
      mean(start + t) = m_beta0 + arma::dot(w, M.col(t) - m_beta0);
      if (return_stdev)
        var(start + t) = m_sigma2 * std::max(0.0, 1.0 - arma::dot(w, kM));
    }
  }

  if (!return_stdev)
    return {mean, arma::vec()};
  return {mean, arma::sqrt(var)};
}

// =============================================================================
// summary
// =============================================================================

std::string NestedKriging::summary() const {
  std::ostringstream oss;
  if (!m_is_fitted) {
    oss << "* covariance: " << m_covType << " (not fitted)";
    return oss.str();
  }
  oss << "* data: " << m_X.n_rows << " x " << m_X.n_cols << " -> " << m_y.n_elem << "\n";
  oss << "* groups: " << m_groups.size() << " (";
  for (arma::uword g = 0; g < m_groups.size(); ++g)
    oss << (g ? "," : "") << m_groups[g].n_elem;
  oss << ")\n";
  oss << "* aggregation: " << aggregationToString(m_aggregation) << "\n";
  if (warped()) {
    oss << "* warping:";
    for (const auto& w : m_warping)
      oss << " " << w;
    oss << "\n";
  }
  oss << "* trend (constant): " << m_beta0 << "\n";
  oss << "* variance: " << m_sigma2 << "\n";
  oss << "* covariance: " << m_covType << ", range: " << m_theta.t();
  return oss.str();
}
