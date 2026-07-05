#ifndef LIBKRIGING_NESTEDKRIGING_HPP
#define LIBKRIGING_NESTEDKRIGING_HPP

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/Covariance.hpp"
#include "libKriging/Kriging.hpp"
#include "libKriging/Trend.hpp"
#include "libKriging/WarpKriging.hpp"
#include "libKriging/libKriging_exports.h"

/** Nested Kriging: divide-and-conquer Gaussian process for large n.
 *
 * (X, y) is partitioned into p groups; one submodel is fitted per group
 * (`Kriging` by default, `WarpKriging` when a warping spec is given);
 * hyperparameters are then unified (common prior) and predictions are
 * aggregated with one of:
 *   - PoE / gPoE / BCM / rBCM : precision-weighted products of experts
 *     (cheap: only submodel means/variances are needed),
 *   - NK : the optimal aggregation of Rullière, Durrande, Bachoc & Chevalier
 *     (Stat. Comput., 2018), which treats the submodel predictors M_i(x) as
 *     observations of the same GP prior and krige Y(x) on them. It is itself
 *     a kriging predictor: it interpolates the data and provides consistent
 *     variances (unlike the PoE family).
 *
 * With warping, the common prior is σ²·k(Φ(x), Φ(x′); θ): a warped kernel is
 * a valid GP prior, so both aggregation families apply unchanged. The common
 * (θ, warp) is estimated by a single reference fit on a global subsample of
 * size min(n, warp_subsample) — warp parameters (embeddings, MLP weights)
 * live on non-convex manifolds and cannot be averaged across groups the way
 * θ can, and one subsample fit is much cheaper than one warp training per
 * group. Submodels are then fitted with optim="none" on the seeded prior.
 *
 * Complexities (n obs, p groups of size ~n/p, q prediction points):
 *   fit      : O(p (n/p)^3) likelihood evals  [vs O(n^3)]
 *   predict  : PoE family O(q n^2/p) ; NK O(q n^2) worst case, dominated by
 *              cross-correlation blocks, parallelized over group pairs.
 *
 * Restrictions (documented, checked at runtime):
 *   - NK aggregation requires a Constant trend (simple-kriging theory);
 *     PoE family works with any trend.
 *   - no nugget/noise channel;
 *   - `normalize` not yet supported (do it outside if needed);
 *   - save/load not yet implemented.
 */
class NestedKriging {
 public:
  enum class Aggregation { PoE, gPoE, BCM, rBCM, NK };
  enum class Partition { Random, KMeans };

  LIBKRIGING_EXPORT static Aggregation aggregationFromString(const std::string& s);
  LIBKRIGING_EXPORT static std::string aggregationToString(Aggregation a);

  NestedKriging() = delete;

  LIBKRIGING_EXPORT explicit NestedKriging(const std::string& covType);

  LIBKRIGING_EXPORT NestedKriging(const arma::vec& y,
                                  const arma::mat& X,
                                  const std::string& covType,
                                  arma::uword nb_groups,
                                  Aggregation aggregation = Aggregation::NK,
                                  Partition partition = Partition::KMeans,
                                  int seed = 123,
                                  const Trend::RegressionModel& regmodel = Trend::RegressionModel::Constant,
                                  const std::string& optim = "BFGS",
                                  const std::string& objective = "LL",
                                  const Kriging::Parameters& parameters = {},
                                  const std::vector<std::string>& warping = {});

  /** Fit p independent submodels, then unify hyperparameters into a common
   * prior and refit with optim="none".
   *
   * Plain path (warping empty, `Kriging` submodels): theta <- group-size
   * weighted geometric mean, sigma2 / beta0 <- weighted means.
   *
   * Warped path (`WarpKriging` submodels): (theta, warp_params) <- from a
   * single reference fit on a global subsample (see set_warp_subsample);
   * sigma2 / beta0 <- weighted means over the seeded submodel fits
   * (sigma2 is profiled per group given the common correlation).
   *
   * @param warping per-dimension warp specs (see WarpKriging), empty for
   *        plain Kriging submodels. */
  LIBKRIGING_EXPORT void fit(const arma::vec& y,
                             const arma::mat& X,
                             arma::uword nb_groups,
                             const Trend::RegressionModel& regmodel = Trend::RegressionModel::Constant,
                             const std::string& optim = "BFGS",
                             const std::string& objective = "LL",
                             const Kriging::Parameters& parameters = {},
                             const std::vector<std::string>& warping = {});

  /** Aggregated prediction at X_n (q x d).
   * @return (mean [q], stdev [q]) ; stdev empty if return_stdev=false. */
  LIBKRIGING_EXPORT std::tuple<arma::vec, arma::vec> predict(const arma::mat& X_n, bool return_stdev = true);

  // --- accessors -----------------------------------------------------------
  [[nodiscard]] const std::string& kernel() const { return m_covType; }
  [[nodiscard]] Aggregation aggregation() const { return m_aggregation; }
  [[nodiscard]] arma::uword nb_groups() const { return m_groups.size(); }
  [[nodiscard]] const std::vector<arma::uvec>& groups() const { return m_groups; }
  [[nodiscard]] bool warped() const { return !m_warping.empty(); }
  [[nodiscard]] const std::vector<std::string>& warping() const { return m_warping; }
  /// Plain-path submodel access (throws when warped)
  [[nodiscard]] const Kriging& submodel(arma::uword i) const;
  /// Warped-path submodel access (throws when plain)
  [[nodiscard]] const libKriging::WarpKriging& wsubmodel(arma::uword i) const;
  [[nodiscard]] const arma::vec& theta() const { return m_theta; }
  [[nodiscard]] double sigma2() const { return m_sigma2; }
  [[nodiscard]] double beta0() const { return m_beta0; }
  [[nodiscard]] const arma::mat& X() const { return m_X; }
  [[nodiscard]] const arma::vec& y() const { return m_y; }

  /// Prediction chunk size for the NK path (memory / speed trade-off:
  /// peak extra memory is ~ n * chunk doubles for the whitened weights).
  LIBKRIGING_EXPORT void set_predict_chunk(arma::uword chunk) { m_chunk = chunk; }

  /// Size of the global subsample used to estimate the warped-prior
  /// hyperparameters (theta, warp). Call before fit(); the reference fit
  /// costs O(min(n, m)^3) per likelihood evaluation.
  LIBKRIGING_EXPORT void set_warp_subsample(arma::uword m) { m_warp_subsample = m; }
  [[nodiscard]] arma::uword warp_subsample() const { return m_warp_subsample; }

  LIBKRIGING_EXPORT std::string summary() const;

 private:
  // configuration
  std::string m_covType;
  Aggregation m_aggregation = Aggregation::NK;
  Partition m_partition_method = Partition::KMeans;
  int m_seed = 123;
  arma::uword m_chunk = 128;

  // data & submodels
  arma::mat m_X;
  arma::vec m_y;
  Trend::RegressionModel m_regmodel = Trend::RegressionModel::Constant;
  std::vector<arma::uvec> m_groups;
  std::vector<std::unique_ptr<Kriging>> m_submodels;       ///< plain path
  std::vector<std::unique_ptr<libKriging::WarpKriging>> m_wsubmodels;  ///< warped path
  std::vector<std::string> m_warping;                      ///< empty = plain
  arma::uword m_warp_subsample = 1000;  ///< subsample size for the warped reference fit

  // unified (common prior) hyperparameters
  arma::vec m_theta;
  double m_sigma2 = -1;
  double m_beta0 = 0;

  // NK precomputations (per group, common prior)
  Covariance::CovFunc m_corr;          ///< resolved correlation kernel (plain path)
  std::vector<arma::mat> m_L;          ///< lower Cholesky of R_g (jittered)
  std::vector<arma::vec> m_alpha;      ///< R_g^{-1} (y_g - beta0)
  bool m_is_fitted = false;

  static constexpr double jitter = 1e-10;

  // helpers
  [[nodiscard]] arma::mat corrMat(const arma::mat& X1, const arma::mat& X2) const;
  void make_partition(arma::uword nb_groups);
  void unify_hyperparameters(const std::string& objective);
  void precompute_nk();
  [[nodiscard]] std::tuple<arma::vec, arma::vec> predict_poe_family(const arma::mat& X_n, bool return_stdev) const;
  [[nodiscard]] std::tuple<arma::vec, arma::vec> predict_nk(const arma::mat& X_n, bool return_stdev) const;
};

#endif  // LIBKRIGING_NESTEDKRIGING_HPP
