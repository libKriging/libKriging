#ifndef LIBKRIGING_MULTIOUTPUTKRIGING_HPP
#define LIBKRIGING_MULTIOUTPUTKRIGING_HPP

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/Kriging.hpp"
#include "libKriging/Trend.hpp"
#include "libKriging/libKriging_exports.h"

/// Starting / fixed hyperparameters shared by all latent submodels.
struct MultiOutputKrigingParameters {
  std::optional<arma::mat> theta;  ///< d (several rows = multistart starting points)
  bool is_theta_estim = true;
};

/** Multi-output Kriging, isotopic design: q outputs observed at the same n
 * points. Y is n × q (rows = observations, columns = outputs — same layout
 * convention as X). See todo/multi-output/ANALYSIS.md for the design.
 *
 * Output models (`outputModel` string):
 *
 *   "pca(K)" / "pca(v)" / "pca"
 *       Karhunen-Loève reduction (Higdon et al., JASA 2008):
 *         Y ≈ 1 ȳᵀ + (A Φᵀ) diag(s),  Φ (q × K) orthonormal,
 *       ȳ the column means, s the column scales (1 unless normalize),
 *       one independent `Kriging` per score column A_k (own θ_k, σ_k², β_k).
 *       K is an integer ≥ 1, or a fraction 0 < v < 1 of explained variance
 *       (default "pca" = "pca(0.99)"), capped at min(n − 1, q).
 *       The truncation residual Ρ = Y − reconstruction (n × q) is modelled as
 *       N(0, Σ_res) with Σ_res = ΡᵀΡ / (n − 1), independent across
 *       prediction points: it is added to the prediction variance and drawn
 *       in simulate (via Ρᵀw, w ~ N(0, I_n), so it stays coherent across
 *       outputs without forming the q × q matrix).
 *
 *   "shared", "separable", "separable(<kernel>)"
 *       Shared-correlation (PP-GaSP) and separable (ICM / R_t ⊗ R_x) models:
 *       parsed but not implemented yet (fit throws).
 *
 * Covariance ordering: joint covariances are over vec(Y_n), i.e. the m
 * prediction points of output 1, then output 2, … (column-major).
 *
 * Restrictions: isotopic only (no missing Y); one regmodel for all latent
 * submodels; no nugget/noise; save/load not yet implemented.
 */
class MultiOutputKriging {
 public:
  using Parameters = MultiOutputKrigingParameters;

  enum class OutputModel {
    Shared,           ///< "shared"
    Separable,        ///< "separable"
    SeparableKernel,  ///< "separable(<kernel>)"
    PCA,              ///< "pca(K)" / "pca(v)"
  };

  MultiOutputKriging() = delete;

  /// @param covType kernel on x (same names as Kriging)
  /// @param outputModel see class documentation
  LIBKRIGING_EXPORT explicit MultiOutputKriging(const std::string& covType, const std::string& outputModel = "pca");

  LIBKRIGING_EXPORT MultiOutputKriging(const arma::mat& Y,
                                       const arma::mat& X,
                                       const std::string& covType,
                                       const std::string& outputModel = "pca",
                                       const Trend::RegressionModel& regmodel = Trend::RegressionModel::Constant,
                                       bool normalize = false,
                                       const std::string& optim = "BFGS",
                                       const std::string& objective = "LL",
                                       const Parameters& parameters = {});

  /** Fit on (X, Y).
   * @param Y n × q outputs
   * @param X n × d inputs
   * @param normalize scale each output by its standard deviation before the
   *        PCA (correlation-based PCA), and normalize X / scores inside each
   *        latent Kriging
   * @param optim, objective forwarded to every latent Kriging */
  LIBKRIGING_EXPORT void fit(const arma::mat& Y,
                             const arma::mat& X,
                             const Trend::RegressionModel& regmodel = Trend::RegressionModel::Constant,
                             bool normalize = false,
                             const std::string& optim = "BFGS",
                             const std::string& objective = "LL",
                             const Parameters& parameters = {});

  /** Prediction at X_n (m × d).
   * @return (mean [m × q], stdev [m × q], cov [mq × mq] over vec(Y_n),
   *          mean derivative [m × d × q]) ; empty when the flag is false.
   * The dense joint cov is meant for small m·q. */
  LIBKRIGING_EXPORT std::tuple<arma::mat, arma::mat, arma::mat, arma::cube> predict(const arma::mat& X_n,
                                                                                    bool return_stdev = true,
                                                                                    bool return_cov = false,
                                                                                    bool return_deriv = false);

  /** Joint conditional trajectories at X_n.
   * @param will_update store the state needed by update_simulate
   * @return m × q × nsim cube (slice s = one joint draw of all outputs) */
  LIBKRIGING_EXPORT arma::cube simulate(int nsim, int seed, const arma::mat& X_n, bool will_update = false);

  /** Re-draw the last simulate() trajectories conditionally on new data
   * (X_u, Y_u), without changing the model. The PCA basis is kept: Y_u is
   * projected on it and each latent Kriging is updated with its scores.
   * @return m × q × nsim cube */
  LIBKRIGING_EXPORT arma::cube update_simulate(const arma::mat& Y_u, const arma::mat& X_u);

  /** Append (X_u, Y_u) (Y_u is m × q, same q).
   * refit = true : PCA basis recomputed and latent models refitted on all
   *                data (same options as the last fit);
   * refit = false: basis kept, Y_u projected, latent models conditioned on
   *                the new scores without re-estimation. */
  LIBKRIGING_EXPORT void update(const arma::mat& Y_u, const arma::mat& X_u, bool refit = true);

  /** Leave-one-out predictions of Y (closed form per latent model).
   * The basis is computed on all data (slightly optimistic).
   * @return (mean [n × q], stdev [n × q]) */
  LIBKRIGING_EXPORT std::tuple<arma::mat, arma::mat> leaveOneOutMat();
  /// Mean squared LOO error over all n·q entries.
  LIBKRIGING_EXPORT double leaveOneOut();

  // --- accessors -----------------------------------------------------------
  [[nodiscard]] const std::string& kernel() const { return m_covType; }
  [[nodiscard]] OutputModel output_model() const { return m_output_model; }
  [[nodiscard]] LIBKRIGING_EXPORT std::string output_model_string() const;
  [[nodiscard]] arma::uword nb_outputs() const { return m_Y.n_cols; }
  [[nodiscard]] const arma::mat& X() const { return m_X; }
  [[nodiscard]] const arma::mat& Y() const { return m_Y; }
  [[nodiscard]] const Trend::RegressionModel& regmodel() const { return m_regmodel; }
  [[nodiscard]] bool normalize() const { return m_normalize; }
  [[nodiscard]] const std::string& optim() const { return m_optim; }
  [[nodiscard]] const std::string& objective() const { return m_objective; }
  [[nodiscard]] const arma::rowvec& centerY() const { return m_centerY; }  ///< ȳ, 1 × q
  [[nodiscard]] const arma::rowvec& scaleY() const { return m_scaleY; }    ///< s, 1 × q

  // PCA mode
  [[nodiscard]] arma::uword nb_components() const { return m_components.size(); }
  [[nodiscard]] const arma::mat& pca_basis() const { return m_pca_basis; }          ///< Φ, q × K
  [[nodiscard]] const arma::vec& pca_explained() const { return m_pca_explained; }  ///< cumulative fraction, K
  [[nodiscard]] const arma::mat& pca_residual() const { return m_pca_residual; }    ///< Ρ, n × q (original scale)
  [[nodiscard]] LIBKRIGING_EXPORT const Kriging& component(arma::uword k) const;

  LIBKRIGING_EXPORT std::string summary() const;

 private:
  // configuration
  std::string m_covType;
  OutputModel m_output_model = OutputModel::PCA;
  std::string m_output_covType;  ///< kernel on t for SeparableKernel
  double m_pca_spec = 0.99;      ///< K (integer ≥ 1) or explained-variance fraction (< 1)
  Trend::RegressionModel m_regmodel = Trend::RegressionModel::Constant;
  bool m_normalize = false;
  std::string m_optim = "BFGS";
  std::string m_objective = "LL";
  Parameters m_parameters;

  // data
  arma::mat m_X;  ///< n × d
  arma::mat m_Y;  ///< n × q
  arma::rowvec m_centerY;
  arma::rowvec m_scaleY;

  // PCA state
  arma::mat m_pca_basis;
  arma::vec m_pca_explained;
  arma::mat m_pca_residual;
  std::vector<std::unique_ptr<Kriging>> m_components;

  // simulate / update_simulate state
  arma::cube m_lastsim_residual;  ///< m × q × nsim truncation draws of the last simulate

  void parse_output_model(const std::string& s);
  void check_fitted(const std::string& where) const;
  /// Scores of Y (n × q, original scale) on the current basis: n × K
  arma::mat project(const arma::mat& Y) const;
  /// Map latent values (m × K) back to outputs (m × q), original scale
  arma::mat reconstruct(const arma::mat& A) const;
  arma::rowvec residual_variance() const;
};

#endif  // LIBKRIGING_MULTIOUTPUTKRIGING_HPP
