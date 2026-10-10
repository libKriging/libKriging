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
  /// "separable(<kernel>)" only: φ of the output kernel, d_t (several rows =
  /// starting points); estimated with θ unless is_theta_estim is false
  std::optional<arma::mat> output_theta;
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
 *   "shared"
 *       Shared correlation, parallel partial GP (Gu & Berger, AoAS 2016):
 *         Y_j ~ GP(F β_j, σ_j² r_θ),  outputs independent given θ,
 *       one θ for all outputs (maximum of the summed concentrated
 *       likelihoods, one Cholesky per evaluation), β_j and σ_j² per output.
 *       Same normalization, scales and optimizer as Kriging: with q = 1 it
 *       gives the same model as Kriging(y, X, …). Objectives "LL" (summed
 *       log-likelihoods) and "LOO" (LOO squared errors summed over outputs,
 *       on the normalized scale). update_simulate conditions the stored draws
 *       exactly (fixed θ, universal kriging).
 *
 *   "separable"
 *       Intrinsic coregionalization model, isotopic (Conti & O'Hagan 2010):
 *         Cov(vec Y) = Σ ⊗ r_θ,  Σ free q × q,
 *       Σ̂ = E*ᵀ E* / n in closed form, θ maximizes the profiled likelihood
 *       −2ℓ = nq log 2π + n log|Σ̂| + q log|R| + nq. Same θ-profile of β and
 *       same predictive mean as "shared" (autokrigeability), but coherent
 *       joint covariance and joint simulations across outputs. Requires
 *       n − p ≥ q and a non-singular Σ̂ (otherwise use "pca" or
 *       "separable(<kernel>)").
 *
 *   "separable(<kernel>)", e.g. "separable(matern5_2)"
 *       Separable model with a parametric output covariance (Rougier 2008):
 *         Cov(vec Y) = σ² R_t(φ) ⊗ r_θ,
 *       R_t the correlation of <kernel> (same names as covType) over the
 *       output coordinates t (q × d_t, set_output_coordinates, required).
 *       −2ℓ = nq log 2πσ̂² + q log|R| + n log|R_t| + nq, σ̂² = tr(R_t⁻¹ E*ᵀE*)/(nq),
 *       maximized in (θ, φ). Same predictive mean as "shared"; coherent
 *       joint covariance with d_t parameters instead of q(q+1)/2, so q may
 *       exceed n (O(q³) per evaluation). Objective "LL" only (LOO does not
 *       depend on φ).
 *
 * Covariance ordering: joint covariances are over vec(Y_n), i.e. the m
 * prediction points of output 1, then output 2, … (column-major).
 *
 * Restrictions: isotopic only (no missing Y); one regmodel for all latent
 * submodels; no nugget/noise. save/load keep the fitted model, not the state
 * of the last simulate (update_simulate needs a new simulate after load).
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
  LIBKRIGING_EXPORT ~MultiOutputKriging();
  LIBKRIGING_EXPORT MultiOutputKriging(MultiOutputKriging&&) noexcept;
  LIBKRIGING_EXPORT MultiOutputKriging& operator=(MultiOutputKriging&&) noexcept;

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

  /** Output coordinates (e.g. time steps), q × d_t. Required by
   * "separable(<kernel>)"; stored and checked against q at fit time
   * otherwise. A vector is taken as a single column. */
  LIBKRIGING_EXPORT void set_output_coordinates(const arma::mat& t);

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
   * (X_u, Y_u), without changing the model. "pca": the basis is kept, Y_u is
   * projected on it and each latent Kriging is updated with its scores.
   * "shared" / "separable": exact conditioning of the stored draws (θ, Σ kept).
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
  [[nodiscard]] const arma::mat& output_coordinates() const { return m_t; }  ///< q × d_t (empty if unset)
  [[nodiscard]] const Trend::RegressionModel& regmodel() const { return m_regmodel; }
  [[nodiscard]] bool normalize() const { return m_normalize; }
  [[nodiscard]] const std::string& optim() const { return m_optim; }
  [[nodiscard]] const std::string& objective() const { return m_objective; }
  /// Output centering / scaling, 1 × q: column means and sd (if normalize)
  /// in "pca", column min and range (if normalize, as Kriging) in "shared".
  [[nodiscard]] const arma::rowvec& centerY() const { return m_centerY; }
  [[nodiscard]] const arma::rowvec& scaleY() const { return m_scaleY; }

  // Shared / separable modes (same scales as the Kriging accessors:
  // normalized when normalize = true)
  [[nodiscard]] LIBKRIGING_EXPORT const arma::vec& theta() const;   ///< θ, d
  [[nodiscard]] LIBKRIGING_EXPORT const arma::vec& sigma2() const;  ///< σ_j², q
  [[nodiscard]] LIBKRIGING_EXPORT const arma::mat& beta() const;    ///< β_j, p × q
  /// Σ (q × q): diag(σ_j²) in "shared", free in "separable", σ² R_t(φ) in
  /// "separable(<kernel>)"
  [[nodiscard]] LIBKRIGING_EXPORT const arma::mat& output_cov() const;
  /// φ of the output kernel, d_t ("separable(<kernel>)" only)
  [[nodiscard]] LIBKRIGING_EXPORT const arma::vec& output_theta() const;
  /** Kronecker factors of the predictive covariance at X_n (m × d):
   * (C_x [m × m] correlation, Σ_raw [q × q] on the original scale) with
   * Cov(vec Y_n) = kron(Σ_raw, C_x), without forming the dense mq × mq. */
  LIBKRIGING_EXPORT std::tuple<arma::mat, arma::mat> predictCovFactors(const arma::mat& X_n);
  /// Summed log-likelihood at the fitted θ
  LIBKRIGING_EXPORT double logLikelihood();
  /// Summed concentrated log-likelihood at θ (normalized scale), with its
  /// gradient in θ when grad is true. "separable(<kernel>)": θ is followed
  /// by φ (d + d_t values, gradient likewise)
  LIBKRIGING_EXPORT std::tuple<double, arma::vec> logLikelihoodFun(const arma::vec& theta, bool grad = false);
  /// LOO mean squared error summed over outputs (normalized scale) at θ,
  /// with its gradient in θ when grad is true
  LIBKRIGING_EXPORT std::tuple<double, arma::vec> leaveOneOutFun(const arma::vec& theta, bool grad = false);

  // PCA mode
  [[nodiscard]] arma::uword nb_components() const { return m_components.size(); }
  [[nodiscard]] const arma::mat& pca_basis() const { return m_pca_basis; }          ///< Φ, q × K
  [[nodiscard]] const arma::vec& pca_explained() const { return m_pca_explained; }  ///< cumulative fraction, K
  [[nodiscard]] const arma::mat& pca_residual() const { return m_pca_residual; }    ///< Ρ, n × q (original scale)
  [[nodiscard]] LIBKRIGING_EXPORT const Kriging& component(arma::uword k) const;

  LIBKRIGING_EXPORT std::string summary() const;

  /// Save the model (configuration, data and fitted state) to a JSON file
  LIBKRIGING_EXPORT void save(const std::string& filename) const;
  /// Load a model saved by save()
  LIBKRIGING_EXPORT static MultiOutputKriging load(const std::string& filename);

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
  arma::mat m_t;  ///< q × d_t output coordinates
  arma::rowvec m_centerY;
  arma::rowvec m_scaleY;

  // PCA state
  arma::mat m_pca_basis;
  arma::vec m_pca_explained;
  arma::mat m_pca_residual;
  std::vector<std::unique_ptr<Kriging>> m_components;

  // simulate / update_simulate state
  arma::cube m_lastsim_residual;  ///< m × q × nsim truncation draws of the last simulate

  // Shared state (KrigingImpl-derived, defined in MultiOutputKriging.cpp)
  class SharedModel;
  class OutputKernel;
  std::unique_ptr<SharedModel> m_shared;

  void parse_output_model(const std::string& s);
  void check_fitted(const std::string& where) const;
  const SharedModel& shared(const std::string& where) const;
  SharedModel& shared(const std::string& where);
  /// Scores of Y (n × q, original scale) on the current basis: n × K
  arma::mat project(const arma::mat& Y) const;
  /// Map latent values (m × K) back to outputs (m × q), original scale
  arma::mat reconstruct(const arma::mat& A) const;
  arma::rowvec residual_variance() const;
};

#endif  // LIBKRIGING_MULTIOUTPUTKRIGING_HPP
