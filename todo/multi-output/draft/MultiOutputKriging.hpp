// =============================================================================
// ESQUISSE — NON BRANCHÉE AU BUILD (aucune implémentation .cpp).
//
// Vérifiée syntaxiquement contre les vrais en-têtes du dépôt :
//   g++ -fsyntax-only -std=c++17 -Isrc/lib/include -Ibuild_test/src/lib \
//       -Idependencies/armadillo-code/include -x c++ \
//       todo/multi-output/draft/MultiOutputKriging.hpp
// (nécessite un dossier de build déjà configuré, pour libKriging_exports.h).
//
// Brouillon d'API pour un krigeage multi-sorties isotopique (todo/multi-output/ANALYSIS.md
// §3 et §5). Classe autonome pour l'instant ; une fois `KrigingImpl` généralisé
// à un `y` matriciel (étape 1 de la feuille de route), les modes Shared /
// Separable / SeparableKernel en dériveront pour réutiliser la factorisation
// (T, M, circ, star) — le mode PCA restera une composition sur des `Kriging`,
// sur le patron de NestedKriging.
//
// Les points marqués « D? » dépendent d'un arbitrage encore ouvert
// (todo/multi-output/ANALYSIS.md §6).
// =============================================================================

#ifndef LIBKRIGING_MULTIOUTPUTKRIGING_HPP
#define LIBKRIGING_MULTIOUTPUTKRIGING_HPP

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/Covariance.hpp"
#include "libKriging/Kriging.hpp"
#include "libKriging/Trend.hpp"
#include "libKriging/libKriging_exports.h"

/// Same role as KrigingParameters, with one entry per output where relevant.
/// Shapes: q outputs, d inputs, p trend columns, d_t output coordinates.
struct MultiOutputKrigingParameters {
  std::optional<arma::vec> sigma2;  ///< q   (Shared) — or scalar σ² as a 1-vector (SeparableKernel)
  bool is_sigma2_estim = true;
  std::optional<arma::mat> theta;  ///< d   (common correlation lengths in x; several rows = multistart)
  bool is_theta_estim = true;
  std::optional<arma::mat> beta;  ///< p × q (one trend vector per output)
  bool is_beta_estim = true;
  std::optional<arma::vec> output_theta;  ///< d_t (SeparableKernel only: ranges of R_t)
  bool is_output_theta_estim = true;
  // D? nugget : un seul rapport alpha = σ²/(σ²+τ²) commun à toutes les sorties
  // (préserve la factorisation partagée). Pas de bruit hétérogène par sortie.
  std::optional<double> nugget_ratio;
  bool is_nugget_ratio_estim = true;
};

/** Multi-output Kriging, isotopic design: q outputs observed at the same n
 * points, Y is n × q (rows = observations, columns = outputs — same layout
 * convention as X).
 *
 * Output models (see todo/multi-output/ANALYSIS.md §1 and §4):
 *
 *   "shared"              PP-GaSP (Gu & Berger, 2016):
 *                           Cov(Y_i(x), Y_j(x')) = δ_ij σ_j² r(x, x'; θ)
 *                         one θ for all outputs, β_j and σ_j² per output.
 *                         One Cholesky of R(θ): O(n³ + q n²).
 *
 *   "separable"           ICM / Conti & O'Hagan (2010):
 *                           Cov = Σ ⊗ r(x, x'; θ),  Σ (q × q) free SPD,
 *                         Σ̂ = Zᵀ R⁻¹ Z / n in closed form (Z = Y − F β̂).
 *                         Same predictive mean as "shared" for a given θ
 *                         (autokrigeability) but joint predictive covariance
 *                         across outputs. Requires n − p ≥ q for Σ̂ to be
 *                         full rank (D? sinon : régularisation de type
 *                         Ledoit-Wolf, ou refus).
 *
 *   "separable(<kernel>)" Σ = σ² R_t(φ) with R_t a parametric kernel on the
 *                         output coordinates t (q × d_t, see
 *                         set_output_coordinates), e.g. time steps:
 *                           Cov = σ² R_t(t, t'; φ) ⊗ r(x, x'; θ).
 *                         Handles q > n; O(n³ + q³) via eigendecomposition
 *                         of each Kronecker factor.
 *
 *   "pca(K)" / "pca(v)"   Karhunen-Loève reduction (Higdon et al., 2008):
 *                           Y ≈ 1 ȳᵀ + A Φᵀ,  Φ (q × K) orthonormal,
 *                         one independent `Kriging` per score column of A
 *                         (own θ_k, σ_k²). K given as an integer ≥ 1, or as
 *                         a fraction 0 < v < 1 of explained variance.
 *                         Truncation variance is added to the prediction
 *                         variance. Composition, no core change.
 *
 * Covariance ordering: every joint covariance is over vec(Y_n), i.e. the
 * m prediction points of output 1, then output 2, …  (column-major, as
 * Armadillo's vectorise). For Kronecker models, Cov(vec Y_n) = Σ ⊗ C_x.
 *
 * Restrictions (checked at runtime):
 *   - isotopic only (no missing Y entries): heterotopic data → stacked ICM
 *     or WarpKriging with a categorical output-index column (ANALYSIS §1.F);
 *   - one regmodel for all outputs (keeps F, hence the GLS, shared);
 *   - no per-observation / per-output noise; only a common nugget ratio;
 *   - objectives: "LL", "LOO" (D? "LMP" plus tard ; LLVecchia/LLNystrom
 *     compatibles en principe avec la factorisation partagée, différés);
 *   - save/load deferred (as in NestedKriging).
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

  /// Parses "shared", "separable", "separable(matern5_2)", "pca(5)",
  /// "pca(0.99)" ; throws on malformed spec.
  LIBKRIGING_EXPORT static OutputModel outputModelFromString(const std::string& s);
  LIBKRIGING_EXPORT static std::string outputModelToString(OutputModel m);

  MultiOutputKriging() = delete;

  /// @param covType kernel on x (same names as Kriging)
  /// @param outputModel see class documentation
  LIBKRIGING_EXPORT explicit MultiOutputKriging(const std::string& covType, const std::string& outputModel = "shared");

  LIBKRIGING_EXPORT MultiOutputKriging(const arma::mat& Y,
                                       const arma::mat& X,
                                       const std::string& covType,
                                       const std::string& outputModel = "shared",
                                       const Trend::RegressionModel& regmodel = Trend::RegressionModel::Constant,
                                       bool normalize = false,
                                       const std::string& optim = "BFGS",
                                       const std::string& objective = "LL",
                                       const Parameters& parameters = {});

  /** Output coordinates (e.g. time steps), q × d_t. Required before fit() for
   * "separable(<kernel>)"; optional otherwise (kept for summaries / bindings).
   * Defaults to 0..q-1 as a single column. */
  LIBKRIGING_EXPORT void set_output_coordinates(const arma::mat& t);

  /** Fit on (X, Y).
   * @param Y n × q outputs (q = 1 is accepted and reduces to Kriging)
   * @param X n × d inputs
   * @param normalize per-output centering/scaling of Y, and of X as in Kriging
   * @param objective "LL" (summed over outputs for "shared"; matrix-normal
   *        likelihood for "separable*", see ANALYSIS §4) or "LOO" (summed
   *        squared LOO errors, closed form with the shared factorization).
   *        For "pca", forwarded to each component's Kriging. */
  LIBKRIGING_EXPORT void fit(const arma::mat& Y,
                             const arma::mat& X,
                             const Trend::RegressionModel& regmodel = Trend::RegressionModel::Constant,
                             bool normalize = false,
                             const std::string& optim = "BFGS",
                             const std::string& objective = "LL",
                             const Parameters& parameters = {});

  /** Prediction at X_n (m × d).
   * @return (mean [m × q], stdev [m × q], cov [mq × mq] over vec(Y_n),
   *          deriv [m × d × q]) ; empty when the matching flag is false.
   * The dense joint cov is meant for small m·q ; for Kronecker models prefer
   * predictCovFactors. D? cov « croisée » seulement (bloc diagonal par sortie
   * si outputModel = "shared", puisque Σ est diagonale). */
  LIBKRIGING_EXPORT std::tuple<arma::mat, arma::mat, arma::mat, arma::cube> predict(const arma::mat& X_n,
                                                                                    bool return_stdev = true,
                                                                                    bool return_cov = false,
                                                                                    bool return_deriv = false);

  /** Kronecker factors of the predictive covariance, Cov(vec Y_n) = Σ ⊗ C_x.
   * Only for "shared" (Σ diagonal), "separable" and "separable(<kernel>)";
   * throws for "pca" (not Kronecker: Σ_k φ_k φ_kᵀ ⊗ C_k + truncation).
   * @return (C_x [m × m] posterior correlation incl. trend uncertainty,
   *          Σ [q × q] output covariance) */
  LIBKRIGING_EXPORT std::tuple<arma::mat, arma::mat> predictCovFactors(const arma::mat& X_n);

  /** Joint conditional trajectories at X_n.
   * @return m × q × nsim cube (slice s = one joint draw of all outputs) */
  LIBKRIGING_EXPORT arma::cube simulate(int nsim, int seed, const arma::mat& X_n, bool will_update = false);

  /// Re-simulate at the last simulate() X_n after assimilating (X_u, Y_u).
  LIBKRIGING_EXPORT arma::cube update_simulate(const arma::mat& Y_u, const arma::mat& X_u);

  /// Append (X_u, Y_u) (Y_u is m × q, same q) and optionally refit.
  LIBKRIGING_EXPORT void update(const arma::mat& Y_u, const arma::mat& X_u, bool refit = true);

  // --- objective functions (Shared / Separable* only; throw for PCA) ---------
  /// gamma = [θ] ("shared", "separable") or [θ, φ] ("separable(<kernel>)")
  LIBKRIGING_EXPORT std::tuple<double, arma::vec> logLikelihoodFun(const arma::vec& gamma,
                                                                   bool return_grad,
                                                                   bool bench = false);
  LIBKRIGING_EXPORT std::tuple<double, arma::vec> leaveOneOutFun(const arma::vec& theta,
                                                                 bool return_grad,
                                                                 bool bench = false);
  LIBKRIGING_EXPORT double logLikelihood();
  LIBKRIGING_EXPORT double leaveOneOut();
  /// LOO means and stdevs, n × q each.
  LIBKRIGING_EXPORT std::tuple<arma::mat, arma::mat> leaveOneOutMat(const arma::vec& theta);

  // --- accessors -----------------------------------------------------------
  [[nodiscard]] const std::string& kernel() const { return m_covType; }
  [[nodiscard]] OutputModel output_model() const { return m_output_model; }
  [[nodiscard]] const std::string& output_kernel() const { return m_output_covType; }
  [[nodiscard]] arma::uword nb_outputs() const { return m_Y.n_cols; }
  [[nodiscard]] const arma::mat& X() const { return m_X; }
  [[nodiscard]] const arma::mat& Y() const { return m_Y; }
  [[nodiscard]] const arma::mat& output_coordinates() const { return m_t; }
  [[nodiscard]] const Trend::RegressionModel& regmodel() const { return m_regmodel; }
  [[nodiscard]] bool normalize() const { return m_normalize; }
  [[nodiscard]] const arma::rowvec& centerY() const { return m_centerY; }  ///< 1 × q
  [[nodiscard]] const arma::rowvec& scaleY() const { return m_scaleY; }    ///< 1 × q

  /// Common θ in x (Shared / Separable*). For PCA, see component(k).theta().
  [[nodiscard]] const arma::vec& theta() const { return m_theta; }
  [[nodiscard]] const arma::mat& beta() const { return m_beta; }      ///< p × q
  [[nodiscard]] const arma::vec& sigma2() const { return m_sigma2; }  ///< q (Shared), 1 (SeparableKernel)
  /// Σ̂ (q × q): diag(σ²) for Shared, free for Separable, σ² R_t(φ) for SeparableKernel.
  [[nodiscard]] LIBKRIGING_EXPORT arma::mat output_cov() const;
  [[nodiscard]] const arma::vec& output_theta() const { return m_output_theta; }  ///< φ (SeparableKernel)
  [[nodiscard]] std::optional<double> nugget_ratio() const { return m_nugget_ratio; }

  // PCA mode only (throw otherwise)
  [[nodiscard]] LIBKRIGING_EXPORT arma::uword nb_components() const;
  [[nodiscard]] LIBKRIGING_EXPORT const arma::mat& pca_basis() const;      ///< Φ, q × K
  [[nodiscard]] LIBKRIGING_EXPORT const arma::rowvec& pca_mean() const;    ///< ȳ, 1 × q
  [[nodiscard]] LIBKRIGING_EXPORT const arma::vec& pca_explained() const;  ///< cumulative fraction, K
  [[nodiscard]] LIBKRIGING_EXPORT const Kriging& component(arma::uword k) const;

  LIBKRIGING_EXPORT std::string summary() const;

  // D? save/load : schéma JSON dédié (content = "MultiOutputKriging"), différé.

 private:
  // configuration
  std::string m_covType;
  OutputModel m_output_model = OutputModel::Shared;
  std::string m_output_covType;  ///< kernel on t for SeparableKernel
  double m_pca_spec = 0.99;      ///< K (≥ 1, integer) or explained-variance fraction (< 1)
  Trend::RegressionModel m_regmodel = Trend::RegressionModel::Constant;
  bool m_normalize = false;
  std::string m_optim;
  std::string m_objective;

  // data
  arma::mat m_X;  ///< n × d
  arma::mat m_Y;  ///< n × q
  arma::mat m_t;  ///< q × d_t
  arma::rowvec m_centerX, m_scaleX;
  arma::rowvec m_centerY, m_scaleY;

  // Shared / Separable* fitted state (→ KrigingImpl once generalized)
  arma::vec m_theta;
  arma::mat m_beta;          ///< p × q
  arma::vec m_sigma2;        ///< q or 1
  arma::mat m_Sigma;         ///< q × q (Separable)
  arma::vec m_output_theta;  ///< φ (SeparableKernel)
  std::optional<double> m_nugget_ratio;
  arma::mat m_F;        ///< n × p, shared by all outputs
  arma::mat m_T;        ///< chol(R), shared
  arma::mat m_M;        ///< T \ F, shared
  arma::mat m_Z;        ///< T \ (Y − F β̂), n × q
  arma::mat m_Ut;       ///< eigenvectors of R_t (SeparableKernel)
  arma::vec m_lambdat;  ///< eigenvalues of R_t (SeparableKernel)

  // PCA state
  arma::rowvec m_pca_mean;
  arma::mat m_pca_basis;
  arma::vec m_pca_explained;
  double m_pca_truncation_var = 0.0;  ///< per-point residual variance outside span(Φ)
  std::vector<std::unique_ptr<Kriging>> m_components;

  // simulate / update_simulate state
  arma::mat m_lastsim_Xn;
  int m_lastsim_seed = 0;
  int m_lastsim_nsim = 0;
};

#endif  // LIBKRIGING_MULTIOUTPUTKRIGING_HPP
