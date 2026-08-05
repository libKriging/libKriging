// =============================================================================
// ESQUISSE — NON BRANCHÉE AU BUILD (aucune implémentation .cpp).
//
// Vérifiée syntaxiquement contre les vrais en-têtes du dépôt :
//   g++ -fsyntax-only -std=c++17 -Isrc/lib/include -Ibuild/src/lib \
//       -Idependencies/armadillo-code/include -x c++ \
//       wip2/draft/MultiFidelityKriging.hpp
// (nécessite un `build/` déjà configuré, pour libKriging_exports.h généré).
//
// Brouillon d'API pour MultiFidelityKriging, calqué sur la structure de
// src/lib/include/libKriging/NestedKriging.hpp (classe de composition sur un
// vecteur de sous-Kriging).
//
// À déplacer vers src/lib/include/libKriging/ une fois les décisions D1-D5 de
// wip2/DESIGN.md tranchées. Les points marqués « D? » dépendent directement
// d'un arbitrage encore ouvert.
// =============================================================================

#ifndef LIBKRIGING_MULTIFIDELITYKRIGING_HPP
#define LIBKRIGING_MULTIFIDELITYKRIGING_HPP

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/Covariance.hpp"
#include "libKriging/Kriging.hpp"
#include "libKriging/Trend.hpp"
#include "libKriging/libKriging_exports.h"

/** Multi-fidelity Kriging: recursive AR(1) co-kriging (Le Gratiet, 2013).
 *
 * s fidelity levels, t=1 the cheapest/least accurate, t=s the reference code:
 *
 *   Z_1(x) ~ GP(f_1(x)' b_1, s_1^2 r_1(.,.;th_1))
 *   Z_t(x) = rho_{t-1}(x) Z_{t-1}(x) + d_t(x),  d_t independent of Z_{t-1}
 *
 * Under NESTED designs D_s subset ... subset D_1, the joint likelihood
 * factorizes into s independent likelihoods, so the fit reduces to s ordinary
 * Kriging fits on residuals: O(sum n_t^3) instead of O((sum n_t)^3).
 *
 * rho is estimated as a trend coefficient by augmenting the level-t regression
 * matrix with y_{t-1}(D_t) * g(x) columns (option (a) of DESIGN.md), or by an
 * outer profiling loop (option (b)).
 *
 * Prediction is recursive:
 *   mu_t(x)  = rho_{t-1}(x) mu_{t-1}(x) + mu_{d_t}(x)
 *   var_t(x) = rho_{t-1}(x)^2 var_{t-1}(x) + var_{d_t}(x)
 *
 * Restrictions to enforce and document:
 *   - nested designs required (D2: strict, or opt-in approximation);
 *   - `normalize` must be global, never per level;
 *   - level ordering: index 0 == LOWEST fidelity (classic source of mistakes);
 *   - s >= 2.
 */
class MultiFidelityKriging {
 public:
  /// Form of the scaling factor rho between consecutive levels.
  enum class RhoModel {
    Constant,  ///< rho(x) = rho_0                       (default)
    Linear,    ///< rho(x) = rho_0 + sum_k rho_k x_k
  };

  LIBKRIGING_EXPORT static RhoModel rhoModelFromString(const std::string& s);
  LIBKRIGING_EXPORT static std::string rhoModelToString(RhoModel m);

  MultiFidelityKriging() = delete;

  LIBKRIGING_EXPORT explicit MultiFidelityKriging(const std::string& covType);

  // ---------------------------------------------------------------------------
  // D3 (OPEN): two candidate signatures for fit. Pick ONE before writing the
  // bindings -- this is the single most expensive decision of the project.
  //
  //   Variant A: list-of-levels. Natural, but a brand-new marshalling
  //   convention must be written for all 5 bindings (painful in the Julia flat
  //   C ABI and the Octave mex layer).
  //
  //   Variant B: flat data + level index vector. Reuses the existing (y, X)
  //   marshalling everywhere, at the cost of a slightly less natural API and an
  //   internal split. Likely halves the binding cost.
  //
  // Both are sketched below; delete the loser.
  // ---------------------------------------------------------------------------

  /// Variant A -- y[t] is n_t, X[t] is n_t x d ; index 0 == lowest fidelity.
  LIBKRIGING_EXPORT void fit(const std::vector<arma::vec>& y,
                             const std::vector<arma::mat>& X,
                             RhoModel rho_model = RhoModel::Constant,
                             const Trend::RegressionModel& regmodel = Trend::RegressionModel::Constant,
                             bool normalize = false,
                             const std::string& optim = "BFGS",
                             const std::string& objective = "LL",
                             const std::vector<Kriging::Parameters>& parameters = {});

  /// Variant B -- y is n, X is n x d, level is n with values in [0, s-1];
  /// level 0 == lowest fidelity.
  LIBKRIGING_EXPORT void fit(const arma::vec& y,
                             const arma::mat& X,
                             const arma::uvec& level,
                             RhoModel rho_model = RhoModel::Constant,
                             const Trend::RegressionModel& regmodel = Trend::RegressionModel::Constant,
                             bool normalize = false,
                             const std::string& optim = "BFGS",
                             const std::string& objective = "LL",
                             const std::vector<Kriging::Parameters>& parameters = {});

  /** Recursive prediction at the highest fidelity level.
   * @return (mean [q], stdev [q]) ; stdev empty if return_stdev=false. */
  LIBKRIGING_EXPORT std::tuple<arma::vec, arma::vec> predict(const arma::mat& X_n, bool return_stdev = true);

  /** Recursive prediction at an arbitrary level (0-based, 0 == lowest). */
  LIBKRIGING_EXPORT std::tuple<arma::vec, arma::vec> predict(const arma::mat& X_n,
                                                             arma::uword level,
                                                             bool return_stdev = true);

  /** Joint recursive simulation: simulate Z_1, then compose
   * Z_t = rho_{t-1} * Z_{t-1} + d_t with the seed propagated PATH BY PATH
   * (required for jointly consistent trajectories).
   * @return q x nsim matrix at the highest fidelity level. */
  LIBKRIGING_EXPORT arma::mat simulate(int nsim, int seed, const arma::mat& X_n);

  /** Add observations at ONE fidelity level.
   * Levels below `level` are untouched; `level` is updated; levels above are
   * refitted (nesting and the y_{t-1}(D_t) column may both have changed). */
  LIBKRIGING_EXPORT void update(const arma::vec& y_u, const arma::mat& X_u, arma::uword level, bool refit = true);

  /// Sum of the per-level log-likelihoods (exact joint LL under nested designs).
  LIBKRIGING_EXPORT double logLikelihood();

  LIBKRIGING_EXPORT std::string summary() const;

  // --- accessors -------------------------------------------------------------
  [[nodiscard]] const std::string& kernel() const { return m_covType; }
  [[nodiscard]] arma::uword nb_levels() const { return m_submodels.size(); }
  [[nodiscard]] RhoModel rho_model() const { return m_rho_model; }
  /// Estimated rho coefficients between level t and t+1 (size nb_levels()-1).
  [[nodiscard]] const std::vector<arma::vec>& rho() const { return m_rho; }
  /// Per-level residual submodel (level 0 models Z_1 itself, level t>0 models d_t).
  [[nodiscard]] LIBKRIGING_EXPORT const Kriging& submodel(arma::uword t) const;

 private:
  // configuration
  std::string m_covType;
  RhoModel m_rho_model = RhoModel::Constant;
  Trend::RegressionModel m_regmodel = Trend::RegressionModel::Constant;
  bool m_normalize = false;

  // data, per level (index 0 == lowest fidelity)
  std::vector<arma::mat> m_X;
  std::vector<arma::vec> m_y;

  // submodels: m_submodels[0] models Z_1, m_submodels[t>0] models delta_t
  std::vector<std::unique_ptr<Kriging>> m_submodels;

  /// rho coefficients, m_rho[t] links level t to level t+1 (size s-1).
  /// Length of each entry: 1 for Constant, d+1 for Linear.
  std::vector<arma::vec> m_rho;

  bool m_is_fitted = false;

  // helpers
  /// Throw unless D_s subset ... subset D_1 (D2: unless the approximate mode
  /// has been explicitly enabled).
  void check_nested_designs() const;
  /// Row indices of D_t inside D_{t-1}.
  [[nodiscard]] arma::uvec nested_indices(arma::uword t) const;
  /// rho_t evaluated at the rows of X_n (q-vector).
  [[nodiscard]] arma::vec eval_rho(arma::uword t, const arma::mat& X_n) const;
  /// Augmented regression matrix [ diag(y_{t-1}(D_t)) G | F ] -- option (a) only.
  [[nodiscard]] arma::mat augmented_trend_matrix(arma::uword t) const;
};

#endif  // LIBKRIGING_MULTIFIDELITYKRIGING_HPP
