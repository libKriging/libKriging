#ifndef LIBKRIGING_KRIGING_HPP
#define LIBKRIGING_KRIGING_HPP

#include <optional>
#include <utility>

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/KrigingImpl.hpp"
#include "libKriging/Trend.hpp"
#include "libKriging/utils/ExplicitCopySpecifier.hpp"

#include "libKriging/libKriging_exports.h"

// Putting this struct inside Kriging gives the following error:
// error: default member initializer for 'is_sigma2_estim' needed within
//    definition of enclosing class 'Kriging' outside of member functions
struct KrigingParameters {
  std::optional<double> sigma2;
  bool is_sigma2_estim = true;
  std::optional<arma::mat> theta;
  bool is_theta_estim = true;
  std::optional<arma::vec> beta;
  bool is_beta_estim = true;
  // Nugget mode only:
  std::optional<double> nugget;
  bool is_nugget_estim = true;
};

/** Ordinary kriging regression
 * @ingroup Regression
 *
 * See docs/math/Kriging.md for the model, objectives and prediction math,
 * and docs/math/Noise.md for the nugget / heterogeneous noise models.
 */
class Kriging : public KrigingImpl {
  Kriging() = delete;
  Kriging(const Kriging& other) = default;  // Should be specialized if non default copy constructor is required

 public:
  using Parameters = KrigingParameters;
  using KModel = KrigingImpl::KModel;

  /// Which noise treatment is in use. See docs/math/Noise.md for the math.
  enum class NoiseModel {
    None,           ///< pure GP: R = corr(theta)
    Nugget,         ///< homogeneous nugget: R = alpha*corr + (1-alpha)*I
    Heterogeneous,  ///< known per-obs noise: R = corr + diag(noise/sigma2)
  };

  /** Subset-of-data pre-fit reduction: picks `n_max` representative rows out
   * of X (by index), meant to be applied BEFORE constructing/fitting a
   * Kriging model on the reduced (X.rows(idx), y(idx)) -- a pure
   * pre-processing layer, no change to the fit engine itself. This is the
   * cheapest of libKriging's large-n options (a single k-means pass, then
   * an ordinary O(n_max^3) exact fit) at the cost of discarding
   * n - n_max points outright, unlike Vecchia/Nystrom/NestedKriging which
   * all still use every point.
   * @param X n x d design.
   * @param n_max target subset size; if n_max >= X.n_rows, returns all
   *        indices (no-op).
   * @param method "kmeans" (default): n_max k-means centroids on X, each
   *        replaced by its nearest actual (not synthetic) data point, so
   *        the subset always consists of real observations; falls back to
   *        "random" if k-means degenerates (e.g. n_max close to n_rows
   *        producing near-empty clusters). "random": uniform subsample
   *        without replacement.
   * @param seed RNG seed (k-means initialization and/or random fallback).
   * @return sorted row-indices into X (and the matching y) to keep. */
  LIBKRIGING_EXPORT static arma::uvec subsetOfData(const arma::mat& X,
                                                   arma::uword n_max,
                                                   const std::string& method = "kmeans",
                                                   int seed = 123);

  // populate_Model with member-state extra_param (alpha or sigma2)
  Kriging::KModel make_Model(const arma::vec& theta, std::map<std::string, double>* bench) const;
  void populate_Model(Kriging::KModel& m, const arma::vec& theta, std::map<std::string, double>* bench) const;

  // populate_Model with explicit extra_param (used during optimization)
  Kriging::KModel make_Model(const arma::vec& theta, double extra_param, std::map<std::string, double>* bench) const;
  void populate_Model(Kriging::KModel& m,
                      const arma::vec& theta,
                      double extra_param,
                      std::map<std::string, double>* bench) const;

  // gamma = [theta] for None, [theta, alpha] for Nugget, [theta, sigma2] for Heterogeneous
  double _logLikelihood(const arma::vec& _gamma,
                        arma::vec* grad_out,
                        Kriging::KModel* okm_data,
                        std::map<std::string, double>* bench) const;
  double _leaveOneOut(const arma::vec& _theta,
                      arma::vec* grad_out,
                      arma::mat* yhat_out,
                      Kriging::KModel* okm_data,
                      std::map<std::string, double>* bench) const;
  double _logMargPost(const arma::vec& _gamma,
                      arma::vec* grad_out,
                      Kriging::KModel* okm_data,
                      std::map<std::string, double>* bench) const;

  // at least, just call make_dist(kernel)
  LIBKRIGING_EXPORT Kriging(const std::string& covType);
  LIBKRIGING_EXPORT Kriging(const std::string& covType, NoiseModel noise_model);

  Kriging(Kriging&&) = default;

  LIBKRIGING_EXPORT Kriging(const arma::vec& y,
                            const arma::mat& X,
                            const std::string& covType,
                            const Trend::RegressionModel& regmodel = Trend::RegressionModel::Constant,
                            bool normalize = false,
                            const std::string& optim = "BFGS",
                            const std::string& objective = "LL",
                            const Parameters& parameters = {});

  LIBKRIGING_EXPORT Kriging(const Kriging& other, ExplicitCopySpecifier);

  [[nodiscard]] NoiseModel noise_model() const { return m_noise_model; }
  [[nodiscard]] double nugget() const { return m_nugget; }
  [[nodiscard]] bool is_nugget_estim() const { return m_est_nugget; }
  [[nodiscard]] const arma::vec& noise() const { return m_noise; }

  /** Fit the kriging object on (X,y):
   * @param y is n length column vector of output
   * @param X is n*d matrix of input
   * @param regmodel is the regression model to be used for the GP mean (choice between contant, linear, quadratic)
   * @param optim is an optimizer name from OptimLib, or 'none' to keep parameters unchanged
   * @param objective is 'LL' (log-likelihood, default), 'LOO' (leave-one-out;
   *        see docs/math/LOO.md), 'LMP' (log-marginal posterior; see
   *        docs/math/LMP.md), or 'LLVecchia'/'LLVecchia(m)' for the Vecchia
   *        approximated log-likelihood with m conditioning neighbors (default
   *        m=30): O(n m^3) per evaluation instead of O(n^3), recommended for
   *        large n in low dimension (see docs/math/Vecchia.md). Ignored if
   *        optim=='none'.
   * @param parameters starting paramteters for optim, or final values if optim=='none'.
   */
  LIBKRIGING_EXPORT void fit(const arma::vec& y,
                             const arma::mat& X,
                             const Trend::RegressionModel& regmodel = Trend::RegressionModel::Constant,
                             bool normalize = false,
                             const std::string& optim = "BFGS",
                             const std::string& objective = "LL",
                             const Parameters& parameters = {});

  // Heterogeneous-noise variant: noise is known per-observation
  LIBKRIGING_EXPORT void fit(const arma::vec& y,
                             const arma::vec& noise,
                             const arma::mat& X,
                             const Trend::RegressionModel& regmodel = Trend::RegressionModel::Constant,
                             bool normalize = false,
                             const std::string& optim = "BFGS",
                             const std::string& objective = "LL",
                             const Parameters& parameters = {});

  LIBKRIGING_EXPORT std::tuple<double, arma::vec> logLikelihoodFun(const arma::vec& gamma,
                                                                   bool return_grad,
                                                                   bool bench);

  LIBKRIGING_EXPORT std::tuple<double, arma::vec> leaveOneOutFun(const arma::vec& theta, bool return_grad, bool bench);

  LIBKRIGING_EXPORT std::tuple<double, arma::vec> logMargPostFun(const arma::vec& theta, bool return_grad, bool bench);

  LIBKRIGING_EXPORT double logLikelihood();
  LIBKRIGING_EXPORT double leaveOneOut();
  LIBKRIGING_EXPORT double logMargPost();

  LIBKRIGING_EXPORT std::tuple<arma::vec, arma::vec> leaveOneOutVec(const arma::vec& theta);

  /** Vecchia approximated log-likelihood at given theta (objective="LLVecchia(m)").
   * Requires the Vecchia sets to be built, i.e. the model to have been fitted
   * with objective="LLVecchia" or "LLVecchia(m)".
   * @return (vll, gradient) ; gradient empty if return_grad=false. */
  LIBKRIGING_EXPORT std::tuple<double, arma::vec> logLikelihoodVecchiaFun(const arma::vec& theta, bool return_grad);

  /// Number of Vecchia conditioning neighbors (0 = not fitted with LLVecchia)
  [[nodiscard]] arma::uword vecchia_neighbors() const { return m_vecchia_m; }

  /** Large-n mode: when set to false BEFORE a fit with objective="LLVecchia(m)",
   * the final exact O(n^3) factorization is skipped. The model then stores
   * theta* plus LLVecchia-profiled beta/sigma2, and `predict` transparently routes
   * to `predictVecchia` (mean/stdev only); return_cov/return_deriv, simulate,
   * update and save are not available on such a "light" model. */
  LIBKRIGING_EXPORT void set_vecchia_exact_commit(bool b) { m_vecchia_exact_commit = b; }
  [[nodiscard]] bool vecchia_exact_commit() const { return m_vecchia_exact_commit; }
  /// True when the current fit skipped the exact factorization (light mode)
  [[nodiscard]] bool is_vecchia_light() const { return m_vecchia_light; }

  /** Vecchia (local) prediction: each point of X_n is kriged on its m nearest
   * observations only — O(q m^3) instead of O(q n^2), embarrassingly parallel.
   * Mean is universal-kriging-style with the committed beta; variance is the
   * simple-kriging one (beta treated as known). Usable after any fit.
   * @param m number of conditioning neighbors (0 = vecchia_neighbors() if
   *          fitted with LLVecchia, else 30)
   * @return (mean [q], stdev [q]) ; stdev empty if return_stdev=false. */
  LIBKRIGING_EXPORT std::tuple<arma::vec, arma::vec> predictVecchia(const arma::mat& X_n,
                                                                    bool return_stdev = true,
                                                                    arma::uword m = 0);

  /** Nystrom (global low-rank) approximated log-likelihood at given theta
   * (objective="LLNystrom(k)"). Requires the model to have been fitted with
   * objective="LLNystrom" or "LLNystrom(k)".
   * @return (ll, gradient) ; gradient empty if return_grad=false. Gradient is
   *         analytic (envelope theorem + Woodbury trace identities, same
   *         principle as the Vecchia gradient) -- see the derivation inside
   *         _logLikelihoodNystrom's grad_out block in Kriging.cpp. */
  LIBKRIGING_EXPORT std::tuple<double, arma::vec> logLikelihoodNystromFun(const arma::vec& theta, bool return_grad);

  /// Nystrom rank used at fit time (0 = not fitted with LLNystrom)
  [[nodiscard]] arma::uword nystrom_rank() const { return m_nystrom_k; }
  /// True when the current fit is a Nystrom (no exact O(n^3) factorization) fit
  [[nodiscard]] bool is_nystrom_light() const { return m_nystrom_light; }

  /** Matrix-free (CG + stochastic log-det) approximated log-likelihood at
   * given theta (objective="LLIterative(m)"). Requires the model to have
   * been fitted with objective="LLIterative" or "LLIterative(m)". Unlike
   * LLVecchia/LLNystrom (whose approximation changes the covariance model
   * itself), this stays close to the EXACT objective -- only its log|R|
   * term is a stochastic (SLQ) estimate, everything else is a CG-converged
   * exact solve -- see docs/math/Iterative.md.
   * @return (ll, gradient) ; gradient empty if return_grad=false. Gradient
   *         via the envelope theorem + a Hutchinson trace estimator sharing
   *         the same probes as the log-determinant term -- see the
   *         derivation inside _logLikelihoodIterative's grad_out block in
   *         Kriging.cpp. */
  LIBKRIGING_EXPORT std::tuple<double, arma::vec> logLikelihoodIterativeFun(const arma::vec& theta, bool return_grad);

  /// Number of Hutchinson/SLQ probes used at fit time (0 = not fitted with LLIterative)
  [[nodiscard]] arma::uword iterative_nprobe() const { return m_iterative_nprobe; }
  /// True when the current fit is an Iterative (no exact O(n^3) factorization) fit
  [[nodiscard]] bool is_iterative_light() const { return m_iterative_light; }

  /** Nystrom (global low-rank) prediction: uses the committed rank-k factors
   * (U, D) from the LLNystrom(k) fit via the Woodbury identity instead of the
   * exact O(n^2) triangular solve — O(n*k*q) instead of O(n^2*q) for q
   * prediction points. Mean is universal-kriging-style with the committed
   * beta; variance is the simple-kriging one (beta treated as known, like
   * predictVecchia). Only usable after an "LLNystrom(k)" fit.
   * @return (mean [q], stdev [q]) ; stdev empty if return_stdev=false. */
  LIBKRIGING_EXPORT std::tuple<arma::vec, arma::vec> predictNystrom(const arma::mat& X_n, bool return_stdev = true);

  /** Nystrom (global low-rank) simulation: draws joint sample trajectories at
   * X_n using the committed rank-k factors (U, D) via Woodbury for the mean
   * and the (dense, but only n_n x n_n -- X_n is expected to be small) joint
   * predictive covariance among the SIMULATION points. Like predictNystrom,
   * the covariance is the simple-kriging one (beta treated as known). Only
   * usable after an "LLNystrom(k)" fit; does not support `will_update` (no
   * update_simulate for Nystrom fits).
   * @return output is n_n*nsim matrix of simulations at X_n */
  LIBKRIGING_EXPORT arma::mat simulateNystrom(int nsim, int seed, const arma::mat& X_n);

  /** Compute the prediction for given points X'
   * @param X_n is m*d matrix of points where to predict output
   * @param return_stdev is true if return also stdev column vector
   * @param return_cov is true if return also cov matrix between X_n
   * @param return_deriv is true if return also derivative at X_n
   * @return output prediction: m means, [m standard deviations], [m*m full covariance matrix]
   */
  LIBKRIGING_EXPORT std::tuple<arma::vec, arma::vec, arma::mat, arma::mat, arma::mat> predict(const arma::mat& X_n,
                                                                                              bool return_stdev,
                                                                                              bool return_cov,
                                                                                              bool return_deriv);

  /** Predict-only alternative to `predict` for an already-fitted model,
   * using matrix-free conjugate gradient (LinearAlgebra::conjugateGradient)
   * instead of the stored O(n^2) Cholesky factor: needs only m_X/m_y/m_F/
   * m_theta/m_beta/m_sigma2 (O(n) storage), at the cost of O(n^2 * iters)
   * compute per solve instead of a single O(n^2) triangular solve. Useful
   * when many predictions are made from a model whose dense factor either
   * was never computed (e.g. after a light Vecchia/Nystrom fit -- though
   * predictVecchia/predictNystrom are cheaper still there) or isn't worth
   * keeping resident just for predict. Mean is universal-kriging-style with
   * the committed beta; stdev requires one extra CG solve PER prediction
   * point (O(n^2 * iters * q) total) and is disabled by default for that
   * reason. Only available for NoiseModel::None.
   * @param max_iter CG iteration budget per solve (0 = 2n; n is CG's exact-arithmetic
   *        bound, but round-off on typically ill-conditioned GP covariance matrices
   *        means more iterations keep helping in practice)
   * @param tol relative residual tolerance (norm(A*x-b)/norm(b)) for early stopping
   * @param use_nystrom_precond build a rank-`precond_rank` Nystrom factor of R
   *        at the model's own (already-fitted) theta and use it as a CG
   *        preconditioner (LinearAlgebra::woodbury_solve as Pinv) -- same
   *        idea as GPyTorch's pivoted-Cholesky preconditioner: fewer CG
   *        iterations to reach `tol` on the typically ill-conditioned R,
   *        at a one-time O(n*k^2) setup cost. Off by default (matches prior
   *        behavior exactly).
   * @param precond_rank rank of that Nystrom preconditioner, if enabled.
   * @return (mean [q], stdev [q]) ; stdev empty if return_stdev=false. */
  LIBKRIGING_EXPORT std::tuple<arma::vec, arma::vec> predictCG(const arma::mat& X_n,
                                                               bool return_stdev = false,
                                                               arma::uword max_iter = 0,
                                                               double tol = 1e-8,
                                                               bool use_nystrom_precond = false,
                                                               arma::uword precond_rank = 50) const;

  /** Draw observed trajectories of kriging at given points X_n
   * @param X_n is m*d matrix of points where to simulate output
   * @param nsim is number of simulations to draw
   * @param seed random seed setup for simulations
   * @param will_update store useful data for possible future update
   * @return output is m*nsim matrix of simulations at X_n
   */
  LIBKRIGING_EXPORT arma::mat simulate(int nsim, int seed, const arma::mat& X_n, const bool will_update = false);
  // Nugget-mode variant: with_nugget controls whether nugget variance is included
  LIBKRIGING_EXPORT arma::mat simulate(int nsim,
                                       int seed,
                                       const arma::mat& X_n,
                                       const bool with_nugget,
                                       const bool will_update);
  // Heterogeneous-mode variant: with_noise is per-observation noise to add to simulations
  LIBKRIGING_EXPORT arma::mat simulate(int nsim,
                                       int seed,
                                       const arma::mat& X_n,
                                       const arma::vec& with_noise,
                                       const bool will_update = false);

  /** Temporary assimilate new conditional data points to already conditioned (X,y), then re-simulate to previous X_n
   * @param y_u is m length column vector of new output
   * @param X_u is m*d matrix of new input
   * @return
   *
   * put is m*nsim matrix of simulations at X_n
   */
  LIBKRIGING_EXPORT arma::mat update_simulate(const arma::vec& y_u, const arma::mat& X_u);
  // Heterogeneous-mode variant: noise_u is per-observation noise for new points
  LIBKRIGING_EXPORT arma::mat update_simulate(const arma::vec& y_u, const arma::vec& noise_u, const arma::mat& X_u);

  /** Add new conditional data points to previous (X,y)
   * @param y_u is m length column vector of new output
   * @param X_u is m*d matrix of new input
   * @param refit is true if re-fit the model after data update
   */
  LIBKRIGING_EXPORT void update(const arma::vec& y_u, const arma::mat& X_u, const bool refit = true);
  // Heterogeneous-mode variant: noise_u is per-observation noise for new points
  LIBKRIGING_EXPORT void update(const arma::vec& y_u,
                                const arma::vec& noise_u,
                                const arma::mat& X_u,
                                const bool refit = true);

  LIBKRIGING_EXPORT std::string summary() const;

  /** Dump current Kriging object into an file
   * @param filename
   */
  LIBKRIGING_EXPORT void save(const std::string filename) const;

  /** Load a new Kriging object from an file
   * @param filename
   */
  LIBKRIGING_EXPORT static Kriging load(const std::string filename);

 private:
  NoiseModel m_noise_model = NoiseModel::None;
  double m_nugget = 0.0;
  bool m_est_nugget = false;
  double m_alpha = 1.0;  // sigma2/(sigma2+nugget) — only meaningful in Nugget mode
  // Simulate state for Nugget/Heterogeneous modes (mirrors NuggetKriging/NoiseKriging)
  bool m_lastsim_with_nugget = false;
  arma::vec m_lastsim_with_noise;

  using FitOfn = std::function<double(const arma::vec&, arma::vec*, KModel*)>;
  FitOfn make_fit_objective(const std::string& objective) const;

  // --- Vecchia approximated likelihood (objective="LLVecchia(m)") -----------------
  // Built once per fit (maxmin ordering + m nearest previously-ordered
  // neighbors on normalized inputs), then reused for every LLVecchia evaluation.
  arma::uword m_vecchia_m = 0;                  ///< 0 = Vecchia mode off
  arma::uvec m_vecchia_order;                   ///< maxmin ordering (row indices of m_X)
  std::vector<arma::uvec> m_vecchia_neighbors;  ///< per ordered point, global row indices

  /// Parse "LLVecchia" (default m=30) or "LLVecchia(m)"; throws on malformed spec.
  static arma::uword parse_vll_m(const std::string& objective);
  /// Build m_vecchia_order / m_vecchia_neighbors from m_X (call after fit_setup_impl).
  void make_vecchia_sets();
  /// Vecchia log-likelihood with profiled sigma2 and (GLS-profiled) beta;
  /// analytic gradient in theta via the envelope theorem. Optional out-params
  /// expose the profiled beta/sigma2 (used by the light-mode commit).
  double _logLikelihoodVecchia(const arma::vec& _theta,
                               arma::vec* grad_out,
                               arma::vec* beta_out = nullptr,
                               double* sigma2_out = nullptr) const;
  bool m_vecchia_exact_commit = true;  ///< false = skip the exact factorization at commit
  bool m_vecchia_light = false;        ///< current fit is a light (non-factorized) Vecchia fit
  /// Throw if the model is a light Vecchia fit (used by simulate/update/save)
  void check_not_vecchia_light(const char* what) const;

  // --- Nystrom approximated likelihood (objective="LLNystrom(k)") ---------------
  // Unlike Vecchia (which by default still performs one exact O(n^3)
  // factorization at commit), Nystrom NEVER factorizes R exactly: the
  // committed model only carries the rank-k factors (m_nystrom_U, m_nystrom_D)
  // from the last likelihood evaluation at theta*, so predict/simulate/update
  // behave like a permanent "light" fit (see m_nystrom_light below).
  arma::uword m_nystrom_k = 0;  ///< rank k (0 = Nystrom mode off)
  arma::mat m_nystrom_U;        ///< committed low-rank factor (n x k): R ~= U*U.t() + diag(D)
  arma::vec m_nystrom_D;        ///< committed residual diagonal (n), jittered to stay > 0
  /// Landmark row-indices (into m_X), chosen ONCE per fit and held fixed
  /// across every theta evaluation during optimization. Fixing the landmark
  /// SET (as opposed to re-selecting it greedily at each theta, which is what
  /// a naive per-call nystromFactor would do) is what makes
  /// _logLikelihoodNystrom smooth in theta -- required for both the
  /// finite-difference gradient here and the analytic one planned in phase 3b.
  /// Chosen via one nystromFactor call at a theta-neutral reference kernel
  /// (isotropic range 1), i.e. purely for spatial coverage.
  arma::uvec m_nystrom_landmarks;
  /// Populate m_nystrom_landmarks from the current m_X/m_dX (call once, after
  /// fit_setup_impl, before optimization starts).
  void make_nystrom_landmarks();

  /// Parse "LLNystrom" (default k=50) or "LLNystrom(k)"; throws on malformed spec.
  static arma::uword parse_nystrom_k(const std::string& objective);
  /// Nystrom log-likelihood with profiled sigma2 and (GLS-profiled) beta, via
  /// a FIXED-landmark Nystrom factorization (R ~= R_ns * R_ss^-1 * R_ns.t(),
  /// landmarks = m_nystrom_landmarks) and Woodbury solves/logdet (no n x n
  /// matrix ever built). Analytic gradient in theta (envelope theorem, same
  /// principle as _logLikelihoodVecchia) computed in O(n*k^2 + n*k*d) when
  /// grad_out is non-null -- see the derivation in the .cpp file-level
  /// comment above this function. Optional out-params expose the profiled
  /// beta/sigma2/factors (used by the commit step and by tests).
  double _logLikelihoodNystrom(const arma::vec& _theta,
                               arma::vec* grad_out = nullptr,
                               arma::vec* beta_out = nullptr,
                               double* sigma2_out = nullptr,
                               arma::mat* U_out = nullptr,
                               arma::vec* D_out = nullptr) const;
  bool m_nystrom_light = false;  ///< true whenever m_nystrom_k > 0 (no exact factorization ever exists)
  /// Throw if the model is a Nystrom fit (used by simulate/update/save)
  void check_not_nystrom_light(const char* what) const;
  /// Nystrom-specific incremental update: extends m_X/m_y/m_F with the new
  /// data (the FIXED landmark set is still valid -- rows were appended, never
  /// reordered/removed), then either re-profiles beta/sigma2/U/D at the
  /// current theta (refit=false), or first does a warm-restart single BFGS
  /// from the current theta over the SAME landmark set (refit=true, no
  /// re-selection) before re-profiling. Both paths stay
  /// O((n_old+n_new)*k^2): no O(n^2) matrix or pairwise-difference cube is
  /// built, unlike the exact/Vecchia update() paths.
  void update_nystrom(const arma::vec& y_u, const arma::mat& X_u, bool refit);

  // --- Iterative (matrix-free CG + stochastic log-det) approximated
  // likelihood (objective="LLIterative(m)") ----------------------------------
  // Unlike Vecchia/Nystrom (whose approximation replaces R by a cheaper
  // structured model -- local conditioning / low rank), this keeps R
  // itself exact: R^-1*y, R^-1*F and the SSE/beta/sigma2 GLS terms are all
  // ordinary matrix-free CG solves (LinearAlgebra::conjugateGradient) that
  // converge to the SAME answer the exact factorization would give, just
  // without ever materializing R. Only log|R| -- the one term CG cannot
  // give directly -- is replaced by a stochastic (SLQ) estimate
  // (LinearAlgebra::stochasticLogDet), using the SAME Rademacher probes for
  // the Hutchinson trace term in the gradient. This is the same overall
  // strategy as GPyTorch's BBMM (see docs/math/Iterative.md).
  arma::uword m_iterative_nprobe = 0;  ///< number of Hutchinson/SLQ probe vectors (0 = mode off)
  /// n x nprobe Rademacher probes, drawn ONCE per fit (fixed seed) and held
  /// fixed across every theta evaluation during optimization -- same
  /// smoothness rationale as Nystrom's fixed landmarks: re-drawing fresh
  /// probes at every evaluation would make the objective noisy/non-smooth
  /// between BFGS iterations.
  arma::mat m_iterative_probes;
  arma::uword m_iterative_cg_max_iter = 0;     ///< CG budget per solve (0 = 2n, like predictCG)
  double m_iterative_cg_tol = 1e-8;            ///< CG relative residual tolerance
  arma::uword m_iterative_lanczos_steps = 20;  ///< SLQ Lanczos steps per probe

  arma::uword m_iterative_precond_rank = 0;  ///< Nystrom preconditioner rank (0 = no preconditioning)
  /// Landmark row-indices (into m_X) for the CG preconditioner, chosen ONCE
  /// per fit (same fixed-landmark rationale as m_nystrom_landmarks: re-
  /// selecting greedily at each theta would make the preconditioner --
  /// and hence the CG-converged objective/gradient -- non-smooth in theta).
  /// The preconditioner itself (R_ss/R_ns -> Woodbury Pinv) is still
  /// rebuilt from these fixed landmarks at the CURRENT theta on every call,
  /// unlike m_nystrom_U/D which are only committed once at theta*.
  arma::uvec m_iterative_precond_landmarks;

  /// Parse "LLIterative" (default m=30), "LLIterative(m)" or
  /// "LLIterative(m,precond_rank)"; throws on malformed spec. precond_rank
  /// defaults to 0 (preconditioning off) when omitted.
  static arma::uword parse_iterative_m(const std::string& objective, arma::uword* precond_rank_out = nullptr);
  /// Draw m_iterative_probes from m_X's row count (call once, after
  /// fit_setup_impl, before optimization starts).
  void make_iterative_probes();
  /// Populate m_iterative_precond_landmarks from the current m_X (call once,
  /// after fit_setup_impl, before optimization starts), same greedy
  /// reference-kernel selection as make_nystrom_landmarks.
  void make_iterative_precond_landmarks();
  /// Iterative log-likelihood with profiled sigma2 and (GLS-profiled) beta,
  /// via matrix-free CG solves and an SLQ log-determinant. Analytic gradient
  /// in theta (envelope theorem, same principle as
  /// _logLikelihoodVecchia/_logLikelihoodNystrom) computed via a shared-probe
  /// Hutchinson trace estimator when grad_out is non-null. Optional
  /// out-params expose the profiled beta/sigma2 (used by the commit step).
  double _logLikelihoodIterative(const arma::vec& _theta,
                                 arma::vec* grad_out = nullptr,
                                 arma::vec* beta_out = nullptr,
                                 double* sigma2_out = nullptr) const;
  bool m_iterative_light = false;  ///< true whenever m_iterative_nprobe > 0 (no exact factorization ever exists)
  /// Throw if the model is an Iterative fit (used by simulate/update_simulate/save;
  /// update() has its own update_iterative() incremental path instead)
  void check_not_iterative_light(const char* what) const;
  /// Iterative-specific incremental update: extends m_X/m_y/m_F with the new
  /// data (the FIXED probes are redrawn at the new n; the FIXED precond
  /// landmark set, if any, stays valid since rows are only ever appended),
  /// then either re-profiles beta/sigma2 at the current theta (refit=false),
  /// or first does a warm-restart single BFGS from the current theta
  /// (refit=true, same fixed probes/landmarks) before re-profiling. Mirrors
  /// update_nystrom's O((n_old+n_new)*...) incremental strategy.
  void update_iterative(const arma::vec& y_u, const arma::mat& X_u, bool refit);

  // Returns dimension of the optimization parameter vector (d for None, d+1 for Nugget/Heterogeneous)
  arma::uword gamma_dim() const;
  // Build current gamma from member state
  arma::vec current_gamma() const;
};

#endif  // LIBKRIGING_KRIGING_HPP
