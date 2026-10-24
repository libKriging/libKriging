#ifndef LIBKRIGING_KRIGING_HPP
#define LIBKRIGING_KRIGING_HPP

#include <functional>
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

  // populate_Model with an explicit Cholesky jitter (noise-free diagonal)
  void populate_Model_jitter(Kriging::KModel& m,
                             const arma::vec& theta,
                             double extra_param,
                             double chol_jitter,
                             std::map<std::string, double>* bench) const;
  // the committed factorization (m_model_chol_jitter), not the objective's
  Kriging::KModel make_committed_Model(const arma::vec& theta, double extra_param) const;

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
                            const Parameters& parameters = {},
                            const std::string& method = "exact");

  LIBKRIGING_EXPORT Kriging(const Kriging& other, ExplicitCopySpecifier);

  /** Deprecation message of a deprecated fit objective ("LLVecchia(m)",
   * "LLNystrom(k)"), empty if `objective` is not deprecated (malformed
   * specs throw std::invalid_argument). */
  LIBKRIGING_EXPORT static std::string deprecated_objective_message(const std::string& objective);

  /** Receiver of deprecation messages (deprecated objective aliases), called
   * at every fit using one. Default (empty handler): printed once per process
   * on arma::cout. The bindings route the messages to their native warning
   * mechanism (Python DeprecationWarning, R warning, Julia depwarn, Octave
   * warning). A handler may throw to turn the warning into an error. */
  using DeprecationHandler = std::function<void(const std::string&)>;
  LIBKRIGING_EXPORT static void set_deprecation_handler(DeprecationHandler handler);

  /** Canonical (objective, method) pair of a fit: method gets its explicit
   * size ("vecchia" -> "vecchia(30)", "nystrom" -> "nystrom(50)"), the
   * deprecated objectives "LLVecchia(m)" / "LLNystrom(k)" become ("LL",
   * "vecchia(m)") / ("LL", "nystrom(k)") (with a deprecation warning), and
   * unknown names raise std::invalid_argument. Every objective ("LL",
   * "LOO", "LMP") accepts every method. */
  LIBKRIGING_EXPORT static std::pair<std::string, std::string> fit_objective_method(const std::string& objective,
                                                                                    const std::string& method);

  [[nodiscard]] NoiseModel noise_model() const { return m_noise_model; }
  [[nodiscard]] bool is_nugget_estim() const { return m_est_nugget; }
  /// Nugget mode: alpha = sigma2 / (sigma2 + nugget) used by the noisy
  /// correlation (scale-free: identical with normalize=true or false)
  [[nodiscard]] double alpha() const { return m_alpha; }
  /// Fixed jitter of the committed factorization (predict / simulate / update);
  /// chol_jitter() is the one of the objective (fit, logLikelihood).
  [[nodiscard]] double model_chol_jitter() const { return m_model_chol_jitter; }

  /** Parameter accessors, always in the ORIGINAL scale of X and y, so that
   * they round-trip with `Parameters` (which are read in the original scale
   * too), whatever `normalize`. With normalize=true the model is held
   * internally on normalized data (X -> (X - centerX) / scaleX,
   * y -> (y - centerY) / scaleY); these accessors convert back:
   *   theta  = theta_n % scaleX,
   *   sigma2 = sigma2_n * scaleY^2, nugget = nugget_n * scaleY^2,
   *   noise  = noise_n * scaleY^2,
   *   beta   such that F(x) * beta == scaleY * F(x_n) * beta_n + centerY
   *          (exact for every trend; the None trend has no coefficient, its
   *          implicit mean is centerY).
   * With normalize=false they return the fitted values unchanged. */
  [[nodiscard]] LIBKRIGING_EXPORT arma::vec theta() const;
  [[nodiscard]] LIBKRIGING_EXPORT double sigma2() const;
  [[nodiscard]] LIBKRIGING_EXPORT arma::vec beta() const;
  [[nodiscard]] LIBKRIGING_EXPORT double nugget() const;
  [[nodiscard]] LIBKRIGING_EXPORT arma::vec noise() const;

  /** Covariance sigma2 * C(X1, X2) between two sets of points, in the
   * original scale of X (inputs) and y (sigma2). */
  LIBKRIGING_EXPORT arma::mat covMat(const arma::mat& X1, const arma::mat& X2);

  /** Fit the kriging object on (X,y):
   * @param y is n length column vector of output
   * @param X is n*d matrix of input
   * @param regmodel is the regression model to be used for the GP mean (choice between contant, linear, quadratic)
   * @param optim is an optimizer name from OptimLib, or 'none' to keep parameters unchanged
   * @param objective is the criterion: 'LL' (log-likelihood, default), 'LOO'
   *        (leave-one-out; see docs/math/LOO.md) or 'LMP' (log-marginal
   *        posterior; see docs/math/LMP.md). Ignored if optim=='none'.
   *        'LLVecchia(m)' / 'LLNystrom(k)' are deprecated aliases of
   *        objective='LL' with method='vecchia(m)' / 'nystrom(k)'.
   * @param parameters starting paramteters for optim, or final values if optim=='none'.
   * @param method is how the covariance is evaluated, for the fit AND for
   *        what the model keeps afterwards: 'exact' (default: O(n^3) dense
   *        Cholesky, every exact method available afterwards),
   *        'vecchia'/'vecchia(m)' (Vecchia approximation with m conditioning
   *        neighbors, default 30: O(n m^3), no O(n^2) matrix; see
   *        docs/math/Vecchia.md) or 'nystrom'/'nystrom(k)' (rank-k Nystrom
   *        approximation, default 50: O(n k^2); see docs/math/Nystrom.md).
   *        An approximate fit keeps no exact factorization: predict /
   *        simulate / logLikelihood then need the same method, and the
   *        exact model at the fitted theta is a separate fit
   *        (objective='LL', method='exact', optim='none', theta given).
   *        Approximate methods are implemented for the objectives 'LL',
   *        'LOO' (LOO under the approximate joint law) and 'LMP' (approximate
   *        log det R and R^-1). With objective 'LL', both 'vecchia(m)'
   *        (response Vecchia) and 'nystrom(k)' support every noise model
   *        (None, Nugget, Heterogeneous); 'LOO' / 'LMP' with an approximate
   *        method require noise model None: other combinations raise
   *        std::invalid_argument.
   */
  LIBKRIGING_EXPORT void fit(const arma::vec& y,
                             const arma::mat& X,
                             const Trend::RegressionModel& regmodel = Trend::RegressionModel::Constant,
                             bool normalize = false,
                             const std::string& optim = "BFGS",
                             const std::string& objective = "LL",
                             const Parameters& parameters = {},
                             const std::string& method = "exact");

  // Heterogeneous-noise variant: noise is known per-observation
  LIBKRIGING_EXPORT void fit(const arma::vec& y,
                             const arma::vec& noise,
                             const arma::mat& X,
                             const Trend::RegressionModel& regmodel = Trend::RegressionModel::Constant,
                             bool normalize = false,
                             const std::string& optim = "BFGS",
                             const std::string& objective = "LL",
                             const Parameters& parameters = {},
                             const std::string& method = "exact");

  /** Objective functions at a given theta (gamma = [theta] without noise,
   * [theta, alpha] with a nugget, [theta, sigma2] with a heterogeneous
   * noise), all in the ORIGINAL scale of X and y, like the accessors: with
   * normalize=true, theta (and a heterogeneous-noise sigma2) are converted
   * to the internal scale and the gradient is chain-ruled back
   * (d/dtheta_i = d/dtheta_n_i / scaleX_i, d/dsigma2 = d/dsigma2_n /
   * scaleY^2; alpha is scale-free). Values are those of the original data:
   *   logLikelihood: log-density of the original y (= normalized one
   *     - n * log(scaleY)), same for the Vecchia / Nystrom variants;
   *   leaveOneOut: mean squared LOO error in y units (= normalized * scaleY^2);
   *   logMargPost: normalized one - (n - p) * log(scaleY) + log|det A| (A the
   *     trend basis change), i.e. the value a normalize=false fit gives.
   * So logLikelihood() == logLikelihoodFun(theta()), and both agree with a
   * normalize=false model at the same (original) parameters. */
  LIBKRIGING_EXPORT std::tuple<double, arma::vec> logLikelihoodFun(const arma::vec& gamma,
                                                                   bool return_grad,
                                                                   bool bench);

  LIBKRIGING_EXPORT std::tuple<double, arma::vec> leaveOneOutFun(const arma::vec& theta, bool return_grad, bool bench);

  LIBKRIGING_EXPORT std::tuple<double, arma::vec> logMargPostFun(const arma::vec& theta, bool return_grad, bool bench);

  LIBKRIGING_EXPORT double logLikelihood();
  /** Log-likelihood at the committed parameters, computed with an explicit
   * method: "exact" (= logLikelihood()), "vecchia" (logLikelihoodVecchiaFun,
   * requires a method="vecchia(m)" fit) or "nystrom" (logLikelihoodNystromFun,
   * requires a method="nystrom(k)" fit). */
  LIBKRIGING_EXPORT double logLikelihood(const std::string& method);
  LIBKRIGING_EXPORT double leaveOneOut();
  /** Leave-one-out criterion at the committed theta, computed with an explicit
   * method: "exact" (= leaveOneOut()), "vecchia" (leaveOneOutVecchiaFun,
   * requires a method="vecchia(m)" fit) or "nystrom" (leaveOneOutNystromFun,
   * requires a method="nystrom(k)" fit). The approximate methods are noise
   * model None only: on a Nugget / Heterogeneous model they raise
   * std::invalid_argument (like fit with objective "LOO" and an approximate
   * method). See docs/math/LOO.md. */
  LIBKRIGING_EXPORT double leaveOneOut(const std::string& method);
  LIBKRIGING_EXPORT double logMargPost();
  /** Log-marginal posterior at the committed theta, computed with an explicit
   * method: "exact" (= logMargPost()), "vecchia" (logMargPostVecchiaFun) or
   * "nystrom" (logMargPostNystromFun), with the same fit requirements as
   * leaveOneOut(method) (approximate methods: noise model None only). See
   * docs/math/LMP.md. */
  LIBKRIGING_EXPORT double logMargPost(const std::string& method);

  LIBKRIGING_EXPORT std::tuple<arma::vec, arma::vec> leaveOneOutVec(const arma::vec& theta);
  /** Leave-one-out predictions (mean, stdev) at given theta with an explicit
   * method ("exact", "vecchia" or "nystrom", same requirements as
   * leaveOneOut(method)): under the approximate joint law, the LOO mean of
   * y_i is y_i - (Qy)_i / Q_ii and its stdev sqrt(sigma2 / Q_ii); theta and
   * the results in the original scale, as leaveOneOutVec(theta). */
  LIBKRIGING_EXPORT std::tuple<arma::vec, arma::vec> leaveOneOutVec(const arma::vec& theta, const std::string& method);

  /** Approximate LOO / LMP below: original scale, as leaveOneOutFun /
   * logMargPostFun: with normalize=true theta is read in the original scale
   * of X, the value is the one of the original data (LOO in y units squared,
   * LMP of the original y) and the gradient is chain-ruled back, so that
   * leaveOneOut("vecchia") == leaveOneOutVecchiaFun(theta()) etc., and a
   * normalize=true model gives the normalize=false values whenever the
   * approximation itself is scale-free (all predecessors as Vecchia
   * neighbors, all points as Nystrom landmarks). Noise model None only: on a
   * Nugget / Heterogeneous model they raise std::invalid_argument. */
  /** Leave-one-out criterion under the Vecchia joint law (precision
   * B' D^-1 B) at given theta; requires a method="vecchia(m)" fit.
   * @return (loo, gradient) ; analytic gradient, empty if return_grad=false. */
  LIBKRIGING_EXPORT std::tuple<double, arma::vec> leaveOneOutVecchiaFun(const arma::vec& theta, bool return_grad);
  /** Leave-one-out criterion under the Nystrom joint law (covariance
   * U U' + diag(D)) at given theta; requires a method="nystrom(k)" fit.
   * @return (loo, gradient) ; analytic gradient, empty if return_grad=false. */
  LIBKRIGING_EXPORT std::tuple<double, arma::vec> leaveOneOutNystromFun(const arma::vec& theta, bool return_grad);
  /** Log-marginal posterior with the Vecchia approximation of R (log det R
   * = sum log D, R^-1 = B' D^-1 B) at given theta; requires a
   * method="vecchia(m)" fit. @return (lmp, gradient) ; analytic gradient. */
  LIBKRIGING_EXPORT std::tuple<double, arma::vec> logMargPostVecchiaFun(const arma::vec& theta, bool return_grad);
  /** Log-marginal posterior with the Nystrom approximation of R (Woodbury
   * solves and determinant lemma) at given theta; requires a
   * method="nystrom(k)" fit. @return (lmp, gradient) ; analytic gradient. */
  LIBKRIGING_EXPORT std::tuple<double, arma::vec> logMargPostNystromFun(const arma::vec& theta, bool return_grad);

  /** Vecchia approximated log-likelihood at given theta (d values) or,
   * with a noise model, gamma = [theta, alpha] (Nugget) / [theta, sigma2]
   * (Heterogeneous) like logLikelihoodFun (theta alone: the committed alpha /
   * sigma2). Requires the model to have been fitted with method="vecchia" or
   * "vecchia(m)" (the Vecchia sets are rebuilt on demand, e.g. after load).
   * Original scale, as logLikelihoodFun: with normalize=true theta and a
   * heterogeneous sigma2 are read in the original scale of X and y (alpha
   * is scale-free), the value is the log-density of the original y and the
   * gradient is chain-ruled back. (The Vecchia ordering and neighbor sets
   * are built in the internal, normalized input space.)
   * @return (vll, gradient) ; gradient (same size as theta) empty if
   *         return_grad=false. */
  LIBKRIGING_EXPORT std::tuple<double, arma::vec> logLikelihoodVecchiaFun(const arma::vec& theta, bool return_grad);

  /// Number of Vecchia conditioning neighbors (0 = not fitted with method "vecchia")
  [[nodiscard]] arma::uword vecchia_neighbors() const { return m_vecchia_m; }

  /** Deprecated: a method="vecchia(m)" fit never keeps the exact O(n^3)
   * factorization any more (the model is "light"); fit the exact model with
   * method="exact" instead. set_vecchia_exact_commit(false) is a no-op,
   * set_vecchia_exact_commit(true) raises std::invalid_argument. */
  [[deprecated("a Vecchia fit is always light: fit with method=\"exact\" for the exact model")]] LIBKRIGING_EXPORT void
  set_vecchia_exact_commit(bool b);
  [[deprecated("a Vecchia fit is always light")]] [[nodiscard]] bool vecchia_exact_commit() const { return false; }
  /// True when the model is a Vecchia fit without exact factorization
  [[nodiscard]] bool is_vecchia_light() const { return m_vecchia_light; }

  /** Vecchia (local) prediction: each point of X_n is kriged on its m nearest
   * observations only — O(q m^3) instead of O(q n^2), embarrassingly parallel.
   * Mean is universal-kriging-style with the committed beta; variance is the
   * simple-kriging one (beta treated as known). Usable after any fit. Noise
   * models follow predict(): the neighborhood block is the noisy correlation;
   * Nugget predicts a noisy observation (variance sigma2 + nugget, an observed
   * location returns its observation), Heterogeneous the latent process.
   * @param m number of conditioning neighbors (0 = vecchia_neighbors() if
   *          fitted with method "vecchia", else 30)
   * @return (mean [q], stdev [q]) ; stdev empty if return_stdev=false. */
  LIBKRIGING_EXPORT std::tuple<arma::vec, arma::vec> predictVecchia(const arma::mat& X_n,
                                                                    bool return_stdev = true,
                                                                    arma::uword m = 0);

  /** Vecchia (local, sequential) simulation: draws joint sample trajectories
   * at X_n by sequential conditioning, in the given row order of X_n: the
   * t-th simulation point is drawn from its simple-kriging conditional
   * (beta treated as known, like predictVecchia) given its m nearest
   * neighbors among the observations AND the t-1 points already simulated
   * (response-first Vecchia, Katzfuss et al. 2020). O(q (n + q) d + q m^3)
   * instead of O(q^3): usable for large n AND large q, after any fit
   * (including a Vecchia fit, where the exact simulate is not). Exact (up to beta known) when
   * m >= n + q - 1; otherwise long-range correlations are truncated and
   * the result depends on the row order of X_n. No `will_update` (no
   * update_simulate on top of it). Noise models follow the exact simulate()
   * conventions: the observations are noisy responses, the simulated values
   * are the latent process unless `with_nugget` (Nugget: noisy response, as
   * simulate(..., with_nugget, ...)); `with_noise` adds independent noise
   * afterwards (noise variances, as simulate(..., with_noise, ...)).
   * @param m number of conditioning neighbors (0 = vecchia_neighbors() if
   *          fitted with method "vecchia", else 30)
   * @param with_nugget Nugget model only: simulate the noisy response
   * @param with_noise empty, a scalar or one noise VARIANCE per row of X_n
   *          (like fit's / update's noise), in the original scale of y
   *          (whatever normalize): Gaussian noise N(0, with_noise) added to
   *          the simulations; negative values raise std::invalid_argument
   * @return output is n_n*nsim matrix of simulations at X_n */
  LIBKRIGING_EXPORT arma::mat simulateVecchia(int nsim,
                                              int seed,
                                              const arma::mat& X_n,
                                              arma::uword m = 0,
                                              bool with_nugget = false,
                                              const arma::vec& with_noise = arma::vec());

  /** Nystrom (global low-rank) approximated log-likelihood at given theta
   * (fit method "nystrom(k)"). Requires the model to have been fitted with
   * method="nystrom" or "nystrom(k)".
   * @return (ll, gradient) ; gradient empty if return_grad=false. Gradient is
   *         analytic (envelope theorem + Woodbury trace identities, same
   *         principle as the Vecchia gradient) -- see the derivation inside
   *         _logLikelihoodNystrom's grad_out block in Kriging.cpp.
   * For the Nugget / Heterogeneous noise models `theta` may also be the full
   * gamma = [theta, alpha] / [theta, sigma2] (gradient then of size d+1,
   * like logLikelihoodFun); with theta only, the committed alpha / sigma2 is
   * used. Original scale, like logLikelihoodFun: with normalize=true, theta
   * (and a heterogeneous sigma2) are converted to the internal scale, the
   * value is that of the original y and the gradient is chain-ruled back. */
  LIBKRIGING_EXPORT std::tuple<double, arma::vec> logLikelihoodNystromFun(const arma::vec& theta, bool return_grad);

  /// Nystrom rank used at fit time (0 = not fitted with method "nystrom")
  [[nodiscard]] arma::uword nystrom_rank() const { return m_nystrom_k; }
  /// True when the current fit is a Nystrom (no exact O(n^3) factorization) fit
  [[nodiscard]] bool is_nystrom_light() const { return m_nystrom_light; }

  /** Nystrom (global low-rank) prediction: uses the committed rank-k factors
   * (U, D) from the method="nystrom(k)" fit via the Woodbury identity instead of the
   * exact O(n^2) triangular solve — O(n*k*q) instead of O(n^2*q) for q
   * prediction points. Mean is universal-kriging-style with the committed
   * beta; variance is the simple-kriging one (beta treated as known, like
   * predictVecchia). Only usable after a method="nystrom(k)" fit.
   * @return (mean [q], stdev [q]) ; stdev empty if return_stdev=false. */
  LIBKRIGING_EXPORT std::tuple<arma::vec, arma::vec> predictNystrom(const arma::mat& X_n, bool return_stdev = true);

  /** Nystrom (global low-rank) simulation: draws joint sample trajectories at
   * X_n using the committed rank-k factors (U, D) via Woodbury for the mean
   * and the (dense, but only n_n x n_n -- X_n is expected to be small) joint
   * predictive covariance among the SIMULATION points. Like predictNystrom,
   * the covariance is the simple-kriging one (beta treated as known). Only
   * usable after a method="nystrom(k)" fit; does not support `will_update` (no
   * update_simulate for Nystrom fits).
   * @return output is n_n*nsim matrix of simulations at X_n */
  LIBKRIGING_EXPORT arma::mat simulateNystrom(int nsim, int seed, const arma::mat& X_n);

  /** Compute the (exact) prediction for given points X'. Requires the exact
   * factorization: on a Vecchia or Nystrom fit it raises an
   * error (use the `method` overload, or predictVecchia / predictNystrom).
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
  /** Prediction with an explicit method: "exact" (predict above), "vecchia"
   * or "vecchia(m)" (predictVecchia, m = 0 -> vecchia_neighbors() or 30),
   * "nystrom" (predictNystrom, requires a method="nystrom" fit). The approximate
   * methods return mean / stdev only: return_cov / return_deriv raise. */
  LIBKRIGING_EXPORT std::tuple<arma::vec, arma::vec, arma::mat, arma::mat, arma::mat>
  predict(const arma::mat& X_n, bool return_stdev, bool return_cov, bool return_deriv, const std::string& method);

  /** Draw observed trajectories of kriging at given points X_n (exact
   * simulation; raises on a Vecchia or Nystrom fit, see the
   * `method` overload)
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
  // Heterogeneous-mode variant: with_noise holds observation-noise VARIANCES
  // (original y scale, like fit's / update's noise: a scalar or one per row of
  // X_n; negative values raise std::invalid_argument); N(0, with_noise) noise
  // is added to the (latent) simulations. update_simulate reuses it.
  LIBKRIGING_EXPORT arma::mat simulate(int nsim,
                                       int seed,
                                       const arma::mat& X_n,
                                       const arma::vec& with_noise,
                                       const bool will_update = false);
  /** Simulation with an explicit method: "exact" (simulate above), "vecchia"
   * or "vecchia(m)" (simulateVecchia), "nystrom" (simulateNystrom, requires an
   * method="nystrom" fit). will_update=true is only available with "exact". */
  LIBKRIGING_EXPORT arma::mat simulate(int nsim,
                                       int seed,
                                       const arma::mat& X_n,
                                       const std::string& method,
                                       const bool will_update = false);
  /// Nugget-mode simulation with an explicit method ("exact": simulate with
  /// with_nugget above; "vecchia" / "vecchia(m)": simulateVecchia;
  /// "nystrom": simulateNystrom, latent process only, so with_nugget=true
  /// raises std::invalid_argument)
  LIBKRIGING_EXPORT arma::mat simulate(int nsim,
                                       int seed,
                                       const arma::mat& X_n,
                                       const bool with_nugget,
                                       const bool will_update,
                                       const std::string& method);
  /// Heterogeneous-mode simulation with an explicit method ("exact": simulate
  /// with with_noise above; "vecchia" / "vecchia(m)": simulateVecchia;
  /// "nystrom": simulateNystrom, latent process only, so a non-zero
  /// with_noise raises std::invalid_argument)
  LIBKRIGING_EXPORT arma::mat simulate(int nsim,
                                       int seed,
                                       const arma::mat& X_n,
                                       const arma::vec& with_noise,
                                       const bool will_update,
                                       const std::string& method);
  // const char* overload: without it, a string literal method would bind to
  // the (with_nugget) bool overload above through the pointer->bool conversion
  LIBKRIGING_EXPORT arma::mat simulate(int nsim,
                                       int seed,
                                       const arma::mat& X_n,
                                       const char* method,
                                       const bool will_update = false) {
    return simulate(nsim, seed, X_n, std::string(method), will_update);
  }

  /** Temporary assimilate new conditional data points to already conditioned (X,y), then re-simulate to previous X_n
   * @param y_u is m length column vector of new output
   * @param X_u is m*d matrix of new input
   * @return
   *
   * put is m*nsim matrix of simulations at X_n
   * Needs a previous simulate(..., will_update=true) on the current model:
   * fit() and update() drop it, after which this raises std::runtime_error
   * ("No previous simulation data available"). A Heterogeneous model needs
   * the noise of the new data: this overload raises std::invalid_argument
   * (before any change) on it; use update_simulate(y_u, noise_u, X_u).
   */
  LIBKRIGING_EXPORT arma::mat update_simulate(const arma::vec& y_u, const arma::mat& X_u);
  // Heterogeneous-mode variant: noise_u is per-observation noise VARIANCE for
  // the new points (like update's noise_u); the with_noise variances of the
  // last simulate(..., with_noise, true) are added to the result again
  LIBKRIGING_EXPORT arma::mat update_simulate(const arma::vec& y_u, const arma::vec& noise_u, const arma::mat& X_u);

  /** Add new conditional data points to previous (X,y)
   * @param y_u is m length column vector of new output
   * @param X_u is m*d matrix of new input
   * @param refit is true if re-fit the model after data update
   * A Heterogeneous model needs the noise of the new data: this overload
   * raises std::invalid_argument (before any change) on it; use
   * update(y_u, noise_u, X_u).
   */
  LIBKRIGING_EXPORT void update(const arma::vec& y_u, const arma::mat& X_u, const bool refit = true);
  // Heterogeneous-mode variant: noise_u is per-observation noise VARIANCE for new points
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
  // JSON (de)serialization behind save()/load(), also used by NestedKriging
  // to embed its submodels in its own save file.
  friend class NestedKriging;
  void dump_to_json(nlohmann::json& j) const;
  /// @param filename only used in error messages
  static Kriging load_from_json(const nlohmann::json& j, const std::string& filename);

  NoiseModel m_noise_model = NoiseModel::None;
  double m_nugget = 0.0;
  bool m_est_nugget = false;
  double m_alpha = 1.0;
  /// Jitter of the committed exact factorization (m_T); m_chol_jitter is the
  /// objective's. Noise-free only. See commit_model_chol_jitter.
  double m_model_chol_jitter = 0.0;  // sigma2/(sigma2+nugget) — only meaningful in Nugget mode
  // Simulate state for Nugget/Heterogeneous modes (mirrors NuggetKriging/NoiseKriging)
  bool m_lastsim_with_nugget = false;
  arma::vec m_lastsim_with_noise;
  // Update-points cache key complement: noise_u of the last update_simulate
  // (empty for the noise-free form). The base cache is keyed on X_u only.
  arma::vec m_lastsimup_noise_u;
  /// KrigingImpl::clear_simulate_state plus the members above
  void clear_simulate_state();
  /// After an exact noise-free optimization (run with the search jitter
  /// LinearAlgebra::kernel_chol_jitter): rebuild the committed model at
  /// m_theta with LinearAlgebra::model_chol_jitter.
  void commit_model_chol_jitter();

  using FitOfn = std::function<double(const arma::vec&, arma::vec*, KModel*)>;
  FitOfn make_fit_objective(const std::string& objective) const;
  /// Validates the noise model against the canonical (objective, method)
  /// pair, and returns the internal objective name evaluated by
  /// make_fit_objective ("LL", "LOO", "LMP", or "<objective>Vecchia(m)" /
  /// "<objective>Nystrom(k)", e.g. "LOOVecchia(30)", "LMPNystrom(50)").
  std::string internal_fit_objective(const std::string& objective, const std::string& method) const;
  /// Internal objective name of the committed (m_objective, m_method) pair
  [[nodiscard]] std::string internal_objective() const;

  // --- Vecchia approximated likelihood (method="vecchia(m)") -----------------
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
  /// analytic gradient via the envelope theorem. _gamma = [theta] (None),
  /// [theta, alpha] (Nugget) or [theta, sigma2] (Heterogeneous; a d-vector
  /// uses the committed alpha / sigma2), noise models by "response" Vecchia
  /// (blocks of the noisy correlation, see docs/math/Vecchia.md). Optional
  /// out-params expose the profiled beta, the scale s of the noisy
  /// correlation (sigma2 for None / Heterogeneous, sigma2 + nugget for
  /// Nugget) and Q/n (the concentrated scale), used by the light-mode commit.
  double _logLikelihoodVecchia(const arma::vec& _gamma,
                               arma::vec* grad_out,
                               arma::vec* beta_out = nullptr,
                               double* sigma2_out = nullptr,
                               double* q_over_n_out = nullptr) const;
  /// Light Vecchia commit at (theta, extra = alpha / sigma2, ignored for
  /// None): beta, sigma2 (and nugget, alpha) profiled by the Vecchia
  /// likelihood with the exact noise-model fit conventions; no factorization.
  void commit_vecchia(const arma::vec& theta, double extra);
  bool m_vecchia_light = false;  ///< current fit is a Vecchia fit without exact factorization
  /// Throw if the model is a light Vecchia fit (used by update_simulate)
  void check_not_vecchia_light(const char* what) const;
  /// Throw if the model has no exact factorization (light Vecchia or Nystrom
  /// fit): `what` is the exact method called, `alternative` the method to use
  void check_exact_available(const char* what, const char* alternative) const;
  /// Light-Vecchia incremental update: extends m_X/m_y/m_F, rebuilds the
  /// Vecchia sets, optionally re-optimizes theta (single warm-start BFGS on
  /// the LLVecchia objective), then re-profiles beta/sigma2 -- no exact
  /// factorization and no O(n^2) matrix, like update_nystrom.
  void update_vecchia_light(const arma::vec& y_u, const arma::mat& X_u, bool refit);
  /// True when the (normalized) output is, to rounding, in the span of the
  /// trend matrix: zero residual whatever theta (see fit()).
  [[nodiscard]] bool y_in_trend_span() const;

  // --- Nystrom approximated likelihood (method="nystrom(k)") ---------------
  // Like Vecchia, Nystrom NEVER factorizes R exactly: the committed model
  // only carries the rank-k factors (m_nystrom_U, m_nystrom_D) from the last
  // likelihood evaluation at theta*, so predict/simulate/update behave like a
  // permanent "light" fit (see m_nystrom_light below).
  arma::uword m_nystrom_k = 0;  ///< rank k (0 = Nystrom mode off)
  arma::mat m_nystrom_U;        ///< committed low-rank factor (n x k): R ~= U*U.t() + diag(D) (noise included)
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
  ///
  /// `_gamma` is theta (d) or, for the Nugget / Heterogeneous noise models,
  /// [theta, alpha] / [theta, sigma2] (d+1; when only theta is given the
  /// committed alpha / sigma2 is used), exactly as the exact _logLikelihood.
  /// The approximated correlation matrix of the observations is
  /// Rhat = U*U.t() + diag(D) for every noise model (Nugget:
  /// alpha*Ct + (1-alpha)*I, Heterogeneous: Ct + diag(noise/sigma2), with
  /// Ct = U0*U0.t() + diag(1 - diag(U0*U0.t())) the Nystrom correlation):
  /// U_out / D_out are the factors of Rhat. sigma2_out is the variance scale
  /// tv of Cov(y) = tv * Rhat (None: sigma2; Nugget: sigma2 + nugget;
  /// Heterogeneous: sigma2), sse_out the GLS residual quadratic form
  /// (y - F beta)' Rhat^-1 (y - F beta).
  double _logLikelihoodNystrom(const arma::vec& _gamma,
                               arma::vec* grad_out = nullptr,
                               arma::vec* beta_out = nullptr,
                               double* sigma2_out = nullptr,
                               arma::mat* U_out = nullptr,
                               arma::vec* D_out = nullptr,
                               double* sse_out = nullptr) const;
  /// Commit a Nystrom fit at gamma (theta [, alpha | sigma2]) as an optimizer
  /// would (BFGS / update conventions of the exact noise-model fits): theta,
  /// beta, sigma2 (and alpha / nugget for the Nugget model) and the rank-k
  /// factors of Rhat, profiled by the Nystrom likelihood at gamma.
  void commit_nystrom(const arma::vec& gamma);
  bool m_nystrom_light = false;  ///< true whenever m_nystrom_k > 0 (no exact factorization ever exists)

  // --- LOO / LMP with an approximate method (see docs/math/LOO.md, LMP.md) ---
  /// Leave-one-out under the Vecchia (vecchia=true) or Nystrom joint law at
  /// theta, with analytic gradient; yhat_out (n x 2: LOO mean, LOO stdev / sigma)
  double _leaveOneOutApprox(bool vecchia, const arma::vec& _theta, arma::vec* grad_out, arma::mat* yhat_out) const;
  /// Log-marginal posterior with the Vecchia / Nystrom approximation of R
  double _logMargPostApprox(bool vecchia, const arma::vec& _theta, arma::vec* grad_out) const;
  /// Commit beta / sigma2 (and the Nystrom factors) at m_theta for an
  /// approximate fit: beta = GLS under the approximate precision, sigma2 =
  /// SSE / n, or SSE / (n - p) for objective "LMP" when lmp_dof (a BFGS
  /// fit / refit, like the exact LMP commit). Vecchia: delegates to
  /// commit_vecchia(m_theta, extra) (extra = alpha | sigma2 with a noise model)
  void commit_approx(bool lmp_dof, double extra = 0.0);
  /// Public approximate LOO (loo=true) / LMP at an original-scale theta:
  /// original-data value and chain-ruled gradient (see leaveOneOutVecchiaFun)
  std::tuple<double, arma::vec> approx_objective_fun(bool loo,
                                                     bool vecchia,
                                                     const arma::vec& theta,
                                                     bool return_grad,
                                                     const char* what);
  /// Throw unless the model was fitted with the Vecchia (vecchia=true) /
  /// Nystrom method; rebuilds the Vecchia sets if needed (after load)
  void ensure_approx_sets(bool vecchia, const char* what);
  /// Throw std::invalid_argument if the model has a noise model (Nugget /
  /// Heterogeneous): LOO / LMP with an approximate method are noise-free only
  /// (same restriction as fit with objective "LOO" / "LMP")
  void check_approx_noise_free(bool loo, bool vecchia, const char* what) const;
  void check_approx_theta(const arma::vec& theta, const char* what) const;
  /// LOO value factor from the internal to the original scale (scaleY^2)
  [[nodiscard]] double loo_scale2() const;
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

  // --- original <-> internal (normalized) scale of the public API ----------
  /// gamma (theta [, alpha | sigma2]) from the original to the internal scale
  [[nodiscard]] arma::vec gamma_from_original(const arma::vec& gamma) const;
  /// chain rule: internal-scale gradient -> original-scale gradient (in place)
  void grad_to_original(arma::vec& grad) const;
  /// original-data log-likelihood minus normalized one (-n log(scaleY))
  [[nodiscard]] double ll_offset() const;
  /// original-scale log marginal posterior minus normalized one
  [[nodiscard]] double lmp_offset() const;
  /// internal-scale value of logMargPost at gamma (augments alpha in Nugget mode)
  double _logMargPostAt(const arma::vec& gamma_n, arma::vec* grad_out, std::map<std::string, double>* bench) const;
  /// fit body; `noise` (original scale) is the heterogeneous noise, or nullptr
  void fit_impl(const arma::vec& y,
                const arma::vec* noise,
                const arma::mat& X,
                const Trend::RegressionModel& regmodel,
                bool normalize,
                const std::string& optim,
                const std::string& objective,
                const Parameters& parameters,
                const std::string& method);

  // Returns dimension of the optimization parameter vector (d for None, d+1 for Nugget/Heterogeneous)
  arma::uword gamma_dim() const;
  // Build current gamma from member state
  arma::vec current_gamma() const;
};

#endif  // LIBKRIGING_KRIGING_HPP
