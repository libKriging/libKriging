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
 */
class Kriging : public KrigingImpl {
  Kriging() = delete;
  Kriging(const Kriging& other) = default;  // Should be specialized if non default copy constructor is required

 public:
  using Parameters = KrigingParameters;
  using KModel = KrigingImpl::KModel;

  /// Which noise treatment is in use.
  enum class NoiseModel {
    None,           ///< pure GP: R = corr(theta)
    Nugget,         ///< homogeneous nugget: R = alpha*corr + (1-alpha)*I
    Heterogeneous,  ///< known per-obs noise: R = corr + diag(noise/sigma2)
  };

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
   * @param objective is 'LOO' or 'LL'. Ignored if optim=='none'.
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

  // Returns dimension of the optimization parameter vector (d for None, d+1 for Nugget/Heterogeneous)
  arma::uword gamma_dim() const;
  // Build current gamma from member state
  arma::vec current_gamma() const;
};

#endif  // LIBKRIGING_KRIGING_HPP
