#ifndef LIBKRIGING_NUGGETKRIGING_HPP
#define LIBKRIGING_NUGGETKRIGING_HPP

#include <optional>
#include <utility>

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/KrigingImpl.hpp"
#include "libKriging/Trend.hpp"
#include "libKriging/utils/ExplicitCopySpecifier.hpp"

#include "libKriging/libKriging_exports.h"

// Putting this struct inside Kriging gives the following error:
// error: default member initializer for 'is_sigma2_estim' needed within
//    definition of enclosing class 'NuggetKriging' outside of member functions
struct NuggetKrigingParameters {
  std::optional<arma::vec> nugget;
  bool is_nugget_estim = true;
  std::optional<arma::vec> sigma2;
  bool is_sigma2_estim = true;
  std::optional<arma::mat> theta;
  bool is_theta_estim = true;
  std::optional<arma::vec> beta;
  bool is_beta_estim = true;
};

/** Ordinary kriging regression
 * @ingroup Regression
 */
class NuggetKriging : public KrigingImpl {
 private:
  NuggetKriging() = delete;
  NuggetKriging(const NuggetKriging& other)
      = default;  // Should be specialized is non default copy constructor is required

 public:
  using Parameters = NuggetKrigingParameters;
  using KModel = KrigingImpl::KModel;

  [[nodiscard]] const double& nugget() const { return m_nugget; };
  [[nodiscard]] const bool& is_nugget_estim() const { return m_est_nugget; };

 private:
  // Nugget-specific state
  double m_nugget{};
  bool m_est_nugget{};

  // Nugget-specific simulation cache
  bool lastsim_with_nugget{};

 public:
  void populate_Model(KModel& m,
                      const arma::vec& theta,
                      const double alpha,
                      std::map<std::string, double>* bench) const;
  NuggetKriging::KModel make_Model(const arma::vec& theta,
                                   const double alpha,
                                   std::map<std::string, double>* bench) const;

  double _logLikelihood(const arma::vec& _theta,
                        arma::vec* grad_out,
                        NuggetKriging::KModel* okm_data,
                        std::map<std::string, double>* bench) const;
  double _logMargPost(const arma::vec& _theta,
                      arma::vec* grad_out,
                      NuggetKriging::KModel* okm_data,
                      std::map<std::string, double>* bench) const;

  // at least, just call make_dist(kernel)
  LIBKRIGING_EXPORT NuggetKriging(const std::string& covType);

  NuggetKriging(NuggetKriging&&) = default;

  LIBKRIGING_EXPORT NuggetKriging(const arma::vec& y,
                                  const arma::mat& X,
                                  const std::string& covType,
                                  const Trend::RegressionModel& regmodel = Trend::RegressionModel::Constant,
                                  bool normalize = false,
                                  const std::string& optim = "BFGS",
                                  const std::string& objective = "LL",
                                  const Parameters& parameters = {});

  LIBKRIGING_EXPORT NuggetKriging(const NuggetKriging& other, ExplicitCopySpecifier);

  static std::function<arma::vec(const arma::vec&)> reparam_to;
  static std::function<arma::vec(const arma::vec&)> reparam_from;
  static std::function<arma::vec(const arma::vec&, const arma::vec&)> reparam_from_deriv;

  static double alpha_lower;
  static double alpha_upper;

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

  LIBKRIGING_EXPORT std::tuple<double, arma::vec> logLikelihoodFun(const arma::vec& theta,
                                                                   bool return_grad,
                                                                   bool bench);

  LIBKRIGING_EXPORT std::tuple<double, arma::vec> logMargPostFun(const arma::vec& theta, bool return_grad, bool bench);

  LIBKRIGING_EXPORT double logLikelihood();
  LIBKRIGING_EXPORT double logMargPost();

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
   * @param nsim is number of simulations to draw
   * @param seed random seed setup for simulations
   * @param X_n is m*d matrix of points where to simulate output
   * @param with_nugget is true if include also nugget effect
   * @param will_update store useful data for possible future update
   * @return output is m*nsim matrix of simulations at X_n
   */
  LIBKRIGING_EXPORT arma::mat simulate(int nsim,
                                       int seed,
                                       const arma::mat& X_n,
                                       const bool with_nugget,
                                       const bool will_update = false);

  /** Temporary assimilate new conditional data points to already conditioned (X,y), then re-simulate to previous X_n
   * @param y_u is m length column vector of new output
   * @param X_u is m*d matrix of new input
   * @return output is m*nsim matrix of simulations at X_n
   */
  LIBKRIGING_EXPORT arma::mat update_simulate(const arma::vec& y_u, const arma::mat& X_u);

  /** Add new conditional data points to previous (X,y)
   * @param y_u is m length column vector of new output
   * @param X_u is m*d matrix of new input
   * @param refit is true if re-fit the model after data update
   */
  LIBKRIGING_EXPORT void update(const arma::vec& y_u, const arma::mat& X_u, const bool refit = true);

  LIBKRIGING_EXPORT std::string summary() const;

  /** Dump current NuggetKriging object into an file
   * @param filename
   */
  LIBKRIGING_EXPORT void save(const std::string filename) const;

  /** Load a new NuggetKriging object from an file
   * @param filename
   */
  LIBKRIGING_EXPORT static NuggetKriging load(const std::string filename);
};

#endif  // LIBKRIGING_NUGGETKRIGING_HPP
