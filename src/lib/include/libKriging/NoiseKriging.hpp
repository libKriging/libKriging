#ifndef LIBKRIGING_NOISEKRIGING_HPP
#define LIBKRIGING_NOISEKRIGING_HPP

#include <optional>
#include <utility>

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/Trend.hpp"
#include "libKriging/utils/ExplicitCopySpecifier.hpp"

#include "libKriging/libKriging_exports.h"

// Putting this struct inside Kriging gives the following error:
// error: default member initializer for 'is_sigma2_estim' needed within
//    definition of enclosing class 'NoiseKriging' outside of member functions
struct NoiseKrigingParameters {
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
class NoiseKriging {
  NoiseKriging() = delete;
  NoiseKriging(const NoiseKriging& other)
      = default;  // Should be specialized is non default copy constructor is required

 public:
  using Parameters = NoiseKrigingParameters;

  [[nodiscard]] const std::string& kernel() const { return m_covType; };
  [[nodiscard]] const std::string& optim() const { return m_optim; };
  [[nodiscard]] const std::string& objective() const { return m_objective; };
  [[nodiscard]] const arma::mat& X() const { return m_X; };
  [[nodiscard]] const arma::rowvec& centerX() const { return m_centerX; };
  [[nodiscard]] const arma::rowvec& scaleX() const { return m_scaleX; };
  [[nodiscard]] const arma::vec& y() const { return m_y; };
  [[nodiscard]] const double& centerY() const { return m_centerY; };
  [[nodiscard]] const double& scaleY() const { return m_scaleY; };
  [[nodiscard]] const bool& normalize() const { return m_normalize; };
  [[nodiscard]] const arma::vec& noise() const { return m_noise; };
  [[nodiscard]] const Trend::RegressionModel& regmodel() const { return m_regmodel; };
  [[nodiscard]] const arma::mat& F() const { return m_F; };
  [[nodiscard]] const arma::mat& T() const { return m_T; };
  [[nodiscard]] const arma::mat& M() const { return m_M; };
  [[nodiscard]] const arma::vec& z() const { return m_z; };
  [[nodiscard]] const arma::vec& beta() const { return m_beta; };
  [[nodiscard]] const bool& is_beta_estim() const { return m_est_beta; };
  [[nodiscard]] const arma::vec& theta() const { return m_theta; };
  [[nodiscard]] const bool& is_theta_estim() const { return m_est_theta; };
  [[nodiscard]] const double& sigma2() const { return m_sigma2; };
  [[nodiscard]] const bool& is_sigma2_estim() const { return m_est_sigma2; };

 private:
  // Main model data
  std::string m_covType;
  arma::mat m_X;
  arma::rowvec m_centerX;
  arma::rowvec m_scaleX;
  arma::vec m_y;
  double m_centerY;
  double m_scaleY;
  bool m_normalize;
  arma::vec m_noise;
  Trend::RegressionModel m_regmodel;
  std::string m_optim;
  std::string m_objective;

  // Auxiliary data
  arma::mat m_dX;
  arma::vec m_maxdX;
  arma::mat m_F;
  arma::mat m_T;
  arma::mat m_R;  // required for the "update" methods
  arma::mat m_M;
  arma::mat m_star;
  arma::mat m_circ;
  arma::vec m_z;
  arma::vec m_beta;
  bool m_est_beta;
  arma::vec m_theta;
  bool m_est_theta;
  double m_sigma2;
  bool m_est_sigma2;

  // Simulation stored data
  arma::mat lastsim_Xn_n;
  arma::mat lastsim_y_n;
  arma::vec lastsim_with_noise;
  int lastsim_nsim;
  int lastsim_seed;
  arma::mat lastsim_F_n;
  arma::mat lastsim_R_nn;
  arma::mat lastsim_L_oCn;
  arma::mat lastsim_L_nCn;
  arma::mat lastsim_L_on;
  arma::mat lastsim_Rinv_on;
  arma::mat lastsim_F_on;
  arma::mat lastsim_Fstar_on;
  arma::mat lastsim_circ_on;
  arma::mat lastsim_Fcirc_on;
  arma::mat lastsim_Fhat_nKo;
  arma::mat lastsim_Ecirc_nKo;

  // Updated simulation stored data
  arma::mat lastsimup_Xn_u;
  arma::mat lastsimup_noise_u;
  arma::mat lastsimup_y_u;
  arma::mat lastsimup_Wtild_nKu;
  arma::mat lastsimup_R_uo;
  arma::mat lastsimup_R_un;
  arma::mat lastsimup_R_uu;

  std::function<double(const arma::vec&, const arma::vec&)> _Cov;
  std::function<arma::vec(const arma::vec&, const arma::vec&)> _DlnCovDtheta;
  std::function<arma::vec(const arma::vec&, const arma::vec&)> _DlnCovDx;
  double _Cov_pow;

  // This will create the dist(xi,xj) function above. Need to parse "kernel".
  void make_Cov(const std::string& covType);

 public:
  struct KModel {
    arma::mat R;
    arma::mat L;
    arma::mat Linv;
    arma::mat Fstar;
    arma::vec ystar;
    arma::mat Rstar;
    arma::mat Qstar;
    arma::vec Estar;
    double SSEstar;
    arma::vec betahat;
  };
  NoiseKriging::KModel make_Model(const arma::vec& theta,
                                  const double sigma2,
                                  std::map<std::string, double>* bench) const;

  double _logLikelihood(const arma::vec& _theta,
                        arma::vec* grad_out,
                        NoiseKriging::KModel* okm_data,
                        std::map<std::string, double>* bench) const;

  // at least, just call make_dist(kernel)
  LIBKRIGING_EXPORT NoiseKriging(const std::string& covType);

  NoiseKriging(NoiseKriging&&) = default;

  LIBKRIGING_EXPORT NoiseKriging(const arma::vec& y,
                                 const arma::vec& noise,
                                 const arma::mat& X,
                                 const std::string& covType,
                                 const Trend::RegressionModel& regmodel = Trend::RegressionModel::Constant,
                                 bool normalize = false,
                                 const std::string& optim = "BFGS",
                                 const std::string& objective = "LL",
                                 const Parameters& parameters = {});

  LIBKRIGING_EXPORT NoiseKriging(const NoiseKriging& other, ExplicitCopySpecifier);

  LIBKRIGING_EXPORT arma::mat covMat(const arma::mat& X1, const arma::mat& X2);

  /** Fit the kriging object on (X,y):
   * @param y is n length column vector of output
   * @param noise is n length column vector of output variances
   * @param X is n*d matrix of input
   * @param regmodel is the regression model to be used for the GP mean (choice between contant, linear, quadratic)
   * @param optim is an optimizer name from OptimLib, or 'none' to keep parameters unchanged
   * @param objective is 'LOO' or 'LL'. Ignored if optim=='none'.
   * @param parameters starting paramteters for optim, or final values if optim=='none'.
   */
  LIBKRIGING_EXPORT void fit(const arma::vec& y,
                             const arma::vec& noise,
                             const arma::mat& X,
                             const Trend::RegressionModel& regmodel = Trend::RegressionModel::Constant,
                             bool normalize = false,
                             const std::string& optim = "BFGS",
                             const std::string& objective = "LL",
                             const Parameters& parameters = {});

  LIBKRIGING_EXPORT std::tuple<double, arma::vec> logLikelihoodFun(const arma::vec& theta,
                                                                   bool return_grad,
                                                                   bool bench);

  LIBKRIGING_EXPORT double logLikelihood();

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
   * @param with_noise is m length column vector of output variances
   * @param will_update store useful data for possible future update
   * @return output is m*nsim matrix of simulations at X_n
   */
  LIBKRIGING_EXPORT arma::mat simulate(int nsim,
                                       int seed,
                                       const arma::mat& X_n,
                                       const arma::vec& with_noise,
                                       const bool will_update = false);

  /** Temporary assimilate new conditional data points to already conditioned (X,y), then re-simulate to previous X_n
   * @param y_u is m length column vector of new output
   * @param noise_u is m length column vector of new output variances
   * @param X_u is m*d matrix of new input
   * @return output is m*nsim matrix of simulations at X_n
   */
  LIBKRIGING_EXPORT arma::mat update_simulate(const arma::vec& y_u, const arma::vec& noise_u, const arma::mat& X_u);

  /** Add new conditional data points to previous (X,y)
   * @param y_u is m length column vector of new output
   * @param noise_u is m length column vector of new output variances
   * @param X_u is m*d matrix of new input
   * @param refit is true if re-fit the model after data update
   */
  LIBKRIGING_EXPORT void update(const arma::vec& y_u,
                                const arma::vec& noise_u,
                                const arma::mat& X_u,
                                const bool refit = true);

  LIBKRIGING_EXPORT std::string summary() const;

  /** Dump current NoiseKriging object into an file
   * @param filename
   */
  LIBKRIGING_EXPORT void save(const std::string filename) const;

  /** Load a new NoiseKriging object from an file
   * @param filename
   */
  LIBKRIGING_EXPORT static NoiseKriging load(const std::string filename);
};

#endif  // LIBKRIGING_NOISEKRIGING_HPP
