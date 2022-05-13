#ifndef LIBKRIGING_NUGGETKRIGING_HPP
#define LIBKRIGING_NUGGETKRIGING_HPP

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/Trend.hpp"

#include "libKriging/libKriging_exports.h"

/** Ordinary kriging regression
 * @ingroup Regression
 */
class NuggetKriging {
 public:
  struct Parameters {
    arma::vec nugget;
    bool has_nugget;
    bool is_nugget_estim;
    arma::vec sigma2;
    bool has_sigma2;
    bool is_sigma2_estim;
    arma::mat theta;
    bool has_theta;
    bool is_theta_estim;
    arma::colvec beta;
    bool has_beta;
    bool is_beta_estim;

    Parameters()
        : nugget(arma::vec()),
          has_nugget(false),
          is_nugget_estim(true),
          sigma2(arma::vec()),
          has_sigma2(false),
          is_sigma2_estim(true),
          theta(arma::mat()),
          has_theta(false),
          is_theta_estim(true),
          beta(arma::vec()),
          has_beta(false),
          is_beta_estim(true) {}

    Parameters(arma::vec n,
               bool h_n,
               bool e_n,
               arma::vec s2,
               bool h_s2,
               bool e_s2,
               arma::mat t,
               bool h_t,
               bool e_t,
               arma::vec b,
               bool h_b,
               bool e_b)
        : nugget(n),
          has_nugget(h_n),
          is_nugget_estim(e_n),
          sigma2(s2),
          has_sigma2(h_s2),
          is_sigma2_estim(e_s2),
          theta(t),
          has_theta(h_t),
          is_theta_estim(e_t),
          beta(b),
          has_beta(h_b),
          is_beta_estim(e_b) {}
  };

  const std::string& kernel() const { return m_covType; };
  const std::string& optim() const { return m_optim; };
  const std::string& objective() const { return m_objective; };
  const arma::mat& X() const { return m_X; };
  const arma::rowvec& centerX() const { return m_centerX; };
  const arma::rowvec& scaleX() const { return m_scaleX; };
  const arma::colvec& y() const { return m_y; };
  const double& centerY() const { return m_centerY; };
  const double& scaleY() const { return m_scaleY; };
  const Trend::RegressionModel& regmodel() const { return m_regmodel; };
  const arma::mat& F() const { return m_F; };
  const arma::mat& T() const { return m_T; };
  const arma::mat& M() const { return m_M; };
  const arma::colvec& z() const { return m_z; };
  const arma::colvec& beta() const { return m_beta; };
  const bool& is_beta_estim() const { return m_est_beta; };
  const arma::vec& theta() const { return m_theta; };
  const bool& is_theta_estim() const { return m_est_theta; };
  const double& sigma2() const { return m_sigma2; };
  const bool& is_sigma2_estim() const { return m_est_sigma2; };
  const double& nugget() const { return m_nugget; };
  const bool& is_nugget_estim() const { return m_est_nugget; };

 private:
  std::string m_covType;
  arma::mat m_X;
  arma::rowvec m_centerX;
  arma::rowvec m_scaleX;
  arma::colvec m_y;
  double m_centerY;
  double m_scaleY;
  Trend::RegressionModel m_regmodel;
  std::string m_optim;
  std::string m_objective;
  arma::mat m_dX;
  arma::mat m_F;
  arma::mat m_T;
  arma::mat m_M;
  arma::colvec m_z;
  arma::colvec m_beta;
  bool m_est_beta;
  arma::vec m_theta;
  bool m_est_theta;
  double m_sigma2;
  bool m_est_sigma2;
  double m_nugget;
  bool m_est_nugget;
  std::function<double(const arma::vec&)>
      CovNorm_fun;  // dist_norm is L1 distance between to points of X, divided by theta
  std::function<arma::vec(const arma::vec&)> Dln_CovNorm;
  double CovNorm_pow;  // power factor used in hessian

  // This will create the dist(xi,xj) function above. Need to parse "kernel".
  void make_Cov(const std::string& covType);

 public:
  struct OKModel {
    arma::mat T;
    arma::mat M;
    arma::colvec z;
    arma::colvec beta;
    bool is_beta_estim;
    double sigma2;
    bool is_sigma2_estim;
    double nugget;
    bool is_nugget_estim;
    double var;
  };

  double _logLikelihood(const arma::vec& _theta, arma::vec* grad_out, NuggetKriging::OKModel* okm_data) const;
  double _logMargPost(const arma::vec& _theta, arma::vec* grad_out, NuggetKriging::OKModel* okm_data) const;

  // at least, just call make_dist(kernel)
  LIBKRIGING_EXPORT NuggetKriging(const std::string& covType);

  LIBKRIGING_EXPORT NuggetKriging(const arma::colvec& y,
                                  const arma::mat& X,
                                  const std::string& covType,
                                  const Trend::RegressionModel& regmodel = Trend::RegressionModel::Constant,
                                  bool normalize = false,
                                  const std::string& optim = "BFGS",
                                  const std::string& objective = "LL",
                                  const Parameters& parameters = Parameters{});

  /** Fit the kriging object on (X,y):
   * @param y is n length column vector of output
   * @param X is n*d matrix of input
   * @param regmodel is the regression model to be used for the GP mean (choice between contant, linear, quadratic)
   * @param optim is an optimizer name from OptimLib, or 'none' to keep parameters unchanged
   * @param objective is 'LOO' or 'LL'. Ignored if optim=='none'.
   * @param parameters starting paramteters for optim, or final values if optim=='none'.
   */
  LIBKRIGING_EXPORT void fit(const arma::colvec& y,
                             const arma::mat& X,
                             const Trend::RegressionModel& regmodel = Trend::RegressionModel::Constant,
                             bool normalize = false,
                             const std::string& optim = "BFGS",
                             const std::string& objective = "LL",
                             const Parameters& parameters = Parameters{});

  LIBKRIGING_EXPORT std::tuple<double, arma::vec> logLikelihoodFun(const arma::vec& theta, const bool grad);

  LIBKRIGING_EXPORT std::tuple<double, arma::vec> logMargPostFun(const arma::vec& theta, const bool grad);

  LIBKRIGING_EXPORT double logLikelihood();
  LIBKRIGING_EXPORT double logMargPost();

  /** Compute the prediction for given points X'
   * @param Xp is m*d matrix of points where to predict output
   * @param std is true if return also stdev column vector
   * @param cov is true if return also cov matrix between Xp
   * @return output prediction: m means, [m standard deviations], [m*m full covariance matrix]
   */
  LIBKRIGING_EXPORT std::tuple<arma::colvec, arma::colvec, arma::mat> predict(const arma::mat& Xp,
                                                                              bool withStd,
                                                                              bool withCov);

  /** Draw sample trajectories of kriging at given points X'
   * @param Xp is m*d matrix of points where to simulate output
   * @param nsim is number of simulations to draw
   * @param seed random seed setup for sample simulations
   * @return output is m*nsim matrix of simulations at Xp
   */
  LIBKRIGING_EXPORT arma::mat simulate(const int nsim, const int seed, const arma::mat& Xp);

  /** Add new conditional data points to previous (X,y)
   * @param newy is m length column vector of new output
   * @param newX is m*d matrix of new input
   */
  LIBKRIGING_EXPORT void update(const arma::vec& newy, const arma::mat& newX);

  LIBKRIGING_EXPORT std::string summary() const;
};

#endif  // LIBKRIGING_NUGGETKRIGING_HPP
