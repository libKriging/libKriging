#ifndef LIBKRIGING_ORDINARYKRIGING_HPP
#define LIBKRIGING_ORDINARYKRIGING_HPP

#include <armadillo>

#include "libKriging/libKriging_exports.h"
// #include "covariance.h"

/** Ordinary kriging regression
 * @ingroup Regression
 */
class OrdinaryKriging {
 public:
  struct Parameters {
    double sigma2;
    bool has_sigma2;
    arma::vec theta;
    bool has_theta;
  };

  enum class RegressionModel { Constant, Linear, Quadratic };

  struct RegressionModelUtils {
    LIBKRIGING_EXPORT static RegressionModel fromString(const std::string& s);
    LIBKRIGING_EXPORT static std::string toString(const RegressionModel& m);
  };

  const arma::mat& X() const { return m_X; };
  const arma::rowvec& centerX() const { return m_centerX; };
  const arma::rowvec& scaleX() const { return m_scaleX; };
  const arma::colvec& y() const { return m_y; };
  const double& centerY() const { return m_centerY; };
  const double& scaleY() const { return m_scaleY; };
  const RegressionModel& regmodel() const { return m_regmodel; };
  const arma::mat& F() const { return m_F; };
  const arma::mat& T() const { return m_T; };
  const arma::mat& M() const { return m_M; };
  const arma::colvec& z() const { return m_z; };
  const arma::colvec& beta() const { return m_beta; };
  const arma::vec& theta() const { return m_theta; };
  const double& sigma2() const { return m_sigma2; };

 private:
  arma::mat m_X;
  arma::rowvec m_centerX;
  arma::rowvec m_scaleX;
  arma::colvec m_y;
  double m_centerY;
  double m_scaleY;
  RegressionModel m_regmodel;
  arma::mat m_F;
  arma::mat m_T;
  arma::mat m_M;
  arma::colvec m_z;
  arma::colvec m_beta;
  arma::vec m_theta;
  double m_sigma2;
  std::function<double(arma::subview_col<double>&&, arma::subview_col<double>&&)> CovNorm_fun;
  std::function<double(arma::subview_col<double>&&, arma::subview_col<double>&&, int)> CovNorm_deriv;
  std::function<double(arma::subview_col<double>&&, arma::subview_col<double>&&, int)> CovNorm_derivFactor;

  // returns distance matrix form Xp to X
  LIBKRIGING_EXPORT arma::mat Cov(const arma::mat& X, const arma::mat& Xp);
  LIBKRIGING_EXPORT arma::mat Cov(const arma::mat& X);
  //  // same for one point
  //  LIBKRIGING_EXPORT arma::colvec Cov(const arma::mat& X, const arma::rowvec& x, const arma::colvec& theta);

  // This will create the dist(xi,xj) function above. Need to parse "kernel".
  void make_Cov(const std::string& covType);

 public:
  struct OKModel {
    arma::mat T;
    arma::mat M;
    arma::colvec z;
    arma::colvec beta;
  };

  double logLikelihood(const arma::vec& _theta, arma::vec* grad_out, OrdinaryKriging::OKModel* okm_data) const;
  double leaveOneOut(const arma::vec& _theta, arma::vec* grad_out, OrdinaryKriging::OKModel* okm_data) const;

  // at least, just call make_dist(kernel)
  LIBKRIGING_EXPORT OrdinaryKriging(const std::string& covType);

  /** Fit the kriging object on (X,y):
   * @param y is n length column vector of output
   * @param X is n*d matrix of input
   * @param regmodel is the regression model to be used for the GP mean (choice between contant, linear, quadratic)
   * @param parameters is starting value for hyper-parameters
   * @param optim_method is an optimizer name from OptimLib, or 'none' to keep parameters unchanged
   * @param optim_objective is 'loo' or 'loglik'. Ignored if optim_method=='none'.
   */
  LIBKRIGING_EXPORT void fit(const arma::colvec& y,
                             const arma::mat& X,
                             const RegressionModel& regmodel = RegressionModel::Constant,
                             bool normalize = false,
                             const std::string& optim = "BFGS",
                             const std::string& objective = "MLE");  //,
  // const Parameters& parameters,
  // const std::string& optim_method,
  // const std::string& optim_objective);

  LIBKRIGING_EXPORT double logLikelihoodFun(const arma::vec& theta);
  LIBKRIGING_EXPORT arma::vec logLikelihoodGrad(const arma::vec& theta);
  LIBKRIGING_EXPORT double leaveOneOutFun(const arma::vec& theta);
  LIBKRIGING_EXPORT arma::vec leaveOneOutGrad(const arma::vec& theta);

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
   * @return output is m*nsim matrix of simulations at Xp
   */
  LIBKRIGING_EXPORT arma::mat simulate(const int nsim, const arma::mat& Xp);

  /** Add new conditional data points to previous (X,y)
   * @param newy is m length column vector of new output
   * @param newX is m*d matrix of new input
   * @param optim_method is an optimizer name from OptimLib, or 'none' to keep previously estimated parameters unchanged
   * @param optim_objective is 'loo' or 'loglik'. Ignored if optim_method=='none'.
   */
  LIBKRIGING_EXPORT void update(const arma::vec& newy,
                                const arma::mat& newX,
                                const std::string& optim_method,
                                const std::string& optim_objective);
};

#endif  // LIBKRIGING_ORDINARYKRIGING_HPP
