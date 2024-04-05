#ifndef LIBKRIGING_NUGGETKRIGING_HPP
#define LIBKRIGING_NUGGETKRIGING_HPP

#include <optional>
#include <utility>

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/Trend.hpp"
#include "libKriging/utils/ExplicitCopySpecifier.hpp"

#include "libKriging/libKriging_exports.h"

// Putting this struct inside Kriging gives the following error:
// error: default member initializer for 'is_sigma2_estim' needed within
//    definition of enclosing class 'NuggetKriging' outside of member functions
struct NuggetKrigingParameters {
  std::optional<arma::fvec> nugget;
  bool is_nugget_estim = true;
  std::optional<arma::fvec> sigma2;
  bool is_sigma2_estim = true;
  std::optional<arma::fmat> theta;
  bool is_theta_estim = true;
  std::optional<arma::fvec> beta;
  bool is_beta_estim = true;
};

/** Ordinary kriging regression
 * @ingroup Regression
 */
class NuggetKriging {
 private:
  NuggetKriging() = delete;
  NuggetKriging(const NuggetKriging& other)
      = default;  // Should be specialized is non default copy constructor is required

 public:
  using Parameters = NuggetKrigingParameters;

  [[nodiscard]] const std::string& kernel() const { return m_covType; };
  [[nodiscard]] const std::string& optim() const { return m_optim; };
  [[nodiscard]] const std::string& objective() const { return m_objective; };
  [[nodiscard]] const arma::fmat& X() const { return m_X; };
  [[nodiscard]] const arma::frowvec& centerX() const { return m_centerX; };
  [[nodiscard]] const arma::frowvec& scaleX() const { return m_scaleX; };
  [[nodiscard]] const arma::fvec& y() const { return m_y; };
  [[nodiscard]] const float& centerY() const { return m_centerY; };
  [[nodiscard]] const float& scaleY() const { return m_scaleY; };
  [[nodiscard]] const bool& normalize() const { return m_normalize; };
  [[nodiscard]] const Trend::RegressionModel& regmodel() const { return m_regmodel; };
  [[nodiscard]] const arma::fmat& F() const { return m_F; };
  [[nodiscard]] const arma::fmat& T() const { return m_T; };
  [[nodiscard]] const arma::fmat& M() const { return m_M; };
  [[nodiscard]] const arma::fvec& z() const { return m_z; };
  [[nodiscard]] const arma::fvec& beta() const { return m_beta; };
  [[nodiscard]] const bool& is_beta_estim() const { return m_est_beta; };
  [[nodiscard]] const arma::fvec& theta() const { return m_theta; };
  [[nodiscard]] const bool& is_theta_estim() const { return m_est_theta; };
  [[nodiscard]] const float& sigma2() const { return m_sigma2; };
  [[nodiscard]] const bool& is_sigma2_estim() const { return m_est_sigma2; };
  [[nodiscard]] const float& nugget() const { return m_nugget; };
  [[nodiscard]] const bool& is_nugget_estim() const { return m_est_nugget; };

 private:
  std::string m_covType;
  arma::fmat m_X;
  arma::frowvec m_centerX;
  arma::frowvec m_scaleX;
  arma::fvec m_y;
  float m_centerY;
  float m_scaleY;
  bool m_normalize;
  Trend::RegressionModel m_regmodel;
  std::string m_optim;
  std::string m_objective;
  arma::fmat m_dX;
  arma::fmat m_F;
  arma::fmat m_T;
  arma::fmat m_M;
  arma::fvec m_z;
  arma::fvec m_beta;
  bool m_est_beta;
  arma::fvec m_theta;
  bool m_est_theta;
  float m_sigma2;
  bool m_est_sigma2;
  float m_nugget;
  bool m_est_nugget;
  std::function<float(const arma::fvec&, const arma::fvec&)> Cov;
  std::function<arma::fvec(const arma::fvec&, const arma::fvec&)> DlnCovDtheta;
  std::function<arma::fvec(const arma::fvec&, const arma::fvec&)> DlnCovDx;
  float Cov_pow;  // power factor used in hessian

  // This will create the dist(xi,xj) function above. Need to parse "kernel".
  void make_Cov(const std::string& covType);

 public:
  struct OKModel {
    arma::fmat T;
    arma::fmat M;
    arma::fvec z;
    arma::fvec beta;
    bool is_beta_estim;
    float sigma2;
    bool is_sigma2_estim;
    float nugget;
    bool is_nugget_estim;
    float var;
  };

  float _logLikelihood(const arma::fvec& _theta,
                        arma::fvec* grad_out,
                        NuggetKriging::OKModel* okm_data,
                        std::map<std::string, double>* bench) const;
  float _logMargPost(const arma::fvec& _theta,
                      arma::fvec* grad_out,
                      NuggetKriging::OKModel* okm_data,
                      std::map<std::string, double>* bench) const;

  // at least, just call make_dist(kernel)
  LIBKRIGING_EXPORT NuggetKriging(const std::string& covType);

  LIBKRIGING_EXPORT NuggetKriging(NuggetKriging&&) = default;

  LIBKRIGING_EXPORT NuggetKriging(const arma::fvec& y,
                                  const arma::fmat& X,
                                  const std::string& covType,
                                  const Trend::RegressionModel& regmodel = Trend::RegressionModel::Constant,
                                  bool normalize = false,
                                  const std::string& optim = "BFGS",
                                  const std::string& objective = "LL",
                                  const Parameters& parameters = Parameters{});

  LIBKRIGING_EXPORT NuggetKriging(const NuggetKriging& other, ExplicitCopySpecifier);

  static std::function<arma::fvec(const arma::fvec&)> reparam_to;
  static std::function<arma::fvec(const arma::fvec&)> reparam_from;
  static std::function<arma::fvec(const arma::fvec&, const arma::fvec&)> reparam_from_deriv;

  static float alpha_lower;
  static float alpha_upper;

  /** Fit the kriging object on (X,y):
   * @param y is n length column vector of output
   * @param X is n*d matrix of input
   * @param regmodel is the regression model to be used for the GP mean (choice between contant, linear, quadratic)
   * @param optim is an optimizer name from OptimLib, or 'none' to keep parameters unchanged
   * @param objective is 'LOO' or 'LL'. Ignored if optim=='none'.
   * @param parameters starting paramteters for optim, or final values if optim=='none'.
   */
  LIBKRIGING_EXPORT void fit(const arma::fvec& y,
                             const arma::fmat& X,
                             const Trend::RegressionModel& regmodel = Trend::RegressionModel::Constant,
                             bool normalize = false,
                             const std::string& optim = "BFGS",
                             const std::string& objective = "LL",
                             const Parameters& parameters = Parameters{});

  LIBKRIGING_EXPORT std::tuple<float, arma::fvec> logLikelihoodFun(const arma::fvec& theta, bool grad, bool bench);

  LIBKRIGING_EXPORT std::tuple<float, arma::fvec> logMargPostFun(const arma::fvec& theta, bool grad, bool bench);

  LIBKRIGING_EXPORT float logLikelihood();
  LIBKRIGING_EXPORT float logMargPost();

  /** Compute the prediction for given points X'
   * @param Xp is m*d matrix of points where to predict output
   * @param std is true if return also stdev column vector
   * @param cov is true if return also cov matrix between Xp
   * @return output prediction: m means, [m standard deviations], [m*m full covariance matrix]
   */
  LIBKRIGING_EXPORT std::tuple<arma::fvec, arma::fvec, arma::fmat, arma::fmat, arma::fmat> predict(const arma::fmat& Xp,
                                                                                                    bool withStd,
                                                                                                    bool withCov,
                                                                                                    bool withDeriv);

  /** Draw sample trajectories of kriging at given points X'
   * @param Xp is m*d matrix of points where to simulate output
   * @param nsim is number of simulations to draw
   * @param seed random seed setup for sample simulations
   * @return output is m*nsim matrix of simulations at Xp
   */
  LIBKRIGING_EXPORT arma::fmat simulate(int nsim, int seed, const arma::fmat& Xp);

  /** Add new conditional data points to previous (X,y)
   * @param newy is m length column vector of new output
   * @param newX is m*d matrix of new input
   */
  LIBKRIGING_EXPORT void update(const arma::fvec& newy, const arma::fmat& newX);

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
