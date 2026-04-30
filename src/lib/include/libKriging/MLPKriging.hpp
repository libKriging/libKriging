#ifndef LIBKRIGING_MLP_KRIGING_HPP
#define LIBKRIGING_MLP_KRIGING_HPP

/**
 * @file MLPKriging.hpp
 * @brief Kriging with a joint MLP feature map (≡ Deep Kernel Learning / NeuralKernelKriging).
 *
 * Thin facade over WarpKriging({"mlp_joint(…)"}, kernel).
 * All GP state, optimisation, and simulation live in the WarpKriging impl.
 */

#include "libKriging/Trend.hpp"
#include "libKriging/WarpKriging.hpp"
#include "libKriging/libKriging_exports.h"
#include "libKriging/utils/lk_armadillo.hpp"

#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace libKriging {

// =========================================================================
//  MLPKriging fit parameters (seed values)
// =========================================================================
struct MLPKrigingParameters {
  std::optional<arma::vec> theta;        ///< θ seed (size d_out)
  std::optional<arma::vec> warp_params;  ///< joint-MLP param seed (flat vec)
};

/**
 * @brief Kriging with a joint MLP feature map.
 *
 * Model:
 *     y(x) = f(Φ(x))^T β  +  ζ(x)
 *     Φ(x) = MLP(x ; W)          // joint, cross-variable
 *     Cov[ζ(x), ζ(x')] = σ² · k_base(Φ(x), Φ(x') ; θ)
 *
 * All parameters (W, θ, β, σ²) are optimised jointly by maximising the
 * concentrated marginal log-likelihood.  σ² and β are profiled out.
 */
class MLPKriging {
 public:
  using Parameters = MLPKrigingParameters;

  // -----------------------------------------------------------------------
  //  Construction
  // -----------------------------------------------------------------------

  LIBKRIGING_EXPORT MLPKriging(const std::vector<arma::uword>& hidden_dims,
                               arma::uword d_out = 2,
                               const std::string& activation = "selu",
                               const std::string& kernel = "gauss");

  LIBKRIGING_EXPORT MLPKriging(const arma::vec& y,
                               const arma::mat& X,
                               const std::vector<arma::uword>& hidden_dims,
                               arma::uword d_out,
                               const std::string& activation,
                               const std::string& kernel,
                               const Trend::RegressionModel& regmodel = Trend::RegressionModel::Constant,
                               bool normalize = false,
                               const std::string& optim = "BFGS+Adam",
                               const std::string& objective = "LL",
                               const std::map<std::string, std::string>& parameters = {});

  // -----------------------------------------------------------------------
  //  Fitting
  // -----------------------------------------------------------------------

  LIBKRIGING_EXPORT void fit(const arma::vec& y,
                             const arma::mat& X,
                             const Trend::RegressionModel& regmodel = Trend::RegressionModel::Constant,
                             bool normalize = false,
                             const std::string& optim = "BFGS+Adam",
                             const std::string& objective = "LL",
                             const std::map<std::string, std::string>& parameters = {});

  LIBKRIGING_EXPORT void fit(const arma::vec& y,
                             const arma::mat& X,
                             const Trend::RegressionModel& regmodel,
                             bool normalize,
                             const std::string& optim,
                             const std::string& objective,
                             const Parameters& parameters);

  // -----------------------------------------------------------------------
  //  Prediction / simulation / update
  // -----------------------------------------------------------------------

  LIBKRIGING_EXPORT std::tuple<arma::vec, arma::vec, arma::mat, arma::mat, arma::mat> predict(const arma::mat& X_n,
                                                                                              bool return_stdev = true,
                                                                                              bool return_cov = false,
                                                                                              bool return_deriv
                                                                                              = false) const;

  LIBKRIGING_EXPORT arma::mat simulate(int nsim, int seed, const arma::mat& X_n, bool will_update = false);

  LIBKRIGING_EXPORT arma::mat update_simulate(const arma::vec& y_u, const arma::mat& X_u);

  LIBKRIGING_EXPORT void update(const arma::vec& y_u, const arma::mat& X_u, const bool refit = true);

  // -----------------------------------------------------------------------
  //  Log-likelihood
  // -----------------------------------------------------------------------

  LIBKRIGING_EXPORT double logLikelihood() const;

  LIBKRIGING_EXPORT std::tuple<double, arma::vec, arma::mat> logLikelihoodFun(const arma::vec& theta,
                                                                              bool return_grad = true,
                                                                              bool return_hess = false) const;

  // -----------------------------------------------------------------------
  //  Accessors
  // -----------------------------------------------------------------------

  LIBKRIGING_EXPORT std::string summary() const;

  const arma::mat& X() const { return m_impl.X(); }
  const arma::rowvec& centerX() const { return m_impl.centerX(); }
  const arma::rowvec& scaleX() const { return m_impl.scaleX(); }
  const arma::vec& y() const { return m_impl.y(); }
  const double& centerY() const { return m_impl.centerY(); }
  const double& scaleY() const { return m_impl.scaleY(); }
  const bool& normalize() const { return m_impl.normalize(); }
  const Trend::RegressionModel& regmodel() const { return m_impl.regmodel(); }
  const arma::mat& F() const { return m_impl.F(); }
  const arma::mat& T() const { return m_impl.T(); }
  const arma::mat& M() const { return m_impl.M(); }
  const arma::vec& z() const { return m_impl.z(); }
  const arma::vec& beta() const { return m_impl.beta(); }
  std::string kernel() const { return m_impl.kernel(); }
  arma::vec theta() const { return m_impl.theta(); }
  double sigma2() const { return m_impl.sigma2(); }
  bool is_fitted() const { return m_impl.is_fitted(); }
  arma::uword feature_dim() const { return m_d_out; }

  const WarpMLPJoint& warp() const { return *m_impl.joint_warp(); }

  const std::vector<arma::uword>& hidden_dims() const { return m_hidden_dims; }
  arma::uword d_out() const { return m_d_out; }
  const std::string& activation() const { return m_activation; }

  LIBKRIGING_EXPORT void save(const std::string filename) const;
  LIBKRIGING_EXPORT static MLPKriging load(const std::string filename);

 private:
  WarpKriging m_impl;
  std::vector<arma::uword> m_hidden_dims;
  arma::uword m_d_out = 2;
  std::string m_activation = "selu";

  static std::string make_warp_spec(const std::vector<arma::uword>& hidden_dims,
                                    arma::uword d_out,
                                    const std::string& activation);
};

}  // namespace libKriging

#endif  // LIBKRIGING_MLP_KRIGING_HPP
