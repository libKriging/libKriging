#ifndef LIBKRIGING_MLP_KRIGING_HPP
#define LIBKRIGING_MLP_KRIGING_HPP

/**
 * @file MLPKriging.hpp
 * @brief Kriging with a joint MLP feature map (≡ Deep Kernel Learning / NeuralKernelKriging).
 *
 * A single multi-layer perceptron Φ : ℝ^d → ℝ^{d_out} warps the FULL input
 * vector jointly (cross-variable interactions), and a Gaussian process is
 * fitted on the warped features Φ(x).
 *
 *     k(x, x') = σ² · k_base(Φ(x), Φ(x') ; θ)
 *
 * Unlike WarpKriging (which concatenates per-variable warps), MLPKriging
 * uses one joint network — this is the \"mlp_joint\" variant extracted into
 * its own class for clarity.
 *
 * The public API mirrors libKriging::Kriging:
 *     fit(), predict(), simulate(), update(), summary(),
 *     logLikelihood(), logLikelihoodFun()
 */

#include "libKriging/Trend.hpp"
#include "libKriging/WarpKriging.hpp"  // for WarpMLPJoint, WarpBaseKernel
#include "libKriging/libKriging_exports.h"
#include "libKriging/utils/lk_armadillo.hpp"

#include <functional>
#include <map>
#include <memory>
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

  /**
   * @brief Light constructor (architecture only — no data).
   * @param hidden_dims  sizes of hidden layers, e.g. {32, 16}
   * @param d_out        output dimensionality of Φ (and size of θ)
   * @param activation   hidden activation ("selu", "relu", "tanh", "sigmoid", "elu")
   * @param kernel       base kernel: "gauss", "matern3_2", "matern5_2", "exp"
   */
  LIBKRIGING_EXPORT MLPKriging(const std::vector<arma::uword>& hidden_dims,
                               arma::uword d_out = 2,
                               const std::string& activation = "selu",
                               const std::string& kernel = "gauss");

  /**
   * @brief Full constructor with immediate fitting.
   */
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

  /// Typed-parameters overload: lets callers seed θ / warp params.
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

  const arma::mat& X() const { return m_X; }
  const arma::rowvec& centerX() const { return m_centerX; }
  const arma::rowvec& scaleX() const { return m_scaleX; }
  const arma::vec& y() const { return m_y; }
  const double& centerY() const { return m_centerY; }
  const double& scaleY() const { return m_scaleY; }
  const bool& normalize() const { return m_normalize; }
  const Trend::RegressionModel& regmodel() const { return m_regmodel; }
  const arma::mat& F() const { return m_F; }
  const arma::mat& T() const { return m_T; }
  const arma::mat& M() const { return m_M; }
  const arma::vec& z() const { return m_z; }
  const arma::vec& beta() const { return m_beta; }
  std::string kernel() const { return m_kernel_name; }
  arma::vec theta() const { return m_theta; }
  double sigma2() const { return m_sigma2; }
  bool is_fitted() const { return m_fitted; }
  arma::uword feature_dim() const { return m_d_out; }

  /// Access to the underlying joint-MLP warping (for inspection).
  const WarpMLPJoint& warp() const { return *m_joint_warp; }

  /// Architecture accessors
  const std::vector<arma::uword>& hidden_dims() const { return m_hidden_dims; }
  arma::uword d_out() const { return m_d_out; }
  const std::string& activation() const { return m_activation; }

  /** Dump current MLPKriging object into a file */
  LIBKRIGING_EXPORT void save(const std::string filename) const;

  /** Load a new MLPKriging object from a file */
  LIBKRIGING_EXPORT static MLPKriging load(const std::string filename);

 private:
  // ---- data ---------------------------------------------------------------
  arma::vec m_y;
  arma::mat m_X;
  arma::mat m_Phi;   ///< warped design (n × d_out)
  arma::mat m_dPhi;  ///< precomputed pairwise diffs (d_out × n*n)

  // ---- joint warp (architecture + learned params) -------------------------
  std::vector<arma::uword> m_hidden_dims;
  arma::uword m_d_out = 2;
  std::string m_activation = "selu";
  std::unique_ptr<WarpMLPJoint> m_joint_warp;  ///< instantiated at fit() time (needs d_in)

  // ---- normalisation ------------------------------------------------------
  bool m_normalize = false;
  arma::rowvec m_centerX, m_scaleX;
  double m_centerY = 0.0, m_scaleY = 1.0;

  // ---- trend --------------------------------------------------------------
  Trend::RegressionModel m_regmodel = Trend::RegressionModel::Constant;
  arma::mat m_F;
  arma::vec m_beta;

  // ---- kernel + hyper-params ----------------------------------------------
  std::string m_kernel_name;
  WarpBaseKernel m_base_kernel = WarpBaseKernel::Gauss;
  std::function<double(const arma::vec&, const arma::vec&)> _Cov;
  std::function<arma::vec(const arma::vec&, const arma::vec&)> _DlnCovDtheta;
  std::function<arma::vec(const arma::vec&, const arma::vec&)> _DlnCovDx;
  void make_Cov(const std::string& kernel);
  arma::vec m_theta;
  double m_sigma2 = 1.0;

  // ---- MLPKModel (mirrors Kriging::KModel) --------------------------------
 public:
  struct MLPKModel {
    arma::mat R;      ///< correlation matrix
    arma::mat L;      ///< Cholesky lower
    arma::mat Rinv;   ///< R⁻¹
    arma::mat Fstar;  ///< L \ F  (whitened trend, ≡ m_M)
    arma::vec ystar;  ///< L \ y
    arma::mat Rstar;  ///< chol_upper(F'R⁻¹F)  (≡ m_circ)
    arma::vec Estar;  ///< L \ (y - Fβ̂)  (whitened residual, ≡ m_z)
    double SSEstar;   ///< Estar'Estar
    arma::vec betahat;
  };
  MLPKModel make_Model(const arma::vec& theta) const;
  void populate_Model(MLPKModel& m, const arma::vec& theta) const;

 private:
  // ---- GP cache -----------------------------------------------------------
  arma::mat m_R;     ///< correlation matrix (n×n)
  arma::mat m_T;     ///< Cholesky(R + nugget), lower
  arma::mat m_M;     ///< C⁻¹ F  (whitened trend basis)
  arma::mat m_circ;  ///< chol_upper(F'R⁻¹F)
  arma::vec m_z;     ///< C⁻¹(y - Fβ)  (whitened residuals)
  arma::mat m_Rinv;  ///< R⁻¹ = C⁻ᵀ C⁻¹, cached
  double m_logdet = 0.0;

  bool m_fitted = false;

  // ---- Simulation cached data (FOXY algorithm) ----------------------------
  arma::mat lastsim_Xn_n;
  arma::mat lastsim_Phi_n;
  arma::mat lastsim_y_n;
  int lastsim_nsim{};
  uint64_t lastsim_seed{};
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

  // Updated simulation cached data
  arma::mat lastsimup_Xn_u;
  arma::mat lastsimup_y_u;
  arma::mat lastsimup_Wtild_nKu;
  arma::mat lastsimup_R_uo;
  arma::mat lastsimup_R_un;
  arma::mat lastsimup_R_uu;

  // ---- optimiser knobs ----------------------------------------------------
  arma::uword m_max_iter_bfgs = 100;
  arma::uword m_max_iter_adam = 10;
  double m_adam_lr = 1e-3;

  // ---- private helpers ----------------------------------------------------
  static WarpBaseKernel parse_kernel(const std::string& name);

  arma::mat build_trend_matrix(const arma::mat& X) const;
  arma::mat apply_warping(const arma::mat& X) const;

  void compute_dPhi();
  arma::mat build_Rcross(const arma::mat& Phi_new, const arma::mat& Phi_train) const;

  void refresh_cache();
  void refresh_cache_theta_only();
  void normalise_data();

  double concentrated_ll() const;

  arma::mat build_dR_dtheta_k(arma::uword k) const;
  std::pair<double, arma::vec> concentrated_ll_and_grad_theta() const;

  arma::mat dK_dPhi(const arma::mat& dL_dK) const;
  arma::vec warp_gradient() const;

  arma::uword total_warp_params() const;
  arma::vec pack_warp_params() const;
  void unpack_warp_params(const arma::vec& w);

  void optimise_joint(const std::string& method);

  MLPKriging clone_for_thread() const;

  void ensure_joint_warp(arma::uword d_in);
};

}  // namespace libKriging

#endif  // LIBKRIGING_MLP_KRIGING_HPP
