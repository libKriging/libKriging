#ifndef LIBKRIGING_NEURALKERNEL_KRIGING_HPP
#define LIBKRIGING_NEURALKERNEL_KRIGING_HPP

/**
 * @file NeuralKernelKriging.hpp
 * @brief Deep Kernel Learning for libKriging
 *
 * Implements the Deep Kernel Learning approach (Wilson et al., AISTATS 2016)
 * as an extension to libKriging.  A neural network φ(x) transforms the input
 * space before a standard GP kernel is applied:
 *
 *     k_DKL(x, x') = σ² · k_base(φ(x), φ(x') | θ)
 *
 * The full model (NN weights + GP hyper-parameters) is trained end-to-end by
 * maximising the marginal log-likelihood.
 *
 * The public API mirrors the existing Kriging class:
 *     fit(), predict(), simulate(), update(), summary(),
 *     logLikelihood(), logLikelihoodFun()
 *
 * Dependencies: Armadillo (already in libKriging), no external ML framework.
 *
 * Reference:
 *   A. G. Wilson, Z. Hu, R. Salakhutdinov, E. P. Xing.
 *   "Deep Kernel Learning", AISTATS 2016.
 */

#include <armadillo>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace libKriging {

// =========================================================================
//  Small MLP implemented with Armadillo (no external ML dependency)
// =========================================================================

/// Supported activation functions
enum class Activation { ReLU, SELU, Tanh, Sigmoid, ELU };

/// Description of a single dense layer
struct DenseLayerSpec {
  arma::uword n_out;           ///< number of output neurons
  Activation activation;       ///< activation after this layer
  bool batch_norm = false;     ///< apply batch-normalisation
};

/**
 * @brief Minimal fully-connected neural network (MLP).
 *
 * Forward pass:  x ∈ ℝ^{d_in}  →  φ(x) ∈ ℝ^{d_out}
 * All matrices are stored row-major: one sample per row.
 *
 * Supports forward and backward (gradient) propagation with respect to all
 * weights, entirely implemented on top of Armadillo.
 */
class MLP {
 public:
  /**
   * @brief Construct an MLP.
   * @param d_in        input dimensionality
   * @param layers      hidden + output layer specifications
   * @param seed        random seed for weight initialisation
   */
  MLP(arma::uword d_in,
      const std::vector<DenseLayerSpec>& layers,
      uint64_t seed = 42);

  /// Default-constructible (empty network)
  MLP() = default;

  // -- Dimensions -----------------------------------------------------------
  arma::uword input_dim() const { return m_d_in; }
  arma::uword output_dim() const { return m_d_out; }
  arma::uword n_params() const { return m_n_params; }
  arma::uword n_layers() const { return m_W.size(); }

  // -- Parameter access (flat vector) ---------------------------------------
  arma::vec   get_params() const;
  void        set_params(const arma::vec& theta);

  // -- Forward pass ---------------------------------------------------------
  /** @brief Evaluate φ(X) for a batch X (n × d_in). */
  arma::mat forward(const arma::mat& X) const;

  // -- Backward pass --------------------------------------------------------
  /**
   * @brief Back-propagate through the network.
   * @param X        input  (n × d_in)
   * @param dL_dPhi  gradient of loss w.r.t. output φ  (n × d_out)
   * @return         gradient of loss w.r.t. all weights (flat vector)
   *
   * This re-runs the forward pass internally to cache activations.
   */
  arma::vec backward(const arma::mat& X, const arma::mat& dL_dPhi) const;

 private:
  arma::uword m_d_in  = 0;
  arma::uword m_d_out = 0;
  arma::uword m_n_params = 0;

  // Per-layer storage
  std::vector<arma::mat> m_W;   ///< weight matrices W_l  (d_prev × d_l)
  std::vector<arma::vec> m_b;   ///< bias vectors    b_l  (d_l)
  std::vector<Activation> m_act;

  // Batch-norm parameters (γ, β per layer, optional)
  std::vector<bool>      m_use_bn;
  std::vector<arma::vec> m_bn_gamma;
  std::vector<arma::vec> m_bn_beta;

  // helpers
  static arma::mat apply_activation(const arma::mat& Z, Activation act);
  static arma::mat activation_derivative(const arma::mat& Z, Activation act);
  void count_params();
};

// =========================================================================
//  Kernel functions operating in the feature space
// =========================================================================

/// Supported base kernel families (applied after the NN transform)
enum class BaseKernel { Gauss, Matern32, Matern52, Exp };

// =========================================================================
//  NeuralKernelKriging  —  main class
// =========================================================================

/**
 * @brief Deep Kernel Learning Kriging model.
 *
 * Model:
 *    y(x) = f(x)^T β  +  ζ(x)
 *
 * where ζ is a zero-mean GP with covariance
 *    Cov[ζ(x), ζ(x')] = σ² · k_base(φ(x), φ(x') ; θ)
 *
 * φ(·) is a learned MLP feature extractor.
 *
 * The public interface mirrors libKriging::Kriging as closely as possible
 * so that it can serve as a drop-in replacement.
 */
class NeuralKernelKriging {
 public:
  // -----------------------------------------------------------------------
  //  Construction
  // -----------------------------------------------------------------------

  /**
   * @brief Construct a NeuralKernelKriging model.
   * @param kernel     base kernel name: "gauss", "matern3_2", "matern5_2", "exp"
   *
   * The MLP architecture is specified later via setNNArchitecture() or
   * through default heuristics at fit() time.
   */
  explicit NeuralKernelKriging(const std::string& kernel = "gauss");

  /**
   * @brief Full constructor with immediate fitting.
   * @param y          response vector  (n)
   * @param X          design matrix    (n × d)
   * @param kernel     base kernel name
   * @param regmodel   regression model: "constant", "linear", "quadratic"
   * @param normalize  normalise inputs to [0,1]
   * @param optim      optimiser: "BFGS", "Adam", "BFGS+Adam"
   * @param objective  "LL" (log-likelihood) or "LOO" (leave-one-out)
   * @param parameters optional map of pre-set parameters
   */
  NeuralKernelKriging(const arma::vec& y,
                      const arma::mat& X,
                      const std::string& kernel,
                      const std::string& regmodel   = "constant",
                      bool normalize                 = false,
                      const std::string& optim       = "BFGS+Adam",
                      const std::string& objective   = "LL",
                      const std::map<std::string, std::string>& parameters = {});

  // -----------------------------------------------------------------------
  //  Neural-network configuration (before fit)
  // -----------------------------------------------------------------------

  /**
   * @brief Set the MLP architecture for the feature extractor.
   * @param hidden_dims   sizes of hidden layers, e.g. {64, 32}
   * @param feature_dim   output dimensionality of φ(x)  (0 = auto)
   * @param activation    "relu", "selu", "tanh", "sigmoid", "elu"
   * @param batch_norm    use batch-normalisation
   * @param seed          random seed for weight init
   */
  void setNNArchitecture(const std::vector<arma::uword>& hidden_dims,
                         arma::uword feature_dim = 0,
                         const std::string& activation = "selu",
                         bool batch_norm = true,
                         uint64_t seed = 42);

  // -----------------------------------------------------------------------
  //  Fitting
  // -----------------------------------------------------------------------

  /**
   * @brief Fit the model to observations.
   *
   * Jointly optimises NN weights and GP hyper-parameters (σ², θ, β)
   * by maximising the marginal log-likelihood (or LOO).
   */
  void fit(const arma::vec& y,
           const arma::mat& X,
           const std::string& regmodel   = "constant",
           bool normalize                = false,
           const std::string& optim      = "BFGS+Adam",
           const std::string& objective  = "LL",
           const std::map<std::string, std::string>& parameters = {});

  // -----------------------------------------------------------------------
  //  Prediction
  // -----------------------------------------------------------------------

  /**
   * @brief Predict at new locations.
   * @param x_new      prediction points  (m × d)
   * @param withStd    return standard deviations?
   * @param withCov    return full covariance matrix?
   * @return tuple (mean, stdev, cov)
   *         - mean:  (m)
   *         - stdev: (m)  or empty
   *         - cov:   (m×m) or empty
   */
  std::tuple<arma::vec, arma::vec, arma::mat>
  predict(const arma::mat& x_new,
          bool withStd = true,
          bool withCov = false) const;

  // -----------------------------------------------------------------------
  //  Simulation
  // -----------------------------------------------------------------------

  /**
   * @brief Draw conditional simulations at new locations.
   * @param nsim   number of simulations
   * @param seed   random seed
   * @param x_new  simulation points (m × d)
   * @return matrix (m × nsim) of simulated values
   */
  arma::mat simulate(int nsim, uint64_t seed, const arma::mat& x_new) const;

  // -----------------------------------------------------------------------
  //  Update (add new observations without full refit)
  // -----------------------------------------------------------------------

  /**
   * @brief Incrementally add observations.
   *
   * Performs a warm-start re-optimisation with few iterations.
   */
  void update(const arma::vec& y_new, const arma::mat& X_new);

  // -----------------------------------------------------------------------
  //  Log-likelihood
  // -----------------------------------------------------------------------

  /**
   * @brief Evaluate the marginal log-likelihood at current parameters.
   */
  double logLikelihood() const;

  /**
   * @brief Evaluate the marginal log-likelihood at given hyper-parameters.
   * @param theta_gp    GP range parameters (d_feature)
   * @param withGrad    also return the gradient?
   * @param withHess    also return the Hessian?  (not yet implemented)
   * @return tuple (ll, grad, hess)
   */
  std::tuple<double, arma::vec, arma::mat>
  logLikelihoodFun(const arma::vec& theta_gp,
                   bool withGrad = true,
                   bool withHess = false) const;

  // -----------------------------------------------------------------------
  //  Accessors
  // -----------------------------------------------------------------------

  std::string summary() const;

  const arma::mat& X() const { return m_X; }
  const arma::vec& y() const { return m_y; }
  std::string kernel() const { return m_kernel_name; }
  arma::vec theta() const { return m_theta; }
  double sigma2() const { return m_sigma2; }
  const MLP& featureExtractor() const { return m_nn; }
  bool is_fitted() const { return m_fitted; }

 private:
  // ---- data ---------------------------------------------------------------
  arma::vec m_y;         ///< observations (n)
  arma::mat m_X;         ///< design matrix (n × d)
  arma::mat m_Phi;       ///< NN-transformed design (n × d_feature) — cached

  // ---- normalisation ------------------------------------------------------
  bool      m_normalize = false;
  arma::rowvec m_X_mean, m_X_std;
  double    m_y_mean = 0.0, m_y_std = 1.0;

  // ---- trend (regression model) -------------------------------------------
  std::string m_regmodel = "constant";
  arma::mat m_F;         ///< trend matrix (n × p)
  arma::vec m_beta;      ///< trend coefficients (p)

  // ---- kernel + hyper-parameters ------------------------------------------
  std::string m_kernel_name;
  BaseKernel  m_base_kernel = BaseKernel::Gauss;
  arma::vec   m_theta;       ///< range parameters in feature space (d_feature)
  double      m_sigma2 = 1.0;///< process variance

  // ---- neural network -----------------------------------------------------
  MLP         m_nn;
  bool        m_nn_configured = false;
  // NN architecture hints (used at fit time if not pre-configured)
  std::vector<arma::uword> m_hidden_dims;
  arma::uword m_feature_dim = 0;

  // ---- cached GP quantities -----------------------------------------------
  arma::mat m_C;         ///< Cholesky factor of K + nugget (lower)
  arma::vec m_alpha;     ///< C^{-T} C^{-1} (y - Fβ)
  double    m_logdet = 0.0;

  bool      m_fitted = false;

  // ---- optimiser settings -------------------------------------------------
  arma::uword m_max_iter_bfgs = 100;
  arma::uword m_max_iter_adam = 500;
  double      m_adam_lr = 1e-3;

  // ---- private helpers ----------------------------------------------------

  /// Parse kernel name string → enum
  static BaseKernel parse_kernel(const std::string& name);

  /// Build trend matrix F from X for the chosen regmodel
  arma::mat build_trend_matrix(const arma::mat& X) const;

  /// Evaluate the base kernel between two feature vectors
  double kernel_scalar(const arma::rowvec& phi_i,
                       const arma::rowvec& phi_j) const;

  /// Build full covariance matrix K from features Φ
  arma::mat build_K(const arma::mat& Phi) const;

  /// Build cross-covariance k(Φ_new, Φ_train)  →  (m × n)
  arma::mat build_Kcross(const arma::mat& Phi_new,
                         const arma::mat& Phi_train) const;

  /// Compute LL and optionally gradient w.r.t. all params (NN + GP)
  std::pair<double, arma::vec>
  compute_loglik_and_grad(const arma::vec& all_params, bool need_grad) const;

  /// Joint optimisation: BFGS on GP params, Adam on NN weights
  void optimise_joint(const std::string& method);

  /// Adam optimiser step for NN weights
  void adam_step(arma::vec& params,
                 const arma::vec& grad,
                 arma::vec& m_adam_m,
                 arma::vec& m_adam_v,
                 arma::uword t,
                 double lr, double beta1, double beta2, double eps) const;

  /// Refresh cached quantities (C, alpha, logdet) after param changes
  void refresh_cache();

  /// Normalise X and y if needed
  void normalise_data();

  /// Auto-configure NN architecture from data dimensions
  void auto_configure_nn(arma::uword d_in);

  /// Pack / unpack all parameters into a single vector
  arma::vec pack_params() const;
  void      unpack_params(const arma::vec& all);

  /// Gradient of K w.r.t. Φ entries  (for backprop through the kernel)
  arma::mat dK_dPhi(const arma::mat& Phi, const arma::mat& dL_dK) const;
};

}  // namespace libKriging

#endif  // LIBKRIGING_NEURALKERNEL_KRIGING_HPP
