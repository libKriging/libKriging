#ifndef LIBKRIGING_WARP_KRIGING_HPP
#define LIBKRIGING_WARP_KRIGING_HPP

/**
 * @file WarpKriging.hpp
 * @brief Kriging with per-variable input warping for libKriging.
 *
 * Each input dimension can be independently warped before the GP kernel
 * is evaluated.  This supports:
 *
 *   Continuous variables:
 *     - None       : identity  (no warping)
 *     - Affine     : w(x) = a·x + b                     [2 params]
 *     - BoxCox     : w(x) = (x^λ − 1)/λ   (λ ≠ 0)      [1 param]
 *     - Kumaraswamy: w(x) = 1−(1−x^a)^b   on [0,1]      [2 params]
 *     - NeuralMono : small monotone network               [≥ 6 params]
 *
 *   Discrete / categorical variables:
 *     - Embedding  : each level l → learned e_l ∈ ℝ^q    [L·q params]
 *
 *   Ordinal variables:
 *     - Ordinal    : ordered positions z_1 < z_2 < … < z_L  [L−1 params]
 *
 * The warped representation Φ(x) is the concatenation of all per-variable
 * outputs.  The GP kernel then operates in this warped space:
 *
 *     k(x, x') = σ² · k_base(Φ(x), Φ(x') ; θ)
 *
 * All warping parameters are optimised jointly with the GP
 * hyper-parameters (σ², θ, β) by maximising the marginal log-likelihood.
 *
 * The public API mirrors libKriging::Kriging:
 *     fit(), predict(), simulate(), update(), summary(),
 *     logLikelihood(), logLikelihoodFun()
 *
 * Reference:
 *   Garrido-Merchán & Hernández-Lobato (2020), "Dealing with categorical
 *   and integer-valued variables in Bayesian Optimization with GP".
 *   Saves et al. (2023), "SMT 2.0: surrogate modeling toolbox with mixed
 *   variables support".
 */

#include <armadillo>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

namespace libKriging {

// =========================================================================
//  Per-variable warping specification
// =========================================================================

/// Warping type identifiers
enum class WarpType {
  // --- continuous ---
  None,            ///< identity (no transformation)
  Affine,          ///< w(x) = a·x + b
  BoxCox,          ///< w(x) = (x^λ - 1)/λ
  Kumaraswamy,     ///< w(x) = 1 - (1-x^a)^b   on [0,1]
  NeuralMono,      ///< small monotone neural network
  MLP,             ///< unconstrained multi-layer perceptron (multi-dim output)

  // --- discrete / categorical ---
  Embedding,       ///< learned embedding vector per level

  // --- ordinal ---
  Ordinal          ///< learned ordered positions on ℝ
};

/**
 * @brief Specification for a single input variable's warping.
 *
 * Usage examples:
 *   WarpSpec::continuous(WarpType::Kumaraswamy)        // 2 params, 1D output
 *   WarpSpec::continuous(WarpType::NeuralMono, 8)      // 8 hidden units
 *   WarpSpec::categorical(5, 2)                        // 5 levels → ℝ²
 *   WarpSpec::ordinal(4)                               // 4 ordered levels → ℝ¹
 *   WarpSpec::mlp({16, 8}, 3)                          // MLP: 1→16→8→3
 *   WarpSpec::none()                                   // pass-through
 */
struct WarpSpec {
  WarpType type  = WarpType::None;
  arma::uword n_levels   = 0;   ///< number of levels (categorical/ordinal)
  arma::uword embed_dim  = 1;   ///< embedding dimension (categorical only)
  arma::uword n_hidden   = 8;   ///< hidden units (NeuralMono only)

  // MLP-specific fields
  std::vector<arma::uword> hidden_dims = {};  ///< hidden layer sizes
  arma::uword d_out      = 1;   ///< output dim (MLP only)
  std::string activation = "selu"; ///< activation function (MLP only)

  /// Convenience factories
  static WarpSpec none();
  static WarpSpec affine();
  static WarpSpec boxcox();
  static WarpSpec kumaraswamy();
  static WarpSpec neural_mono(arma::uword n_hidden = 8);
  static WarpSpec categorical(arma::uword n_levels, arma::uword embed_dim = 2);
  static WarpSpec ordinal(arma::uword n_levels);

  /**
   * @brief MLP warping: unconstrained multi-layer perceptron.
   * @param hidden_dims   hidden layer sizes, e.g. {16, 8}
   * @param d_out         output dimensionality in feature space
   * @param activation    "relu", "selu", "tanh", "sigmoid", "elu"
   *
   * Unlike NeuralMono, this warp is NOT monotone and can output
   * a multi-dimensional feature vector.  Most expressive warp,
   * but needs more data to avoid overfitting.
   */
  static WarpSpec mlp(const std::vector<arma::uword>& hidden_dims,
                      arma::uword d_out = 2,
                      const std::string& activation = "selu");
};

// =========================================================================
//  Single-variable warping function (polymorphic via variant)
// =========================================================================

/**
 * @brief Abstract interface for a single-variable warping function.
 *
 * Each warp maps a scalar (or integer level) to a vector in ℝ^{d_out}.
 */
class IWarp {
 public:
  virtual ~IWarp() = default;

  /// Output dimensionality of this warp
  virtual arma::uword output_dim() const = 0;

  /// Number of learnable parameters
  virtual arma::uword n_params() const = 0;

  /// Get parameters as a flat vector
  virtual arma::vec get_params() const = 0;

  /// Set parameters from a flat vector
  virtual void set_params(const arma::vec& p) = 0;

  /// Forward:  warp a column of n scalar values → (n × d_out) matrix
  virtual arma::mat forward(const arma::vec& x) const = 0;

  /// Gradient of the output w.r.t. the parameters (for a batch)
  /// Given dL/dΦ (n × d_out), returns dL/d(params)
  virtual arma::vec backward(const arma::vec& x,
                             const arma::mat& dL_dPhi) const = 0;

  /// Human-readable description
  virtual std::string describe() const = 0;
};

// --- Concrete warp implementations ----------------------------------------

class WarpNone final : public IWarp {
 public:
  arma::uword output_dim() const override { return 1; }
  arma::uword n_params() const override { return 0; }
  arma::vec get_params() const override { return {}; }
  void set_params(const arma::vec&) override {}
  arma::mat forward(const arma::vec& x) const override;
  arma::vec backward(const arma::vec& x,
                     const arma::mat& dL_dPhi) const override;
  std::string describe() const override { return "None (identity)"; }
};

class WarpAffine final : public IWarp {
 public:
  WarpAffine();
  arma::uword output_dim() const override { return 1; }
  arma::uword n_params() const override { return 2; }
  arma::vec get_params() const override;
  void set_params(const arma::vec& p) override;
  arma::mat forward(const arma::vec& x) const override;
  arma::vec backward(const arma::vec& x,
                     const arma::mat& dL_dPhi) const override;
  std::string describe() const override;
 private:
  double m_a = 1.0, m_b = 0.0;
};

class WarpBoxCox final : public IWarp {
 public:
  WarpBoxCox();
  arma::uword output_dim() const override { return 1; }
  arma::uword n_params() const override { return 1; }
  arma::vec get_params() const override;
  void set_params(const arma::vec& p) override;
  arma::mat forward(const arma::vec& x) const override;
  arma::vec backward(const arma::vec& x,
                     const arma::mat& dL_dPhi) const override;
  std::string describe() const override;
 private:
  double m_lambda = 1.0;  ///< stored as unconstrained (real line)
};

class WarpKumaraswamy final : public IWarp {
 public:
  WarpKumaraswamy();
  arma::uword output_dim() const override { return 1; }
  arma::uword n_params() const override { return 2; }
  arma::vec get_params() const override;
  void set_params(const arma::vec& p) override;
  arma::mat forward(const arma::vec& x) const override;
  arma::vec backward(const arma::vec& x,
                     const arma::mat& dL_dPhi) const override;
  std::string describe() const override;
 private:
  double m_log_a = 0.0, m_log_b = 0.0;  ///< a,b > 0 via exp()
};

class WarpNeuralMono final : public IWarp {
 public:
  explicit WarpNeuralMono(arma::uword n_hidden = 8, uint64_t seed = 42);
  arma::uword output_dim() const override { return 1; }
  arma::uword n_params() const override;
  arma::vec get_params() const override;
  void set_params(const arma::vec& p) override;
  arma::mat forward(const arma::vec& x) const override;
  arma::vec backward(const arma::vec& x,
                     const arma::mat& dL_dPhi) const override;
  std::string describe() const override;
 private:
  arma::uword m_H;
  // Architecture:  x → |W1| x + b1 → softplus → |W2| h + b2
  // Positive weights enforced via exp(raw_W)
  arma::vec m_raw_W1;   ///< (H)   weights layer 1 (unconstrained)
  arma::vec m_b1;       ///< (H)   bias layer 1
  arma::vec m_raw_W2;   ///< (H)   weights layer 2 (unconstrained)
  double    m_b2 = 0.0; ///< scalar bias layer 2
};

/**
 * @brief MLP warping: unconstrained multi-layer perceptron.
 *
 * Maps a scalar x ∈ ℝ to a vector φ(x) ∈ ℝ^{d_out} via:
 *
 *     x → [Linear → activation]×L → Linear → φ(x)
 *
 * Unlike NeuralMono:
 *   - NOT constrained to be monotone (free weights)
 *   - Supports multi-dimensional output (d_out > 1)
 *   - Configurable depth and activation
 *
 * This is the most expressive warping and subsumes all continuous
 * warps (Affine, BoxCox, Kumaraswamy, NeuralMono) as special cases.
 * Use when you have enough data to afford the extra parameters.
 */
class WarpMLP final : public IWarp {
 public:
  /// Supported activation functions
  enum class Act { ReLU, SELU, Tanh, Sigmoid, ELU };

  /**
   * @param hidden_dims  sizes of hidden layers, e.g. {16, 8}
   * @param d_out        output dimensionality
   * @param activation   activation function for hidden layers
   * @param seed         random seed for weight initialisation
   */
  WarpMLP(const std::vector<arma::uword>& hidden_dims,
          arma::uword d_out = 2,
          Act activation = Act::SELU,
          uint64_t seed = 42);

  arma::uword output_dim() const override { return m_d_out; }
  arma::uword n_params() const override { return m_n_params; }
  arma::vec get_params() const override;
  void set_params(const arma::vec& p) override;
  arma::mat forward(const arma::vec& x) const override;
  arma::vec backward(const arma::vec& x,
                     const arma::mat& dL_dPhi) const override;
  std::string describe() const override;

  /// Parse activation name from string
  static Act parse_act(const std::string& s);

 private:
  arma::uword m_d_out;
  arma::uword m_n_params = 0;
  Act m_act;

  // Per-layer weights and biases
  //   Layer l:  Z_l = H_{l-1} * W_l + b_l^T
  //   W_l ∈ ℝ^{d_{l-1} × d_l},  b_l ∈ ℝ^{d_l}
  //   Input dim is 1 (scalar), output of last layer = d_out (linear)
  std::vector<arma::mat> m_W;
  std::vector<arma::vec> m_b;

  void count_params();
  static arma::mat apply_act(const arma::mat& Z, Act act);
  static arma::mat act_deriv(const arma::mat& Z, Act act);
};

class WarpEmbedding final : public IWarp {
 public:
  WarpEmbedding(arma::uword n_levels, arma::uword embed_dim,
                uint64_t seed = 42);
  arma::uword output_dim() const override { return m_embed_dim; }
  arma::uword n_params() const override;
  arma::vec get_params() const override;
  void set_params(const arma::vec& p) override;
  arma::mat forward(const arma::vec& x) const override;
  arma::vec backward(const arma::vec& x,
                     const arma::mat& dL_dPhi) const override;
  std::string describe() const override;
 private:
  arma::uword m_n_levels;
  arma::uword m_embed_dim;
  arma::mat   m_E;         ///< embedding matrix (n_levels × embed_dim)
};

class WarpOrdinal final : public IWarp {
 public:
  explicit WarpOrdinal(arma::uword n_levels, uint64_t seed = 42);
  arma::uword output_dim() const override { return 1; }
  arma::uword n_params() const override;
  arma::vec get_params() const override;
  void set_params(const arma::vec& p) override;
  arma::mat forward(const arma::vec& x) const override;
  arma::vec backward(const arma::vec& x,
                     const arma::mat& dL_dPhi) const override;
  std::string describe() const override;
 private:
  arma::uword m_n_levels;
  arma::vec   m_raw_gaps;  ///< (L-1) unconstrained; actual gaps = exp(raw)
};

// =========================================================================
//  Supported base kernels
// =========================================================================

enum class WarpBaseKernel { Gauss, Matern32, Matern52, Exp };

// =========================================================================
//  WarpKriging  —  main class
// =========================================================================

/**
 * @brief Kriging model with per-variable input warping.
 *
 * Model:
 *    y(x) = f(x)^T β  +  ζ(x)
 *
 * where ζ is a zero-mean GP with covariance
 *    Cov[ζ(x), ζ(x')] = σ² · k_base(Φ(x), Φ(x') ; θ)
 *
 * Φ(x) = [ w_1(x_1) , w_2(x_2) , … , w_d(x_d) ] is the concatenation
 * of per-variable warpings.
 *
 * Public interface mirrors libKriging::Kriging.
 */
class WarpKriging {
 public:
  // -----------------------------------------------------------------------
  //  Construction
  // -----------------------------------------------------------------------

  /**
   * @brief Construct with warping specification per variable.
   * @param warping   vector of WarpSpec, one per input dimension
   * @param kernel    base kernel: "gauss", "matern3_2", "matern5_2", "exp"
   */
  WarpKriging(const std::vector<WarpSpec>& warping,
              const std::string& kernel = "gauss");

  /**
   * @brief Full constructor with immediate fitting.
   */
  WarpKriging(const arma::vec& y,
              const arma::mat& X,
              const std::vector<WarpSpec>& warping,
              const std::string& kernel,
              const std::string& regmodel   = "constant",
              bool normalize                = false,
              const std::string& optim      = "BFGS+Adam",
              const std::string& objective  = "LL",
              const std::map<std::string, std::string>& parameters = {});

  // -----------------------------------------------------------------------
  //  Fitting
  // -----------------------------------------------------------------------

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

  std::tuple<arma::vec, arma::vec, arma::mat>
  predict(const arma::mat& x_new,
          bool withStd = true,
          bool withCov = false) const;

  // -----------------------------------------------------------------------
  //  Simulation
  // -----------------------------------------------------------------------

  arma::mat simulate(int nsim, uint64_t seed, const arma::mat& x_new) const;

  // -----------------------------------------------------------------------
  //  Update
  // -----------------------------------------------------------------------

  void update(const arma::vec& y_new, const arma::mat& X_new);

  // -----------------------------------------------------------------------
  //  Log-likelihood
  // -----------------------------------------------------------------------

  double logLikelihood() const;

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
  bool is_fitted() const { return m_fitted; }
  arma::uword feature_dim() const { return m_feature_dim; }
  const std::vector<WarpSpec>& warping() const { return m_warp_specs; }

  /// Access a specific warp function (e.g. to inspect embeddings)
  const IWarp& warp(arma::uword dim) const { return *m_warps[dim]; }

 private:
  // ---- data ---------------------------------------------------------------
  arma::vec m_y;
  arma::mat m_X;
  arma::mat m_Phi;       ///< warped design (n × feature_dim)

  // ---- warping ------------------------------------------------------------
  std::vector<WarpSpec>                  m_warp_specs;
  std::vector<std::unique_ptr<IWarp>>    m_warps;
  arma::uword                            m_feature_dim = 0;

  // ---- normalisation ------------------------------------------------------
  bool m_normalize = false;
  arma::rowvec m_X_mean, m_X_std;
  double m_y_mean = 0.0, m_y_std = 1.0;
  // Per-variable normalisation (only continuous variables)
  std::vector<bool> m_is_continuous;

  // ---- trend --------------------------------------------------------------
  std::string m_regmodel = "constant";
  arma::mat m_F;
  arma::vec m_beta;

  // ---- kernel + hyper-params -----------------------------------------------
  std::string    m_kernel_name;
  WarpBaseKernel m_base_kernel = WarpBaseKernel::Gauss;
  arma::vec      m_theta;
  double         m_sigma2 = 1.0;

  // ---- GP cache -----------------------------------------------------------
  arma::mat m_C;         ///< Cholesky(K + nugget), lower
  arma::vec m_alpha;     ///< K^{-1}(y - Fβ)
  double    m_logdet = 0.0;

  bool m_fitted = false;

  // ---- optimiser ----------------------------------------------------------
  arma::uword m_max_iter_bfgs = 100;
  arma::uword m_max_iter_adam = 500;
  double      m_adam_lr = 1e-3;

  // ---- private helpers ----------------------------------------------------
  static WarpBaseKernel parse_kernel(const std::string& name);
  std::unique_ptr<IWarp> make_warp(const WarpSpec& spec) const;
  void build_warps();

  arma::mat build_trend_matrix(const arma::mat& X) const;
  arma::mat apply_warping(const arma::mat& X) const;

  double kernel_scalar(const arma::rowvec& phi_i,
                       const arma::rowvec& phi_j) const;
  arma::mat build_K(const arma::mat& Phi) const;
  arma::mat build_Kcross(const arma::mat& Phi_new,
                         const arma::mat& Phi_train) const;

  static double compute_ll_internal(const arma::vec& y,
                                    const arma::mat& F,
                                    const arma::vec& beta,
                                    const arma::vec& alpha,
                                    double logdet);

  void refresh_cache();
  void normalise_data();

  arma::vec pack_params() const;
  void      unpack_params(const arma::vec& all);
  arma::uword total_warp_params() const;

  std::pair<double, arma::vec>
  compute_loglik_and_grad(const arma::vec& all_params, bool need_grad) const;

  arma::mat dK_dPhi(const arma::mat& Phi, const arma::mat& dL_dK) const;

  void optimise_joint(const std::string& method);
  void adam_step(arma::vec& params, const arma::vec& grad,
                 arma::vec& mm, arma::vec& vm,
                 arma::uword t, double lr,
                 double beta1, double beta2, double eps) const;
};

}  // namespace libKriging

#endif  // LIBKRIGING_WARP_KRIGING_HPP
