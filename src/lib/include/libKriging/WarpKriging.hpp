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

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/libKriging_exports.h"

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
  None,         ///< identity (no transformation)
  Affine,       ///< w(x) = a·x + b
  BoxCox,       ///< w(x) = (x^λ - 1)/λ
  Kumaraswamy,  ///< w(x) = 1 - (1-x^a)^b   on [0,1]
  NeuralMono,   ///< small monotone neural network
  MLP,          ///< unconstrained multi-layer perceptron (multi-dim output)

  // --- discrete / categorical ---
  Embedding,  ///< learned embedding vector per level

  // --- ordinal ---
  Ordinal,  ///< learned ordered positions on ℝ

  // --- joint (multi-input) ---
  MLPJoint  ///< MLP taking ALL inputs jointly (≡ NeuralKernelKriging)
};

/**
 * @brief Specification for a single input variable's warping.
 *
 * Can be constructed programmatically or parsed from a string via
 * \c from_string().  The string format mirrors the trend/kernel style
 * of libKriging:
 *
 *   "none"                   — identity (no warping)
 *   "affine"                 — w(x) = a·x + b               [2 params]
 *   "boxcox"                 — Box-Cox transform              [1 param]
 *   "kumaraswamy"            — Kumaraswamy CDF on [0,1]       [2 params]
 *   "neural_mono(H)"         — monotone network, H hidden     [3H+1 params]
 *   "mlp(h1:h2,q,act)"       — MLP with layers h1→h2→q, activation act
 *   "categorical(L,q)"       — L levels embedded in ℝ^q
 *   "ordinal(L)"             — L ordered levels               [L−1 params]
 *
 * Defaults when arguments are omitted:
 *   "neural_mono"   ⟹  "neural_mono(8)"
 *   "mlp"           ⟹  "mlp(16:8,2,selu)"
 *   "categorical(5)"⟹  "categorical(5,2)"
 *
 * Usage in C++:
 *   WarpKriging model(y, X, {"kumaraswamy", "categorical(5,2)"}, "gauss");
 */
struct WarpSpec {
  WarpType type = WarpType::None;
  arma::uword n_levels = 0;   ///< number of levels (categorical/ordinal)
  arma::uword embed_dim = 1;  ///< embedding dimension (categorical only)
  arma::uword n_hidden = 8;   ///< hidden units (NeuralMono only)

  // MLP-specific fields
  std::vector<arma::uword> hidden_dims = {};  ///< hidden layer sizes
  arma::uword d_out = 1;                      ///< output dim (MLP only)
  std::string activation = "selu";            ///< activation function (MLP only)

  // ---- String parsing / serialisation ------------------------------------

  /**
   * @brief Parse a warp specification from a string.
   *
   * Format:  "type"  or  "type(arg1,arg2,...)"
   * For MLP hidden layers use ':' separator: "mlp(16:8,3,selu)"
   *
   * @throws std::invalid_argument if the string cannot be parsed
   */
  LIBKRIGING_EXPORT static WarpSpec from_string(const std::string& str);

  /**
   * @brief Convert back to the canonical string form.
   */
  LIBKRIGING_EXPORT std::string to_string() const;

  // ---- Convenience factories (kept for programmatic use) -----------------
  static WarpSpec none();
  static WarpSpec affine();
  static WarpSpec boxcox();
  static WarpSpec kumaraswamy();
  static WarpSpec neural_mono(arma::uword n_hidden = 8);
  static WarpSpec categorical(arma::uword n_levels, arma::uword embed_dim = 2);
  static WarpSpec ordinal(arma::uword n_levels);
  static WarpSpec mlp(const std::vector<arma::uword>& hidden_dims,
                      arma::uword d_out = 2,
                      const std::string& activation = "selu");

  /**
   * @brief Joint MLP warping: takes ALL input dimensions together.
   *
   * String format:  "mlp_joint(h1:h2,d_out,act)"
   *   or            "mlp_joint(h1:h2)"  (d_out=2, act=selu by default)
   *
   * This subsumes NeuralKernelKriging: a single MLP φ(x) ∈ ℝ^{d_out}
   * maps the entire input vector x ∈ ℝ^d jointly (cross-variable interactions).
   *
   * When used, warping must be exactly {"mlp_joint(…)"} — one entry.
   */
  static WarpSpec mlp_joint(const std::vector<arma::uword>& hidden_dims,
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
class LIBKRIGING_EXPORT IWarp {
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
  virtual arma::vec backward(const arma::vec& x, const arma::mat& dL_dPhi) const = 0;

  /// Human-readable description
  virtual std::string describe() const = 0;
};

// --- Concrete warp implementations ----------------------------------------

class LIBKRIGING_EXPORT WarpNone final : public IWarp {
 public:
  arma::uword output_dim() const override { return 1; }
  arma::uword n_params() const override { return 0; }
  arma::vec get_params() const override { return {}; }
  void set_params(const arma::vec&) override {}
  arma::mat forward(const arma::vec& x) const override;
  arma::vec backward(const arma::vec& x, const arma::mat& dL_dPhi) const override;
  std::string describe() const override { return "None (identity)"; }
};

class LIBKRIGING_EXPORT WarpAffine final : public IWarp {
 public:
  WarpAffine();
  arma::uword output_dim() const override { return 1; }
  arma::uword n_params() const override { return 2; }
  arma::vec get_params() const override;
  void set_params(const arma::vec& p) override;
  arma::mat forward(const arma::vec& x) const override;
  arma::vec backward(const arma::vec& x, const arma::mat& dL_dPhi) const override;
  std::string describe() const override;

 private:
  double m_a = 1.0, m_b = 0.0;
};

class LIBKRIGING_EXPORT WarpBoxCox final : public IWarp {
 public:
  WarpBoxCox();
  arma::uword output_dim() const override { return 1; }
  arma::uword n_params() const override { return 1; }
  arma::vec get_params() const override;
  void set_params(const arma::vec& p) override;
  arma::mat forward(const arma::vec& x) const override;
  arma::vec backward(const arma::vec& x, const arma::mat& dL_dPhi) const override;
  std::string describe() const override;

 private:
  double m_lambda = 1.0;  ///< stored as unconstrained (real line)
};

class LIBKRIGING_EXPORT WarpKumaraswamy final : public IWarp {
 public:
  WarpKumaraswamy();
  arma::uword output_dim() const override { return 1; }
  arma::uword n_params() const override { return 2; }
  arma::vec get_params() const override;
  void set_params(const arma::vec& p) override;
  arma::mat forward(const arma::vec& x) const override;
  arma::vec backward(const arma::vec& x, const arma::mat& dL_dPhi) const override;
  std::string describe() const override;

 private:
  double m_log_a = 0.0, m_log_b = 0.0;  ///< a,b > 0 via exp()
};

class LIBKRIGING_EXPORT WarpNeuralMono final : public IWarp {
 public:
  explicit WarpNeuralMono(arma::uword n_hidden = 8, uint64_t seed = 42);
  arma::uword output_dim() const override { return 1; }
  arma::uword n_params() const override;
  arma::vec get_params() const override;
  void set_params(const arma::vec& p) override;
  arma::mat forward(const arma::vec& x) const override;
  arma::vec backward(const arma::vec& x, const arma::mat& dL_dPhi) const override;
  std::string describe() const override;

 private:
  arma::uword m_H;
  // Architecture:  x → |W1| x + b1 → softplus → |W2| h + b2
  // Positive weights enforced via exp(raw_W)
  arma::vec m_raw_W1;  ///< (H)   weights layer 1 (unconstrained)
  arma::vec m_b1;      ///< (H)   bias layer 1
  arma::vec m_raw_W2;  ///< (H)   weights layer 2 (unconstrained)
  double m_b2 = 0.0;   ///< scalar bias layer 2
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
class LIBKRIGING_EXPORT WarpMLP final : public IWarp {
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
  arma::vec backward(const arma::vec& x, const arma::mat& dL_dPhi) const override;
  std::string describe() const override;

  /// Parse activation name from string
  static Act parse_act(const std::string& s);

  /// Activation functions (public for use by WarpMLPJoint)
  static arma::mat apply_act(const arma::mat& Z, Act act);
  static arma::mat act_deriv(const arma::mat& Z, Act act);

 private:
  arma::uword m_d_out;
  arma::uword m_n_params = 0;
  Act m_act;

  std::vector<arma::mat> m_W;
  std::vector<arma::vec> m_b;

  void count_params();
};

/**
 * @brief Joint MLP: takes the FULL input vector x ∈ ℝ^d and maps to ℝ^{d_out}.
 *
 * This is NOT an IWarp (which is per-variable). It is used when the
 * warping spec is "mlp_joint(…)", replacing all per-variable warps.
 *
 * Subsumes NeuralKernelKriging: the entire input space is transformed
 * jointly, allowing the network to learn cross-variable interactions.
 *
 *     Φ(x) = MLP(x₁, x₂, …, xₐ) ∈ ℝ^{d_out}
 */
class LIBKRIGING_EXPORT WarpMLPJoint {
 public:
  using Act = WarpMLP::Act;

  WarpMLPJoint(arma::uword d_in,
               const std::vector<arma::uword>& hidden_dims,
               arma::uword d_out = 2,
               Act activation = Act::SELU,
               uint64_t seed = 42);

  arma::uword input_dim() const { return m_d_in; }
  arma::uword output_dim() const { return m_d_out; }
  arma::uword n_params() const { return m_n_params; }

  arma::vec get_params() const;
  void set_params(const arma::vec& p);

  /// Forward: X (n × d_in) → Φ (n × d_out)
  arma::mat forward(const arma::mat& X) const;

  /// Backward: given dL/dΦ (n × d_out), compute dL/d(params)
  arma::vec backward(const arma::mat& X, const arma::mat& dL_dPhi) const;

  std::string describe() const;

 private:
  arma::uword m_d_in, m_d_out, m_n_params = 0;
  Act m_act;
  std::vector<arma::mat> m_W;
  std::vector<arma::vec> m_b;
  void count_params();
};

class LIBKRIGING_EXPORT WarpEmbedding final : public IWarp {
 public:
  WarpEmbedding(arma::uword n_levels, arma::uword embed_dim, uint64_t seed = 42);
  arma::uword output_dim() const override { return m_embed_dim; }
  arma::uword n_params() const override;
  arma::vec get_params() const override;
  void set_params(const arma::vec& p) override;
  arma::mat forward(const arma::vec& x) const override;
  arma::vec backward(const arma::vec& x, const arma::mat& dL_dPhi) const override;
  std::string describe() const override;

 private:
  arma::uword m_n_levels;
  arma::uword m_embed_dim;
  arma::mat m_E;  ///< embedding matrix (n_levels × embed_dim)
};

class LIBKRIGING_EXPORT WarpOrdinal final : public IWarp {
 public:
  explicit WarpOrdinal(arma::uword n_levels, uint64_t seed = 42);
  arma::uword output_dim() const override { return 1; }
  arma::uword n_params() const override;
  arma::vec get_params() const override;
  void set_params(const arma::vec& p) override;
  arma::mat forward(const arma::vec& x) const override;
  arma::vec backward(const arma::vec& x, const arma::mat& dL_dPhi) const override;
  std::string describe() const override;

 private:
  arma::uword m_n_levels;
  arma::vec m_raw_gaps;  ///< (L-1) unconstrained; actual gaps = exp(raw)
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
   * @brief Construct with warping specification per variable (strings).
   * @param warping   vector of warp spec strings, one per input dimension
   *                  e.g. {"kumaraswamy", "categorical(5,2)", "none"}
   * @param kernel    base kernel: "gauss", "matern3_2", "matern5_2", "exp"
   */
  LIBKRIGING_EXPORT WarpKriging(const std::vector<std::string>& warping, const std::string& kernel = "gauss");

  /**
   * @brief Full constructor with string warping and immediate fitting.
   */
  LIBKRIGING_EXPORT WarpKriging(const arma::vec& y,
                                const arma::mat& X,
                                const std::vector<std::string>& warping,
                                const std::string& kernel,
                                const std::string& regmodel = "constant",
                                bool normalize = false,
                                const std::string& optim = "BFGS+Adam",
                                const std::string& objective = "LL",
                                const std::map<std::string, std::string>& parameters = {});

  // -----------------------------------------------------------------------
  //  Fitting
  // -----------------------------------------------------------------------

  LIBKRIGING_EXPORT void fit(const arma::vec& y,
                             const arma::mat& X,
                             const std::string& regmodel = "constant",
                             bool normalize = false,
                             const std::string& optim = "BFGS+Adam",
                             const std::string& objective = "LL",
                             const std::map<std::string, std::string>& parameters = {});

  // -----------------------------------------------------------------------
  //  Prediction
  // -----------------------------------------------------------------------

  LIBKRIGING_EXPORT std::tuple<arma::vec, arma::vec, arma::mat, arma::mat, arma::mat> predict(
      const arma::mat& x_new,
      bool withStd = true,
      bool withCov = false,
      bool withDeriv = false) const;

  // -----------------------------------------------------------------------
  //  Simulation
  // -----------------------------------------------------------------------

  LIBKRIGING_EXPORT arma::mat simulate(int nsim, uint64_t seed, const arma::mat& x_new) const;

  // -----------------------------------------------------------------------
  //  Update
  // -----------------------------------------------------------------------

  LIBKRIGING_EXPORT void update(const arma::vec& y_new, const arma::mat& X_new);

  // -----------------------------------------------------------------------
  //  Log-likelihood
  // -----------------------------------------------------------------------

  LIBKRIGING_EXPORT double logLikelihood() const;

  LIBKRIGING_EXPORT std::tuple<double, arma::vec, arma::mat> logLikelihoodFun(const arma::vec& theta_gp,
                                                                              bool withGrad = true,
                                                                              bool withHess = false) const;

  // -----------------------------------------------------------------------
  //  Accessors
  // -----------------------------------------------------------------------

  LIBKRIGING_EXPORT std::string summary() const;

  const arma::mat& X() const { return m_X; }
  const arma::vec& y() const { return m_y; }
  std::string kernel() const { return m_kernel_name; }
  arma::vec theta() const { return m_theta; }
  double sigma2() const { return m_sigma2; }
  bool is_fitted() const { return m_fitted; }
  arma::uword feature_dim() const { return m_feature_dim; }
  const std::vector<WarpSpec>& warping() const { return m_warp_specs; }

  /// Get warping specs as a vector of strings
  std::vector<std::string> warping_strings() const {
    std::vector<std::string> result;
    for (const auto& s : m_warp_specs)
      result.push_back(s.to_string());
    return result;
  }

  /// Access a specific warp function (e.g. to inspect embeddings)
  const IWarp& warp(arma::uword dim) const { return *m_warps[dim]; }

 private:
  // ---- data ---------------------------------------------------------------
  arma::vec m_y;
  arma::mat m_X;
  arma::mat m_Phi;   ///< warped design (n × feature_dim)
  arma::mat m_dPhi;  ///< precomputed pairwise diffs (feature_dim × n*n), like Kriging's m_dX

  // ---- warping ------------------------------------------------------------
  std::vector<WarpSpec> m_warp_specs;
  std::vector<std::unique_ptr<IWarp>> m_warps;  ///< per-variable
  std::unique_ptr<WarpMLPJoint> m_joint_warp;   ///< joint MLP (if any)
  bool m_has_joint = false;
  arma::uword m_feature_dim = 0;

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
  std::string m_kernel_name;
  WarpBaseKernel m_base_kernel = WarpBaseKernel::Gauss;
  std::function<double(const arma::vec&, const arma::vec&)> _Cov;
  std::function<arma::vec(const arma::vec&, const arma::vec&)> _DlnCovDtheta;
  std::function<arma::vec(const arma::vec&, const arma::vec&)> _DlnCovDx;
  void make_Cov(const std::string& kernel);
  arma::vec m_theta;
  double m_sigma2 = 1.0;

  // ---- GP cache -----------------------------------------------------------
  arma::mat m_R;      ///< correlation matrix (n×n)
  arma::mat m_C;      ///< Cholesky(R + nugget), lower
  arma::vec m_alpha;  ///< R^{-1}(y - Fβ)
  double m_logdet = 0.0;

  bool m_fitted = false;

  // ---- optimiser ----------------------------------------------------------
  arma::uword m_max_iter_bfgs = 100;
  arma::uword m_max_iter_adam = 10;
  double m_adam_lr = 1e-3;

  // ---- private helpers ----------------------------------------------------
  static WarpBaseKernel parse_kernel(const std::string& name);
  std::unique_ptr<IWarp> make_warp(const WarpSpec& spec) const;
  void build_warps();

  arma::mat build_trend_matrix(const arma::mat& X) const;
  arma::mat apply_warping(const arma::mat& X) const;

  // ---- Precomputed pairwise differences -----------------------------------
  /// Compute m_dPhi from m_Phi (feature_dim × n*n layout, like Kriging's m_dX)
  void compute_dPhi();

  // ---- Correlation matrix (σ²=1) -----------------------------------------
  /// Build cross-correlation  r(Φ_new, Φ_train)  →  (m × n)
  arma::mat build_Rcross(const arma::mat& Phi_new, const arma::mat& Phi_train) const;

  // ---- Concentrated profile log-likelihood --------------------------------
  //
  //  σ̂² and β̂ are computed analytically from R(θ) and y:
  //    β̂ = (F^T R⁻¹ F)⁻¹ F^T R⁻¹ y
  //    σ̂² = (y - Fβ̂)^T R⁻¹ (y - Fβ̂) / n
  //    LL_conc(θ) = -n/2 [1 + log(2π) + log(σ̂²)] - ½ log|R|
  //

  /// Refresh all cached quantities (Φ, R, Cholesky, β̂, σ̂², α) from
  /// the current warp params and θ.
  void refresh_cache();
  /// Like refresh_cache but skips recomputing Φ (use when only θ changed).
  void refresh_cache_theta_only();
  void normalise_data();

  /// Compute concentrated LL from current cache
  double concentrated_ll() const;

  // ---- Analytical gradient ∂LL/∂θ ----------------------------------------
  /// Build matrix ∂R/∂θ_k  (n×n)  for the k-th range parameter
  arma::mat build_dR_dtheta_k(const arma::mat& Phi, arma::uword k) const;

  /// Compute concentrated LL and its gradient w.r.t. log(θ)
  std::pair<double, arma::vec> concentrated_ll_and_grad_theta() const;

  // ---- Gradient ∂LL/∂(warp params) via backprop through kernel ------------
  arma::mat dK_dPhi(const arma::mat& Phi, const arma::mat& dL_dK) const;
  arma::vec warp_gradient() const;

  // ---- Warp param packing (θ is NOT in here — optimised separately) ------
  arma::uword total_warp_params() const;
  arma::vec pack_warp_params() const;
  void unpack_warp_params(const arma::vec& w);

  // ---- Optimisation -------------------------------------------------------
  //
  //  Bi-level strategy:
  //    Outer loop: Adam on warp params
  //      Inner loop: L-BFGS on log(θ) with analytical gradient
  //        σ̂² and β̂ concentrated out (computed analytically)
  //

  /// Joint optimisation (bi-level Adam+BFGS or joint L-BFGS-B)
  void optimise_joint(const std::string& method);

  /// Internal: initialise from pre-parsed WarpSpecs
  void init_from_specs(const std::vector<WarpSpec>& specs, const std::string& kernel);
};

}  // namespace libKriging

#endif  // LIBKRIGING_WARP_KRIGING_HPP
