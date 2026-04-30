#ifndef LIBKRIGING_KRIGINGIMPL_HPP
#define LIBKRIGING_KRIGINGIMPL_HPP

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <tuple>

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/Trend.hpp"
#include "libKriging/libKriging_exports.h"
#include "libKriging/utils/nlohmann/json_fwd.hpp"

/// Shared state and helpers for the Kriging / NuggetKriging / NoiseKriging
/// family. Holds data members, accessors, the covariance functor bundle, and
/// the common `covMat` implementation. Class-specific state (nugget, noise,
/// reparam statics, …) lives in the derived classes.
class KrigingImpl {
 public:
  /// GP fit artifacts produced by populate_Model.  The diagonal treatment
  /// of the covariance matrix (plain correlation / nugget / per-point noise)
  /// is handled by the owning class in its populate_Model override; what this
  /// struct carries is identical across the three variants.
  struct KModel {
    arma::mat R;      ///< normalized covariance (correlation + optional diag)
    arma::mat L;      ///< Cholesky lower of R
    arma::mat Linv;   ///< L⁻¹
    arma::mat Rinv;   ///< R⁻¹ = L⁻ᵀ L⁻¹
    arma::mat Fstar;  ///< L \ F (whitened trend)
    arma::vec ystar;  ///< L \ y (whitened observations)
    arma::mat Rstar;  ///< chol_upper(F' R⁻¹ F)
    arma::mat Qstar;  ///< Q factor from QR of Fstar (LOO path)
    arma::vec Estar;  ///< L \ (y - F β̂) (whitened residual)
    double SSEstar;   ///< Estar' Estar
    arma::vec betahat;
  };

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

  LIBKRIGING_EXPORT arma::mat covMat(const arma::mat& X1, const arma::mat& X2);

  /// Optional feature map Φ : (n × d_in) → (n × d_out).  Used by simulate_impl
  /// and update_simulate_impl to transform normalized inputs to the kernel space
  /// before covariance computations.  Empty (default) = identity (no transform).
  using FeatureMap = std::function<arma::mat(const arma::mat&)>;

  static arma::vec ones;

 protected:
  KrigingImpl() = default;
  KrigingImpl(const KrigingImpl&) = default;
  KrigingImpl(KrigingImpl&&) = default;
  KrigingImpl& operator=(const KrigingImpl&) = default;
  KrigingImpl& operator=(KrigingImpl&&) = default;
  ~KrigingImpl() = default;

  // Main model data
  std::string m_covType;
  arma::mat m_X;
  arma::rowvec m_centerX;
  arma::rowvec m_scaleX;
  arma::vec m_y;
  double m_centerY{};
  double m_scaleY{};
  bool m_normalize{};
  Trend::RegressionModel m_regmodel;
  std::string m_optim;
  std::string m_objective;

  // Auxiliary data
  arma::mat m_dX;
  arma::vec m_maxdX;
  /// Per-observation noise variances (in normalized-y space). Empty when
  /// the model has no per-point noise channel (Kriging / NuggetKriging /
  /// MLPKriging always empty; NoiseKriging always set; WarpKriging optional).
  /// Hoisted here so save/load and update hooks can reach it uniformly.
  arma::vec m_noise;
  arma::mat m_F;
  arma::mat m_T;
  arma::mat m_R;  // required for the "update" methods
  arma::mat m_M;
  arma::mat m_star;
  arma::mat m_circ;
  arma::vec m_z;
  arma::mat m_Rinv;
  arma::vec m_beta;
  bool m_est_beta{};
  arma::vec m_theta;
  bool m_est_theta{};
  double m_sigma2{};
  bool m_est_sigma2{};
  bool m_is_empty = true;  // forces the model to be made from scratch first time (no update)

  // Simulation stored data (members common to all three variants)
  arma::mat lastsim_Xn_n;
  arma::mat lastsim_y_n;
  int lastsim_nsim{};
  int lastsim_seed{};
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
  arma::mat lastsimup_y_u;
  arma::mat lastsimup_Wtild_nKu;
  arma::mat lastsimup_R_uo;
  arma::mat lastsimup_R_un;
  arma::mat lastsimup_R_uu;

  // Covariance functor bundle
  std::function<double(const arma::vec&, const arma::vec&)> _Cov;
  std::function<arma::vec(const arma::vec&, const arma::vec&)> _DlnCovDtheta;
  std::function<arma::vec(const arma::vec&, const arma::vec&)> _DlnCovDx;
  double _Cov_pow{};

  // Parse `covType` and populate the _Cov / _DlnCov* functors + _Cov_pow.
  void make_Cov(const std::string& covType);

  /// Shared GP-fit scaffolding. Builds the Cholesky factor (via `cholCov` or
  /// the rank-augmenting `update_cholCov`), whitens F/y, forms Rstar, and
  /// computes the GLS β̂ + residual SSE.
  ///
  /// `alpha` scales the correlation (=1 for plain/noise, =σ²/(σ²+τ²) for
  /// nugget). `diag_norm` is a per-point diagonal (use the `ones` sentinel
  /// for plain/nugget, `1 + noise/σ²` for noise). `update_eligible` is the
  /// class-specific predicate for reusing the previous factorization.
  ///
  /// On return, `m.betahat` holds the GLS estimate regardless of
  /// `m_est_beta`; derived wrappers are responsible for zeroing it (and
  /// optionally recomputing SSE) when `m_est_beta` is false.
  void populate_Model(KModel& m,
                      const arma::vec& theta,
                      double alpha,
                      const arma::vec& diag_norm,
                      bool update_eligible,
                      std::map<std::string, double>* bench) const;

  /// Preallocate a `KModel` sized from (n=m_X.n_rows, p=m_F.n_cols).
  /// `Linv` and `Rinv` are left empty (computed on demand / during populate).
  KModel allocate_KModel() const;

  /// Unified predict implementation shared by Kriging / NuggetKriging /
  /// NoiseKriging.  The three variants only differ in:
  ///
  ///   * `R_on_factor`: multiplies the off-diagonal of the (o,n) correlation
  ///     block (1.0 for plain/noise, α=σ²/(σ²+τ²) for nugget).  When a predict
  ///     point coincides with an observation point, the entry is forced to
  ///     1.0 regardless of this factor (preserves Nug's noiseless-at-obs
  ///     semantics; no-op for K/Noise where factor=1).
  ///   * `R_nn_factor` / `R_nn_diag`: passed through to `covMat_sym_X` when
  ///     `return_cov` is true (K: 1 / empty; Nug: α / ones; Noise: 1 / empty).
  ///   * `var_scale`: σ² multiplier applied to the variance/cov/derivative
  ///     outputs (K: σ² with optional LMP correction; Nug: σ²/α with LMP;
  ///     Noise: m_sigma2 — LMP unsupported).
  std::tuple<arma::vec, arma::vec, arma::mat, arma::mat, arma::mat> predict_impl(const arma::mat& X_n,
                                                                                 bool return_stdev,
                                                                                 bool return_cov,
                                                                                 bool return_deriv,
                                                                                 double R_on_factor,
                                                                                 double R_nn_factor,
                                                                                 const arma::vec& R_nn_diag,
                                                                                 double var_scale);

  /// Unified simulate scaffolding shared by the three variants.  Builds R_nn,
  /// R_on, draws y_n ~ N(yhat_n, σ² · Sigma/Sigma_divisor), and (when
  /// `will_update`) stores the common `lastsim_*` members needed by
  /// `update_simulate`.  The Noise-specific post-hoc noise term and the
  /// class-specific `lastsim_with_*` flags stay in the wrappers.
  ///
  /// Parameters:
  ///   * `R_on_factor`: multiplies off-diagonal R_on entries (Nug: α).
  ///   * `R_on_coincident_to_one`: when true, force R_on[i,j]=1 whenever
  ///     Xn_o[:,i] == Xn_n[:,j] (Nug simulate with `with_nugget=true`).
  ///   * `R_nn_factor` / `R_nn_diag`: passed through to `covMat_sym_X`.
  ///   * `Sigma_divisor`: Sigma_nKo is divided by this before the Cholesky
  ///     used to sample (Nug: α; K/Noise: 1).
  ///   * `use_qr_for_circ`: when true, build `lastsim_circ_on` via QR of the
  ///     augmented `Fstar_on` (K/Nug); when false, via `chol_upper` of the
  ///     Gram matrix (Noise).  Preserves the existing bit-level divergence.
  arma::mat simulate_impl(int nsim,
                          int seed,
                          const arma::mat& X_n,
                          bool will_update,
                          double R_on_factor,
                          bool R_on_coincident_to_one,
                          double R_nn_factor,
                          const arma::vec& R_nn_diag,
                          double Sigma_divisor,
                          bool use_qr_for_circ,
                          const FeatureMap& phi = {});

  /// Unified update_simulate shared by the three variants. The wrapper is
  /// responsible for class-specific input validation (Noise: noise_u length)
  /// and post-hoc adjustments (Noise: += eps using lastsim_with_noise). When
  /// `allow_cache=false` the impl forces a full recompute — used by Noise to
  /// preserve existing behavior where `lastsimup_noise_u` is never written.
  ///
  /// Parameters mirror `simulate_impl`:
  ///   * `R_uu_factor` / `R_uu_diag`: passed through to `covMat_sym_X`.
  ///   * `R_uo_factor`: multiplies off-diagonal R_uo entries (Nug: α).
  ///   * `R_un_factor`: multiplies off-diagonal R_un entries (Nug: α).
  ///   * `R_un_coincident_to_one`: when true, force R_un[i,j]=1 whenever
  ///     Xn_u[:,i] == lastsim_Xn_n[:,j] (Nug update_simulate with nugget).
  ///   * `Sigma_divisor`: divides the Sigma_uKno Cholesky argument (Nug: α).
  arma::mat update_simulate_impl(const arma::vec& y_u,
                                 const arma::mat& X_u,
                                 bool allow_cache,
                                 double R_uu_factor,
                                 const arma::vec& R_uu_diag,
                                 double R_uo_factor,
                                 double R_un_factor,
                                 bool R_un_coincident_to_one,
                                 double Sigma_divisor,
                                 const FeatureMap& phi = {});

  /// Unified pre-optim data setup for `fit()`. Normalizes y/X (when
  /// `normalize=true`), populates `m_centerX/m_scaleX/m_centerY/m_scaleY`,
  /// `m_X/m_y`, `m_dX/m_maxdX`, `m_regmodel/m_F`, sets `m_est_beta` and
  /// `m_beta` (if forced via `beta` with `is_beta_estim=false`).
  ///
  /// Returns the normalized `theta0` matrix (empty if `theta` is not set).
  /// Wrappers handle class-specific extras (Noise: normalize and store
  /// `m_noise = noise / scaleY²`) and the optim==none / BFGS branches.
  ///
  /// `theta` and `beta` are taken as the matching subset of each class's
  /// `Parameters` struct (which has different `sigma2` types — sigma2 stays
  /// in the wrapper).
  arma::mat fit_setup_impl(const arma::vec& y,
                           const arma::mat& X,
                           const Trend::RegressionModel& regmodel,
                           bool normalize,
                           bool is_beta_estim,
                           const std::optional<arma::vec>& beta,
                           const std::optional<arma::mat>& theta);

  /// Dump the inherited GP state to a JSON object using the
  /// Kriging/NuggetKriging/NoiseKriging schema. Used by `save()` in those
  /// three classes; WarpKriging/MLPKriging use a different schema and roll
  /// their own serializer. The caller is responsible for writing the
  /// `version` and `content` tags as well as any class-specific fields.
  void dump_common_to_json(nlohmann::json& j) const;

  /// Inverse of `dump_common_to_json`. The caller is responsible for
  /// validating `version`/`content`, instantiating the derived class with
  /// the right `covType`, and then calling this on its `KrigingImpl` base.
  /// Loads `m_noise` only when present (older saved files predating the
  /// hoist will lack it; we default to empty).
  void load_common_from_json(const nlohmann::json& j);

  /// Write the shared upper portion of `summary()` to `oss`: the not-fitted
  /// short form, or the data/trend/variance/covariance-header/range section.
  /// Auto-emits the `* noise:` line when `m_noise` is non-empty (NoiseKriging
  /// always; WarpKriging when fit with noise). Returns `true` if the model
  /// has data — i.e. the caller should continue to print class-specific
  /// covariance extras (NuggetKriging: nugget line) and call
  /// `summary_bottom`. Returns `false` if the unfitted short form was written
  /// and the caller should stop.
  [[nodiscard]] bool summary_top(std::ostringstream& oss, const arma::mat* X_display_override = nullptr) const;

  /// Write the shared lower portion of `summary()` to `oss`: the
  /// `* fit:` block (objective + optim).
  void summary_bottom(std::ostringstream& oss) const;

  /// Move KModel fields (R, L→T, Rinv, Fstar→M, Rstar→circ, betahat→beta,
  /// Estar→z) into the base members.  Caller computes m_logdet afterward.
  void commit_model(KModel& m);

  /// Unified incremental-update (no refit) shared by the three variants.
  /// Normalizes the new data, extends `m_X`/`m_y`, runs the class-specific
  /// `extend_class_data` hook (Noise: also extends `m_noise`; K/Nug: no-op),
  /// recomputes `m_dX`/`m_maxdX`/`m_F`, calls `build_model` (which delegates
  /// to the class-specific `make_Model` carrying alpha / sigma2), then moves
  /// the resulting `KModel` into `m_T/m_R/m_M/m_circ/m_star/m_Rinv` and
  /// updates `m_beta/m_z/m_sigma2` according to `m_est_beta`/`m_est_sigma2`.
  /// Wrappers are responsible for input validation (dimension checks).
  void update_no_refit_impl(const arma::vec& y_u,
                            const arma::mat& X_u,
                            const std::function<void()>& extend_class_data,
                            const std::function<KModel()>& build_model);

  // --- Objective-function evaluation helpers ---------------------------------

  /// Print a bench timing map in the standard "| key | value |" format.
  static void print_bench(const std::map<std::string, double>& bench);

  /// Dispatch an objective function (scalar + gradient, no hessian) with
  /// optional benchmark timing.  `fn(grad_out, bench_out)` must return the
  /// objective value and write the gradient into *grad_out when non-null.
  template <typename Fn>
  static std::tuple<double, arma::vec> eval_objective(arma::uword n_params,
                                                      bool return_grad,
                                                      bool bench,
                                                      Fn&& fn) {
    double val = -1;
    arma::vec grad;
    if (bench) {
      std::map<std::string, double> bench_map;
      if (return_grad) {
        grad = arma::vec(n_params);
        val = fn(&grad, &bench_map);
      } else
        val = fn(nullptr, &bench_map);
      print_bench(bench_map);
    } else {
      if (return_grad) {
        grad = arma::vec(n_params);
        val = fn(&grad, nullptr);
      } else
        val = fn(nullptr, nullptr);
    }
    return {val, std::move(grad)};
  }

  /// Shared inner loop for ∂LL/∂θ_k.  Accumulates term1_vec[k] and
  /// term2_vec[k] (k=0..d-1) over upper-triangle (i,j) pairs using m_dX and
  /// _DlnCovDtheta.  Caller is responsible for pre-sizing both output vectors
  /// to d and zeroing them.
  void compute_ll_grad_theta_vecs(const arma::mat& R,
                                  const arma::mat& Rinv,
                                  const arma::mat& x,
                                  const arma::vec& theta,
                                  arma::vec& term1_vec,
                                  arma::vec& term2_vec) const;

  /// Shared for-k gradient loop for ∂LMP/∂θ_k (both Kriging and
  /// NuggetKriging _logMargPost).  Returns the ans vector (length d) from
  /// the Wb_k accumulation.  Caller appends class-specific prior correction.
  arma::vec compute_lmp_theta_ans(const KModel& m,
                                  const arma::vec& theta,
                                  double sigma2,
                                  const arma::mat& Rinv_X_Xt_Rinv_X_inv_Xt_Rinv,
                                  const arma::mat& Q_output,
                                  std::map<std::string, double>* bench) const;
};

#endif  // LIBKRIGING_KRIGINGIMPL_HPP
