  
  
    Investigation: bringing WarpKriging / MLPKriging under KrigingImpl

  I read both headers, the relevant .cpp sections, and compared against what KrigingImpl currently abstracts. Here is the assessment.

  1. Class topology and shape

  See [refactor_table.md](refactor_table.md) for the up-to-date per-class
  routing/inheritance table (last updated 2026-04-25, post W-1/W-2/W-3).

  2. What Φ-space means for KrigingImpl reuse

  KrigingImpl currently assumes m_X is the kernel input: populate_Model reads m_dX directly, predict_impl/simulate_impl/update_simulate_impl recompute distances from m_X and the new
  X_n/X_u. Warp/MLP need the kernel evaluated on Φ(X), so two clean options:

  - (A) Replace m_X/m_dX with Φ-space contents in derived classes. m_X already stores normalized data; under this option Warp/MLP would store Φ(X_normalized) in m_X, and m_dX would be
  m_dPhi. Predict/update would have to warp X_n/X_u before calling the impl (the impl would then operate purely in Φ-space). Pro: zero changes to KrigingImpl. Con: m_X() accessor becomes
  Φ-space, breaking the public-API meaning users have today (R/Python tests inspect X()).
  - (B) Add a virtual hook feature_map(X) -> Φ. KrigingImpl keeps m_X in original (normalized) input space but routes through feature_map everywhere it currently does Xn_n = ...,
  distances, covMat. Default implementation returns X unchanged (identity). Pro: preserves accessor semantics. Con: introduces virtuality on a hot path, and requires re-routing every     
  covMat_*/compute_dX/covMat_sym_X call site in KrigingImpl.
                                                                                                                                                                                           
  I'd recommend (A) with a renamed protected member — store m_Phi separately in the derived class but populate m_X = m_Phi from the derived fit_setup/update_no_refit hooks. The base never
   needs to know the difference, and you keep X() returning the original via an override in the derived class that exposes its own m_X_raw.
                                                                                                                                                                                           
  3. Per-method reuse assessment                                                                                                                                                           
   
  Directly reusable as-is (small wrappers):                                                                                                                                                
  - make_Cov — Warp/MLP have nearly identical functor bundles; use WarpBaseKernel→string adapter or fold WarpBaseKernel into m_covType.
  - covMat() accessor — works once Φ-space distances are correct.                                                                                                                          
  - populate_Model — Warp's WKModel/MLP's MLPKModel are subsets of KModel (no Linv/Qstar). They could simply use the base KModel. Diagonal handling already supported via diag_norm
  parameter (Warp's 1+num_nugget+noise/σ² and MLP's 1+1e-8 map cleanly).                                                                                                                   
  - allocate_KModel — same.                                                                                                                                                                
                                                                                                                                                                                           
  Reusable with modest wrapper effort:                                                                                                                                                     
  - predict_impl — works in Φ-space if (A) is taken. The finite-difference derivative path is currently absent from KrigingImpl (it uses analytical gradients via _DlnCovDx); Warp/MLP need
   finite-difference because of warp Jacobians. This is the biggest blocker for predict_impl reuse — would need an extra bool use_fd_jacobian parameter, OR Warp/MLP expose a J_warp       
  callback.                                                                                                                                                                         
  - simulate_impl — Warp/MLP simulation differs in lastsim_Phi_n cache (one extra member) and Warp's validate_discrete_columns call. Otherwise the FOXY scaffolding is identical. With (A),
   Warp's Phi is already in the base's lastsim_Xn_n slot, but per-class lastsim_Phi_n becomes redundant.                                                                                   
  - update_simulate_impl — same shape; Warp's noise path matches Noise's with_noise logic well. The lastsimup_noise_u cache key makes Warp closer to Noise than to K/Nug.                  
  - update_no_refit_impl — directly reusable through the existing extend_class_data/build_model hooks. Warp's hook would re-warp m_X and call compute_dPhi; MLP's would do the same with
  its joint warp.                                                                                                                                                                          
                                                                                                                                                                                           
  Not reusable (genuinely class-specific):                                                                                                                                                 
  - fit() — bi-level Adam+BFGS optimizer is structurally different from K/Nug/Noise's L-BFGS-B multistart. The pre-optim setup can reuse fit_setup_impl with one caveat: Warp's per-dim    
  normalization (only continuous dims) doesn't fit the rowvec model — needs an is_continuous mask parameter or the derived class still owns normalization.                                 
  - optimise_joint() — outer Adam loop on warp params with inner BFGS on θ has no analogue in the existing impl.                                                                           
  - concentrated_ll_and_grad_theta() — Warp/MLP compute the gradient w.r.t. log(θ) inline rather than going through _logLikelihoodGrad. Fold or keep separate.                             
  - dK_dPhi/warp_gradient — required for the Adam outer loop; entirely Warp/MLP-specific. 
  
  
  
  4. Concrete complications

  1. Two distinct KModel structs. Warp's WKModel and MLP's MLPKModel lack Linv and Qstar. Easiest fix: drop them and use the
  base KModel (those fields stay empty when not filled — that's already how Linv works in the base).
  2. m_noise lives only in Warp. WarpKriging supports per-point observation noise through the same channel as NoiseKriging
  (commit history shows it was added later). NoiseKriging keeps m_noise in the wrapper; Warp could do the same. Or hoist m_noise
   to KrigingImpl as an optional/empty vec — that would actually clean up Noise too.
  3. m_is_continuous mask for Warp. Discrete dims must not be normalized, must not be passed through compute_dX blindly, and
  must be validated to be non-negative integers. Either keep all of this in Warp (and have it call fit_setup_impl with
  normalize=false + do its own per-dim normalization first), or extend fit_setup_impl with an optional mask.
  4. MLPKriging has no update_simulate noise / no warp_params freezing. Its API is a strict subset of WarpKriging's. Treat
  MLPKriging as WarpKriging with a single mlp_joint(...) warp — the existing WarpType::MLPJoint path already does this. The
  cleanest "MLP under KrigingImpl" is to merge MLPKriging into WarpKriging entirely (or make MLPKriging a thin facade over
  WarpKriging with mlp_joint warping), then port WarpKriging once.
  5. Per-thread cloning (clone_for_thread) needs to deep-clone std::unique_ptr<IWarp> and std::unique_ptr<WarpMLPJoint>.
  KrigingImpl currently has no clone hook because K/Nug/Noise are trivially copyable. Either add a virtual clone or keep the
  multistart loop in the derived class (which it already is).
  6. Bench files. bench-warpkriging.cpp and bench-mlpkriging.cpp are pre-existing build failures from commit 81b894e. They'll
  need updating in lockstep with any refactor that changes Warp/MLP fit() signatures.
  
  
  5. Risk and recommended phasing

  ┌──────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────┬─────────────────────────┬────────────────────────────────┐
  │    Stage     │                                                  Scope                                                  │          Risk           │         Tests to track         │
  ├──────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────┼────────────────────────────────┤
  │ W-0          │ Decide MLPKriging fate: collapse into WarpKriging with mlp_joint warp, or keep separate                 │ low (design-only)       │ —                              │
  ├──────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────┼────────────────────────────────┤
  │ W-1          │ Drop WKModel/MLPKModel, use base KModel. No inheritance yet.                                            │ low                     │ Warp + MLP test suites         │
  ├──────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────┼────────────────────────────────┤
  │ W-2          │ Make WarpKriging contain a KrigingImpl-like helper or inherit (private) and route                       │ medium                  │ full suite                     │
  │              │ populate_Model/covMat/make_Cov through the base. Keep all simulate/predict in Warp for now.             │                         │                                │
  ├──────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────┼────────────────────────────────┤
  │ W-3          │ Route update_no_refit_impl and fit_setup_impl from Warp. The Adam+BFGS outer loop stays in Warp.        │ medium                  │ full suite                     │
  │              │ Per-dim normalization stays in Warp.                                                                    │                         │                                │
  ├──────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────┼────────────────────────────────┤
  │ W-4          │ Route simulate_impl + update_simulate_impl. Requires storing Φ in m_X slot under option (A) and         │ high — bit-stability of │ seeded                         │
  │              │ confirming lastsim_Xn_n semantics.                                                                      │  FOXY draws             │ simulate/update_simulate tests │
  ├──────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────┼────────────────────────────────┤
  │ W-5          │ Route predict_impl — only if a J_warp callback is added to the base, since Warp/MLP need FD Jacobians.  │ medium                  │ predict + derivative tests     │
  ├──────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────┼────────────────────────────────┤
  │ W-6          │ Collapse MLPKriging into WarpKriging                                                                    │ low if W-0 chose this   │ MLP tests                      │
  │ (optional)   │                                                                                                         │                         │                                │
  └──────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────┴─────────────────────────┴────────────────────────────────┘

  6. Recommendation

  Given how invasive option (A)/(B) is and how much of WarpKriging's value lives in code KrigingImpl can't naturally absorb (the warp gradients, the bi-level optimizer, per-dim
  normalization), the realistic win is stages W-1 through W-3 — kill the duplicate KModel structs, share populate_Model/covMat/make_Cov/fit_setup_impl/update_no_refit_impl. That removes
  maybe 250–400 LOC of duplication and aligns the diagonal-handling semantics across the family.

  Stages W-4 and W-5 give marginal gains in exchange for substantial bit-stability risk (FOXY caches, FD vs analytical Jacobian). I'd defer them unless a separate motivation emerges.

  The highest-leverage independent decision is W-0: if MLPKriging can become a thin facade over WarpKriging (mlp_joint already exists as a WarpSpec), that's a ~1300-line deletion with no
  impact on KrigingImpl at all — worth doing first.