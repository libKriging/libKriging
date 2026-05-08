<!-- cspell:disable -->

# Class topology and shape

_Last updated 2026-05-06 — post W-1/W-2/W-3 + A/A2/B/D/F + G + C2 + **N (noise unification)**._
_NuggetKriging and NoiseKriging have been merged into Kriging via the `NoiseModel` enum (`None` | `Nugget` | `Heterogeneous`)._

```text
┌─────────────────────┬────────────────────────────────────────────────────────────────┬────────────────────────────────────────────────┬───────────────────────────────────────────────────────┐
│                     │                           Kriging                              │                  WarpKriging                   │                    MLPKriging                         │
│                     │               (None / Nugget / Heterogeneous)                  │                                                │                                                       │
├─────────────────────┼────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ File LOC            │ 1854  (was 1897 + 1470 + 1116 across 3 files pre-N)            │ 2661  (was 2940)                               │ 177  (was 1361)                                       │
│                     │ NoiseKriging.cpp + NuggetKriging.cpp removed (stage N)         │ KrigingImpl.cpp: 879 (was 647 — +232 helpers)  │                                                       │
├─────────────────────┼────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Inherits KrigingImpl│ protected (pre-existing)                                       │ protected (W-2)                                │ does NOT inherit — holds WarpKriging m_impl (A)       │
├─────────────────────┼────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Kernel input slot   │ inherited m_X (= normalized X)                                 │ inherited m_X (= Φ(X_norm), Option A)          │ via m_impl.m_X (Φ-space)                              │
│                     │                                                                │ raw kept in m_X_raw                            │ raw kept in m_impl.m_X_raw                            │
├─────────────────────┼────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Distance cache      │ inherited m_dX / m_maxdX                                       │ inherited m_dX / m_maxdX (set in              │ via m_impl                                            │
│                     │                                                                │ refresh_cache — G)                             │                                                       │
├─────────────────────┼────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ m_noise location    │ inherited; empty (None), always set (Heterogeneous),           │ inherited (optional, set when fit with noise)  │ via m_impl (always empty — MLP has no noise path)     │
│                     │ unused (Nugget uses m_nugget/m_est_nugget instead)             │ — hoisted in D                                 │                                                       │
├─────────────────────┼────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ KModel struct       │ shared KrigingImpl::KModel                                     │ shared KrigingImpl::KModel (W-1)               │ via m_impl (no own KModel)                            │
├─────────────────────┼────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ populate_Model      │ KrigingImpl::populate_Model (diag branch per NoiseModel)       │ KrigingImpl::populate_Model (W-2)              │ via m_impl                                            │
├─────────────────────┼────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ make_Cov            │ KrigingImpl::make_Cov                                          │ KrigingImpl::make_Cov via name normalizer (W-2)│ via m_impl                                            │
├─────────────────────┼────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ covMat() accessor   │ inherited                                                       │ inherited                                      │ via m_impl                                            │
├─────────────────────┼────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Trend matrix        │ regressionModelMatrix(m_regmodel, m_X)                         │ same (Φ-space; build_trend_matrix wraps)       │ via m_impl                                            │
├─────────────────────┼────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Normalization       │ rowvec on all dims (base helper)                               │ per-dim continuous-only (centerX(j)=0,         │ via m_impl                                            │
│                     │                                                                │ scaleX(j)=1 for non-continuous → reduces to    │                                                       │
│                     │                                                                │ rowvec equivalence)                            │                                                       │
├─────────────────────┼────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Diag sentinel       │ None→1  |  Nugget→1+α  |  Heterogeneous→1+noise_i/σ²          │ 1 + num_nugget [+ noise/σ²]                    │ 1 + 1e-8 (via m_impl with mlp_joint warp spec)        │
├─────────────────────┼────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Closed-form σ̂²      │ None→yes  |  Nugget→no (joint optim of (θ,α))                 │ yes when m_noise empty, else 1-D               │ yes (via m_impl, noise always empty)                  │
│                     │ Heterogeneous→no (per-point noise)                             │ golden-section over σ²                         │                                                       │
├─────────────────────┼────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Optim variables     │ None→θ(+β when fixed)  |  Nugget→(θ,α)  |  Noise→(θ,σ²)      │ (θ, warp_params, [σ² when noise])              │ (θ, warp_params) via m_impl                           │
├─────────────────────┼────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Optim algorithm     │ L-BFGS-B multistart                                            │ bi-level Adam(warp) ⊕ L-BFGS(θ)                │ via m_impl (bi-level Adam ⊕ L-BFGS with mlp_joint)    │
│                     │                                                                │ (stays in Warp); m_optim/m_objective now set   │                                                       │
├─────────────────────┼────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ update_no_refit     │ KrigingImpl::update_no_refit_impl                              │ KrigingImpl::update_no_refit_impl (W-3)        │ via m_impl (A)                                        │
│                     │                                                                │ +hook: extend m_X_raw / m_noise / re-warp m_X  │                                                       │
├─────────────────────┼────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ save / load         │ dump_common / load_common (B) with nugget/noise extras         │ dump_to_json calls dump_common_to_json + Warp  │ dump_to_json / load_from_json via m_impl (A2+G)       │
│                     │                                                                │ extras; load_from_json branches v2/v3 (G)      │ v3 schema; v2 legacy files still loadable             │
│                     │                                                                │ v3 schema ("covType"/"X_raw"); v2 still loads  │                                                       │
├─────────────────────┼────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ summary()           │ summary_top + summary_bottom (F); noise/nugget details         │ summary_top(&m_X_raw) + warpings block +       │ Delegates to m_impl.summary() + header rename (A)     │
│                     │ auto-included per NoiseModel                                   │ summary_bottom (G); X_display_override param   │                                                       │
│                     │                                                                │ shows input-space ranges not Φ-space           │                                                       │
├─────────────────────┼────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ fit_setup_impl      │ KrigingImpl::fit_setup_impl                                    │ NOT routed — θ₀ normalized by scaleX in base   │ via m_impl (A)                                        │
│                     │                                                                │ helper, but Warp's θ lives in feature_dim ≠    │                                                       │
│                     │                                                                │ input-d, so dimensions mismatch                │                                                       │
├─────────────────────┼────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ predict_impl        │ KrigingImpl::predict_impl                                      │ KrigingImpl::predict_impl (C2) with phi_fn +   │ via m_impl (A)                                        │
│                     │                                                                │ jac_fn lambdas; Xn_n_input saved before phi    │                                                       │
│                     │                                                                │ overwrites Xn_n; deriv chain-ruled to d_input  │                                                       │
├─────────────────────┼────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ simulate_impl       │ KrigingImpl::simulate_impl                                     │ KrigingImpl::simulate_impl (W-4) via phi_fn;   │ via m_impl (A)                                        │
│ update_simulate_impl│ KrigingImpl::update_simulate_impl                              │ wrapper keeps with_noise/lastsim_with_noise +  │                                                       │
│                     │                                                                │ noise_ok cache check (Warp-only extras)        │                                                       │
├─────────────────────┼────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ logLikelihoodFun θ- │ free helper (no member θ swap)                                 │ inline save/restore m_theta around             │ via m_impl (A)                                        │
│ swap pattern        │                                                                │ refresh_cache (E deferred — only 2 callsites,  │                                                       │
│                     │                                                                │ refresh_cache call is class-specific anyway)   │                                                       │
└─────────────────────┴────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────┴───────────────────────────────────────────────────────┘
```

## Net LOC delta on this branch

```text
                    pre-N     post-N    Δ vs pre-N    notes
Kriging.cpp         1837      1854      +17           NoiseModel enum + noise/nugget fit paths
NuggetKriging.cpp   1264         0     −1264          removed — merged into Kriging (stage N)
NoiseKriging.cpp     960         0      −960          removed — merged into Kriging (stage N)
WarpKriging.cpp     2661      2768     +107           noise path added (stage N)
MLPKriging.cpp       177       178       +1
KrigingImpl.cpp      879       877       −2
                                        ──────
                                        −2101 net vs pre-N state  (−3754 vs pre-W1 state)
```

## Binding summary (stage N — noise unification)

| Binding | Backward-compat shim | New unified API |
|---------|----------------------|-----------------|
| **R** | `NoiseKrigingClass.R` / `NuggetKrigingClass.R` kept as S3 wrappers delegating to `Kriging(noise_model=...)` | `Kriging(noise=vector)` / `Kriging(noise="nugget")` |
| **Python** | Old `PyNoiseKriging` / `PyNuggetKriging` files retained but NOT compiled | `Kriging(noise=array)` / `Kriging(noise="nugget")` |
| **Julia** | `NuggetKriging(...)` / `NoiseKriging(...)` kept as deprecated wrappers (`Base.depwarn`) | `Kriging(y, X, kernel; noise=vec\|"nugget")` |
| **Octave** | `NuggetKriging::*` / `NoiseKriging::*` dispatcher cases redirect to `Kriging_binding` | `Kriging(y, X, ..., [], 'heterogeneous', noise)` |

## Completed stages

| Stage | Description | Status |
|-------|-------------|--------|
| W-1 | Drop WKModel/MLPKModel, use base KModel | ✓ done |
| W-2 | Inherit KrigingImpl; route populate_Model / covMat / make_Cov | ✓ done |
| W-3 | Route update_no_refit_impl; extend m_X_raw / m_noise / re-warp hook | ✓ done |
| B | dump_common / load_common for Kriging / Nugget / Noise | ✓ done |
| D | Hoist m_noise to KrigingImpl | ✓ done |
| F | summary_top / summary_bottom helpers | ✓ done |
| A | MLPKriging as thin facade over WarpKriging(mlp_joint) | ✓ done |
| A2 | MLPKriging save/load via WarpKriging::dump_to_json / load_from_json | ✓ done |
| G | WarpKriging summary/dump/load via base helpers; v3 schema; m_maxdX/m_optim/m_objective set in fit | ✓ done |
| C2 | WarpKriging::predict via predict_impl; FeatureJacobian type; chain-rule deriv to input space | ✓ done |
| W-4 | WarpKriging::simulate / update_simulate via simulate_impl / update_simulate_impl + phi_fn | ✓ done (in G commit) |
| N | Remove NuggetKriging.cpp / NoiseKriging.cpp; unify into Kriging with NoiseModel enum; update all bindings (R/Python/Julia/Octave); re-render all notebooks | ✓ done |

## Deferred / open opportunities

- **E — `with_temp_theta` helper:** only 2 callsites (Warp + MLP), and the `refresh_cache`
  calls inside are class-specific. Abstraction tax exceeds duplication tax.
- **Python cleanup:** `NoiseKriging_binding.cpp` / `NuggetKriging_binding.cpp` and their headers are
  dead code (not compiled). Safe to delete once backward-compat shims are no longer needed.

## Implementation notes for C2

- `FeatureJacobian = std::function<arma::mat(const arma::vec&)>` — receives normalized input
  column (d_input,), returns Jacobian (d_phi × d_input).
- `predict_impl` is now `const`; gains `phi` and `jac` parameters (default `{}`).
- Critical: `Xn_n_input` (d_input × n_n) saved **before** `Xn_n` is overwritten to phi-space
  (`Xn_n = trans(phi(Xn_n))`). The jac lambda receives `Xn_n_input.col(i)` — normalized input,
  not phi-space — so FD perturbations are in the right space.
- Chain rule: `DR_on_i = DR_on_i_phi * J_i` (n_o × d_input), `DF_n_i = J_i.t() * DF_n_i_phi`
  (d_input × p). Output matrices `Dyhat_n` / `Dysd2_n` allocated at d_input width.

## Implementation notes for N (noise unification)

- `Kriging::NoiseModel` enum: `None` | `Nugget` | `Heterogeneous`. Stored as `m_noise_model`.
- `Nugget` mode: `m_nugget` (double) + `m_est_nugget` (bool). Diagonal sentinel = `1 + alpha` (where `alpha = nugget/sigma2`). Joint optim over `(theta, alpha)`.
- `Heterogeneous` mode: `m_noise` (vec, length n). Diagonal sentinel = `1 + noise_i/sigma2`. Optim over `(theta, sigma2)`.
- Backward-compat: R and Julia keep `NoiseKriging`/`NuggetKriging` as user-visible class names; Python exposes only `Kriging` with `noise_model` string.
- T/M/z normalization difference vs old classes: new unified Kriging stores `T = Chol(R + diag(noise/σ²))` (correlation units); old `NoiseKriging` stored `Chol(σ²R + diag(noise))` (covariance units). Differs by `sqrt(σ²)`. All user-facing values (theta, sigma2, beta, predictions, loglik) match exactly.
- compat-r CI: compares new Kriging API output vs v0.9.3 `NoiseKriging`/`NuggetKriging` reference; T/M/z excluded from comparison; macOS logMargPost excluded (bug in old arm64 binary).

## Status

Branch is at its natural stopping point. All tests pass. Stages W-1 through W-4, A/A2, B, C2, D, F, G, and N are complete. Only E remains deferred.
