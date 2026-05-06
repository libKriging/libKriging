<!-- cspell:disable -->

# Class topology and shape

_Last updated 2026-05-06 — post W-1/W-2/W-3 + A/A2/B/D/F + G + C2 + **N (noise unification)**._

```text
┌─────────────────────┬──────────────────────────────────────┬────────────────────────┬────────────────────┬────────────────────────────────────────────────┬───────────────────────────────────────────────────────┐
│                     │               Kriging                │     NuggetKriging      │    NoiseKriging    │                  WarpKriging                   │                    MLPKriging                         │
├─────────────────────┼──────────────────────────────────────┼────────────────────────┼────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ File LOC            │ 1837  (was 1897)                     │ 1264  (was 1470)       │ 960  (was 1116)    │ 2661  (was 2940)                               │ 177  (was 1361)                                       │
│                     │                                      │                        │                    │ KrigingImpl.cpp: 879 (was 647 — +232 helpers)  │                                                       │
├─────────────────────┼──────────────────────────────────────┼────────────────────────┼────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Inherits KrigingImpl│ protected (pre-existing)             │ protected              │ protected          │ protected (W-2)                                │ does NOT inherit — holds WarpKriging m_impl (A)       │
├─────────────────────┼──────────────────────────────────────┼────────────────────────┼────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Kernel input slot   │ inherited m_X (= normalized X)       │ inherited m_X          │ inherited m_X      │ inherited m_X (= Φ(X_norm), Option A)          │ via m_impl.m_X (Φ-space)                              │
│                     │                                      │                        │                    │ raw kept in m_X_raw                            │ raw kept in m_impl.m_X_raw                            │
├─────────────────────┼──────────────────────────────────────┼────────────────────────┼────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Distance cache      │ inherited m_dX / m_maxdX             │ inherited              │ inherited          │ inherited m_dX / m_maxdX (set in              │ via m_impl                                            │
│                     │                                      │                        │                    │ refresh_cache — G)                             │                                                       │
├─────────────────────┼──────────────────────────────────────┼────────────────────────┼────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ m_noise location    │ inherited (always empty)             │ inherited (always      │ inherited (always  │ inherited (optional, set when fit with noise)  │ via m_impl (always empty — MLP has no noise path)     │
│                     │                                      │ empty)                 │ set, length n)     │ — hoisted in D                                 │                                                       │
├─────────────────────┼──────────────────────────────────────┼────────────────────────┼────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ KModel struct       │ shared KrigingImpl::KModel           │ shared                 │ shared             │ shared KrigingImpl::KModel (W-1)               │ via m_impl (no own KModel)                            │
├─────────────────────┼──────────────────────────────────────┼────────────────────────┼────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ populate_Model      │ KrigingImpl::populate_Model          │ KrigingImpl::pop…      │ KrigingImpl::pop…  │ KrigingImpl::populate_Model (W-2)              │ via m_impl                                            │
├─────────────────────┼──────────────────────────────────────┼────────────────────────┼────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ make_Cov            │ KrigingImpl::make_Cov                │ KrigingImpl::make_Cov  │ KrigingImpl::…     │ KrigingImpl::make_Cov via name normalizer (W-2)│ via m_impl                                            │
├─────────────────────┼──────────────────────────────────────┼────────────────────────┼────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ covMat() accessor   │ inherited                            │ inherited              │ inherited          │ inherited                                      │ via m_impl                                            │
├─────────────────────┼──────────────────────────────────────┼────────────────────────┼────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Trend matrix        │ regressionModelMatrix(m_regmodel,    │ same                   │ same               │ same (Φ-space; build_trend_matrix wraps)       │ via m_impl                                            │
│                     │ m_X)                                 │                        │                    │                                                │                                                       │
├─────────────────────┼──────────────────────────────────────┼────────────────────────┼────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Normalization       │ rowvec on all dims (base helper)     │ rowvec, base           │ rowvec, base       │ per-dim continuous-only (centerX(j)=0,         │ via m_impl                                            │
│                     │                                      │                        │                    │ scaleX(j)=1 for non-continuous → reduces to    │                                                       │
│                     │                                      │                        │                    │ rowvec equivalence)                            │                                                       │
├─────────────────────┼──────────────────────────────────────┼────────────────────────┼────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Diag sentinel       │ ones (=1)                            │ ones(n) (=1+τ)         │ 1+noise/σ²         │ 1 + num_nugget [+ noise/σ²]                    │ 1 + 1e-8 (via m_impl with mlp_joint warp spec)        │
├─────────────────────┼──────────────────────────────────────┼────────────────────────┼────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Closed-form σ̂²      │ yes                                  │ no (joint optim of     │ no (per-point      │ yes when m_noise empty, else 1-D               │ yes (via m_impl, noise always empty)                  │
│                     │                                      │ (θ,α))                 │ noise)             │ golden-section over σ²                         │                                                       │
├─────────────────────┼──────────────────────────────────────┼────────────────────────┼────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Optim variables     │ θ (+β when fixed)                    │ (θ, α)                 │ (θ, σ²)            │ (θ, warp_params, [σ² when noise])              │ (θ, warp_params) via m_impl                           │
├─────────────────────┼──────────────────────────────────────┼────────────────────────┼────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Optim algorithm     │ L-BFGS-B multistart                  │ L-BFGS-B multistart    │ L-BFGS-B           │ bi-level Adam(warp) ⊕ L-BFGS(θ)                │ via m_impl (bi-level Adam ⊕ L-BFGS with mlp_joint)    │
│                     │                                      │                        │ multistart         │ (stays in Warp); m_optim/m_objective now set   │                                                       │
├─────────────────────┼──────────────────────────────────────┼────────────────────────┼────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ update_no_refit     │ KrigingImpl::update_no_refit_impl    │ same                   │ same               │ KrigingImpl::update_no_refit_impl (W-3)        │ via m_impl (A)                                        │
│                     │                                      │                        │                    │ +hook: extend m_X_raw / m_noise / re-warp m_X  │                                                       │
├─────────────────────┼──────────────────────────────────────┼────────────────────────┼────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ save / load         │ dump_common / load_common (B)        │ dump/load_common (B)   │ dump/load_common   │ dump_to_json calls dump_common_to_json + Warp  │ dump_to_json / load_from_json via m_impl (A2+G)       │
│                     │                                      │ + nugget/est_nugget    │ (B) — m_noise auto │ extras; load_from_json branches v2/v3 (G)      │ v3 schema; v2 legacy files still loadable             │
│                     │                                      │ extras                 │ via dump_common    │ v3 schema ("covType"/"X_raw"); v2 still loads  │                                                       │
├─────────────────────┼──────────────────────────────────────┼────────────────────────┼────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ summary()           │ summary_top + summary_bottom (F)     │ summary_top + nugget   │ summary_top +      │ summary_top(&m_X_raw) + warpings block +       │ Delegates to m_impl.summary() + header rename (A)     │
│                     │                                      │ line + summary_bottom  │ summary_bottom     │ summary_bottom (G); X_display_override param   │                                                       │
│                     │                                      │ (F)                    │ (F) — m_noise auto │ shows input-space ranges not Φ-space           │                                                       │
├─────────────────────┼──────────────────────────────────────┼────────────────────────┼────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ fit_setup_impl      │ KrigingImpl::fit_setup_impl          │ same                   │ same               │ NOT routed — θ₀ normalized by scaleX in base   │ via m_impl (A)                                        │
│                     │                                      │                        │                    │ helper, but Warp's θ lives in feature_dim ≠    │                                                       │
│                     │                                      │                        │                    │ input-d, so dimensions mismatch                │                                                       │
├─────────────────────┼──────────────────────────────────────┼────────────────────────┼────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ predict_impl        │ KrigingImpl::predict_impl            │ same                   │ same               │ KrigingImpl::predict_impl (C2) with phi_fn +   │ via m_impl (A)                                        │
│                     │                                      │                        │                    │ jac_fn lambdas; Xn_n_input saved before phi    │                                                       │
│                     │                                      │                        │                    │ overwrites Xn_n; deriv chain-ruled to d_input  │                                                       │
├─────────────────────┼──────────────────────────────────────┼────────────────────────┼────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ simulate_impl       │ KrigingImpl::simulate_impl           │ same                   │ same               │ KrigingImpl::simulate_impl (W-4) via phi_fn;   │ via m_impl (A)                                        │
│ update_simulate_impl│ KrigingImpl::update_simulate_impl    │ same                   │ same               │ wrapper keeps with_noise/lastsim_with_noise +  │                                                       │
│                     │                                      │                        │                    │ noise_ok cache check (Warp-only extras)        │                                                       │
├─────────────────────┼──────────────────────────────────────┼────────────────────────┼────────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ logLikelihoodFun θ- │ free helper (no member θ swap)       │ free helper            │ free helper        │ inline save/restore m_theta around             │ via m_impl (A)                                        │
│ swap pattern        │                                      │                        │                    │ refresh_cache (E deferred — only 2 callsites,  │                                                       │
│                     │                                      │                        │                    │ refresh_cache call is class-specific anyway)   │                                                       │
└─────────────────────┴──────────────────────────────────────┴────────────────────────┴────────────────────┴────────────────────────────────────────────────┴───────────────────────────────────────────────────────┘
```

## Net LOC delta on this branch

```text
                    pre-W1   post-W1/2/3   post-B/D/F   post-A    post-G    post-C2   post-N    Δ vs pre-W1
Kriging.cpp         1897     1897          1794         1794       1837       1837      1854      −43    (B + F + N: NoiseModel enum + noise/nugget fit paths)
NuggetKriging.cpp   1470     1470          1368         1368       1264       1264         0     −1470   (N: removed — merged into Kriging)
NoiseKriging.cpp    1116     1116          1010         1010        960        960         0     −1116   (N: removed — merged into Kriging)
WarpKriging.cpp     2940     2918          2918         2802       2765       2661      2768      −172   (W-2/W-3 + A + G + C2; +107 post-C2 for noise path)
MLPKriging.cpp      1361     1329          1329          262        177        177       178     −1183   (A: thin facade + G)
KrigingImpl.cpp     —/647    647           768           784        852        879       877      +230   (helpers added)
                                                                                                 ──────
                                                                                                 −3754 net  (vs pre-W1 state)
```

## Binding changes (stage N)

| Binding | NoiseKriging/NuggetKriging classes | New unified API |
|---------|-----------------------------------|-----------------|
| **R** | `NoiseKrigingClass.R` / `NuggetKrigingClass.R` kept as backward-compat S3 wrappers; internally call `Kriging(noise_model=...)` via `new_NoiseKriging` / `new_NuggetKriging` → `new Kriging(kernel, NoiseModel::Heterogeneous/Nugget)` | `Kriging(noise=vector)` / `Kriging(noise="nugget")` |
| **Python** | Old `PyNoiseKriging` / `PyNuggetKriging` files retained but NOT compiled; `pylibkriging.cpp` exposes `PyKriging` only with `noise_model` string param | `Kriging(noise=array)` / `Kriging(noise="nugget")` |
| **Julia** | `NuggetKriging(...)` / `NoiseKriging(...)` kept as deprecated wrappers emitting `Base.depwarn`; delegate to `Kriging(...; noise="nugget")` / `Kriging(...; noise=vec)` | `Kriging(y, X, kernel; noise=vec\|"nugget")` |
| **Octave** | `NuggetKriging::*` / `NoiseKriging::*` dispatcher cases redirect to unified `Kriging_binding`; MxMapper segfault fixed | `Kriging(y, X, ..., [], 'heterogeneous', noise)` |

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
- K / Nug / Noise: call sites unchanged (phi={}, jac={}); no overhead, no behavioral change.

## Implementation notes for N (noise unification)

- `Kriging::NoiseModel` enum: `None` | `Nugget` | `Heterogeneous`. Stored as `m_noise_model`.
- `Nugget` mode: `m_nugget` (double) + `m_est_nugget` (bool). Diagonal sentinel = `1 + alpha` (where `alpha = nugget/sigma2`). Joint optim over `(theta, alpha)`.
- `Heterogeneous` mode: `m_noise` (vec, length n). Diagonal sentinel = `1 + noise_i/sigma2`. Optim over `(theta, sigma2)`.
- Backward-compat: R and Julia keep `NoiseKriging`/`NuggetKriging` as user-visible class names; Python exposes only `Kriging` with `noise_model` string.
- T/M/z normalization difference vs old classes: new unified Kriging stores `T = Chol(R + diag(noise/σ²))` (correlation units); old `NoiseKriging` stored `Chol(σ²R + diag(noise))` (covariance units). Differs by `sqrt(σ²)`. All user-facing values (theta, sigma2, beta, predictions, loglik) match exactly.
- compat-r CI: compares new Kriging API output vs v0.9.3 `NoiseKriging`/`NuggetKriging` reference; T/M/z excluded from comparison; macOS logMargPost excluded (bug in old arm64 binary).

## Status

Branch is at its natural stopping point. All tests pass. Stages W-1 through W-4, A/A2, B, C2, D, F, G, and N are complete. Only E remains deferred.
