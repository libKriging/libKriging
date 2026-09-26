// UNVERIFIED Apple-Metal (MSL) port of src/lib/cuda/CudaLinearAlgebraKernel.cu.
// FLOAT32 -- Metal Shading Language has no double on Apple-Silicon GPUs. See
// src/lib/metal/MetalLinearAlgebra.hpp for the load-bearing caveat.
//
// This file is the reference copy of the shader source; the runtime actually
// compiles the identical string embedded in MetalLinearAlgebraKernel.cpp
// (via MTL::Device::newLibrary(source:)), so no .metallib build step is
// needed. covKind: 0=gauss 1=exp 2=matern3_2 3=matern5_2. All matrices are
// column-major (column c at ptr + c*n), same layout as the CUDA kernels.
//
// Every O(n) reduction here (the R*v matvec, the batched dot products, the
// Nystrom U^T Z GEMM and combine) accumulates through the Neumaier compensated
// sum below instead of a bare float. A naive float32 sum of n terms carries
// O(n)*eps rounding error -- ~3-4 lost significant digits at n~1e3 -- which is
// what made the float32 CG/SLQ path visibly disagree (~1e-3) with the double
// CPU path. Compensated summation keeps that error ~eps regardless of n for
// ~4 extra flops per term, which is what lets MetalLinearAlgebra.cpp's CG
// floor its tolerance near 1e-6 rather than 1e-4.
#include <metal_stdlib>
using namespace metal;

// Neumaier compensated summation, float32. `comp` carries the running
// round-off lost by `sum`; add it back once at the end via kacc_final.
struct KAcc { float sum; float comp; };
inline void kacc_init(thread KAcc& a, float v) { a.sum = v; a.comp = 0.0f; }
inline void kacc_add(thread KAcc& a, float v) {
  float t = a.sum + v;
  a.comp += (fabs(a.sum) >= fabs(v)) ? ((a.sum - t) + v) : ((v - t) + a.sum);
  a.sum = t;
}
inline float kacc_final(thread KAcc& a) { return a.sum + a.comp; }

inline float lk_cov_pair(int kind, device const float* Xi, device const float* Xj, device const float* theta, int dimX) {
  float c, sum = 0.0f, sum_sq = 0.0f;
  if (kind == 0) {
    for (int k = 0; k < dimX; ++k) { float v = (Xi[k] - Xj[k]) / theta[k]; sum_sq += v * v; }
    c = exp(-0.5f * sum_sq);
  } else if (kind == 1) {
    for (int k = 0; k < dimX; ++k) sum += fabs((Xi[k] - Xj[k]) / theta[k]);
    c = exp(-sum);
  } else if (kind == 2) {
    for (int k = 0; k < dimX; ++k) { float d = 1.7320508f * fabs((Xi[k] - Xj[k]) / theta[k]); sum += d - log(1.0f + d); }
    c = exp(-sum);
  } else {
    for (int k = 0; k < dimX; ++k) { float d = 2.2360680f * fabs((Xi[k] - Xj[k]) / theta[k]); sum += d - log(1.0f + d + (d * d) / 3.0f); }
    c = exp(-sum);
  }
  return c;
}

inline void lk_dlncov_pair(int kind, device const float* Xi, device const float* Xj, device const float* theta, int dimX, thread float* out) {
  if (kind == 0) {
    for (int k = 0; k < dimX; ++k) { float dx = Xi[k] - Xj[k]; out[k] = (dx * dx) / (theta[k] * theta[k] * theta[k]); }
  } else if (kind == 1) {
    for (int k = 0; k < dimX; ++k) out[k] = fabs(Xi[k] - Xj[k]) / (theta[k] * theta[k]);
  } else if (kind == 2) {
    for (int k = 0; k < dimX; ++k) { float d = 1.7320508f * fabs((Xi[k] - Xj[k]) / theta[k]); out[k] = (d * d) / (1.0f + d) / theta[k]; }
  } else {
    for (int k = 0; k < dimX; ++k) { float d = 2.2360680f * fabs((Xi[k] - Xj[k]) / theta[k]); float a = 1.0f + d, b = (d * d) / 3.0f; out[k] = (a * b) / (a + b) / theta[k]; }
  }
}

struct RmulP { int n; int dimX; int covKind; int ncols; };

kernel void rmul_batched(device const float* Xt [[buffer(0)]], device const float* theta [[buffer(1)]],
                         device const float* P [[buffer(2)]], device float* Ap [[buffer(3)]],
                         constant RmulP& p [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
  const int i = int(gid.x), c = int(gid.y);
  if (i >= p.n || c >= p.ncols) return;
  device const float* Pc = P + (size_t)c * p.n;
  device const float* Xi = Xt + (size_t)i * p.dimX;
  KAcc acc; kacc_init(acc, Pc[i]);
  for (int j = 0; j < p.n; ++j) { if (j == i) continue; kacc_add(acc, lk_cov_pair(p.covKind, Xi, Xt + (size_t)j * p.dimX, theta, p.dimX) * Pc[j]); }
  Ap[(size_t)c * p.n + i] = kacc_final(acc);
}

// --- Dense-R fast path -----------------------------------------------
// Mechanical port of CudaLinearAlgebraKernel.cu's build_cov_kernel /
// HipLinearAlgebraKernel.hip.cpp's dense_matvec_kernel: for a whole
// objective evaluation (one CG solve on [F|y|probes], dozens of Lanczos
// steps, all at the SAME theta), materialize R ONCE instead of
// re-evaluating every covariance pair's transcendentals (rmul_batched
// above) on every single iteration/step. See MetalLinearAlgebra.cpp's
// DenseCovCache for the host-side cache and budget gate.
struct BuildCovP { int n; int dimX; int covKind; };

// grid (n,n), i<=j only (R is symmetric) -- half the covariance
// evaluations of a naive full n x n fill.
kernel void build_cov(device const float* Xt [[buffer(0)]], device const float* theta [[buffer(1)]],
                      device float* R [[buffer(2)]], constant BuildCovP& p [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]]) {
  const int i = int(gid.x), j = int(gid.y);
  if (i >= p.n || j >= p.n || i > j) return;
  device const float* Xi = Xt + (size_t)i * p.dimX;
  device const float* Xj = Xt + (size_t)j * p.dimX;
  const float c = lk_cov_pair(p.covKind, Xi, Xj, theta, p.dimX);
  const size_t idx_ij = (size_t)i + (size_t)j * p.n;
  const size_t idx_ji = (size_t)j + (size_t)i * p.n;
  R[idx_ij] = c;
  R[idx_ji] = c;
}

struct DenseMvP { int n; int ncols; int ldc; };

// Same grid/block shape as rmul_batched, but reading a cached matrix
// instead of evaluating covariance transcendentals per pair -- pure FMA,
// no exp/log/pow, which is the entire point of the dense fast path.
// Deliberately a plain sequential sum over ALL j (diagonal included in
// its natural position, not special-cased like rmul_batched's `acc =
// Pc[i]` shortcut): this same kernel also serves dRmulBatched's future
// dense path, where R would be a dR/dtheta_k block whose diagonal is
// exactly 0, not 1 -- see HipLinearAlgebraKernel.hip.cpp's matching
// comment for why hard-coding "diag==1" there is a correctness bug
// waiting to happen.
kernel void dense_matvec(device const float* R [[buffer(0)]], device const float* V [[buffer(1)]],
                         device float* Out [[buffer(2)]], constant DenseMvP& p [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
  const int i = int(gid.x), c = int(gid.y);
  if (i >= p.n || c >= p.ncols) return;
  device const float* Vc = V + (size_t)c * p.n;
  KAcc acc; kacc_init(acc, 0.0f);
  for (int j = 0; j < p.n; ++j) kacc_add(acc, R[(size_t)j * p.n + i] * Vc[j]);
  Out[(size_t)c * p.ldc + i] = kacc_final(acc);
}

kernel void drmul_batched(device const float* Xt [[buffer(0)]], device const float* theta [[buffer(1)]],
                          device const float* V [[buffer(2)]], device float* Out [[buffer(3)]],
                          constant RmulP& p [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
  const int i = int(gid.x), c = int(gid.y);
  if (i >= p.n || c >= p.ncols) return;
  KAcc acc[32]; float dln[32];
  for (int k = 0; k < p.dimX; ++k) kacc_init(acc[k], 0.0f);
  device const float* Vc = V + (size_t)c * p.n;
  device const float* Xi = Xt + (size_t)i * p.dimX;
  for (int j = 0; j < p.n; ++j) {
    if (j == i) continue;
    device const float* Xj = Xt + (size_t)j * p.dimX;
    float cij = lk_cov_pair(p.covKind, Xi, Xj, theta, p.dimX);
    lk_dlncov_pair(p.covKind, Xi, Xj, theta, p.dimX, dln);
    float vj = Vc[j];
    for (int k = 0; k < p.dimX; ++k) kacc_add(acc[k], cij * dln[k] * vj);
  }
  device float* Oc = Out + (size_t)c * p.dimX * p.n;
  for (int k = 0; k < p.dimX; ++k) Oc[(size_t)k * p.n + i] = kacc_final(acc[k]);
}

struct NC { int n; int ncols; };

kernel void batched_dot(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
                        device float* out [[buffer(2)]], constant NC& p [[buffer(3)]],
                        uint c [[thread_position_in_grid]]) {
  if (int(c) >= p.ncols) return;
  device const float* Ac = A + (size_t)c * p.n;
  device const float* Bc = B + (size_t)c * p.n;
  KAcc s; kacc_init(s, 0.0f);
  for (int i = 0; i < p.n; ++i) kacc_add(s, Ac[i] * Bc[i]);
  out[c] = kacc_final(s);
}

kernel void batched_axpy(device const float* alpha [[buffer(0)]], device const float* X [[buffer(1)]],
                         device float* Y [[buffer(2)]], constant NC& p [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
  if (int(gid.x) >= p.n || int(gid.y) >= p.ncols) return;
  size_t idx = (size_t)gid.y * p.n + gid.x;
  Y[idx] += alpha[gid.y] * X[idx];
}

kernel void batched_update_p(device const float* R [[buffer(0)]], device const float* beta [[buffer(1)]],
                             device float* P [[buffer(2)]], constant NC& p [[buffer(3)]],
                             uint2 gid [[thread_position_in_grid]]) {
  if (int(gid.x) >= p.n || int(gid.y) >= p.ncols) return;
  size_t idx = (size_t)gid.y * p.n + gid.x;
  P[idx] = R[idx] + beta[gid.y] * P[idx];
}

// tol is PER-COLUMN (buffer, length ncols) rather than a scalar in CgP:
// each column freezes independently once it reaches ITS OWN tol[c],
// letting a caller fuse right-hand-side groups that need different
// tolerances (e.g. Kriging.cpp's mBCG fusion of [F|y|probes] -- F/y want
// the tighter cg_tol, probes want the looser probes_cg_tol) into ONE
// Krylov pass instead of a separate call per group -- mirrors
// LinearAlgebraCuda::conjugateGradient's per-column tol vector.
struct CgP { int ncols; };

kernel void cg_alpha(device const float* rz_old [[buffer(0)]], device const float* pAp [[buffer(1)]],
                     device atomic_int* active [[buffer(2)]], device float* alpha [[buffer(3)]],
                     device float* neg_alpha [[buffer(4)]], constant CgP& p [[buffer(5)]],
                     uint c [[thread_position_in_grid]]) {
  if (int(c) >= p.ncols) return;
  if (atomic_load_explicit(&active[c], memory_order_relaxed) == 0) { alpha[c] = 0.0f; neg_alpha[c] = 0.0f; return; }
  float q = pAp[c];
  if (!(q > 0.0f)) { atomic_store_explicit(&active[c], 0, memory_order_relaxed); alpha[c] = 0.0f; neg_alpha[c] = 0.0f; return; }
  float a = rz_old[c] / q;
  alpha[c] = a; neg_alpha[c] = -a;
}

kernel void cg_beta(device const float* rr_new [[buffer(0)]], device const float* bnorm [[buffer(1)]],
                    device const float* tol [[buffer(2)]],
                    device atomic_int* active [[buffer(3)]], device float* rz_old [[buffer(4)]],
                    device float* beta [[buffer(5)]], constant CgP& p [[buffer(6)]],
                    uint c [[thread_position_in_grid]]) {
  if (int(c) >= p.ncols) return;
  if (atomic_load_explicit(&active[c], memory_order_relaxed) == 0) { beta[c] = 0.0f; return; }
  float rn = rr_new[c];
  if (sqrt(rn) / bnorm[c] < tol[c]) { atomic_store_explicit(&active[c], 0, memory_order_relaxed); beta[c] = 0.0f; return; }
  beta[c] = rn / rz_old[c]; rz_old[c] = rn;
}

kernel void cg_restart(device const float* rr [[buffer(0)]], device const float* bnorm [[buffer(1)]],
                       device const float* tol [[buffer(2)]],
                       device atomic_int* active [[buffer(3)]], device float* rz_old [[buffer(4)]],
                       constant CgP& p [[buffer(5)]], uint c [[thread_position_in_grid]]) {
  if (int(c) >= p.ncols) return;
  if (atomic_load_explicit(&active[c], memory_order_relaxed) == 0) return;
  rz_old[c] = rr[c];
  if (sqrt(rr[c]) / bnorm[c] < tol[c]) atomic_store_explicit(&active[c], 0, memory_order_relaxed);
}

kernel void cg_restart_precond(device const float* rr [[buffer(0)]], device const float* rz [[buffer(1)]],
                               device const float* bnorm [[buffer(2)]], device const float* tol [[buffer(3)]],
                               device atomic_int* active [[buffer(4)]],
                               device float* rz_old [[buffer(5)]], constant CgP& p [[buffer(6)]],
                               uint c [[thread_position_in_grid]]) {
  if (int(c) >= p.ncols) return;
  if (atomic_load_explicit(&active[c], memory_order_relaxed) == 0) return;
  rz_old[c] = rz[c];
  if (sqrt(rr[c]) / bnorm[c] < tol[c]) atomic_store_explicit(&active[c], 0, memory_order_relaxed);
}

kernel void cg_beta_precond(device const float* rr [[buffer(0)]], device const float* rz_new [[buffer(1)]],
                            device const float* bnorm [[buffer(2)]], device const float* tol [[buffer(3)]],
                            device atomic_int* active [[buffer(4)]],
                            device float* rz_old [[buffer(5)]], device float* beta [[buffer(6)]],
                            constant CgP& p [[buffer(7)]], uint c [[thread_position_in_grid]]) {
  if (int(c) >= p.ncols) return;
  if (atomic_load_explicit(&active[c], memory_order_relaxed) == 0) { beta[c] = 0.0f; return; }
  if (sqrt(rr[c]) / bnorm[c] < tol[c]) { atomic_store_explicit(&active[c], 0, memory_order_relaxed); beta[c] = 0.0f; return; }
  beta[c] = rz_new[c] / rz_old[c]; rz_old[c] = rz_new[c];
}

kernel void cg_any_active(device const int* active [[buffer(0)]], device atomic_int* flag [[buffer(1)]],
                          constant int& ncols [[buffer(2)]], uint c [[thread_position_in_grid]]) {
  if (int(c) >= ncols) return;
  if (active[c]) atomic_fetch_or_explicit(flag, 1, memory_order_relaxed);
}

struct PcP { int n; int k; int ncols; };

kernel void precond_scale_rows(device const float* Dinv [[buffer(0)]], device const float* r [[buffer(1)]],
                               device float* out [[buffer(2)]], constant PcP& p [[buffer(3)]],
                               uint2 gid [[thread_position_in_grid]]) {
  if (int(gid.x) >= p.n || int(gid.y) >= p.ncols) return;
  size_t idx = (size_t)gid.y * p.n + gid.x;
  out[idx] = Dinv[gid.x] * r[idx];
}

kernel void precond_gemm_Ut(device const float* U [[buffer(0)]], device const float* Z [[buffer(1)]],
                            device float* t [[buffer(2)]], constant PcP& p [[buffer(3)]],
                            uint2 gid [[thread_position_in_grid]]) {
  const int kk = int(gid.x), c = int(gid.y);
  if (kk >= p.k || c >= p.ncols) return;
  device const float* Uk = U + (size_t)kk * p.n;
  device const float* Zc = Z + (size_t)c * p.n;
  KAcc acc; kacc_init(acc, 0.0f);
  for (int i = 0; i < p.n; ++i) kacc_add(acc, Uk[i] * Zc[i]);
  t[(size_t)c * p.k + kk] = kacc_final(acc);
}

kernel void precond_trisolve(device const float* Mchol [[buffer(0)]], device float* t [[buffer(1)]],
                             constant PcP& p [[buffer(2)]], uint c [[thread_position_in_grid]]) {
  if (int(c) >= p.ncols) return;
  device float* tc = t + (size_t)c * p.k;
  for (int i = 0; i < p.k; ++i) {
    float s = tc[i];
    for (int j = 0; j < i; ++j) s -= Mchol[(size_t)j * p.k + i] * tc[j];
    tc[i] = s / Mchol[(size_t)i * p.k + i];
  }
  for (int i = p.k - 1; i >= 0; --i) {
    float s = tc[i];
    for (int j = i + 1; j < p.k; ++j) s -= Mchol[(size_t)i * p.k + j] * tc[j];
    tc[i] = s / Mchol[(size_t)i * p.k + i];
  }
}

kernel void precond_combine(device const float* U [[buffer(0)]], device const float* Dinv [[buffer(1)]],
                            device const float* r [[buffer(2)]], device const float* s [[buffer(3)]],
                            device float* z [[buffer(4)]], constant PcP& p [[buffer(5)]],
                            uint2 gid [[thread_position_in_grid]]) {
  const int i = int(gid.x), c = int(gid.y);
  if (i >= p.n || c >= p.ncols) return;
  device const float* sc = s + (size_t)c * p.k;
  KAcc acc; kacc_init(acc, 0.0f);
  for (int kk = 0; kk < p.k; ++kk) kacc_add(acc, U[(size_t)kk * p.n + i] * sc[kk]);
  size_t idx = (size_t)c * p.n + i;
  z[idx] = Dinv[i] * (r[idx] - kacc_final(acc));
}

// --- Device-resident batched Lanczos (Stochastic Lanczos Quadrature) ------
// Mechanical port of HipLinearAlgebraKernel.hip.cpp's lanczos_alpha/beta/
// reorth_dot/reorth_sub kernels (which themselves stand in for CUDA's
// cublasDgemmStridedBatched reorthogonalization pair -- no BLAS dependency
// here either). Keeps every probe's entire Krylov history resident on the
// GPU for the whole SLQ recurrence instead of round-tripping through
// rmulBatched once per Lanczos step for the host to do
// reorthogonalization/bookkeeping in Armadillo -- see
// MetalLinearAlgebra.hpp's stochasticLogDetBatched doc comment.

// alpha/beta write straight into their step's slot of the ls*ncols
// alpha_all/beta_all history buffers (offset step_idx*ncols, device-side)
// instead of a small transient buffer a host-side copy_dev would then need
// to place into that slot -- lk_metal_copy_dev/memset_dev have no offset
// parameter (see MetalLinearAlgebraKernel.cpp), and there's no reason to
// add one when the kernel can just compute the right address itself.
struct LzAlphaP { int ncols; int step_idx; };

kernel void lanczos_alpha(device const float* dot [[buffer(0)]], device const int* active [[buffer(1)]],
                          device float* alpha_all [[buffer(2)]], device float* neg_alpha [[buffer(3)]],
                          constant LzAlphaP& p [[buffer(4)]], uint c [[thread_position_in_grid]]) {
  if (int(c) >= p.ncols) return;
  const float a = active[c] ? dot[c] : 0.0f;
  alpha_all[(size_t)p.step_idx * p.ncols + c] = a;
  neg_alpha[c] = -a;
}

struct LzBetaP { int ncols; int step_idx; int is_last; };

// End-of-step Lanczos bookkeeping -- see HipLinearAlgebraKernel.hip.cpp's
// lanczos_beta_kernel doc comment for the exact semantics (once active[c]
// is 0, every output is 0, which is what keeps an inactive probe's Krylov
// vectors exactly zero from m_eff[c] on). The invariant-breakdown floor is
// 1e-6 here, not CUDA/HIP's 1e-12 -- there's no float32 residual signal
// below that (see the compensated-sum comment at the top of this file).
kernel void lanczos_beta(device const float* dot2 [[buffer(0)]], device int* active [[buffer(1)]],
                         device int* m_eff [[buffer(2)]], device float* beta_all [[buffer(3)]],
                         device float* inv_bj_out [[buffer(4)]], device float* neg_beta_prev_out [[buffer(5)]],
                         constant LzBetaP& p [[buffer(6)]], uint c [[thread_position_in_grid]]) {
  if (int(c) >= p.ncols) return;
  if (!active[c]) {
    beta_all[(size_t)p.step_idx * p.ncols + c] = 0.0f;
    inv_bj_out[c] = 0.0f;
    neg_beta_prev_out[c] = 0.0f;
    return;
  }
  const float bj = sqrt(dot2[c]);
  const bool invariant = bj < 1e-6f;
  if (invariant) {
    active[c] = 0;
    m_eff[c] = p.step_idx + 1;
  }
  const bool no_next = (p.is_last != 0) || invariant;
  beta_all[(size_t)p.step_idx * p.ncols + c] = no_next ? 0.0f : bj;
  inv_bj_out[c] = no_next ? 0.0f : 1.0f / bj;
  neg_beta_prev_out[c] = no_next ? 0.0f : -bj;
}

struct LzReP { int n; int npr; int ls; int m; };

// grid = (m, npr): one thread per (history step jj, probe p) pair, dotting
// step jj's Krylov vector for probe p (V's step-major layout: step k's
// n x npr block lives at V + k*npr*n) against W's probe-p column, writing
// probe p's length-ls history slot at t[p*ls + jj]. Stands in for CUDA's
// cublasDgemmStridedBatched(CUBLAS_OP_T, ...) call -- covering every jj in
// [0,m) in ONE launch keeps this at one dispatch per Lanczos step.
kernel void lanczos_reorth_dot(device const float* V [[buffer(0)]], device const float* W [[buffer(1)]],
                               device float* t [[buffer(2)]], constant LzReP& p [[buffer(3)]],
                               uint2 gid [[thread_position_in_grid]]) {
  const int jj = int(gid.x), pr = int(gid.y);
  if (jj >= p.m || pr >= p.npr) return;
  device const float* Vp = V + ((size_t)jj * p.npr + pr) * p.n;
  device const float* Wp = W + (size_t)pr * p.n;
  KAcc s; kacc_init(s, 0.0f);
  for (int i = 0; i < p.n; ++i) kacc_add(s, Vp[i] * Wp[i]);
  t[(size_t)pr * p.ls + jj] = kacc_final(s);
}

// One thread per (row i, probe p): W[i,p] -= sum_{k=0}^{m-1} V[step k][i,p]
// * t[p*ls+k] -- probe p's whole Krylov history read straight out of the
// caller's step-major V buffer, same layout lanczos_reorth_dot/the matvec
// read it in. Stands in for CUDA's second cublasDgemmStridedBatched call.
kernel void lanczos_reorth_sub(device const float* V [[buffer(0)]], device const float* t [[buffer(1)]],
                               device float* W [[buffer(2)]], constant LzReP& p [[buffer(3)]],
                               uint2 gid [[thread_position_in_grid]]) {
  const int i = int(gid.x), pr = int(gid.y);
  if (i >= p.n || pr >= p.npr) return;
  device const float* tp = t + (size_t)pr * p.ls;
  KAcc acc; kacc_init(acc, 0.0f);
  for (int k = 0; k < p.m; ++k) kacc_add(acc, V[((size_t)k * p.npr + pr) * p.n + i] * tp[k]);
  W[(size_t)pr * p.n + i] -= kacc_final(acc);
}
