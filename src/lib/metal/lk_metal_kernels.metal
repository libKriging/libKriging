// UNVERIFIED Apple-Metal (MSL) port of src/lib/cuda/CudaLinearAlgebraKernel.cu.
// FLOAT32 -- Metal Shading Language has no double on Apple-Silicon GPUs. See
// src/lib/metal/MetalLinearAlgebra.hpp for the load-bearing caveat.
//
// This file is the reference copy of the shader source; the runtime actually
// compiles the identical string embedded in MetalLinearAlgebraKernel.cpp
// (via MTL::Device::newLibrary(source:)), so no .metallib build step is
// needed. covKind: 0=gauss 1=exp 2=matern3_2 3=matern5_2. All matrices are
// column-major (column c at ptr + c*n), same layout as the CUDA kernels.
#include <metal_stdlib>
using namespace metal;

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
  float acc = Pc[i];
  for (int j = 0; j < p.n; ++j) { if (j == i) continue; acc += lk_cov_pair(p.covKind, Xi, Xt + (size_t)j * p.dimX, theta, p.dimX) * Pc[j]; }
  Ap[(size_t)c * p.n + i] = acc;
}

kernel void drmul_batched(device const float* Xt [[buffer(0)]], device const float* theta [[buffer(1)]],
                          device const float* V [[buffer(2)]], device float* Out [[buffer(3)]],
                          constant RmulP& p [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
  const int i = int(gid.x), c = int(gid.y);
  if (i >= p.n || c >= p.ncols) return;
  float acc[32]; float dln[32];
  for (int k = 0; k < p.dimX; ++k) acc[k] = 0.0f;
  device const float* Vc = V + (size_t)c * p.n;
  device const float* Xi = Xt + (size_t)i * p.dimX;
  for (int j = 0; j < p.n; ++j) {
    if (j == i) continue;
    device const float* Xj = Xt + (size_t)j * p.dimX;
    float cij = lk_cov_pair(p.covKind, Xi, Xj, theta, p.dimX);
    lk_dlncov_pair(p.covKind, Xi, Xj, theta, p.dimX, dln);
    float vj = Vc[j];
    for (int k = 0; k < p.dimX; ++k) acc[k] += cij * dln[k] * vj;
  }
  device float* Oc = Out + (size_t)c * p.dimX * p.n;
  for (int k = 0; k < p.dimX; ++k) Oc[(size_t)k * p.n + i] = acc[k];
}

struct NC { int n; int ncols; };

kernel void batched_dot(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
                        device float* out [[buffer(2)]], constant NC& p [[buffer(3)]],
                        uint c [[thread_position_in_grid]]) {
  if (int(c) >= p.ncols) return;
  device const float* Ac = A + (size_t)c * p.n;
  device const float* Bc = B + (size_t)c * p.n;
  float s = 0.0f;
  for (int i = 0; i < p.n; ++i) s += Ac[i] * Bc[i];
  out[c] = s;
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

struct CgP { float tol; int ncols; };

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
                    device atomic_int* active [[buffer(2)]], device float* rz_old [[buffer(3)]],
                    device float* beta [[buffer(4)]], constant CgP& p [[buffer(5)]],
                    uint c [[thread_position_in_grid]]) {
  if (int(c) >= p.ncols) return;
  if (atomic_load_explicit(&active[c], memory_order_relaxed) == 0) { beta[c] = 0.0f; return; }
  float rn = rr_new[c];
  if (sqrt(rn) / bnorm[c] < p.tol) { atomic_store_explicit(&active[c], 0, memory_order_relaxed); beta[c] = 0.0f; return; }
  beta[c] = rn / rz_old[c]; rz_old[c] = rn;
}

kernel void cg_restart(device const float* rr [[buffer(0)]], device const float* bnorm [[buffer(1)]],
                       device atomic_int* active [[buffer(2)]], device float* rz_old [[buffer(3)]],
                       constant CgP& p [[buffer(4)]], uint c [[thread_position_in_grid]]) {
  if (int(c) >= p.ncols) return;
  if (atomic_load_explicit(&active[c], memory_order_relaxed) == 0) return;
  rz_old[c] = rr[c];
  if (sqrt(rr[c]) / bnorm[c] < p.tol) atomic_store_explicit(&active[c], 0, memory_order_relaxed);
}

kernel void cg_restart_precond(device const float* rr [[buffer(0)]], device const float* rz [[buffer(1)]],
                               device const float* bnorm [[buffer(2)]], device atomic_int* active [[buffer(3)]],
                               device float* rz_old [[buffer(4)]], constant CgP& p [[buffer(5)]],
                               uint c [[thread_position_in_grid]]) {
  if (int(c) >= p.ncols) return;
  if (atomic_load_explicit(&active[c], memory_order_relaxed) == 0) return;
  rz_old[c] = rz[c];
  if (sqrt(rr[c]) / bnorm[c] < p.tol) atomic_store_explicit(&active[c], 0, memory_order_relaxed);
}

kernel void cg_beta_precond(device const float* rr [[buffer(0)]], device const float* rz_new [[buffer(1)]],
                            device const float* bnorm [[buffer(2)]], device atomic_int* active [[buffer(3)]],
                            device float* rz_old [[buffer(4)]], device float* beta [[buffer(5)]],
                            constant CgP& p [[buffer(6)]], uint c [[thread_position_in_grid]]) {
  if (int(c) >= p.ncols) return;
  if (atomic_load_explicit(&active[c], memory_order_relaxed) == 0) { beta[c] = 0.0f; return; }
  if (sqrt(rr[c]) / bnorm[c] < p.tol) { atomic_store_explicit(&active[c], 0, memory_order_relaxed); beta[c] = 0.0f; return; }
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
  float acc = 0.0f;
  for (int i = 0; i < p.n; ++i) acc += Uk[i] * Zc[i];
  t[(size_t)c * p.k + kk] = acc;
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
  float acc = 0.0f;
  for (int kk = 0; kk < p.k; ++kk) acc += U[(size_t)kk * p.n + i] * sc[kk];
  size_t idx = (size_t)c * p.n + i;
  z[idx] = Dinv[i] * (r[idx] - acc);
}
