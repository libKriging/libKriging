// UNVERIFIED SYCL / Intel oneAPI port of src/lib/cuda/CudaLinearAlgebraKernel.cu
// -- see src/lib/sycl/SyclLinearAlgebra.hpp for the full caveat. Compiled by
// the SYCL compiler (icpx -fsycl / clang++ -fsycl); the SYCL headers are
// confined to this file. Never compiled or run: no oneAPI toolchain / Intel
// GPU in the environment this was written in.
//
// The kernel bodies are line-for-line the CUDA ones (same closed forms for
// the covariance and its d(ln cov)/d(theta), same CG scalar arithmetic,
// same Woodbury apply). The differences from CUDA are model-level only:
//   - one process-wide sycl::queue with in_order semantics replaces the
//     default CUDA stream; every launch .wait()s (simple + correct; a
//     verified follow-up can relax that for overlap).
//   - USM device pointers (sycl::malloc_device) replace cudaMalloc.
//   - grids become sycl::range<2>{n, ncols} (or range<1>) parallel_for.
//   - the tiled rmul path (rmul_batched_tiled_kernel / sum_partials /
//     chooseJBlocks -- a CUDA occupancy heuristic) is dropped; the plain
//     range<2> already distributes across an Intel GPU's slices/EUs, so
//     lk_sycl_rmul_batched_scratch_elems returns 0.

#include "SyclLinearAlgebraKernel.hpp"

#ifdef LIBKRIGING_USE_SYCL_ITERATIVE

#include <sycl/sycl.hpp>

#include <cstddef>
#include <stdexcept>

namespace {

// covKind: 0=gauss, 1=exp, 2=matern3_2, 3=matern5_2 (LinearAlgebraSycl's
// CovKind enum in SyclLinearAlgebra.cpp, passed through as a plain int).
inline double lk_cov_pair(int kind, const double* Xi, const double* Xj, const double* theta, int dimX) {
  double c, sum = 0.0, sum_sq = 0.0;
  switch (kind) {
    case 0:  // gauss
      for (int k = 0; k < dimX; ++k) {
        double v = (Xi[k] - Xj[k]) / theta[k];
        sum_sq += v * v;
      }
      c = sycl::exp(-0.5 * sum_sq);
      break;
    case 1:  // exp
      for (int k = 0; k < dimX; ++k)
        sum += sycl::fabs((Xi[k] - Xj[k]) / theta[k]);
      c = sycl::exp(-sum);
      break;
    case 2:  // matern3_2
      for (int k = 0; k < dimX; ++k) {
        double d = 1.7320508075688772 * sycl::fabs((Xi[k] - Xj[k]) / theta[k]);
        sum += d - sycl::log1p(d);
      }
      c = sycl::exp(-sum);
      break;
    default:  // matern5_2
      for (int k = 0; k < dimX; ++k) {
        double d = 2.23606797749979 * sycl::fabs((Xi[k] - Xj[k]) / theta[k]);
        sum += d - sycl::log1p(d + (d * d) / 3.0);
      }
      c = sycl::exp(-sum);
      break;
  }
  return c;
}

constexpr int LK_SYCL_MAX_DIMX = 32;

// out[k] = d(ln cov)/d(theta_k), matching Covariance::DlnCovDtheta_* exactly.
inline void lk_dlncov_pair(int kind, const double* Xi, const double* Xj, const double* theta, int dimX, double* out) {
  switch (kind) {
    case 0:  // gauss
      for (int k = 0; k < dimX; ++k) {
        double dx = Xi[k] - Xj[k];
        out[k] = (dx * dx) / (theta[k] * theta[k] * theta[k]);
      }
      break;
    case 1:  // exp
      for (int k = 0; k < dimX; ++k)
        out[k] = sycl::fabs(Xi[k] - Xj[k]) / (theta[k] * theta[k]);
      break;
    case 2:  // matern3_2
      for (int k = 0; k < dimX; ++k) {
        double d = 1.7320508075688772 * sycl::fabs((Xi[k] - Xj[k]) / theta[k]);
        out[k] = (d * d) / (1.0 + d) / theta[k];
      }
      break;
    default:  // matern5_2
      for (int k = 0; k < dimX; ++k) {
        double d = 2.23606797749979 * sycl::fabs((Xi[k] - Xj[k]) / theta[k]);
        double a = 1.0 + d, b = (d * d) / 3.0;
        out[k] = (a * b) / (a + b) / theta[k];
      }
      break;
  }
}

// One process-wide in-order queue on the first SYCL GPU. Constructed on
// first use; throws sycl::exception if no GPU is present, which
// lk_sycl_available() catches.
sycl::queue& q() {
  static sycl::queue Q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
  return Q;
}

const bool g_available = [] {
  try {
    (void)q();
    return true;
  } catch (...) {
    return false;
  }
}();

}  // namespace

extern "C" int lk_sycl_available(void) {
  return g_available ? 1 : 0;
}

extern "C" void* lk_sycl_malloc(unsigned long bytes) {
  void* p = sycl::malloc_device(static_cast<std::size_t>(bytes), q());
  if (!p)
    throw std::runtime_error("lk_sycl_malloc: sycl::malloc_device returned null");
  return p;
}

extern "C" void lk_sycl_free(void* p) {
  if (p)
    sycl::free(p, q());
}

extern "C" void lk_sycl_memcpy(void* dst, const void* src, unsigned long bytes, int /*kind*/) {
  q().memcpy(dst, src, static_cast<std::size_t>(bytes)).wait();
}

extern "C" void lk_sycl_memset(void* p, int value, unsigned long bytes) {
  q().memset(p, value, static_cast<std::size_t>(bytes)).wait();
}

extern "C" void lk_sycl_sync(void) {
  q().wait();
}

extern "C" int lk_sycl_rmul_batched_scratch_elems(int, int) {
  return 0;
}

extern "C" void lk_sycl_rmul_batched_launch(const double* Xt, int n, int dimX, const double* theta, int covKind,
                                            const double* P, int ncols, double* Ap, double*) {
  q().parallel_for(sycl::range<2>(static_cast<std::size_t>(n), static_cast<std::size_t>(ncols)),
                   [=](sycl::id<2> id) {
                     const int i = static_cast<int>(id[0]);
                     const int c = static_cast<int>(id[1]);
                     const double* Pc = P + static_cast<std::size_t>(c) * n;
                     const double* Xi = Xt + static_cast<std::size_t>(i) * dimX;
                     double acc = Pc[i];  // diag = 1
                     for (int j = 0; j < n; ++j) {
                       if (j == i)
                         continue;
                       acc += lk_cov_pair(covKind, Xi, Xt + static_cast<std::size_t>(j) * dimX, theta, dimX) * Pc[j];
                     }
                     Ap[static_cast<std::size_t>(c) * n + i] = acc;
                   })
      .wait();
}

extern "C" void lk_sycl_drmul_batched_launch(const double* Xt, int n, int dimX, const double* theta, int covKind,
                                             const double* V, int ncols, double* Out) {
  q().parallel_for(sycl::range<2>(static_cast<std::size_t>(n), static_cast<std::size_t>(ncols)),
                   [=](sycl::id<2> id) {
                     const int i = static_cast<int>(id[0]);
                     const int c = static_cast<int>(id[1]);
                     double acc[LK_SYCL_MAX_DIMX];
                     double dln[LK_SYCL_MAX_DIMX];
                     for (int k = 0; k < dimX; ++k)
                       acc[k] = 0.0;
                     const double* Vc = V + static_cast<std::size_t>(c) * n;
                     const double* Xi = Xt + static_cast<std::size_t>(i) * dimX;
                     for (int j = 0; j < n; ++j) {
                       if (j == i)
                         continue;
                       const double* Xj = Xt + static_cast<std::size_t>(j) * dimX;
                       const double cij = lk_cov_pair(covKind, Xi, Xj, theta, dimX);
                       lk_dlncov_pair(covKind, Xi, Xj, theta, dimX, dln);
                       const double vj = Vc[j];
                       for (int k = 0; k < dimX; ++k)
                         acc[k] += cij * dln[k] * vj;
                     }
                     double* Oc = Out + static_cast<std::size_t>(c) * dimX * n;
                     for (int k = 0; k < dimX; ++k)
                       Oc[static_cast<std::size_t>(k) * n + i] = acc[k];
                   })
      .wait();
}

extern "C" void lk_sycl_batched_dot_launch(const double* A, const double* B, int n, int ncols, double* out) {
  q().parallel_for(sycl::range<1>(static_cast<std::size_t>(ncols)), [=](sycl::id<1> id) {
     const int c = static_cast<int>(id[0]);
     const double* Ac = A + static_cast<std::size_t>(c) * n;
     const double* Bc = B + static_cast<std::size_t>(c) * n;
     double s = 0.0;
     for (int i = 0; i < n; ++i)
       s += Ac[i] * Bc[i];
     out[c] = s;
   }).wait();
}

extern "C" void lk_sycl_batched_axpy_launch(const double* alpha, const double* X, double* Y, int n, int ncols) {
  q().parallel_for(sycl::range<2>(static_cast<std::size_t>(n), static_cast<std::size_t>(ncols)),
                   [=](sycl::id<2> id) {
                     const std::size_t idx = id[1] * n + id[0];
                     Y[idx] += alpha[id[1]] * X[idx];
                   })
      .wait();
}

extern "C" void lk_sycl_batched_update_p_launch(const double* R, const double* beta, double* P, int n, int ncols) {
  q().parallel_for(sycl::range<2>(static_cast<std::size_t>(n), static_cast<std::size_t>(ncols)),
                   [=](sycl::id<2> id) {
                     const std::size_t idx = id[1] * n + id[0];
                     P[idx] = R[idx] + beta[id[1]] * P[idx];
                   })
      .wait();
}

extern "C" void lk_sycl_cg_alpha_launch(const double* rz_old, const double* pAp, int ncols, int* active, double* alpha,
                                        double* neg_alpha) {
  q().parallel_for(sycl::range<1>(static_cast<std::size_t>(ncols)), [=](sycl::id<1> id) {
     const int c = static_cast<int>(id[0]);
     if (!active[c]) {
       alpha[c] = 0.0;
       neg_alpha[c] = 0.0;
       return;
     }
     const double p = pAp[c];
     if (!(p > 0.0)) {
       active[c] = 0;
       alpha[c] = 0.0;
       neg_alpha[c] = 0.0;
       return;
     }
     const double a = rz_old[c] / p;
     alpha[c] = a;
     neg_alpha[c] = -a;
   }).wait();
}

extern "C" void lk_sycl_cg_beta_launch(const double* rr_new, const double* bnorm, double tol, int ncols, int* active,
                                       double* rz_old, double* beta) {
  q().parallel_for(sycl::range<1>(static_cast<std::size_t>(ncols)), [=](sycl::id<1> id) {
     const int c = static_cast<int>(id[0]);
     if (!active[c]) {
       beta[c] = 0.0;
       return;
     }
     const double rn = rr_new[c];
     if (sycl::sqrt(rn) / bnorm[c] < tol) {
       active[c] = 0;
       beta[c] = 0.0;
       return;
     }
     beta[c] = rn / rz_old[c];
     rz_old[c] = rn;
   }).wait();
}

extern "C" void lk_sycl_cg_restart_launch(const double* rr, const double* bnorm, double tol, int ncols, int* active,
                                          double* rz_old) {
  q().parallel_for(sycl::range<1>(static_cast<std::size_t>(ncols)), [=](sycl::id<1> id) {
     const int c = static_cast<int>(id[0]);
     if (!active[c])
       return;
     rz_old[c] = rr[c];
     if (sycl::sqrt(rr[c]) / bnorm[c] < tol)
       active[c] = 0;
   }).wait();
}

extern "C" void lk_sycl_cg_restart_precond_launch(const double* rr, const double* rz, const double* bnorm, double tol,
                                                  int ncols, int* active, double* rz_old) {
  q().parallel_for(sycl::range<1>(static_cast<std::size_t>(ncols)), [=](sycl::id<1> id) {
     const int c = static_cast<int>(id[0]);
     if (!active[c])
       return;
     rz_old[c] = rz[c];
     if (sycl::sqrt(rr[c]) / bnorm[c] < tol)
       active[c] = 0;
   }).wait();
}

extern "C" void lk_sycl_cg_beta_precond_launch(const double* rr, const double* rz_new, const double* bnorm, double tol,
                                               int ncols, int* active, double* rz_old, double* beta) {
  q().parallel_for(sycl::range<1>(static_cast<std::size_t>(ncols)), [=](sycl::id<1> id) {
     const int c = static_cast<int>(id[0]);
     if (!active[c]) {
       beta[c] = 0.0;
       return;
     }
     if (sycl::sqrt(rr[c]) / bnorm[c] < tol) {
       active[c] = 0;
       beta[c] = 0.0;
       return;
     }
     beta[c] = rz_new[c] / rz_old[c];
     rz_old[c] = rz_new[c];
   }).wait();
}

extern "C" void lk_sycl_cg_any_active_launch(const int* active, int ncols, int* flag) {
  q().parallel_for(sycl::range<1>(static_cast<std::size_t>(ncols)), [=](sycl::id<1> id) {
     if (active[id[0]]) {
       sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> a(*flag);
       a.fetch_or(1);
     }
   }).wait();
}

extern "C" void lk_sycl_precond_apply_launch(const double* U, int n, int k, const double* Dinv, const double* Mchol,
                                             const double* r, int ncols, double* z, double* scratch_nc,
                                             double* scratch_kc) {
  // scratch_nc = Dinv .* r
  q().parallel_for(sycl::range<2>(static_cast<std::size_t>(n), static_cast<std::size_t>(ncols)),
                   [=](sycl::id<2> id) {
                     const std::size_t idx = id[1] * n + id[0];
                     scratch_nc[idx] = Dinv[id[0]] * r[idx];
                   })
      .wait();
  // scratch_kc[kk,c] = sum_i U[i,kk] * scratch_nc[i,c]
  q().parallel_for(sycl::range<2>(static_cast<std::size_t>(k), static_cast<std::size_t>(ncols)),
                   [=](sycl::id<2> id) {
                     const int kk = static_cast<int>(id[0]);
                     const int c = static_cast<int>(id[1]);
                     const double* Uk = U + static_cast<std::size_t>(kk) * n;
                     const double* Zc = scratch_nc + static_cast<std::size_t>(c) * n;
                     double acc = 0.0;
                     for (int i = 0; i < n; ++i)
                       acc += Uk[i] * Zc[i];
                     scratch_kc[static_cast<std::size_t>(c) * k + kk] = acc;
                   })
      .wait();
  // solve (L L^T) s = t in place, one column per work-item
  q().parallel_for(sycl::range<1>(static_cast<std::size_t>(ncols)), [=](sycl::id<1> id) {
     double* tc = scratch_kc + id[0] * k;
     for (int i = 0; i < k; ++i) {
       double s = tc[i];
       for (int j = 0; j < i; ++j)
         s -= Mchol[static_cast<std::size_t>(j) * k + i] * tc[j];
       tc[i] = s / Mchol[static_cast<std::size_t>(i) * k + i];
     }
     for (int i = k - 1; i >= 0; --i) {
       double s = tc[i];
       for (int j = i + 1; j < k; ++j)
         s -= Mchol[static_cast<std::size_t>(i) * k + j] * tc[j];
       tc[i] = s / Mchol[static_cast<std::size_t>(i) * k + i];
     }
   }).wait();
  // z[i,c] = Dinv[i] * (r[i,c] - sum_kk U[i,kk] * s[kk,c])
  q().parallel_for(sycl::range<2>(static_cast<std::size_t>(n), static_cast<std::size_t>(ncols)),
                   [=](sycl::id<2> id) {
                     const int i = static_cast<int>(id[0]);
                     const int c = static_cast<int>(id[1]);
                     const double* sc = scratch_kc + static_cast<std::size_t>(c) * k;
                     double acc = 0.0;
                     for (int kk = 0; kk < k; ++kk)
                       acc += U[static_cast<std::size_t>(kk) * n + i] * sc[kk];
                     const std::size_t idx = static_cast<std::size_t>(c) * n + i;
                     z[idx] = Dinv[i] * (r[idx] - acc);
                   })
      .wait();
}

#endif  // LIBKRIGING_USE_SYCL_ITERATIVE
