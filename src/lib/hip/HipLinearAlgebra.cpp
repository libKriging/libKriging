// UNVERIFIED port of src/lib/cuda/CudaLinearAlgebra.cpp -- see
// HipLinearAlgebra.hpp for why. Host-compiler-compiled (NOT hipcc) on
// purpose, for the same reason as the CUDA backend: this file touches
// arma::mat/arma::vec (via .memptr()/.n_elem/etc), and must be compiled by
// the SAME compiler as the rest of libKriging to guarantee it agrees with
// Kriging.cpp/KrigingImpl.cpp on those types' memory layout. The only
// hipcc-compiled code this file talks to is the plain-C, pointer-only
// lk_hip_*_launch family declared in HipLinearAlgebraKernel.hpp.
//
// This is otherwise a close-to-line-for-line translation of
// CudaLinearAlgebra.cpp's conjugateGradient (API renames: cuda* -> hip*),
// not a redesign -- the CG algorithm, the batched/lockstep/frozen-column
// approach, and the restart logic are identical to the CUDA backend. See
// that file for the design rationale.
#include "HipLinearAlgebra.hpp"

#ifdef LIBKRIGING_USE_HIP_ITERATIVE

#include "HipLinearAlgebraKernel.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

#define LK_HIP_CHECK(expr)                                                           \
  do {                                                                               \
    hipError_t lk_hip_status__ = (expr);                                              \
    if (lk_hip_status__ != hipSuccess) {                                              \
      std::ostringstream lk_hip_oss__;                                               \
      lk_hip_oss__ << "HIP error at " << __FILE__ << ":" << __LINE__ << ": "         \
                   << hipGetErrorString(lk_hip_status__);                            \
      throw std::runtime_error(lk_hip_oss__.str());                                  \
    }                                                                                \
  } while (0)

enum class CovKind : int { Gauss = 0, Exp = 1, Matern32 = 2, Matern52 = 3 };

bool covKindFromString(const std::string& covType, CovKind* out) {
  if (covType == "gauss") {
    *out = CovKind::Gauss;
    return true;
  }
  if (covType == "exp") {
    *out = CovKind::Exp;
    return true;
  }
  if (covType == "matern3_2") {
    *out = CovKind::Matern32;
    return true;
  }
  if (covType == "matern5_2") {
    *out = CovKind::Matern52;
    return true;
  }
  return false;
}

}  // namespace

namespace LinearAlgebraHip {

bool available() {
  static const bool cached = [] {
    int count = 0;
    hipError_t status = hipGetDeviceCount(&count);
    return status == hipSuccess && count > 0;
  }();
  return cached;
}

namespace {
bool g_enabled_initialized = false;
bool g_enabled = false;
std::mutex g_enabled_mutex;
}  // namespace

bool enabled() {
  std::lock_guard<std::mutex> lock(g_enabled_mutex);
  if (!g_enabled_initialized) {
    g_enabled = available();
    g_enabled_initialized = true;
  }
  return g_enabled;
}

void set_enabled(bool value) {
  std::lock_guard<std::mutex> lock(g_enabled_mutex);
  g_enabled = value;
  g_enabled_initialized = true;
}

bool supports(const std::string& covType) {
  CovKind kind;
  return covKindFromString(covType, &kind);
}

arma::mat conjugateGradient(const arma::mat& Xt, const arma::vec& theta, const std::string& covType,
                            const arma::mat& B, arma::uword max_iter, double tol) {
  CovKind kind;
  if (!covKindFromString(covType, &kind))
    throw std::invalid_argument("LinearAlgebraHip::conjugateGradient: unsupported covType '" + covType + "'");

  const int n = static_cast<int>(Xt.n_cols);
  const int dimX = static_cast<int>(Xt.n_rows);
  const int ncols = static_cast<int>(B.n_cols);

  // Upload X and theta once -- reused for every CG iteration, never
  // re-transferred mid-solve.
  double *d_Xt, *d_theta;
  LK_HIP_CHECK(hipMalloc(&d_Xt, sizeof(double) * static_cast<std::size_t>(n) * dimX));
  LK_HIP_CHECK(hipMalloc(&d_theta, sizeof(double) * dimX));
  LK_HIP_CHECK(hipMemcpy(d_Xt, Xt.memptr(), sizeof(double) * static_cast<std::size_t>(n) * dimX,
                        hipMemcpyHostToDevice));
  LK_HIP_CHECK(hipMemcpy(d_theta, theta.memptr(), sizeof(double) * dimX, hipMemcpyHostToDevice));

  const std::size_t mat_bytes = sizeof(double) * static_cast<std::size_t>(n) * ncols;
  double *d_b, *d_x, *d_r, *d_p, *d_Ap;
  LK_HIP_CHECK(hipMalloc(&d_b, mat_bytes));
  LK_HIP_CHECK(hipMalloc(&d_x, mat_bytes));
  LK_HIP_CHECK(hipMalloc(&d_r, mat_bytes));
  LK_HIP_CHECK(hipMalloc(&d_p, mat_bytes));
  LK_HIP_CHECK(hipMalloc(&d_Ap, mat_bytes));

  const std::size_t col_bytes = sizeof(double) * static_cast<std::size_t>(ncols);
  double *d_scratch, *d_alpha, *d_beta;
  LK_HIP_CHECK(hipMalloc(&d_scratch, col_bytes));  // reused for pAp then for r.r each iteration
  LK_HIP_CHECK(hipMalloc(&d_alpha, col_bytes));
  LK_HIP_CHECK(hipMalloc(&d_beta, col_bytes));

  // n and ncols don't change across a CG solve, so the matvec's own scratch
  // requirement (see lk_hip_rmul_batched_scratch_elems) is fixed too --
  // allocate it once here rather than inside the per-iteration matvec call.
  const int rmul_scratch_elems = lk_hip_rmul_batched_scratch_elems(n, ncols);
  double* d_rmul_scratch = nullptr;
  if (rmul_scratch_elems > 0)
    LK_HIP_CHECK(hipMalloc(&d_rmul_scratch, sizeof(double) * static_cast<std::size_t>(rmul_scratch_elems)));

  LK_HIP_CHECK(hipMemcpy(d_b, B.memptr(), mat_bytes, hipMemcpyHostToDevice));
  LK_HIP_CHECK(hipMemset(d_x, 0, mat_bytes));
  LK_HIP_CHECK(hipMemcpy(d_r, d_b, mat_bytes, hipMemcpyDeviceToDevice));  // r = b - A*0
  LK_HIP_CHECK(hipMemcpy(d_p, d_r, mat_bytes, hipMemcpyDeviceToDevice));  // p = z = r (no precond)

  std::vector<double> bnorm(ncols), rz_old(ncols), host_scratch(ncols), host_alpha(ncols), host_beta(ncols);
  std::vector<bool> active(ncols);
  for (arma::uword c = 0; c < B.n_cols; ++c)
    bnorm[c] = arma::norm(B.col(c));
  for (int c = 0; c < ncols; ++c) {
    active[c] = bnorm[c] != 0.0;  // x=0 already solves A*x=0 for a zero column
    rz_old[c] = bnorm[c] * bnorm[c];  // r=b initially, so r.r = |b|^2
  }

  constexpr arma::uword restart_every = 50;
  bool any_active = std::any_of(active.begin(), active.end(), [](bool a) { return a; });

  for (arma::uword it = 0; any_active && it < max_iter; ++it) {
    lk_hip_rmul_batched_launch(d_Xt, n, dimX, d_theta, static_cast<int>(kind), d_p, ncols, d_Ap, d_rmul_scratch);
    LK_HIP_CHECK(hipGetLastError());
    lk_hip_batched_dot_launch(d_p, d_Ap, n, ncols, d_scratch);  // pAp
    LK_HIP_CHECK(hipGetLastError());
    LK_HIP_CHECK(hipMemcpy(host_scratch.data(), d_scratch, col_bytes, hipMemcpyDeviceToHost));

    for (int c = 0; c < ncols; ++c) {
      if (!active[c] || host_scratch[c] <= 0.0) {
        if (active[c])
          active[c] = false;  // breakdown guard, same as the CPU path
        host_alpha[c] = 0.0;
        continue;
      }
      host_alpha[c] = rz_old[c] / host_scratch[c];
    }
    LK_HIP_CHECK(hipMemcpy(d_alpha, host_alpha.data(), col_bytes, hipMemcpyHostToDevice));
    lk_hip_batched_axpy_launch(d_alpha, d_p, d_x, n, ncols);  // x += alpha*p
    LK_HIP_CHECK(hipGetLastError());

    if ((it + 1) % restart_every == 0) {
      // Full restart: recompute the exact residual r = b - A*x from
      // scratch for every still-active column, same rationale as
      // LinearAlgebra::conjugateGradient's restart_every (corrects
      // round-off drift in the recursively updated residual).
      lk_hip_rmul_batched_launch(d_Xt, n, dimX, d_theta, static_cast<int>(kind), d_x, ncols, d_Ap,
                                 d_rmul_scratch);  // Ap = A*x
      LK_HIP_CHECK(hipGetLastError());
      LK_HIP_CHECK(hipMemcpy(d_r, d_b, mat_bytes, hipMemcpyDeviceToDevice));  // r = b
      for (int c = 0; c < ncols; ++c)
        host_alpha[c] = -1.0;  // reuse d_alpha as an all-(-1) column vector
      LK_HIP_CHECK(hipMemcpy(d_alpha, host_alpha.data(), col_bytes, hipMemcpyHostToDevice));
      lk_hip_batched_axpy_launch(d_alpha, d_Ap, d_r, n, ncols);  // r += -1*(A*x) = b - A*x
      LK_HIP_CHECK(hipGetLastError());

      lk_hip_batched_dot_launch(d_r, d_r, n, ncols, d_scratch);  // r.r == rnorm^2
      LK_HIP_CHECK(hipGetLastError());
      LK_HIP_CHECK(hipMemcpy(host_scratch.data(), d_scratch, col_bytes, hipMemcpyDeviceToHost));
      for (int c = 0; c < ncols; ++c) {
        if (!active[c])
          continue;
        rz_old[c] = host_scratch[c];
        if (std::sqrt(host_scratch[c]) / bnorm[c] < tol)
          active[c] = false;
      }
      LK_HIP_CHECK(hipMemcpy(d_p, d_r, mat_bytes, hipMemcpyDeviceToDevice));  // restart: p = z = r
      any_active = std::any_of(active.begin(), active.end(), [](bool a) { return a; });
      continue;
    }

    for (int c = 0; c < ncols; ++c)
      host_alpha[c] = active[c] ? -host_alpha[c] : 0.0;
    LK_HIP_CHECK(hipMemcpy(d_alpha, host_alpha.data(), col_bytes, hipMemcpyHostToDevice));
    lk_hip_batched_axpy_launch(d_alpha, d_Ap, d_r, n, ncols);  // r -= alpha*Ap
    LK_HIP_CHECK(hipGetLastError());

    lk_hip_batched_dot_launch(d_r, d_r, n, ncols, d_scratch);  // r.r == rz_new == rnorm^2
    LK_HIP_CHECK(hipGetLastError());
    LK_HIP_CHECK(hipMemcpy(host_scratch.data(), d_scratch, col_bytes, hipMemcpyDeviceToHost));

    for (int c = 0; c < ncols; ++c) {
      if (!active[c]) {
        host_beta[c] = 0.0;
        continue;
      }
      if (std::sqrt(host_scratch[c]) / bnorm[c] < tol) {
        active[c] = false;
        host_beta[c] = 0.0;
        continue;
      }
      host_beta[c] = host_scratch[c] / rz_old[c];
      rz_old[c] = host_scratch[c];
    }
    LK_HIP_CHECK(hipMemcpy(d_beta, host_beta.data(), col_bytes, hipMemcpyHostToDevice));
    lk_hip_batched_update_p_launch(d_r, d_beta, d_p, n, ncols);  // p = r + beta*p
    LK_HIP_CHECK(hipGetLastError());

    any_active = std::any_of(active.begin(), active.end(), [](bool a) { return a; });
  }

  arma::mat X(n, ncols, arma::fill::none);
  LK_HIP_CHECK(hipMemcpy(X.memptr(), d_x, mat_bytes, hipMemcpyDeviceToHost));

  hipFree(d_Xt);
  hipFree(d_theta);
  hipFree(d_b);
  hipFree(d_x);
  hipFree(d_r);
  hipFree(d_p);
  hipFree(d_Ap);
  hipFree(d_scratch);
  hipFree(d_alpha);
  hipFree(d_beta);
  if (d_rmul_scratch)
    hipFree(d_rmul_scratch);

  return X;
}

}  // namespace LinearAlgebraHip

#endif  // LIBKRIGING_USE_HIP_ITERATIVE
