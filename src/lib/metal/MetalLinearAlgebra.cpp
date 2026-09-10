// UNVERIFIED Apple-Metal host orchestration -- see MetalLinearAlgebra.hpp for
// the full caveat (never compiled or run; FLOAT32-only). Compiled by the
// normal compiler (no Metal / metal-cpp symbols here): only calls the
// plain-C lk_metal_* surface, which owns every MTL::Buffer and does the
// double<->float narrowing. Same ABI-safety rule as the CUDA backend.
//
// The CG loop is the same algorithm as CudaLinearAlgebra.cpp's (batched,
// lockstep columns, on-device alpha/beta scalars, periodic restart) -- only
// written out here rather than macro-aliased, because the float device
// buffers need explicit typed upload/download.

#include "MetalLinearAlgebra.hpp"

#ifdef LIBKRIGING_USE_METAL_ITERATIVE

#include "MetalLinearAlgebraKernel.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <mutex>
#include <vector>

namespace {

enum : int { GAUSS = 0, EXP = 1, MATERN32 = 2, MATERN52 = 3 };

bool covKind(const std::string& covType, int* out) {
  if (covType == "gauss") { *out = GAUSS; return true; }
  if (covType == "exp") { *out = EXP; return true; }
  if (covType == "matern3_2") { *out = MATERN32; return true; }
  if (covType == "matern5_2") { *out = MATERN52; return true; }
  return false;
}

// RAII float device buffer sized for `count` elements.
struct FBuf {
  void* h = nullptr;
  explicit FBuf(std::size_t count) : h(lk_metal_malloc(count * sizeof(float))) {}
  FBuf(std::size_t count, const double* src) : h(lk_metal_malloc(count * sizeof(float))) {
    lk_metal_upload_f64_as_f32(h, src, count);
  }
  ~FBuf() { lk_metal_free(h); }
  FBuf(const FBuf&) = delete;
  FBuf& operator=(const FBuf&) = delete;
  operator void*() const { return h; }
};
struct IBuf {
  void* h = nullptr;
  IBuf(std::size_t count, const int* src) : h(lk_metal_malloc(count * sizeof(int))) {
    lk_metal_upload_i32(h, src, count);
  }
  explicit IBuf(std::size_t count) : h(lk_metal_malloc(count * sizeof(int))) {}
  ~IBuf() { lk_metal_free(h); }
  IBuf(const IBuf&) = delete;
  IBuf& operator=(const IBuf&) = delete;
  operator void*() const { return h; }
};

}  // namespace

namespace LinearAlgebraMetal {

bool available() {
  static const bool cached = lk_metal_available() != 0;
  return cached;
}

namespace {
bool g_init = false, g_enabled = false;
std::mutex g_mtx;
}  // namespace

bool enabled() {
  std::lock_guard<std::mutex> lock(g_mtx);
  if (!g_init) {
    g_enabled = available();
    g_init = true;
  }
  return g_enabled;
}
void set_enabled(bool v) {
  std::lock_guard<std::mutex> lock(g_mtx);
  g_enabled = v;
  g_init = true;
}
bool supports(const std::string& covType) {
  int k;
  return covKind(covType, &k);
}

arma::mat rmulBatched(const arma::mat& Xt, const arma::vec& theta, const std::string& covType, const arma::mat& V) {
  int kind;
  if (!covKind(covType, &kind))
    throw std::invalid_argument("LinearAlgebraMetal::rmulBatched: unsupported covType '" + covType + "'");
  const int n = static_cast<int>(Xt.n_cols);
  const int dimX = static_cast<int>(Xt.n_rows);
  const int ncols = static_cast<int>(V.n_cols);
  FBuf dXt(static_cast<std::size_t>(n) * dimX, Xt.memptr());
  FBuf dTheta(dimX, theta.memptr());
  FBuf dV(static_cast<std::size_t>(n) * ncols, V.memptr());
  FBuf dAv(static_cast<std::size_t>(n) * ncols);
  lk_metal_rmul_batched_launch(dXt, n, dimX, dTheta, kind, dV, ncols, dAv, nullptr);
  arma::mat Av(static_cast<arma::uword>(n), static_cast<arma::uword>(ncols), arma::fill::none);
  lk_metal_download_f32_as_f64(Av.memptr(), dAv, static_cast<std::size_t>(n) * ncols);
  return Av;
}

arma::mat dRmulBatched(const arma::mat& Xt, const arma::vec& theta, const std::string& covType, const arma::mat& V) {
  int kind;
  if (!covKind(covType, &kind))
    throw std::invalid_argument("LinearAlgebraMetal::dRmulBatched: unsupported covType '" + covType + "'");
  const int n = static_cast<int>(Xt.n_cols);
  const int dimX = static_cast<int>(Xt.n_rows);
  const int ncols = static_cast<int>(V.n_cols);
  if (dimX > kMaxDimX)
    throw std::invalid_argument("LinearAlgebraMetal::dRmulBatched: dimX > 32");
  FBuf dXt(static_cast<std::size_t>(n) * dimX, Xt.memptr());
  FBuf dTheta(dimX, theta.memptr());
  FBuf dV(static_cast<std::size_t>(n) * ncols, V.memptr());
  FBuf dOut(static_cast<std::size_t>(n) * dimX * ncols);
  lk_metal_drmul_batched_launch(dXt, n, dimX, dTheta, kind, dV, ncols, dOut);
  arma::mat Out(static_cast<arma::uword>(n), static_cast<arma::uword>(dimX) * ncols, arma::fill::none);
  lk_metal_download_f32_as_f64(Out.memptr(), dOut, static_cast<std::size_t>(n) * dimX * ncols);
  return Out;
}

arma::mat conjugateGradient(const arma::mat& Xt,
                            const arma::vec& theta,
                            const std::string& covType,
                            const arma::mat& B,
                            arma::uword max_iter,
                            double tol,
                            const arma::mat& precU,
                            const arma::vec& precDinv,
                            const arma::mat& precMcholLower) {
  int kind;
  if (!covKind(covType, &kind))
    throw std::invalid_argument("LinearAlgebraMetal::conjugateGradient: unsupported covType '" + covType + "'");
  const int n = static_cast<int>(Xt.n_cols);
  const int dimX = static_cast<int>(Xt.n_rows);
  const int ncols = static_cast<int>(B.n_cols);
  const bool preconditioned = (precU.n_elem > 0);
  const int pk = preconditioned ? static_cast<int>(precU.n_cols) : 0;
  const std::size_t mat = static_cast<std::size_t>(n) * ncols;

  // FLOAT32 CG on an ill-conditioned interpolation kernel loses digits fast;
  // don't chase a tolerance single precision can't reach.
  const double eff_tol = std::max(tol, 1e-4);

  FBuf dXt(static_cast<std::size_t>(n) * dimX, Xt.memptr());
  FBuf dTheta(dimX, theta.memptr());
  FBuf dB(mat, B.memptr()), dX(mat), dR(mat), dP(mat), dAp(mat);
  FBuf dScratch(ncols), dScratch2(ncols), dAlpha(ncols), dNegAlpha(ncols), dBeta(ncols), dRzOld(ncols), dBnorm(ncols),
      dNegOnes(ncols);
  IBuf dActive(ncols), dFlag(1);

  std::vector<double> bnorm(ncols), rz0(ncols), negones(ncols, -1.0);
  std::vector<int> active(ncols);
  bool any = false;
  for (int c = 0; c < ncols; ++c) {
    bnorm[c] = arma::norm(B.col(c));
    active[c] = bnorm[c] != 0.0 ? 1 : 0;
    rz0[c] = bnorm[c] * bnorm[c];
    any = any || active[c];
  }
  lk_metal_upload_f64_as_f32(dBnorm, bnorm.data(), ncols);
  lk_metal_upload_f64_as_f32(dNegOnes, negones.data(), ncols);
  lk_metal_upload_i32(dActive, active.data(), ncols);
  lk_metal_memset_dev(dX, 0, mat * sizeof(float));
  lk_metal_copy_dev(dR, dB, mat * sizeof(float));

  std::unique_ptr<FBuf> dU, dDinv, dMchol, dZ, dPrecNc, dPrecKc;
  if (preconditioned) {
    dU = std::make_unique<FBuf>(static_cast<std::size_t>(n) * pk, precU.memptr());
    dDinv = std::make_unique<FBuf>(static_cast<std::size_t>(n), precDinv.memptr());
    dMchol = std::make_unique<FBuf>(static_cast<std::size_t>(pk) * pk, precMcholLower.memptr());
    dZ = std::make_unique<FBuf>(mat);
    dPrecNc = std::make_unique<FBuf>(mat);
    dPrecKc = std::make_unique<FBuf>(static_cast<std::size_t>(pk) * ncols);
  }
  auto precondApply = [&](void* in, void* out) {
    lk_metal_precond_apply_launch(*dU, n, pk, *dDinv, *dMchol, in, ncols, out, *dPrecNc, *dPrecKc);
  };

  if (preconditioned) {
    precondApply(dR, *dZ);
    lk_metal_copy_dev(dP, *dZ, mat * sizeof(float));
    lk_metal_batched_dot_launch(dR, *dZ, n, ncols, dRzOld);
  } else {
    lk_metal_copy_dev(dP, dR, mat * sizeof(float));
    lk_metal_upload_f64_as_f32(dRzOld, rz0.data(), ncols);
  }

  constexpr arma::uword restart_every = 50, sync_every = 10;
  int host_flag = any ? 1 : 0;
  for (arma::uword it = 0; host_flag && it < max_iter; ++it) {
    lk_metal_rmul_batched_launch(dXt, n, dimX, dTheta, kind, dP, ncols, dAp, nullptr);
    lk_metal_batched_dot_launch(dP, dAp, n, ncols, dScratch);
    lk_metal_cg_alpha_launch(dRzOld, dScratch, ncols, dActive, dAlpha, dNegAlpha);
    lk_metal_batched_axpy_launch(dAlpha, dP, dX, n, ncols);

    if ((it + 1) % restart_every == 0) {
      lk_metal_rmul_batched_launch(dXt, n, dimX, dTheta, kind, dX, ncols, dAp, nullptr);
      lk_metal_copy_dev(dR, dB, mat * sizeof(float));
      lk_metal_batched_axpy_launch(dNegOnes, dAp, dR, n, ncols);
      lk_metal_batched_dot_launch(dR, dR, n, ncols, dScratch);
      if (preconditioned) {
        precondApply(dR, *dZ);
        lk_metal_batched_dot_launch(dR, *dZ, n, ncols, dScratch2);
        lk_metal_cg_restart_precond_launch(dScratch, dScratch2, dBnorm, eff_tol, ncols, dActive, dRzOld);
        lk_metal_copy_dev(dP, *dZ, mat * sizeof(float));
      } else {
        lk_metal_cg_restart_launch(dScratch, dBnorm, eff_tol, ncols, dActive, dRzOld);
        lk_metal_copy_dev(dP, dR, mat * sizeof(float));
      }
    } else {
      lk_metal_batched_axpy_launch(dNegAlpha, dAp, dR, n, ncols);
      lk_metal_batched_dot_launch(dR, dR, n, ncols, dScratch);
      if (preconditioned) {
        precondApply(dR, *dZ);
        lk_metal_batched_dot_launch(dR, *dZ, n, ncols, dScratch2);
        lk_metal_cg_beta_precond_launch(dScratch, dScratch2, dBnorm, eff_tol, ncols, dActive, dRzOld, dBeta);
        lk_metal_batched_update_p_launch(*dZ, dBeta, dP, n, ncols);
      } else {
        lk_metal_cg_beta_launch(dScratch, dBnorm, eff_tol, ncols, dActive, dRzOld, dBeta);
        lk_metal_batched_update_p_launch(dR, dBeta, dP, n, ncols);
      }
    }

    if ((it + 1) % sync_every == 0 || (it + 1) % restart_every == 0) {
      lk_metal_memset_dev(dFlag, 0, sizeof(int));
      lk_metal_cg_any_active_launch(dActive, ncols, dFlag);
      lk_metal_download_i32(&host_flag, dFlag, 1);
    }
  }

  arma::mat X(static_cast<arma::uword>(n), static_cast<arma::uword>(ncols), arma::fill::none);
  lk_metal_download_f32_as_f64(X.memptr(), dX, mat);
  return X;
}

}  // namespace LinearAlgebraMetal

#endif  // LIBKRIGING_USE_METAL_ITERATIVE
