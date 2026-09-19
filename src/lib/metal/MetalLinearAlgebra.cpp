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

#include "libKriging/LinearAlgebra.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
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

// Dense-R fast path: materialize R once per (Xt, theta, covType) instead of
// re-evaluating covariance transcendentals on every CG iteration / Lanczos
// step -- see lk_metal_kernels.metal's build_cov/dense_matvec doc comment.
// Mechanical port of CudaLinearAlgebra.cpp's/HipLinearAlgebra.cpp's
// DenseCovCache: process-wide, single-entry cache keyed by VALUE (not
// pointer -- callers pass temporaries and Armadillo reuses freed memory),
// gated by a device-memory budget (LK_ITERATIVE_METAL_DENSE_MAX_MB,
// default 4096 MiB; 0 forces the matrix-free kernel). Only R is cached
// (not the dRmulBatched dR/dtheta_k blocks -- used once per gradient
// eval, nothing to amortize).
struct DenseCovCache {
  int n = 0;
  int dimX = 0;
  int kind = -1;
  std::vector<double> xt;
  std::vector<double> theta;
  void* d_Xt = nullptr;
  void* d_theta = nullptr;
  void* d_R = nullptr;
};

DenseCovCache& covCache() {
  static DenseCovCache cache;
  return cache;
}

bool covCacheMatches(const DenseCovCache& c, const arma::mat& Xt, const arma::vec& theta, int kind) {
  if (c.d_Xt == nullptr || c.n != static_cast<int>(Xt.n_cols) || c.dimX != static_cast<int>(Xt.n_rows)
      || c.kind != kind)
    return false;
  if (c.xt.size() != Xt.n_elem || c.theta.size() != theta.n_elem)
    return false;
  return std::equal(c.xt.begin(), c.xt.end(), Xt.memptr()) && std::equal(c.theta.begin(), c.theta.end(), theta.memptr());
}

// Binds the cache to (Xt, theta, kind), uploading Xt/theta if the key
// changed, and returns it. d_R is left null on a key change; covCacheR()
// below is what actually materializes it (callers that only need the
// matrix-free path must not pay for a build they won't use).
DenseCovCache& covCacheBind(const arma::mat& Xt, const arma::vec& theta, int kind) {
  DenseCovCache& c = covCache();
  if (covCacheMatches(c, Xt, theta, kind))
    return c;

  lk_metal_free(c.d_Xt);
  lk_metal_free(c.d_theta);
  lk_metal_free(c.d_R);
  c.d_Xt = c.d_theta = c.d_R = nullptr;

  const int n = static_cast<int>(Xt.n_cols);
  const int dimX = static_cast<int>(Xt.n_rows);
  c.d_Xt = lk_metal_malloc(static_cast<std::size_t>(n) * dimX * sizeof(float));
  c.d_theta = lk_metal_malloc(static_cast<std::size_t>(dimX) * sizeof(float));
  lk_metal_upload_f64_as_f32(c.d_Xt, Xt.memptr(), static_cast<std::size_t>(n) * dimX);
  lk_metal_upload_f64_as_f32(c.d_theta, theta.memptr(), static_cast<std::size_t>(dimX));

  c.n = n;
  c.dimX = dimX;
  c.kind = kind;
  c.xt.assign(Xt.memptr(), Xt.memptr() + Xt.n_elem);
  c.theta.assign(theta.memptr(), theta.memptr() + theta.n_elem);
  return c;
}

bool denseFitsBudget(double need_mb) {
  std::size_t budget_mb = 4096;
  if (const char* e = std::getenv("LK_ITERATIVE_METAL_DENSE_MAX_MB")) {
    try {
      budget_mb = static_cast<std::size_t>(std::stoull(e));
    } catch (...) { /* keep default */
    }
  }
  return budget_mb > 0 && need_mb <= static_cast<double>(budget_mb);
}

// Cached dense R (float32), built on first use. Returns nullptr when the
// dense path is over budget, so callers keep their matrix-free fallback.
void* covCacheR(DenseCovCache& c) {
  if (c.d_R)
    return c.d_R;
  const double need_mb = static_cast<double>(c.n) * c.n * sizeof(float) / (1024.0 * 1024.0);
  if (!denseFitsBudget(need_mb))
    return nullptr;
  c.d_R = lk_metal_malloc(static_cast<std::size_t>(c.n) * c.n * sizeof(float));
  lk_metal_build_cov_launch(c.d_Xt, c.n, c.dimX, c.d_theta, c.kind, c.d_R);
  return c.d_R;
}

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
  const int ncols = static_cast<int>(V.n_cols);
  DenseCovCache& cov = covCacheBind(Xt, theta, kind);
  void* dRmat = covCacheR(cov);
  FBuf dV(static_cast<std::size_t>(n) * ncols, V.memptr());
  FBuf dAv(static_cast<std::size_t>(n) * ncols);
  if (dRmat) {
    lk_metal_dense_matvec_launch(dRmat, n, dV, ncols, dAv, n);
  } else {
    lk_metal_rmul_batched_launch(cov.d_Xt, n, cov.dimX, cov.d_theta, kind, dV, ncols, dAv, nullptr);
  }
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
                            const arma::vec& tol,
                            const arma::mat& precU,
                            const arma::vec& precDinv,
                            const arma::mat& precMcholLower,
                            arma::uword* n_unconverged_out,
                            const arma::mat* X0) {
  int kind;
  if (!covKind(covType, &kind))
    throw std::invalid_argument("LinearAlgebraMetal::conjugateGradient: unsupported covType '" + covType + "'");
  const int n = static_cast<int>(Xt.n_cols);
  const int dimX = static_cast<int>(Xt.n_rows);
  const int ncols = static_cast<int>(B.n_cols);
  const bool preconditioned = (precU.n_elem > 0);
  const int pk = preconditioned ? static_cast<int>(precU.n_cols) : 0;
  const std::size_t mat = static_cast<std::size_t>(n) * ncols;

  // FLOAT32 CG still can't reach a double solver's ~1e-12, but with the
  // compensated-sum reductions in lk_metal_kernels.metal the matvec/dot are
  // accurate to ~eps, so the relative residual can be driven to ~1e-6 before
  // round-off stalls it -- floor there rather than at the old, far looser
  // 1e-4 (which by itself accounted for the visible gap vs the CPU path).
  // tol is PER-COLUMN (mBCG fusion: Kriging.cpp's [F|y|probes] solve wants
  // F/y at the tighter cg_tol and probes at the looser probes_cg_tol in ONE
  // Krylov pass -- see lk_metal_kernels.metal's CgP doc comment).
  const arma::vec eff_tol = arma::clamp(tol, 1e-6, arma::datum::inf);
  FBuf dTol(static_cast<std::size_t>(ncols), eff_tol.memptr());

  // Dense-R fast path: a whole objective evaluation (this CG solve, the SLQ
  // Lanczos recurrence, the probes CG solve) runs at a FIXED theta, so the
  // first call here pays the upload + O(n^2) build and every later matvec
  // at the same theta reuses it -- see DenseCovCache's doc comment.
  DenseCovCache& cov = covCacheBind(Xt, theta, kind);
  void* dRmat = covCacheR(cov);
  auto matvec = [&](void* in, std::size_t in_byte_offset, void* out) {
    if (dRmat)
      lk_metal_dense_matvec_launch(dRmat, n, in, ncols, out, n, static_cast<long>(in_byte_offset));
    else
      lk_metal_rmul_batched_launch(cov.d_Xt, n, dimX, cov.d_theta, kind, in, ncols, out, nullptr,
                                   static_cast<long>(in_byte_offset));
  };

  FBuf dB(mat, B.memptr()), dX(mat), dR(mat), dP(mat), dAp(mat);
  FBuf dScratch(ncols), dScratch2(ncols), dAlpha(ncols), dNegAlpha(ncols), dBeta(ncols), dRzOld(ncols), dBnorm(ncols),
      dNegOnes(ncols);
  IBuf dActive(ncols), dFlag(1);

  std::vector<double> bnorm(ncols), negones(ncols, -1.0);
  for (int c = 0; c < ncols; ++c)
    bnorm[c] = arma::norm(B.col(c));
  lk_metal_upload_f64_as_f32(dBnorm, bnorm.data(), ncols);
  lk_metal_upload_f64_as_f32(dNegOnes, negones.data(), ncols);

  // neg_ones is uploaded above (constant, independent of B/X0) since the X0
  // branch below needs it for the "r = b - A*x0" axpy.
  if (X0 != nullptr) {
    lk_metal_upload_f64_as_f32(dX, X0->memptr(), mat);
    matvec(dX, 0, dAp);  // Ap = A*x0
    lk_metal_copy_dev(dR, dB, mat * sizeof(float));                                    // r = b
    lk_metal_batched_axpy_launch(dNegOnes, dAp, dR, n, ncols);                         // r -= Ap
  } else {
    lk_metal_memset_dev(dX, 0, mat * sizeof(float));
    lk_metal_copy_dev(dR, dB, mat * sizeof(float));  // r = b - A*0
  }

  std::unique_ptr<FBuf> dU, dDinv, dMchol, dZ, dPrecNc, dPrecKc;
  if (preconditioned) {
    dU = std::make_unique<FBuf>(static_cast<std::size_t>(n) * pk, precU.memptr());
    // WoodburyFactorization floors D at LinearAlgebra::num_nugget (1e-10),
    // so Dinv can reach ~1e10 wherever the Nystrom factor already captures
    // ~all of a row's variance. precond_combine's z = Dinv*(r - U*s) then
    // needs U*s to cancel r to within Dinv's own ~1e10 scale to land on an
    // O(1) z -- fine in double (M_chol/U/r keep ~15-16 digits, leaving ~5-6
    // after the cancellation) but catastrophic in float32 (~7 digits total,
    // so ~-3 digits left => pure noise, verified empirically: preconditioned
    // CG landed 12x WORSE than plain CG at a tight iteration budget before
    // this clamp). Clamp Dinv here (not D upstream in WoodburyFactorization,
    // which is shared with the exact double CUDA/HIP/SYCL backends and must
    // keep the tight floor) so the float32 cancellation stays bounded by
    // ~eps_f32*1e6 ~ 0.1 instead of ~eps_f32*1e10 ~ 1e3. CG remains
    // mathematically valid for any SPD preconditioner, so this only makes
    // the preconditioner slightly less exact on those rows, never incorrect.
    arma::vec precDinvClamped = arma::clamp(precDinv, 0.0, 1e6);
    dDinv = std::make_unique<FBuf>(static_cast<std::size_t>(n), precDinvClamped.memptr());
    dMchol = std::make_unique<FBuf>(static_cast<std::size_t>(pk) * pk, precMcholLower.memptr());
    dZ = std::make_unique<FBuf>(mat);
    dPrecNc = std::make_unique<FBuf>(mat);
    dPrecKc = std::make_unique<FBuf>(static_cast<std::size_t>(pk) * ncols);
  }
  auto precondApply = [&](void* in, void* out) {
    lk_metal_precond_apply_launch(*dU, n, pk, *dDinv, *dMchol, in, ncols, out, *dPrecNc, *dPrecKc);
  };

  // rz_old is always computed ON DEVICE from the actual initial residual
  // (r=b when X0 is null, so <r,r> == |b|^2 exactly; with X0 given, r != b
  // in general, so a host-computed bnorm^2 shortcut would have been wrong).
  if (preconditioned) {
    precondApply(dR, *dZ);
    lk_metal_copy_dev(dP, *dZ, mat * sizeof(float));
    lk_metal_batched_dot_launch(dR, *dZ, n, ncols, dRzOld);
  } else {
    lk_metal_copy_dev(dP, dR, mat * sizeof(float));
    lk_metal_batched_dot_launch(dR, dR, n, ncols, dRzOld);
  }

  // active: a column starts inactive when b=0 (x=0 trivially solves it) OR
  // when X0 already meets tol for it (a good warm start needing zero
  // further iterations) -- the latter can only happen when X0 != nullptr.
  // Needs the true residual norm r.r, which is dRzOld in the
  // unpreconditioned case but is r.z (not r.r) once preconditioned, so
  // fetch it separately there.
  std::vector<double> resid_sq(ncols);
  if (preconditioned) {
    lk_metal_batched_dot_launch(dR, dR, n, ncols, dScratch);
    lk_metal_download_f32_as_f64(resid_sq.data(), dScratch, ncols);
  } else {
    lk_metal_download_f32_as_f64(resid_sq.data(), dRzOld, ncols);
  }
  std::vector<int> active(ncols);
  bool any = false;
  for (int c = 0; c < ncols; ++c) {
    const bool zero_b = (bnorm[c] == 0.0);
    const bool already_at_tol
        = (X0 != nullptr) && !zero_b && (std::sqrt(std::max(resid_sq[c], 0.0)) < eff_tol(c) * bnorm[c]);
    active[c] = (!zero_b && !already_at_tol) ? 1 : 0;
    any = any || active[c];
  }
  lk_metal_upload_i32(dActive, active.data(), ncols);

  constexpr arma::uword restart_every = 50, sync_every = 10;
  int host_flag = any ? 1 : 0;
  for (arma::uword it = 0; host_flag && it < max_iter; ++it) {
    matvec(dP, 0, dAp);
    lk_metal_batched_dot_launch(dP, dAp, n, ncols, dScratch);
    lk_metal_cg_alpha_launch(dRzOld, dScratch, ncols, dActive, dAlpha, dNegAlpha);
    lk_metal_batched_axpy_launch(dAlpha, dP, dX, n, ncols);

    if ((it + 1) % restart_every == 0) {
      matvec(dX, 0, dAp);
      lk_metal_copy_dev(dR, dB, mat * sizeof(float));
      lk_metal_batched_axpy_launch(dNegOnes, dAp, dR, n, ncols);
      lk_metal_batched_dot_launch(dR, dR, n, ncols, dScratch);
      if (preconditioned) {
        precondApply(dR, *dZ);
        lk_metal_batched_dot_launch(dR, *dZ, n, ncols, dScratch2);
        lk_metal_cg_restart_precond_launch(dScratch, dScratch2, dBnorm, dTol, ncols, dActive, dRzOld);
        lk_metal_copy_dev(dP, *dZ, mat * sizeof(float));
      } else {
        lk_metal_cg_restart_launch(dScratch, dBnorm, dTol, ncols, dActive, dRzOld);
        lk_metal_copy_dev(dP, dR, mat * sizeof(float));
      }
    } else {
      lk_metal_batched_axpy_launch(dNegAlpha, dAp, dR, n, ncols);
      lk_metal_batched_dot_launch(dR, dR, n, ncols, dScratch);
      if (preconditioned) {
        precondApply(dR, *dZ);
        lk_metal_batched_dot_launch(dR, *dZ, n, ncols, dScratch2);
        lk_metal_cg_beta_precond_launch(dScratch, dScratch2, dBnorm, dTol, ncols, dActive, dRzOld, dBeta);
        lk_metal_batched_update_p_launch(*dZ, dBeta, dP, n, ncols);
      } else {
        lk_metal_cg_beta_launch(dScratch, dBnorm, dTol, ncols, dActive, dRzOld, dBeta);
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

  // See CudaLinearAlgebra.cpp's matching readback: dActive still holds, per
  // column, whether the loop hit max_iter before that column reached tol
  // (a converged/deactivated column was already zeroed).
  std::vector<int> active_final(static_cast<std::size_t>(ncols));
  lk_metal_download_i32(active_final.data(), dActive, ncols);
  arma::uword n_unconverged = 0;
  for (int c = 0; c < ncols; ++c)
    if (active_final[static_cast<std::size_t>(c)])
      ++n_unconverged;
  if (n_unconverged_out != nullptr)
    *n_unconverged_out = n_unconverged;
  LinearAlgebra::cgNonConvergenceWarning(n_unconverged, static_cast<arma::uword>(ncols),
                                        static_cast<arma::uword>(n), max_iter);

  return X;
}

// Scalar-tol convenience overload for the common case (every column shares
// one tolerance) -- broadcasts into the per-column vector above. Matches
// LinearAlgebraCuda::conjugateGradient's matching overload.
arma::mat conjugateGradient(const arma::mat& Xt,
                            const arma::vec& theta,
                            const std::string& covType,
                            const arma::mat& B,
                            arma::uword max_iter,
                            double tol,
                            const arma::mat& precU,
                            const arma::vec& precDinv,
                            const arma::mat& precMcholLower,
                            arma::uword* n_unconverged_out,
                            const arma::mat* X0) {
  return conjugateGradient(Xt, theta, covType, B, max_iter, arma::vec(B.n_cols, arma::fill::value(tol)), precU,
                           precDinv, precMcholLower, n_unconverged_out, X0);
}

// Device-resident batched Lanczos -- see MetalLinearAlgebra.hpp's doc
// comment. Mechanical port of LinearAlgebraHip::stochasticLogDetBatched.
// d_V holds every probe's ENTIRE Krylov history for the whole call,
// step-major: step j's n x npr block lives at n*npr*j floats into the
// buffer (bound via lk_metal_rmul_batched_launch/lk_metal_batched_axpy_
// launch's byte-offset parameters) -- lanczos_reorth_dot/sub read that
// same layout directly (device-side offset math, see the .metal file).
double stochasticLogDetBatched(const arma::mat& Xt, const arma::vec& theta, const std::string& covType,
                               arma::uword lanczos_steps_in, const arma::mat& probes) {
  int kind;
  if (!covKind(covType, &kind))
    throw std::invalid_argument("LinearAlgebraMetal::stochasticLogDetBatched: unsupported covType '" + covType + "'");

  const int n = static_cast<int>(Xt.n_cols);
  const int dimX = static_cast<int>(Xt.n_rows);
  const int npr = static_cast<int>(probes.n_cols);
  const int ls = static_cast<int>(std::min<arma::uword>(lanczos_steps_in, Xt.n_cols));
  if (npr == 0 || ls == 0)
    return 0.0;

  std::vector<double> znorm(static_cast<std::size_t>(npr));
  std::vector<int> active_h(static_cast<std::size_t>(npr));
  arma::mat V0(static_cast<arma::uword>(n), static_cast<arma::uword>(npr), arma::fill::zeros);
  for (int p = 0; p < npr; ++p) {
    znorm[static_cast<std::size_t>(p)] = arma::norm(probes.col(static_cast<arma::uword>(p)));
    if (znorm[static_cast<std::size_t>(p)] != 0.0) {
      V0.col(static_cast<arma::uword>(p)) = probes.col(static_cast<arma::uword>(p)) / znorm[static_cast<std::size_t>(p)];
      active_h[static_cast<std::size_t>(p)] = 1;
    } else {
      active_h[static_cast<std::size_t>(p)] = 0;
    }
  }

  // Dense-R fast path (see DenseCovCache's doc comment): the whole SLQ
  // Lanczos recurrence below runs at a FIXED theta -- ls steps, each one
  // matvec -- so this reuses whatever conjugateGradient's earlier call (in
  // the SAME objective evaluation) already materialized, instead of
  // re-uploading Xt/theta and re-evaluating covariance transcendentals on
  // every one of those steps.
  DenseCovCache& cov = covCacheBind(Xt, theta, kind);
  void* dRmat = covCacheR(cov);
  auto matvec = [&](void* in, std::size_t in_byte_offset, void* out) {
    if (dRmat)
      lk_metal_dense_matvec_launch(dRmat, n, in, npr, out, n, static_cast<long>(in_byte_offset));
    else
      lk_metal_rmul_batched_launch(cov.d_Xt, n, dimX, cov.d_theta, kind, in, npr, out, nullptr,
                                   static_cast<long>(in_byte_offset));
  };

  const std::size_t step_elems = static_cast<std::size_t>(n) * npr;
  const std::size_t step_bytes = step_elems * sizeof(float);

  FBuf dV(step_elems * static_cast<std::size_t>(ls));
  lk_metal_memset_dev(dV, 0, step_elems * static_cast<std::size_t>(ls) * sizeof(float));
  lk_metal_upload_f64_as_f32(dV, V0.memptr(), step_elems);  // step 0's block == the whole buffer's start
  FBuf dW(step_elems);
  FBuf dDot(npr), dDot2(npr), dNegAlpha(npr), dInvBj(npr), dNegBetaPrev(npr);
  FBuf dT(static_cast<std::size_t>(ls) * npr);
  FBuf dAlphaAll(static_cast<std::size_t>(ls) * npr), dBetaAll(static_cast<std::size_t>(ls) * npr);
  IBuf dActive(npr, active_h.data());
  std::vector<int> m_eff_init(static_cast<std::size_t>(npr), ls);
  IBuf dMEff(npr, m_eff_init.data());

  for (int j = 0; j < ls; ++j) {
    const std::size_t vj_off = step_bytes * static_cast<std::size_t>(j);
    matvec(dV, vj_off, dW);
    if (j > 0) {
      const std::size_t vprev_off = step_bytes * static_cast<std::size_t>(j - 1);
      lk_metal_batched_axpy_launch(dNegBetaPrev, dV, dW, n, npr, static_cast<long>(vprev_off), 0);  // w -= beta_prev*v_prev
    }
    lk_metal_batched_dot_launch(dW, dV, n, npr, dDot, 0, static_cast<long>(vj_off));  // alpha_j = <w, Vj>
    lk_metal_lanczos_alpha_launch(dDot, npr, j, dActive, dAlphaAll, dNegAlpha);
    lk_metal_batched_axpy_launch(dNegAlpha, dV, dW, n, npr, static_cast<long>(vj_off), 0);  // w -= alpha_j*Vj

    const int m = j + 1;
    lk_metal_lanczos_reorth_dot_launch(dV, dW, n, npr, ls, m, dT);
    lk_metal_lanczos_reorth_sub_launch(dV, dT, n, npr, ls, m, dW);

    lk_metal_batched_dot_launch(dW, dW, n, npr, dDot2);  // bj^2
    const int is_last = (j + 1 == ls) ? 1 : 0;
    lk_metal_lanczos_beta_launch(dDot2, npr, j, is_last, dActive, dMEff, dBetaAll, dInvBj, dNegBetaPrev);

    if (j + 1 < ls) {
      const std::size_t vnext_off = step_bytes * static_cast<std::size_t>(j + 1);
      lk_metal_batched_axpy_launch(dInvBj, dW, dV, n, npr, 0, static_cast<long>(vnext_off));  // Vnext (==0) += inv_bj*w
    }
  }

  std::vector<double> alpha_h(static_cast<std::size_t>(ls) * npr), beta_h(static_cast<std::size_t>(ls) * npr);
  lk_metal_download_f32_as_f64(alpha_h.data(), dAlphaAll, alpha_h.size());
  lk_metal_download_f32_as_f64(beta_h.data(), dBetaAll, beta_h.size());
  std::vector<int> m_eff_h(static_cast<std::size_t>(npr));
  lk_metal_download_i32(m_eff_h.data(), dMEff, npr);

  // Same tail as LinearAlgebra::stochasticLogDetBatched: per-probe
  // tridiagonal eigendecomposition (O(nprobe*lanczos_steps^2), negligible
  // next to the matvecs above -- not worth a device port).
  double total = 0.0;
  for (int p = 0; p < npr; ++p) {
    if (znorm[static_cast<std::size_t>(p)] == 0.0)
      continue;
    const int me = m_eff_h[static_cast<std::size_t>(p)];
    arma::mat T(static_cast<arma::uword>(me), static_cast<arma::uword>(me), arma::fill::zeros);
    for (int jj = 0; jj < me; ++jj)
      T(static_cast<arma::uword>(jj), static_cast<arma::uword>(jj)) = alpha_h[static_cast<std::size_t>(jj) * npr + p];
    for (int jj = 0; jj + 1 < me; ++jj) {
      const double b = beta_h[static_cast<std::size_t>(jj) * npr + p];
      T(static_cast<arma::uword>(jj), static_cast<arma::uword>(jj + 1)) = b;
      T(static_cast<arma::uword>(jj + 1), static_cast<arma::uword>(jj)) = b;
    }
    arma::vec eigval;
    arma::mat eigvec;
    arma::eig_sym(eigval, eigvec, T);
    double quad = 0.0;
    for (int jj = 0; jj < me; ++jj) {
      const double lambda = std::max(eigval(static_cast<arma::uword>(jj)), LinearAlgebra::num_nugget);
      quad += eigvec(0, static_cast<arma::uword>(jj)) * eigvec(0, static_cast<arma::uword>(jj)) * std::log(lambda);
    }
    total += quad;
  }
  return (static_cast<double>(n) / static_cast<double>(npr)) * total;
}

}  // namespace LinearAlgebraMetal

#endif  // LIBKRIGING_USE_METAL_ITERATIVE
