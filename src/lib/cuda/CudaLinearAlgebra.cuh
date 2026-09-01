#ifndef LIBKRIGING_SRC_LIB_CUDA_CUDALINEARALGEBRA_CUH
#define LIBKRIGING_SRC_LIB_CUDA_CUDALINEARALGEBRA_CUH

// Only compiled/declared when the project is configured with
// -DENABLE_CUDA_ITERATIVE=ON (see root CMakeLists.txt). Kept behind this
// macro so a default (non-CUDA) build never sees CUDA types/symbols, and
// LIBKRIGING_USE_CUDA_ITERATIVE is only ever defined by CMake, never
// auto-detected at configure time.
#ifdef LIBKRIGING_USE_CUDA_ITERATIVE

#include "libKriging/utils/lk_armadillo.hpp"

#include <string>

#include "libKriging/libKriging_exports.h"

// GPU counterpart of LinearAlgebra::conjugateGradient, scoped to the
// matrix-free R*v matvec used by LLIterative (Kriging::_logLikelihoodIterative)
// and predictIterative (KrigingImpl::predictIterative_impl). R is never
// materialized on the GPU either -- see CudaLinearAlgebra.cu's rmul_kernel --
// matching the CPU path's O(n) memory invariant. Only handles the
// no-preconditioner case; callers fall back to LinearAlgebra::conjugateGradient
// otherwise (unsupported covType, or a Nystrom preconditioner requested).
namespace LinearAlgebraCuda {

// True iff a CUDA device was found at runtime (lazy-initialized, cached).
// Independent of the compile-time flag: a build compiled with CUDA support
// can still run on a machine with no GPU, in which case this is false and
// callers must fall back to the CPU path.
LIBKRIGING_EXPORT bool available();

// Runtime on/off switch, defaulting to available(). Exists so the same
// binary can compare CPU vs GPU (e.g. in a benchmark) without recompiling.
LIBKRIGING_EXPORT bool enabled();
LIBKRIGING_EXPORT void set_enabled(bool value);

// Covariance kernels with a device-side implementation (see
// CudaLinearAlgebra.cu): "gauss", "exp", "matern3_2", "matern5_2".
LIBKRIGING_EXPORT bool supports(const std::string& covType);

// Matrix-free CG solve of R(Xt,theta)*Y = B (R = correlation matrix implied
// by Xt (d x n, one point per column), theta and covType). Solves every
// column of B in lockstep -- one batched matvec + a couple of batched
// reductions per iteration cover all columns at once (see
// CudaLinearAlgebraKernel.cu), rather than looping one column at a time --
// with the whole CG loop running on the GPU (single upload of Xt/theta/B,
// single download of the result, no host<->device round trip per
// iteration beyond the small per-column scalars CG's convergence check
// needs). A column that converges before others is frozen (its
// contribution to further updates zeroed) rather than dropped, since a
// batched launch can't cheaply shrink its own column count mid-loop. Same
// convergence contract as LinearAlgebra::conjugateGradient (relative
// residual < tol or max_iter iterations, periodic exact-residual restart).
LIBKRIGING_EXPORT arma::mat conjugateGradient(const arma::mat& Xt,
                                              const arma::vec& theta,
                                              const std::string& covType,
                                              const arma::mat& B,
                                              arma::uword max_iter,
                                              double tol = 1e-8);

}  // namespace LinearAlgebraCuda

#endif  // LIBKRIGING_USE_CUDA_ITERATIVE

#endif  // LIBKRIGING_SRC_LIB_CUDA_CUDALINEARALGEBRA_CUH
