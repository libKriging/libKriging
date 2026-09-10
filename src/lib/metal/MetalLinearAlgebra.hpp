#ifndef LIBKRIGING_SRC_LIB_METAL_METALLINEARALGEBRA_HPP
#define LIBKRIGING_SRC_LIB_METAL_METALLINEARALGEBRA_HPP

// UNVERIFIED Apple-Metal port of src/lib/cuda/CudaLinearAlgebra.cuh -- see
// src/lib/hip/HipLinearAlgebra.hpp for the "never compiled or run" caveat
// (no macOS / Metal toolchain here), which applies here with one EXTRA,
// load-bearing caveat:
//
//   Metal Shading Language has NO double on Apple-Silicon GPUs (M1/M2/M3
//   and later are float32 / float16 only). So this backend runs the entire
//   iterative path -- the matrix-free R*v matvec, the CG recurrence, the
//   SLQ Lanczos, the Woodbury apply -- in FLOAT32. That is a genuine
//   numerical difference from the double CUDA/HIP/SYCL backends, not a
//   mechanical detail: CG on an ill-conditioned interpolation kernel loses
//   digits fast in single precision, so predictIterative's `tol` floors
//   itself at ~1e-4 here (see MetalLinearAlgebra.cpp) and the SLQ
//   log-determinant is correspondingly noisier. Treat this as a
//   proof-of-portability skeleton that needs a real design review (mixed
//   precision? Kahan sums? refuse ill-conditioned theta?) before use, not
//   as a drop-in equal of the other backends.
//
// Only compiled/declared with -DENABLE_METAL_ITERATIVE=ON (macOS + Metal +
// metal-cpp headers via -DMETAL_CPP_DIR). LIBKRIGING_USE_METAL_ITERATIVE is
// only ever defined by CMake.
#ifdef LIBKRIGING_USE_METAL_ITERATIVE

#include "libKriging/utils/lk_armadillo.hpp"

#include <string>

#include "libKriging/libKriging_exports.h"

// Same public API as LinearAlgebraCuda / LinearAlgebraHip / LinearAlgebraSycl
// (double at the boundary; the float32 truncation happens inside). See the
// CUDA header for the per-function contract.
namespace LinearAlgebraMetal {

LIBKRIGING_EXPORT bool available();
LIBKRIGING_EXPORT bool enabled();
LIBKRIGING_EXPORT void set_enabled(bool value);
LIBKRIGING_EXPORT bool supports(const std::string& covType);

LIBKRIGING_EXPORT arma::mat conjugateGradient(const arma::mat& Xt,
                                              const arma::vec& theta,
                                              const std::string& covType,
                                              const arma::mat& B,
                                              arma::uword max_iter,
                                              double tol = 1e-8,
                                              const arma::mat& precU = arma::mat(),
                                              const arma::vec& precDinv = arma::vec(),
                                              const arma::mat& precMcholLower = arma::mat());

LIBKRIGING_EXPORT arma::mat rmulBatched(const arma::mat& Xt,
                                        const arma::vec& theta,
                                        const std::string& covType,
                                        const arma::mat& V);

LIBKRIGING_EXPORT arma::mat dRmulBatched(const arma::mat& Xt,
                                         const arma::vec& theta,
                                         const std::string& covType,
                                         const arma::mat& V);

constexpr int kMaxDimX = 32;

}  // namespace LinearAlgebraMetal

#endif  // LIBKRIGING_USE_METAL_ITERATIVE

#endif  // LIBKRIGING_SRC_LIB_METAL_METALLINEARALGEBRA_HPP
