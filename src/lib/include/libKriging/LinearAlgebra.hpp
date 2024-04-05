#ifndef LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_LINEARALGEBRA_HPP
#define LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_LINEARALGEBRA_HPP

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/libKriging_exports.h"

class LinearAlgebra {
 public:
  static const arma::solve_opts::opts default_solve_opts;

  static float num_nugget;
  LIBKRIGING_EXPORT static void set_num_nugget(float nugget);
  LIBKRIGING_EXPORT static float get_num_nugget();

  static bool warn_chol;
  LIBKRIGING_EXPORT static void set_chol_warning(bool warn);

  static bool chol_rcond_check;
  LIBKRIGING_EXPORT static void check_chol_rcond(bool c);

  static int max_inc_choldiag;
  LIBKRIGING_EXPORT static arma::fmat safe_chol_lower(arma::fmat X);
  static arma::fmat safe_chol_lower(arma::fmat X, int warn);

  static float min_rcond;
  LIBKRIGING_EXPORT static float rcond_chol(arma::fmat chol);
  static float min_rcond_approx;
  LIBKRIGING_EXPORT static float rcond_approx_chol(arma::fmat chol);

  LIBKRIGING_EXPORT static arma::fmat cholCov(arma::fmat* R,
                                      const arma::fmat& _dX,
                                      const arma::fvec& _theta,
                                      std::function<float(const arma::fvec&, const arma::fvec&)> Cov,
                                      const float factor, const arma::fvec diag);
  LIBKRIGING_EXPORT static arma::fmat update_cholCov(arma::fmat* R,
                                        const arma::fmat& _dX,
                                        const arma::fvec& _theta, 
                                        std::function<float(const arma::fvec&, const arma::fvec&)> Cov,
                                        const float factor, const arma::fvec diag,
                                        const arma::fmat& Told);

  LIBKRIGING_EXPORT static arma::fmat chol_block(const arma::fmat C, const arma::fmat Loo, const arma::fmat Coo);

  LIBKRIGING_EXPORT static arma::fmat solve(const arma::fmat& A, const arma::fmat& B);

  LIBKRIGING_EXPORT static arma::fmat rsolve(const arma::fmat& A, const arma::fmat& B);

  LIBKRIGING_EXPORT static arma::fmat crossprod(const arma::fmat& A);

  LIBKRIGING_EXPORT static arma::fmat tcrossprod(const arma::fmat& A);

  LIBKRIGING_EXPORT static arma::fmat diagcrossprod(const arma::fmat& A);

  LIBKRIGING_EXPORT static arma::fcolvec diagABA(const arma::fmat& A, const arma::fmat& B);
};

#endif  // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_LINEARALGEBRA_HPP
