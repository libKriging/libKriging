#ifndef LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_LINEARALGEBRA_HPP
#define LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_LINEARALGEBRA_HPP

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/libKriging_exports.h"

class LinearAlgebra {
 public:
  static arma::solve_opts::opts default_solve_opts;

  static double num_nugget;
  LIBKRIGING_EXPORT static void set_num_nugget(double nugget);
  LIBKRIGING_EXPORT static double get_num_nugget();

  static bool warn_chol;
  LIBKRIGING_EXPORT static void set_chol_warning(bool warn);

  static bool chol_rcond_check;
  LIBKRIGING_EXPORT static void check_chol_rcond(bool c);
  LIBKRIGING_EXPORT static bool chol_rcond_checked();

  static int max_inc_choldiag;
  LIBKRIGING_EXPORT static arma::mat safe_chol_lower(arma::mat X);
  static arma::mat safe_chol_lower_retry(arma::mat X, int warn);

  static double min_rcond;
  LIBKRIGING_EXPORT static double rcond_chol(arma::mat chol);
  static double min_rcond_approx;
  LIBKRIGING_EXPORT static double rcond_approx_chol(arma::mat chol);

  LIBKRIGING_EXPORT static arma::mat cholCov(arma::mat* R,
                                             const arma::mat& _dX,
                                             const arma::vec& _theta,
                                             std::function<double(const arma::vec&, const arma::vec&)> _Cov,
                                             const double factor,
                                             const arma::vec diag);
  LIBKRIGING_EXPORT static arma::mat update_cholCov(arma::mat* R,
                                                    const arma::mat& _dX,
                                                    const arma::vec& _theta,
                                                    std::function<double(const arma::vec&, const arma::vec&)> _Cov,
                                                    const double factor,
                                                    const arma::vec diag,
                                                    const arma::mat& T_old,
                                                    const arma::mat& R_old);

  LIBKRIGING_EXPORT static arma::mat chol_block(const arma::mat C, const arma::mat Loo);

  LIBKRIGING_EXPORT static arma::mat solve(const arma::mat& A, const arma::mat& B);

  LIBKRIGING_EXPORT static arma::mat rsolve(const arma::mat& A, const arma::mat& B);

  LIBKRIGING_EXPORT static arma::mat crossprod(const arma::mat& A);

  LIBKRIGING_EXPORT static arma::mat tcrossprod(const arma::mat& A);

  LIBKRIGING_EXPORT static arma::mat diagcrossprod(const arma::mat& A);

  LIBKRIGING_EXPORT static arma::colvec diagABA(const arma::mat& A, const arma::mat& B);

  // Fast pointer-based computation of pairwise differences
  // Computes dX where dX.col(i*n+j) = X.row(i) - X.row(j) for all i,j
  // Result is a (d x n*n) matrix where d = X.n_cols and n = X.n_rows
  LIBKRIGING_EXPORT static arma::mat compute_dX(const arma::mat& X);

  // Compute symmetric covariance matrix R from pre-computed differences dX
  // R[i,j] = R[j,i] = factor * Cov(dX.col(i*n+j), theta) for i < j
  // diag is set after factor multiplication
  LIBKRIGING_EXPORT static void covMat_sym_dX(arma::mat* R,
                                               const arma::mat& dX,
                                               const arma::vec& theta,
                                               std::function<double(const arma::vec&, const arma::vec&)> Cov,
                                               double factor = 1.0,
                                               const arma::vec& diag = arma::vec());

  // Compute symmetric covariance matrix R directly from X
  // R[i,j] = R[j,i] = factor * Cov(X.col(i) - X.col(j), theta) for i < j
  // X is assumed to be (d x n) with observations in columns
  LIBKRIGING_EXPORT static void covMat_sym_X(arma::mat* R,
                                              const arma::mat& X,
                                              const arma::vec& theta,
                                              std::function<double(const arma::vec&, const arma::vec&)> Cov,
                                              double factor = 1.0,
                                              const arma::vec& diag = arma::vec());

  // Compute rectangular covariance matrix R between X1 and X2
  // R[i,j] = factor * Cov(X1.col(i) - X2.col(j), theta)
  // X1 is (d x n1), X2 is (d x n2) with observations in columns
  LIBKRIGING_EXPORT static void covMat_rect(arma::mat* R,
                                             const arma::mat& X1,
                                             const arma::mat& X2,
                                             const arma::vec& theta,
                                             std::function<double(const arma::vec&, const arma::vec&)> Cov,
                                             double factor = 1.0);

  // Efficient computation of trace(A * B) = sum_i sum_j A(i,j) * B(j,i)
  // Avoids explicit matrix multiplication
  LIBKRIGING_EXPORT static double trace_prod(const arma::mat& A, const arma::mat& B);
};

#endif  // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_LINEARALGEBRA_HPP
