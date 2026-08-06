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

  // Nystrom / partial-pivoted-Cholesky low-rank approximation of a covariance
  // matrix R (n x n, built lazily from X/_theta/_Cov), never materialized in
  // full and without ever building an O(n^2) pairwise-difference cube either
  // (unlike cholCov's _dX): R ~= U * U.t() + diag(*diag_resid), with U
  // (n x k_eff), k_eff <= k. X is (n x d), rows = observations (the m_X
  // convention, not cholCov's transposed _dX layout). Greedy pivoting
  // selects, at each step, the point with the largest residual variance
  // (Harbrecht et al. 2012); stops early if the max residual diagonal falls
  // below `tol` (k_eff < k). `landmarks_out`, if given, receives the selected
  // pivot row-indices (into the n points) in selection order. `diag` follows
  // cholCov's convention: empty = ones(n), else used verbatim.
  LIBKRIGING_EXPORT static arma::mat nystromFactor(arma::vec* diag_resid,
                                                   const arma::mat& X,
                                                   const arma::vec& _theta,
                                                   std::function<double(const arma::vec&, const arma::vec&)> _Cov,
                                                   double factor,
                                                   const arma::vec& diag,
                                                   arma::uword k,
                                                   double tol = 1e-12,
                                                   arma::uvec* landmarks_out = nullptr);

  // Solve (D + U*U.t()) * X = B via the Woodbury identity, without ever
  // materializing the n x n matrix D + U*U.t(). U is n x k (as returned by
  // nystromFactor), D is the strictly-positive diagonal (n). Cost O(n*k^2 + k^3)
  // instead of O(n^3) for a dense solve. Caller must ensure D > 0 (e.g. add a
  // jitter floor to nystromFactor's diag_resid beforehand).
  LIBKRIGING_EXPORT static arma::mat woodbury_solve(const arma::mat& U, const arma::vec& D, const arma::mat& B);

  // log|D + U*U.t()| via the matrix determinant lemma: log|D| + log|I_k + U.t() D^-1 U|.
  // Same complexity/preconditions as woodbury_solve.
  LIBKRIGING_EXPORT static double woodbury_logdet(const arma::mat& U, const arma::vec& D);

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

  // Matrix-free conjugate gradient solve of A*X = B, where the SPD matrix A
  // is applied only through the caller-supplied matrix-vector product
  // `Amul` -- A itself is never materialized (O(n) memory instead of
  // O(n^2)). Solves each column of B independently (no block-CG sharing of
  // Krylov subspaces across columns). Stops per-column when the relative
  // residual norm(A*x-b)/norm(b) drops below `tol`, or after `max_iter`
  // iterations (in exact arithmetic, CG converges in at most n iterations;
  // `max_iter` is typically set to n or a smaller early-stopping budget).
  // Trades O(n^2) storage for O(n^2 * iters) compute per column, vs a single
  // O(n^2) dense triangular solve from a precomputed Cholesky factor -- only
  // worthwhile when that factor either doesn't exist or isn't kept in memory.
  LIBKRIGING_EXPORT static arma::mat conjugateGradient(const std::function<arma::vec(const arma::vec&)>& Amul,
                                                       const arma::mat& B,
                                                       arma::uword max_iter,
                                                       double tol = 1e-8);

  LIBKRIGING_EXPORT static arma::mat solve_lower(const arma::mat& L, const arma::mat& B);
  LIBKRIGING_EXPORT static arma::mat solve_upper(const arma::mat& U, const arma::mat& B);
  LIBKRIGING_EXPORT static arma::mat rsolve_upper(const arma::mat& U, const arma::mat& B);
  LIBKRIGING_EXPORT static arma::mat inv_sympd(const arma::mat& L);
  LIBKRIGING_EXPORT static arma::mat chol_upper(const arma::mat& A);
  LIBKRIGING_EXPORT static void qr_econ(arma::mat& Q, arma::mat& R, const arma::mat& A);
};

#endif  // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_LINEARALGEBRA_HPP
