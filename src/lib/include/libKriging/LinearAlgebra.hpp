#ifndef LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_LINEARALGEBRA_HPP
#define LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_LINEARALGEBRA_HPP

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/libKriging_exports.h"

#include <functional>
#include <vector>

class LinearAlgebra {
 public:
  static arma::solve_opts::opts default_solve_opts;

  static double num_nugget;
  LIBKRIGING_EXPORT static void set_num_nugget(double nugget);
  LIBKRIGING_EXPORT static double get_num_nugget();

  static bool warn_chol;
  LIBKRIGING_EXPORT static void set_chol_warning(bool warn);

  // Whether conjugateGradient(Batched) prints a [WARNING] when a solve hits
  // max_iter without every column reaching tol (see cgNonConvergenceWarning).
  // Defaults to true (unlike warn_chol): a non-converged iterative solve is
  // a silent-by-default correctness issue -- log-likelihood, gradient and
  // predictions all become quietly wrong -- not a routine numerical detail,
  // so this stays on unless a caller deliberately opts out (e.g. a
  // benchmark exploring a known-non-converging regime on purpose).
  LIBKRIGING_EXPORT static bool warn_cg;
  LIBKRIGING_EXPORT static void set_cg_warning(bool warn);

  // Shared by every conjugateGradient(Batched) implementation -- CPU and
  // every GPU backend (CUDA/HIP/SYCL/Metal) -- so the same message/format
  // is used everywhere a CG solve can silently under-converge. Prints
  // nothing when n_unconverged == 0 or warn_cg is false.
  LIBKRIGING_EXPORT static void cgNonConvergenceWarning(arma::uword n_unconverged,
                                                        arma::uword ncols,
                                                        arma::uword n,
                                                        arma::uword max_iter);

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

  // One-time O(n*k^2 + k^3) factorization of D + U*U.t() (Woodbury), reused
  // afterward for many O(n*k + k^2) solves via solve(). Exists specifically
  // for the CG-preconditioner use case (Kriging::_logLikelihoodIterative,
  // KrigingImpl::predictIterative_impl): those call the preconditioner's
  // apply once PER CG ITERATION, so building a fresh woodbury_solve() call
  // each time -- which redoes the full O(n*k^2+k^3) factorization on every
  // single apply -- makes each "cheap" preconditioner application cost as
  // much as (or more than) the O(n^2) matvec it's meant to make cheaper to
  // skip, silently erasing any iteration-count benefit from preconditioning
  // (found by profiling: a 79s unpreconditioned predictIterative solve
  // became 137s "preconditioned" at n=500, k=50, despite genuinely
  // converging in fewer iterations -- see git history). Build ONE of these
  // per CG solve (not per iteration) and call solve() from the Pinv
  // closure instead.
  class WoodburyFactorization {
   public:
    LIBKRIGING_EXPORT WoodburyFactorization(const arma::mat& U, const arma::vec& D);
    LIBKRIGING_EXPORT arma::mat solve(const arma::mat& B) const;

    // L^{-1} * B  (whitenL) and L^{-T} * B  (whitenLt) for a square factor L
    // of P = D + U*U.t() with L*L.t() == P (L is not symmetric or
    // triangular). Each applies in O(n*k) after a shared O(n*k^2 + k^3)
    // one-time setup (built lazily) -- a thin-QR + k x k eigendecomposition
    // of the low-rank structure, never an n x n factor. Lets
    // Kriging::_logLikelihoodIterative run its SLQ log-determinant on the
    // symmetric whitened operator Rtilde = L^{-1} R L^{-T}, whose SLQ
    // estimate is log|R| - log|P| (log|P| added back via woodbury_logdet) --
    // so the Nystrom preconditioner tightens the log-determinant, not just
    // the CG solves.
    LIBKRIGING_EXPORT arma::mat whitenL(const arma::mat& B) const;
    LIBKRIGING_EXPORT arma::mat whitenLt(const arma::mat& B) const;

    // Read-only access to the factors, so a matching preconditioner apply
    // can be run elsewhere (e.g. on the GPU, in LinearAlgebraCuda's
    // preconditioned CG) without re-deriving safe_chol_lower(M) and risking
    // a subtly different jittering.
    LIBKRIGING_EXPORT const arma::mat& U() const { return m_U; }
    LIBKRIGING_EXPORT const arma::vec& Dinv() const { return m_Dinv; }
    LIBKRIGING_EXPORT const arma::mat& McholLower() const { return m_M_chol_lower; }

   private:
    void ensure_whiten_factors() const;  // lazily builds the whitenL/whitenLt factors

    arma::mat m_U;   // the n x k Nystrom factor, kept for U()/GPU apply
    arma::vec m_Dinv;
    arma::mat m_Ut;  // U.t(), kept separately from m_DinvU: solve()'s rhs needs U.t()*DinvB, not DinvU.t()*DinvB
    arma::mat m_DinvU;
    arma::mat m_M_chol_lower;
    // whitenL()/whitenLt()'s lazily-built low-rank factors: K^{-1/2} acts as
    // I + Q*m_whiten_core*Q', then scaled by D^{-1/2} (mutable: a pure
    // cache, the accessors stay logically const).
    mutable arma::vec m_whiten_Dinvhalf;
    mutable arma::mat m_whiten_Q;
    mutable arma::mat m_whiten_core;
  };

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
  // Optional `Pinv` applies an approximate inverse of a preconditioner M
  // (M^-1 * v) to accelerate convergence on ill-conditioned A -- e.g.
  // LinearAlgebra::woodbury_solve bound to a Nystrom factor of A itself.
  // Left empty (default), this reduces to plain CG. Standard
  // preconditioned-CG recurrence (z = Pinv(r) replaces r in the
  // Fletcher-Reeves ratio and search direction); the same periodic
  // exact-residual restart as plain CG applies here too.
  LIBKRIGING_EXPORT static arma::mat conjugateGradient(const std::function<arma::vec(const arma::vec&)>& Amul,
                                                       const arma::mat& B,
                                                       arma::uword max_iter,
                                                       double tol = 1e-8,
                                                       const std::function<arma::vec(const arma::vec&)>& Pinv
                                                       = std::function<arma::vec(const arma::vec&)>());

  // Block CG with a SHARED matvec: same per-column convergence contract as
  // conjugateGradient (each column an independent Krylov solve, no block-CG
  // subspace sharing), but every active column is advanced in lockstep so
  // `AmulBatched` / `PinvBatched` are called once per iteration on the whole
  // n x ncols block -- one covariance sweep instead of ncols for LLIterative's
  // matrix-free R*V. Used for LLIterative's [F|y] and probe solves.
  // n_unconverged_out, when non-null, receives the number of B's columns
  // that still hadn't reached tol when the loop hit max_iter (0 = every
  // column converged). A [WARNING] is also printed via
  // cgNonConvergenceWarning whenever that count is > 0 (see warn_cg).
  LIBKRIGING_EXPORT static arma::mat conjugateGradientBatched(
      const std::function<arma::mat(const arma::mat&)>& AmulBatched,
      const arma::mat& B,
      arma::uword max_iter,
      double tol = 1e-8,
      const std::function<arma::mat(const arma::mat&)>& PinvBatched
      = std::function<arma::mat(const arma::mat&)>(),
      arma::uword* n_unconverged_out = nullptr);

  // Stochastic Lanczos Quadrature (SLQ) estimate of log|A| for an SPD matrix
  // A of size n, given only as a matrix-vector product `Amul` -- A itself is
  // never materialized (Ubaru, Chen & Saad 2017). For `nprobe` independent
  // Rademacher probe vectors z_i (entries +-1, so z_i.t()*z_i = n exactly),
  // an `lanczos_steps`-step Lanczos tridiagonalization of A starting from
  // z_i (full reorthogonalization against all prior Lanczos vectors, since
  // `lanczos_steps` is meant to stay modest relative to n) gives a small
  // tridiagonal T_i whose eigendecomposition V*diag(lambda)*V.t() yields
  // z_i.t()*log(A)*z_i ~= n * sum_j V(0,j)^2 * log(lambda_j); averaging
  // over probes and using trace(log(A)) = log|A| gives the estimate. Cost
  // O(nprobe * lanczos_steps * n^2) (matvec-dominated) instead of O(n^3)
  // for an exact Cholesky-based log-determinant -- the same idea GPyTorch's
  // BBMM uses for its own log-determinant term. `seed` makes repeated calls
  // with the same probes reproducible (needed for a smooth objective across
  // nearby theta evaluations during optimization).
  LIBKRIGING_EXPORT static double stochasticLogDet(const std::function<arma::vec(const arma::vec&)>& Amul,
                                                   arma::uword n,
                                                   arma::uword nprobe,
                                                   arma::uword lanczos_steps,
                                                   const arma::mat& probes);

  // Same estimator, every probe's Lanczos advanced in lockstep so the
  // caller's `AmulBatched` is invoked once per Lanczos step on the whole
  // n x nprobe block instead of nprobe times on single vectors -- lets the
  // matvec (and hence the SLQ log-determinant) run batched on the GPU. See
  // LinearAlgebra.cpp and docs/math/Iterative.md.
  LIBKRIGING_EXPORT static double stochasticLogDetBatched(
      const std::function<arma::mat(const arma::mat&)>& AmulBatched,
      arma::uword n,
      arma::uword nprobe,
      arma::uword lanczos_steps,
      const arma::mat& probes);

  // Generates `nprobe` Rademacher (+-1 entries) probe vectors of length n,
  // as columns of an n x nprobe matrix -- meant to be generated ONCE (fixed
  // seed) and reused across every theta evaluation of an iterative
  // objective, exactly like LLNystrom's fixed landmarks: re-drawing fresh
  // probes at every evaluation would make the objective noisy/non-smooth
  // between optimizer iterations.
  LIBKRIGING_EXPORT static arma::mat rademacherProbes(arma::uword n, arma::uword nprobe, unsigned seed);

  LIBKRIGING_EXPORT static arma::mat solve_lower(const arma::mat& L, const arma::mat& B);
  LIBKRIGING_EXPORT static arma::mat solve_upper(const arma::mat& U, const arma::mat& B);
  LIBKRIGING_EXPORT static arma::mat rsolve_upper(const arma::mat& U, const arma::mat& B);
  LIBKRIGING_EXPORT static arma::mat inv_sympd(const arma::mat& L);
  LIBKRIGING_EXPORT static arma::mat chol_upper(const arma::mat& A);
  LIBKRIGING_EXPORT static void qr_econ(arma::mat& Q, arma::mat& R, const arma::mat& A);
};

#endif  // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_LINEARALGEBRA_HPP
