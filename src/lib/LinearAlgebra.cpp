// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio
#include <cmath>
// clang-format on

#include "libKriging/LinearAlgebra.hpp"

#include <thread>
#ifdef _OPENMP
#include <omp.h>

// Helper function to safely get optimal thread count
// Windows MSVC OpenMP can sometimes return unexpected values
inline int get_optimal_threads(int max_default = 2) {
  int max_threads = omp_get_max_threads();
  // Defensive: ensure we get a valid thread count
  if (max_threads <= 0) {
    return 1;  // Fallback to serial if OpenMP returns invalid value
  }
  return (max_threads > max_default) ? max_default : max_threads;
}
#endif

#include "libKriging/Bench.hpp"
#include "libKriging/Covariance.hpp"
#include "libKriging/utils/lk_armadillo.hpp"

#include <mutex>

arma::solve_opts::opts LinearAlgebra::default_solve_opts = arma::solve_opts::fast + arma::solve_opts::no_approx;

double LinearAlgebra::num_nugget = 1E-10;

LIBKRIGING_EXPORT void LinearAlgebra::set_num_nugget(double nugget) {
  LinearAlgebra::num_nugget = nugget;
};

LIBKRIGING_EXPORT double LinearAlgebra::get_num_nugget() {
  return LinearAlgebra::num_nugget;
};

LIBKRIGING_EXPORT arma::mat LinearAlgebra::safe_chol_lower(arma::mat X) {
  return LinearAlgebra::safe_chol_lower_retry(X, 0);
}

LIBKRIGING_EXPORT bool LinearAlgebra::warn_chol = false;

LIBKRIGING_EXPORT void LinearAlgebra::set_chol_warning(bool warn) {
  LinearAlgebra::warn_chol = warn;
};

LIBKRIGING_EXPORT bool LinearAlgebra::chol_rcond_check = true;

LIBKRIGING_EXPORT void LinearAlgebra::check_chol_rcond(bool c) {
  LinearAlgebra::chol_rcond_check = c;
};

LIBKRIGING_EXPORT bool LinearAlgebra::chol_rcond_checked() {
  return LinearAlgebra::chol_rcond_check;
};

int LinearAlgebra::max_inc_choldiag = 10;

// Recursive turn-around for ill-condition of correlation matrix. Used in *Kriging::fit & *Kriging::simulate
//' @ref: Andrianakis, I. and Challenor, P. G. (2012). The effect of the nugget on Gaussian pro-cess emulators of
// computer models. Comput. Stat. Data Anal., 56(12):4215–4228.
arma::mat LinearAlgebra::safe_chol_lower_retry(arma::mat X, int inc_cond) {
  arma::mat L = arma::mat(X.n_rows, X.n_cols, arma::fill::none);
  // auto t0 = Bench::tic();
  bool ok = arma::chol(L, X, "lower");
  // t0 = Bench::toc(nullptr, "        arma::chol" ,t0);
  bool wrong_rcond = LinearAlgebra::chol_rcond_check;
  if (ok) {
    // wrong_rcond = wrong_rcond && (LinearAlgebra::rcond_approx_chol(L) < LinearAlgebra::min_rcond_approx);
    ////t0 = Bench::toc(nullptr, "        rcond_approx" ,t0);
    wrong_rcond = wrong_rcond && (LinearAlgebra::rcond_chol(L) < LinearAlgebra::min_rcond);
    // t0 = Bench::toc(nullptr, "        rcond" ,t0);
  }
  if (!ok || wrong_rcond) {
    if (inc_cond > max_inc_choldiag) {
      throw std::runtime_error("[ERROR] Exceed max numerical nugget (" + std::to_string(inc_cond) + " x 1e"
                               + std::to_string(log10(LinearAlgebra::num_nugget)) + ") added to force chol matrix");
    } else if (LinearAlgebra::num_nugget <= 0.0) {
      throw std::runtime_error("[ERROR] Cannot add numerical nugget which is not strictly positive: "
                               + std::to_string(LinearAlgebra::num_nugget));
    } else {
      X.diag() += LinearAlgebra::num_nugget * std::pow(10, inc_cond);
      return LinearAlgebra::safe_chol_lower_retry(X, inc_cond + 1);
    }
    // t0 = Bench::toc(nullptr, "        inc_cond" ,t0);
  } else {
    if (warn_chol && (inc_cond > 0)) {
      arma::cout << "[WARNING] Added " << LinearAlgebra::num_nugget << " * 10^" << inc_cond << " numerical nugget to force Cholesky decomposition" << arma::endl;
    }
    return L;
  }
}

double LinearAlgebra::min_rcond = 1e-18;

// MKL's dgecon (called by arma::rcond) is not thread-safe when invoked from
// multiple std::threads concurrently. Serialize all rcond calls with a mutex.
static std::mutex s_rcond_mutex;

LIBKRIGING_EXPORT double LinearAlgebra::rcond_chol(arma::mat chol) {
  std::lock_guard<std::mutex> lock(s_rcond_mutex);
  double rcond = arma::rcond(chol);
  rcond *= rcond;
  if (warn_chol)
    if (rcond < (chol.n_rows * min_rcond)) {
      arma::cout << "[WARNING] rcond " << rcond << " is below minimal value." << arma::endl;
    }
  return rcond;
}

double LinearAlgebra::min_rcond_approx = 1e-10;
// Proxy to arma::rcond
// @ref: N. J. Higham, "A survey of condition number estimation for triangular matrices," SIAM Review, vol. 29, no. 4,
// pp. 575–596, Dec. 1987.
LIBKRIGING_EXPORT double LinearAlgebra::rcond_approx_chol(arma::mat chol) {
  double m = chol.at(0, 0);
  double M = chol.at(0, 0);
  if (chol.n_rows > 1)
    for (arma::uword i = 1; i < chol.n_rows; i++) {
      double cii = chol.at(i, i);
      if (cii < m) {
        m = cii;
      } else if (cii > M) {
        M = cii;
      }
    }
  double rcond = m / M;
  rcond = rcond * rcond;
  if (warn_chol)
    if (rcond < (chol.n_rows * min_rcond_approx)) {
      arma::cout << "[WARNING] rcond_approx " << rcond << " is below minimal value." << arma::endl;
    }
  return rcond;
}

LIBKRIGING_EXPORT arma::mat LinearAlgebra::cholCov(arma::mat* R,
                                                   const arma::mat& _dX,
                                                   const arma::vec& _theta,
                                                   std::function<double(const arma::vec&, const arma::vec&)> _Cov,
                                                   const double factor,
                                                   const arma::vec diag) {
  arma::uword n = (*R).n_rows;

  // auto t0 = Bench::tic();
  for (arma::uword i = 0; i < n; i++) {
    //(*R).at(i, i) = 1.0;
    for (arma::uword j = 0; j < i; j++) {
      (*R).at(i, j) = (*R).at(j, i) = _Cov(_dX.col(i * n + j), _theta);
    }
  }
  (*R) *= factor;  // !!! requires that diag is setup after
  // t0 = Bench::toc(nullptr, "    _Cov: " + std::to_string(n) + "/" + std::to_string(n),t0);

  // Slower:
  // std::vector<std::thread> col_threads(n);
  // std::vector<arma::colvec> col_vecs(n);
  // for (arma::uword i = 0; i < n; i++) {
  //   col_threads[i] = std::thread([i, &col_vecs, _dX, _theta, _Cov, factor, n](){
  //     arma::colvec col_vecs_i = arma::colvec(n,arma::fill::none);
  //     for (arma::uword j = 0; j < n; j++) {
  //       col_vecs_i.at(j) = _Cov(_dX.col(i * n + j), _theta) * factor;
  //     }
  //     col_vecs[i] = col_vecs_i;
  //   });
  //   //(*R).at(i, j) = (*R).at(j, i) = _Cov(_dX.col(i * n + j), _theta) * factor;
  //   ////t0 = Bench::toc(nullptr, "    _Cov: " + std::to_string(i) + "/" + std::to_string(n),t0);
  // }
  // for (arma::uword i = 0; i < n; i++) {
  //   col_threads[i].join();
  //   (*R).col(i) = col_vecs[i];
  // }
  // //t0 = Bench::toc(nullptr, "    _Cov (threads): " + std::to_string(n) + "/" + std::to_string(n),t0);

  // Same speed:
  // #pragma omp parallel for shared(*R)
  //  for (arma::uword i = 0; i < n; i++) {
  //    (*R).at(i, i) = 1.0;
  //    for (arma::uword j = 0; j < i; j++) {
  //      (*R).at(i, j) = (*R).at(j, i) = _Cov(_dX.col(i * n + j), _theta) * factor;
  //    }
  //  }
  //  //t0 = Bench::toc(nullptr, "    _Cov (omp): " + std::to_string(n) + "/" + std::to_string(n),t0);

  if (diag.n_elem == 0) {
    (*R).diag().ones();  //(*R).diag() = arma::vec(n, arma::fill::ones);
  } else {
    (*R).diag() = diag;
  }
  // t0 = Bench::toc(nullptr, "    _Cov: diag",t0);

  // Cholesky decompostion of covariance matrix

  arma::mat L
      = LinearAlgebra::safe_chol_lower(*R);  // Do NOT trimatl T (slower because copy): trimatl(chol(R, "lower"));
  // t0 = Bench::toc(nullptr, "    Chol",t0);

  return L;
}

LIBKRIGING_EXPORT arma::mat LinearAlgebra::update_cholCov(
    arma::mat* R,
    const arma::mat& _dX,
    const arma::vec& _theta,
    std::function<double(const arma::vec&, const arma::vec&)> _Cov,
    const double factor,
    const arma::vec diag,
    const arma::mat& T_old,
    const arma::mat& R_old) {
  arma::uword n_old = T_old.n_rows;
  arma::uword n = (*R).n_rows;

  // auto t0 = Bench::tic();
  (*R).submat(0, 0, n_old - 1, n_old - 1)
      = R_old;  // T_old * T_old.t();// hope that does not cost too much... (we dont save previous R)
  // t0 = Bench::toc(nullptr, "    _Cov: restore old",t0);
  for (arma::uword i = n_old; i < n; i++) {
    for (arma::uword j = 0; j < i; j++) {
      (*R).at(i, j) = (*R).at(j, i) = _Cov(_dX.col(i * n + j), _theta);
    }
    ////t0 = Bench::toc(nullptr, "    _Cov: " + std::to_string(i) + "/" + std::to_string(n),t0);
  }
  //(*R).submat(n_old, n_old, n-1, n-1) *= factor; // !!! requires that diag is setup after
  (*R).submat(n_old, 0, n - 1, n_old - 1) *= factor;
  (*R).submat(0, n_old, n - 1, n - 1) *= factor;
  // t0 = Bench::toc(nullptr, "    _Cov: " + std::to_string(n) + "/" + std::to_string(n),t0);

  if (diag.n_elem == 0) {
    (*R).diag().ones();  //(*R).diag() = arma::vec(n, arma::fill::ones);
  } else {
    (*R).diag() = diag;
  }
  // t0 = Bench::toc(nullptr, "    _Cov: diag",t0);

  arma::mat L = LinearAlgebra::chol_block(*R, T_old);
  // t0 = Bench::toc(nullptr, "    Chol Block",t0);

  return L;
}

// We want to compute the cholesky root of C, knowing the cholesky root of a block of C: Co,o.
// We use the notation of the doc "libKriging" update by Yves Deville (o=old, u=new):
// C = | Co,o Co,u | = | Lo,o   0  | | Lo,o^T Lu,o^T | = | Lo,o*Lo,o^T          Lo,o*Lu,o^T        |
//     | Cu,o Cu,u |   | Lu,o Lu,u | |   0    Lu,u^T |   | Lu,o*Lo,o^T   Lu,o*Lu,o^T + Lu,u*Lu,u^T |
// so, by id.:
//   Lo,o is the cholesky root of Co,o (known)
//   Lu,o = Cu,o Lo,o^-T
//   Lu,u is the cholesky root of Cu,u - Lu,o Lu,o^T
LIBKRIGING_EXPORT arma::mat LinearAlgebra::chol_block(const arma::mat C, const arma::mat Loo) {
  arma::uword n = C.n_rows;
  arma::uword no = Loo.n_rows;  // old size. n-1 if we just add one observation.

  // auto t0 = Bench::tic();
  // arma::mat Cuo = C.submat(no, 0,  n-1, no-1);
  ////t0 = Bench::toc(nullptr, "        >Cuo",t0);
  arma::mat Cou = C.submat(0, no, no - 1, n - 1);
  // t0 = Bench::toc(nullptr, "        >Cou",t0);
  arma::mat Cuu = C.submat(no, no, n - 1, n - 1);
  // t0 = Bench::toc(nullptr, "        >Cuu",t0);

  arma::mat L = arma::mat(n, n, arma::fill::none);
  L.submat(0, 0, no - 1, no - 1) = Loo;
  // t0 = Bench::toc(nullptr, "        <Loo",t0);
  // arma::mat Luo = Cuo * arma::solve( Loo, arma::eye<arma::mat>(no, no) ).t(); // Lu,o = Cu,o Lo,o^-T
  ////t0 = Bench::toc(nullptr, "        Luo = Cuo / Loo.t()",t0);
  arma::mat Lou = LinearAlgebra::solve(Loo, Cou);
  // t0 = Bench::toc(nullptr, "        Lou = Loo \\ Cou ",t0);
  L.submat(no, 0, n - 1, no - 1) = Lou.t();  // Luo;
  // t0 = Bench::toc(nullptr, "        <Luo",t0);
  // Lu,u = chol( Cu,u - Lu,o Lu,o^T ). Forming this Schur complement
  // explicitly can lose positive-definiteness through catastrophic
  // cancellation when the new points are (numerically) almost perfectly
  // predictable from the old ones -- i.e. their conditional variance
  // given the old block is genuinely tiny -- even though the full matrix
  // C is still safely factorizable directly (a monolithic Cholesky
  // propagates rounding error more gracefully than this two-step
  // block/Schur-complement computation). If the block update still fails
  // after safe_chol_lower's own nugget-retry budget is exhausted, fall
  // back to a full from-scratch factorization of C instead of throwing.
  try {
    L.submat(no, no, n - 1, n - 1) = LinearAlgebra::safe_chol_lower(Cuu - LinearAlgebra::crossprod(Lou));
    // t0 = Bench::toc(nullptr, "        <Luu = chol( Cuu - Luo * Luo.t() )",t0);
  } catch (const std::exception&) {
    if (warn_chol)
      arma::cout << "[WARNING] Cholesky block update failed (near-singular conditional block); "
                 << "falling back to a full Cholesky factorization" << arma::endl;
    return arma::trimatl(LinearAlgebra::safe_chol_lower(C));
  }

  arma::mat lowL = arma::trimatl(L);
  // t0 = Bench::toc(nullptr, "        trimatl L",t0);

  return lowL;
}

// Greedy partial-pivoted Cholesky (Harbrecht, Peters & Schneider 2012), a.k.a.
// the Nystrom approximation used e.g. as GPyTorch's pivoted-Cholesky
// preconditioner. At each step, picks the point with the largest residual
// (uncaptured) diagonal variance as the next pivot/landmark, appends a column
// to U so that U*U.t() matches R exactly on the pivot's row/column, and
// deflates the residual diagonal accordingly. Never touches an off-diagonal
// entry of R that isn't on a pivot column, and never materializes an n x n
// matrix or an O(n^2) pairwise-difference cube (unlike cholCov's _dX):
// covariance values are computed on demand, row by row, straight from X. Cost
// O(n*k) covariance evaluations and O(n*k^2) flops total for rank k, vs O(n^2)
// to build R and O(n^3) to factorize it exactly.
LIBKRIGING_EXPORT arma::mat LinearAlgebra::nystromFactor(arma::vec* diag_resid,
                                                         const arma::mat& X,
                                                         const arma::vec& _theta,
                                                         std::function<double(const arma::vec&, const arma::vec&)> _Cov,
                                                         const double factor,
                                                         const arma::vec& diag,
                                                         arma::uword k,
                                                         const double tol,
                                                         arma::uvec* landmarks_out) {
  const arma::uword n = X.n_rows;
  k = std::min(k, n);
  const arma::vec diag0 = (diag.n_elem == 0) ? arma::vec(n, arma::fill::ones) : diag;

  arma::vec d = diag0;  // current residual (uncaptured) diagonal
  arma::mat U(n, k, arma::fill::none);
  arma::uvec landmarks(k, arma::fill::zeros);

  arma::uword k_eff = 0;
  for (arma::uword t = 0; t < k; ++t) {
    const arma::uword piv = d.index_max();
    const double dpiv = d(piv);
    if (dpiv < tol)
      break;
    landmarks(t) = piv;

    arma::vec col(n, arma::fill::none);
    for (arma::uword i = 0; i < n; ++i)
      col(i) = (i == piv) ? diag0(piv) : factor * _Cov(arma::trans(X.row(i) - X.row(piv)), _theta);

    if (t > 0)
      col -= U.cols(0, t - 1) * U.row(piv).cols(0, t - 1).t();

    const double s = std::sqrt(std::max(dpiv, tol));
    U.col(t) = col / s;

    d -= arma::square(U.col(t));
    d(piv) = 0.0;
    d.transform([](double v) { return v < 0 ? 0.0 : v; });

    ++k_eff;
  }

  if (k_eff < k)
    U = (k_eff == 0) ? arma::mat(n, 0) : arma::mat(U.cols(0, k_eff - 1));

  if (landmarks_out)
    *landmarks_out = landmarks.head(k_eff);

  const arma::vec captured = (k_eff == 0) ? arma::vec(n, arma::fill::zeros) : arma::sum(arma::square(U), 1);
  *diag_resid = arma::clamp(diag0 - captured, 0.0, arma::datum::inf);

  return U;
}

LIBKRIGING_EXPORT arma::mat LinearAlgebra::woodbury_solve(const arma::mat& U, const arma::vec& D, const arma::mat& B) {
  const arma::vec Dinv = 1.0 / D;
  const arma::mat DinvU = U.each_col() % Dinv;                             // diag(Dinv) * U        (n x k)
  const arma::mat DinvB = B.each_col() % Dinv;                             // diag(Dinv) * B        (n x m)
  arma::mat M = arma::eye<arma::mat>(U.n_cols, U.n_cols) + U.t() * DinvU;  // I_k + U' Dinv U (k x k)
  return DinvB - DinvU * arma::solve(M, U.t() * DinvB, LinearAlgebra::default_solve_opts);
}

LIBKRIGING_EXPORT double LinearAlgebra::woodbury_logdet(const arma::mat& U, const arma::vec& D) {
  const arma::vec Dinv = 1.0 / D;
  const arma::mat DinvU = U.each_col() % Dinv;
  const arma::mat M = arma::eye<arma::mat>(U.n_cols, U.n_cols) + U.t() * DinvU;
  double val;
  double sign;
  arma::log_det(val, sign, M);
  return arma::sum(arma::log(D)) + val;
}

// Solve A*X=B : X = A \ B
LIBKRIGING_EXPORT arma::mat LinearAlgebra::solve(const arma::mat& A, const arma::mat& B) {
  return arma::solve(A, B, LinearAlgebra::default_solve_opts);
}

LIBKRIGING_EXPORT arma::mat LinearAlgebra::conjugateGradient(const std::function<arma::vec(const arma::vec&)>& Amul,
                                                             const arma::mat& B,
                                                             arma::uword max_iter,
                                                             double tol,
                                                             const std::function<arma::vec(const arma::vec&)>& Pinv) {
  const bool preconditioned = static_cast<bool>(Pinv);
  const arma::uword n = B.n_rows;
  arma::mat X(n, B.n_cols, arma::fill::zeros);
  for (arma::uword c = 0; c < B.n_cols; ++c) {
    const arma::vec b = B.col(c);
    const double bnorm = arma::norm(b);
    if (bnorm == 0.0)
      continue;  // x=0 already solves A*x=0

    arma::vec x(n, arma::fill::zeros);
    arma::vec r = b;  // b - A*x0, x0 = 0
    arma::vec z = preconditioned ? Pinv(r) : r;
    arma::vec p = z;
    double rz_old = arma::dot(r, z);

    // GP covariance matrices are typically ill-conditioned (smooth kernels,
    // many points): the recursively-updated residual (r -= alpha*Ap) drifts
    // from the true residual under round-off well before max_iter is
    // reached, and pushing past that point can make x measurably WORSE, not
    // better (observed empirically: unstable growth after ~2n iterations on
    // a matern5_2 fit). Periodically recompute the exact residual from
    // scratch (one extra matvec every `restart_every` iterations) -- a
    // standard CG robustness fix -- to correct that drift.
    constexpr arma::uword restart_every = 50;

    for (arma::uword it = 0; it < max_iter; ++it) {
      const arma::vec Ap = Amul(p);
      const double pAp = arma::dot(p, Ap);
      if (pAp <= 0.0)
        break;  // breakdown guard: shouldn't happen for a genuinely SPD A
      const double alpha = rz_old / pAp;
      x += alpha * p;

      if ((it + 1) % restart_every == 0) {
        // Full restart: the just-recomputed residual reflects the TRUE
        // state at x, so the previous rz_old (from the drifted residual) is
        // no longer a meaningful reference for the Fletcher-Reeves ratio --
        // blending it into beta (as the non-restart branch does) sends the
        // search direction off in a bad direction instead of correcting it.
        // Reset p = z, i.e. restart CG fresh from the current x.
        r = b - Amul(x);
        if (arma::norm(r) / bnorm < tol)
          break;
        z = preconditioned ? Pinv(r) : r;
        rz_old = arma::dot(r, z);
        p = z;
        continue;
      }

      r -= alpha * Ap;
      const double rnorm = arma::norm(r);
      if (rnorm / bnorm < tol)
        break;
      z = preconditioned ? Pinv(r) : r;
      const double rz_new = arma::dot(r, z);
      p = z + (rz_new / rz_old) * p;
      rz_old = rz_new;
    }
    X.col(c) = x;
  }
  return X;
}

LIBKRIGING_EXPORT arma::mat LinearAlgebra::rademacherProbes(arma::uword n, arma::uword nprobe, unsigned seed) {
  arma::arma_rng::set_seed(static_cast<arma::arma_rng::seed_type>(seed));
  arma::mat probes(n, nprobe, arma::fill::none);
  probes.randu();  // in [0,1)
  probes.transform([](double v) { return v < 0.5 ? -1.0 : 1.0; });
  return probes;
}

LIBKRIGING_EXPORT double LinearAlgebra::stochasticLogDet(const std::function<arma::vec(const arma::vec&)>& Amul,
                                                         arma::uword n,
                                                         arma::uword nprobe,
                                                         arma::uword lanczos_steps,
                                                         const arma::mat& probes) {
  lanczos_steps = std::min(lanczos_steps, n);
  double total = 0.0;
  for (arma::uword p = 0; p < nprobe; ++p) {
    const double znorm = arma::norm(probes.col(p));
    if (znorm == 0.0)
      continue;

    // m-step Lanczos tridiagonalization of A, starting from probes.col(p)/znorm,
    // with full reorthogonalization (lanczos_steps stays modest relative to
    // n, so the extra O(m^2*n) cost is cheap next to the O(m*n^2) matvecs).
    arma::mat V(n, lanczos_steps, arma::fill::none);
    arma::vec alpha(lanczos_steps, arma::fill::zeros);
    arma::vec beta(lanczos_steps, arma::fill::zeros);  // beta(j) links v_{j+1} and v_j, beta(0) unused

    V.col(0) = probes.col(p) / znorm;
    arma::vec v_prev(n, arma::fill::zeros);
    double beta_prev = 0.0;
    arma::uword m_eff = lanczos_steps;
    for (arma::uword j = 0; j < lanczos_steps; ++j) {
      arma::vec w = Amul(V.col(j)) - beta_prev * v_prev;
      alpha(j) = arma::dot(w, V.col(j));
      w -= alpha(j) * V.col(j);
      // full reorthogonalization against all previous Lanczos vectors
      for (arma::uword i = 0; i <= j; ++i)
        w -= arma::dot(w, V.col(i)) * V.col(i);
      const double bj = arma::norm(w);
      if (j + 1 == lanczos_steps)
        break;
      if (bj < 1e-12) {
        m_eff = j + 1;  // invariant subspace found: A*V(:,0:j) stays within span(V(:,0:j))
        break;
      }
      beta(j) = bj;
      V.col(j + 1) = w / bj;
      v_prev = V.col(j);
      beta_prev = bj;
    }

    arma::mat T(m_eff, m_eff, arma::fill::zeros);
    for (arma::uword j = 0; j < m_eff; ++j)
      T(j, j) = alpha(j);
    for (arma::uword j = 0; j + 1 < m_eff; ++j) {
      T(j, j + 1) = beta(j);
      T(j + 1, j) = beta(j);
    }

    arma::vec eigval;
    arma::mat eigvec;
    arma::eig_sym(eigval, eigvec, T);

    double quad = 0.0;
    for (arma::uword j = 0; j < m_eff; ++j) {
      const double lambda = std::max(eigval(j), LinearAlgebra::num_nugget);
      quad += eigvec(0, j) * eigvec(0, j) * std::log(lambda);
    }
    total += quad;  // the leading znorm^2 == n for exact Rademacher entries is folded into the (n/nprobe) below
  }
  return (static_cast<double>(n) / static_cast<double>(nprobe)) * total;
}

// Solve X*A=B : X = B / A
LIBKRIGING_EXPORT arma::mat LinearAlgebra::rsolve(const arma::mat& A, const arma::mat& B) {
  // Force evaluation of ALL transposes to avoid LAPACK dimension mismatch (MKL ERROR Parameter 7)
  // Explicitly allocate and copy to ensure contiguous memory with correct leading dimensions
  // CRITICAL: MKL's DGELS requires proper matrix layout; views/templates cause Parameter 7 errors

  // Pre-allocate with correct dimensions and copy transposed data
  arma::mat At(A.n_cols, A.n_rows);
  At = A.t();  // Force copy, not view

  arma::mat Bt(B.n_cols, B.n_rows);
  Bt = B.t();  // Force copy, not view

  // Solve and store result
  arma::mat temp = arma::solve(At, Bt, LinearAlgebra::default_solve_opts);

  // Transpose result with explicit allocation
  arma::mat result(temp.n_cols, temp.n_rows);
  result = temp.t();  // Force copy, not view

  return result;
}

LIBKRIGING_EXPORT arma::mat LinearAlgebra::crossprod(const arma::mat& A) {
  // Use BLAS GEMM via Armadillo for A^T * A
  // This is faster than manual loops as it uses optimized BLAS library
  return arma::trans(A) * A;
}

LIBKRIGING_EXPORT arma::mat LinearAlgebra::tcrossprod(const arma::mat& A) {
  // Use BLAS GEMM via Armadillo for A * A^T
  // This is faster than manual loops as it uses optimized BLAS library
  return A * arma::trans(A);
}

LIBKRIGING_EXPORT arma::mat LinearAlgebra::diagcrossprod(const arma::mat& A) {
  return arma::diagmat(arma::sum(arma::square(A), 1));
}

LIBKRIGING_EXPORT arma::colvec LinearAlgebra::diagABA(const arma::mat& A, const arma::mat& B) {
  arma::mat D = trimatu(2 * B);
  D.diag() = B.diag();
  D = (A * D) % A;
  return sum(D, 1);
}

// Fast pointer-based computation of pairwise differences
// Benchmarked to be ~10x faster than the original implementation
// Original: uses division/modulo and armadillo indexing (expensive)
// This version: uses direct pointer access with manual indexing (cache-friendly)
// Now with OpenMP parallelization for large n
LIBKRIGING_EXPORT arma::mat LinearAlgebra::compute_dX(const arma::mat& X) {
  arma::uword n = X.n_rows;
  arma::uword d = X.n_cols;
  arma::mat dX(d, n * n, arma::fill::zeros);

  const double* X_mem = X.memptr();
  double* dX_mem = dX.memptr();

  #ifdef _OPENMP
  if (n >= 200) {  // Only use OpenMP for large enough matrices
    int optimal_threads = get_optimal_threads(2);
    #pragma omp parallel for schedule(dynamic, 8) num_threads(optimal_threads) if(n >= 200)
    for (arma::sword i = 0; i < static_cast<arma::sword>(n); i++) {
      for (arma::sword j = i + 1; j < static_cast<arma::sword>(n); j++) {
        arma::uword ij = i * n + j;
        arma::uword ji = j * n + i;
        for (arma::uword k = 0; k < d; k++) {
          // X is column-major: X(row, col) = X_mem[row + col * n_rows]
          double diff = X_mem[i + k * n] - X_mem[j + k * n];
          // dX is column-major: dX(row, col) = dX_mem[row + col * n_rows]
          dX_mem[k + ij * d] = diff;
          dX_mem[k + ji * d] = diff;
        }
      }
    }
  } else {
  #endif
    for (arma::uword i = 0; i < n; i++) {
      for (arma::uword j = i + 1; j < n; j++) {
        arma::uword ij = i * n + j;
        arma::uword ji = j * n + i;
        for (arma::uword k = 0; k < d; k++) {
          // X is column-major: X(row, col) = X_mem[row + col * n_rows]
          double diff = X_mem[i + k * n] - X_mem[j + k * n];
          // dX is column-major: dX(row, col) = dX_mem[row + col * n_rows]
          dX_mem[k + ij * d] = diff;
          dX_mem[k + ji * d] = diff;
        }
      }
    }
  #ifdef _OPENMP
  }
  #endif

  return dX;
}

LIBKRIGING_EXPORT void LinearAlgebra::covMat_sym_dX(arma::mat* R,
                                                     const arma::mat& dX,
                                                     const arma::vec& theta,
                                                     std::function<double(const arma::vec&, const arma::vec&)> Cov,
                                                     double factor,
                                                     const arma::vec& diag) {
  arma::uword n = (*R).n_rows;

  // First compute off-diagonal elements with OpenMP parallelization
  // Use dynamic scheduling for load balancing (lower triangle has uneven work)
  #ifdef _OPENMP
  if (n >= 200) {  // Only use OpenMP for large enough matrices (avoid overhead for small n)
    // Limit threads to avoid overhead - optimal is 4-8 threads based on benchmarks
    int optimal_threads = get_optimal_threads(2);
    #pragma omp parallel for schedule(dynamic, 8) num_threads(optimal_threads) if(n >= 200)
    for (arma::sword i = 0; i < static_cast<arma::sword>(n); i++) {
      for (arma::sword j = 0; j < i; j++) {
        double cov_val = Cov(dX.col(i * n + j), theta) * factor;
        (*R).at(i, j) = (*R).at(j, i) = cov_val;
      }
    }
  } else {
  #endif
    // Serial version for small matrices or when OpenMP is disabled
    for (arma::uword i = 0; i < n; i++) {
      for (arma::uword j = 0; j < i; j++) {
        double cov_val = Cov(dX.col(i * n + j), theta) * factor;
        (*R).at(i, j) = (*R).at(j, i) = cov_val;
      }
    }
  #ifdef _OPENMP
  }
  #endif

  // Then set diagonal
  if (diag.n_elem == 0) {
    for (arma::uword i = 0; i < n; i++) {
      (*R).at(i, i) = factor;  // factor * 1
    }
  } else {
    (*R).diag() = diag;
  }
}

LIBKRIGING_EXPORT void LinearAlgebra::covMat_sym_X(arma::mat* R,
                                                    const arma::mat& X,
                                                    const arma::vec& theta,
                                                    std::function<double(const arma::vec&, const arma::vec&)> Cov,
                                                    double factor,
                                                    const arma::vec& diag) {
  arma::uword n = (*R).n_rows;
  arma::uword d = X.n_rows;

  // Use pointer-based access for better performance
  const double* X_mem = X.memptr();

  // Block size for cache optimization (64 elements fit well in L1 cache)
  const arma::uword BLOCK_SIZE = 64;

  // First compute off-diagonal elements with block-based OpenMP parallelization
  // Use dynamic scheduling because lower triangle has uneven work distribution
  #ifdef _OPENMP
  if (n >= 200) {  // Only use OpenMP for large enough matrices (avoid overhead for small n)
    // Limit threads to avoid overhead - optimal is 4-8 threads based on benchmarks
    int optimal_threads = get_optimal_threads(2);
    #pragma omp parallel for schedule(dynamic, 4) num_threads(optimal_threads) if(n >= 200)
    for (arma::sword bi = 0; bi < static_cast<arma::sword>(n); bi += BLOCK_SIZE) {
      arma::uword block_end_i = (bi + BLOCK_SIZE < n) ? bi + BLOCK_SIZE : n;

      // Pre-allocate diff vector once per thread (thread-local)
      arma::vec diff(d);
      double* diff_mem = diff.memptr();

      for (arma::uword i = bi; i < block_end_i; i++) {
        for (arma::uword j = 0; j < i; j++) {
          for (arma::uword k = 0; k < d; k++) {
            diff_mem[k] = X_mem[k + i * d] - X_mem[k + j * d];
          }
          double cov_val = Cov(diff, theta) * factor;
          (*R).at(i, j) = (*R).at(j, i) = cov_val;
        }
      }
    }
  } else {
  #endif
    // Serial version for small matrices or when OpenMP is disabled
    // Pre-allocate diff vector to avoid repeated allocations
    arma::vec diff(d);
    double* diff_mem = diff.memptr();

    for (arma::uword i = 0; i < n; i++) {
      for (arma::uword j = 0; j < i; j++) {
        for (arma::uword k = 0; k < d; k++) {
          diff_mem[k] = X_mem[k + i * d] - X_mem[k + j * d];
        }
        double cov_val = Cov(diff, theta) * factor;
        (*R).at(i, j) = (*R).at(j, i) = cov_val;
      }
    }
  #ifdef _OPENMP
  }
  #endif

  // Then set diagonal
  if (diag.n_elem == 0) {
    for (arma::uword i = 0; i < n; i++) {
      (*R).at(i, i) = factor;  // factor * 1
    }
  } else {
    (*R).diag() = diag;
  }
}

LIBKRIGING_EXPORT void LinearAlgebra::covMat_rect(arma::mat* R,
                                                   const arma::mat& X1,
                                                   const arma::mat& X2,
                                                   const arma::vec& theta,
                                                   std::function<double(const arma::vec&, const arma::vec&)> Cov,
                                                   double factor) {
  arma::uword n1 = X1.n_cols;
  arma::uword n2 = X2.n_cols;
  arma::uword d = X1.n_rows;

  // Use pointer-based access for better performance
  const double* X1_mem = X1.memptr();
  const double* X2_mem = X2.memptr();

  // Block size for cache optimization
  const arma::uword BLOCK_SIZE = 64;

  // Block-based parallelization for better cache locality
  // Rectangular matrices have uniform work distribution, use static scheduling
  #ifdef _OPENMP
  arma::uword total_work = n1 * n2;
  if (total_work >= 40000) {  // Only use OpenMP for sufficient work (avoid overhead for small matrices)
    // Limit threads to avoid overhead - optimal is 4-8 threads based on benchmarks
    int optimal_threads = get_optimal_threads(2);
    #pragma omp parallel num_threads(optimal_threads) if(total_work >= 40000)
    {
      // Pre-allocate diff vector once per thread (thread-local)
      arma::vec diff(d);
      double* diff_mem = diff.memptr();

      #pragma omp for schedule(static) collapse(2)
      for (arma::sword bi = 0; bi < static_cast<arma::sword>(n1); bi += BLOCK_SIZE) {
        for (arma::sword bj = 0; bj < static_cast<arma::sword>(n2); bj += BLOCK_SIZE) {
          arma::uword block_end_i = (bi + BLOCK_SIZE < static_cast<arma::sword>(n1)) ? bi + BLOCK_SIZE : n1;
          arma::uword block_end_j = (bj + BLOCK_SIZE < static_cast<arma::sword>(n2)) ? bj + BLOCK_SIZE : n2;

          // Process block with good cache locality
          for (arma::uword i = bi; i < block_end_i; i++) {
            for (arma::uword j = bj; j < block_end_j; j++) {
              for (arma::uword k = 0; k < d; k++) {
                diff_mem[k] = X1_mem[k + i * d] - X2_mem[k + j * d];
              }
              (*R).at(i, j) = Cov(diff, theta) * factor;
            }
          }
        }
      }
    }
  } else {
  #endif
    // Serial version for small matrices or when OpenMP is disabled
    // Pre-allocate diff vector to avoid repeated allocations
    arma::vec diff(d);
    double* diff_mem = diff.memptr();

    for (arma::uword i = 0; i < n1; i++) {
      for (arma::uword j = 0; j < n2; j++) {
        for (arma::uword k = 0; k < d; k++) {
          diff_mem[k] = X1_mem[k + i * d] - X2_mem[k + j * d];
        }
        (*R).at(i, j) = Cov(diff, theta) * factor;
      }
    }
  #ifdef _OPENMP
  }
  #endif
}

// Efficient computation of trace(A * B) = sum_i sum_j A(i,j) * B(j,i)
// This avoids the explicit matrix multiplication A * B
LIBKRIGING_EXPORT double LinearAlgebra::trace_prod(const arma::mat& A, const arma::mat& B) {
  arma::uword n = A.n_rows;
  arma::uword m = A.n_cols;

  if (B.n_rows != m || B.n_cols != n) {
    throw std::invalid_argument("trace_prod: incompatible matrix dimensions");
  }

  double sum = 0.0;
  const double* A_mem = A.memptr();
  const double* B_mem = B.memptr();

  // A is stored column-major: A(i,j) = A_mem[i + j*n]
  // B is stored column-major: B(i,j) = B_mem[i + j*m]
  // We need: sum_i sum_j A(i,j) * B(j,i)
  //        = sum_i sum_j A_mem[i + j*n] * B_mem[j + i*m]

  for (arma::uword j = 0; j < m; j++) {
    for (arma::uword i = 0; i < n; i++) {
      sum += A_mem[i + j * n] * B_mem[j + i * m];
    }
  }

  return sum;
}
LIBKRIGING_EXPORT arma::mat LinearAlgebra::solve_lower(const arma::mat& L, const arma::mat& B) {
  return arma::solve(arma::trimatl(L), B);
}

LIBKRIGING_EXPORT arma::mat LinearAlgebra::solve_upper(const arma::mat& U, const arma::mat& B) {
  return arma::solve(arma::trimatu(U), B);
}

LIBKRIGING_EXPORT arma::mat LinearAlgebra::rsolve_upper(const arma::mat& U, const arma::mat& B) {
  // X*U = B → U'*X' = B' → X = solve_lower(U', B')'
  return arma::solve(arma::trimatl(U.t()), B.t()).t();
}

LIBKRIGING_EXPORT arma::mat LinearAlgebra::inv_sympd(const arma::mat& L) {
  return arma::solve(arma::trimatu(L.t()), arma::solve(arma::trimatl(L), arma::eye(L.n_rows, L.n_rows)));
}

LIBKRIGING_EXPORT arma::mat LinearAlgebra::chol_upper(const arma::mat& A) {
  return arma::chol(A);
}

LIBKRIGING_EXPORT void LinearAlgebra::qr_econ(arma::mat& Q, arma::mat& R, const arma::mat& A) {
  arma::qr_econ(Q, R, A);
}
