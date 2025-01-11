// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio
#include <cmath>
// clang-format on

#include "libKriging/LinearAlgebra.hpp"

#include <thread>
#include "libKriging/Bench.hpp"
#include "libKriging/Covariance.hpp"
#include "libKriging/utils/lk_armadillo.hpp"

arma::solve_opts::opts LinearAlgebra::default_solve_opts
    = arma::solve_opts::fast + arma::solve_opts::no_approx;

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

bool LinearAlgebra::warn_chol = false;

LIBKRIGING_EXPORT void LinearAlgebra::set_chol_warning(bool warn) {
  LinearAlgebra::warn_chol = warn;
};

bool LinearAlgebra::chol_rcond_check = true;

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
      X.diag() += LinearAlgebra::num_nugget;  // inc diagonal
      return LinearAlgebra::safe_chol_lower_retry(X, inc_cond + 1);
    }
    // t0 = Bench::toc(nullptr, "        inc_cond" ,t0);
  } else {
    if (warn_chol && (inc_cond > 0))
      arma::cout << "[WARNING] Added " << inc_cond << " numerical nugget to force Cholesky decomposition" << arma::endl;
    return L;
  }
}

double LinearAlgebra::min_rcond = 1e-18;

LIBKRIGING_EXPORT double LinearAlgebra::rcond_chol(arma::mat chol) {
  double rcond = arma::rcond(chol);
  rcond *= rcond;
  if (warn_chol)
    if (rcond < (chol.n_rows * min_rcond))
      arma::cout << "[WARNING] rcond " << rcond << " is below minimal value." << arma::endl;
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
    if (rcond < (chol.n_rows * min_rcond_approx))
      arma::cout << "[WARNING] rcond_approx " << rcond << " is below minimal value." << arma::endl;
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
  L.submat(no, no, n - 1, n - 1) = LinearAlgebra::safe_chol_lower(
      Cuu - LinearAlgebra::crossprod(Lou));  // Luo * Luo.t() ); // Lu,u = chol( Cu,u - Lu,o Lu,o^T )
  // t0 = Bench::toc(nullptr, "        <Luu = chol( Cuu - Luo * Luo.t() )",t0);

  arma::mat lowL = arma::trimatl(L);
  // t0 = Bench::toc(nullptr, "        trimatl L",t0);

  return lowL;
}

// Solve A*X=B : X = A \ B
LIBKRIGING_EXPORT arma::mat LinearAlgebra::solve(const arma::mat& A, const arma::mat& B) {
  return arma::solve(A, B, LinearAlgebra::default_solve_opts);
}

// Solve X*A=B : X = B / A
LIBKRIGING_EXPORT arma::mat LinearAlgebra::rsolve(const arma::mat& A, const arma::mat& B) {
  return arma::solve(A.t(), B.t(), LinearAlgebra::default_solve_opts).t();
}

LIBKRIGING_EXPORT arma::mat LinearAlgebra::crossprod(const arma::mat& A) {
  // return A.t() * A;
  arma::mat AtA = arma::mat(A.n_cols, A.n_cols, arma::fill::none);
  for (arma::uword i = 0; i < A.n_cols; i++) {
    for (arma::uword j = 0; j <= i; j++) {
      AtA.at(i, j) = arma::dot(A.col(i), A.col(j));
      AtA.at(j, i) = AtA.at(i, j);
    }
  }
  return AtA;
}

LIBKRIGING_EXPORT arma::mat LinearAlgebra::tcrossprod(const arma::mat& A) {
  // return A * A.t();
  arma::mat AAt = arma::mat(A.n_rows, A.n_rows, arma::fill::none);
  for (arma::uword i = 0; i < A.n_rows; i++) {
    for (arma::uword j = 0; j <= i; j++) {
      AAt.at(i, j) = arma::dot(A.row(i), A.row(j));
      AAt.at(j, i) = AAt.at(i, j);
    }
  }
  return AAt;
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