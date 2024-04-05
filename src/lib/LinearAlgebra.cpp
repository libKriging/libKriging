// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio
#include <cmath>
// clang-format on

#include "libKriging/LinearAlgebra.hpp"

#include "libKriging/utils/lk_armadillo.hpp"
#include "libKriging/Covariance.hpp"
#include "libKriging/Bench.hpp"
#include <thread>

LIBKRIGING_EXPORT
const arma::solve_opts::opts LinearAlgebra::default_solve_opts
    = arma::solve_opts::fast + arma::solve_opts::no_approx + arma::solve_opts::no_band + arma::solve_opts::no_sympd;

float LinearAlgebra::num_nugget = 1E-10;

LIBKRIGING_EXPORT void LinearAlgebra::set_num_nugget(float nugget) {
  LinearAlgebra::num_nugget = nugget;
};

LIBKRIGING_EXPORT float LinearAlgebra::get_num_nugget() {
  return LinearAlgebra::num_nugget;
};

LIBKRIGING_EXPORT arma::fmat LinearAlgebra::safe_chol_lower(arma::fmat X) {
  return LinearAlgebra::safe_chol_lower(X, 0);
}

bool LinearAlgebra::warn_chol = false;

LIBKRIGING_EXPORT void LinearAlgebra::set_chol_warning(bool warn) {
  LinearAlgebra::warn_chol = warn;
};

bool LinearAlgebra::chol_rcond_check = false;

LIBKRIGING_EXPORT void LinearAlgebra::check_chol_rcond(bool c) {
  LinearAlgebra::chol_rcond_check = c;
};

int LinearAlgebra::max_inc_choldiag = 10;

// Recursive turn-around for ill-condition of correlation matrix. Used in *Kriging::fit & *Kriging::simulate
//' @ref: Andrianakis, I. and Challenor, P. G. (2012). The effect of the nugget on Gaussian pro-cess emulators of
// computer models. Comput. Stat. Data Anal., 56(12):4215–4228.
LIBKRIGING_EXPORT arma::fmat LinearAlgebra::safe_chol_lower(arma::fmat X, int inc_cond) {
  arma::fmat L = arma::fmat(X.n_rows, X.n_cols, arma::fill::none);
  auto t0 = Bench::tic();
  bool ok = arma::chol(L, X, "lower");
  t0 = Bench::toc(nullptr, "        arma::chol" ,t0);
  bool wrong_rcond = LinearAlgebra::chol_rcond_check;
  if (ok) {
    //wrong_rcond = wrong_rcond && (LinearAlgebra::rcond_approx_chol(L) < LinearAlgebra::min_rcond_approx);
    //t0 = Bench::toc(nullptr, "        rcond_approx" ,t0);
    wrong_rcond = wrong_rcond && (LinearAlgebra::rcond_chol(L) < LinearAlgebra::min_rcond);
    t0 = Bench::toc(nullptr, "        rcond" ,t0);
  }
  if (!ok || wrong_rcond) {
    if (inc_cond > max_inc_choldiag) {
      throw std::runtime_error("[ERROR] Exceed max numerical nugget ("+std::to_string(inc_cond)+" x 1e"+std::to_string(log10(LinearAlgebra::num_nugget))+") added to force chol matrix");
    } else if (LinearAlgebra::num_nugget <= 0.0) {
      throw std::runtime_error("[ERROR] Cannot add numerical nugget which is not strictly positive: "
                               + std::to_string(LinearAlgebra::num_nugget));
    } else {
      X.diag() += LinearAlgebra::num_nugget;  // inc diagonal
      return LinearAlgebra::safe_chol_lower(X, inc_cond + 1);
    }
    t0 = Bench::toc(nullptr, "        inc_cond" ,t0);
  } else {
    if (warn_chol && (inc_cond > 0))
      arma::cout << "[WARNING] Added " << inc_cond << " numerical nugget to force Cholesky decomposition"
                 << arma::endl;
    return L;
  }
}

float LinearAlgebra::min_rcond = 1e-18;

LIBKRIGING_EXPORT float LinearAlgebra::rcond_chol(arma::fmat chol) {
  float rcond = arma::rcond(chol);
  rcond *= rcond;
  if (warn_chol)
    if (rcond < (chol.n_rows * min_rcond))
      arma::cout << "[WARNING] rcond " << rcond << " is below minimal value." << arma::endl;
  return rcond;
}

float LinearAlgebra::min_rcond_approx = 1e-10;
// Proxy to arma::rcond
// @ref: N. J. Higham, "A survey of condition number estimation for triangular matrices," SIAM Review, vol. 29, no. 4,
// pp. 575–596, Dec. 1987.
LIBKRIGING_EXPORT float LinearAlgebra::rcond_approx_chol(arma::fmat chol) {
  float m = chol.at(0, 0);
  float M = chol.at(0, 0);
  if (chol.n_rows > 1)
    for (arma::uword i = 1; i < chol.n_rows; i++) {
      float cii = chol.at(i, i);
      if (cii < m) {
        m = cii;
      } else if (cii > M) {
        M = cii;
      }
    }
  float rcond = m / M;
  rcond = rcond * rcond;
  if (warn_chol)
    if (rcond < (chol.n_rows * min_rcond_approx))
      arma::cout << "[WARNING] rcond_approx " << rcond << " is below minimal value." << arma::endl;
  return rcond;
}

LIBKRIGING_EXPORT arma::fmat LinearAlgebra::cholCov(arma::fmat* R, 
const arma::fmat& _dX, const arma::fvec& _theta, 
std::function<float (const arma::fvec &, const arma::fvec &)> Cov, 
const float factor, const arma::fvec diag) {
  arma::uword n = (*R).n_rows;

  auto t0 = Bench::tic();
  for (arma::uword i = 0; i < n; i++) {
    //(*R).at(i, i) = 1.0;
    for (arma::uword j = 0; j < i; j++) {
      (*R).at(i, j) = (*R).at(j, i) = Cov(_dX.col(i * n + j), _theta) * factor;
    }
  }
  t0 = Bench::toc(nullptr, "    Cov: " + std::to_string(n) + "/" + std::to_string(n),t0);

// Slower:
  // std::vector<std::thread> col_threads(n);
  // std::vector<arma::fcolvec> col_vecs(n);
  // for (arma::uword i = 0; i < n; i++) {
  //   col_threads[i] = std::thread([i, &col_vecs, _dX, _theta, Cov, factor, n](){
  //     arma::fcolvec col_vecs_i = arma::fcolvec(n,arma::fill::none);
  //     for (arma::uword j = 0; j < n; j++) {
  //       col_vecs_i.at(j) = Cov(_dX.col(i * n + j), _theta) * factor;
  //     }
  //     col_vecs[i] = col_vecs_i;
  //   });
  //   //(*R).at(i, j) = (*R).at(j, i) = Cov(_dX.col(i * n + j), _theta) * factor;
  //   //t0 = Bench::toc(nullptr, "    Cov: " + std::to_string(i) + "/" + std::to_string(n),t0);
  // }
  // for (arma::uword i = 0; i < n; i++) {
  //   col_threads[i].join();
  //   (*R).col(i) = col_vecs[i];
  // }
  // t0 = Bench::toc(nullptr, "    Cov (threads): " + std::to_string(n) + "/" + std::to_string(n),t0);

// Same speed:
//#pragma omp parallel for shared(*R) 
//  for (arma::uword i = 0; i < n; i++) {
//    (*R).at(i, i) = 1.0;
//    for (arma::uword j = 0; j < i; j++) {
//      (*R).at(i, j) = (*R).at(j, i) = Cov(_dX.col(i * n + j), _theta) * factor;
//    }
//  }
//  t0 = Bench::toc(nullptr, "    Cov (omp): " + std::to_string(n) + "/" + std::to_string(n),t0);
  
  if (diag.n_elem == 0) {
    (*R).diag().ones();//(*R).diag() = arma::fvec(n, arma::fill::ones);
  } else {
    (*R).diag() = diag;
  }
  t0 = Bench::toc(nullptr, "    Cov: diag",t0);

  // Cholesky decompostion of covariance matrix

  arma::fmat L = LinearAlgebra::safe_chol_lower(*R);  // Do NOT trimatl T (slower because copy): trimatl(chol(R, "lower"));
  t0 = Bench::toc(nullptr, "    Chol",t0);

  return L;
}

LIBKRIGING_EXPORT arma::fmat LinearAlgebra::update_cholCov(arma::fmat* R, 
const arma::fmat& _dX, const arma::fvec& _theta, 
std::function<float (const arma::fvec &, const arma::fvec &)> Cov, 
const float factor, const arma::fvec diag,
const arma::fmat& T_old) {
  arma::uword n_old = T_old.n_rows;
  arma::fmat R_old = T_old * T_old.t(); // hope that does not cost too much... (we dont save previous R)

  arma::uword n = (*R).n_rows;

  auto t0 = Bench::tic();
  (*R).submat(0, 0, n_old-1, n_old-1) = R_old;
  t0 = Bench::toc(nullptr, "    Cov: copy old",t0);
  for (arma::uword i = n_old; i < n; i++) {
    (*R).at(i, i) = 1.0;
    for (arma::uword j = 0; j < i; j++) {
      (*R).at(i, j) = (*R).at(j, i) = Cov(_dX.col(i * n + j), _theta) * factor;
    }
    //t0 = Bench::toc(nullptr, "    Cov: " + std::to_string(i) + "/" + std::to_string(n),t0);
  }
  t0 = Bench::toc(nullptr, "    Cov: " + std::to_string(n) + "/" + std::to_string(n),t0);
  (*R).diag() = diag;
  t0 = Bench::toc(nullptr, "    Cov: diag",t0);

  arma::fmat L = LinearAlgebra::chol_block(*R, T_old, R_old);
  t0 = Bench::toc(nullptr, "    Chol Block",t0);

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
LIBKRIGING_EXPORT arma::fmat LinearAlgebra::chol_block(const arma::fmat C, const arma::fmat Loo, const arma::fmat Coo) {
  arma::uword n = C.n_rows;
  arma::uword no = Coo.n_rows; // old size. n-1 if we just add one observation.  

  auto t0 = Bench::tic();
  arma::fmat Cuo = C.submat(no, 0,  n-1, no-1);
  t0 = Bench::toc(nullptr, "        >Cuo",t0);
  arma::fmat Cuu = C.submat(no, no, n-1, n-1);  
  t0 = Bench::toc(nullptr, "        >Cuu",t0);

  arma::fmat L = arma::fmat(n, n);
  L.submat( 0,  0,  no-1, no-1 ) = Loo;
  t0 = Bench::toc(nullptr, "        <Loo",t0);
  arma::fmat Luo = Cuo * arma::solve( Loo, arma::eye<arma::fmat>(no, no) ).t(); // Lu,o = Cu,o Lo,o^-T
  t0 = Bench::toc(nullptr, "        Luo = Cuo / Loo.t()",t0);
  L.submat( no, 0,  n-1,  no-1 ) = Luo;
  t0 = Bench::toc(nullptr, "        <Luo",t0);  
  L.submat( no, no, n-1,  n-1 ) = LinearAlgebra::safe_chol_lower( Cuu - Luo * Luo.t() ); // Lu,u = chol( Cu,u - Lu,o Lu,o^T )  
  t0 = Bench::toc(nullptr, "        <Luu = chol( Cuu - Luo * Luo.t() )",t0);

  arma::fmat lowL = arma::trimatl(L);
  t0 = Bench::toc(nullptr, "        trimatl L",t0);

  return lowL;
}

// Solve A*X=B : X = A \ B
LIBKRIGING_EXPORT arma::fmat LinearAlgebra::solve(const arma::fmat& A, const arma::fmat& B) {
  return arma::solve(A, B, LinearAlgebra::default_solve_opts);
}

// Solve X*A=B : X = B / A
LIBKRIGING_EXPORT arma::fmat LinearAlgebra::rsolve(const arma::fmat& A, const arma::fmat& B) {
  return arma::solve(A.t(), B.t(), LinearAlgebra::default_solve_opts).t();
}

LIBKRIGING_EXPORT arma::fmat LinearAlgebra::crossprod(const arma::fmat& A) {
  return A.t() * A;
}

LIBKRIGING_EXPORT arma::fmat LinearAlgebra::tcrossprod(const arma::fmat& A) {
  return A * A.t();
}

LIBKRIGING_EXPORT arma::fmat LinearAlgebra::diagcrossprod(const arma::fmat& A) {
  return arma::diagmat(arma::sum(arma::square(A), 1));   
}

LIBKRIGING_EXPORT arma::fcolvec LinearAlgebra::diagABA(const arma::fmat& A, const arma::fmat& B) {
  arma::fmat D = trimatu(2 * B);
  D.diag() = B.diag();
  D = (A * D) % A;
  return sum(D, 1);
}