// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio
#include <cmath>
// clang-format on

#include "libKriging/LinearAlgebra.hpp"

#include "libKriging/utils/lk_armadillo.hpp"

LIBKRIGING_EXPORT
const arma::solve_opts::opts LinearAlgebra::default_solve_opts
    = arma::solve_opts::fast + arma::solve_opts::no_approx + arma::solve_opts::no_band + arma::solve_opts::no_sympd;

double LinearAlgebra::num_nugget = 1E-10;

LIBKRIGING_EXPORT void LinearAlgebra::set_num_nugget(double nugget) {
  LinearAlgebra::num_nugget = nugget;
};

LIBKRIGING_EXPORT double LinearAlgebra::get_num_nugget() {
  return LinearAlgebra::num_nugget;
};

LIBKRIGING_EXPORT arma::mat LinearAlgebra::safe_chol_lower(arma::mat X) {
  return LinearAlgebra::safe_chol_lower(X, 0);
}

bool LinearAlgebra::warn_chol = false;

LIBKRIGING_EXPORT void LinearAlgebra::set_chol_warning(bool warn) {
  LinearAlgebra::warn_chol = warn;
};

int LinearAlgebra::max_inc_choldiag = 10;

// Recursive turn-around for ill-condition of correlation matrix. Used in *Kriging::fit & *Kriging::simulate
//' @ref: Andrianakis, I. and Challenor, P. G. (2012). The effect of the nugget on Gaussian pro-cess emulators of
// computer models. Comput. Stat. Data Anal., 56(12):4215–4228.
LIBKRIGING_EXPORT arma::mat LinearAlgebra::safe_chol_lower(arma::mat X, int inc_cond) {
  arma::mat L = arma::mat(X.n_rows, X.n_cols);
  bool ok = arma::chol(L, X, "lower");
  if (!ok) {
    if (inc_cond > max_inc_choldiag) {
      throw std::runtime_error("[ERROR] Exceed max numerical nugget added to force chol matrix");
    } else if (LinearAlgebra::num_nugget <= 0.0) {
      throw std::runtime_error("[ERROR] Cannot add anumerical nugget which is not strictly positive: "
                               + std::to_string(LinearAlgebra::num_nugget));
    } else {
      X.diag() += LinearAlgebra::num_nugget;  // inc diagonal
      return LinearAlgebra::safe_chol_lower(X, inc_cond + 1);
    }
  } else {
    if (inc_cond > 0)
      if (warn_chol)
        arma::cout << "[WARNING] Added " << inc_cond << " numerical nugget to force Cholesky decomposition"
                   << arma::endl;
    return L;
  }
}

double LinearAlgebra::min_rcond = 1e-10;

// Proxy to arma::rcond
// @ref: N. J. Higham, "A survey of condition number estimation for triangular matrices," SIAM Review, vol. 29, no. 4,
// pp. 575–596, Dec. 1987.
LIBKRIGING_EXPORT double LinearAlgebra::rcond_chol(arma::mat chol) {
  double m = chol.at(0, 0);
  double M = chol.at(0, 0);
  if (chol.n_rows > 1)
    for (arma::uword i = 1; i < chol.n_rows; i++) {
      if (chol.at(i, i) < m) {
        m = chol.at(i, i);
      } else if (chol.at(i, i) > M) {
        M = chol.at(i, i);
      }
    }
  double rcond = m / M;
  rcond = rcond * rcond;
  if (warn_chol)
    if (rcond < (chol.n_rows * min_rcond))
      arma::cout << "[WARNING] rcond " << rcond << " is below minimal value." << arma::endl;
  return rcond;
}

LIBKRIGING_EXPORT arma::mat LinearAlgebra::cholCov(arma::mat* R, 
const arma::mat& _dX, const arma::vec& _theta, 
std::function<double (const arma::vec &, const arma::vec &)> Cov) {
  arma::uword n = (*R).n_rows;

  for (arma::uword i = 0; i < n; i++) {
    (*R).at(i, i) = 1;
    for (arma::uword j = 0; j < i; j++) {
      (*R).at(i, j) = (*R).at(j, i) = Cov(_dX.col(i * n + j), _theta);
    }
  }

  // Cholesky decompostion of covariance matrix
  return LinearAlgebra::safe_chol_lower(*R);  // Do NOT trimatl T (slower because copy): trimatl(chol(R, "lower"));
}

LIBKRIGING_EXPORT arma::mat LinearAlgebra::update_cholCov(arma::mat* R, 
const arma::mat& _dX, const arma::vec& _theta, 
std::function<double (const arma::vec &, const arma::vec &)> Cov,
const arma::mat& T_old) {
  arma::uword n_old = T_old.n_rows;
  arma::mat R_old = T_old * T_old.t(); // hope that does not cost too much... (we dont save previous R)

  arma::uword n = (*R).n_rows;

  (*R).submat(0, 0, n_old-1, n_old-1) = R_old;
  for (arma::uword i = n_old; i < n; i++) {
    (*R).at(i, i) = 1;
    for (arma::uword j = 0; j < i; j++) {
      (*R).at(i, j) = (*R).at(j, i) = Cov(_dX.col(i * n + j), _theta);
    }
  }

  return LinearAlgebra::chol_block(*R, T_old, R_old);
}

// We want to compute the cholesky root of C, knowing the cholesky root of a block of C: Co,o.
// We use the notation of the doc "libKriging" update by Yves Deville (o=old, u=new):
// C = | Co,o Co,u | = | Lo,o   0  | | Lo,o^T Lu,o^T | = | Lo,o*Lo,o^T          Lo,o*Lu,o^T        |
//     | Cu,o Cu,u |   | Lu,o Lu,u | |   0    Lu,u^T |   | Lu,o*Lo,o^T   Lu,o*Lu,o^T + Lu,u*Lu,u^T |
// so, by id.:
//   Lo,o is the cholesky root of Co,o (known)
//   Lu,o = Cu,o Lo,o^-T
//   Lu,u is the cholesky root of Cu,u - Lu,o Lu,o^T
LIBKRIGING_EXPORT arma::mat LinearAlgebra::chol_block(const arma::mat C, const arma::mat Loo, const arma::mat Coo) {
  arma::uword n = C.n_rows;
  arma::uword no = Coo.n_rows; // old size. n-1 if we just add one observation.  

        arma::cout << "C:" << arma::size(C) << arma::endl;
        arma::cout << "Coo:" << arma::size(Coo) << arma::endl;
        arma::cout << "Loo:" << arma::size(Loo) << arma::endl;


  arma::mat Cuo = C.submat(no, 0,  n-1, no-1);
  arma::mat Cuu = C.submat(no, no, n-1, n-1);  

        arma::cout << "Cuo:" << arma::size(Cuo) << arma::endl;
        arma::cout << "Cuu:" << arma::size(Cuu) << arma::endl;

  arma::mat L = arma::mat(n, n);
  L.submat( 0,  0,  no-1, no-1 ) = Loo;
  arma::mat Luo = Cuo * arma::solve( Loo, arma::eye<arma::mat>(no, no) ).t(); // Lu,o = Cu,o Lo,o^-T
        arma::cout << "Luo:" << arma::size(Luo) << arma::endl;
  L.submat( no, 0,  n-1,  no-1 ) = Luo;
  L.submat( no, no, n-1,  n-1 ) = LinearAlgebra::safe_chol_lower( Cuu - Luo * Luo.t() ); // Lu,u = chol( Cu,u - Lu,o Lu,o^T )  

  return arma::trimatl(L);
}