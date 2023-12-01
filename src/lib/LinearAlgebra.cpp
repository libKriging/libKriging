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
  arma::mat R = arma::mat(X.n_rows, X.n_cols);
  bool ok = arma::chol(R, X, "lower");
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
    return R;
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
