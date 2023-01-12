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
arma::mat LinearAlgebra::safe_chol_lower(arma::mat X, int inc_cond) {
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
