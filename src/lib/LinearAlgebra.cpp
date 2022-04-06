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
  arma::mat R = arma::mat(X.n_rows, X.n_cols);
  bool ok = arma::chol(R, X, "lower");
  if (!ok) {
    if (LinearAlgebra::num_nugget <= 0.0)
      throw std::runtime_error("[ERROR] Cannot add anumerical nugget which is not strictly positive: "
                               + std::to_string(LinearAlgebra::num_nugget));
    else {
      arma::cout << "[WARNING]: chol failed, adding numerical nugget (on diagonal)" << arma::endl;
      X /= 1.0 + LinearAlgebra::num_nugget;
      X.diag().ones();
      return LinearAlgebra::safe_chol_lower(X);
    }
  } else
    return R;
}
