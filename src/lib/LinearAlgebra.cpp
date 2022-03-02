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
