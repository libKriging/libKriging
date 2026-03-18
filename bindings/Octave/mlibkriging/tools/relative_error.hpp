#ifndef LIBKRIGING_BINDINGS_OCTAVE_TOOLS_RELATIVE_ERROR_HPP
#define LIBKRIGING_BINDINGS_OCTAVE_TOOLS_RELATIVE_ERROR_HPP

#include "libKriging/utils/lk_armadillo.hpp"

double relative_error(const arma::vec& x, const arma::vec& y);

#endif  // LIBKRIGING_BINDINGS_OCTAVE_TOOLS_RELATIVE_ERROR_HPP
