#ifndef LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_UTILS_DATA_FROM_ARMA_VEC_HPP
#define LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_UTILS_DATA_FROM_ARMA_VEC_HPP

#include "libKriging/utils/lk_armadillo.hpp"

namespace arma {
inline double* data(vec& x) {
  return x.memptr();
}
}  // namespace arma

#endif  // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_UTILS_DATA_FROM_ARMA_VEC_HPP
