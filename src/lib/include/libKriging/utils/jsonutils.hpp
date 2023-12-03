#ifndef LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_JSONUTILS_HPP
#define LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_JSONUTILS_HPP

#include <tao/json/events/from_value.hpp>
#include <tao/json.hpp>

#include "libKriging/utils/lk_armadillo.hpp"

tao::json::value to_value(const arma::rowvec &t);

#endif  // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_JSONUTILS_HPP