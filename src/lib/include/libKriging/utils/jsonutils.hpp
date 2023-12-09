#ifndef LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_JSONUTILS_HPP
#define LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_JSONUTILS_HPP

#include "libKriging/utils/lk_armadillo.hpp"
#include "libKriging/utils/nlohmann/json.hpp"

nlohmann::json to_json(const arma::rowvec& t);

arma::rowvec rowvec_from_json(const nlohmann::json& data);

#endif  // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_JSONUTILS_HPP