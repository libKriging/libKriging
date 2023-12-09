#ifndef LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_JSONUTILS_HPP
#define LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_JSONUTILS_HPP

#include "libKriging/utils/lk_armadillo.hpp"
#include "libKriging/utils/nlohmann/json.hpp"

#include "libKriging/libKriging_exports.h"

LIBKRIGING_EXPORT nlohmann::json to_json(const arma::rowvec& t);
LIBKRIGING_EXPORT nlohmann::json to_json(const arma::colvec& t);
LIBKRIGING_EXPORT nlohmann::json to_json(const arma::mat& t);

LIBKRIGING_EXPORT arma::rowvec rowvec_from_json(const nlohmann::json& data);
LIBKRIGING_EXPORT arma::colvec colvec_from_json(const nlohmann::json& data);
LIBKRIGING_EXPORT arma::mat mat_from_json(const nlohmann::json& data);

#endif  // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_JSONUTILS_HPP