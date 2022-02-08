#ifndef LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_HASH_FUNCTION_HPP
#define LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_HASH_FUNCTION_HPP

#include <armadillo>
#include <functional>
#include "libKriging/Kriging.hpp"
#include "libKriging/utils/cache_details.hpp"

template <>
struct std::hash<arma::vec> {
  std::size_t operator()(const arma::vec& s) const noexcept {
    size_t result = 0;
    std::hash<arma::vec::value_type> hasher{};
    for (auto&& e : s) {
      size_t hash = hasher(e);
      result = details::composeHash(result, hash);
    }
    return result;
  }
};

template <>
struct std::hash<arma::mat> {
  std::size_t operator()(const arma::mat& s) const noexcept {
    size_t result = 0;
    std::hash<arma::mat::value_type> hasher{};
    for (auto&& e : s) {
      size_t hash = hasher(e);
      result = details::composeHash(result, hash);
    }
    return result;
  }
};

template <>
struct std::hash<arma::vec*> {
  std::size_t operator()(const arma::vec* s) const noexcept {
    if (s != nullptr)
      return std::hash<arma::vec>{}(*s);
    else
      return 0;
  }
};

template <>
struct std::hash<arma::mat*> {
  std::size_t operator()(const arma::mat* s) const noexcept {
    if (s != nullptr)
      return std::hash<arma::mat>{}(*s);
    else
      return 0;
  }
};

template <>
struct std::hash<Kriging::OKModel*> {
  std::size_t operator()(const Kriging::OKModel* s) const noexcept {
    if (s != nullptr) {
      struct OKModel {
        arma::mat T;
        arma::mat M;
        arma::colvec z;
        arma::colvec beta;
        bool estim_beta;
        double sigma2;
        bool estim_sigma2;
      };

      //      return std::hash<arma::mat>{}(*s);
    } else {
      return 0;
    }
  }
};

#endif  // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_HASH_FUNCTION_HPP
