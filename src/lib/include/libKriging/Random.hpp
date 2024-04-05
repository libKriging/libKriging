#ifndef LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_RANDOM_HPP
#define LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_RANDOM_HPP

#include <random>

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/libKriging_exports.h"

class Random {
 public:
  static std::mt19937 engine;  // Mersenne twister random number engine

  static unsigned int seed;

 public:
  LIBKRIGING_EXPORT static void reset_seed(unsigned int seed);

  static void init();

  LIBKRIGING_EXPORT static std::function<float()> randu;

  LIBKRIGING_EXPORT static std::function<arma::vec(const int)> randu_vec;

  LIBKRIGING_EXPORT static std::function<arma::mat(const int, const int)> randu_mat;

  LIBKRIGING_EXPORT static std::function<arma::mat(const int, const int)> randn_mat;
};

#endif  // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_RANDOM_HPP
