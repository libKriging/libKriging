#ifndef LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_RANDOM_HPP
#define LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_RANDOM_HPP

#include <random>

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/libKriging_exports.h"

class Random {

  private:

      static std::mt19937 engine;  // Mersenne twister random number engine

  public:

      static void set_seed(const int seed) {engine(seed)};

      static std::function<double()> runif;

      static std::function<arma::vec(const int)> runif_vec;

      static std::function<arma::mat(const int, const int)> runif_mat;
};

#endif  // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_RANDOM_HPP
