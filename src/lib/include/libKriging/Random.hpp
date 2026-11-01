#ifndef LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_RANDOM_HPP
#define LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_RANDOM_HPP

#include <random>

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/libKriging_exports.h"

// The engine and its seed are per thread: concurrent calls from user threads
// (a multistart fit draws its starts, simulate() reseeds then draws) do not
// race on them and give the results of the same calls run one after the
// other. reset_seed() applies to the calls of the thread that made it; a new
// thread starts from seed 123.
class Random {
 public:
  static thread_local std::mt19937 engine;  // Mersenne twister random number engine

  static thread_local unsigned int seed;

 public:
  LIBKRIGING_EXPORT static void reset_seed(unsigned int seed);

  static void init();

  LIBKRIGING_EXPORT static std::function<double()> randu;

  LIBKRIGING_EXPORT static std::function<arma::vec(const int)> randu_vec;

  LIBKRIGING_EXPORT static std::function<arma::mat(const int, const int)> randu_mat;

  LIBKRIGING_EXPORT static std::function<arma::mat(const int, const int)> randn_mat;
};

#endif  // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_RANDOM_HPP
