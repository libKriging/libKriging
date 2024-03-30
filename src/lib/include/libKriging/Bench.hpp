#ifndef LIBKRIGING_BENCH_HPP
#define LIBKRIGING_BENCH_HPP

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/Kriging.hpp"
#include "libKriging/libKriging_exports.h"

class Bench {
 public:
  static const bool NO_BENCH = false;
  static std::chrono::high_resolution_clock::time_point tic();
  static std::chrono::high_resolution_clock::time_point toc(std::map<std::string, double>* bench,
                                                            std::string what,
                                                            std::chrono::high_resolution_clock::time_point t0);
  static std::string pad(std::string str, const size_t num, const char paddingChar);
};

#endif  // LIBKRIGING_BENCH_HPP
