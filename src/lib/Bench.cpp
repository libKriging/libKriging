// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio
#include <cmath>
// clang-format on

#include "libKriging/Bench.hpp"
#include "libKriging/Kriging.hpp"

#include "libKriging/utils/lk_armadillo.hpp"

#include <tuple>

// Usefull functions to evaluate time spent

std::chrono::high_resolution_clock::time_point Bench::tic() {
  return std::chrono::high_resolution_clock::now();
}

std::chrono::high_resolution_clock::time_point Bench::toc(std::map<std::string, double>* bench,
                                                          std::string what,
                                                          std::chrono::high_resolution_clock::time_point t0) {
  if (bench == nullptr)
    return t0;

  const auto t = std::chrono::high_resolution_clock::now();
  if ((*bench).count(what) > 0)
    (*bench)[what] += (std::chrono::duration<double>(t - t0)).count() * 1000;
  else
    (*bench)[what] = (std::chrono::duration<double>(t - t0)).count() * 1000;
  // arma::cout << what << ":     " << (std::chrono::duration<double>(t - t0)).count() * 1000 << arma::endl;
  return t;
}

std::string Bench::pad(std::string str, const size_t num, const char paddingChar = ' ') {
  if (num > str.size())
    return str.insert(str.size(), num - str.size(), paddingChar);
  return str;
}