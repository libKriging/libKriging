// clang-format off
// Must before any other include
#include "libKriging/utils/lkalloc.hpp"

#include <RcppArmadillo.h>
// clang-format on

#include "libKriging/Gpu.hpp"

// [[Rcpp::export]]
std::string lk_gpu_compiled_backends() {
  return libKriging::gpu::compiled_backends();
}

// [[Rcpp::export]]
bool lk_gpu_available() {
  return libKriging::gpu::available();
}

// [[Rcpp::export]]
std::string lk_gpu_backend() {
  return libKriging::gpu::active_backend();
}

// [[Rcpp::export]]
bool lk_gpu_enabled() {
  return libKriging::gpu::enabled();
}

// [[Rcpp::export]]
void lk_set_gpu_enabled(bool value) {
  libKriging::gpu::set_enabled(value);
}
