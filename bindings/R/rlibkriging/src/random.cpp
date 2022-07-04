// clang-format off
// Must before any other include
#include "libKriging/utils/lkalloc.hpp"

#include <RcppArmadillo.h>
// clang-format on

#include "libKriging/Random.hpp"

// [[Rcpp::export]]
void random_reset_seed(unsigned int seed) {
  Random* r = new Random();
  Rcpp::XPtr<Random> impl_ptr(r);
  impl_ptr->reset_seed(seed);
}
