// clang-format off
// Must before any other include
#include "libKriging/utils/lkalloc.hpp"

#include <RcppArmadillo.h>
// clang-format on

#include "libKriging/Gpu.hpp"

// GPU backend of the iterative method (method "iterative(...)"): Gpu.hpp,
// docs/dev/AllCMakeOptions.md. A CPU-only build (the default package)
// reports no backend and its settings have no effect.

//' @rdname gpu_settings
//' @export
// [[Rcpp::export]]
std::string gpu_compiled_backends() {
  return Gpu::compiled_backends();
}

//' @rdname gpu_settings
//' @export
// [[Rcpp::export]]
bool gpu_available() {
  return Gpu::available();
}

//' @rdname gpu_settings
//' @export
// [[Rcpp::export]]
std::string gpu_active_backend() {
  return Gpu::active_backend();
}

//' @rdname gpu_settings
//' @export
// [[Rcpp::export]]
bool gpu_get_enabled() {
  return Gpu::get_enabled();
}

//' @rdname gpu_settings
//' @export
// [[Rcpp::export]]
void gpu_set_enabled(bool enabled) {
  Gpu::set_enabled(enabled);
}

//' @rdname gpu_settings
//' @export
// [[Rcpp::export]]
int gpu_get_dense_max_mb() {
  return Gpu::get_dense_max_mb();
}

//' @rdname gpu_settings
//' @export
// [[Rcpp::export]]
void gpu_set_dense_max_mb(int mb) {
  Gpu::set_dense_max_mb(mb);
}

//' @rdname gpu_settings
//' @export
// [[Rcpp::export]]
bool gpu_get_mixed_precision() {
  return Gpu::get_mixed_precision();
}

//' @rdname gpu_settings
//' @export
// [[Rcpp::export]]
void gpu_set_mixed_precision(bool on) {
  Gpu::set_mixed_precision(on);
}

//' @rdname gpu_settings
//' @export
// [[Rcpp::export]]
bool gpu_get_verbose() {
  return Gpu::get_verbose();
}

//' @rdname gpu_settings
//' @export
// [[Rcpp::export]]
void gpu_set_verbose(bool on) {
  Gpu::set_verbose(on);
}
