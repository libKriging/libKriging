// clang-format off
// Must before any other include
#include "libKriging/utils/lkalloc.hpp"

#include <RcppArmadillo.h>
// clang-format on

#include "libKriging/Parallel.hpp"

// Thread budget of the library and its distribution between the levels
// (multistart workers, OpenMP loops, BLAS threads): Parallel.hpp,
// docs/dev/parallel.md. The optim_* thread settings are the same settings.

//' @rdname parallel_settings
//' @export
// [[Rcpp::export]]
double parallel_get_num_threads_percent() {
  return Parallel::get_num_threads_percent();
}

//' @rdname parallel_settings
//' @export
// [[Rcpp::export]]
void parallel_set_num_threads_percent(double percent) {
  Parallel::set_num_threads_percent(percent);
}

//' @rdname parallel_settings
//' @export
// [[Rcpp::export]]
int parallel_get_thread_pool_size() {
  return Parallel::get_thread_pool_size();
}

//' @rdname parallel_settings
//' @export
// [[Rcpp::export]]
void parallel_set_thread_pool_size(int pool_size) {
  Parallel::set_thread_pool_size(pool_size);
}

//' @rdname parallel_settings
//' @export
// [[Rcpp::export]]
int parallel_get_thread_start_delay_ms() {
  return Parallel::get_thread_start_delay_ms();
}

//' @rdname parallel_settings
//' @export
// [[Rcpp::export]]
void parallel_set_thread_start_delay_ms(int delay_ms) {
  Parallel::set_thread_start_delay_ms(delay_ms);
}

//' @rdname parallel_settings
//' @export
// [[Rcpp::export]]
int parallel_get_blas_threads_min_n() {
  return Parallel::get_blas_threads_min_n();
}

//' @rdname parallel_settings
//' @export
// [[Rcpp::export]]
void parallel_set_blas_threads_min_n(int min_n) {
  Parallel::set_blas_threads_min_n(min_n);
}

//' @rdname parallel_settings
//' @export
// [[Rcpp::export]]
int parallel_get_omp_min_rows() {
  return Parallel::get_omp_min_rows();
}

//' @rdname parallel_settings
//' @export
// [[Rcpp::export]]
void parallel_set_omp_min_rows(int n) {
  Parallel::set_omp_min_rows(n);
}

//' @rdname parallel_settings
//' @export
// [[Rcpp::export]]
int parallel_get_num_threads() {
  return Parallel::get_num_threads();
}

//' @rdname parallel_settings
//' @export
// [[Rcpp::export]]
std::string parallel_blas_backend() {
  return Parallel::blas_backend();
}

//' @rdname parallel_settings
//' @export
// [[Rcpp::export]]
int parallel_blas_num_threads() {
  return Parallel::blas_num_threads();
}

//' @rdname parallel_settings
//' @export
// [[Rcpp::export]]
bool parallel_omp_enabled() {
  return Parallel::omp_enabled();
}
