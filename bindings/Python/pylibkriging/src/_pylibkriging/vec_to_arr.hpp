#ifndef LIBKRIGING_BINDINGS_PYTHON_SRC__PYLIBKRIGING_VEC_TO_ARR_HPP
#define LIBKRIGING_BINDINGS_PYTHON_SRC__PYLIBKRIGING_VEC_TO_ARR_HPP

#include "libKriging/utils/lk_armadillo.hpp"  // should always be before any armadillo include

#include <pybind11/numpy.h>

/// Armadillo column or row vector -> 1-D numpy array (copy).
/// Every vector-valued output of the Python binding uses it, so that a
/// length-n result has shape (n,) for all classes (carma's col_to_arr /
/// row_to_arr give (n, 1) / (1, n)). Matrices keep using carma::mat_to_arr.
template <typename eT>
pybind11::array_t<eT> vec_to_arr(const arma::Col<eT>& v) {
  return pybind11::array_t<eT>(static_cast<pybind11::ssize_t>(v.n_elem), v.memptr());
}

template <typename eT>
pybind11::array_t<eT> vec_to_arr(const arma::Row<eT>& v) {
  return pybind11::array_t<eT>(static_cast<pybind11::ssize_t>(v.n_elem), v.memptr());
}

#endif  // LIBKRIGING_BINDINGS_PYTHON_SRC__PYLIBKRIGING_VEC_TO_ARR_HPP
