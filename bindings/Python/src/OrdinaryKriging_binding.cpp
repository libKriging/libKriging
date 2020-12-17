#include "OrdinaryKriging_binding.hpp"

#include <armadillo>
#include <libKriging/OrdinaryKriging.hpp>
#include <random>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <armadillo>

#include <carma/carma.h>

PyOrdinaryKriging::PyOrdinaryKriging(const std::string& kernel) : m_internal{new OrdinaryKriging{kernel}} {}

PyOrdinaryKriging::~PyOrdinaryKriging() {}

void PyOrdinaryKriging::fit(const py::array_t<double>& y, const py::array_t<double>& X) {
  arma::mat mat_y = carma::arr_to_col<double>(y, true);
  arma::mat mat_X = carma::arr_to_mat<double>(X, true);
  m_internal->fit(mat_y, mat_X);
}

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
PyOrdinaryKriging::predict(const py::array_t<double>& X, bool withStd, bool withCov) {
  arma::mat mat_X = carma::arr_to_mat<double>(X, true);
  auto [y_predict, y_stderr, y_cov] = m_internal->predict(mat_X, withStd, withCov);
  return std::make_tuple(
      carma::col_to_arr(y_predict, true), carma::col_to_arr(y_stderr, true), carma::mat_to_arr(y_cov, true));
}
