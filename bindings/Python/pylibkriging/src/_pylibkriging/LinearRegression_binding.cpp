#include "LinearRegression_binding.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <armadillo>
#include <carma>
#include <libKriging/LinearRegression.hpp>
#include <random>

PyLinearRegression::PyLinearRegression() : m_internal{new LinearRegression{}} {}

PyLinearRegression::~PyLinearRegression() {}

void PyLinearRegression::fit(const py::array_t<double>& y, const py::array_t<double>& X) {
  arma::mat mat_y = carma::arr_to_col<double>(y);
  arma::mat mat_X = carma::arr_to_mat<double>(X);
  m_internal->fit(mat_y, mat_X);
}

std::tuple<py::array_t<double>, py::array_t<double>> PyLinearRegression::predict(const py::array_t<double>& X) {
  arma::mat mat_X = carma::arr_to_mat<double>(X);
  auto [y_predict, y_stderr] = m_internal->predict(mat_X);
  return std::make_tuple(carma::col_to_arr(y_predict, true), carma::col_to_arr(y_stderr, true));
}
