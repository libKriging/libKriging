#ifndef LIBKRIGING_BINDINGS_PYTHON_SRC_BINDINGTEST_HPP
#define LIBKRIGING_BINDINGS_PYTHON_SRC_BINDINGTEST_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::array_t<double> direct_binding(py::array_t<double>& input1, py::array_t<double>& input2);

py::array_t<double> one_side_carma_binding(py::array_t<double>& input1, py::array_t<double>& input2);

py::array_t<double> two_side_carma_binding(py::array_t<double>& input1, py::array_t<double>& input2);

#endif  // LIBKRIGING_BINDINGS_PYTHON_SRC_BINDINGTEST_HPP
