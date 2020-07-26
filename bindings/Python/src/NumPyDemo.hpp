//
// Created by Pascal Havé on 27/06/2020.
//

#ifndef LIBKRIGING_BINDINGS_PYTHON_SRC_NUMPYDEMO_HPP
#define LIBKRIGING_BINDINGS_PYTHON_SRC_NUMPYDEMO_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::array_t<double> add_arrays(py::array_t<double> input1, py::array_t<double> input2);
py::array_t<double> add_arrays2(py::array_t<double> input1, py::array_t<double> input2);

#endif  // LIBKRIGING_BINDINGS_PYTHON_SRC_NUMPYDEMO_HPP
