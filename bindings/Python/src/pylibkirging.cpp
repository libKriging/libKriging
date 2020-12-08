#include <pybind11/pybind11.h>
#include <iostream>

#include "AddDemo.hpp"
#include "NumPyDemo.hpp"
#include "LinearRegression_binding.hpp"

#include <carma/carma.h>

namespace py = pybind11;

PYBIND11_MODULE(pylibkriging, m) {
  m.doc() = R"pbdoc(
        pylibkriging example plugin
        -----------------------

        .. currentmodule:: pylibkriging

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

  m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

  m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

  m.def("add_arrays", &add_arrays, "Add two NumPy arrays");

  m.attr("__version__") = VERSION_INFO;

  py::class_<PyLinearRegression>(m, "PyLinearRegression")
      .def(py::init<>())
      .def("fit", &PyLinearRegression::fit)
      .def("predict", &PyLinearRegression::predict);

  py::class_<LinearRegression>(m, "LinearRegression")
      .def(py::init<>())
      .def("fit", &LinearRegression::fit)
      .def("predict", &LinearRegression::predict);
}
