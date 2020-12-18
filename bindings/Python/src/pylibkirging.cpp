#include <pybind11/pybind11.h>
#include <iostream>

#include <libKriging/Kriging.hpp>
#include <libKriging/LinearRegression.hpp>
#include "AddDemo.hpp"
#include "Kriging_binding.hpp"
#include "LinearRegression_binding.hpp"
#include "NumPyDemo.hpp"

#include <carma/carma.h>

// To compare string at compile time (before latest C++)
constexpr bool strings_equal(char const* a, char const* b) {
  return *a == *b && (*a == '\0' || strings_equal(a + 1, b + 1));
}

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

  if constexpr (strings_equal(BUILD_TYPE, "Debug")) {
    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def(
        "subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

    m.def("add_arrays", &add_arrays, "Add two NumPy arrays");
  }

  m.attr("__version__") = VERSION_INFO;
  m.attr("__build_type__") = BUILD_TYPE;

  // Custom manual wrapper (for testing)
  py::class_<PyLinearRegression>(m, "PyLinearRegression")
      .def(py::init<>())
      .def("fit", &PyLinearRegression::fit)
      .def("predict", &PyLinearRegression::predict);

  // Automated wrappers
  py::class_<LinearRegression>(m, "LinearRegression")
      .def(py::init<>())
      .def("fit", &LinearRegression::fit)
      .def("predict", &LinearRegression::predict);

  // Quick and dirty manual wrapper (cf optional argument mapping)
  py::class_<PyKriging>(m, "Kriging")
      .def(py::init<const std::string&>())
      .def("fit", &PyKriging::fit)
      .def("predict", &PyKriging::predict);
}
