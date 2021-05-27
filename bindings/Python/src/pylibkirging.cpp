#include <pybind11/pybind11.h>

#include <iostream>
#include <libKriging/LinearRegression.hpp>

#include "Kriging_binding.hpp"
#include "LinearRegression_binding.hpp"
#include "NumPyDemo.hpp"
#include "RandomGenerator.hpp"

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
    m.def("add_arrays", &add_arrays, R"pbdoc(
        Add two NumPy arrays

        This is a demo for debugging numpy stuff with carma mapper
    )pbdoc");
  }

  m.attr("__version__") = KRIGING_VERSION_INFO;
  m.attr("__build_type__") = BUILD_TYPE;

  // Basic tools
  py::class_<RandomGenerator>(m, "RandomGenerator")
      .def(py::init<unsigned int>())
      .def("uniform", &RandomGenerator::uniform);

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

  py::enum_<Kriging::RegressionModel>(m, "RegressionModel")
      .value("Constant", Kriging::RegressionModel::Constant)
      .value("Linear", Kriging::RegressionModel::Linear)
      .value("Interactive", Kriging::RegressionModel::Interactive)
      .value("Quadratic", Kriging::RegressionModel::Quadratic)
      .export_values();

  // Quick and dirty manual wrapper (cf optional argument mapping)
  py::class_<PyKriging>(m, "Kriging2")
      .def(py::init<const std::string&>())
      .def("fit", &PyKriging::fit)
      .def("predict", &PyKriging::predict)
      .def("leaveOneOut", &PyKriging::leaveOneOutEval)  // TODO rename as leaveOneOutEval ??
      .def("logLikelihood", &PyKriging::logLikelihoodEval);

  // Automated mapper
  py::class_<Kriging>(m, "Kriging")
      .def(py::init<const std::string&>())
      .def("fit", &Kriging::fit)  // TODO mettre des arguments nommé optionnels: py::arg("name") =
      .def("predict", &Kriging::predict)
      .def("leaveOneOut", &Kriging::leaveOneOutEval)  // TODO rename as leaveOneOutEval ??
      .def("logLikelihood", &Kriging::logLikelihoodEval);

  // Quick and dirty manual wrapper (cf optional argument mapping)
  py::class_<Kriging::Parameters>(m, "Parameters").def(py::init<>());
}
