#include <pybind11/pybind11.h>

#include "libKriging/utils/lk_armadillo.hpp"

#include <carma>
#include <iostream>
#include <libKriging/LinearRegression.hpp>

#include "BindingTest.hpp"
#include "Kriging_binding.hpp"
#include "LinearRegression_binding.hpp"
#include "RandomGenerator.hpp"

// To compare string at compile time (before latest C++)
constexpr bool strings_equal(char const* a, char const* b) {
  return *a == *b && (*a == '\0' || strings_equal(a + 1, b + 1));
}

namespace py = pybind11;

PYBIND11_MODULE(_pylibkriging, m) {
  // to avoid mixing allocators from default libKriging and Python
  lkalloc::set_allocation_functions(cnalloc::npy_malloc, cnalloc::npy_free);

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
    m.def("direct_binding", &direct_binding, R"pbdoc(
        Pure Numpy debugging demo

    )pbdoc");

    m.def("one_side_carma_binding", &one_side_carma_binding, R"pbdoc(
        Arma debugging demo

    )pbdoc");

    m.def("two_side_carma_binding", &two_side_carma_binding, R"pbdoc(
            libkriging link debugging demo
    
    )pbdoc");
  }

  m.attr("__version__") = KRIGING_VERSION_INFO;
  m.attr("__build_type__") = BUILD_TYPE;

  // Basic tools
  py::class_<RandomGenerator>(m, "RandomGenerator")
      .def(py::init<unsigned int>())
      .def("uniform", &RandomGenerator::uniform);

  // Custom manual wrapper (for testing)
  py::class_<PyLinearRegression>(m, "WrappedPyLinearRegression")
      .def(py::init<>())
      .def("fit", &PyLinearRegression::fit)
      .def("predict", &PyLinearRegression::predict);

  // Automated wrappers
  py::class_<LinearRegression>(m, "PyLinearRegression")
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
  py::class_<Kriging::Parameters>(m, "Parameters").def(py::init<>());

  // Quick and dirty manual wrapper (cf optional argument mapping)
  // Backup solution // FIXME remove it if not necessary
  py::class_<PyKriging>(m, "WrappedPyKriging")
      .def(py::init<const std::string&>())
      .def(py::init<const py::array_t<double>&,
                    const py::array_t<double>&,
                    const std::string&,
                    const Kriging::RegressionModel&,
                    bool,
                    const std::string&,
                    const std::string&,
                    const Kriging::Parameters&>(),
           py::arg("y"),
           py::arg("X"),
           py::arg("kernel"),
           py::arg("regmodel") = Kriging::RegressionModel::Constant,
           py::arg("normalize") = false,
           py::arg("optim") = "BFGS",
           py::arg("objective") = "LL",
           py::arg("parameters") = Kriging::Parameters{})
      .def("fit", &PyKriging::fit)
      .def("predict", &PyKriging::predict)
      .def("simulate", &PyKriging::simulate)
      .def("update", &PyKriging::update)
      .def("describeModel", &PyKriging::describeModel)
      .def("leaveOneOut", &PyKriging::leaveOneOutEval)
      .def("logLikelihood", &PyKriging::logLikelihoodEval)
      .def("logMargPost", &PyKriging::logMargPostEval);

  // Automated mapper
  py::class_<Kriging>(m, "PyKriging")
      .def(py::init<const std::string&>())
      .def(py::init<const arma::colvec&,
                    const arma::mat&,
                    const std::string&,
                    const Kriging::RegressionModel&,
                    bool,
                    const std::string&,
                    const std::string&,
                    const Kriging::Parameters&>(),
           py::arg("y"),
           py::arg("X"),
           py::arg("kernel"),
           py::arg("regmodel") = Kriging::RegressionModel::Constant,
           py::arg("normalize") = false,
           py::arg("optim") = "BFGS",
           py::arg("objective") = "LL",
           py::arg("parameters") = Kriging::Parameters{})
      .def("fit",
           &Kriging::fit,
           py::arg("y"),
           py::arg("X"),
           py::arg("regmodel") = Kriging::RegressionModel::Constant,
           py::arg("normalize") = false,
           py::arg("optim") = "BFGS",
           py::arg("objective") = "LL",
           py::arg("parameters") = Kriging::Parameters{})
      .def("predict", &Kriging::predict)
      .def("simulate", &Kriging::simulate)
      .def("update", &Kriging::update)
      .def("describeModel", &Kriging::describeModel)
      .def("leaveOneOut", &Kriging::leaveOneOutEval)
      .def("logLikelihood", &Kriging::logLikelihoodEval)
      .def("logMargPost", &Kriging::logMargPostEval);
}
