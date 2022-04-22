#include <pybind11/pybind11.h>

#include "libKriging/utils/lk_armadillo.hpp"

#include <carma>
#include <iostream>
#include <libKriging/LinearRegression.hpp>

#include "BindingTest.hpp"
#include "Kriging_binding.hpp"
#include "LinearRegression_binding.hpp"
#include "NuggetKriging_binding.hpp"
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

  py::enum_<Trend::RegressionModel>(m, "RegressionModel")
      .value("Constant", Trend::RegressionModel::Constant)
      .value("Linear", Trend::RegressionModel::Linear)
      .value("Interactive", Trend::RegressionModel::Interactive)
      .value("Quadratic", Trend::RegressionModel::Quadratic)
      .export_values();

  // Quick and dirty manual wrapper (cf optional argument mapping)
  py::class_<Kriging::Parameters>(m, "KrigingParameters").def(py::init<>());

  // Quick and dirty manual wrapper (cf optional argument mapping)
  // Backup solution // FIXME remove it if not necessary
  py::class_<PyKriging>(m, "WrappedPyKriging")
      .def(py::init<const std::string&>())
      .def(py::init<const py::array_t<double>&,
                    const py::array_t<double>&,
                    const std::string&,
                    const Trend::RegressionModel&,
                    bool,
                    const std::string&,
                    const std::string&,
                    const Kriging::Parameters&>(),
           py::arg("y"),
           py::arg("X"),
           py::arg("kernel"),
           py::arg("regmodel") = Trend::RegressionModel::Constant,
           py::arg("normalize") = false,
           py::arg("optim") = "BFGS",
           py::arg("objective") = "LL",
           py::arg("parameters") = Kriging::Parameters{})
      .def("fit", &PyKriging::fit)
      .def("predict", &PyKriging::predict)
      .def("simulate", &PyKriging::simulate)
      .def("update", &PyKriging::update)
      .def("summary", &PyKriging::summary)
      .def("leaveOneOut", &PyKriging::leaveOneOutEval)
      .def("logLikelihood", &PyKriging::logLikelihoodEval)
      .def("logMargPost", &PyKriging::logMargPostEval);

  // Automated mapper
  py::class_<Kriging>(m, "PyKriging")
      .def(py::init<const std::string&>())
      .def(py::init<const arma::colvec&,
                    const arma::mat&,
                    const std::string&,
                    const Trend::RegressionModel&,
                    bool,
                    const std::string&,
                    const std::string&,
                    const Kriging::Parameters&>(),
           py::arg("y"),
           py::arg("X"),
           py::arg("kernel"),
           py::arg("regmodel") = Trend::RegressionModel::Constant,
           py::arg("normalize") = false,
           py::arg("optim") = "BFGS",
           py::arg("objective") = "LL",
           py::arg("parameters") = Kriging::Parameters{})
      .def("fit",
           &Kriging::fit,
           py::arg("y"),
           py::arg("X"),
           py::arg("regmodel") = Trend::RegressionModel::Constant,
           py::arg("normalize") = false,
           py::arg("optim") = "BFGS",
           py::arg("objective") = "LL",
           py::arg("parameters") = Kriging::Parameters{})
      .def("predict", &Kriging::predict)
      .def("simulate", &Kriging::simulate)
      .def("update", &Kriging::update)
      .def("summary", &Kriging::summary)
      .def("leaveOneOut", &Kriging::leaveOneOutEval)
      .def("logLikelihood", &Kriging::logLikelihoodEval)
      .def("logMargPost", &Kriging::logMargPostEval)

      .def("kernel", &Kriging::kernel)
      .def("optim", &Kriging::optim)
      .def("objective", &Kriging::objective)
      .def("X", &Kriging::X)
      .def("centerX", &Kriging::centerX)
      .def("scaleX", &Kriging::scaleX)
      .def("y", &Kriging::y)
      .def("centerY", &Kriging::centerY)
      .def("scaleY", &Kriging::scaleY)
      .def("regmodel", &Kriging::regmodel)
      .def("F", &Kriging::F)
      .def("T", &Kriging::T)
      .def("M", &Kriging::M)
      .def("z", &Kriging::z)
      .def("beta", &Kriging::beta)
      .def("is_beta_estim", &Kriging::is_beta_estim)
      .def("theta", &Kriging::theta)
      .def("is_theta_estim", &Kriging::is_theta_estim)
      .def("sigma2", &Kriging::sigma2)
      .def("is_sigma2_estim", &Kriging::is_sigma2_estim);

  // Quick and dirty manual wrapper (cf optional argument mapping)
  py::class_<NuggetKriging::Parameters>(m, "NuggetKrigingParameters").def(py::init<>());

  // Quick and dirty manual wrapper (cf optional argument mapping)
  // Backup solution // FIXME remove it if not necessary
  py::class_<PyNuggetKriging>(m, "WrappedPyNuggetKriging")
      .def(py::init<const std::string&>())
      .def(py::init<const py::array_t<double>&,
                    const py::array_t<double>&,
                    const std::string&,
                    const Trend::RegressionModel&,
                    bool,
                    const std::string&,
                    const std::string&,
                    const NuggetKriging::Parameters&>(),
           py::arg("y"),
           py::arg("X"),
           py::arg("kernel"),
           py::arg("regmodel") = Trend::RegressionModel::Constant,
           py::arg("normalize") = false,
           py::arg("optim") = "BFGS",
           py::arg("objective") = "LL",
           py::arg("parameters") = NuggetKriging::Parameters{})
      .def("fit", &PyNuggetKriging::fit)
      .def("predict", &PyNuggetKriging::predict)
      .def("simulate", &PyNuggetKriging::simulate)
      .def("update", &PyNuggetKriging::update)
      .def("summary", &PyNuggetKriging::summary)
      .def("logLikelihood", &PyNuggetKriging::logLikelihoodEval)
      .def("logMargPost", &PyNuggetKriging::logMargPostEval);

  // Automated mapper
  py::class_<NuggetKriging>(m, "PyNuggetKriging")
      .def(py::init<const std::string&>())
      .def(py::init<const arma::colvec&,
                    const arma::mat&,
                    const std::string&,
                    const Trend::RegressionModel&,
                    bool,
                    const std::string&,
                    const std::string&,
                    const NuggetKriging::Parameters&>(),
           py::arg("y"),
           py::arg("X"),
           py::arg("kernel"),
           py::arg("regmodel") = Trend::RegressionModel::Constant,
           py::arg("normalize") = false,
           py::arg("optim") = "BFGS",
           py::arg("objective") = "LL",
           py::arg("parameters") = NuggetKriging::Parameters{})
      .def("fit",
           &NuggetKriging::fit,
           py::arg("y"),
           py::arg("X"),
           py::arg("regmodel") = Trend::RegressionModel::Constant,
           py::arg("normalize") = false,
           py::arg("optim") = "BFGS",
           py::arg("objective") = "LL",
           py::arg("parameters") = NuggetKriging::Parameters{})
      .def("predict", &NuggetKriging::predict)
      .def("simulate", &NuggetKriging::simulate)
      .def("update", &NuggetKriging::update)
      .def("summary", &NuggetKriging::summary)
      .def("logLikelihood", &NuggetKriging::logLikelihoodEval)
      .def("logMargPost", &NuggetKriging::logMargPostEval)

      .def("kernel", &NuggetKriging::kernel)
      .def("optim", &NuggetKriging::optim)
      .def("objective", &NuggetKriging::objective)
      .def("X", &NuggetKriging::X)
      .def("centerX", &NuggetKriging::centerX)
      .def("scaleX", &NuggetKriging::scaleX)
      .def("y", &NuggetKriging::y)
      .def("centerY", &NuggetKriging::centerY)
      .def("scaleY", &NuggetKriging::scaleY)
      .def("regmodel", &NuggetKriging::regmodel)
      .def("F", &NuggetKriging::F)
      .def("T", &NuggetKriging::T)
      .def("M", &NuggetKriging::M)
      .def("z", &NuggetKriging::z)
      .def("beta", &NuggetKriging::beta)
      .def("is_beta_estim", &NuggetKriging::is_beta_estim)
      .def("theta", &NuggetKriging::theta)
      .def("is_theta_estim", &NuggetKriging::is_theta_estim)
      .def("sigma2", &NuggetKriging::sigma2)
      .def("is_sigma2_estim", &NuggetKriging::is_sigma2_estim)
      .def("nugget", &NuggetKriging::nugget)
      .def("is_nugget_estim", &NuggetKriging::is_nugget_estim);
}
