#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // required to use std::nullopt as default value

#include "libKriging/utils/lk_armadillo.hpp"

#include <carma>
#include <iostream>
#include <libKriging/LinearRegression.hpp>

// Should be included Only in Debug build
#include "ArrayBindingTest.hpp"
#include "DictTest.hpp"

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

    m.def("check_dict_entry", &check_dict_entry, R"pbdoc(libkriging dict debugging demo)pbdoc");
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

  const Trend::RegressionModel default_regmodel = Trend::RegressionModel::Constant;
  const bool default_normalize = false;
  const std::string default_optim = "BFGS";
  const std::string default_objective = "LL";

  py::class_<Kriging::Parameters>(m, "KrigingParameters")
      .def(py::init<>())
      .def(py::init<std::optional<double>, bool, std::optional<arma::mat>, bool, std::optional<arma::vec>, bool>(),
           py::arg("sigma2") = std::nullopt,
           py::arg("is_sigma2_estim") = true,
           py::arg("theta") = std::nullopt,
           py::arg("is_theta_estim") = true,
           py::arg("beta") = std::nullopt,
           py::arg("is_beta_estim") = true);

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
           py::arg("regmodel") = default_regmodel,
           py::arg("normalize") = default_normalize,
           py::arg("optim") = default_optim,
           py::arg("objective") = default_objective,
           py::arg("parameters") = Kriging::Parameters{})
      .def("fit", &PyKriging::fit)
      .def("predict", &PyKriging::predict)
      .def("simulate", &PyKriging::simulate)
      .def("update", &PyKriging::update)
      .def("summary", &PyKriging::summary)
      .def("leaveOneOutFun", &PyKriging::leaveOneOutFun)
      .def("logLikelihoodFun", &PyKriging::logLikelihoodFun)
      .def("logMargPostFun", &PyKriging::logMargPostFun)
      .def("logLikelihood", &PyKriging::logLikelihood)
      .def("logMargPost", &PyKriging::logMargPost)

      .def("kernel", &PyKriging::kernel)
      .def("optim", &PyKriging::optim)
      .def("objective", &PyKriging::objective)
      .def("X", &PyKriging::X)
      .def("centerX", &PyKriging::centerX)
      .def("scaleX", &PyKriging::scaleX)
      .def("y", &PyKriging::y)
      .def("centerY", &PyKriging::centerY)
      .def("scaleY", &PyKriging::scaleY)
      .def("regmodel", &PyKriging::regmodel)
      .def("F", &PyKriging::F)
      .def("T", &PyKriging::T)
      .def("M", &PyKriging::M)
      .def("z", &PyKriging::z)
      .def("beta", &PyKriging::beta)
      .def("is_beta_estim", &PyKriging::is_beta_estim)
      .def("theta", &PyKriging::theta)
      .def("is_theta_estim", &PyKriging::is_theta_estim)
      .def("sigma2", &PyKriging::sigma2)
      .def("is_sigma2_estim", &PyKriging::is_sigma2_estim);

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
           py::arg("regmodel") = default_regmodel,
           py::arg("normalize") = default_normalize,
           py::arg("optim") = default_optim,
           py::arg("objective") = default_objective,
           py::arg("parameters") = Kriging::Parameters{})
      .def("fit",
           &Kriging::fit,
           py::arg("y"),
           py::arg("X"),
           py::arg("regmodel") = default_regmodel,
           py::arg("normalize") = default_normalize,
           py::arg("optim") = default_optim,
           py::arg("objective") = default_objective,
           py::arg("parameters") = Kriging::Parameters{})
      .def("predict", &Kriging::predict)
      .def("simulate", &Kriging::simulate)
      .def("update", &Kriging::update)
      .def("summary", &Kriging::summary)
      .def("leaveOneOutFun", &Kriging::leaveOneOutFun)
      .def("logLikelihoodFun", &Kriging::logLikelihoodFun)
      .def("logMargPostFun", &Kriging::logMargPostFun)
      .def("leaveOneOut", &Kriging::leaveOneOut)
      .def("logLikelihood", &Kriging::logLikelihood)
      .def("logMargPost", &Kriging::logMargPost)

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
           py::arg("regmodel") = default_regmodel,
           py::arg("normalize") = default_normalize,
           py::arg("optim") = default_optim,
           py::arg("objective") = default_objective,
           py::arg("parameters") = NuggetKriging::Parameters{})
      .def("fit", &PyNuggetKriging::fit)
      .def("predict", &PyNuggetKriging::predict)
      .def("simulate", &PyNuggetKriging::simulate)
      .def("update", &PyNuggetKriging::update)
      .def("summary", &PyNuggetKriging::summary)
      .def("logLikelihoodFun", &PyNuggetKriging::logLikelihoodFun)
      .def("logMargPostFun", &PyNuggetKriging::logMargPostFun)
      .def("logLikelihood", &PyNuggetKriging::logLikelihood)
      .def("logMargPost", &PyNuggetKriging::logMargPost)

      .def("kernel", &PyNuggetKriging::kernel)
      .def("optim", &PyNuggetKriging::optim)
      .def("objective", &PyNuggetKriging::objective)
      .def("X", &PyNuggetKriging::X)
      .def("centerX", &PyNuggetKriging::centerX)
      .def("scaleX", &PyNuggetKriging::scaleX)
      .def("y", &PyNuggetKriging::y)
      .def("centerY", &PyNuggetKriging::centerY)
      .def("scaleY", &PyNuggetKriging::scaleY)
      .def("regmodel", &PyNuggetKriging::regmodel)
      .def("F", &PyNuggetKriging::F)
      .def("T", &PyNuggetKriging::T)
      .def("M", &PyNuggetKriging::M)
      .def("z", &PyNuggetKriging::z)
      .def("beta", &PyNuggetKriging::beta)
      .def("is_beta_estim", &PyNuggetKriging::is_beta_estim)
      .def("theta", &PyNuggetKriging::theta)
      .def("is_theta_estim", &PyNuggetKriging::is_theta_estim)
      .def("sigma2", &PyNuggetKriging::sigma2)
      .def("is_sigma2_estim", &PyNuggetKriging::is_sigma2_estim)
      .def("nugget", &PyNuggetKriging::nugget)
      .def("is_nugget_estim", &PyNuggetKriging::is_nugget_estim);

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
           py::arg("regmodel") = default_regmodel,
           py::arg("normalize") = default_normalize,
           py::arg("optim") = default_optim,
           py::arg("objective") = default_objective,
           py::arg("parameters") = NuggetKriging::Parameters{})
      .def("fit",
           &NuggetKriging::fit,
           py::arg("y"),
           py::arg("X"),
           py::arg("regmodel") = default_regmodel,
           py::arg("normalize") = default_normalize,
           py::arg("optim") = default_optim,
           py::arg("objective") = default_objective,
           py::arg("parameters") = NuggetKriging::Parameters{})
      .def("predict", &NuggetKriging::predict)
      .def("simulate", &NuggetKriging::simulate)
      .def("update", &NuggetKriging::update)
      .def("summary", &NuggetKriging::summary)
      .def("logLikelihoodFun", &NuggetKriging::logLikelihoodFun)
      .def("logMargPostFun", &NuggetKriging::logMargPostFun)
      .def("logLikelihood", &NuggetKriging::logLikelihood)
      .def("logMargPost", &NuggetKriging::logMargPost)

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
