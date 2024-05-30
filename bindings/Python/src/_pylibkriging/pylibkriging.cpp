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
#include "NoiseKriging_binding.hpp"
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
      .value("None", Trend::RegressionModel::None)
      .value("Constant", Trend::RegressionModel::Constant)
      .value("Linear", Trend::RegressionModel::Linear)
      .value("Interactive", Trend::RegressionModel::Interactive)
      .value("Quadratic", Trend::RegressionModel::Quadratic)
      .export_values();

  const std::string default_regmodel = "constant";
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
                    const std::string&,
                    bool,
                    const std::string&,
                    const std::string&,
                    const py::dict&>(),
           py::arg("y"),
           py::arg("X"),
           py::arg("kernel"),
           py::arg("regmodel") = default_regmodel,
           py::arg("normalize") = default_normalize,
           py::arg("optim") = default_optim,
           py::arg("objective") = default_objective,
           py::arg("parameters") = py::dict{})
      .def(py::init<const PyKriging&>())
      .def("copy", &PyKriging::copy)
      .def("fit", &PyKriging::fit)
      .def("predict", &PyKriging::predict)
      .def("simulate", &PyKriging::simulate)
      .def("update", &PyKriging::update)
      .def("update_simulate", &PyKriging::update_simulate)
      .def("summary", &PyKriging::summary)
      .def("save", &PyKriging::save)
      .def_static("load", &PyKriging::load)
      .def("leaveOneOutFun", &PyKriging::leaveOneOutFun)
      .def("leaveOneOutVec", &PyKriging::leaveOneOutVec)
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
      .def("normalize", &PyKriging::normalize)
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

  // Quick and dirty manual wrapper (cf optional argument mapping)
  py::class_<NuggetKriging::Parameters>(m, "NuggetKrigingParameters").def(py::init<>());

  // Quick and dirty manual wrapper (cf optional argument mapping)
  // Backup solution // FIXME remove it if not necessary
  py::class_<PyNuggetKriging>(m, "WrappedPyNuggetKriging")
      .def(py::init<const std::string&>())
      .def(py::init<const py::array_t<double>&,
                    const py::array_t<double>&,
                    const std::string&,
                    const std::string&,
                    bool,
                    const std::string&,
                    const std::string&,
                    const py::dict&>(),
           py::arg("y"),
           py::arg("X"),
           py::arg("kernel"),
           py::arg("regmodel") = default_regmodel,
           py::arg("normalize") = default_normalize,
           py::arg("optim") = default_optim,
           py::arg("objective") = default_objective,
           py::arg("parameters") = py::dict{})
      .def(py::init<const PyNuggetKriging&>())
      .def("copy", &PyNuggetKriging::copy)
      .def("fit", &PyNuggetKriging::fit)
      .def("predict", &PyNuggetKriging::predict)
      .def("simulate", &PyNuggetKriging::simulate)
      .def("update", &PyNuggetKriging::update)
      .def("update_simulate", &PyNuggetKriging::update_simulate)
      .def("summary", &PyNuggetKriging::summary)
      .def("save", &PyNuggetKriging::save)
      .def_static("load", &PyNuggetKriging::load)
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
      .def("normalize", &PyNuggetKriging::normalize)
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

  // Quick and dirty manual wrapper (cf optional argument mapping)
  py::class_<NoiseKriging::Parameters>(m, "NoiseKrigingParameters").def(py::init<>());

  // Quick and dirty manual wrapper (cf optional argument mapping)
  // Backup solution // FIXME remove it if not necessary
  py::class_<PyNoiseKriging>(m, "WrappedPyNoiseKriging")
      .def(py::init<const std::string&>())
      .def(py::init<const py::array_t<double>&,
                    const py::array_t<double>&,
                    const py::array_t<double>&,
                    const std::string&,
                    const std::string&,
                    bool,
                    const std::string&,
                    const std::string&,
                    const py::dict&>(),
           py::arg("y"),
           py::arg("noise"),
           py::arg("X"),
           py::arg("kernel"),
           py::arg("regmodel") = default_regmodel,
           py::arg("normalize") = default_normalize,
           py::arg("optim") = default_optim,
           py::arg("objective") = default_objective,
           py::arg("parameters") = py::dict{})
      .def(py::init<const PyNoiseKriging&>())
      .def("copy", &PyNoiseKriging::copy)
      .def("fit", &PyNoiseKriging::fit)
      .def("predict", &PyNoiseKriging::predict)
      .def("simulate", &PyNoiseKriging::simulate)
      .def("update", &PyNoiseKriging::update)
      .def("update_simulate", &PyNoiseKriging::update_simulate)
      .def("summary", &PyNoiseKriging::summary)
      .def("save", &PyNoiseKriging::save)
      .def_static("load", &PyNoiseKriging::load)
      .def("logLikelihoodFun", &PyNoiseKriging::logLikelihoodFun)
      .def("logLikelihood", &PyNoiseKriging::logLikelihood)

      .def("kernel", &PyNoiseKriging::kernel)
      .def("optim", &PyNoiseKriging::optim)
      .def("objective", &PyNoiseKriging::objective)
      .def("X", &PyNoiseKriging::X)
      .def("centerX", &PyNoiseKriging::centerX)
      .def("scaleX", &PyNoiseKriging::scaleX)
      .def("y", &PyNoiseKriging::y)
      .def("centerY", &PyNoiseKriging::centerY)
      .def("scaleY", &PyNoiseKriging::scaleY)
      .def("normalize", &PyNoiseKriging::normalize)
      .def("noise", &PyNoiseKriging::noise)
      .def("regmodel", &PyNoiseKriging::regmodel)
      .def("F", &PyNoiseKriging::F)
      .def("T", &PyNoiseKriging::T)
      .def("M", &PyNoiseKriging::M)
      .def("z", &PyNoiseKriging::z)
      .def("beta", &PyNoiseKriging::beta)
      .def("is_beta_estim", &PyNoiseKriging::is_beta_estim)
      .def("theta", &PyNoiseKriging::theta)
      .def("is_theta_estim", &PyNoiseKriging::is_theta_estim)
      .def("sigma2", &PyNoiseKriging::sigma2)
      .def("is_sigma2_estim", &PyNoiseKriging::is_sigma2_estim);
}