#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // required to use std::nullopt as default value

#include "libKriging/utils/lk_armadillo.hpp"

#include <carma>
#include <iostream>
#include <libKriging/KrigingLoader.hpp>
#include <libKriging/Optim.hpp>

// Should be included Only in Debug build
#include "ArrayBindingTest.hpp"
#include "DictTest.hpp"

#include "Kriging_binding.hpp"
#include "NestedKriging_binding.hpp"
#include "MLPKriging_binding.hpp"
#include "RandomGenerator.hpp"
#include "WarpKriging_binding.hpp"

// To compare string at compile time (before latest C++)
constexpr bool strings_equal(char const* a, char const* b) {
  return *a == *b && (*a == '\0' || strings_equal(a + 1, b + 1));
}

namespace py = pybind11;

static py::object load_any(const std::string& filename) {
  auto ktype = KrigingLoader::describe(filename);
  switch (ktype) {
    case KrigingLoader::KrigingType::Kriging:
      return py::cast(PyKriging::load(filename));
    case KrigingLoader::KrigingType::WarpKriging:
      return py::cast(PyWarpKriging::load(filename));
    case KrigingLoader::KrigingType::MLPKriging:
      return py::cast(PyMLPKriging::load(filename));
    default:
      throw std::runtime_error("Unknown Kriging type in file: " + filename);
  }
}

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

  m.def("load", &load_any, py::arg("filename"), "Load any Kriging model from file, auto-detecting its class.");

  // Basic tools
  py::class_<RandomGenerator>(m, "RandomGenerator")
      .def(py::init<unsigned int>())
      .def("uniform", &RandomGenerator::uniform);

  py::enum_<Trend::RegressionModel>(m, "RegressionModel")
      .value("None", Trend::RegressionModel::None)
      .value("Constant", Trend::RegressionModel::Constant)
      .value("Linear", Trend::RegressionModel::Linear)
      .value("Interactive", Trend::RegressionModel::Interactive)
      .value("Quadratic", Trend::RegressionModel::Quadratic)
      .export_values();

  const std::string default_regmodel = "constant";

/* --- NestedKriging --- */
  py::class_<PyNestedKriging>(m, "WrappedPyNestedKriging")
      .def(py::init<const std::string&>(), py::arg("kernel"))
      .def(py::init<const py::array_t<double>&,
                    const py::array_t<double>&,
                    const std::string&,
                    unsigned long,
                    const std::string&,
                    const std::string&,
                    int,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const py::dict&,
                    const std::vector<std::string>&>(),
           py::arg("y"),
           py::arg("X"),
           py::arg("kernel"),
           py::arg("nb_groups"),
           py::arg("aggregation") = "NK",
           py::arg("partition") = "kmeans",
           py::arg("seed") = 123,
           py::arg("regmodel") = default_regmodel,
           py::arg("optim") = default_optim,
           py::arg("objective") = default_objective,
           py::arg("parameters") = py::dict{},
           py::arg("warping") = std::vector<std::string>{})
      .def("fit",
           &PyNestedKriging::fit,
           py::arg("y"),
           py::arg("X"),
           py::arg("nb_groups"),
           py::arg("regmodel") = default_regmodel,
           py::arg("optim") = default_optim,
           py::arg("objective") = default_objective,
           py::arg("parameters") = py::dict{},
           py::arg("warping") = std::vector<std::string>{})
      .def("predict", &PyNestedKriging::predict, py::arg("X"), py::arg("return_stdev") = true)
      .def("summary", &PyNestedKriging::summary)
      .def("kernel", &PyNestedKriging::kernel)
      .def("warping", &PyNestedKriging::warping)
      .def("aggregation", &PyNestedKriging::aggregation)
      .def("nb_groups", &PyNestedKriging::nb_groups)
      .def("groups", &PyNestedKriging::groups)
      .def("theta", &PyNestedKriging::theta)
      .def("sigma2", &PyNestedKriging::sigma2)
      .def("beta0", &PyNestedKriging::beta0)
      .def("X", &PyNestedKriging::X)
      .def("y", &PyNestedKriging::y)
      .def("set_predict_chunk", &PyNestedKriging::set_predict_chunk, py::arg("chunk"))
      .def("__repr__", [](const PyNestedKriging& k) { return k.summary(); });

  const bool default_normalize = false;
  const std::string default_optim = "BFGS";
  const std::string default_objective = "LL";

  py::class_<Kriging::Parameters>(m, "KrigingParameters")
      .def(py::init<>())
      .def(py::init<std::optional<double>,
                    bool,
                    std::optional<arma::mat>,
                    bool,
                    std::optional<arma::vec>,
                    bool,
                    std::optional<double>,
                    bool>(),
           py::arg("sigma2") = std::nullopt,
           py::arg("is_sigma2_estim") = true,
           py::arg("theta") = std::nullopt,
           py::arg("is_theta_estim") = true,
           py::arg("beta") = std::nullopt,
           py::arg("is_beta_estim") = true,
           py::arg("nugget") = std::nullopt,
           py::arg("is_nugget_estim") = true);

  // Unified Kriging wrapper supporting None/Nugget/Heterogeneous noise modes
  py::class_<PyKriging>(m, "WrappedPyKriging")
      .def(py::init<const std::string&>(), py::arg("kernel"))
      .def(py::init<const std::string&, const std::string&>(), py::arg("kernel"), py::arg("noise_model"))
      .def(py::init<const py::array_t<double>&,
                    const py::array_t<double>&,
                    const std::string&,
                    const std::string&,
                    bool,
                    const std::string&,
                    const std::string&,
                    const py::dict&,
                    const py::object&>(),
           py::arg("y"),
           py::arg("X"),
           py::arg("kernel"),
           py::arg("regmodel") = default_regmodel,
           py::arg("normalize") = default_normalize,
           py::arg("optim") = default_optim,
           py::arg("objective") = default_objective,
           py::arg("parameters") = py::dict{},
           py::arg("noise") = py::none())
      .def(py::init<const PyKriging&>())
      .def("copy", &PyKriging::copy)
      .def("fit",
           &PyKriging::fit,
           py::arg("y"),
           py::arg("X"),
           py::arg("regmodel") = default_regmodel,
           py::arg("normalize") = default_normalize,
           py::arg("optim") = default_optim,
           py::arg("objective") = default_objective,
           py::arg("parameters") = py::dict{},
           py::arg("noise") = py::none())
      .def("predict",
           &PyKriging::predict,
           py::arg("X"),
           py::arg("return_stdev") = true,
           py::arg("return_cov") = false,
           py::arg("return_deriv") = false)
      .def("simulate",
           &PyKriging::simulate,
           py::arg("nsim") = 1,
           py::arg("seed") = 123,
           py::arg("X"),
           py::arg("will_update") = false,
           py::arg("with_noise") = py::none())
      .def("update",
           &PyKriging::update,
           py::arg("y_u"),
           py::arg("X_u"),
           py::arg("refit") = true,
           py::arg("noise_u") = py::none())
      .def("update_simulate",
           &PyKriging::update_simulate,
           py::arg("y_u"),
           py::arg("X_u"),
           py::arg("noise_u") = py::none())
      .def("summary", &PyKriging::summary)
      .def("save", &PyKriging::save)
      .def_static("load", &PyKriging::load)
      .def("leaveOneOutFun", &PyKriging::leaveOneOutFun)
      .def("leaveOneOutVec", &PyKriging::leaveOneOutVec)
      .def("logLikelihoodFun",
           &PyKriging::logLikelihoodFun,
           py::arg("theta"),
           py::arg("return_grad") = false,
           py::arg("want_hess") = false)
      .def("logMargPostFun", &PyKriging::logMargPostFun)
      .def("logLikelihood", &PyKriging::logLikelihood)
      .def("logMargPost", &PyKriging::logMargPost)
      .def("leaveOneOut", &PyKriging::leaveOneOut)
      .def("covMat", &PyKriging::covMat)
      .def("model", &PyKriging::model)

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
      .def("is_sigma2_estim", &PyKriging::is_sigma2_estim)
      .def("noise_model", &PyKriging::noise_model)
      .def("nugget", &PyKriging::nugget)
      .def("is_nugget_estim", &PyKriging::is_nugget_estim)
      .def("noise", &PyKriging::noise);

  const std::string default_warp_optim = "BFGS+Adam";

  py::class_<PyWarpKriging>(m, "WrappedPyWarpKriging")
      .def(py::init<const std::vector<std::string>&, const std::string&>(),
           py::arg("warping"),
           py::arg("kernel") = "gauss")
      .def(py::init<const py::array_t<double>&,
                    const py::array_t<double>&,
                    const std::vector<std::string>&,
                    const std::string&,
                    const std::string&,
                    bool,
                    const std::string&,
                    const std::string&,
                    const py::dict&,
                    py::object>(),
           py::arg("y"),
           py::arg("X"),
           py::arg("warping"),
           py::arg("kernel") = "gauss",
           py::arg("regmodel") = default_regmodel,
           py::arg("normalize") = default_normalize,
           py::arg("optim") = default_warp_optim,
           py::arg("objective") = default_objective,
           py::arg("parameters") = py::dict{},
           py::arg("noise") = py::none())
      .def("copy", &PyWarpKriging::copy)
      .def("save", &PyWarpKriging::save)
      .def_static("load", &PyWarpKriging::load)
      .def("fit",
           &PyWarpKriging::fit,
           py::arg("y"),
           py::arg("X"),
           py::arg("regmodel") = default_regmodel,
           py::arg("normalize") = default_normalize,
           py::arg("optim") = default_warp_optim,
           py::arg("objective") = default_objective,
           py::arg("parameters") = py::dict{},
           py::arg("noise") = py::none())
      .def("predict",
           &PyWarpKriging::predict,
           py::arg("X"),
           py::arg("return_stdev") = true,
           py::arg("return_cov") = false,
           py::arg("return_deriv") = false)
      .def("simulate",
           &PyWarpKriging::simulate,
           py::arg("nsim") = 1,
           py::arg("seed") = 123,
           py::arg("X"),
           py::arg("will_update") = false)
      .def("update_simulate", &PyWarpKriging::update_simulate, py::arg("y_u"), py::arg("X_u"))
      .def("update", &PyWarpKriging::update, py::arg("y_u"), py::arg("X_u"), py::arg("refit") = true)
      .def("summary", &PyWarpKriging::summary)
      .def("logLikelihood", &PyWarpKriging::logLikelihood)
      .def("logLikelihoodFun",
           &PyWarpKriging::logLikelihoodFun,
           py::arg("theta"),
           py::arg("return_grad") = false,
           py::arg("return_hess") = false)
      .def("kernel", &PyWarpKriging::kernel)
      .def("X", &PyWarpKriging::X)
      .def("centerX", &PyWarpKriging::centerX)
      .def("scaleX", &PyWarpKriging::scaleX)
      .def("y", &PyWarpKriging::y)
      .def("centerY", &PyWarpKriging::centerY)
      .def("scaleY", &PyWarpKriging::scaleY)
      .def("normalize", &PyWarpKriging::normalize)
      .def("regmodel", &PyWarpKriging::regmodel)
      .def("F", &PyWarpKriging::F)
      .def("T", &PyWarpKriging::T)
      .def("M", &PyWarpKriging::M)
      .def("z", &PyWarpKriging::z)
      .def("beta", &PyWarpKriging::beta)
      .def("theta", &PyWarpKriging::theta)
      .def("sigma2", &PyWarpKriging::sigma2)
      .def("is_fitted", &PyWarpKriging::is_fitted)
      .def("feature_dim", &PyWarpKriging::feature_dim)
      .def("warping", &PyWarpKriging::warping);

  py::class_<PyMLPKriging>(m, "WrappedPyMLPKriging")
      .def(py::init<const std::vector<std::size_t>&, std::size_t, const std::string&, const std::string&>(),
           py::arg("hidden_dims"),
           py::arg("d_out") = 2,
           py::arg("activation") = "selu",
           py::arg("kernel") = "gauss")
      .def(py::init<const py::array_t<double>&,
                    const py::array_t<double>&,
                    const std::vector<std::size_t>&,
                    std::size_t,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    bool,
                    const std::string&,
                    const std::string&,
                    const py::dict&>(),
           py::arg("y"),
           py::arg("X"),
           py::arg("hidden_dims"),
           py::arg("d_out") = 2,
           py::arg("activation") = "selu",
           py::arg("kernel") = "gauss",
           py::arg("regmodel") = default_regmodel,
           py::arg("normalize") = default_normalize,
           py::arg("optim") = default_warp_optim,
           py::arg("objective") = default_objective,
           py::arg("parameters") = py::dict{})
      .def("fit",
           &PyMLPKriging::fit,
           py::arg("y"),
           py::arg("X"),
           py::arg("regmodel") = default_regmodel,
           py::arg("normalize") = default_normalize,
           py::arg("optim") = default_warp_optim,
           py::arg("objective") = default_objective,
           py::arg("parameters") = py::dict{})
      .def("predict",
           &PyMLPKriging::predict,
           py::arg("X"),
           py::arg("return_stdev") = true,
           py::arg("return_cov") = false,
           py::arg("return_deriv") = false)
      .def("simulate",
           &PyMLPKriging::simulate,
           py::arg("nsim") = 1,
           py::arg("seed") = 123,
           py::arg("X"),
           py::arg("will_update") = false)
      .def("update_simulate", &PyMLPKriging::update_simulate, py::arg("y_u"), py::arg("X_u"))
      .def("update", &PyMLPKriging::update, py::arg("y_u"), py::arg("X_u"), py::arg("refit") = true)
      .def("summary", &PyMLPKriging::summary)
      .def("logLikelihood", &PyMLPKriging::logLikelihood)
      .def("logLikelihoodFun",
           &PyMLPKriging::logLikelihoodFun,
           py::arg("theta"),
           py::arg("return_grad") = false,
           py::arg("return_hess") = false)
      .def("kernel", &PyMLPKriging::kernel)
      .def("X", &PyMLPKriging::X)
      .def("centerX", &PyMLPKriging::centerX)
      .def("scaleX", &PyMLPKriging::scaleX)
      .def("y", &PyMLPKriging::y)
      .def("centerY", &PyMLPKriging::centerY)
      .def("scaleY", &PyMLPKriging::scaleY)
      .def("normalize", &PyMLPKriging::normalize)
      .def("regmodel", &PyMLPKriging::regmodel)
      .def("F", &PyMLPKriging::F)
      .def("T", &PyMLPKriging::T)
      .def("M", &PyMLPKriging::M)
      .def("z", &PyMLPKriging::z)
      .def("beta", &PyMLPKriging::beta)
      .def("theta", &PyMLPKriging::theta)
      .def("sigma2", &PyMLPKriging::sigma2)
      .def("is_fitted", &PyMLPKriging::is_fitted)
      .def("feature_dim", &PyMLPKriging::feature_dim)
      .def("hidden_dims", &PyMLPKriging::hidden_dims)
      .def("activation", &PyMLPKriging::activation)
      .def("copy", &PyMLPKriging::copy)
      .def("save", &PyMLPKriging::save)
      .def_static("load", &PyMLPKriging::load);

  // Optim class - static methods for optimization settings
  py::class_<Optim>(m, "Optim")
      .def(py::init<>())
      .def_static("is_reparametrized", &Optim::is_reparametrized)
      .def_static("use_reparametrize", &Optim::use_reparametrize)
      .def_static("get_theta_lower_factor", &Optim::get_theta_lower_factor)
      .def_static("set_theta_lower_factor", &Optim::set_theta_lower_factor)
      .def_static("get_theta_upper_factor", &Optim::get_theta_upper_factor)
      .def_static("set_theta_upper_factor", &Optim::set_theta_upper_factor)
      .def_static("variogram_bounds_heuristic_used", &Optim::variogram_bounds_heuristic_used)
      .def_static("use_variogram_bounds_heuristic", &Optim::use_variogram_bounds_heuristic)
      .def_static("get_log_level", &Optim::get_log_level)
      .def_static("set_log_level", &Optim::set_log_level)
      .def_static("get_max_iteration", &Optim::get_max_iteration)
      .def_static("set_max_iteration", &Optim::set_max_iteration)
      .def_static("get_gradient_tolerance", &Optim::get_gradient_tolerance)
      .def_static("set_gradient_tolerance", &Optim::set_gradient_tolerance)
      .def_static("get_objective_rel_tolerance", &Optim::get_objective_rel_tolerance)
      .def_static("set_objective_rel_tolerance", &Optim::set_objective_rel_tolerance)
      .def_static("get_thread_start_delay_ms", &Optim::get_thread_start_delay_ms)
      .def_static("set_thread_start_delay_ms", &Optim::set_thread_start_delay_ms)
      .def_static("get_thread_pool_size", &Optim::get_thread_pool_size)
      .def_static("set_thread_pool_size", &Optim::set_thread_pool_size);
}