#ifndef LIBKRIGING_BINDINGS_PYTHON_SRC_KRIGING_BINDING_HPP
#define LIBKRIGING_BINDINGS_PYTHON_SRC_KRIGING_BINDING_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <libKriging/Kriging.hpp>
#include <string>
#include <tuple>

namespace py = pybind11;

class PyKriging {
 public:
  PyKriging(const std::string& kernel);
  ~PyKriging();

  void fit(const py::array_t<double>& y,
           const py::array_t<double>& X,
           const Kriging::RegressionModel& regmodel = Kriging::RegressionModel::Constant,
           bool normalize = false,
           const std::string& optim = "BFGS",
           const std::string& objective = "LL",
           const Kriging::Parameters& parameters = Kriging::Parameters{});

  // TODO The result should be a namedtuple
  // see
  // - https://docs.python.org/3/library/collections.html#namedtuple-factory-function-for-tuples-with-named-fields
  // - https://github.com/pybind/pybind11/issues/1244
  std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>> predict(const py::array_t<double>& X,
                                                                                    bool withStd,
                                                                                    bool withCov);

  std::tuple<double, py::array_t<double>> leaveOneOutEval(const py::array_t<double>& theta, const bool want_grad);

  std::tuple<double, py::array_t<double>, py::array_t<double>> logLikelihoodEval(const py::array_t<double>& theta,
                                                                                 const bool want_grad,
                                                                                 const bool want_hess);

  std::tuple<double, py::array_t<double>> logMargPostEval(const py::array_t<double>& theta, const bool want_grad);

 private:
  std::unique_ptr<Kriging> m_internal;
};

#endif  // LIBKRIGING_BINDINGS_PYTHON_SRC_KRIGING_BINDING_HPP
