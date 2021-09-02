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
  PyKriging(const py::array_t<double>& y,
            const py::array_t<double>& X,
            const std::string& covType,
            const Kriging::RegressionModel& regmodel,
            bool normalize,
            const std::string& optim,
            const std::string& objective,
            const Kriging::Parameters& parameters);
  ~PyKriging();

  void fit(const py::array_t<double>& y,
           const py::array_t<double>& X,
           const Kriging::RegressionModel& regmodel,
           bool normalize,
           const std::string& optim,
           const std::string& objective,
           const Kriging::Parameters& parameters);

  // TODO The result should be a namedtuple
  // see
  // - https://docs.python.org/3/library/collections.html#namedtuple-factory-function-for-tuples-with-named-fields
  // - https://github.com/pybind/pybind11/issues/1244
  std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>> predict(const py::array_t<double>& X,
                                                                                    bool withStd,
                                                                                    bool withCov);

  py::array_t<double> simulate(const int nsim, const int seed, const py::array_t<double>& Xp);

  void update(const py::array_t<double>& newy, const py::array_t<double>& newX, bool normalize);

  std::string describeModel() const;

  std::tuple<double, py::array_t<double>> leaveOneOutEval(const py::array_t<double>& theta, const bool want_grad);

  std::tuple<double, py::array_t<double>, py::array_t<double>> logLikelihoodEval(const py::array_t<double>& theta,
                                                                                 const bool want_grad,
                                                                                 const bool want_hess);

  std::tuple<double, py::array_t<double>> logMargPostEval(const py::array_t<double>& theta, const bool want_grad);

 private:
  std::unique_ptr<Kriging> m_internal;
};

#endif  // LIBKRIGING_BINDINGS_PYTHON_SRC_KRIGING_BINDING_HPP
