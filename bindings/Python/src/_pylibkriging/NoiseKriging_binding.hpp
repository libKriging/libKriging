#ifndef LIBKRIGING_BINDINGS_PYTHON_SRC_NOISEKRIGING_BINDING_HPP
#define LIBKRIGING_BINDINGS_PYTHON_SRC_NOISEKRIGING_BINDING_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <libKriging/NoiseKriging.hpp>
#include <libKriging/Trend.hpp>

#include <string>
#include <tuple>

namespace py = pybind11;

class PyNoiseKriging {
 private:
  PyNoiseKriging(std::unique_ptr<NoiseKriging>&& internal) : m_internal(std::move(internal)) {}

 public:
  PyNoiseKriging(const std::string& kernel);
  PyNoiseKriging(const py::array_t<double>& y,
                 const py::array_t<double>& noise,
                 const py::array_t<double>& X,
                 const std::string& covType,
                 const std::string& regmodel,
                 bool normalize,
                 const std::string& optim,
                 const std::string& objective,
                 const NoiseKriging::Parameters& parameters);
  PyNoiseKriging(const py::array_t<double>& y,
                 const py::array_t<double>& noise,
                 const py::array_t<double>& X,
                 const std::string& covType,
                 const std::string& regmodel,
                 bool normalize,
                 const std::string& optim,
                 const std::string& objective,
                 const py::dict& dict);
  ~PyNoiseKriging();

  PyNoiseKriging(const PyNoiseKriging& other);

  PyNoiseKriging copy() const;

  void fit(const py::array_t<double>& y,
           const py::array_t<double>& noise,
           const py::array_t<double>& X,
           const std::string& regmodel,
           bool normalize,
           const std::string& optim,
           const std::string& objective,
           const py::dict& dict);

  // TODO The result should be a namedtuple
  // see
  // - https://docs.python.org/3/library/collections.html#namedtuple-factory-function-for-tuples-with-named-fields
  // - https://github.com/pybind/pybind11/issues/1244
  std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>>
  predict(const py::array_t<double>& X, bool withStd, bool withCov, bool withDeriv);

  py::array_t<double> simulate(const int nsim, const int seed, const py::array_t<double>& Xp);

  void update(const py::array_t<double>& newy, const py::array_t<double>& newnoise, const py::array_t<double>& newX, const bool refit);

  std::string summary() const;

  void save(const std::string filename) const;

  static PyNoiseKriging load(const std::string filename);

  std::tuple<double, py::array_t<double>> logLikelihoodFun(const py::array_t<double>& theta_alpha,
                                                           const bool want_grad);

  double logLikelihood();

  std::string kernel();
  std::string optim();
  std::string objective();
  py::array_t<double> X();
  py::array_t<double> centerX();
  py::array_t<double> scaleX();
  py::array_t<double> y();
  double centerY();
  double scaleY();
  py::array_t<double> noise();
  bool normalize();
  std::string regmodel();  // Trend::toString(km->regmodel())
  py::array_t<double> F();
  py::array_t<double> T();
  py::array_t<double> M();
  py::array_t<double> z();
  py::array_t<double> beta();
  bool is_beta_estim();
  py::array_t<double> theta();
  bool is_theta_estim();
  double sigma2();
  bool is_sigma2_estim();

 private:
  std::unique_ptr<NoiseKriging> m_internal;
};

#endif  // LIBKRIGING_BINDINGS_PYTHON_SRC_NOISEKRIGING_BINDING_HPP
