#ifndef LIBKRIGING_BINDINGS_PYTHON_SRC_NUGGETKRIGING_BINDING_HPP
#define LIBKRIGING_BINDINGS_PYTHON_SRC_NUGGETKRIGING_BINDING_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <libKriging/NuggetKriging.hpp>
#include <libKriging/Trend.hpp>

#include <string>
#include <tuple>

namespace py = pybind11;

class PyNuggetKriging {
 private:
  PyNuggetKriging(std::unique_ptr<NuggetKriging>&& internal) : m_internal(std::move(internal)) {}

 public:
  PyNuggetKriging(const std::string& kernel);
  PyNuggetKriging(const py::array_t<double>& y,
                  const py::array_t<double>& X,
                  const std::string& covType,
                  const std::string& regmodel,
                  bool normalize,
                  const std::string& optim,
                  const std::string& objective,
                  const NuggetKriging::Parameters& parameters);
  PyNuggetKriging(const py::array_t<double>& y,
                  const py::array_t<double>& X,
                  const std::string& covType,
                  const std::string& regmodel,
                  bool normalize,
                  const std::string& optim,
                  const std::string& objective,
                  const py::dict& dict);
  ~PyNuggetKriging();

  PyNuggetKriging(const PyNuggetKriging& other);

  PyNuggetKriging copy() const;

  void fit(const py::array_t<double>& y,
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
  predict(const py::array_t<double>& X_n, bool return_stdev, bool return_cov, bool return_deriv);

  py::array_t<double> simulate(const int nsim, const int seed, const py::array_t<double>& X_n, const bool with_nugget, const bool will_update);

  void update(const py::array_t<double>& y_u, const py::array_t<double>& X_u, const bool refit);

  std::string summary() const;

  void save(const std::string filename) const;

  static PyNuggetKriging load(const std::string filename);

  std::tuple<double, py::array_t<double>> logLikelihoodFun(const py::array_t<double>& theta_alpha,
                                                           const bool return_grad);

  std::tuple<double, py::array_t<double>> logMargPostFun(const py::array_t<double>& theta_alpha, const bool return_grad);

  double logLikelihood();
  double logMargPost();

  std::string kernel();
  std::string optim();
  std::string objective();
  py::array_t<double> X();
  py::array_t<double> centerX();
  py::array_t<double> scaleX();
  py::array_t<double> y();
  double centerY();
  double scaleY();
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
  double nugget();
  bool is_nugget_estim();

 private:
  std::unique_ptr<NuggetKriging> m_internal;
};

#endif  // LIBKRIGING_BINDINGS_PYTHON_SRC_NUGGETKRIGING_BINDING_HPP
