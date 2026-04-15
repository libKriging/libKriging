#ifndef LIBKRIGING_BINDINGS_PYTHON_SRC_MLPKRIGING_BINDING_HPP
#define LIBKRIGING_BINDINGS_PYTHON_SRC_MLPKRIGING_BINDING_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <libKriging/MLPKriging.hpp>

#include <string>
#include <tuple>
#include <vector>

namespace py = pybind11;

class PyMLPKriging {
 private:
  PyMLPKriging(std::unique_ptr<libKriging::MLPKriging>&& internal) : m_internal(std::move(internal)) {}

 public:
  PyMLPKriging(const std::vector<std::size_t>& hidden_dims,
               std::size_t d_out,
               const std::string& activation,
               const std::string& kernel);

  PyMLPKriging(const py::array_t<double>& y,
               const py::array_t<double>& X,
               const std::vector<std::size_t>& hidden_dims,
               std::size_t d_out,
               const std::string& activation,
               const std::string& kernel,
               const std::string& regmodel,
               bool normalize,
               const std::string& optim,
               const std::string& objective,
               const py::dict& parameters);
  ~PyMLPKriging();

  void fit(const py::array_t<double>& y,
           const py::array_t<double>& X,
           const std::string& regmodel,
           bool normalize,
           const std::string& optim,
           const std::string& objective,
           const py::dict& parameters);

  std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>>
  predict(const py::array_t<double>& X_n, bool return_stdev, bool return_cov, bool return_deriv = false);

  py::array_t<double> simulate(const int nsim, const int seed, const py::array_t<double>& X_n);

  void update(const py::array_t<double>& y_u, const py::array_t<double>& X_u, const bool refit = true);

  std::string summary() const;

  double logLikelihood();

  std::tuple<double, py::array_t<double>, py::array_t<double>> logLikelihoodFun(const py::array_t<double>& theta,
                                                                                const bool return_grad,
                                                                                const bool return_hess);

  std::string kernel();
  py::array_t<double> X();
  py::array_t<double> y();
  py::array_t<double> theta();
  double sigma2();
  bool is_fitted();
  int feature_dim();
  std::vector<std::size_t> hidden_dims();
  std::string activation();

  [[nodiscard]] PyMLPKriging copy() const;

  void save(const std::string filename) const;
  static PyMLPKriging load(const std::string filename);

 private:
  std::unique_ptr<libKriging::MLPKriging> m_internal;
};

#endif  // LIBKRIGING_BINDINGS_PYTHON_SRC_MLPKRIGING_BINDING_HPP
