#ifndef LIBKRIGING_BINDINGS_PYTHON_SRC_MULTIOUTPUTKRIGING_BINDING_HPP
#define LIBKRIGING_BINDINGS_PYTHON_SRC_MULTIOUTPUTKRIGING_BINDING_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <libKriging/MultiOutputKriging.hpp>
#include <libKriging/Trend.hpp>

#include <memory>
#include <string>
#include <tuple>

#include "Kriging_binding.hpp"

namespace py = pybind11;

class PyMultiOutputKriging {
 public:
  // Kernel-only constructor (no data)
  PyMultiOutputKriging(const std::string& kernel, const std::string& output_model);

  // Full constructor with data
  PyMultiOutputKriging(const py::array_t<double>& Y,
                       const py::array_t<double>& X,
                       const std::string& kernel,
                       const std::string& output_model,
                       const std::string& regmodel,
                       bool normalize,
                       const std::string& optim,
                       const std::string& objective,
                       const py::dict& dict,
                       const py::object& output_coordinates);
  ~PyMultiOutputKriging();

  void fit(const py::array_t<double>& Y,
           const py::array_t<double>& X,
           const std::string& regmodel,
           bool normalize,
           const std::string& optim,
           const std::string& objective,
           const py::dict& dict);

  void set_output_coordinates(const py::array_t<double>& t);

  std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>>
  predict(const py::array_t<double>& X_n, bool return_stdev, bool return_cov, bool return_deriv);

  py::array_t<double> simulate(int nsim, int seed, const py::array_t<double>& X_n, bool will_update);
  py::array_t<double> update_simulate(const py::array_t<double>& Y_u, const py::array_t<double>& X_u);
  void update(const py::array_t<double>& Y_u, const py::array_t<double>& X_u, bool refit);

  std::tuple<py::array_t<double>, py::array_t<double>> leaveOneOutMat();
  double leaveOneOut();

  [[nodiscard]] std::string summary() const;

  // accessors
  [[nodiscard]] std::string kernel() const;
  [[nodiscard]] std::string output_model() const;
  [[nodiscard]] unsigned long nb_outputs() const;
  [[nodiscard]] py::array_t<double> X() const;
  [[nodiscard]] py::array_t<double> Y() const;
  [[nodiscard]] py::array_t<double> output_coordinates() const;
  [[nodiscard]] std::string regmodel() const;
  [[nodiscard]] bool normalize() const;
  [[nodiscard]] std::string optim() const;
  [[nodiscard]] std::string objective() const;
  [[nodiscard]] py::array_t<double> centerY() const;
  [[nodiscard]] py::array_t<double> scaleY() const;
  [[nodiscard]] unsigned long nb_components() const;
  [[nodiscard]] py::array_t<double> pca_basis() const;
  [[nodiscard]] py::array_t<double> pca_explained() const;
  [[nodiscard]] py::array_t<double> pca_residual() const;
  /// Copy of the k-th latent Kriging (independent of this model afterwards)
  [[nodiscard]] PyKriging component(unsigned long k) const;

 private:
  std::unique_ptr<MultiOutputKriging> m_internal;
};

#endif  // LIBKRIGING_BINDINGS_PYTHON_SRC_MULTIOUTPUTKRIGING_BINDING_HPP
