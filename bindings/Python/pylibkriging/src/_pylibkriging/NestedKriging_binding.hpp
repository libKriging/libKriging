#ifndef LIBKRIGING_BINDINGS_PYTHON_SRC_NESTEDKRIGING_BINDING_HPP
#define LIBKRIGING_BINDINGS_PYTHON_SRC_NESTEDKRIGING_BINDING_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <libKriging/NestedKriging.hpp>
#include <libKriging/Trend.hpp>

#include <memory>
#include <string>
#include <tuple>

namespace py = pybind11;

class PyNestedKriging {
 public:
  // Kernel-only constructor (no data)
  explicit PyNestedKriging(const std::string& kernel);

  // Full constructor with data
  PyNestedKriging(const py::array_t<double>& y,
                  const py::array_t<double>& X,
                  const std::string& kernel,
                  unsigned long nb_groups,
                  const std::string& aggregation,
                  const std::string& partition,
                  int seed,
                  const std::string& regmodel,
                  const std::string& optim,
                  const std::string& objective,
                  const py::dict& dict);
  ~PyNestedKriging();

  void fit(const py::array_t<double>& y,
           const py::array_t<double>& X,
           unsigned long nb_groups,
           const std::string& regmodel,
           const std::string& optim,
           const std::string& objective,
           const py::dict& dict);

  std::tuple<py::array_t<double>, py::array_t<double>> predict(const py::array_t<double>& X_n, bool return_stdev);

  [[nodiscard]] std::string summary() const;

  // accessors
  [[nodiscard]] std::string kernel() const;
  [[nodiscard]] std::string aggregation() const;
  [[nodiscard]] unsigned long nb_groups() const;
  [[nodiscard]] py::list groups() const;
  [[nodiscard]] py::array_t<double> theta() const;
  [[nodiscard]] double sigma2() const;
  [[nodiscard]] double beta0() const;
  [[nodiscard]] py::array_t<double> X() const;
  [[nodiscard]] py::array_t<double> y() const;

  void set_predict_chunk(unsigned long chunk);

 private:
  std::unique_ptr<NestedKriging> m_internal;
};

#endif  // LIBKRIGING_BINDINGS_PYTHON_SRC_NESTEDKRIGING_BINDING_HPP
