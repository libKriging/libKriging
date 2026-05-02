#ifndef LIBKRIGING_BINDINGS_PYTHON_SRC_KRIGING_BINDING_HPP
#define LIBKRIGING_BINDINGS_PYTHON_SRC_KRIGING_BINDING_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <libKriging/Kriging.hpp>
#include <libKriging/Trend.hpp>

#include <string>
#include <tuple>

namespace py = pybind11;

class PyKriging {
 private:
  PyKriging(std::unique_ptr<Kriging>&& internal) : m_internal(std::move(internal)) {}

 public:
  // Kernel-only constructor (no data)
  PyKriging(const std::string& kernel);
  // Kernel + noise_model constructor (no data)
  PyKriging(const std::string& kernel, const std::string& noise_model);

  // Full constructors with data (noise=py::none() for None/Nugget modes)
  PyKriging(const py::array_t<double>& y,
            const py::array_t<double>& X,
            const std::string& covType,
            const std::string& regmodel,
            bool normalize,
            const std::string& optim,
            const std::string& objective,
            const py::dict& dict,
            const py::object& noise);
  ~PyKriging();

  PyKriging(const PyKriging& other);

  [[nodiscard]] PyKriging copy() const;

  void fit(const py::array_t<double>& y,
           const py::array_t<double>& X,
           const std::string& regmodel,
           bool normalize,
           const std::string& optim,
           const std::string& objective,
           const py::dict& dict,
           const py::object& noise);

  std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>>
  predict(const py::array_t<double>& X_n, bool return_stdev, bool return_cov, bool return_deriv);

  // with_noise: None => plain, True => with nugget, array => heterogeneous noise
  py::array_t<double> simulate(const int nsim,
                                const int seed,
                                const py::array_t<double>& X_n,
                                const bool will_update,
                                const py::object& with_noise);

  // noise_u: py::none() for None/Nugget, array for Heterogeneous
  void update(const py::array_t<double>& y_u,
              const py::array_t<double>& X_u,
              const bool refit,
              const py::object& noise_u);

  py::array_t<double> update_simulate(const py::array_t<double>& y_u,
                                       const py::array_t<double>& X_u,
                                       const py::object& noise_u);

  std::string summary() const;

  void save(const std::string filename) const;

  static PyKriging load(const std::string filename);

  std::tuple<double, py::array_t<double>> leaveOneOutFun(const py::array_t<double>& theta, const bool return_grad);

  std::tuple<py::array_t<double>, py::array_t<double>> leaveOneOutVec(const py::array_t<double>& theta);

  std::tuple<double, py::array_t<double>, py::array_t<double>> logLikelihoodFun(
      const py::array_t<double>& theta,
      const bool return_grad,
      const bool want_hess = false);

  std::tuple<double, py::array_t<double>> logMargPostFun(const py::array_t<double>& theta, const bool return_grad);

  double logLikelihood();
  double leaveOneOut();
  double logMargPost();

  py::array_t<double> covMat(const py::array_t<double>& X1, const py::array_t<double>& X2);

  py::dict model() const;

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
  std::string regmodel();
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

  // Noise-related accessors
  std::string noise_model();
  double nugget();
  bool is_nugget_estim();
  py::array_t<double> noise();

 private:
  std::unique_ptr<Kriging> m_internal;

  static Kriging::NoiseModel parse_noise_model(const std::string& nm);
  static std::string noise_model_to_string(Kriging::NoiseModel nm);
};

#endif  // LIBKRIGING_BINDINGS_PYTHON_SRC_KRIGING_BINDING_HPP
