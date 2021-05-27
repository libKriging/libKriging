#include "Kriging_binding.hpp"

#include <carma>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <armadillo>
#include <libKriging/Kriging.hpp>
#include <random>

PyKriging::PyKriging(const std::string& kernel) : m_internal{new Kriging{kernel}} {}

PyKriging::~PyKriging() {}

void PyKriging::fit(const py::array_t<double>& y,
                    const py::array_t<double>& X,
                    const Kriging::RegressionModel& regmodel,
                    bool normalize,
                    const std::string& optim,
                    const std::string& objective,
                    const Kriging::Parameters& parameters) {
  arma::mat mat_y = carma::arr_to_col<double>(y);
  arma::mat mat_X = carma::arr_to_mat<double>(X);
  m_internal->fit(mat_y, mat_X, regmodel, normalize, optim, objective, parameters);
}

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
PyKriging::predict(const py::array_t<double>& X, bool withStd, bool withCov) {
  arma::mat mat_X = carma::arr_to_mat<double>(X);
  auto [y_predict, y_stderr, y_cov] = m_internal->predict(mat_X, withStd, withCov);
  return std::make_tuple(
      carma::col_to_arr(y_predict, true), carma::col_to_arr(y_stderr, true), carma::mat_to_arr(y_cov, true));
}

std::tuple<double, py::array_t<double>> PyKriging::leaveOneOutEval(const py::array_t<double>& theta,
                                                                   const bool want_grad) {
  arma::vec vec_theta = carma::arr_to_col<double>(theta);
  auto [llo, grad] = m_internal->leaveOneOutEval(vec_theta, want_grad);
  return {llo, carma::col_to_arr(grad)};
}

std::tuple<double, py::array_t<double>, py::array_t<double>>
PyKriging::logLikelihoodEval(const py::array_t<double>& theta, const bool want_grad, const bool want_hess) {
  arma::vec vec_theta = carma::arr_to_col<double>(theta);
  auto [llo, grad, hess] = m_internal->logLikelihoodEval(vec_theta, want_grad, want_hess);
  return {
      llo,
      carma::col_to_arr(grad),
      // carma::mat_to_arr(hess)  // FIXME error in hessian transmission
      {}  //
  };
}
