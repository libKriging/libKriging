#include "Kriging_binding.hpp"

#include "libKriging/utils/lk_armadillo.hpp"

#include <carma>
#include <libKriging/Kriging.hpp>
#include <random>

PyKriging::PyKriging(const std::string& kernel) : m_internal{new Kriging{kernel}} {}

PyKriging::PyKriging(const py::array_t<double>& y,
                     const py::array_t<double>& X,
                     const std::string& covType,
                     const Kriging::RegressionModel& regmodel,
                     bool normalize,
                     const std::string& optim,
                     const std::string& objective,
                     const Kriging::Parameters& parameters) {
  arma::colvec mat_y = carma::arr_to_col_view<double>(y);
  arma::mat mat_X = carma::arr_to_mat_view<double>(X);
  m_internal = std::make_unique<Kriging>(mat_y, mat_X, covType, regmodel, normalize, optim, objective, parameters);
}

PyKriging::~PyKriging() {}

void PyKriging::fit(const py::array_t<double>& y,
                    const py::array_t<double>& X,
                    const Kriging::RegressionModel& regmodel,
                    bool normalize,
                    const std::string& optim,
                    const std::string& objective,
                    const Kriging::Parameters& parameters) {
  arma::mat mat_y = carma::arr_to_col_view<double>(y);
  arma::mat mat_X = carma::arr_to_mat_view<double>(X);
  m_internal->fit(mat_y, mat_X, regmodel, normalize, optim, objective, parameters);
}

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
PyKriging::predict(const py::array_t<double>& X, bool withStd, bool withCov) {
  arma::mat mat_X = carma::arr_to_mat_view<double>(X);
  auto [y_predict, y_stderr, y_cov] = m_internal->predict(mat_X, withStd, withCov);
  return std::make_tuple(
      carma::col_to_arr(y_predict, true), carma::col_to_arr(y_stderr, true), carma::mat_to_arr(y_cov, true));
}

py::array_t<double> PyKriging::simulate(const int nsim, const int seed, const py::array_t<double>& Xp) {
  arma::mat mat_X = carma::arr_to_mat_view<double>(Xp);
  auto result = m_internal->simulate(nsim, seed, mat_X);
  return carma::mat_to_arr(result, true);
}

void PyKriging::update(const py::array_t<double>& newy, const py::array_t<double>& newX, bool normalize) {
  arma::mat mat_y = carma::arr_to_col<double>(newy);
  arma::mat mat_X = carma::arr_to_mat<double>(newX);
  m_internal->update(mat_y, mat_X, normalize);
}

std::string PyKriging::describeModel() const {
  return m_internal->describeModel();
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

std::tuple<double, py::array_t<double>> PyKriging::logMargPostEval(const py::array_t<double>& theta,
                                                                   const bool want_grad) {
  arma::vec vec_theta = carma::arr_to_col<double>(theta);
  auto [lmp, grad] = m_internal->logMargPostEval(vec_theta, want_grad);
  return {lmp, carma::col_to_arr(grad)};
}
