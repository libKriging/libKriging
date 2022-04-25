#include "Kriging_binding.hpp"

#include "libKriging/utils/lk_armadillo.hpp"

#include <carma>

#include <libKriging/Kriging.hpp>
#include <libKriging/Trend.hpp>

#include <random>

PyKriging::PyKriging(const std::string& kernel) : m_internal{new Kriging{kernel}} {}

PyKriging::PyKriging(const py::array_t<double>& y,
                     const py::array_t<double>& X,
                     const std::string& covType,
                     const Trend::RegressionModel& regmodel,
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
                    const Trend::RegressionModel& regmodel,
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

std::string PyKriging::summary() const {
  return m_internal->summary();
}

std::tuple<double, py::array_t<double>> PyKriging::leaveOneOutFun(const py::array_t<double>& theta,
                                                                  const bool want_grad) {
  arma::vec vec_theta = carma::arr_to_col<double>(theta);
  auto [llo, grad] = m_internal->leaveOneOutFun(vec_theta, want_grad);
  return {llo, carma::col_to_arr(grad)};
}

double PyKriging::leaveOneOut() {
  return m_internal->leaveOneOut();
}

std::tuple<double, py::array_t<double>, py::array_t<double>>
PyKriging::logLikelihoodFun(const py::array_t<double>& theta, const bool want_grad, const bool want_hess) {
  arma::vec vec_theta = carma::arr_to_col<double>(theta);
  auto [llo, grad, hess] = m_internal->logLikelihoodFun(vec_theta, want_grad, want_hess);
  return {
      llo,
      carma::col_to_arr(grad),
      // carma::mat_to_arr(hess)  // FIXME error in hessian transmission
      {}  //
  };
}

double PyKriging::logLikelihood() {
  return m_internal->logLikelihood();
}

std::tuple<double, py::array_t<double>> PyKriging::logMargPostFun(const py::array_t<double>& theta,
                                                                  const bool want_grad) {
  arma::vec vec_theta = carma::arr_to_col<double>(theta);
  auto [lmp, grad] = m_internal->logMargPostFun(vec_theta, want_grad);
  return {lmp, carma::col_to_arr(grad)};
}

double PyKriging::logMargPost() {
  return m_internal->logMargPost();
}
