#include "NuggetKriging_binding.hpp"

#include "libKriging/utils/lk_armadillo.hpp"

#include <carma>

#include <libKriging/NuggetKriging.hpp>
#include <libKriging/Trend.hpp>

#include <random>

PyNuggetKriging::PyNuggetKriging(const std::string& kernel) : m_internal{new NuggetKriging{kernel}} {}

PyNuggetKriging::PyNuggetKriging(const py::array_t<double>& y,
                                 const py::array_t<double>& X,
                                 const std::string& covType,
                                 const Trend::RegressionModel& regmodel,
                                 bool normalize,
                                 const std::string& optim,
                                 const std::string& objective,
                                 const NuggetKriging::Parameters& parameters) {
  arma::colvec mat_y = carma::arr_to_col_view<double>(y);
  arma::mat mat_X = carma::arr_to_mat_view<double>(X);
  m_internal
      = std::make_unique<NuggetKriging>(mat_y, mat_X, covType, regmodel, normalize, optim, objective, parameters);
}

PyNuggetKriging::~PyNuggetKriging() {}

void PyNuggetKriging::fit(const py::array_t<double>& y,
                          const py::array_t<double>& X,
                          const Trend::RegressionModel& regmodel,
                          bool normalize,
                          const std::string& optim,
                          const std::string& objective,
                          const NuggetKriging::Parameters& parameters) {
  arma::mat mat_y = carma::arr_to_col_view<double>(y);
  arma::mat mat_X = carma::arr_to_mat_view<double>(X);
  m_internal->fit(mat_y, mat_X, regmodel, normalize, optim, objective, parameters);
}

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>>
PyNuggetKriging::predict(const py::array_t<double>& X, bool withStd, bool withCov, bool withDeriv) {
  arma::mat mat_X = carma::arr_to_mat_view<double>(X);
  auto [y_predict, y_stderr, y_cov, y_mean_deriv, y_stderr_deriv] = m_internal->predict(mat_X, withStd, withCov, withDeriv);
    return std::make_tuple(
      carma::col_to_arr(y_predict, true), 
      carma::col_to_arr(y_stderr, true), 
      carma::mat_to_arr(y_cov, true), 
      carma::mat_to_arr(y_mean_deriv, true), 
      carma::mat_to_arr(y_stderr_deriv, true));
}

py::array_t<double> PyNuggetKriging::simulate(const int nsim, const int seed, const py::array_t<double>& Xp) {
  arma::mat mat_X = carma::arr_to_mat_view<double>(Xp);
  auto result = m_internal->simulate(nsim, seed, mat_X);
  return carma::mat_to_arr(result, true);
}

void PyNuggetKriging::update(const py::array_t<double>& newy, const py::array_t<double>& newX) {
  arma::mat mat_y = carma::arr_to_col<double>(newy);
  arma::mat mat_X = carma::arr_to_mat<double>(newX);
  m_internal->update(mat_y, mat_X);
}

std::string PyNuggetKriging::summary() const {
  return m_internal->summary();
}

std::tuple<double, py::array_t<double>> PyNuggetKriging::logLikelihoodFun(
    const py::array_t<double>& theta_alpha,
    const bool want_grad) {
  arma::vec vec_theta_alpha = carma::arr_to_col<double>(theta_alpha);
  auto [llo, grad] = m_internal->logLikelihoodFun(vec_theta_alpha, want_grad);
  return {llo, carma::col_to_arr(grad)};
}

double PyNuggetKriging::logLikelihood() {
  return m_internal->logLikelihood();
}

std::tuple<double, py::array_t<double>> PyNuggetKriging::logMargPostFun(const py::array_t<double>& theta_alpha,
                                                                        const bool want_grad) {
  arma::vec vec_theta_alpha = carma::arr_to_col<double>(theta_alpha);
  auto [lmp, grad] = m_internal->logMargPostFun(vec_theta_alpha, want_grad);
  return {lmp, carma::col_to_arr(grad)};
}

double PyNuggetKriging::logMargPost() {
  return m_internal->logMargPost();
}

std::string PyNuggetKriging::kernel() {
  return m_internal->kernel();
}

std::string PyNuggetKriging::optim() {
  return m_internal->optim();
}

std::string PyNuggetKriging::objective() {
  return m_internal->objective();
}

py::array_t<double> PyNuggetKriging::X() {
  return carma::mat_to_arr(m_internal->X());
}

py::array_t<double> PyNuggetKriging::centerX() {
  return carma::row_to_arr(m_internal->centerX());
}

py::array_t<double> PyNuggetKriging::scaleX() {
  return carma::row_to_arr(m_internal->scaleX());
}

py::array_t<double> PyNuggetKriging::y() {
  return carma::col_to_arr(m_internal->y());
}

double PyNuggetKriging::centerY() {
  return m_internal->centerY();
}

double PyNuggetKriging::scaleY() {
  return m_internal->scaleY();
}

std::string PyNuggetKriging::regmodel() {
  return Trend::toString(m_internal->regmodel());
}

py::array_t<double> PyNuggetKriging::F() {
  return carma::mat_to_arr(m_internal->F());
}

py::array_t<double> PyNuggetKriging::T() {
  return carma::mat_to_arr(m_internal->T());
}

py::array_t<double> PyNuggetKriging::M() {
  return carma::mat_to_arr(m_internal->M());
}

py::array_t<double> PyNuggetKriging::z() {
  return carma::col_to_arr(m_internal->z());
}

py::array_t<double> PyNuggetKriging::beta() {
  return carma::col_to_arr(m_internal->beta());
}

bool PyNuggetKriging::is_beta_estim() {
  return m_internal->is_beta_estim();
}

py::array_t<double> PyNuggetKriging::theta() {
  return carma::col_to_arr(m_internal->theta());
}

bool PyNuggetKriging::is_theta_estim() {
  return m_internal->is_theta_estim();
}

double PyNuggetKriging::sigma2() {
  return m_internal->sigma2();
}

bool PyNuggetKriging::is_sigma2_estim() {
  return m_internal->is_sigma2_estim();
}

double PyNuggetKriging::nugget() {
  return m_internal->nugget();
}

bool PyNuggetKriging::is_nugget_estim() {
  return m_internal->is_nugget_estim();
}