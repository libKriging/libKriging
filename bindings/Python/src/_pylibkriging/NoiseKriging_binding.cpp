#include "NoiseKriging_binding.hpp"

#include "libKriging/utils/lk_armadillo.hpp"

#include <carma>

#include <libKriging/NoiseKriging.hpp>
#include <libKriging/Trend.hpp>

#include <random>
#include "py_to_cpp_cast.hpp"

PyNoiseKriging::PyNoiseKriging(const std::string& kernel) : m_internal{new NoiseKriging{kernel}} {}

PyNoiseKriging::PyNoiseKriging(const py::array_t<double>& y,
                               const py::array_t<double>& noise,
                               const py::array_t<double>& X,
                               const std::string& covType,
                               const std::string& regmodel,
                               bool normalize,
                               const std::string& optim,
                               const std::string& objective,
                               const NoiseKriging::Parameters& parameters) {
  arma::colvec mat_y = carma::arr_to_col_view<double>(y);
  arma::colvec mat_noise = carma::arr_to_col_view<double>(noise);
  arma::mat mat_X = carma::arr_to_mat_view<double>(X);
  m_internal = std::make_unique<NoiseKriging>(
      mat_y, mat_noise, mat_X, covType, Trend::fromString(regmodel), normalize, optim, objective, parameters);
}

PyNoiseKriging::PyNoiseKriging(const py::array_t<double>& y,
                               const py::array_t<double>& noise,
                               const py::array_t<double>& X,
                               const std::string& covType,
                               const std::string& regmodel,
                               bool normalize,
                               const std::string& optim,
                               const std::string& objective,
                               const py::dict& dict) {
  arma::colvec mat_y = carma::arr_to_col_view<double>(y);
  arma::colvec mat_noise = carma::arr_to_col_view<double>(noise);
  arma::mat mat_X = carma::arr_to_mat_view<double>(X);
  NoiseKriging::Parameters parameters{get_entry<arma::vec>(dict, "sigma2"),
                                      get_entry<bool>(dict, "is_sigma2_estim").value_or(true),
                                      get_entry<arma::mat>(dict, "theta"),
                                      get_entry<bool>(dict, "is_theta_estim").value_or(true),
                                      get_entry<arma::colvec>(dict, "beta"),
                                      get_entry<bool>(dict, "is_beta_estim").value_or(true)};
  m_internal = std::make_unique<NoiseKriging>(
      mat_y, mat_noise, mat_X, covType, Trend::fromString(regmodel), normalize, optim, objective, parameters);
}

PyNoiseKriging::~PyNoiseKriging() {}

PyNoiseKriging::PyNoiseKriging(const PyNoiseKriging& other)
    : m_internal{std::make_unique<NoiseKriging>(*other.m_internal, ExplicitCopySpecifier{})} {}

PyNoiseKriging PyNoiseKriging::copy() const {
  return PyNoiseKriging(*this);
}

void PyNoiseKriging::fit(const py::array_t<double>& y,
                         const py::array_t<double>& noise,
                         const py::array_t<double>& X,
                         const std::string& regmodel,
                         bool normalize,
                         const std::string& optim,
                         const std::string& objective,
                         const py::dict& dict) {
  arma::mat mat_y = carma::arr_to_col_view<double>(y);
  arma::mat mat_noise = carma::arr_to_col_view<double>(noise);
  arma::mat mat_X = carma::arr_to_mat_view<double>(X);
  NoiseKriging::Parameters parameters{get_entry<arma::vec>(dict, "sigma2"),
                                      get_entry<bool>(dict, "is_sigma2_estim").value_or(true),
                                      get_entry<arma::mat>(dict, "theta"),
                                      get_entry<bool>(dict, "is_theta_estim").value_or(true),
                                      get_entry<arma::colvec>(dict, "beta"),
                                      get_entry<bool>(dict, "is_beta_estim").value_or(true)};
  m_internal->fit(mat_y, mat_noise, mat_X, Trend::fromString(regmodel), normalize, optim, objective, parameters);
}

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>>
PyNoiseKriging::predict(const py::array_t<double>& X, bool return_stdev, bool return_cov, bool return_deriv) {
  arma::mat mat_X = carma::arr_to_mat_view<double>(X);
  auto [y_predict, y_stderr, y_cov, y_mean_deriv, y_stderr_deriv]
      = m_internal->predict(mat_X, return_stdev, return_cov, return_deriv);
  return std::make_tuple(carma::col_to_arr(y_predict, true),
                         carma::col_to_arr(y_stderr, true),
                         carma::mat_to_arr(y_cov, true),
                         carma::mat_to_arr(y_mean_deriv, true),
                         carma::mat_to_arr(y_stderr_deriv, true));
}

py::array_t<double> PyNoiseKriging::simulate(const int nsim, const int seed, const py::array_t<double>& Xp, const bool will_update) {
  arma::mat mat_X = carma::arr_to_mat_view<double>(Xp);
  auto result = m_internal->simulate(nsim, seed, mat_X, will_update);
  return carma::mat_to_arr(result, true);
}

void PyNoiseKriging::update(const py::array_t<double>& newy,
                            const py::array_t<double>& newnoise,
                            const py::array_t<double>& newX,
                             const bool refit) {
  arma::mat mat_y = carma::arr_to_col<double>(newy);
  arma::mat mat_noise = carma::arr_to_col<double>(newnoise);
  arma::mat mat_X = carma::arr_to_mat<double>(newX);
  m_internal->update(mat_y, mat_noise, mat_X, refit);
}

std::string PyNoiseKriging::summary() const {
  return m_internal->summary();
}

void PyNoiseKriging::save(const std::string filename) const {
  return m_internal->save(filename);
}

PyNoiseKriging PyNoiseKriging::load(const std::string filename) {
  return PyNoiseKriging(std::make_unique<NoiseKriging>(NoiseKriging::load(filename)));
}

std::tuple<double, py::array_t<double>> PyNoiseKriging::logLikelihoodFun(const py::array_t<double>& theta_sigma2,
                                                                         const bool want_grad) {
  arma::vec vec_theta_sigma2 = carma::arr_to_col<double>(theta_sigma2);
  auto [llo, grad] = m_internal->logLikelihoodFun(vec_theta_sigma2, want_grad, false);
  return {llo, carma::col_to_arr(grad)};
}

double PyNoiseKriging::logLikelihood() {
  return m_internal->logLikelihood();
}

std::string PyNoiseKriging::kernel() {
  return m_internal->kernel();
}

std::string PyNoiseKriging::optim() {
  return m_internal->optim();
}

std::string PyNoiseKriging::objective() {
  return m_internal->objective();
}

py::array_t<double> PyNoiseKriging::X() {
  return carma::mat_to_arr(m_internal->X());
}

py::array_t<double> PyNoiseKriging::centerX() {
  return carma::row_to_arr(m_internal->centerX());
}

py::array_t<double> PyNoiseKriging::scaleX() {
  return carma::row_to_arr(m_internal->scaleX());
}

py::array_t<double> PyNoiseKriging::y() {
  return carma::col_to_arr(m_internal->y());
}

double PyNoiseKriging::centerY() {
  return m_internal->centerY();
}

double PyNoiseKriging::scaleY() {
  return m_internal->scaleY();
}

py::array_t<double> PyNoiseKriging::noise() {
  return carma::col_to_arr(m_internal->noise());
}

bool PyNoiseKriging::normalize() {
  return m_internal->normalize();
}

std::string PyNoiseKriging::regmodel() {
  return Trend::toString(m_internal->regmodel());
}

py::array_t<double> PyNoiseKriging::F() {
  return carma::mat_to_arr(m_internal->F());
}

py::array_t<double> PyNoiseKriging::T() {
  return carma::mat_to_arr(m_internal->T());
}

py::array_t<double> PyNoiseKriging::M() {
  return carma::mat_to_arr(m_internal->M());
}

py::array_t<double> PyNoiseKriging::z() {
  return carma::col_to_arr(m_internal->z());
}

py::array_t<double> PyNoiseKriging::beta() {
  return carma::col_to_arr(m_internal->beta());
}

bool PyNoiseKriging::is_beta_estim() {
  return m_internal->is_beta_estim();
}

py::array_t<double> PyNoiseKriging::theta() {
  return carma::col_to_arr(m_internal->theta());
}

bool PyNoiseKriging::is_theta_estim() {
  return m_internal->is_theta_estim();
}

double PyNoiseKriging::sigma2() {
  return m_internal->sigma2();
}

bool PyNoiseKriging::is_sigma2_estim() {
  return m_internal->is_sigma2_estim();
}
