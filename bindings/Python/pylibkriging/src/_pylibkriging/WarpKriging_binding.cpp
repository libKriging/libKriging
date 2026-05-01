#include "WarpKriging_binding.hpp"

#include "libKriging/utils/lk_armadillo.hpp"

#include <carma>

#include <libKriging/Trend.hpp>
#include <libKriging/WarpKriging.hpp>

namespace lk = libKriging;

// Convert py::dict to std::map<std::string, std::string>
static std::map<std::string, std::string> dict_to_string_map(const py::dict& dict) {
  std::map<std::string, std::string> result;
  for (auto item : dict) {
    result[py::str(item.first)] = py::str(item.second);
  }
  return result;
}

PyWarpKriging::PyWarpKriging(const std::vector<std::string>& warping, const std::string& kernel)
    : m_internal{std::make_unique<lk::WarpKriging>(warping, kernel)} {}

PyWarpKriging::PyWarpKriging(const py::array_t<double>& y,
                             const py::array_t<double>& X,
                             const std::vector<std::string>& warping,
                             const std::string& kernel,
                             const std::string& regmodel,
                             bool normalize,
                             const std::string& optim,
                             const std::string& objective,
                             const py::dict& parameters) {
  arma::colvec mat_y = carma::arr_to_col_view<double>(y);
  arma::mat mat_X = carma::arr_to_mat_view<double>(X);
  m_internal = std::make_unique<lk::WarpKriging>(mat_y,
                                                 mat_X,
                                                 warping,
                                                 kernel,
                                                 Trend::fromString(regmodel),
                                                 normalize,
                                                 optim,
                                                 objective,
                                                 dict_to_string_map(parameters));
}

PyWarpKriging::~PyWarpKriging() {}

PyWarpKriging PyWarpKriging::copy() const {
  // WarpKriging is not copyable (contains unique_ptr members),
  // so we re-fit from the stored data.
  auto clone = std::make_unique<lk::WarpKriging>(m_internal->warping_strings(), m_internal->kernel());
  if (m_internal->is_fitted()) {
    clone->fit(m_internal->y(), m_internal->X());
  }
  return PyWarpKriging(std::move(clone));
}

void PyWarpKriging::fit(const py::array_t<double>& y,
                        const py::array_t<double>& X,
                        const std::string& regmodel,
                        bool normalize,
                        const std::string& optim,
                        const std::string& objective,
                        const py::dict& parameters) {
  arma::colvec mat_y = carma::arr_to_col_view<double>(y);
  arma::mat mat_X = carma::arr_to_mat_view<double>(X);
  m_internal->fit(mat_y, mat_X, Trend::fromString(regmodel), normalize, optim, objective, dict_to_string_map(parameters));
}

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>>
PyWarpKriging::predict(const py::array_t<double>& X_n, bool return_stdev, bool return_cov, bool return_deriv) {
  arma::mat mat_X = carma::arr_to_mat_view<double>(X_n);
  auto [mean, stdev, cov, mean_deriv, stdev_deriv] = m_internal->predict(mat_X, return_stdev, return_cov, return_deriv);
  return std::make_tuple(carma::col_to_arr(mean, true),
                         carma::col_to_arr(stdev, true),
                         carma::mat_to_arr(cov, true),
                         carma::mat_to_arr(mean_deriv, true),
                         carma::mat_to_arr(stdev_deriv, true));
}

py::array_t<double> PyWarpKriging::simulate(const int nsim,
                                            const int seed,
                                            const py::array_t<double>& X_n,
                                            const bool will_update) {
  arma::mat mat_X = carma::arr_to_mat_view<double>(X_n);
  auto result = m_internal->simulate(nsim, seed, mat_X, will_update);
  return carma::mat_to_arr(result, true);
}

py::array_t<double> PyWarpKriging::update_simulate(const py::array_t<double>& y_u, const py::array_t<double>& X_u) {
  arma::colvec mat_y = carma::arr_to_col<double>(y_u);
  arma::mat mat_X = carma::arr_to_mat<double>(X_u);
  auto result = m_internal->update_simulate(mat_y, mat_X);
  return carma::mat_to_arr(result, true);
}

void PyWarpKriging::update(const py::array_t<double>& y_u, const py::array_t<double>& X_u, const bool refit) {
  arma::colvec mat_y = carma::arr_to_col<double>(y_u);
  arma::mat mat_X = carma::arr_to_mat<double>(X_u);
  m_internal->update(mat_y, mat_X, refit);
}

std::string PyWarpKriging::summary() const {
  return m_internal->summary();
}

double PyWarpKriging::logLikelihood() {
  return m_internal->logLikelihood();
}

std::tuple<double, py::array_t<double>, py::array_t<double>>
PyWarpKriging::logLikelihoodFun(const py::array_t<double>& theta, const bool return_grad, const bool return_hess) {
  arma::vec vec_theta = carma::arr_to_col<double>(theta);
  auto [ll, grad, hess] = m_internal->logLikelihoodFun(vec_theta, return_grad, return_hess);
  return {ll, carma::col_to_arr(grad), carma::mat_to_arr(hess)};
}

std::string PyWarpKriging::kernel() {
  return m_internal->kernel();
}

py::array_t<double> PyWarpKriging::X() {
  return carma::mat_to_arr(m_internal->X());
}

py::array_t<double> PyWarpKriging::centerX() {
  return carma::row_to_arr(m_internal->centerX());
}

py::array_t<double> PyWarpKriging::scaleX() {
  return carma::row_to_arr(m_internal->scaleX());
}

py::array_t<double> PyWarpKriging::y() {
  return carma::col_to_arr(m_internal->y());
}

double PyWarpKriging::centerY() {
  return m_internal->centerY();
}

double PyWarpKriging::scaleY() {
  return m_internal->scaleY();
}

bool PyWarpKriging::normalize() {
  return m_internal->normalize();
}

std::string PyWarpKriging::regmodel() {
  return Trend::toString(m_internal->regmodel());
}

py::array_t<double> PyWarpKriging::F() {
  return carma::mat_to_arr(m_internal->F());
}

py::array_t<double> PyWarpKriging::T() {
  return carma::mat_to_arr(m_internal->T());
}

py::array_t<double> PyWarpKriging::M() {
  return carma::mat_to_arr(m_internal->M());
}

py::array_t<double> PyWarpKriging::z() {
  return carma::col_to_arr(m_internal->z());
}

py::array_t<double> PyWarpKriging::beta() {
  return carma::col_to_arr(m_internal->beta());
}

py::array_t<double> PyWarpKriging::theta() {
  return carma::col_to_arr(m_internal->theta());
}

double PyWarpKriging::sigma2() {
  return m_internal->sigma2();
}

bool PyWarpKriging::is_fitted() {
  return m_internal->is_fitted();
}

int PyWarpKriging::feature_dim() {
  return static_cast<int>(m_internal->feature_dim());
}

std::vector<std::string> PyWarpKriging::warping() {
  return m_internal->warping_strings();
}

void PyWarpKriging::save(const std::string filename) const {
  return m_internal->save(filename);
}

PyWarpKriging PyWarpKriging::load(const std::string filename) {
  return PyWarpKriging(std::make_unique<lk::WarpKriging>(lk::WarpKriging::load(filename)));
}
