#include "WarpKriging_binding.hpp"

#include "libKriging/utils/lk_armadillo.hpp"

#include <carma>

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
  m_internal = std::make_unique<lk::WarpKriging>(
      mat_y, mat_X, warping, kernel, regmodel, normalize, optim, objective, dict_to_string_map(parameters));
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
  m_internal->fit(mat_y, mat_X, regmodel, normalize, optim, objective, dict_to_string_map(parameters));
}

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
PyWarpKriging::predict(const py::array_t<double>& X_n, bool return_stdev, bool return_cov) {
  arma::mat mat_X = carma::arr_to_mat_view<double>(X_n);
  auto [mean, stdev, cov] = m_internal->predict(mat_X, return_stdev, return_cov);
  return std::make_tuple(carma::col_to_arr(mean, true),
                         carma::col_to_arr(stdev, true),
                         carma::mat_to_arr(cov, true));
}

py::array_t<double> PyWarpKriging::simulate(const int nsim, const int seed, const py::array_t<double>& X_n) {
  arma::mat mat_X = carma::arr_to_mat_view<double>(X_n);
  auto result = m_internal->simulate(nsim, static_cast<uint64_t>(seed), mat_X);
  return carma::mat_to_arr(result, true);
}

void PyWarpKriging::update(const py::array_t<double>& y_u, const py::array_t<double>& X_u) {
  arma::colvec mat_y = carma::arr_to_col<double>(y_u);
  arma::mat mat_X = carma::arr_to_mat<double>(X_u);
  m_internal->update(mat_y, mat_X);
}

std::string PyWarpKriging::summary() const {
  return m_internal->summary();
}

double PyWarpKriging::logLikelihood() {
  return m_internal->logLikelihood();
}

std::tuple<double, py::array_t<double>> PyWarpKriging::logLikelihoodFun(const py::array_t<double>& theta,
                                                                        const bool return_grad) {
  arma::vec vec_theta = carma::arr_to_col<double>(theta);
  auto [ll, grad, hess] = m_internal->logLikelihoodFun(vec_theta, return_grad, false);
  return {ll, carma::col_to_arr(grad)};
}

std::string PyWarpKriging::kernel() {
  return m_internal->kernel();
}

py::array_t<double> PyWarpKriging::X() {
  return carma::mat_to_arr(m_internal->X());
}

py::array_t<double> PyWarpKriging::y() {
  return carma::col_to_arr(m_internal->y());
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
