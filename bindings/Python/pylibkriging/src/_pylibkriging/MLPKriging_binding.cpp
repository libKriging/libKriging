#include "MLPKriging_binding.hpp"

#include "libKriging/utils/lk_armadillo.hpp"

#include <carma>

#include <libKriging/MLPKriging.hpp>

namespace lk = libKriging;

static std::map<std::string, std::string> dict_to_string_map(const py::dict& dict) {
  std::map<std::string, std::string> result;
  for (auto item : dict) {
    result[py::str(item.first)] = py::str(item.second);
  }
  return result;
}

static std::vector<arma::uword> to_arma_uwords(const std::vector<std::size_t>& v) {
  std::vector<arma::uword> out;
  out.reserve(v.size());
  for (auto x : v)
    out.push_back(static_cast<arma::uword>(x));
  return out;
}

PyMLPKriging::PyMLPKriging(const std::vector<std::size_t>& hidden_dims,
                           std::size_t d_out,
                           const std::string& activation,
                           const std::string& kernel)
    : m_internal{std::make_unique<lk::MLPKriging>(
          to_arma_uwords(hidden_dims), static_cast<arma::uword>(d_out), activation, kernel)} {}

PyMLPKriging::PyMLPKriging(const py::array_t<double>& y,
                           const py::array_t<double>& X,
                           const std::vector<std::size_t>& hidden_dims,
                           std::size_t d_out,
                           const std::string& activation,
                           const std::string& kernel,
                           const std::string& regmodel,
                           bool normalize,
                           const std::string& optim,
                           const std::string& objective,
                           const py::dict& parameters) {
  arma::colvec mat_y = carma::arr_to_col_view<double>(y);
  arma::mat mat_X = carma::arr_to_mat_view<double>(X);
  m_internal = std::make_unique<lk::MLPKriging>(mat_y,
                                                mat_X,
                                                to_arma_uwords(hidden_dims),
                                                static_cast<arma::uword>(d_out),
                                                activation,
                                                kernel,
                                                regmodel,
                                                normalize,
                                                optim,
                                                objective,
                                                dict_to_string_map(parameters));
}

PyMLPKriging::~PyMLPKriging() {}

void PyMLPKriging::fit(const py::array_t<double>& y,
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

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>>
PyMLPKriging::predict(const py::array_t<double>& X_n, bool return_stdev, bool return_cov, bool return_deriv) {
  arma::mat mat_X = carma::arr_to_mat_view<double>(X_n);
  auto [mean, stdev, cov, mean_deriv, stdev_deriv]
      = m_internal->predict(mat_X, return_stdev, return_cov, return_deriv);
  return std::make_tuple(carma::col_to_arr(mean, true),
                         carma::col_to_arr(stdev, true),
                         carma::mat_to_arr(cov, true),
                         carma::mat_to_arr(mean_deriv, true),
                         carma::mat_to_arr(stdev_deriv, true));
}

py::array_t<double> PyMLPKriging::simulate(const int nsim, const int seed, const py::array_t<double>& X_n) {
  arma::mat mat_X = carma::arr_to_mat_view<double>(X_n);
  auto result = m_internal->simulate(nsim, static_cast<uint64_t>(seed), mat_X);
  return carma::mat_to_arr(result, true);
}

void PyMLPKriging::update(const py::array_t<double>& y_u, const py::array_t<double>& X_u) {
  arma::colvec mat_y = carma::arr_to_col<double>(y_u);
  arma::mat mat_X = carma::arr_to_mat<double>(X_u);
  m_internal->update(mat_y, mat_X);
}

std::string PyMLPKriging::summary() const {
  return m_internal->summary();
}

double PyMLPKriging::logLikelihood() {
  return m_internal->logLikelihood();
}

std::tuple<double, py::array_t<double>, py::array_t<double>> PyMLPKriging::logLikelihoodFun(
    const py::array_t<double>& theta,
    const bool return_grad,
    const bool want_hess) {
  arma::vec vec_theta = carma::arr_to_col<double>(theta);
  auto [ll, grad, hess] = m_internal->logLikelihoodFun(vec_theta, return_grad, want_hess);
  return {ll, carma::col_to_arr(grad), {}};
}

std::string PyMLPKriging::kernel() {
  return m_internal->kernel();
}

py::array_t<double> PyMLPKriging::X() {
  return carma::mat_to_arr(m_internal->X());
}

py::array_t<double> PyMLPKriging::y() {
  return carma::col_to_arr(m_internal->y());
}

py::array_t<double> PyMLPKriging::theta() {
  return carma::col_to_arr(m_internal->theta());
}

double PyMLPKriging::sigma2() {
  return m_internal->sigma2();
}

bool PyMLPKriging::is_fitted() {
  return m_internal->is_fitted();
}

int PyMLPKriging::feature_dim() {
  return static_cast<int>(m_internal->feature_dim());
}

std::vector<std::size_t> PyMLPKriging::hidden_dims() {
  const auto& hd = m_internal->hidden_dims();
  std::vector<std::size_t> out;
  out.reserve(hd.size());
  for (auto x : hd)
    out.push_back(static_cast<std::size_t>(x));
  return out;
}

std::string PyMLPKriging::activation() {
  return m_internal->activation();
}
