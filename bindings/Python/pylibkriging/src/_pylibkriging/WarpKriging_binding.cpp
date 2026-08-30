#include "WarpKriging_binding.hpp"

#include "libKriging/utils/lk_armadillo.hpp"

#include <carma>

#include <libKriging/Trend.hpp>
#include <libKriging/WarpKriging.hpp>

#include <set>

namespace lk = libKriging;

// The recognised optimiser knobs, forwarded as a string map.
static const std::set<std::string> kWarpTuningKeys = {"adam_lr", "max_iter_adam", "max_iter_bfgs"};

// Deep-copy a numpy array/scalar into an owned arma::vec WITHOUT mutating the
// caller's array (carma::arr_to_col may steal the source buffer).
static arma::vec to_owned_col(const py::handle& obj) {
  auto arr = py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(obj);
  return arma::vec(carma::arr_to_col_view<double>(arr));
}

// Split a `parameters` dict into WarpKriging's typed seeds (numeric `theta` /
// `warp_params` / `noise`) and the string optimiser knobs. Unknown keys raise.
static std::pair<lk::WarpKriging::Parameters, std::map<std::string, std::string>> split_warp_parameters(
    const py::dict& dict) {
  lk::WarpKriging::Parameters wp;
  std::map<std::string, std::string> tuning;
  for (auto item : dict) {
    const std::string key = py::str(item.first);
    if (key == "theta")
      wp.theta = to_owned_col(item.second);
    else if (key == "warp_params")
      wp.warp_params = to_owned_col(item.second);
    else if (key == "noise")
      wp.noise = to_owned_col(item.second);
    else if (kWarpTuningKeys.count(key))
      tuning[key] = std::string(py::str(item.second));
    else
      throw std::invalid_argument("WarpKriging: unknown parameter '" + key + "'");
  }
  return {wp, tuning};
}

// Merge an explicit `noise=` argument (numpy array or scalar) into `wp.noise`.
static void merge_noise_arg(lk::WarpKriging::Parameters& wp, const py::object& noise) {
  if (noise.is_none())
    return;
  if (py::isinstance<py::str>(noise))
    throw std::invalid_argument(
        "WarpKriging: noise=\"nugget\" is not supported (no homogeneous-nugget estimation); "
        "pass a numeric per-observation noise-variance vector instead.");
  wp.noise = to_owned_col(noise);
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
                             const py::dict& parameters,
                             py::object noise) {
  arma::colvec mat_y = carma::arr_to_col_view<double>(y);
  arma::mat mat_X = carma::arr_to_mat_view<double>(X);
  m_internal = std::make_unique<lk::WarpKriging>(warping, kernel);
  auto [wparams, tuning] = split_warp_parameters(parameters);
  merge_noise_arg(wparams, noise);
  m_internal->fit(mat_y, mat_X, Trend::fromString(regmodel), normalize, optim, objective, wparams, tuning);
}

PyWarpKriging::~PyWarpKriging() {}

PyWarpKriging PyWarpKriging::copy() const {
  return PyWarpKriging(std::make_unique<lk::WarpKriging>(m_internal->clone_for_thread()));
}

void PyWarpKriging::fit(const py::array_t<double>& y,
                        const py::array_t<double>& X,
                        const std::string& regmodel,
                        bool normalize,
                        const std::string& optim,
                        const std::string& objective,
                        const py::dict& parameters,
                        py::object noise) {
  arma::colvec mat_y = carma::arr_to_col_view<double>(y);
  arma::mat mat_X = carma::arr_to_mat_view<double>(X);
  auto [wparams, tuning] = split_warp_parameters(parameters);
  merge_noise_arg(wparams, noise);
  m_internal->fit(mat_y, mat_X, Trend::fromString(regmodel), normalize, optim, objective, wparams, tuning);
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

py::array_t<double> PyWarpKriging::update_simulate(const py::array_t<double>& y_u,
                                                   const py::array_t<double>& X_u,
                                                   py::object noise_u) {
  arma::colvec mat_y = carma::arr_to_col<double>(y_u);
  arma::mat mat_X = carma::arr_to_mat<double>(X_u);
  arma::vec vec_noise_u;
  if (!noise_u.is_none())
    vec_noise_u = to_owned_col(noise_u);
  auto result = m_internal->update_simulate(mat_y, mat_X, vec_noise_u);
  return carma::mat_to_arr(result, true);
}

void PyWarpKriging::update(const py::array_t<double>& y_u,
                           const py::array_t<double>& X_u,
                           const bool refit,
                           py::object noise_u) {
  arma::colvec mat_y = carma::arr_to_col<double>(y_u);
  arma::mat mat_X = carma::arr_to_mat<double>(X_u);
  if (!noise_u.is_none()) {
    arma::vec vec_noise_u = to_owned_col(noise_u);
    m_internal->update(mat_y, mat_X, refit, vec_noise_u);
  } else {
    m_internal->update(mat_y, mat_X, refit);
  }
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

py::array_t<double> PyWarpKriging::covMat(const py::array_t<double>& X1, const py::array_t<double>& X2) {
  arma::mat mat_X1 = carma::arr_to_mat_view<double>(X1);
  arma::mat mat_X2 = carma::arr_to_mat_view<double>(X2);
  return carma::mat_to_arr(m_internal->covMat(mat_X1, mat_X2));
}

std::string PyWarpKriging::kernel() {
  return m_internal->kernel();
}

std::string PyWarpKriging::optim() {
  return m_internal->optim();
}

std::string PyWarpKriging::objective() {
  return m_internal->objective();
}

py::array_t<double> PyWarpKriging::noise() {
  return carma::col_to_arr(m_internal->noise());
}

py::array_t<double> PyWarpKriging::warp_params() {
  return carma::col_to_arr(m_internal->warp_params());
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
