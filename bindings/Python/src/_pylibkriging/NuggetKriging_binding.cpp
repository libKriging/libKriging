#include "NuggetKriging_binding.hpp"

#include "libKriging/utils/lk_armadillo.hpp"

#include <carma>

#include <libKriging/NuggetKriging.hpp>
#include <libKriging/Trend.hpp>

#include <random>
#include "py_to_cpp_cast.hpp"

PyNuggetKriging::PyNuggetKriging(const std::string& kernel) : m_internal{new NuggetKriging{kernel}} {}

PyNuggetKriging::PyNuggetKriging(const py::array_t<double>& y,
                                 const py::array_t<double>& X,
                                 const std::string& covType,
                                 const std::string& regmodel,
                                 bool normalize,
                                 const std::string& optim,
                                 const std::string& objective,
                                 const NuggetKriging::Parameters& parameters) {
  arma::colvec mat_y = carma::arr_to_col_view<double>(y);
  arma::mat mat_X = carma::arr_to_mat_view<double>(X);
  m_internal = std::make_unique<NuggetKriging>(
      mat_y, mat_X, covType, Trend::fromString(regmodel), normalize, optim, objective, parameters);
}

PyNuggetKriging::PyNuggetKriging(const py::array_t<double>& y,
                                 const py::array_t<double>& X,
                                 const std::string& covType,
                                 const std::string& regmodel,
                                 bool normalize,
                                 const std::string& optim,
                                 const std::string& objective,
                                 const py::dict& dict) {
  arma::colvec mat_y = carma::arr_to_col_view<double>(y);
  arma::mat mat_X = carma::arr_to_mat_view<double>(X);
  NuggetKriging::Parameters parameters{get_entry<arma::vec>(dict, "nugget"),
                                       get_entry<bool>(dict, "is_nugget_estim").value_or(true),
                                       get_entry<arma::vec>(dict, "sigma2"),
                                       get_entry<bool>(dict, "is_sigma2_estim").value_or(true),
                                       get_entry<arma::mat>(dict, "theta"),
                                       get_entry<bool>(dict, "is_theta_estim").value_or(true),
                                       get_entry<arma::colvec>(dict, "beta"),
                                       get_entry<bool>(dict, "is_beta_estim").value_or(true)};
  m_internal = std::make_unique<NuggetKriging>(
      mat_y, mat_X, covType, Trend::fromString(regmodel), normalize, optim, objective, parameters);
}

PyNuggetKriging::~PyNuggetKriging() {}

PyNuggetKriging::PyNuggetKriging(const PyNuggetKriging& other)
    : m_internal{std::make_unique<NuggetKriging>(*other.m_internal, ExplicitCopySpecifier{})} {}

PyNuggetKriging PyNuggetKriging::copy() const {
  return PyNuggetKriging(*this);
}

void PyNuggetKriging::fit(const py::array_t<double>& y,
                          const py::array_t<double>& X,
                          const std::string& regmodel,
                          bool normalize,
                          const std::string& optim,
                          const std::string& objective,
                          const py::dict& dict) {
  arma::mat mat_y = carma::arr_to_col_view<double>(y);
  arma::mat mat_X = carma::arr_to_mat_view<double>(X);
  NuggetKriging::Parameters parameters{get_entry<arma::vec>(dict, "nugget"),
                                       get_entry<bool>(dict, "is_nugget_estim").value_or(true),
                                       get_entry<arma::vec>(dict, "sigma2"),
                                       get_entry<bool>(dict, "is_sigma2_estim").value_or(true),
                                       get_entry<arma::mat>(dict, "theta"),
                                       get_entry<bool>(dict, "is_theta_estim").value_or(true),
                                       get_entry<arma::colvec>(dict, "beta"),
                                       get_entry<bool>(dict, "is_beta_estim").value_or(true)};
  m_internal->fit(mat_y, mat_X, Trend::fromString(regmodel), normalize, optim, objective, parameters);
}

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>>
PyNuggetKriging::predict(const py::array_t<double>& X_n, bool return_stdev, bool return_cov, bool return_deriv) {
  arma::mat mat_X = carma::arr_to_mat_view<double>(X_n);
  auto [y_predict, y_stderr, y_cov, y_mean_deriv, y_stderr_deriv]
      = m_internal->predict(mat_X, return_stdev, return_cov, return_deriv);
  return std::make_tuple(carma::col_to_arr(y_predict, true),
                         carma::col_to_arr(y_stderr, true),
                         carma::mat_to_arr(y_cov, true),
                         carma::mat_to_arr(y_mean_deriv, true),
                         carma::mat_to_arr(y_stderr_deriv, true));
}

py::array_t<double> PyNuggetKriging::simulate(const int nsim,
                                              const int seed,
                                              const py::array_t<double>& X_n,
                                              const bool with_nugget,
                                              const bool will_update) {
  arma::mat mat_X = carma::arr_to_mat_view<double>(X_n);
  auto result = m_internal->simulate(nsim, seed, mat_X, with_nugget, will_update);
  return carma::mat_to_arr(result, true);
}

void PyNuggetKriging::update(const py::array_t<double>& y_u, const py::array_t<double>& X_u, const bool refit) {
  arma::mat mat_y = carma::arr_to_col<double>(y_u);
  arma::mat mat_X = carma::arr_to_mat<double>(X_u);
  m_internal->update(mat_y, mat_X, refit);
}

void PyNuggetKriging::update_simulate(const py::array_t<double>& y_u, const py::array_t<double>& X_u) {
  arma::mat mat_y = carma::arr_to_col<double>(y_u);
  arma::mat mat_X = carma::arr_to_mat<double>(X_u);
  m_internal->update_simulate(mat_y, mat_X);
}

std::string PyNuggetKriging::summary() const {
  return m_internal->summary();
}

void PyNuggetKriging::save(const std::string filename) const {
  return m_internal->save(filename);
}

PyNuggetKriging PyNuggetKriging::load(const std::string filename) {
  return PyNuggetKriging(std::make_unique<NuggetKriging>(NuggetKriging::load(filename)));
}

std::tuple<double, py::array_t<double>> PyNuggetKriging::logLikelihoodFun(const py::array_t<double>& theta_alpha,
                                                                          const bool return_grad) {
  arma::vec vec_theta_alpha = carma::arr_to_col<double>(theta_alpha);
  auto [llo, grad] = m_internal->logLikelihoodFun(vec_theta_alpha, return_grad, false);
  return {llo, carma::col_to_arr(grad)};
}

double PyNuggetKriging::logLikelihood() {
  return m_internal->logLikelihood();
}

std::tuple<double, py::array_t<double>> PyNuggetKriging::logMargPostFun(const py::array_t<double>& theta_alpha,
                                                                        const bool return_grad) {
  arma::vec vec_theta_alpha = carma::arr_to_col<double>(theta_alpha);
  auto [lmp, grad] = m_internal->logMargPostFun(vec_theta_alpha, return_grad, false);
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

bool PyNuggetKriging::normalize() {
  return m_internal->normalize();
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

py::array_t<double> PyNuggetKriging::covMat(const py::array_t<double>& X1, const py::array_t<double>& X2) {
  arma::mat mat_X1 = carma::arr_to_mat<double>(X1);
  arma::mat mat_X2 = carma::arr_to_mat<double>(X2);
  return carma::mat_to_arr(m_internal->covMat(mat_X1, mat_X2), true);
}

py::dict PyNuggetKriging::model() const {
  py::dict d;
  d["kernel"] = m_internal->kernel();
  d["optim"] = m_internal->optim();
  d["objective"] = m_internal->objective();
  d["theta"] = carma::col_to_arr(m_internal->theta());
  d["is_theta_estim"] = m_internal->is_theta_estim();
  d["sigma2"] = m_internal->sigma2();
  d["is_sigma2_estim"] = m_internal->is_sigma2_estim();
  d["nugget"] = m_internal->nugget();
  d["is_nugget_estim"] = m_internal->is_nugget_estim();
  d["X"] = carma::mat_to_arr(m_internal->X());
  d["centerX"] = carma::row_to_arr(m_internal->centerX());
  d["scaleX"] = carma::row_to_arr(m_internal->scaleX());
  d["y"] = carma::col_to_arr(m_internal->y());
  d["centerY"] = m_internal->centerY();
  d["scaleY"] = m_internal->scaleY();
  d["normalize"] = m_internal->normalize();
  d["regmodel"] = Trend::toString(m_internal->regmodel());
  d["beta"] = carma::col_to_arr(m_internal->beta());
  d["is_beta_estim"] = m_internal->is_beta_estim();
  d["F"] = carma::mat_to_arr(m_internal->F());
  d["T"] = carma::mat_to_arr(m_internal->T());
  d["M"] = carma::mat_to_arr(m_internal->M());
  d["z"] = carma::col_to_arr(m_internal->z());
  return d;
}
