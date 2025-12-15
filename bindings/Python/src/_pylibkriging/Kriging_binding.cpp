#include "Kriging_binding.hpp"

#include "libKriging/utils/lk_armadillo.hpp"

#include <carma>

#include <libKriging/Kriging.hpp>
#include <libKriging/Trend.hpp>
#include "py_to_cpp_cast.hpp"

#include <random>

PyKriging::PyKriging(const std::string& kernel) : m_internal{new Kriging{kernel}} {}

PyKriging::PyKriging(const py::array_t<double>& y,
                     const py::array_t<double>& X,
                     const std::string& covType,
                     const std::string& regmodel,
                     bool normalize,
                     const std::string& optim,
                     const std::string& objective,
                     const Kriging::Parameters& parameters) {
  arma::colvec mat_y = carma::arr_to_col_view<double>(y);
  arma::mat mat_X = carma::arr_to_mat_view<double>(X);
  m_internal = std::make_unique<Kriging>(
      mat_y, mat_X, covType, Trend::fromString(regmodel), normalize, optim, objective, parameters);
}

PyKriging::PyKriging(const py::array_t<double>& y,
                     const py::array_t<double>& X,
                     const std::string& covType,
                     const std::string& regmodel,
                     bool normalize,
                     const std::string& optim,
                     const std::string& objective,
                     const py::dict& dict) {
  arma::colvec mat_y = carma::arr_to_col_view<double>(y);
  arma::mat mat_X = carma::arr_to_mat_view<double>(X);
  Kriging::Parameters parameters{get_entry<double>(dict, "sigma2"),
                                 get_entry<bool>(dict, "is_sigma2_estim").value_or(true),
                                 get_entry<arma::mat>(dict, "theta"),
                                 get_entry<bool>(dict, "is_theta_estim").value_or(true),
                                 get_entry<arma::colvec>(dict, "beta"),
                                 get_entry<bool>(dict, "is_beta_estim").value_or(true)};
  m_internal = std::make_unique<Kriging>(
      mat_y, mat_X, covType, Trend::fromString(regmodel), normalize, optim, objective, parameters);
}

PyKriging::~PyKriging() {}

PyKriging::PyKriging(const PyKriging& other)
    : m_internal{std::make_unique<Kriging>(*other.m_internal, ExplicitCopySpecifier{})} {}

PyKriging PyKriging::copy() const {
  return PyKriging(*this);
}

void PyKriging::fit(const py::array_t<double>& y,
                    const py::array_t<double>& X,
                    const std::string& regmodel,
                    bool normalize,
                    const std::string& optim,
                    const std::string& objective,
                    const py::dict& dict) {
  arma::mat mat_y = carma::arr_to_col_view<double>(y);
  arma::mat mat_X = carma::arr_to_mat_view<double>(X);
  Kriging::Parameters parameters{get_entry<double>(dict, "sigma2"),
                                 get_entry<bool>(dict, "is_sigma2_estim").value_or(true),
                                 get_entry<arma::mat>(dict, "theta"),
                                 get_entry<bool>(dict, "is_theta_estim").value_or(true),
                                 get_entry<arma::colvec>(dict, "beta"),
                                 get_entry<bool>(dict, "is_beta_estim").value_or(true)};
  m_internal->fit(mat_y, mat_X, Trend::fromString(regmodel), normalize, optim, objective, parameters);
}

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>>
PyKriging::predict(const py::array_t<double>& X_n, bool return_stdev, bool return_cov, bool return_deriv) {
  arma::mat mat_X = carma::arr_to_mat_view<double>(X_n);
  auto [y_predict, y_stderr, y_cov, y_mean_deriv, y_stderr_deriv]
      = m_internal->predict(mat_X, return_stdev, return_cov, return_deriv);
  return std::make_tuple(carma::col_to_arr(y_predict, true),
                         carma::col_to_arr(y_stderr, true),
                         carma::mat_to_arr(y_cov, true),
                         carma::mat_to_arr(y_mean_deriv, true),
                         carma::mat_to_arr(y_stderr_deriv, true));
}

py::array_t<double> PyKriging::simulate(const int nsim,
                                        const int seed,
                                        const py::array_t<double>& X_n,
                                        const bool will_update) {
  arma::mat mat_X = carma::arr_to_mat_view<double>(X_n);
  auto result = m_internal->simulate(nsim, seed, mat_X, will_update);
  return carma::mat_to_arr(result, true);
}

void PyKriging::update(const py::array_t<double>& y_u, const py::array_t<double>& X_u, const bool refit) {
  arma::mat mat_y = carma::arr_to_col<double>(y_u);
  arma::mat mat_X = carma::arr_to_mat<double>(X_u);
  m_internal->update(mat_y, mat_X, refit);
}

void PyKriging::update_simulate(const py::array_t<double>& y_u, const py::array_t<double>& X_u) {
  arma::mat mat_y = carma::arr_to_col<double>(y_u);
  arma::mat mat_X = carma::arr_to_mat<double>(X_u);
  m_internal->update_simulate(mat_y, mat_X);
}

std::string PyKriging::summary() const {
  return m_internal->summary();
}

void PyKriging::save(const std::string filename) const {
  return m_internal->save(filename);
}

PyKriging PyKriging::load(const std::string filename) {
  return PyKriging(std::make_unique<Kriging>(Kriging::load(filename)));
}

std::tuple<double, py::array_t<double>> PyKriging::leaveOneOutFun(const py::array_t<double>& theta,
                                                                  const bool return_grad) {
  arma::vec vec_theta = carma::arr_to_col<double>(theta);
  auto [llo, grad] = m_internal->leaveOneOutFun(vec_theta, return_grad, false);
  return {llo, carma::col_to_arr(grad)};
}

std::tuple<py::array_t<double>, py::array_t<double>> PyKriging::leaveOneOutVec(const py::array_t<double>& theta) {
  arma::vec vec_theta = carma::arr_to_col<double>(theta);
  auto [yhat_mean, yhat_sd] = m_internal->leaveOneOutVec(vec_theta);
  return {carma::col_to_arr(yhat_mean), carma::col_to_arr(yhat_sd)};
}

double PyKriging::leaveOneOut() {
  return m_internal->leaveOneOut();
}

std::tuple<double, py::array_t<double>, py::array_t<double>>
PyKriging::logLikelihoodFun(const py::array_t<double>& theta, const bool return_grad, const bool want_hess) {
  arma::vec vec_theta = carma::arr_to_col<double>(theta);
  auto [llo, grad, hess] = m_internal->logLikelihoodFun(vec_theta, return_grad, want_hess, false);
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
                                                                  const bool return_grad) {
  arma::vec vec_theta = carma::arr_to_col<double>(theta);
  auto [lmp, grad] = m_internal->logMargPostFun(vec_theta, return_grad, false);
  return {lmp, carma::col_to_arr(grad)};
}

double PyKriging::logMargPost() {
  return m_internal->logMargPost();
}

std::string PyKriging::kernel() {
  return m_internal->kernel();
}

std::string PyKriging::optim() {
  return m_internal->optim();
}

std::string PyKriging::objective() {
  return m_internal->objective();
}

py::array_t<double> PyKriging::X() {
  return carma::mat_to_arr(m_internal->X());
}

py::array_t<double> PyKriging::centerX() {
  return carma::row_to_arr(m_internal->centerX());
}

py::array_t<double> PyKriging::scaleX() {
  return carma::row_to_arr(m_internal->scaleX());
}

py::array_t<double> PyKriging::y() {
  return carma::col_to_arr(m_internal->y());
}

double PyKriging::centerY() {
  return m_internal->centerY();
}

double PyKriging::scaleY() {
  return m_internal->scaleY();
}

bool PyKriging::normalize() {
  return m_internal->normalize();
}

std::string PyKriging::regmodel() {
  return Trend::toString(m_internal->regmodel());
}

py::array_t<double> PyKriging::F() {
  return carma::mat_to_arr(m_internal->F());
}

py::array_t<double> PyKriging::T() {
  return carma::mat_to_arr(m_internal->T());
}

py::array_t<double> PyKriging::M() {
  return carma::mat_to_arr(m_internal->M());
}

py::array_t<double> PyKriging::z() {
  return carma::col_to_arr(m_internal->z());
}

py::array_t<double> PyKriging::beta() {
  return carma::col_to_arr(m_internal->beta());
}

bool PyKriging::is_beta_estim() {
  return m_internal->is_beta_estim();
}

py::array_t<double> PyKriging::theta() {
  return carma::col_to_arr(m_internal->theta());
}

bool PyKriging::is_theta_estim() {
  return m_internal->is_theta_estim();
}

double PyKriging::sigma2() {
  return m_internal->sigma2();
}

bool PyKriging::is_sigma2_estim() {
  return m_internal->is_sigma2_estim();
}

py::array_t<double> PyKriging::covMat(const py::array_t<double>& X1, const py::array_t<double>& X2) {
  arma::mat mat_X1 = carma::arr_to_mat<double>(X1);
  arma::mat mat_X2 = carma::arr_to_mat<double>(X2);
  return carma::mat_to_arr(m_internal->covMat(mat_X1, mat_X2), true);
}

py::dict PyKriging::model() const {
  py::dict d;
  d["kernel"] = m_internal->kernel();
  d["optim"] = m_internal->optim();
  d["objective"] = m_internal->objective();
  d["theta"] = carma::col_to_arr(m_internal->theta());
  d["is_theta_estim"] = m_internal->is_theta_estim();
  d["sigma2"] = m_internal->sigma2();
  d["is_sigma2_estim"] = m_internal->is_sigma2_estim();
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