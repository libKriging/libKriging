#include "MultiOutputKriging_binding.hpp"

#include "libKriging/utils/lk_armadillo.hpp"

#include <carma>

#include <libKriging/MultiOutputKriging.hpp>
#include <libKriging/Trend.hpp>
#include <libKriging/utils/ExplicitCopySpecifier.hpp>
#include "py_to_cpp_cast.hpp"

#include <stdexcept>

static MultiOutputKriging::Parameters params_from_dict(const py::dict& dict) {
  for (const auto& kv : dict) {
    const auto key = kv.first.cast<std::string>();
    if (key != "theta" && key != "is_theta_estim")
      throw std::invalid_argument("MultiOutputKriging: unsupported parameter '" + key
                                  + "' (only 'theta' and 'is_theta_estim')");
  }
  MultiOutputKriging::Parameters p;
  p.theta = get_entry<arma::mat>(dict, "theta");
  p.is_theta_estim = get_entry<bool>(dict, "is_theta_estim").value_or(true);
  return p;
}

// Y (and output coordinates) may be given as a 1-D array: one output (resp. one coordinate column)
static arma::mat to_mat(const py::array_t<double>& a) {
  if (a.ndim() == 1)
    return arma::mat(carma::arr_to_col<double>(a));
  if (a.ndim() != 2)
    throw std::invalid_argument("MultiOutputKriging: expected a 1-D or 2-D array, got " + std::to_string(a.ndim())
                                + "-D");
  return carma::arr_to_mat<double>(a);
}

static py::array_t<double> rowvec_to_arr(const arma::rowvec& v) {
  return py::array_t<double>(static_cast<py::ssize_t>(v.n_elem), v.memptr());
}

static py::array_t<double> vec_to_arr(const arma::vec& v) {
  return py::array_t<double>(static_cast<py::ssize_t>(v.n_elem), v.memptr());
}

PyMultiOutputKriging::PyMultiOutputKriging(const std::string& kernel, const std::string& output_model)
    : m_internal{std::make_unique<MultiOutputKriging>(kernel, output_model)} {}

PyMultiOutputKriging::PyMultiOutputKriging(const py::array_t<double>& Y,
                                           const py::array_t<double>& X,
                                           const std::string& kernel,
                                           const std::string& output_model,
                                           const std::string& regmodel,
                                           bool normalize,
                                           const std::string& optim,
                                           const std::string& objective,
                                           const py::dict& dict,
                                           const py::object& output_coordinates)
    : PyMultiOutputKriging(kernel, output_model) {
  if (!output_coordinates.is_none())
    set_output_coordinates(output_coordinates.cast<py::array_t<double>>());
  fit(Y, X, regmodel, normalize, optim, objective, dict);
}

PyMultiOutputKriging::~PyMultiOutputKriging() {}

void PyMultiOutputKriging::fit(const py::array_t<double>& Y,
                               const py::array_t<double>& X,
                               const std::string& regmodel,
                               bool normalize,
                               const std::string& optim,
                               const std::string& objective,
                               const py::dict& dict) {
  m_internal->fit(
      to_mat(Y), to_mat(X), Trend::fromString(regmodel), normalize, optim, objective, params_from_dict(dict));
}

void PyMultiOutputKriging::set_output_coordinates(const py::array_t<double>& t) {
  m_internal->set_output_coordinates(to_mat(t));
}

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>>
PyMultiOutputKriging::predict(const py::array_t<double>& X_n, bool return_stdev, bool return_cov, bool return_deriv) {
  auto [mean, stdev, cov, deriv] = m_internal->predict(to_mat(X_n), return_stdev, return_cov, return_deriv);
  return std::make_tuple(carma::mat_to_arr(mean, true),
                         carma::mat_to_arr(stdev, true),
                         carma::mat_to_arr(cov, true),
                         carma::cube_to_arr(deriv, true));
}

py::array_t<double> PyMultiOutputKriging::simulate(int nsim,
                                                   int seed,
                                                   const py::array_t<double>& X_n,
                                                   bool will_update) {
  arma::cube sims = m_internal->simulate(nsim, seed, to_mat(X_n), will_update);
  return carma::cube_to_arr(sims, true);  // m × q × nsim
}

py::array_t<double> PyMultiOutputKriging::update_simulate(const py::array_t<double>& Y_u,
                                                          const py::array_t<double>& X_u) {
  arma::cube sims = m_internal->update_simulate(to_mat(Y_u), to_mat(X_u));
  return carma::cube_to_arr(sims, true);
}

void PyMultiOutputKriging::update(const py::array_t<double>& Y_u, const py::array_t<double>& X_u, bool refit) {
  m_internal->update(to_mat(Y_u), to_mat(X_u), refit);
}

std::tuple<py::array_t<double>, py::array_t<double>> PyMultiOutputKriging::leaveOneOutMat() {
  auto [mean, stdev] = m_internal->leaveOneOutMat();
  return std::make_tuple(carma::mat_to_arr(mean, true), carma::mat_to_arr(stdev, true));
}

double PyMultiOutputKriging::leaveOneOut() {
  return m_internal->leaveOneOut();
}

std::string PyMultiOutputKriging::summary() const {
  return m_internal->summary();
}

std::string PyMultiOutputKriging::kernel() const {
  return m_internal->kernel();
}

std::string PyMultiOutputKriging::output_model() const {
  return m_internal->output_model_string();
}

unsigned long PyMultiOutputKriging::nb_outputs() const {
  return m_internal->nb_outputs();
}

py::array_t<double> PyMultiOutputKriging::X() const {
  arma::mat X = m_internal->X();
  return carma::mat_to_arr(X, true);
}

py::array_t<double> PyMultiOutputKriging::Y() const {
  arma::mat Y = m_internal->Y();
  return carma::mat_to_arr(Y, true);
}

py::array_t<double> PyMultiOutputKriging::output_coordinates() const {
  arma::mat t = m_internal->output_coordinates();
  return carma::mat_to_arr(t, true);
}

std::string PyMultiOutputKriging::regmodel() const {
  return Trend::toString(m_internal->regmodel());
}

bool PyMultiOutputKriging::normalize() const {
  return m_internal->normalize();
}

std::string PyMultiOutputKriging::optim() const {
  return m_internal->optim();
}

std::string PyMultiOutputKriging::objective() const {
  return m_internal->objective();
}

py::array_t<double> PyMultiOutputKriging::centerY() const {
  return rowvec_to_arr(m_internal->centerY());
}

py::array_t<double> PyMultiOutputKriging::scaleY() const {
  return rowvec_to_arr(m_internal->scaleY());
}

unsigned long PyMultiOutputKriging::nb_components() const {
  return m_internal->nb_components();
}

py::array_t<double> PyMultiOutputKriging::pca_basis() const {
  arma::mat B = m_internal->pca_basis();
  return carma::mat_to_arr(B, true);
}

py::array_t<double> PyMultiOutputKriging::pca_explained() const {
  return vec_to_arr(m_internal->pca_explained());
}

py::array_t<double> PyMultiOutputKriging::pca_residual() const {
  arma::mat R = m_internal->pca_residual();
  return carma::mat_to_arr(R, true);
}

py::array_t<double> PyMultiOutputKriging::theta() const {
  return vec_to_arr(m_internal->theta());
}

py::array_t<double> PyMultiOutputKriging::sigma2() const {
  return vec_to_arr(m_internal->sigma2());
}

py::array_t<double> PyMultiOutputKriging::beta() const {
  arma::mat B = m_internal->beta();
  return carma::mat_to_arr(B, true);
}

double PyMultiOutputKriging::logLikelihood() {
  return m_internal->logLikelihood();
}

std::tuple<double, py::array_t<double>> PyMultiOutputKriging::logLikelihoodFun(const py::array_t<double>& theta,
                                                                               bool return_grad) {
  const arma::vec th = arma::vectorise(to_mat(theta));
  auto [ll, g] = m_internal->logLikelihoodFun(th, return_grad);
  return std::make_tuple(ll, vec_to_arr(g));
}

PyKriging PyMultiOutputKriging::component(unsigned long k) const {
  return PyKriging(std::make_unique<Kriging>(m_internal->component(k), ExplicitCopySpecifier{}));
}

std::tuple<double, py::array_t<double>> PyMultiOutputKriging::leaveOneOutFun(const py::array_t<double>& theta,
                                                                             bool return_grad) {
  const arma::vec th = arma::vectorise(to_mat(theta));
  auto [loo, g] = m_internal->leaveOneOutFun(th, return_grad);
  return std::make_tuple(loo, vec_to_arr(g));
}

py::array_t<double> PyMultiOutputKriging::output_cov() const {
  arma::mat S = m_internal->output_cov();
  return carma::mat_to_arr(S, true);
}

std::tuple<py::array_t<double>, py::array_t<double>> PyMultiOutputKriging::predictCovFactors(
    const py::array_t<double>& X_n) {
  auto [Cx, S] = m_internal->predictCovFactors(to_mat(X_n));
  return std::make_tuple(carma::mat_to_arr(Cx, true), carma::mat_to_arr(S, true));
}
