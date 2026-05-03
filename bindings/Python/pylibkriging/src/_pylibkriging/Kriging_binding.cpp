#include "Kriging_binding.hpp"

#include "libKriging/utils/lk_armadillo.hpp"

#include <carma>

#include <libKriging/Kriging.hpp>
#include <libKriging/Trend.hpp>
#include "py_to_cpp_cast.hpp"

#include <random>
#include <stdexcept>

// --- Helpers ---

Kriging::NoiseModel PyKriging::parse_noise_model(const std::string& nm) {
  if (nm == "none" || nm.empty())
    return Kriging::NoiseModel::None;
  if (nm == "nugget")
    return Kriging::NoiseModel::Nugget;
  if (nm == "heterogeneous")
    return Kriging::NoiseModel::Heterogeneous;
  throw std::invalid_argument("Unknown noise_model '" + nm + "'; expected 'none', 'nugget', or 'heterogeneous'");
}

std::string PyKriging::noise_model_to_string(Kriging::NoiseModel nm) {
  switch (nm) {
    case Kriging::NoiseModel::None:
      return "none";
    case Kriging::NoiseModel::Nugget:
      return "nugget";
    case Kriging::NoiseModel::Heterogeneous:
      return "heterogeneous";
  }
  return "none";
}

// Parse parameters dict, including optional nugget/is_nugget_estim
static Kriging::Parameters params_from_dict(const py::dict& dict) {
  return Kriging::Parameters{get_entry<double>(dict, "sigma2"),
                             get_entry<bool>(dict, "is_sigma2_estim").value_or(true),
                             get_entry<arma::mat>(dict, "theta"),
                             get_entry<bool>(dict, "is_theta_estim").value_or(true),
                             get_entry<arma::colvec>(dict, "beta"),
                             get_entry<bool>(dict, "is_beta_estim").value_or(true),
                             get_entry<double>(dict, "nugget"),
                             get_entry<bool>(dict, "is_nugget_estim").value_or(true)};
}

// --- Constructors ---

PyKriging::PyKriging(const std::string& kernel) : m_internal{new Kriging{kernel}} {}

PyKriging::PyKriging(const std::string& kernel, const std::string& noise_model)
    : m_internal{new Kriging{kernel, parse_noise_model(noise_model)}} {}

PyKriging::PyKriging(const py::array_t<double>& y,
                     const py::array_t<double>& X,
                     const std::string& covType,
                     const std::string& regmodel,
                     bool normalize,
                     const std::string& optim,
                     const std::string& objective,
                     const py::dict& dict,
                     const py::object& noise) {
  arma::colvec mat_y = carma::arr_to_col_view<double>(y);
  arma::mat mat_X = carma::arr_to_mat_view<double>(X);
  Kriging::Parameters parameters = params_from_dict(dict);

  if (noise.is_none()) {
    // No noise: pure GP (NoiseModel::None)
    m_internal = std::make_unique<Kriging>(
        mat_y, mat_X, covType, Trend::fromString(regmodel), normalize, optim, objective, parameters);
  } else if (py::isinstance<py::str>(noise)) {
    // noise="nugget": estimated nugget mode
    std::string noise_str = noise.cast<std::string>();
    if (noise_str != "nugget")
      throw std::invalid_argument("noise string must be 'nugget', got '" + noise_str + "'");
    m_internal = std::make_unique<Kriging>(covType, Kriging::NoiseModel::Nugget);
    m_internal->fit(mat_y, mat_X, Trend::fromString(regmodel), normalize, optim, objective, parameters);
  } else {
    // noise is array or scalar: heterogeneous noise
    arma::colvec mat_noise;
    if (py::isinstance<py::float_>(noise) || py::isinstance<py::int_>(noise)) {
      double noise_val = noise.cast<double>();
      mat_noise = arma::colvec(mat_y.n_elem, arma::fill::value(noise_val));
    } else {
      mat_noise = carma::arr_to_col_view<double>(noise.cast<py::array_t<double>>());
    }
    m_internal = std::make_unique<Kriging>(covType, Kriging::NoiseModel::Heterogeneous);
    m_internal->fit(mat_y, mat_noise, mat_X, Trend::fromString(regmodel), normalize, optim, objective, parameters);
  }
}

PyKriging::~PyKriging() {}

PyKriging::PyKriging(const PyKriging& other)
    : m_internal{std::make_unique<Kriging>(*other.m_internal, ExplicitCopySpecifier{})} {}

PyKriging PyKriging::copy() const {
  return PyKriging(*this);
}

// --- fit ---

void PyKriging::fit(const py::array_t<double>& y,
                    const py::array_t<double>& X,
                    const std::string& regmodel,
                    bool normalize,
                    const std::string& optim,
                    const std::string& objective,
                    const py::dict& dict,
                    const py::object& noise) {
  arma::mat mat_y = carma::arr_to_col_view<double>(y);
  arma::mat mat_X = carma::arr_to_mat_view<double>(X);
  Kriging::Parameters parameters = params_from_dict(dict);

  if (noise.is_none()) {
    m_internal->fit(mat_y, mat_X, Trend::fromString(regmodel), normalize, optim, objective, parameters);
  } else if (py::isinstance<py::str>(noise)) {
    // noise="nugget": just fit without noise vector (Nugget mode uses the non-noise fit overload)
    m_internal->fit(mat_y, mat_X, Trend::fromString(regmodel), normalize, optim, objective, parameters);
  } else {
    arma::colvec mat_noise;
    if (py::isinstance<py::float_>(noise) || py::isinstance<py::int_>(noise)) {
      double noise_val = noise.cast<double>();
      mat_noise = arma::colvec(mat_y.n_elem, arma::fill::value(noise_val));
    } else {
      mat_noise = carma::arr_to_col_view<double>(noise.cast<py::array_t<double>>());
    }
    m_internal->fit(mat_y, mat_noise, mat_X, Trend::fromString(regmodel), normalize, optim, objective, parameters);
  }
}

// --- predict ---

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

// --- simulate ---

py::array_t<double> PyKriging::simulate(const int nsim,
                                        const int seed,
                                        const py::array_t<double>& X_n,
                                        const bool will_update,
                                        const py::object& with_noise) {
  arma::mat mat_X = carma::arr_to_mat_view<double>(X_n);

  if (with_noise.is_none()) {
    // Plain simulate (NoiseModel::None)
    auto result = m_internal->simulate(nsim, seed, mat_X, will_update);
    return carma::mat_to_arr(result, true);
  } else if (py::isinstance<py::bool_>(with_noise)) {
    // Nugget mode: simulate(nsim, seed, X_n, with_nugget, will_update)
    bool flag = with_noise.cast<bool>();
    auto result = m_internal->simulate(nsim, seed, mat_X, flag, will_update);
    return carma::mat_to_arr(result, true);
  } else {
    // Heterogeneous mode: simulate(nsim, seed, X_n, noise_vec, will_update)
    arma::colvec noise_vec;
    if (py::isinstance<py::float_>(with_noise) || py::isinstance<py::int_>(with_noise)) {
      double noise_val = with_noise.cast<double>();
      noise_vec = arma::colvec(mat_X.n_rows, arma::fill::value(noise_val));
    } else {
      noise_vec = carma::arr_to_col_view<double>(with_noise.cast<py::array_t<double>>());
    }
    auto result = m_internal->simulate(nsim, seed, mat_X, noise_vec, will_update);
    return carma::mat_to_arr(result, true);
  }
}

// --- update ---

void PyKriging::update(const py::array_t<double>& y_u,
                       const py::array_t<double>& X_u,
                       const bool refit,
                       const py::object& noise_u) {
  arma::mat mat_y = carma::arr_to_col<double>(y_u);
  arma::mat mat_X = carma::arr_to_mat<double>(X_u);

  if (noise_u.is_none()) {
    m_internal->update(mat_y, mat_X, refit);
  } else {
    arma::colvec mat_noise;
    if (py::isinstance<py::float_>(noise_u) || py::isinstance<py::int_>(noise_u)) {
      double noise_val = noise_u.cast<double>();
      mat_noise = arma::colvec(mat_y.n_elem, arma::fill::value(noise_val));
    } else {
      mat_noise = carma::arr_to_col<double>(noise_u.cast<py::array_t<double>>());
    }
    m_internal->update(mat_y, mat_noise, mat_X, refit);
  }
}

// --- update_simulate ---

py::array_t<double> PyKriging::update_simulate(const py::array_t<double>& y_u,
                                               const py::array_t<double>& X_u,
                                               const py::object& noise_u) {
  arma::vec mat_y = carma::arr_to_col<double>(y_u);
  arma::mat mat_X = carma::arr_to_mat<double>(X_u);

  if (noise_u.is_none()) {
    arma::mat result = m_internal->update_simulate(mat_y, mat_X);
    return carma::mat_to_arr(result, true);
  } else {
    arma::colvec mat_noise;
    if (py::isinstance<py::float_>(noise_u) || py::isinstance<py::int_>(noise_u)) {
      double noise_val = noise_u.cast<double>();
      mat_noise = arma::colvec(mat_y.n_elem, arma::fill::value(noise_val));
    } else {
      mat_noise = carma::arr_to_col<double>(noise_u.cast<py::array_t<double>>());
    }
    arma::mat result = m_internal->update_simulate(mat_y, mat_noise, mat_X);
    return carma::mat_to_arr(result, true);
  }
}

// --- misc ---

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
PyKriging::logLikelihoodFun(const py::array_t<double>& theta, const bool return_grad, const bool /*want_hess*/) {
  arma::vec vec_theta = carma::arr_to_col<double>(theta);
  auto [llo, grad] = m_internal->logLikelihoodFun(vec_theta, return_grad, false);
  return {llo, carma::col_to_arr(grad), {}};
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

// --- accessors ---

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

// --- noise-related accessors ---

std::string PyKriging::noise_model() {
  return noise_model_to_string(m_internal->noise_model());
}

double PyKriging::nugget() {
  return m_internal->nugget();
}

bool PyKriging::is_nugget_estim() {
  return m_internal->is_nugget_estim();
}

py::array_t<double> PyKriging::noise() {
  return carma::col_to_arr(m_internal->noise());
}

// --- covMat ---

py::array_t<double> PyKriging::covMat(const py::array_t<double>& X1, const py::array_t<double>& X2) {
  arma::mat mat_X1 = carma::arr_to_mat<double>(X1);
  arma::mat mat_X2 = carma::arr_to_mat<double>(X2);
  arma::mat result = m_internal->covMat(mat_X1, mat_X2);
  return carma::mat_to_arr(result, true);
}

// --- model ---

py::dict PyKriging::model() const {
  py::dict d;
  d["kernel"] = m_internal->kernel();
  d["optim"] = m_internal->optim();
  d["objective"] = m_internal->objective();
  d["noise_model"] = noise_model_to_string(m_internal->noise_model());

  arma::vec theta = m_internal->theta();
  d["theta"] = carma::col_to_arr(theta);
  d["is_theta_estim"] = m_internal->is_theta_estim();
  d["sigma2"] = m_internal->sigma2();
  d["is_sigma2_estim"] = m_internal->is_sigma2_estim();

  if (m_internal->noise_model() == Kriging::NoiseModel::Nugget) {
    d["nugget"] = m_internal->nugget();
    d["is_nugget_estim"] = m_internal->is_nugget_estim();
  }
  if (m_internal->noise_model() == Kriging::NoiseModel::Heterogeneous) {
    arma::vec noise = m_internal->noise();
    d["noise"] = carma::col_to_arr(noise);
  }

  arma::mat X = m_internal->X();
  d["X"] = carma::mat_to_arr(X);
  arma::rowvec centerX = m_internal->centerX();
  d["centerX"] = carma::row_to_arr(centerX);
  arma::rowvec scaleX = m_internal->scaleX();
  d["scaleX"] = carma::row_to_arr(scaleX);
  arma::vec y = m_internal->y();
  d["y"] = carma::col_to_arr(y);
  d["centerY"] = m_internal->centerY();
  d["scaleY"] = m_internal->scaleY();
  d["normalize"] = m_internal->normalize();
  d["regmodel"] = Trend::toString(m_internal->regmodel());

  arma::vec beta = m_internal->beta();
  d["beta"] = carma::col_to_arr(beta);
  d["is_beta_estim"] = m_internal->is_beta_estim();
  arma::mat F = m_internal->F();
  d["F"] = carma::mat_to_arr(F);
  arma::mat T = m_internal->T();
  d["T"] = carma::mat_to_arr(T);
  arma::mat M = m_internal->M();
  d["M"] = carma::mat_to_arr(M);
  arma::vec z = m_internal->z();
  d["z"] = carma::col_to_arr(z);
  return d;
}