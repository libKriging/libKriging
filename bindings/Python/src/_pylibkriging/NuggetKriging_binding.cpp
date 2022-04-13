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

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
PyNuggetKriging::predict(const py::array_t<double>& X, bool withStd, bool withCov) {
  arma::mat mat_X = carma::arr_to_mat_view<double>(X);
  auto [y_predict, y_stderr, y_cov] = m_internal->predict(mat_X, withStd, withCov);
  return std::make_tuple(
      carma::col_to_arr(y_predict, true), carma::col_to_arr(y_stderr, true), carma::mat_to_arr(y_cov, true));
}

py::array_t<double> PyNuggetKriging::simulate(const int nsim, const int seed, const py::array_t<double>& Xp) {
  arma::mat mat_X = carma::arr_to_mat_view<double>(Xp);
  auto result = m_internal->simulate(nsim, seed, mat_X);
  return carma::mat_to_arr(result, true);
}

void PyNuggetKriging::update(const py::array_t<double>& newy, const py::array_t<double>& newX, bool normalize) {
  arma::mat mat_y = carma::arr_to_col<double>(newy);
  arma::mat mat_X = carma::arr_to_mat<double>(newX);
  m_internal->update(mat_y, mat_X, normalize);
}

std::string PyNuggetKriging::summary() const {
  return m_internal->summary();
}

std::tuple<double, py::array_t<double>, py::array_t<double>> PyNuggetKriging::logLikelihoodEval(
    const py::array_t<double>& theta,
    const bool want_grad) {
  arma::vec vec_theta = carma::arr_to_col<double>(theta);
  auto [llo, grad] = m_internal->logLikelihoodEval(vec_theta, want_grad);
  return {
      llo,
      carma::col_to_arr(grad),
      // carma::mat_to_arr(hess)  // FIXME error in hessian transmission
      {}  //
  };
}

std::tuple<double, py::array_t<double>> PyNuggetKriging::logMargPostEval(const py::array_t<double>& theta,
                                                                         const bool want_grad) {
  arma::vec vec_theta = carma::arr_to_col<double>(theta);
  auto [lmp, grad] = m_internal->logMargPostEval(vec_theta, want_grad);
  return {lmp, carma::col_to_arr(grad)};
}
