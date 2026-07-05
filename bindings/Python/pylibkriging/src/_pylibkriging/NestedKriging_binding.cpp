#include "NestedKriging_binding.hpp"

#include "libKriging/utils/lk_armadillo.hpp"

#include <carma>

#include <libKriging/NestedKriging.hpp>
#include <libKriging/Trend.hpp>
#include "py_to_cpp_cast.hpp"

#include <stdexcept>

// Same convention as PyKriging::params_from_dict (nugget entries unused in v1)
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

static NestedKriging::Partition parse_partition(const std::string& s) {
  if (s == "kmeans")
    return NestedKriging::Partition::KMeans;
  if (s == "random")
    return NestedKriging::Partition::Random;
  throw std::invalid_argument("Unknown partition '" + s + "'; expected 'kmeans' or 'random'");
}

PyNestedKriging::PyNestedKriging(const std::string& kernel) : m_internal{new NestedKriging{kernel}} {}

PyNestedKriging::PyNestedKriging(const py::array_t<double>& y,
                                 const py::array_t<double>& X,
                                 const std::string& kernel,
                                 unsigned long nb_groups,
                                 const std::string& aggregation,
                                 const std::string& partition,
                                 int seed,
                                 const std::string& regmodel,
                                 const std::string& optim,
                                 const std::string& objective,
                                 const py::dict& dict,
                                 const std::vector<std::string>& warping) {
  arma::colvec mat_y = carma::arr_to_col_view<double>(y);
  arma::mat mat_X = carma::arr_to_mat_view<double>(X);
  m_internal = std::make_unique<NestedKriging>(mat_y,
                                               mat_X,
                                               kernel,
                                               nb_groups,
                                               NestedKriging::aggregationFromString(aggregation),
                                               parse_partition(partition),
                                               seed,
                                               Trend::fromString(regmodel),
                                               optim,
                                               objective,
                                               params_from_dict(dict),
                                               warping);
}

PyNestedKriging::~PyNestedKriging() {}

void PyNestedKriging::fit(const py::array_t<double>& y,
                          const py::array_t<double>& X,
                          unsigned long nb_groups,
                          const std::string& regmodel,
                          const std::string& optim,
                          const std::string& objective,
                          const py::dict& dict,
                          const std::vector<std::string>& warping) {
  arma::colvec mat_y = carma::arr_to_col_view<double>(y);
  arma::mat mat_X = carma::arr_to_mat_view<double>(X);
  m_internal->fit(mat_y, mat_X, nb_groups, Trend::fromString(regmodel), optim, objective, params_from_dict(dict), warping);
}

std::tuple<py::array_t<double>, py::array_t<double>> PyNestedKriging::predict(const py::array_t<double>& X_n,
                                                                              bool return_stdev) {
  arma::mat mat_X = carma::arr_to_mat_view<double>(X_n);
  auto [mean, stdev] = m_internal->predict(mat_X, return_stdev);
  return {carma::col_to_arr(mean, true), carma::col_to_arr(stdev, true)};
}

std::string PyNestedKriging::summary() const {
  return m_internal->summary();
}

std::string PyNestedKriging::kernel() const {
  return m_internal->kernel();
}

std::vector<std::string> PyNestedKriging::warping() const {
  return m_internal->warping();
}

std::string PyNestedKriging::aggregation() const {
  return NestedKriging::aggregationToString(m_internal->aggregation());
}

unsigned long PyNestedKriging::nb_groups() const {
  return m_internal->nb_groups();
}

py::list PyNestedKriging::groups() const {
  py::list out;
  for (const auto& g : m_internal->groups())
    out.append(carma::col_to_arr(arma::conv_to<arma::vec>::from(g), true));
  return out;
}

py::array_t<double> PyNestedKriging::theta() const {
  return carma::col_to_arr(m_internal->theta(), true);
}

double PyNestedKriging::sigma2() const {
  return m_internal->sigma2();
}

double PyNestedKriging::beta0() const {
  return m_internal->beta0();
}

py::array_t<double> PyNestedKriging::X() const {
  return carma::mat_to_arr(m_internal->X(), true);
}

py::array_t<double> PyNestedKriging::y() const {
  return carma::col_to_arr(m_internal->y(), true);
}

void PyNestedKriging::set_predict_chunk(unsigned long chunk) {
  m_internal->set_predict_chunk(chunk);
}
