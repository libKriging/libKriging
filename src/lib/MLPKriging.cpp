/**
 * @file MLPKriging.cpp
 * @brief Thin facade over WarpKriging({"mlp_joint(…)"}, kernel).
 */

#include "libKriging/MLPKriging.hpp"
#include "libKriging/utils/jsonutils.hpp"
#include "libKriging/utils/nlohmann/json.hpp"
#include "libKriging/utils/utils.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace libKriging {

// -------------------------------------------------------------------------
//  Helper: build the "mlp_joint(h1:h2,d_out,act)" spec string
// -------------------------------------------------------------------------
std::string MLPKriging::make_warp_spec(const std::vector<arma::uword>& hidden_dims,
                                       arma::uword d_out,
                                       const std::string& activation) {
  std::string s = "mlp_joint(";
  for (arma::uword i = 0; i < hidden_dims.size(); ++i) {
    if (i > 0)
      s += ":";
    s += std::to_string(hidden_dims[i]);
  }
  s += "," + std::to_string(d_out) + "," + activation + ")";
  return s;
}

// -------------------------------------------------------------------------
//  Constructors
// -------------------------------------------------------------------------
MLPKriging::MLPKriging(const std::vector<arma::uword>& hidden_dims,
                       arma::uword d_out,
                       const std::string& activation,
                       const std::string& kernel)
    : m_impl({make_warp_spec(hidden_dims, d_out, activation)}, kernel),
      m_hidden_dims(hidden_dims),
      m_d_out(d_out),
      m_activation(activation) {}

MLPKriging::MLPKriging(const arma::vec& y,
                       const arma::mat& X,
                       const std::vector<arma::uword>& hidden_dims,
                       arma::uword d_out,
                       const std::string& activation,
                       const std::string& kernel,
                       const Trend::RegressionModel& regmodel,
                       bool normalize,
                       const std::string& optim,
                       const std::string& objective,
                       const std::map<std::string, std::string>& parameters)
    : MLPKriging(hidden_dims, d_out, activation, kernel) {
  fit(y, X, regmodel, normalize, optim, objective, parameters);
}

// -------------------------------------------------------------------------
//  fit
// -------------------------------------------------------------------------
void MLPKriging::fit(const arma::vec& y,
                     const arma::mat& X,
                     const Trend::RegressionModel& regmodel,
                     bool normalize,
                     const std::string& optim,
                     const std::string& objective,
                     const std::map<std::string, std::string>& parameters) {
  m_impl.fit(y, X, regmodel, normalize, optim, objective, parameters);
}

void MLPKriging::fit(const arma::vec& y,
                     const arma::mat& X,
                     const Trend::RegressionModel& regmodel,
                     bool normalize,
                     const std::string& optim,
                     const std::string& objective,
                     const Parameters& parameters) {
  WarpKrigingParameters wp;
  wp.theta = parameters.theta;
  wp.warp_params = parameters.warp_params;
  m_impl.fit(y, X, regmodel, normalize, optim, objective, wp);
}

// -------------------------------------------------------------------------
//  predict / simulate / update
// -------------------------------------------------------------------------
std::tuple<arma::vec, arma::vec, arma::mat, arma::mat, arma::mat> MLPKriging::predict(const arma::mat& X_n,
                                                                                       bool return_stdev,
                                                                                       bool return_cov,
                                                                                       bool return_deriv) const {
  return m_impl.predict(X_n, return_stdev, return_cov, return_deriv);
}

arma::mat MLPKriging::simulate(int nsim, int seed, const arma::mat& X_n, bool will_update) {
  return m_impl.simulate(nsim, seed, X_n, will_update, /*with_noise=*/{});
}

arma::mat MLPKriging::update_simulate(const arma::vec& y_u, const arma::mat& X_u) {
  return m_impl.update_simulate(y_u, X_u, /*noise_u=*/{});
}

void MLPKriging::update(const arma::vec& y_u, const arma::mat& X_u, const bool refit) {
  m_impl.update(y_u, X_u, refit, /*noise_u=*/{});
}

// -------------------------------------------------------------------------
//  Log-likelihood
// -------------------------------------------------------------------------
double MLPKriging::logLikelihood() const {
  return m_impl.logLikelihood();
}

std::tuple<double, arma::vec, arma::mat> MLPKriging::logLikelihoodFun(const arma::vec& theta,
                                                                       bool return_grad,
                                                                       bool return_hess) const {
  return m_impl.logLikelihoodFun(theta, return_grad, return_hess);
}

// -------------------------------------------------------------------------
//  summary
// -------------------------------------------------------------------------
std::string MLPKriging::summary() const {
  std::string s = m_impl.summary();
  // Replace the "* WarpKriging" header with "* MLPKriging"
  const std::string from = "* WarpKriging";
  const std::string to   = "* MLPKriging";
  auto pos = s.find(from);
  if (pos != std::string::npos)
    s.replace(pos, from.size(), to);
  return s;
}

// -------------------------------------------------------------------------
//  save / load  — maintain exact MLPKriging JSON schema (backward compat)
// -------------------------------------------------------------------------
void MLPKriging::save(const std::string filename) const {
  nlohmann::json j;
  j["version"] = 3;
  j["content"] = "MLPKriging";
  j["hidden_dims"] = std::vector<arma::uword>(m_hidden_dims.begin(), m_hidden_dims.end());
  j["d_out"] = m_d_out;
  j["activation"] = m_activation;
  m_impl.dump_to_json(j);
  // MLPKriging schema omits these WarpKriging-only fields
  j.erase("warping");
  j.erase("is_continuous");
  j.erase("feature_dim");
  std::ofstream f(filename);
  f << std::setw(4) << j;
}

MLPKriging MLPKriging::load(const std::string filename) {
  std::ifstream f(filename);
  nlohmann::json j = nlohmann::json::parse(f);

  uint32_t version = j["version"].template get<uint32_t>();
  if (version < 2 || version > 3)
    throw std::runtime_error(asString("Bad version to load from '", filename, "'; found ", version, ", requires 2 or 3"));
  std::string content = j["content"].template get<std::string>();
  if (content != "MLPKriging")
    throw std::runtime_error(
        asString("Bad content to load from '", filename, "'; found '", content, "', requires 'MLPKriging'"));

  auto hd = j["hidden_dims"].template get<std::vector<arma::uword>>();
  arma::uword d_out_val = j["d_out"].template get<arma::uword>();
  std::string activation = j["activation"].template get<std::string>();
  std::string kernel = j.contains("covType") ? j["covType"].template get<std::string>()
                                             : j["kernel"].template get<std::string>();
  MLPKriging mk(hd, d_out_val, activation, kernel);
  mk.m_impl.load_from_json(j);
  return mk;
}

}  // namespace libKriging
