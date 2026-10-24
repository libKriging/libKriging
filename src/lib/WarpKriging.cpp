/**
 * @file WarpKriging.cpp
 * @brief Per-variable warping Kriging implementation for libKriging.
 * See WarpKriging.hpp for documentation.
 */

#include "libKriging/WarpKriging.hpp"
#include "libKriging/AdamBFGS.hpp"
#include "libKriging/Covariance.hpp"
#include "libKriging/Kriging.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/Optim.hpp"
#include "libKriging/Random.hpp"

#include "libKriging/utils/jsonutils.hpp"
#include "libKriging/utils/nlohmann/json.hpp"
#include "libKriging/utils/utils.hpp"

#include "BlasThreads.hpp"
#include "TaskPool.hpp"
#include "ThreadBudget.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace libKriging {

namespace {
// Inner θ L-BFGS budget of the bi-level "BFGS+Adam" optimiser (max_iter_bfgs = 0).
constexpr arma::uword kDefaultInnerMaxIterBFGS = 100;
// Automatic budget of the joint "BFGS" optimiser when per-variable warp params
// are present: kJointIterPerParam iterations per optimised parameter, capped.
// The global Optim::max_iteration (20 by default) is sized for a handful of θ
// and stops a joint warp+θ problem with tens of parameters far from its
// optimum (#382). A joint MLP (mlp_joint / MLPKriging) instead keeps
// Optim::max_iteration per L-BFGS-B call, resumed while it still improves
// (see run_joint_bfgs): with more weights than observations, a long run mostly
// overfits the training points, for a mean no better and an overconfident
// predictive variance, at 10-20x the cost (#407).
constexpr arma::uword kJointIterPerParam = 20;
constexpr arma::uword kJointMaxIterCap = 1000;
// Early stop of the Adam phase: steps without a relative LL gain > tol.
constexpr arma::uword kAdamPatience = 20;
constexpr double kAdamRelTol = 1e-6;
// At the first plateau, back to the best point with half the step, twice,
// before stopping: the fixed step oscillated around a ridge (mlp warps: +167
// LL median over 15 data sets, x2 fit time; docs/dev/CholJitter.md,
// Adam phase).
constexpr arma::uword kAdamLrDecays = 2;
// Overflow-safe softplus log(1 + e^z): the naive form is +inf for z > 709.
inline double softplus(double z) {
  return std::max(z, 0.0) + std::log1p(std::exp(-std::abs(z)));
}
// Collapse guard (#406): between two consecutive training inputs, the warp's
// mean slope may fall down to kCollapseMinRatio times its overall mean slope
// over the training range for free; below, the fit objective pays
// kCollapseWeight · log²(ratio / kCollapseMinRatio) per such gap.
constexpr double kCollapseMinRatio = 0.1;
constexpr double kCollapseWeight = 100.0;

using detail::run_task_pool;
}  // namespace

// *************************************************************************
//  WarpSpec  factories
// *************************************************************************

WarpSpec WarpSpec::none() {
  WarpSpec s;
  s.type = WarpType::None;
  return s;
}
WarpSpec WarpSpec::affine() {
  WarpSpec s;
  s.type = WarpType::Affine;
  return s;
}
WarpSpec WarpSpec::boxcox() {
  WarpSpec s;
  s.type = WarpType::BoxCox;
  return s;
}
WarpSpec WarpSpec::kumaraswamy() {
  WarpSpec s;
  s.type = WarpType::Kumaraswamy;
  return s;
}

WarpSpec WarpSpec::neural_mono(arma::uword nh) {
  WarpSpec s;
  s.type = WarpType::NeuralMono;
  s.n_hidden = nh;
  return s;
}

WarpSpec WarpSpec::categorical(arma::uword nlevels, arma::uword edim) {
  WarpSpec s;
  s.type = WarpType::Embedding;
  s.n_levels = nlevels;
  s.embed_dim = edim;
  return s;
}

WarpSpec WarpSpec::ordinal(arma::uword nlevels) {
  WarpSpec s;
  s.type = WarpType::Ordinal;
  s.n_levels = nlevels;
  return s;
}

WarpSpec WarpSpec::mlp(const std::vector<arma::uword>& hdims, arma::uword dout, const std::string& act) {
  WarpSpec s;
  s.type = WarpType::MLP;
  s.hidden_dims = hdims;
  s.d_out = dout;
  s.activation = act;
  return s;
}

WarpSpec WarpSpec::knots(arma::uword n_knots, const std::vector<double>& knot_positions) {
  WarpSpec s;
  s.type = WarpType::Knots;
  s.n_knots = n_knots;
  s.knot_positions = knot_positions;
  return s;
}

WarpSpec WarpSpec::mlp_joint(const std::vector<arma::uword>& hdims, arma::uword dout, const std::string& act) {
  WarpSpec s;
  s.type = WarpType::MLPJoint;
  s.hidden_dims = hdims;
  s.d_out = dout;
  s.activation = act;
  return s;
}

// -------------------------------------------------------------------------
//  WarpSpec::from_string  —  parse "type(arg1,arg2,...)" format
//
//  Accepted formats:
//    "none"  "affine"  "boxcox"  "kumaraswamy"
//    "neural_mono"  or  "neural_mono(16)"
//    "mlp"  or  "mlp(16:8,3,selu)"
//    "categorical(5)"  or  "categorical(5,2)"
//    "ordinal(4)"
// -------------------------------------------------------------------------
WarpSpec WarpSpec::from_string(const std::string& str) {
  // Trim whitespace
  auto trim = [](const std::string& s) -> std::string {
    auto b = s.find_first_not_of(" \t");
    if (b == std::string::npos)
      return "";
    auto e = s.find_last_not_of(" \t");
    return s.substr(b, e - b + 1);
  };

  std::string input = trim(str);

  // Split "type(args)" into type and args
  std::string type_str, args_str;
  auto paren = input.find('(');
  if (paren != std::string::npos) {
    type_str = trim(input.substr(0, paren));
    auto close = input.rfind(')');
    if (close == std::string::npos || close <= paren)
      throw std::invalid_argument("WarpSpec::from_string: unmatched '(' in: " + str);
    args_str = trim(input.substr(paren + 1, close - paren - 1));
  } else {
    type_str = input;
  }

  // Lowercase type
  std::transform(type_str.begin(), type_str.end(), type_str.begin(), ::tolower);

  // Helper: split by comma
  auto split = [&](const std::string& s, char delim) -> std::vector<std::string> {
    std::vector<std::string> tokens;
    std::istringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, delim))
      tokens.push_back(trim(tok));
    return tokens;
  };

  // ---- Dispatch on type ------------------------------------------------

  if (type_str == "none")
    return WarpSpec::none();

  if (type_str == "affine")
    return WarpSpec::affine();

  if (type_str == "boxcox")
    return WarpSpec::boxcox();

  if (type_str == "kumaraswamy")
    return WarpSpec::kumaraswamy();

  if (type_str == "neural_mono") {
    arma::uword nh = 8;
    if (!args_str.empty())
      nh = static_cast<arma::uword>(std::stoul(args_str));
    return WarpSpec::neural_mono(nh);
  }

  if (type_str == "mlp") {
    // Default: "mlp(16:8,2,selu)"
    std::vector<arma::uword> hdims = {16, 8};
    arma::uword dout = 2;
    std::string act = "selu";

    if (!args_str.empty()) {
      auto parts = split(args_str, ',');
      if (parts.size() >= 1) {
        // Parse hidden dims with ':' separator
        hdims.clear();
        auto layers = split(parts[0], ':');
        for (const auto& l : layers)
          hdims.push_back(static_cast<arma::uword>(std::stoul(l)));
      }
      if (parts.size() >= 2)
        dout = static_cast<arma::uword>(std::stoul(parts[1]));
      if (parts.size() >= 3)
        act = parts[2];
    }
    return WarpSpec::mlp(hdims, dout, act);
  }

  // Helper: parse a bracket-delimited list of quoted strings, e.g.
  //   '["red","green","blue"]'  →  {"red", "green", "blue"}
  // Returns the names and the position just past the closing ']'.
  auto parse_name_list
      = [&](const std::string& s, std::size_t start) -> std::pair<std::vector<std::string>, std::size_t> {
    if (start >= s.size() || s[start] != '[')
      throw std::invalid_argument("WarpSpec::from_string: expected '[' at position " + std::to_string(start));
    auto close = s.find(']', start);
    if (close == std::string::npos)
      throw std::invalid_argument("WarpSpec::from_string: unmatched '[' in: " + str);
    std::string inner = s.substr(start + 1, close - start - 1);
    std::vector<std::string> names;
    std::istringstream iss(inner);
    std::string tok;
    while (std::getline(iss, tok, ',')) {
      tok = trim(tok);
      // Strip quotes (single or double)
      if (tok.size() >= 2 && ((tok.front() == '"' && tok.back() == '"') || (tok.front() == '\'' && tok.back() == '\'')))
        tok = tok.substr(1, tok.size() - 2);
      if (!tok.empty())
        names.push_back(tok);
    }
    return {names, close + 1};
  };

  if (type_str == "categorical") {
    if (args_str.empty())
      throw std::invalid_argument(
          "WarpSpec::from_string: categorical requires n_levels, e.g. 'categorical(5)' or 'categorical(5,2)'"
          " or 'categorical([\"a\",\"b\",\"c\"],2)'");
    // Check for bracket-delimited level names
    if (args_str.front() == '[') {
      auto [names, pos] = parse_name_list(args_str, 0);
      if (names.empty())
        throw std::invalid_argument("WarpSpec::from_string: categorical level name list is empty");
      arma::uword nl = static_cast<arma::uword>(names.size());
      arma::uword ed = 2;  // default embed_dim
      // Parse optional embed_dim after the '],'
      if (pos < args_str.size()) {
        std::string rest = trim(args_str.substr(pos));
        if (!rest.empty() && rest.front() == ',')
          rest = trim(rest.substr(1));
        if (!rest.empty())
          ed = static_cast<arma::uword>(std::stoul(rest));
      }
      auto spec = WarpSpec::categorical(nl, ed);
      spec.level_names = std::move(names);
      return spec;
    }
    auto parts = split(args_str, ',');
    arma::uword nl = static_cast<arma::uword>(std::stoul(parts[0]));
    arma::uword ed = (parts.size() >= 2) ? static_cast<arma::uword>(std::stoul(parts[1])) : 2;
    return WarpSpec::categorical(nl, ed);
  }

  if (type_str == "ordinal") {
    if (args_str.empty())
      throw std::invalid_argument(
          "WarpSpec::from_string: ordinal requires n_levels, e.g. 'ordinal(4)'"
          " or 'ordinal([\"low\",\"med\",\"high\"])'");
    // Check for bracket-delimited level names
    if (args_str.front() == '[') {
      auto [names, pos] = parse_name_list(args_str, 0);
      if (names.empty())
        throw std::invalid_argument("WarpSpec::from_string: ordinal level name list is empty");
      arma::uword nl = static_cast<arma::uword>(names.size());
      auto spec = WarpSpec::ordinal(nl);
      spec.level_names = std::move(names);
      return spec;
    }
    arma::uword nl = static_cast<arma::uword>(std::stoul(args_str));
    return WarpSpec::ordinal(nl);
  }

  // --- mlp_joint(h1:h2[:...][,d_out][,activation]) ---
  // Same parsing as mlp but produces MLPJoint type
  if (type_str == "mlp_joint") {
    if (args_str.empty())
      return WarpSpec::mlp_joint({32, 16}, 2, "selu");

    auto parts = split(args_str, ',');
    std::vector<arma::uword> hdims;
    if (!parts.empty()) {
      auto hparts = split(parts[0], ':');
      for (const auto& hp : hparts)
        hdims.push_back(static_cast<arma::uword>(std::stoul(hp)));
    }
    if (hdims.empty())
      hdims = {32, 16};

    arma::uword dout = 2;
    if (parts.size() >= 2)
      dout = static_cast<arma::uword>(std::stoul(parts[1]));

    std::string act = "selu";
    if (parts.size() >= 3) {
      act = parts[2];
      std::transform(act.begin(), act.end(), act.begin(), ::tolower);
    }
    return WarpSpec::mlp_joint(hdims, dout, act);
  }

  // --- knots([K] | [t1:t2:...:tK]) ---
  // "knots"           → 3 uniform interior knots
  // "knots(K)"        → K uniform interior knots
  // "knots(t1:t2:…)"  → explicit knot positions in (0,1) (colon-separated floats)
  if (type_str == "knots") {
    if (args_str.empty())
      return WarpSpec::knots(3);

    // Try to determine whether args_str is a single integer or floats
    auto kparts = split(args_str, ':');
    if (kparts.size() == 1) {
      // Could be an integer K or a single float knot position
      bool is_int = true;
      for (char c : kparts[0]) {
        if (!std::isdigit(c)) {
          is_int = false;
          break;
        }
      }
      if (is_int) {
        arma::uword K = static_cast<arma::uword>(std::stoul(kparts[0]));
        return WarpSpec::knots(K);
      }
    }
    // Parse as explicit float knot positions
    std::vector<double> positions;
    for (const auto& part : kparts)
      positions.push_back(std::stod(part));
    return WarpSpec::knots(static_cast<arma::uword>(positions.size()), positions);
  }

  throw std::invalid_argument("WarpSpec::from_string: unknown warp type: '" + type_str + "'");
}

// -------------------------------------------------------------------------
//  WarpSpec::to_string
// -------------------------------------------------------------------------
std::string WarpSpec::to_string() const {
  switch (type) {
    case WarpType::None:
      return "none";
    case WarpType::Affine:
      return "affine";
    case WarpType::BoxCox:
      return "boxcox";
    case WarpType::Kumaraswamy:
      return "kumaraswamy";
    case WarpType::NeuralMono:
      return "neural_mono(" + std::to_string(n_hidden) + ")";
    case WarpType::MLP: {
      std::string s = "mlp(";
      for (arma::uword i = 0; i < hidden_dims.size(); ++i) {
        if (i > 0)
          s += ":";
        s += std::to_string(hidden_dims[i]);
      }
      s += "," + std::to_string(d_out) + "," + activation + ")";
      return s;
    }
    case WarpType::Embedding: {
      std::string s = "categorical(";
      if (!level_names.empty()) {
        s += "[";
        for (arma::uword i = 0; i < level_names.size(); ++i) {
          if (i > 0)
            s += ",";
          s += "\"" + level_names[i] + "\"";
        }
        s += "]";
      } else {
        s += std::to_string(n_levels);
      }
      s += "," + std::to_string(embed_dim) + ")";
      return s;
    }
    case WarpType::Ordinal: {
      std::string s = "ordinal(";
      if (!level_names.empty()) {
        s += "[";
        for (arma::uword i = 0; i < level_names.size(); ++i) {
          if (i > 0)
            s += ",";
          s += "\"" + level_names[i] + "\"";
        }
        s += "]";
      } else {
        s += std::to_string(n_levels);
      }
      s += ")";
      return s;
    }
    case WarpType::MLPJoint: {
      std::string s = "mlp_joint(";
      for (arma::uword i = 0; i < hidden_dims.size(); ++i) {
        if (i > 0)
          s += ":";
        s += std::to_string(hidden_dims[i]);
      }
      s += "," + std::to_string(d_out) + "," + activation + ")";
      return s;
    }
    case WarpType::Knots: {
      if (!knot_positions.empty() && knot_positions.size() == n_knots) {
        // Emit explicit positions
        std::string s = "knots(";
        for (arma::uword i = 0; i < n_knots; ++i) {
          if (i > 0)
            s += ":";
          std::ostringstream oss;
          oss << std::setprecision(15) << knot_positions[i];
          s += oss.str();
        }
        s += ")";
        return s;
      }
      return "knots(" + std::to_string(n_knots) + ")";
    }
  }
  return "unknown";
}

// *************************************************************************
//  WarpNone
// *************************************************************************

arma::mat WarpNone::forward(const arma::vec& x) const {
  return arma::mat(x);  // (n × 1)
}

arma::vec WarpNone::deriv_input(double /*x_val*/) const {
  return {1.0};
}

arma::vec WarpNone::backward(const arma::vec& /*x*/, const arma::mat& /*dL_dPhi*/) const {
  return {};  // no params → empty gradient
}

// *************************************************************************
//  WarpAffine  :  w(x) = a·x + b
// *************************************************************************

WarpAffine::WarpAffine() : m_a(1.0), m_b(0.0) {}

arma::vec WarpAffine::get_params() const {
  return {m_a, m_b};
}

void WarpAffine::set_params(const arma::vec& p) {
  m_a = p(0);
  m_b = p(1);
}

arma::mat WarpAffine::forward(const arma::vec& x) const {
  return arma::mat(m_a * x + m_b);
}

arma::vec WarpAffine::deriv_input(double /*x_val*/) const {
  return {m_a};
}

arma::vec WarpAffine::backward(const arma::vec& x, const arma::mat& dL_dPhi) const {
  // dL/da = Σ dL/dφ_i · x_i,   dL/db = Σ dL/dφ_i
  arma::vec grad(2);
  grad(0) = arma::dot(dL_dPhi.col(0), x);
  grad(1) = arma::accu(dL_dPhi.col(0));
  return grad;
}

std::string WarpAffine::describe() const {
  std::ostringstream s;
  s << "Affine(a=" << m_a << ", b=" << m_b << ")";
  return s.str();
}

// *************************************************************************
//  WarpBoxCox  :  w(x) = (u^λ − 1)/λ   (λ via unconstrained param)
//  If λ ≈ 0 → log(u).   u(x) = max((x − xlo)/(xhi − xlo), 1e-10) maps the
//  training range onto (0, 1], so the transform stays well-posed for x <= 0
//  and for inputs far from O(1). The default range [0, 1] reproduces the
//  historical  u = max(x, 1e-10)  exactly.
// *************************************************************************

namespace {
constexpr double kBoxCoxFloor = 1e-10;
}

WarpBoxCox::WarpBoxCox() : m_lambda(1.0) {}

arma::vec WarpBoxCox::get_params() const {
  return {m_lambda};
}

void WarpBoxCox::set_params(const arma::vec& p) {
  m_lambda = p(0);
}

void WarpBoxCox::set_input_range(double lo, double hi) {
  if (std::isfinite(lo) && std::isfinite(hi) && (hi - lo) > 1e-12) {
    m_xlo = lo;
    m_xhi = hi;
  } else if (std::isfinite(lo)) {
    // Degenerate span (e.g. a constant column): centre it in the reference
    // interval so the warp sees a well-posed mid-domain value.
    m_xlo = lo - 0.5;
    m_xhi = lo + 0.5;
  } else {
    m_xlo = 0.0;
    m_xhi = 1.0;
  }
}

double WarpBoxCox::to_pos(double x) const {
  return std::max((x - m_xlo) / (m_xhi - m_xlo), kBoxCoxFloor);
}

arma::mat WarpBoxCox::forward(const arma::vec& x) const {
  arma::vec out(x.n_elem);
  for (arma::uword i = 0; i < x.n_elem; ++i) {
    double xi = to_pos(x(i));
    if (std::abs(m_lambda) < 1e-6)
      out(i) = std::log(xi);
    else
      out(i) = (std::pow(xi, m_lambda) - 1.0) / m_lambda;
  }
  return arma::mat(out);
}

arma::vec WarpBoxCox::deriv_input(double x_val) const {
  double xi = to_pos(x_val);
  // Chain rule through the affine input map u(x): dΦ/dx = (dΦ/du) / (xhi - xlo).
  double dphi_du = (std::abs(m_lambda) < 1e-6) ? 1.0 / xi : std::pow(xi, m_lambda - 1.0);
  return {dphi_du / (m_xhi - m_xlo)};
}

arma::vec WarpBoxCox::backward(const arma::vec& x, const arma::mat& dL_dPhi) const {
  // dw/dλ = [u^λ (λ ln(u) − 1) + 1] / λ²  (for λ ≠ 0)
  double grad_lambda = 0.0;
  for (arma::uword i = 0; i < x.n_elem; ++i) {
    double xi = to_pos(x(i));
    double dw_dl;
    if (std::abs(m_lambda) < 1e-6) {
      dw_dl = 0.5 * std::log(xi) * std::log(xi);  // Taylor approx
    } else {
      double xp = std::pow(xi, m_lambda);
      dw_dl = (xp * (m_lambda * std::log(xi) - 1.0) + 1.0) / (m_lambda * m_lambda);
    }
    grad_lambda += dL_dPhi(i, 0) * dw_dl;
  }
  return {grad_lambda};
}

std::string WarpBoxCox::describe() const {
  std::ostringstream s;
  s << "BoxCox(lambda=" << m_lambda << ", range=[" << m_xlo << "," << m_xhi << "])";
  return s.str();
}

// *************************************************************************
//  WarpKumaraswamy  :  w(x) = 1 − (1 − x^a)^b    on [0,1]
//  a, b > 0 enforced via exp(raw)
// *************************************************************************

WarpKumaraswamy::WarpKumaraswamy() : m_log_a(0.0), m_log_b(0.0) {}

arma::vec WarpKumaraswamy::get_params() const {
  return {m_log_a, m_log_b};
}

void WarpKumaraswamy::set_params(const arma::vec& p) {
  m_log_a = p(0);
  m_log_b = p(1);
}

void WarpKumaraswamy::param_bounds(arma::vec& lo, arma::vec& hi) const {
  const double m = std::log(kMaxShape);
  lo = {-m, -m};
  hi = {m, m};
}

void WarpKumaraswamy::set_input_range(double lo, double hi) {
  if (std::isfinite(lo) && std::isfinite(hi) && (hi - lo) > 1e-12) {
    m_xlo = lo;
    m_xhi = hi;
  } else if (std::isfinite(lo)) {
    // Degenerate span (e.g. a constant column): centre it in the reference
    // interval so the warp sees a well-posed mid-domain value.
    m_xlo = lo - 0.5;
    m_xhi = lo + 0.5;
  } else {
    m_xlo = 0.0;
    m_xhi = 1.0;
  }
}

double WarpKumaraswamy::to_unit(double x) const {
  double u = (x - m_xlo) / (m_xhi - m_xlo);
  return std::clamp(u, 1e-10, 1.0 - 1e-10);
}

arma::mat WarpKumaraswamy::forward(const arma::vec& x) const {
  double a = std::exp(m_log_a);
  double b = std::exp(m_log_b);
  arma::vec out(x.n_elem);
  for (arma::uword i = 0; i < x.n_elem; ++i) {
    double xi = to_unit(x(i));
    out(i) = 1.0 - std::pow(1.0 - std::pow(xi, a), b);
  }
  return arma::mat(out);
}

arma::vec WarpKumaraswamy::deriv_input(double x_val) const {
  double a = std::exp(m_log_a);
  double b = std::exp(m_log_b);
  double xi = to_unit(x_val);
  double xa = std::pow(xi, a);
  double u = 1.0 - xa;
  // Chain rule through the affine input map u(x): dΦ/dx = (dΦ/du) / (xhi - xlo).
  return {a * b * std::pow(xi, a - 1.0) * std::pow(u, b - 1.0) / (m_xhi - m_xlo)};
}

arma::vec WarpKumaraswamy::backward(const arma::vec& x, const arma::mat& dL_dPhi) const {
  double a = std::exp(m_log_a);
  double b = std::exp(m_log_b);
  double grad_log_a = 0.0, grad_log_b = 0.0;

  for (arma::uword i = 0; i < x.n_elem; ++i) {
    double xi = to_unit(x(i));
    double xa = std::pow(xi, a);
    double u = 1.0 - xa;         // 1 - x^a
    double ub = std::pow(u, b);  // (1 - x^a)^b

    // dw/da = b · (1−x^a)^{b−1} · x^a · ln(x) · a   (chain via log_a)
    double dw_da = b * std::pow(u, b - 1.0) * xa * std::log(xi);
    // dw/d(log_a) = dw/da · da/d(log_a) = dw/da · a
    grad_log_a += dL_dPhi(i, 0) * dw_da * a;

    // dw/db = -(1−x^a)^b · ln(1−x^a)
    double dw_db = -ub * std::log(std::max(u, 1e-30));
    // dw/d(log_b) = dw/db · b
    grad_log_b += dL_dPhi(i, 0) * dw_db * b;
  }
  return {grad_log_a, grad_log_b};
}

std::string WarpKumaraswamy::describe() const {
  std::ostringstream s;
  s << "Kumaraswamy(a=" << std::exp(m_log_a) << ", b=" << std::exp(m_log_b) << ", range=[" << m_xlo << "," << m_xhi
    << "])";
  return s.str();
}

// *************************************************************************
//  WarpNeuralMono  :  monotone 1-hidden-layer network
//
//  w(x) = |W2|^T softplus(|W1| x + b1) + b2
//
//  Monotonicity is guaranteed by using positive weights (stored as exp(raw)).
// *************************************************************************

WarpNeuralMono::WarpNeuralMono(arma::uword n_hidden, uint64_t seed) : m_H(n_hidden) {
  arma::arma_rng::set_seed(seed);
  double scale = std::sqrt(2.0 / 1.0);  // Kaiming init, fan_in = 1
  m_raw_W1 = arma::randn<arma::vec>(m_H) * scale;
  m_b1 = arma::zeros<arma::vec>(m_H);
  m_raw_W2 = arma::randn<arma::vec>(m_H) * scale;
  m_b2 = 0.0;
}

arma::uword WarpNeuralMono::n_params() const {
  return m_H + m_H + m_H + 1;  // W1 + b1 + W2 + b2
}

arma::vec WarpNeuralMono::get_params() const {
  arma::vec p(n_params());
  arma::uword idx = 0;
  p.subvec(idx, idx + m_H - 1) = m_raw_W1;
  idx += m_H;
  p.subvec(idx, idx + m_H - 1) = m_b1;
  idx += m_H;
  p.subvec(idx, idx + m_H - 1) = m_raw_W2;
  idx += m_H;
  p(idx) = m_b2;
  return p;
}

void WarpNeuralMono::set_params(const arma::vec& p) {
  arma::uword idx = 0;
  m_raw_W1 = p.subvec(idx, idx + m_H - 1);
  idx += m_H;
  m_b1 = p.subvec(idx, idx + m_H - 1);
  idx += m_H;
  m_raw_W2 = p.subvec(idx, idx + m_H - 1);
  idx += m_H;
  m_b2 = p(idx);
}

void WarpNeuralMono::set_input_range(double lo, double hi) {
  if (std::isfinite(lo) && std::isfinite(hi) && (hi - lo) > 1e-12) {
    m_xlo = lo;
    m_xhi = hi;
  } else if (std::isfinite(lo)) {
    // Degenerate span (e.g. a constant column): centre it in the reference
    // interval so the warp sees a well-posed mid-domain value.
    m_xlo = lo - 0.5;
    m_xhi = lo + 0.5;
  } else {
    m_xlo = 0.0;
    m_xhi = 1.0;
  }
}

double WarpNeuralMono::to_scaled(double x) const {
  return (x - m_xlo) / (m_xhi - m_xlo);  // affine only, no clamp
}

arma::mat WarpNeuralMono::forward(const arma::vec& x) const {
  arma::vec W1 = arma::exp(m_raw_W1);  // positive weights
  arma::vec W2 = arma::exp(m_raw_W2);

  arma::uword n = x.n_elem;
  arma::vec out(n);

  for (arma::uword i = 0; i < n; ++i) {
    // hidden = softplus(W1 * u(x_i) + b1)
    arma::vec z = W1 * to_scaled(x(i)) + m_b1;
    arma::vec h(m_H);
    for (arma::uword j = 0; j < m_H; ++j)
      h(j) = softplus(z(j));

    out(i) = arma::dot(W2, h) + m_b2;
  }
  return arma::mat(out);
}

arma::vec WarpNeuralMono::deriv_input(double x_val) const {
  arma::vec W1 = arma::exp(m_raw_W1);
  arma::vec W2 = arma::exp(m_raw_W2);
  // out = W2 · softplus(W1*u + b1) + b2,  u = (x - xlo)/(xhi - xlo)
  // ∂out/∂x = [Σ_j W2_j · σ(W1_j*u + b1_j) · W1_j] · du/dx
  arma::vec z = W1 * to_scaled(x_val) + m_b1;
  double deriv = 0.0;
  for (arma::uword j = 0; j < m_H; ++j) {
    double sig_j = 1.0 / (1.0 + std::exp(-z(j)));
    deriv += W2(j) * sig_j * W1(j);
  }
  return {deriv / (m_xhi - m_xlo)};
}

arma::vec WarpNeuralMono::backward(const arma::vec& x, const arma::mat& dL_dPhi) const {
  arma::vec W1 = arma::exp(m_raw_W1);
  arma::vec W2 = arma::exp(m_raw_W2);

  arma::vec grad(n_params(), arma::fill::zeros);
  arma::uword n = x.n_elem;

  arma::vec g_rW1(m_H, arma::fill::zeros);
  arma::vec g_b1(m_H, arma::fill::zeros);
  arma::vec g_rW2(m_H, arma::fill::zeros);
  double g_b2 = 0.0;

  for (arma::uword i = 0; i < n; ++i) {
    double dl = dL_dPhi(i, 0);
    double ui = to_scaled(x(i));
    arma::vec z = W1 * ui + m_b1;
    arma::vec h(m_H), sig(m_H);
    for (arma::uword j = 0; j < m_H; ++j) {
      h(j) = softplus(z(j));
      sig(j) = 1.0 / (1.0 + std::exp(-z(j)));  // sigmoid = d(softplus)/dz
    }

    // d(out)/d(W2) = h  → d/d(raw_W2) = h * W2 (chain via exp)
    g_rW2 += dl * (h % W2);
    g_b2 += dl;

    // d(out)/d(h) = W2
    arma::vec dout_dh = W2;

    // d(h_j)/d(z_j) = sigmoid(z_j)
    arma::vec dh_dz = sig;

    // d(z_j)/d(W1_j) = u_i  → d/d(raw_W1) = u_i * W1 (chain via exp)
    g_rW1 += dl * (dout_dh % dh_dz) * ui % W1;
    g_b1 += dl * (dout_dh % dh_dz);
  }

  arma::uword idx = 0;
  grad.subvec(idx, idx + m_H - 1) = g_rW1;
  idx += m_H;
  grad.subvec(idx, idx + m_H - 1) = g_b1;
  idx += m_H;
  grad.subvec(idx, idx + m_H - 1) = g_rW2;
  idx += m_H;
  grad(idx) = g_b2;
  return grad;
}

std::string WarpNeuralMono::describe() const {
  std::ostringstream s;
  s << "NeuralMono(H=" << m_H << ", " << n_params() << " params, range=[" << m_xlo << "," << m_xhi << "])";
  return s.str();
}

// *************************************************************************
//  WarpMLP  :  unconstrained multi-layer perceptron
//
//  x (scalar) → [Linear → activation]×L → Linear → φ(x) ∈ ℝ^{d_out}
//
//  Weights are free (not constrained to be positive), so this warp
//  is NOT monotone.  Multi-dimensional output for richer feature spaces.
//  Kaiming init for ReLU/SELU/ELU, Glorot for Tanh/Sigmoid.
// *************************************************************************

// -- activation helpers (static) ------------------------------------------

arma::mat WarpMLP::apply_act(const arma::mat& Z, Act act) {
  switch (act) {
    case Act::ReLU:
      return arma::clamp(Z, 0.0, arma::datum::inf);
    case Act::SELU: {
      const double alpha = 1.6732632423543772;
      const double lambda = 1.0507009873554805;
      arma::mat out = Z;
      out.transform([&](double z) { return lambda * (z >= 0.0 ? z : alpha * (std::exp(z) - 1.0)); });
      return out;
    }
    case Act::Tanh:
      return arma::tanh(Z);
    case Act::Sigmoid:
      return 1.0 / (1.0 + arma::exp(-Z));
    case Act::ELU: {
      arma::mat out = Z;
      out.transform([](double z) { return z >= 0.0 ? z : std::exp(z) - 1.0; });
      return out;
    }
  }
  return Z;
}

arma::mat WarpMLP::act_deriv(const arma::mat& Z, Act act) {
  switch (act) {
    case Act::ReLU: {
      arma::mat d(arma::size(Z));
      for (arma::uword i = 0; i < Z.n_elem; ++i)
        d(i) = Z(i) > 0.0 ? 1.0 : 0.0;
      return d;
    }
    case Act::SELU: {
      const double alpha = 1.6732632423543772;
      const double lambda = 1.0507009873554805;
      arma::mat d(arma::size(Z));
      for (arma::uword i = 0; i < Z.n_elem; ++i)
        d(i) = Z(i) >= 0.0 ? lambda : lambda * alpha * std::exp(Z(i));
      return d;
    }
    case Act::Tanh: {
      arma::mat t = arma::tanh(Z);
      return 1.0 - t % t;
    }
    case Act::Sigmoid: {
      arma::mat s = 1.0 / (1.0 + arma::exp(-Z));
      return s % (1.0 - s);
    }
    case Act::ELU: {
      arma::mat d(arma::size(Z));
      for (arma::uword i = 0; i < Z.n_elem; ++i)
        d(i) = Z(i) >= 0.0 ? 1.0 : std::exp(Z(i));
      return d;
    }
  }
  return arma::ones(arma::size(Z));
}

WarpMLP::Act WarpMLP::parse_act(const std::string& s) {
  std::string lower = s;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  if (lower == "relu")
    return Act::ReLU;
  if (lower == "selu")
    return Act::SELU;
  if (lower == "tanh")
    return Act::Tanh;
  if (lower == "sigmoid")
    return Act::Sigmoid;
  if (lower == "elu")
    return Act::ELU;
  throw std::invalid_argument("WarpMLP: unknown activation: " + s);
}

// -- construction -----------------------------------------------------------

WarpMLP::WarpMLP(const std::vector<arma::uword>& hidden_dims, arma::uword d_out, Act activation, uint64_t seed)
    : m_d_out(d_out), m_act(activation) {
  if (hidden_dims.empty())
    throw std::invalid_argument("WarpMLP: need at least one hidden layer");

  arma::arma_rng::set_seed(seed);

  // Build layers:  1 → hidden[0] → hidden[1] → … → d_out
  arma::uword prev = 1;  // scalar input
  for (arma::uword l = 0; l < hidden_dims.size(); ++l) {
    arma::uword cur = hidden_dims[l];
    // Kaiming init for ReLU/SELU/ELU, Glorot for Tanh/Sigmoid
    double scale;
    if (m_act == Act::Tanh || m_act == Act::Sigmoid)
      scale = std::sqrt(6.0 / static_cast<double>(prev + cur));
    else
      scale = std::sqrt(2.0 / static_cast<double>(prev));

    m_W.push_back(scale * arma::randn<arma::mat>(prev, cur));
    m_b.push_back(arma::zeros<arma::vec>(cur));
    prev = cur;
  }
  // Output layer (linear, no activation)
  double scale_out = std::sqrt(2.0 / static_cast<double>(prev));
  m_W.push_back(scale_out * arma::randn<arma::mat>(prev, d_out));
  m_b.push_back(arma::zeros<arma::vec>(d_out));

  count_params();
}

void WarpMLP::count_params() {
  m_n_params = 0;
  for (arma::uword l = 0; l < m_W.size(); ++l)
    m_n_params += m_W[l].n_elem + m_b[l].n_elem;
}

void WarpMLP::set_input_range(double lo, double hi) {
  if (std::isfinite(lo) && std::isfinite(hi) && (hi - lo) > 1e-12) {
    m_xlo = lo;
    m_xhi = hi;
  } else if (std::isfinite(lo)) {
    // Degenerate span (e.g. a constant column): centre it in the reference
    // interval so the warp sees a well-posed mid-domain value.
    m_xlo = lo - 0.5;
    m_xhi = lo + 0.5;
  } else {
    m_xlo = 0.0;
    m_xhi = 1.0;
  }
}

double WarpMLP::to_scaled(double x) const {
  return (x - m_xlo) / (m_xhi - m_xlo);  // affine only, no clamp
}

// -- param serialisation ----------------------------------------------------

arma::vec WarpMLP::get_params() const {
  arma::vec p(m_n_params);
  arma::uword idx = 0;
  for (arma::uword l = 0; l < m_W.size(); ++l) {
    std::memcpy(p.memptr() + idx, m_W[l].memptr(), m_W[l].n_elem * sizeof(double));
    idx += m_W[l].n_elem;
    std::memcpy(p.memptr() + idx, m_b[l].memptr(), m_b[l].n_elem * sizeof(double));
    idx += m_b[l].n_elem;
  }
  return p;
}

void WarpMLP::set_params(const arma::vec& p) {
  arma::uword idx = 0;
  for (arma::uword l = 0; l < m_W.size(); ++l) {
    std::memcpy(m_W[l].memptr(), p.memptr() + idx, m_W[l].n_elem * sizeof(double));
    idx += m_W[l].n_elem;
    std::memcpy(m_b[l].memptr(), p.memptr() + idx, m_b[l].n_elem * sizeof(double));
    idx += m_b[l].n_elem;
  }
}

// -- forward ----------------------------------------------------------------

arma::mat WarpMLP::forward(const arma::vec& x) const {
  // x is (n × 1) scalar input; affinely map to [0,1] then reshape for matmul
  arma::mat H(x.n_elem, 1);
  H.col(0) = (x - m_xlo) / (m_xhi - m_xlo);

  const arma::uword L = m_W.size();
  for (arma::uword l = 0; l < L; ++l) {
    arma::mat Z = H * m_W[l];
    Z.each_row() += m_b[l].t();

    // Activation on all layers except the last (output = linear)
    if (l + 1 < L)
      H = apply_act(Z, m_act);
    else
      H = Z;
  }
  return H;  // (n × d_out)
}

arma::vec WarpMLP::deriv_input(double x_val) const {
  const arma::uword L = m_W.size();

  // Forward pass for single scalar input — cache pre-activations
  std::vector<arma::mat> Z_cache(L);
  arma::mat H(1, 1);
  H(0, 0) = to_scaled(x_val);
  for (arma::uword l = 0; l < L; ++l) {
    arma::mat Z = H * m_W[l];
    Z.each_row() += m_b[l].t();
    Z_cache[l] = Z;
    if (l + 1 < L)
      H = apply_act(Z, m_act);
  }

  // Backward: J = W_{L-1}^T · diag(σ'(z_{L-2})) · ... · W_0^T → (d_out × 1)
  arma::mat J = m_W[L - 1].t();  // (d_out × w_{L-1})
  for (int l = static_cast<int>(L) - 2; l >= 0; --l) {
    arma::mat act_d = act_deriv(Z_cache[l], m_act);  // (1 × w_{l+1})
    J.each_row() %= act_d;
    J = J * m_W[l].t();
  }
  // Chain rule through the affine input map u(x): dΦ/dx = (dΦ/du) / (xhi - xlo).
  return J.col(0) / (m_xhi - m_xlo);  // (d_out)-vector (d_in=1 for per-dim warp)
}

// -- backward ---------------------------------------------------------------

arma::vec WarpMLP::backward(const arma::vec& x, const arma::mat& dL_dPhi) const {
  const arma::uword L = m_W.size();

  // Forward pass with caching
  std::vector<arma::mat> Z_cache(L);
  std::vector<arma::mat> H_cache(L + 1);
  H_cache[0] = arma::mat(x.n_elem, 1);
  H_cache[0].col(0) = (x - m_xlo) / (m_xhi - m_xlo);

  for (arma::uword l = 0; l < L; ++l) {
    arma::mat Z = H_cache[l] * m_W[l];
    Z.each_row() += m_b[l].t();
    Z_cache[l] = Z;
    if (l + 1 < L)
      H_cache[l + 1] = apply_act(Z, m_act);
    else
      H_cache[l + 1] = Z;
  }

  // Backward pass
  arma::vec grad(m_n_params, arma::fill::zeros);
  arma::mat delta = dL_dPhi;  // (n × d_out)

  arma::uword idx = m_n_params;

  for (int l = static_cast<int>(L) - 1; l >= 0; --l) {
    bool is_last = (static_cast<arma::uword>(l) + 1 == L);

    // Through activation (identity on last layer)
    if (!is_last)
      delta = delta % act_deriv(Z_cache[l], m_act);

    // ∂L/∂b_l
    arma::vec db = arma::sum(delta, 0).t();
    idx -= m_b[l].n_elem;
    grad.subvec(idx, idx + m_b[l].n_elem - 1) = db;

    // ∂L/∂W_l = H_{l}^T · delta
    arma::mat dW = H_cache[l].t() * delta;
    idx -= m_W[l].n_elem;
    grad.subvec(idx, idx + m_W[l].n_elem - 1) = arma::vectorise(dW);

    // Propagate
    if (l > 0)
      delta = delta * m_W[l].t();
  }
  return grad;
}

// -- describe ---------------------------------------------------------------

std::string WarpMLP::describe() const {
  std::ostringstream s;
  s << "MLP(1";
  for (arma::uword l = 0; l < m_W.size(); ++l)
    s << " -> " << m_W[l].n_cols;
  s << ", " << m_n_params << " params, range=[" << m_xlo << "," << m_xhi << "])";
  return s.str();
}

// *************************************************************************
//  WarpMLPJoint  :  full-input MLP  x ∈ ℝ^d → φ(x) ∈ ℝ^{d_out}
//
//  Identical architecture to WarpMLP but with d_in > 1.
//  Reuses WarpMLP::apply_act / act_deriv (static methods).
// *************************************************************************

WarpMLPJoint::WarpMLPJoint(arma::uword d_in,
                           const std::vector<arma::uword>& hidden_dims,
                           arma::uword d_out,
                           Act activation,
                           uint64_t seed)
    : m_d_in(d_in), m_d_out(d_out), m_act(activation) {
  if (hidden_dims.empty())
    throw std::invalid_argument("WarpMLPJoint: need at least one hidden layer");

  m_xlo = arma::zeros<arma::rowvec>(d_in);  // default: identity input map on [0, 1]
  m_xhi = arma::ones<arma::rowvec>(d_in);

  arma::arma_rng::set_seed(seed);

  arma::uword prev = d_in;
  for (auto cur : hidden_dims) {
    double scale
        = (m_act == Act::Tanh || m_act == Act::Sigmoid) ? std::sqrt(6.0 / (prev + cur)) : std::sqrt(2.0 / prev);
    m_W.push_back(scale * arma::randn<arma::mat>(prev, cur));
    m_b.push_back(arma::zeros<arma::vec>(cur));
    prev = cur;
  }
  double scale_out = std::sqrt(2.0 / prev);
  m_W.push_back(scale_out * arma::randn<arma::mat>(prev, d_out));
  m_b.push_back(arma::zeros<arma::vec>(d_out));
  count_params();
}

void WarpMLPJoint::count_params() {
  m_n_params = 0;
  for (arma::uword l = 0; l < m_W.size(); ++l)
    m_n_params += m_W[l].n_elem + m_b[l].n_elem;
}

arma::vec WarpMLPJoint::get_params() const {
  arma::vec p(m_n_params);
  arma::uword idx = 0;
  for (arma::uword l = 0; l < m_W.size(); ++l) {
    std::memcpy(p.memptr() + idx, m_W[l].memptr(), m_W[l].n_elem * sizeof(double));
    idx += m_W[l].n_elem;
    std::memcpy(p.memptr() + idx, m_b[l].memptr(), m_b[l].n_elem * sizeof(double));
    idx += m_b[l].n_elem;
  }
  return p;
}

void WarpMLPJoint::set_params(const arma::vec& p) {
  arma::uword idx = 0;
  for (arma::uword l = 0; l < m_W.size(); ++l) {
    std::memcpy(m_W[l].memptr(), p.memptr() + idx, m_W[l].n_elem * sizeof(double));
    idx += m_W[l].n_elem;
    std::memcpy(m_b[l].memptr(), p.memptr() + idx, m_b[l].n_elem * sizeof(double));
    idx += m_b[l].n_elem;
  }
}

void WarpMLPJoint::set_input_ranges(const arma::rowvec& lo, const arma::rowvec& hi) {
  m_xlo = arma::zeros<arma::rowvec>(m_d_in);
  m_xhi = arma::ones<arma::rowvec>(m_d_in);
  for (arma::uword j = 0; j < m_d_in && j < lo.n_elem && j < hi.n_elem; ++j) {
    if (std::isfinite(lo(j)) && std::isfinite(hi(j)) && (hi(j) - lo(j)) > 1e-12) {
      m_xlo(j) = lo(j);
      m_xhi(j) = hi(j);
    } else if (std::isfinite(lo(j))) {
      // Degenerate column → centre it in the reference interval.
      m_xlo(j) = lo(j) - 0.5;
      m_xhi(j) = lo(j) + 0.5;
    }
  }
}

arma::mat WarpMLPJoint::to_scaled(const arma::mat& X) const {
  arma::mat U = X;
  U.each_row() -= m_xlo;
  U.each_row() /= (m_xhi - m_xlo);  // affine only, no clamp
  return U;
}

arma::mat WarpMLPJoint::forward(const arma::mat& X) const {
  arma::mat H = to_scaled(X);  // (n × d_in)
  const arma::uword L = m_W.size();
  for (arma::uword l = 0; l < L; ++l) {
    arma::mat Z = H * m_W[l];
    Z.each_row() += m_b[l].t();
    if (l + 1 < L)
      H = WarpMLP::apply_act(Z, m_act);
    else
      H = Z;
  }
  return H;  // (n × d_out)
}

arma::vec WarpMLPJoint::backward(const arma::mat& X, const arma::mat& dL_dPhi) const {
  const arma::uword L = m_W.size();

  // Forward with caching (H_cache[0] = affinely scaled inputs)
  std::vector<arma::mat> Z_cache(L);
  std::vector<arma::mat> H_cache(L + 1);
  H_cache[0] = to_scaled(X);

  for (arma::uword l = 0; l < L; ++l) {
    arma::mat Z = H_cache[l] * m_W[l];
    Z.each_row() += m_b[l].t();
    Z_cache[l] = Z;
    if (l + 1 < L)
      H_cache[l + 1] = WarpMLP::apply_act(Z, m_act);
    else
      H_cache[l + 1] = Z;
  }

  // Backward — param gradients only (∂/∂W, ∂/∂b), unaffected by the fixed
  // affine input map beyond the scaled values already in H_cache[0].
  arma::vec grad(m_n_params, arma::fill::zeros);
  arma::mat delta = dL_dPhi;
  arma::uword idx = m_n_params;

  for (int l = static_cast<int>(L) - 1; l >= 0; --l) {
    bool is_last = (static_cast<arma::uword>(l) + 1 == L);
    if (!is_last)
      delta = delta % WarpMLP::act_deriv(Z_cache[l], m_act);

    arma::vec db = arma::sum(delta, 0).t();
    idx -= m_b[l].n_elem;
    grad.subvec(idx, idx + m_b[l].n_elem - 1) = db;

    arma::mat dW = H_cache[l].t() * delta;
    idx -= m_W[l].n_elem;
    grad.subvec(idx, idx + m_W[l].n_elem - 1) = arma::vectorise(dW);

    if (l > 0)
      delta = delta * m_W[l].t();
  }
  return grad;
}

arma::mat WarpMLPJoint::jacobian_input(const arma::rowvec& x) const {
  const arma::uword L = m_W.size();

  // Forward pass on the affinely scaled input — cache pre-activations Z_l
  std::vector<arma::mat> Z_cache(L);
  arma::mat H = to_scaled(arma::mat(x));
  for (arma::uword l = 0; l < L; ++l) {
    arma::mat Z = H * m_W[l];
    Z.each_row() += m_b[l].t();
    Z_cache[l] = Z;
    if (l + 1 < L)
      H = WarpMLP::apply_act(Z, m_act);
  }

  // Backward pass: J = W_{L-1}^T · diag(σ'(z_{L-2})) · W_{L-2}^T · ... · diag(σ'(z_0)) · W_0^T
  arma::mat J = m_W[L - 1].t();  // (d_out × w_{L-1})
  for (int l = static_cast<int>(L) - 2; l >= 0; --l) {
    arma::mat act_d = WarpMLP::act_deriv(Z_cache[l], m_act);  // (1 × w_{l+1})
    J.each_row() %= act_d;
    J = J * m_W[l].t();
  }
  // Chain rule through the per-column affine input map: ∂Φ/∂x_j = (∂Φ/∂u_j) / (xhi_j - xlo_j)
  J.each_row() /= (m_xhi - m_xlo);
  return J;  // (d_out × d_in)
}

std::string WarpMLPJoint::describe() const {
  std::ostringstream s;
  s << "MLPJoint(" << m_d_in;
  for (arma::uword l = 0; l < m_W.size(); ++l)
    s << " -> " << m_W[l].n_cols;
  s << ", " << m_n_params << " params, ranges=[" << m_xlo << "; " << m_xhi << "])";
  return s.str();
}

// *************************************************************************
//  WarpEmbedding  :  level l → row l of E ∈ ℝ^{L × q}
// *************************************************************************

WarpEmbedding::WarpEmbedding(arma::uword n_levels, arma::uword embed_dim, uint64_t seed)
    : m_n_levels(n_levels), m_embed_dim(embed_dim) {
  arma::arma_rng::set_seed(seed);
  double scale = std::sqrt(1.0 / embed_dim);
  m_E = scale * arma::randn<arma::mat>(n_levels, embed_dim);
}

arma::uword WarpEmbedding::n_params() const {
  return m_n_levels * m_embed_dim;
}

arma::vec WarpEmbedding::get_params() const {
  return arma::vectorise(m_E);  // column-major
}

void WarpEmbedding::set_params(const arma::vec& p) {
  m_E = arma::reshape(p, m_n_levels, m_embed_dim);
}

arma::mat WarpEmbedding::forward(const arma::vec& x) const {
  arma::uword n = x.n_elem;
  arma::mat out(n, m_embed_dim);
  for (arma::uword i = 0; i < n; ++i) {
    arma::uword level = static_cast<arma::uword>(std::round(x(i)));
    if (level >= m_n_levels)
      throw std::out_of_range("WarpEmbedding: level " + std::to_string(level) + " >= n_levels "
                              + std::to_string(m_n_levels));
    out.row(i) = m_E.row(level);
  }
  return out;
}

arma::vec WarpEmbedding::deriv_input(double /*x_val*/) const {
  return arma::vec(m_embed_dim, arma::fill::zeros);
}

arma::vec WarpEmbedding::backward(const arma::vec& x, const arma::mat& dL_dPhi) const {
  arma::mat dE(m_n_levels, m_embed_dim, arma::fill::zeros);
  for (arma::uword i = 0; i < x.n_elem; ++i) {
    arma::uword level = static_cast<arma::uword>(std::round(x(i)));
    dE.row(level) += dL_dPhi.row(i);
  }
  return arma::vectorise(dE);
}

std::string WarpEmbedding::describe() const {
  std::ostringstream s;
  s << "Embedding(L=" << m_n_levels << ", q=" << m_embed_dim << ")";
  return s.str();
}

// *************************************************************************
//  WarpOrdinal  :  level l → z_l = Σ_{k<l} exp(gap_k)
//  Enforces z_0 = 0 < z_1 < z_2 < … < z_{L-1}
// *************************************************************************

WarpOrdinal::WarpOrdinal(arma::uword n_levels, uint64_t seed) : m_n_levels(n_levels) {
  arma::arma_rng::set_seed(seed);
  m_raw_gaps = arma::zeros<arma::vec>(n_levels - 1);
}

arma::uword WarpOrdinal::n_params() const {
  return m_n_levels - 1;
}

arma::vec WarpOrdinal::get_params() const {
  return m_raw_gaps;
}

void WarpOrdinal::set_params(const arma::vec& p) {
  m_raw_gaps = p;
}

arma::mat WarpOrdinal::forward(const arma::vec& x) const {
  // Precompute positions:  z_0 = 0, z_l = z_{l-1} + exp(gap_{l-1})
  arma::vec positions(m_n_levels);
  positions(0) = 0.0;
  for (arma::uword l = 1; l < m_n_levels; ++l)
    positions(l) = positions(l - 1) + std::exp(m_raw_gaps(l - 1));

  arma::uword n = x.n_elem;
  arma::vec out(n);
  for (arma::uword i = 0; i < n; ++i) {
    arma::uword level = static_cast<arma::uword>(std::round(x(i)));
    if (level >= m_n_levels)
      throw std::out_of_range("WarpOrdinal: level out of range");
    out(i) = positions(level);
  }
  return arma::mat(out);
}

arma::vec WarpOrdinal::deriv_input(double /*x_val*/) const {
  return {0.0};
}

arma::vec WarpOrdinal::backward(const arma::vec& x, const arma::mat& dL_dPhi) const {
  // d(z_l)/d(gap_k) = exp(gap_k)  if k < l,  0 otherwise
  arma::vec grad(m_n_levels - 1, arma::fill::zeros);
  for (arma::uword i = 0; i < x.n_elem; ++i) {
    arma::uword level = static_cast<arma::uword>(std::round(x(i)));
    double dl = dL_dPhi(i, 0);
    for (arma::uword k = 0; k < std::min(level, m_n_levels - 1); ++k)
      grad(k) += dl * std::exp(m_raw_gaps(k));
  }
  return grad;
}

std::string WarpOrdinal::describe() const {
  // Compute actual positions
  arma::vec pos(m_n_levels);
  pos(0) = 0.0;
  for (arma::uword l = 1; l < m_n_levels; ++l)
    pos(l) = pos(l - 1) + std::exp(m_raw_gaps(l - 1));
  std::ostringstream s;
  s << "Ordinal(L=" << m_n_levels << ", positions=" << pos.t() << ")";
  return s.str();
}

// *************************************************************************
//  clone() implementations
// *************************************************************************

std::unique_ptr<IWarp> WarpAffine::clone() const {
  auto c = std::make_unique<WarpAffine>();
  c->set_params(get_params());
  return c;
}

std::unique_ptr<IWarp> WarpBoxCox::clone() const {
  auto c = std::make_unique<WarpBoxCox>();
  c->set_params(get_params());
  c->set_input_range(m_xlo, m_xhi);
  return c;
}

std::unique_ptr<IWarp> WarpKumaraswamy::clone() const {
  auto c = std::make_unique<WarpKumaraswamy>();
  c->set_params(get_params());
  c->set_input_range(m_xlo, m_xhi);
  return c;
}

std::unique_ptr<IWarp> WarpNeuralMono::clone() const {
  auto c = std::make_unique<WarpNeuralMono>(m_H);
  c->set_params(get_params());
  c->set_input_range(m_xlo, m_xhi);
  return c;
}

std::unique_ptr<IWarp> WarpMLP::clone() const {
  // Reconstruct hidden_dims from weight matrices
  // m_W[0..n-2] are hidden layers; m_W[n-1] is output layer
  std::vector<arma::uword> hidden_dims;
  for (size_t i = 0; i + 1 < m_W.size(); ++i)
    hidden_dims.push_back(m_W[i].n_cols);
  auto c = std::make_unique<WarpMLP>(hidden_dims, m_d_out, m_act);
  c->set_params(get_params());
  c->set_input_range(m_xlo, m_xhi);
  return c;
}

std::unique_ptr<WarpMLPJoint> WarpMLPJoint::clone() const {
  // Reconstruct hidden_dims from weight matrices
  std::vector<arma::uword> hidden_dims;
  for (size_t i = 0; i + 1 < m_W.size(); ++i)
    hidden_dims.push_back(m_W[i].n_cols);
  auto c = std::make_unique<WarpMLPJoint>(m_d_in, hidden_dims, m_d_out, m_act);
  c->set_params(get_params());
  c->set_input_ranges(m_xlo, m_xhi);
  return c;
}

std::unique_ptr<IWarp> WarpEmbedding::clone() const {
  auto c = std::make_unique<WarpEmbedding>(m_n_levels, m_embed_dim);
  c->set_params(get_params());
  return c;
}

std::unique_ptr<IWarp> WarpOrdinal::clone() const {
  auto c = std::make_unique<WarpOrdinal>(m_n_levels);
  c->set_params(get_params());
  return c;
}

// *************************************************************************
//  WarpKnots  —  piecewise-linear monotone warping (Xiong et al. 2007)
//
//  Breakpoints:  0 = t₀ < t₁ < … < t_K < t_{K+1} = 1
//  Parameters:   log-slopes r₀, …, r_K  (unconstrained)
//  Slopes:       sₖ = exp(rₖ) > 0
//  Input map:    u(x) = clamp((x − xlo) / (xhi − xlo), 0, 1)     [xlo,xhi] = training range
//  Warp:         w(u) = Σ_{j<k} sⱼ·Δtⱼ  +  sₖ·(u − tₖ)
//                for u ∈ [tₖ, t_{k+1})
//
//  The reference domain of the warp is the unit interval. Inputs are affinely
//  mapped from their training range [xlo, xhi] onto [0, 1] first (set via
//  set_input_range()); the default identity range [0, 1] keeps models fit on
//  unit-cube inputs bit-for-bit unchanged.
// *************************************************************************

WarpKnots::WarpKnots(arma::uword n_knots, const std::vector<double>& knot_positions) : m_K(n_knots) {
  if (n_knots < 1)
    throw std::invalid_argument("WarpKnots: n_knots must be >= 1");

  m_breaks.resize(m_K + 2);
  m_breaks[0] = 0.0;
  m_breaks[m_K + 1] = 1.0;

  if (knot_positions.size() == m_K) {
    for (arma::uword k = 0; k < m_K; ++k)
      m_breaks[k + 1] = knot_positions[k];
  } else if (knot_positions.empty()) {
    for (arma::uword k = 0; k < m_K; ++k)
      m_breaks[k + 1] = (k + 1.0) / (m_K + 1.0);
  } else {
    throw std::invalid_argument("WarpKnots: knot_positions must be empty (uniform) or have n_knots elements");
  }

  for (arma::uword k = 0; k < m_K + 1; ++k) {
    if (m_breaks[k] >= m_breaks[k + 1])
      throw std::invalid_argument(
          "WarpKnots: breakpoints must be strictly increasing; "
          "knot positions must lie strictly in (0,1)");
  }

  m_log_slopes = arma::zeros<arma::vec>(m_K + 1);
}

void WarpKnots::set_input_range(double lo, double hi) {
  if (std::isfinite(lo) && std::isfinite(hi) && (hi - lo) > 1e-12) {
    m_xlo = lo;
    m_xhi = hi;
  } else if (std::isfinite(lo)) {
    // Degenerate span (e.g. a constant column): centre it in the reference
    // interval so the warp sees a well-posed mid-domain value.
    m_xlo = lo - 0.5;
    m_xhi = lo + 0.5;
  } else {
    m_xlo = 0.0;
    m_xhi = 1.0;
  }
}

double WarpKnots::to_unit(double x) const {
  double u = (x - m_xlo) / (m_xhi - m_xlo);
  return std::clamp(u, 0.0, 1.0);
}

arma::uword WarpKnots::find_interval(double x) const {
  // Binary search for k such that m_breaks[k] <= x < m_breaks[k+1]
  arma::uword lo = 0, hi = m_K;
  while (lo < hi) {
    arma::uword mid = (lo + hi + 1) / 2;
    if (x >= m_breaks[mid])
      lo = mid;
    else
      hi = mid - 1;
  }
  return lo;
}

arma::vec WarpKnots::get_params() const {
  return m_log_slopes;
}

void WarpKnots::set_params(const arma::vec& p) {
  m_log_slopes = p;
}

arma::mat WarpKnots::forward(const arma::vec& x) const {
  arma::uword K1 = m_K + 1;
  arma::vec slopes = arma::exp(m_log_slopes);

  // Cumulative warp values at each breakpoint
  arma::vec cum(K1 + 1, arma::fill::zeros);
  for (arma::uword j = 0; j < K1; ++j)
    cum(j + 1) = cum(j) + slopes(j) * (m_breaks[j + 1] - m_breaks[j]);

  arma::vec out(x.n_elem);
  for (arma::uword i = 0; i < x.n_elem; ++i) {
    double xi = to_unit(x(i));
    arma::uword k = find_interval(xi);
    out(i) = cum(k) + slopes(k) * (xi - m_breaks[k]);
  }
  return arma::mat(out);
}

arma::vec WarpKnots::deriv_input(double x_val) const {
  double xi = to_unit(x_val);
  arma::uword k = find_interval(xi);
  // Chain rule through the affine input map u(x): dΦ/dx = s_k · du/dx.
  return {std::exp(m_log_slopes(k)) / (m_xhi - m_xlo)};
}

arma::vec WarpKnots::backward(const arma::vec& x, const arma::mat& dL_dPhi) const {
  arma::uword K1 = m_K + 1;
  arma::vec slopes = arma::exp(m_log_slopes);
  arma::vec grad(K1, arma::fill::zeros);

  for (arma::uword i = 0; i < x.n_elem; ++i) {
    double xi = to_unit(x(i));
    double g = dL_dPhi(i, 0);
    arma::uword k = find_interval(xi);

    // For fully traversed intervals j < k:
    // d(w)/d(log_s_j) = slopes(j) * (t_{j+1} - t_j)  via chain rule through exp
    for (arma::uword j = 0; j < k; ++j)
      grad(j) += g * slopes(j) * (m_breaks[j + 1] - m_breaks[j]);

    // For the partial interval k:
    // d(w)/d(log_s_k) = slopes(k) * (xi - t_k)
    grad(k) += g * slopes(k) * (xi - m_breaks[k]);
  }
  return grad;
}

std::string WarpKnots::describe() const {
  std::ostringstream s;
  s << "Knots(K=" << m_K << ", breaks=[";
  for (arma::uword k = 0; k <= m_K + 1; ++k) {
    if (k > 0)
      s << ",";
    s << std::setprecision(4) << m_breaks[k];
  }
  s << "], slopes=[";
  for (arma::uword k = 0; k <= m_K; ++k) {
    if (k > 0)
      s << ",";
    s << std::setprecision(4) << std::exp(m_log_slopes(k));
  }
  s << "], range=[" << std::setprecision(4) << m_xlo << "," << m_xhi << "])";
  return s.str();
}

std::unique_ptr<IWarp> WarpKnots::clone() const {
  std::vector<double> pos(m_K);
  for (arma::uword k = 0; k < m_K; ++k)
    pos[k] = m_breaks[k + 1];
  auto c = std::make_unique<WarpKnots>(m_K, pos);
  c->set_params(get_params());
  c->set_input_range(m_xlo, m_xhi);
  return c;
}

// -------------------------------------------------------------------------
WarpBaseKernel WarpKriging::parse_kernel(const std::string& name) {
  std::string s = name;
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
  if (s == "gauss" || s == "rbf")
    return WarpBaseKernel::Gauss;
  if (s == "matern3_2" || s == "matern32")
    return WarpBaseKernel::Matern32;
  if (s == "matern5_2" || s == "matern52")
    return WarpBaseKernel::Matern52;
  if (s == "exp" || s == "exponential")
    return WarpBaseKernel::Exp;
  throw std::invalid_argument("Unknown kernel: " + name);
}

// -------------------------------------------------------------------------
std::unique_ptr<IWarp> WarpKriging::make_warp(const WarpSpec& spec) const {
  switch (spec.type) {
    case WarpType::None:
      return std::make_unique<WarpNone>();
    case WarpType::Affine:
      return std::make_unique<WarpAffine>();
    case WarpType::BoxCox:
      return std::make_unique<WarpBoxCox>();
    case WarpType::Kumaraswamy:
      return std::make_unique<WarpKumaraswamy>();
    case WarpType::NeuralMono:
      return std::make_unique<WarpNeuralMono>(spec.n_hidden);
    case WarpType::MLP:
      return std::make_unique<WarpMLP>(spec.hidden_dims, spec.d_out, WarpMLP::parse_act(spec.activation));
    case WarpType::Embedding:
      return std::make_unique<WarpEmbedding>(spec.n_levels, spec.embed_dim);
    case WarpType::Ordinal:
      return std::make_unique<WarpOrdinal>(spec.n_levels);
    case WarpType::Knots:
      return std::make_unique<WarpKnots>(spec.n_knots, spec.knot_positions);
    case WarpType::MLPJoint:
      // MLPJoint is not a per-variable IWarp — handled separately in build_warps
      throw std::invalid_argument("make_warp: MLPJoint is handled by build_warps, not make_warp");
  }
  throw std::invalid_argument("Unknown WarpType");
}

void WarpKriging::build_warps() {
  m_warps.clear();
  m_feature_dim = 0;
  m_is_continuous.clear();
  m_is_joint = false;
  m_joint_warp.reset();

  // MLPJoint: single joint MLP taking all inputs together — no per-variable warps.
  if (m_warp_specs.size() == 1 && m_warp_specs[0].type == WarpType::MLPJoint) {
    m_is_joint = true;
    m_feature_dim = m_warp_specs[0].d_out;
    // m_joint_warp is instantiated later in ensure_joint_warp() (needs d_in from X).
    return;
  }

  for (const auto& spec : m_warp_specs) {
    if (spec.type == WarpType::MLPJoint)
      throw std::invalid_argument("WarpKriging: mlp_joint must be the only warp spec");
  }

  for (const auto& spec : m_warp_specs) {
    auto w = make_warp(spec);
    m_feature_dim += w->output_dim();
    m_warps.push_back(std::move(w));
    m_is_continuous.push_back(spec.type != WarpType::Embedding && spec.type != WarpType::Ordinal);
  }
}

// -------------------------------------------------------------------------
//  Push the per-variable training-input range into each warp.
//  Uses m_X_raw as it stands *after* normalise_data() (so the range is in the
//  same coordinates the warp actually receives at fit and predict time).
//  Continuous per-variable warps that have no reference domain (none, affine)
//  ignore it; discrete columns are skipped. For the joint MLP the per-column
//  ranges go in one shot.
// -------------------------------------------------------------------------
void WarpKriging::calibrate_warps() {
  calibrate_warps(m_X_raw);
}

void WarpKriging::calibrate_warps(const arma::mat& Xn) {
  if (Xn.n_rows == 0)
    return;
  if (m_is_joint) {
    if (m_joint_warp)
      m_joint_warp->set_input_ranges(arma::min(Xn, 0), arma::max(Xn, 0));
    return;
  }
  if (m_warps.empty())
    return;
  for (arma::uword j = 0; j < m_warps.size() && j < Xn.n_cols; ++j) {
    if (j < m_is_continuous.size() && !m_is_continuous[j])
      continue;
    const arma::vec col = Xn.col(j);
    m_warps[j]->set_input_range(col.min(), col.max());
  }
}

void WarpKriging::recalibrate_warps(const arma::mat& X_ref) {
  if (!m_fitted || X_ref.n_rows == 0)
    return;
  // Bring X_ref into the same (normalised) coordinates the warps see, then
  // recompute Φ / Cholesky / β̂ / σ̂² on the new warp so training-point
  // predictions stay consistent. Used by NestedKriging so every warped
  // submodel shares one input-range calibration alongside the shared
  // (theta, warp_params).
  arma::mat Xn = X_ref;
  Xn.each_row() -= m_centerX;
  Xn.each_row() /= m_scaleX;
  calibrate_warps(Xn);
  m_cache_jitter = m_chol_jitter;
  refresh_cache();
  commit_model_chol_jitter();
}

void WarpKriging::ensure_joint_warp(arma::uword d_in) {
  if (m_joint_warp && m_joint_warp->input_dim() == d_in)
    return;
  const auto& spec = m_warp_specs[0];
  m_joint_warp
      = std::make_unique<WarpMLPJoint>(d_in, spec.hidden_dims, spec.d_out, WarpMLP::parse_act(spec.activation));
}

// -------------------------------------------------------------------------
//  Validate discrete (categorical / ordinal) columns of X
// -------------------------------------------------------------------------
void WarpKriging::validate_discrete_columns(const arma::mat& X, const std::string& caller) const {
  if (m_is_joint)
    return;
  for (arma::uword j = 0; j < m_warp_specs.size(); ++j) {
    const auto& spec = m_warp_specs[j];
    if (spec.type != WarpType::Embedding && spec.type != WarpType::Ordinal)
      continue;

    const arma::vec& col = X.col(j);
    for (arma::uword i = 0; i < col.n_elem; ++i) {
      double v = col(i);
      // Check that value is a non-negative integer
      if (v < 0.0 || v != std::floor(v) || !std::isfinite(v)) {
        std::string type_name = (spec.type == WarpType::Embedding) ? "categorical" : "ordinal";
        throw std::invalid_argument(caller + ": column " + std::to_string(j) + " (" + type_name
                                    + ") contains non-integer value " + std::to_string(v) + " at row "
                                    + std::to_string(i));
      }
      arma::uword level = static_cast<arma::uword>(v);
      if (level >= spec.n_levels) {
        std::string type_name = (spec.type == WarpType::Embedding) ? "categorical" : "ordinal";
        std::string msg = caller + ": column " + std::to_string(j) + " (" + type_name + ") has level "
                          + std::to_string(level) + " at row " + std::to_string(i) + " but n_levels is "
                          + std::to_string(spec.n_levels);
        if (!spec.level_names.empty()) {
          msg += " (valid names: ";
          for (arma::uword k = 0; k < spec.level_names.size(); ++k) {
            if (k > 0)
              msg += ", ";
            msg += std::to_string(k) + "=\"" + spec.level_names[k] + "\"";
          }
          msg += ")";
        }
        throw std::invalid_argument(msg);
      }
    }
  }
}

// -------------------------------------------------------------------------
//  Constructors (string-based public API)
// -------------------------------------------------------------------------
static std::vector<WarpSpec> parse_warp_strings(const std::vector<std::string>& strs) {
  std::vector<WarpSpec> specs;
  specs.reserve(strs.size());
  for (const auto& s : strs)
    specs.push_back(WarpSpec::from_string(s));
  return specs;
}

void WarpKriging::make_Cov(const std::string& kernel) {
  // Normalise aliases to the canonical names used by Covariance::resolve
  std::string s = kernel;
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
  if (s == "rbf")
    s = "gauss";
  else if (s == "exponential")
    s = "exp";
  else if (s == "matern32")
    s = "matern3_2";
  else if (s == "matern52")
    s = "matern5_2";

  KrigingImpl::make_Cov(s);
}

void WarpKriging::init_from_specs(const std::vector<WarpSpec>& specs, const std::string& kernel) {
  // Inherited members default-initialise to zero/empty; restore the safe
  // defaults the previous WarpKriging-owned members provided (especially
  // m_scaleY=1, m_regmodel=Constant) so accessors are well-defined pre-fit.
  m_normalize = false;
  m_centerY = 0.0;
  m_scaleY = 1.0;
  m_sigma2 = 1.0;
  m_regmodel = Trend::RegressionModel::Constant;

  m_warp_specs = specs;
  m_covType = kernel;
  m_base_kernel = parse_kernel(kernel);
  make_Cov(kernel);
  build_warps();
}

WarpKriging::WarpKriging(const std::vector<std::string>& warping, const std::string& kernel) {
  init_from_specs(parse_warp_strings(warping), kernel);
}

WarpKriging::WarpKriging(const arma::vec& y,
                         const arma::mat& X,
                         const std::vector<std::string>& warping,
                         const std::string& kernel,
                         const Trend::RegressionModel& regmodel,
                         bool normalize,
                         const std::string& optim,
                         const std::string& objective,
                         const std::map<std::string, std::string>& parameters,
                         const std::string& method) {
  init_from_specs(parse_warp_strings(warping), kernel);
  fit(y, X, regmodel, normalize, optim, objective, parameters, method);
}

// -------------------------------------------------------------------------
//  Apply warping:  X → Φ  (per-variable concatenation)
// -------------------------------------------------------------------------
arma::mat WarpKriging::apply_warping(const arma::mat& X) const {
  if (m_is_joint) {
    if (!m_joint_warp)
      throw std::runtime_error("apply_warping: joint warp not initialized (call ensure_joint_warp first)");
    return m_joint_warp->forward(X);
  }
  const arma::uword n = X.n_rows;
  arma::mat Phi(n, m_feature_dim);

  arma::uword col = 0;
  for (arma::uword j = 0; j < m_warps.size(); ++j) {
    arma::mat wj = m_warps[j]->forward(X.col(j));
    arma::uword dj = m_warps[j]->output_dim();
    Phi.cols(col, col + dj - 1) = wj;
    col += dj;
  }
  return Phi;
}

// -------------------------------------------------------------------------
//  Trend matrix — built on warped features Φ(X) so it scales with feature_dim
// -------------------------------------------------------------------------
arma::mat WarpKriging::build_trend_matrix(const arma::mat& X) const {
  return Trend::regressionModelMatrix(m_regmodel, apply_warping(X));
}

// -------------------------------------------------------------------------
//  Data normalisation  (only continuous variables are normalised)
// -------------------------------------------------------------------------
void WarpKriging::normalise_data() {
  arma::uword d = m_X_raw.n_cols;
  m_centerX = arma::zeros<arma::rowvec>(d);
  m_scaleX = arma::ones<arma::rowvec>(d);

  if (m_normalize) {
    for (arma::uword j = 0; j < d; ++j) {
      bool is_cont = m_is_joint || (j < m_is_continuous.size() && m_is_continuous[j]);
      if (is_cont) {
        m_centerX(j) = arma::mean(m_X_raw.col(j));
        m_scaleX(j) = arma::stddev(m_X_raw.col(j));
        if (m_scaleX(j) < 1e-12)
          m_scaleX(j) = 1.0;
        m_X_raw.col(j) = (m_X_raw.col(j) - m_centerX(j)) / m_scaleX(j);
      }
    }
    m_centerY = arma::mean(m_y);
    m_scaleY = arma::stddev(m_y);
    if (m_scaleY < 1e-12)
      m_scaleY = 1.0;
    m_y = (m_y - m_centerY) / m_scaleY;
  } else {
    m_centerY = 0.0;
    m_scaleY = 1.0;
  }
}

// =========================================================================
//  CONCENTRATED PROFILE LOG-LIKELIHOOD + BI-LEVEL OPTIMISATION
//
//  σ̂² and β̂ are profiled out (computed analytically).
//  θ is optimised in the inner loop via L-BFGS with analytical gradient.
//  Warp params are optimised in the outer loop via Adam.
// =========================================================================

// -------------------------------------------------------------------------
//  Correlation function  (R_ij = corr(φ_i, φ_j; θ),  R_ii = 1,  σ² factored out)
// -------------------------------------------------------------------------
arma::mat WarpKriging::build_Rcross(const arma::mat& Phi_new, const arma::mat& Phi_train) const {
  const arma::uword m = Phi_new.n_rows;
  const arma::uword n = Phi_train.n_rows;
  arma::mat Rc(m, n);
  // covMat_rect expects transposed inputs (d × n)
  LinearAlgebra::covMat_rect(&Rc, Phi_new.t(), Phi_train.t(), m_theta, _Cov, 1.0);
  return Rc;
}

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
//  WKModel factory
// -------------------------------------------------------------------------
void WarpKriging::populate_Model(WKModel& m, const arma::vec& theta, double sigma2) const {
  const arma::uword n = m_y.n_elem;

  // Cholesky update detection (same logic as Kriging::populate_Model)
  const bool do_update = m_fitted && (m_theta.size() == theta.size()) && (theta - m_theta).is_zero()
                         && (m_T.memptr() != nullptr) && (n > m_T.n_rows);

  // Preemptive nugget: Embedding/Ordinal warpings can collapse same-level
  // training rows in Phi, making R rank-deficient in a way Kriging never
  // encounters. m_cache_jitter on the diagonal stabilizes Cholesky and its
  // derivative during the search: m_chol_jitter, the kernel's fixed jitter
  // (LinearAlgebra::kernel_chol_jitter, same as Kriging) when it has one, else
  // LinearAlgebra::num_nugget (default 1e-10, set_num_nugget()). The
  // committed model keeps at most num_nugget (committed_chol_jitter).
  //
  // When m_noise is non-empty, the normalized covariance becomes
  //   C̃ = R_off-diag + diag(1 + nug + noise_i/σ²)
  // so the same cholCov(factor=1, diag) path carries observation-noise.
  arma::vec diag(n, arma::fill::value(1.0 + m_cache_jitter));
  if (!m_noise.is_empty()) {
    diag += m_noise / sigma2;
  }

  KrigingImpl::populate_Model(m, theta, /*alpha=*/1.0, diag, do_update, /*bench=*/nullptr);
}

WarpKriging::WKModel WarpKriging::make_Model(const arma::vec& theta, double sigma2) const {
  WKModel m = KrigingImpl::allocate_KModel();
  populate_Model(m, theta, sigma2);
  return m;
}

//  refresh_cache  — rebuild Φ, R, Cholesky, β̂ (GLS), σ̂² (MLE), α
// -------------------------------------------------------------------------
void WarpKriging::refresh_cache() {
  m_X = apply_warping(m_X_raw);
  // Extreme warp params can overflow the features (e.g. pow() in Kumaraswamy
  // or Box-Cox): fail here, the optimisers fall back to their best evaluated
  // point, instead of handing a NaN matrix to chol() ("given matrix is not
  // symmetric", #406).
  if (!m_X.is_finite())
    throw std::runtime_error("WarpKriging: non-finite warped features (warp parameters out of range)");
  m_dX = LinearAlgebra::compute_dX(m_X);
  m_maxdX = arma::max(arma::abs(m_dX), 1);
  refresh_cache_theta_only();
}

/// Rebuild R, Cholesky, β̂, σ̂², α from current m_X and m_theta.
/// Skips recomputing m_X and m_dX — use when only θ changed.
///
/// Without noise: σ² admits the closed-form MLE σ² = SSEstar / n and the
/// populate_Model diagonal is independent of σ², so a single factorization
/// suffices.  With per-observation noise the diagonal depends on σ² (via
/// noise_i/σ²), breaking the closed form.  We then run a 1D golden-section
/// search on log(σ²) minimizing -ll(σ² | θ), with 20-30 extra Cholesky
/// factorizations per refresh.  The θ-gradient remains correct by the
/// envelope theorem (σ² sits at its MLE given θ).
void WarpKriging::refresh_cache_theta_only() {
  const arma::uword n = m_y.n_elem;
  m_F = build_trend_matrix(m_X_raw);

  if (m_noise.is_empty()) {
    // Closed-form concentrated σ² (unchanged no-noise path).
    WKModel m = make_Model(m_theta, 1.0);
    m_sigma2 = std::max(m.SSEstar / n, 1e-20);
    KrigingImpl::commit_model(m);
    m_logdet = 2.0 * arma::sum(arma::log(m_T.diag()));
    return;
  }

  // Noise present: minimize neg_ll over σ² in log-space via golden-section.
  auto neg_ll = [&](double sigma2) -> double {
    WKModel m = make_Model(m_theta, sigma2);
    double logdet = 2.0 * arma::sum(arma::log(m.L.diag()));
    return 0.5 * (n * std::log(2.0 * arma::datum::pi * sigma2) + logdet + m.SSEstar / sigma2);
  };

  const double var_y = std::max(arma::var(m_y), 1e-10);
  double a = std::log(var_y * 1e-6);
  double b = std::log(var_y * 1e3);
  const double phi = 0.5 * (std::sqrt(5.0) - 1.0);
  double x1 = b - phi * (b - a), x2 = a + phi * (b - a);
  double f1 = neg_ll(std::exp(x1)), f2 = neg_ll(std::exp(x2));
  for (int iter = 0; iter < 100 && (b - a) > 1e-6; ++iter) {
    if (f1 < f2) {
      b = x2;
      x2 = x1;
      f2 = f1;
      x1 = b - phi * (b - a);
      f1 = neg_ll(std::exp(x1));
    } else {
      a = x1;
      x1 = x2;
      f1 = f2;
      x2 = a + phi * (b - a);
      f2 = neg_ll(std::exp(x2));
    }
  }
  const double sigma2_opt = std::exp(0.5 * (a + b));
  WKModel m = make_Model(m_theta, sigma2_opt);
  m_sigma2 = sigma2_opt;
  KrigingImpl::commit_model(m);
  m_logdet = 2.0 * arma::sum(arma::log(m_T.diag()));
}

// -------------------------------------------------------------------------
//  Concentrated profile log-likelihood
//    Without noise: σ² = SSEstar/n closed-form, so SSEstar/σ² = n and
//      LL(θ) = -n/2 [1 + log(2π) + log(σ̂²)] - ½ log|R|
//    With noise:  σ² sits at its inner-Brent MLE; SSEstar/σ² is NOT n:
//      LL(θ, σ²) = -n/2 log(2π σ²) - ½ log|R̃| - ½ SSEstar / σ²
// -------------------------------------------------------------------------
double WarpKriging::concentrated_ll() const {
  const double n = static_cast<double>(m_y.n_elem);
  if (m_noise.is_empty()) {
    return -0.5 * n * (1.0 + std::log(2.0 * arma::datum::pi) + std::log(m_sigma2)) - 0.5 * m_logdet;
  }
  const double SSEstar = arma::dot(m_z, m_z);
  return -0.5 * (n * std::log(2.0 * arma::datum::pi * m_sigma2) + m_logdet + SSEstar / m_sigma2);
}

double WarpKriging::logLikelihood() const {
  if (!m_fitted)
    throw std::runtime_error("WarpKriging: model not fitted");
  return std::isnan(m_objective_ll) ? concentrated_ll() : m_objective_ll;
}

double WarpKriging::committed_chol_jitter() const {
  // At most num_nugget (1e-10), not 0 as Kriging: warps can bring training
  // rows together in Phi (Kumaraswamy, saturated MLP, categorical levels), and
  // an exact factorization then leaves no predictive variance there (stdev 0,
  // NaN stdev derivative / NLPD: WarpKrigingPerWarpTest, WarpKrigingCollapseTest).
  // docs/dev/CholJitter.md.
  return std::min(m_chol_jitter, LinearAlgebra::num_nugget);
}

void WarpKriging::commit_model_chol_jitter() {
  m_cache_jitter = m_chol_jitter;
  refresh_cache_theta_only();
  m_objective_ll = concentrated_ll();
  m_model_chol_jitter = committed_chol_jitter();
  m_cache_jitter = m_model_chol_jitter;
  if (m_model_chol_jitter != m_chol_jitter)
    refresh_cache_theta_only();
}

std::tuple<double, arma::vec, arma::mat> WarpKriging::logLikelihoodFun(const arma::vec& theta,
                                                                       bool return_grad,
                                                                       bool /*return_hess*/) const {
  auto* self = const_cast<WarpKriging*>(this);
  arma::vec old_theta = m_theta;
  const double old_cache_jitter = m_cache_jitter;
  self->m_theta = theta;
  self->m_cache_jitter = m_chol_jitter;  // the objective: search jitter
  self->refresh_cache();

  double ll = concentrated_ll();
  arma::vec grad;
  if (return_grad) {
    auto [ll2, g] = concentrated_ll_and_grad_theta();
    grad = g;
    (void)ll2;
  }

  self->m_theta = old_theta;
  self->m_cache_jitter = old_cache_jitter;
  self->refresh_cache();
  return {ll, grad, arma::mat()};
}

// -------------------------------------------------------------------------
//  ∂R/∂θ_k  (analytical, n×n matrix)
// -------------------------------------------------------------------------
arma::mat WarpKriging::build_dR_dtheta_k(const arma::mat& /*Phi*/, arma::uword k) const {
  const arma::uword n = m_X.n_rows;
  arma::mat dR(n, n, arma::fill::zeros);

  for (arma::uword i = 0; i < n; ++i) {
    for (arma::uword j = i + 1; j < n; ++j) {
      arma::vec dlnCov = _DlnCovDtheta(m_dX.col(i * n + j), m_theta);
      double dR_ij = m_R(i, j) * dlnCov(k);
      dR(i, j) = dR_ij;
      dR(j, i) = dR_ij;
    }
  }
  return dR;
}

// -------------------------------------------------------------------------
//  Concentrated LL + analytical gradient w.r.t. log(θ)
//
//  ∂LL/∂θ_k = ½ tr[(α αᵀ / σ̂² - R⁻¹) · ∂R/∂θ_k]
//  ∂LL/∂(log θ_k) = ∂LL/∂θ_k · θ_k
//
//  Fused: computes all dR_k in a single pass over (i,j) pairs, sharing
//  diff, r2, r with the same loop structure as build_R.
// -------------------------------------------------------------------------
std::pair<double, arma::vec> WarpKriging::concentrated_ll_and_grad_theta() const {
  double ll = concentrated_ll();

  const arma::uword n = m_y.n_elem;
  const arma::uword d = m_theta.n_elem;

  arma::vec alpha = LinearAlgebra::solve_upper(m_T.t(), m_z);
  if ((m_Rinv.memptr() == nullptr) || (arma::size(m_Rinv) != arma::size(m_T)))
    m_Rinv = LinearAlgebra::inv_sympd(m_T);
  const arma::mat& Rinv = m_Rinv;  // lazily built above if not already cached

  arma::mat dLL_dR = 0.5 * (alpha * alpha.t() / m_sigma2 - Rinv);

  // Accumulate tr(dLL_dR · dR_k) for all k in a single pass over (i,j).
  // Returns grad in theta-space; callers apply Optim::reparam_from_deriv when reparametrize is on.
  arma::vec grad_theta(d, arma::fill::zeros);

  for (arma::uword i = 0; i < n; ++i) {
    for (arma::uword j = i + 1; j < n; ++j) {
      // Use _DlnCovDtheta like Kriging: ∂R_ij/∂θ_k = R_ij * DlnCovDtheta_k
      arma::vec dlnCov = _DlnCovDtheta(m_dX.col(i * n + j), m_theta);
      double R_ij = m_R(i, j);
      double w = 2.0 * dLL_dR(i, j);  // symmetry: (i,j) + (j,i)

      for (arma::uword k = 0; k < d; ++k) {
        grad_theta(k) += w * R_ij * dlnCov(k);
      }
    }
  }

  return {ll, grad_theta};
}

// -------------------------------------------------------------------------
//  Gradient of LL w.r.t. warp params  (backprop through R, θ held fixed)
//
//  Mirrors concentrated_ll_and_grad_theta(): dLL/dR_ij = 0.5*(alpha_i alpha_j
//  / σ̂² - Rinv_ij) (validated by the passing θ-gradient FD test), then
//  dR_ij/dΦ_i = R_ij * DlnCovDx(dPhi_ij, θ)  (sign: dPhi = Φ_i - Φ_j).
//
//  NOTE: an earlier version of this routine backpropagated through
//  K = σ̂²·R instead of R directly, using dL_dK = 0.5*(alpha alphaᵀ - Kinv)
//  paired with K_ij = σ̂²·R_ij. That introduced a spurious σ̂² scaling
//  mismatch between the two terms (Kinv already carries a 1/σ̂², but the
//  alpha·alphaᵀ term did not), which for small/large σ̂² blew up the
//  analytical gradient by many orders of magnitude relative to a finite
//  difference of the true log-likelihood -- verified by an end-to-end FD
//  check (see git history / diagnostic test). Working directly in
//  R-space (as done here) sidesteps the issue entirely.
// -------------------------------------------------------------------------
arma::mat WarpKriging::dK_dPhi(const arma::mat& Phi, const arma::mat& dL_dR) const {
  const arma::uword n = Phi.n_rows;
  const arma::uword d = Phi.n_cols;
  arma::mat dL_dPhi(n, d, arma::fill::zeros);

  // R_ij = C(dPhi_ij, θ), with dPhi_ij = Φ_i - Φ_j.
  // Φ_i affects BOTH the (i,j) and (j,i) slots of the (conceptually
  // unconstrained) gradient matrix dL_dR: ∂R_ij/∂Φ_i = R_ij·DlnCovDx(dPhi_ij,θ)
  // and, since DlnCovDx is odd in its argument and R_ji=R_ij,
  // ∂R_ji/∂Φ_i = R_ji·DlnCovDx(dPhi_ji,θ)·(-1) = R_ij·DlnCovDx(dPhi_ij,θ)
  // — i.e. the SAME quantity. As dL_dR is symmetric (dL_dR(i,j)=dL_dR(j,i)),
  // both slots contribute equally, giving a factor of 2 (matching the
  // `w = 2.0 * dLL_dR(i,j)` convention already used in
  // concentrated_ll_and_grad_theta() for the analogous θ-gradient sum).
  //
  // NOTE: LinearAlgebra::compute_dX(X) does NOT store an antisymmetric
  // pairwise-difference array: for a pair (p,q) with p<q it writes the same
  // value diff = X_p - X_q at BOTH column p*n+q and column q*n+p (see its
  // implementation) -- i.e. m_dX.col(a*n+b) always equals X_min(a,b) -
  // X_max(a,b), regardless of whether a<b or a>b. Reusing that cache with
  // the naive assumption dPhi_ij = m_dX.col(i*n+j) is therefore only
  // correct when i<j; for i>j it silently returns -dPhi_ij, which (since
  // DlnCovDx is odd) flips the sign of every term where the row index i is
  // the larger one. That bug used to corrupt every row of dL_dPhi except
  // row 0 (verified by an isolated finite-difference check against
  // dK_dPhi() with cached vs recomputed differences). Recomputing the
  // difference directly from Phi below sidesteps the cache entirely and
  // is guaranteed to have the correct sign.
  for (arma::uword i = 0; i < n; ++i) {
    for (arma::uword j = 0; j < n; ++j) {
      if (i == j)
        continue;
      double coeff = dL_dR(i, j);
      if (std::abs(coeff) < 1e-15)
        continue;

      double R_ij = m_R(i, j);
      arma::vec dPhi_ij = Phi.row(i).t() - Phi.row(j).t();
      arma::vec dlnCdx = _DlnCovDx(dPhi_ij, m_theta);
      // ∂R_ij/∂Φ_i_k = R_ij * dlnCdx_k (factor 2: see comment above)
      dL_dPhi.row(i) += 2.0 * coeff * R_ij * dlnCdx.t();
    }
  }
  return dL_dPhi;
}

arma::vec WarpKriging::warp_gradient() const {
  if ((m_Rinv.memptr() == nullptr) || (arma::size(m_Rinv) != arma::size(m_T)))
    m_Rinv = LinearAlgebra::inv_sympd(m_T);
  const arma::mat& Rinv = m_Rinv;  // lazily built above if not already cached
  arma::vec alpha = LinearAlgebra::solve_upper(m_T.t(), m_z);
  arma::mat dLL_dR = 0.5 * (alpha * alpha.t() / m_sigma2 - Rinv);

  arma::mat dLL_dPhi = dK_dPhi(m_X, dLL_dR);

  if (m_is_joint) {
    if (!m_joint_warp)
      return {};
    return m_joint_warp->backward(m_X_raw, dLL_dPhi);
  }

  arma::uword n_warp = total_warp_params();
  arma::vec grad(n_warp, arma::fill::zeros);

  arma::uword col = 0, idx = 0;
  for (arma::uword j = 0; j < m_warps.size(); ++j) {
    arma::uword dj = m_warps[j]->output_dim();
    arma::uword np = m_warps[j]->n_params();
    if (np > 0) {
      arma::mat dLL_dPhi_j = dLL_dPhi.cols(col, col + dj - 1);
      arma::vec gw = m_warps[j]->backward(m_X_raw.col(j), dLL_dPhi_j);
      grad.subvec(idx, idx + np - 1) = gw;
      idx += np;
    }
    col += dj;
  }
  return grad;
}

// -------------------------------------------------------------------------
//  Warp param packing  (θ is NOT here — optimised separately)
// -------------------------------------------------------------------------
arma::uword WarpKriging::total_warp_params() const {
  if (m_is_joint)
    return m_joint_warp ? m_joint_warp->n_params() : 0;
  arma::uword total = 0;
  for (const auto& w : m_warps)
    total += w->n_params();
  return total;
}

unsigned int WarpKriging::joint_bfgs_max_iter(arma::uword n_total) const {
  if (m_max_iter_bfgs > 0)
    return static_cast<unsigned int>(m_max_iter_bfgs);
  const arma::uword base = static_cast<arma::uword>(std::max(Optim::max_iteration, 1));
  if (total_warp_params() == 0 || m_is_joint)
    return static_cast<unsigned int>(base);
  return static_cast<unsigned int>(std::max(base, std::min(kJointIterPerParam * n_total, kJointMaxIterCap)));
}

arma::vec WarpKriging::random_warp_params(uint64_t seed) const {
  if (m_is_joint) {
    if (!m_joint_warp)
      return {};
    const auto& spec = m_warp_specs[0];
    WarpMLPJoint fresh(
        m_joint_warp->input_dim(), spec.hidden_dims, spec.d_out, WarpMLP::parse_act(spec.activation), seed);
    return fresh.get_params();
  }
  arma::vec wp = pack_warp_params();
  arma::uword idx = 0;
  for (arma::uword k = 0; k < m_warps.size(); ++k) {
    arma::uword np = m_warps[k]->n_params();
    if (np > 0 && m_warp_specs[k].type == WarpType::MLP) {
      const auto& spec = m_warp_specs[k];
      WarpMLP fresh(spec.hidden_dims, spec.d_out, WarpMLP::parse_act(spec.activation), seed + k);
      wp.subvec(idx, idx + np - 1) = fresh.get_params();
    }
    idx += np;
  }
  return wp;
}

arma::vec WarpKriging::pack_warp_params() const {
  if (m_is_joint)
    return m_joint_warp ? m_joint_warp->get_params() : arma::vec();
  arma::uword n_warp = total_warp_params();
  if (n_warp == 0)
    return {};
  arma::vec wp(n_warp);
  arma::uword idx = 0;
  for (const auto& w : m_warps) {
    arma::uword np = w->n_params();
    if (np > 0) {
      wp.subvec(idx, idx + np - 1) = w->get_params();
      idx += np;
    }
  }
  return wp;
}

void WarpKriging::unpack_warp_params(const arma::vec& wp) {
  if (m_is_joint) {
    if (m_joint_warp)
      m_joint_warp->set_params(wp);
    return;
  }
  arma::uword idx = 0;
  for (auto& w : m_warps) {
    arma::uword np = w->n_params();
    if (np > 0) {
      w->set_params(wp.subvec(idx, idx + np - 1));
      idx += np;
    }
  }
}

void WarpKriging::warp_param_bounds(arma::vec& lo, arma::vec& hi) const {
  arma::uword n_warp = total_warp_params();
  lo.set_size(n_warp);
  hi.set_size(n_warp);
  lo.fill(-arma::datum::inf);
  hi.fill(arma::datum::inf);
  if (m_is_joint)
    return;
  arma::uword idx = 0;
  for (const auto& w : m_warps) {
    arma::uword np = w->n_params();
    if (np > 0) {
      arma::vec l, h;
      w->param_bounds(l, h);
      lo.subvec(idx, idx + np - 1) = l;
      hi.subvec(idx, idx + np - 1) = h;
      idx += np;
    }
  }
}

// -------------------------------------------------------------------------
//  collapse_penalty()  —  guard against warps collapsing training inputs
//
//  The concentrated LL of a noise-free GP is unbounded when a warp maps
//  several training points with (near) equal outputs onto (near) the same
//  feature: R gets near-duplicate rows and −½ log|R| → +∞. A monotone 1D warp
//  can do this on a whole sub-range of its input (e.g. Kumaraswamy with a
//  large b squeezing [0.4, 1] onto 1), a degenerate optimum with collapsed
//  predictive variance (#406). For each guarded warp, with training inputs
//  sorted and de-duplicated, the gap k between x_(k) and x_(k+1) has the
//  relative mean slope
//      g_k = |Δφ_k| / Δx_k  /  (span φ / span x)
//  (1 everywhere for an affine warp, so a global rescaling — θ's job — is
//  not penalised) and costs  w · max(0, log(g_min / g_k))².
// -------------------------------------------------------------------------
std::pair<double, arma::vec> WarpKriging::collapse_penalty() const {
  arma::uword n_warp = total_warp_params();
  arma::vec grad(n_warp, arma::fill::zeros);
  double pen = 0.0;
  if (m_is_joint || n_warp == 0 || m_X.n_rows < 3)
    return {pen, grad};
  const double log_gmin = std::log(kCollapseMinRatio);
  const double tiny = 1e-300;

  arma::uword col = 0, idx = 0;
  for (arma::uword j = 0; j < m_warps.size(); ++j) {
    const arma::uword dj = m_warps[j]->output_dim();
    const arma::uword np = m_warps[j]->n_params();
    if (np > 0 && dj == 1 && m_warps[j]->collapse_guard()) {
      const arma::vec x = m_X_raw.col(j);
      const arma::vec phi = m_X.col(col);
      const double x_span = x.max() - x.min();
      const arma::uword i_max = phi.index_max(), i_min = phi.index_min();
      const double phi_span = std::max(phi(i_max) - phi(i_min), tiny);
      if (x_span > 0) {
        const arma::uvec ord = arma::sort_index(x);
        const double x_tol = 1e-12 * x_span;
        arma::vec dpen_dphi(x.n_elem, arma::fill::zeros);
        double pen_j = 0.0;
        double dpen_dlogspan = 0.0;  // ∂pen/∂log(span φ)
        // Walk consecutive distinct inputs (ties carry no gap).
        arma::uword prev = ord(0);
        for (arma::uword k = 1; k < ord.n_elem; ++k) {
          const arma::uword cur = ord(k);
          const double dx = x(cur) - x(prev);
          if (dx <= x_tol)
            continue;
          const double dphi = phi(cur) - phi(prev);
          const double adphi = std::max(std::abs(dphi), tiny);
          const double log_g = std::log(adphi) - std::log(dx) - std::log(phi_span) + std::log(x_span);
          const double r = log_gmin - log_g;
          if (r > 0) {
            pen_j += kCollapseWeight * r * r;
            // ∂pen/∂log g = −2 w r ;  log g = log|Δφ| − log span φ + const
            const double dpen_dlogg = -2.0 * kCollapseWeight * r;
            const double s = (dphi >= 0) ? 1.0 : -1.0;
            dpen_dphi(cur) += dpen_dlogg * s / adphi;
            dpen_dphi(prev) -= dpen_dlogg * s / adphi;
            dpen_dlogspan -= dpen_dlogg;
          }
          prev = cur;
        }
        if (pen_j > 0) {
          dpen_dphi(i_max) += dpen_dlogspan / phi_span;
          dpen_dphi(i_min) -= dpen_dlogspan / phi_span;
          pen += pen_j;
          grad.subvec(idx, idx + np - 1) = m_warps[j]->backward(x, arma::mat(dpen_dphi));
        }
      }
    }
    idx += np;
    col += dj;
  }
  return {pen, grad};
}

// -------------------------------------------------------------------------
//  calibrate_variance()  —  K-fold calibration of the predictive variance
//
//  A joint MLP has many more weights than there are observations: maximising
//  the likelihood, it learns features in which the training outputs are
//  almost exactly a smooth function (e.g. one feature ≈ y), so the GP on top
//  is very confident (long θ, tiny predictive variance) while the features
//  are off between the training points (#407: 95% intervals covering 1% of
//  the test points). The likelihood, or a leave-one-out at fixed features,
//  cannot see it: only refitting without the held-out points does. Each fold
//  is refitted from the starting point of the optimiser start that gave the
//  fit (not from the fit itself, which already knows the held-out points: a
//  warm start barely moves away from it), and σ² is scaled for
//  predict/simulate by the mean squared standardised held-out residual, if
//  above 1: the calibration only corrects overconfidence (that start was
//  picked on all the data, so the folds are slightly optimistic).
// -------------------------------------------------------------------------
void WarpKriging::calibrate_variance() {
  m_var_scale = 1.0;
  const arma::uword n = m_y.n_elem;
  const arma::uword K = m_calibration_folds;
  if (!m_is_joint || K < 2 || n < 2 * K || !m_noise.is_empty() || total_warp_params() == 0
      || m_start_theta.n_elem != m_theta.n_elem || m_start_warp_params.n_elem != total_warp_params())
    return;

  // Back to the raw scale: fit() normalises its inputs again.
  const arma::vec y_raw = m_y * m_scaleY + m_centerY;
  arma::mat X_raw = m_X_raw;
  X_raw.each_row() %= m_scaleX;
  X_raw.each_row() += m_centerX;

  // Same single-start optimiser and budget as the start that gave the fit.
  Parameters start;
  start.theta = m_start_theta;
  start.warp_params = m_start_warp_params;
  const std::string method = Optim::parse_method(m_optim, "BFGS").first;

  // The folds are independent: refitted concurrently (see run_task_pool), the
  // models built beforehand since their warp constructors reseed the RNG.
  std::vector<WarpKriging> folds;
  folds.reserve(K);
  for (arma::uword f = 0; f < K; ++f) {
    folds.emplace_back(warping_strings(), m_covType);
    WarpKriging& wf = folds.back();
    wf.m_max_iter_bfgs = m_max_iter_bfgs;
    wf.m_max_iter_adam = m_max_iter_adam;
    wf.m_adam_lr = m_adam_lr;
    wf.m_calibration_folds = 0;  // no nested calibration
  }
  std::vector<double> fold_z2(K, 0.0);
  std::vector<arma::uword> fold_nz(K, 0);
  std::mutex log_mutex;
  auto refit_fold = [&](int task_id) {
    const arma::uword f = static_cast<arma::uword>(task_id);
    // Interleaved folds: f, f + K, f + 2K, …
    arma::uvec test = arma::regspace<arma::uvec>(f, K, n - 1);
    std::vector<arma::uword> train_idx;
    train_idx.reserve(n - test.n_elem);
    for (arma::uword i = 0; i < n; ++i)
      if (i % K != f)
        train_idx.push_back(i);
    const arma::uvec train(train_idx);
    try {
      WarpKriging& wf = folds[f];
      wf.fit(y_raw.elem(train), X_raw.rows(train), m_regmodel, m_normalize, method, "LL", start, {}, "exact");
      auto [mean, stdev, _c, _md, _sd] = wf.predict(X_raw.rows(test), true, false, false);
      for (arma::uword i = 0; i < test.n_elem; ++i) {
        const double v = stdev(i) * stdev(i);
        const double e = y_raw(test(i)) - mean(i);
        if (v > 0 && std::isfinite(v) && std::isfinite(e)) {
          fold_z2[f] += e * e / v;
          ++fold_nz[f];
        }
      }
    } catch (const std::exception& e) {
      if (Optim::log_level > Optim::log_none) {
        std::lock_guard<std::mutex> lock(log_mutex);
        arma::cout << "Warning: WarpKriging variance calibration, fold " << (f + 1) << " failed: " << e.what()
                   << arma::endl;
      }
    }
  };
  run_task_pool(static_cast<int>(K), refit_fold, "calibration_folds");

  // Summed in fold order, whatever the order the folds finished in.
  double sum_z2 = 0.0;
  arma::uword n_z = 0;
  for (arma::uword f = 0; f < K; ++f) {
    sum_z2 += fold_z2[f];
    n_z += fold_nz[f];
  }
  if (2 * n_z < n)  // too few held-out residuals to calibrate on
    return;
  const double factor = sum_z2 / static_cast<double>(n_z);
  if (std::isfinite(factor) && factor > 1.0)
    m_var_scale = factor;
}

// -------------------------------------------------------------------------
//  clone_for_thread()  —  deep copy for parallel multistart
// -------------------------------------------------------------------------
WarpKriging WarpKriging::clone_for_thread() const {
  // Use the light constructor (warping strings + kernel, no data)
  WarpKriging c(warping_strings(), m_covType);

  // Copy immutable data
  c.m_y = m_y;
  c.m_X_raw = m_X_raw;
  c.m_F = m_F;
  c.m_normalize = m_normalize;
  c.m_centerX = m_centerX;
  c.m_scaleX = m_scaleX;
  c.m_centerY = m_centerY;
  c.m_scaleY = m_scaleY;
  c.m_is_continuous = m_is_continuous;
  c.m_regmodel = m_regmodel;
  c.m_feature_dim = m_feature_dim;
  c.m_is_joint = m_is_joint;
  c.m_max_iter_bfgs = m_max_iter_bfgs;
  c.m_max_iter_adam = m_max_iter_adam;
  c.m_adam_lr = m_adam_lr;
  c.m_calibration_folds = m_calibration_folds;
  c.m_var_scale = m_var_scale;
  c.m_chol_jitter = m_chol_jitter;  // the clones search with the same diagonal
  c.m_model_chol_jitter = m_model_chol_jitter;
  c.m_cache_jitter = m_cache_jitter;
  c.m_objective_ll = m_objective_ll;
  c.m_fitted = true;

  // Copy mutable GP state
  c.m_theta = m_theta;
  c.m_sigma2 = m_sigma2;
  c.m_noise = m_noise;
  c.m_beta = m_beta;
  c.m_z = m_z;
  c.m_M = m_M;
  c.m_circ = m_circ;
  c.m_R = m_R;
  c.m_T = m_T;
  c.m_Rinv = m_Rinv;
  c.m_logdet = m_logdet;
  c.m_X = m_X;
  c.m_dX = m_dX;

  // Deep-copy warp objects
  if (m_is_joint) {
    if (m_joint_warp)
      c.m_joint_warp = m_joint_warp->clone();
  } else {
    c.m_warps.clear();
    for (const auto& w : m_warps)
      c.m_warps.push_back(w->clone());
  }

  return c;
}

// -------------------------------------------------------------------------
//  Joint optimisation  (bi-level via AdamBFGS)
//
//  x_outer = warp params      (optimised by Adam)
//  x_inner = log(θ)           (optimised by L-BFGS-B)
//  σ̂² and β̂ are concentrated out at each evaluation.
// -------------------------------------------------------------------------
void WarpKriging::optimise_joint(const std::string& method) {
  arma::uword n_warp = total_warp_params();
  arma::uword d_theta = m_theta.n_elem;

  arma::vec wp0 = pack_warp_params();

  // Data-driven θ bounds from feature-space pairwise diffs (shared helper).
  arma::vec maxdPhi = arma::max(arma::abs(m_dX), 1);
  auto theta_bounds_pair = Optim::theta_bounds(maxdPhi, m_dX, m_y, m_y.n_elem);
  arma::vec theta_lower = theta_bounds_pair.first;
  arma::vec theta_upper = theta_bounds_pair.second;
  theta_lower = arma::clamp(theta_lower, 1e-10, arma::datum::inf);
  theta_upper = arma::max(theta_lower * 2.0, theta_upper);

  // Honor Optim::reparametrize flag via local helpers.
  auto to_gamma = [](const arma::vec& t) { return Optim::reparametrize ? Optim::reparam_to(t) : t; };
  auto from_gamma = [](const arma::vec& g) { return Optim::reparametrize ? Optim::reparam_from(g) : g; };
  auto grad_theta_to_gamma = [](const arma::vec& theta, const arma::vec& g_theta) {
    return Optim::reparametrize ? Optim::reparam_from_deriv(theta, g_theta) : g_theta;
  };

  arma::vec gamma_lower = to_gamma(theta_lower);
  arma::vec gamma_upper = to_gamma(theta_upper);

  // Parse multistart count from method (shared helper).
  // Box constraints of the warp params (±inf where unbounded).
  arma::vec wp_lower, wp_upper;
  warp_param_bounds(wp_lower, wp_upper);

  auto parsed = Optim::parse_method(method, "BFGS");
  const std::string base_method = parsed.first;
  const int multistart = parsed.second;

  // Helper lambda: run joint L-BFGS-B on a WarpKriging instance.
  // Returns the negative LL at the optimum (lower is better).
  auto run_joint_bfgs = [&](WarpKriging& wk) -> double {
    arma::uword nw = wk.total_warp_params();
    arma::uword dt = wk.m_theta.n_elem;
    arma::uword n_total = nw + dt;
    lbfgsb::Optimizer optimizer{static_cast<unsigned int>(n_total)};
    optimizer.iprint = -1;
    optimizer.max_iter = wk.joint_bfgs_max_iter(n_total);
    optimizer.pgtol = Optim::gradient_tolerance;
    optimizer.factr = Optim::objective_rel_tolerance / 1E-13;

    // Save the initial theta for retry contraction.
    arma::vec theta0 = wk.m_theta;
    // Starting warp params, projected onto their box constraints.
    arma::vec wp_init
        = (nw > 0) ? arma::vec(arma::min(arma::max(wk.pack_warp_params(), wp_lower), wp_upper)) : arma::vec();

    arma::vec x0(n_total);
    if (nw > 0)
      x0.head(nw) = wp_init;
    x0.tail(dt) = to_gamma(theta0);

    arma::vec lb(n_total), ub(n_total);
    arma::ivec btype(n_total);
    // Warp params: per-parameter box constraints (IWarp::param_bounds),
    // L-BFGS-B bound type 0/1/2/3 = none/lower/both/upper.
    for (arma::uword i = 0; i < nw; ++i) {
      const bool has_lo = std::isfinite(wp_lower(i)), has_hi = std::isfinite(wp_upper(i));
      lb(i) = has_lo ? wp_lower(i) : -1e20;
      ub(i) = has_hi ? wp_upper(i) : 1e20;
      btype(i) = has_lo ? (has_hi ? 2 : 1) : (has_hi ? 3 : 0);
    }
    lb.tail(dt) = gamma_lower;
    ub.tail(dt) = gamma_upper;
    btype.tail(dt).fill(2);

    // Best point evaluated so far: a long joint run can step into a region
    // where the correlation matrix cannot be factorised (the objective then
    // throws); fall back to this point instead of failing the whole fit.
    double eval_best_f = std::numeric_limits<double>::infinity();
    arma::vec eval_best_x;

    auto obj_fn = [&](const arma::vec& x, arma::vec& grad) -> double {
      if (nw > 0)
        wk.unpack_warp_params(x.head(nw));
      wk.m_theta = from_gamma(x.tail(dt));
      try {
        wk.refresh_cache();
      } catch (const std::exception&) {
        // Point the correlation matrix cannot be factorised at (or with
        // non-finite features), typically a too-long line-search step: report
        // it as much worse than the best point so far, so the line search
        // backtracks instead of the whole run stopping there (#406).
        if (!std::isfinite(eval_best_f))
          throw;
        grad.zeros();
        return eval_best_f + 1e3 * std::max(1.0, std::abs(eval_best_f));
      }

      auto [ll, g_theta] = wk.concentrated_ll_and_grad_theta();
      grad.tail(dt) = -grad_theta_to_gamma(wk.m_theta, g_theta);
      if (nw > 0) {
        // Penalised objective: LL − collapse penalty (#406).
        auto [pen, g_pen] = wk.collapse_penalty();
        ll -= pen;
        grad.head(nw) = -wk.warp_gradient() + g_pen;
      }
      if (std::isfinite(ll) && -ll < eval_best_f) {
        eval_best_f = -ll;
        eval_best_x = x;
      }
      return -ll;
    };

    // Retry loop: on abnormal termination / stuck at bound / no progress,
    // contract θ toward lower bound and restart (mirrors Kriging).
    int retry = 0;
    double best_f = std::numeric_limits<double>::infinity();
    arma::vec best_x = x0;
    arma::vec x = x0;
    while (retry <= Optim::max_restart) {
      const double prev_best_f = best_f;
      lbfgsb::OptimizeResult res{};
      try {
        res = optimizer.minimize(obj_fn, x, lb.memptr(), ub.memptr(), btype.memptr());
      } catch (const std::exception& e) {
        if (!std::isfinite(eval_best_f))
          throw;
        if (Optim::log_level > Optim::log_none)
          arma::cout << "Warning: WarpKriging joint BFGS stopped at the best evaluated point: " << e.what()
                     << arma::endl;
        if (eval_best_f < best_f) {
          best_f = eval_best_f;
          best_x = eval_best_x;
        }
        break;
      }
      if (res.f_opt < best_f) {
        best_f = res.f_opt;
        best_x = x;
      }
      // Iteration budget exhausted while still making progress: resume from
      // the point reached instead of contracting θ back toward its lower
      // bound (which would discard the warp params learnt so far).
      if ((retry < Optim::max_restart) && (nw > 0) && (res.task.rfind("STOP: TOTAL NO. of ITERATIONS", 0) == 0)
          && (!std::isfinite(prev_best_f)
              || (prev_best_f - res.f_opt) > Optim::objective_rel_tolerance * std::max(1.0, std::abs(res.f_opt)))) {
        x = best_x;
        retry++;
        continue;
      }
      arma::vec theta_cur = from_gamma(x.tail(dt));
      double sol_to_lb = arma::min(arma::abs(theta_cur - theta_lower));
      // "No improvement" within the optimizer's own tolerance (as Kriging):
      // a difference at the rounding level must not decide a restart.
      const double f_tol = optimizer.factr * arma::datum::eps * std::max(1.0, std::abs(best_f));
      const bool worse = res.f_opt > best_f + f_tol;
      if ((retry < Optim::max_restart)
          && ((res.task.rfind("ABNORMAL_TERMINATION_IN_LNSRCH", 0) == 0) || (res.num_iters <= 2)
              || (sol_to_lb < arma::datum::eps) || worse)) {
        // Contract theta starting point toward lower bound.
        arma::vec theta_restart = (theta0 + theta_lower) / std::pow(2.0, retry + 1);
        x.tail(dt) = to_gamma(theta_restart);
        if (nw > 0)
          x.head(nw) = wp_init;
        retry++;
      } else {
        break;
      }
    }

    if (nw > 0)
      wk.unpack_warp_params(best_x.head(nw));
    wk.m_theta = from_gamma(best_x.tail(dt));
    wk.refresh_cache();
    return best_f;
  };

  // Helper lambda: run Adam+BFGS bi-level on a WarpKriging instance.
  auto run_adam_bfgs = [&](WarpKriging& wk) -> double {
    arma::uword nw = wk.total_warp_params();
    arma::uword dt = wk.m_theta.n_elem;
    AdamBFGS opt(nw, dt);
    opt.max_iter_adam = wk.m_max_iter_adam;
    opt.adam_lr = wk.m_adam_lr;
    opt.patience = kAdamPatience;
    opt.rel_tol = kAdamRelTol;
    opt.lr_decays = kAdamLrDecays;
    opt.max_iter_bfgs = (wk.m_max_iter_bfgs > 0) ? wk.m_max_iter_bfgs : kDefaultInnerMaxIterBFGS;
    opt.bfgs_pgtol = Optim::gradient_tolerance;
    opt.bfgs_factr = Optim::objective_rel_tolerance / 1E-13;
    opt.maximize = true;
    opt.outer_lower = wp_lower;
    opt.outer_upper = wp_upper;
    // Same restart as the joint BFGS (and Kriging): θ contracted toward its
    // lower bound, kept inside the bounds (L-BFGS-B does not project its
    // starting point)
    opt.max_restart = Optim::max_restart;
    opt.inner_restart = [&](const arma::vec& gamma0, int retry) -> arma::vec {
      const arma::vec theta_restart
          = arma::max(arma::min(arma::vec((from_gamma(gamma0) + theta_lower) / std::pow(2.0, retry + 1)), theta_upper),
                      theta_lower);
      return to_gamma(theta_restart);
    };

    arma::vec current_wp = arma::min(arma::max(wk.pack_warp_params(), wp_lower), wp_upper);
    if (nw > 0 && arma::any(current_wp != wk.pack_warp_params())) {
      // Start projected onto the bounds: the objective's first call only
      // refreshes θ, so the features must match the projected params already.
      wk.unpack_warp_params(current_wp);
      wk.refresh_cache();
    }
    arma::vec current_gamma = to_gamma(wk.m_theta);

    auto obj_fn = [&wk, &current_wp, &from_gamma, &grad_theta_to_gamma](const arma::vec& x_outer,
                                                                        const arma::vec& x_inner,
                                                                        arma::vec* grad_outer,
                                                                        arma::vec* grad_inner) -> double {
      bool warp_changed = false;
      if (x_outer.n_elem > 0) {
        if (current_wp.n_elem != x_outer.n_elem || arma::any(current_wp != x_outer)) {
          wk.unpack_warp_params(x_outer);
          current_wp = x_outer;
          warp_changed = true;
        }
      }

      wk.m_theta = from_gamma(x_inner);

      if (warp_changed)
        wk.refresh_cache();
      else
        wk.refresh_cache_theta_only();

      double ll = wk.concentrated_ll();
      // Penalised objective: LL − collapse penalty (#406). The penalty only
      // depends on the warp params, so the inner θ gradient is unchanged.
      auto [pen, g_pen] = wk.collapse_penalty();
      ll -= pen;

      if (grad_inner) {
        auto [ll2, g_theta] = wk.concentrated_ll_and_grad_theta();
        *grad_inner = grad_theta_to_gamma(wk.m_theta, g_theta);
        (void)ll2;
      }
      if (grad_outer && x_outer.n_elem > 0) {
        *grad_outer = wk.warp_gradient() - g_pen;
      }
      return ll;
    };

    auto result = opt.optimize(current_wp, current_gamma, gamma_lower, gamma_upper, obj_fn);

    if (nw > 0)
      wk.unpack_warp_params(result.x_outer);
    wk.m_theta = from_gamma(result.x_inner);
    wk.refresh_cache();
    return -(wk.concentrated_ll() - wk.collapse_penalty().first);  // neg penalised LL, as the joint BFGS
  };

  // Structure to hold optimization results from each thread
  struct OptimizationResult {
    double neg_ll = std::numeric_limits<double>::infinity();
    arma::vec theta;
    arma::vec warp_params;
    double sigma2 = 0;
    arma::vec beta;
    arma::vec z;
    arma::mat M;
    arma::mat circ;
    arma::mat C;
    arma::mat Rinv;
    arma::mat R;
    arma::mat Phi;
    arma::mat dPhi;
    double logdet = 0;
    bool success = false;
    arma::vec start_theta;  // starting point of this start (see calibrate_variance)
    arma::vec start_warp_params;
  };

  // Helper: extract result from a WarpKriging instance after optimization
  auto extract_result = [](WarpKriging& wk, double neg_ll) -> OptimizationResult {
    OptimizationResult r;
    r.neg_ll = neg_ll;
    r.theta = wk.m_theta;
    r.warp_params = wk.pack_warp_params();
    r.sigma2 = wk.m_sigma2;
    r.beta = wk.m_beta;
    r.z = wk.m_z;
    r.M = wk.m_M;
    r.circ = wk.m_circ;
    r.C = wk.m_T;
    r.Rinv = wk.m_Rinv;
    r.R = wk.m_R;
    r.Phi = wk.m_X;
    r.dPhi = wk.m_dX;
    r.logdet = wk.m_logdet;
    r.success = true;
    return r;
  };

  // Helper: restore best result to `this`
  auto restore_best = [&](const OptimizationResult& r) {
    if (n_warp > 0)
      unpack_warp_params(r.warp_params);
    m_theta = r.theta;
    m_sigma2 = r.sigma2;
    m_beta = r.beta;
    m_z = r.z;
    m_M = r.M;
    m_circ = r.circ;
    m_T = r.C;
    m_Rinv = r.Rinv;
    m_R = r.R;
    m_X = r.Phi;
    m_dX = r.dPhi;
    m_logdet = r.logdet;
  };

  // Helper: run parallel multistart with a given optimizer function
  auto run_parallel_multistart = [&](auto& optimizer_fn) {
    // Random starting points for θ, from the same generator as Kriging
    // (Random, reset to its seed at each fit: Random::reset_seed applies).
    // Per-variable warps: log-uniform in [theta_lower, theta_upper] as Kriging
    // (θ is a scale parameter, theta_lower > 0). Joint MLP: uniform, θ lives on
    // the scale of the learnt features and long-range starts let the weights
    // adapt; the log-uniform draw lost 124 LL units (median) on a projected
    // Branin (docs/dev/CholJitter.md, starting points).
    Random::init();
    arma::mat theta0_rand = Random::randu_mat(multistart, d_theta);
    for (arma::uword j = 0; j < d_theta; ++j) {
      const double lo = theta_lower.at(j), up = theta_upper.at(j);
      if (m_is_joint)
        theta0_rand.col(j) = lo + theta0_rand.col(j) * (up - lo);
      else
        theta0_rand.col(j) = arma::exp(std::log(lo) + theta0_rand.col(j) * (std::log(up) - std::log(lo)));
    }
    // First start uses the current theta
    theta0_rand.row(0) = m_theta.t();

    // First start uses the current warp params; the others re-draw the MLP
    // weights (if any), so that restarts also explore the network's
    // parameter space instead of only θ. Drawn here, before the workers
    // start, since the warp constructors reseed the global RNG.
    arma::vec wp_init = pack_warp_params();
    std::vector<arma::vec> wp_starts(multistart, wp_init);
    if (n_warp > 0)
      for (int i = 1; i < multistart; ++i)
        wp_starts[i] = random_warp_params(static_cast<uint64_t>(42 + 1000 * i));

    std::vector<OptimizationResult> results(multistart);
    std::mutex results_mutex;

    // Create per-thread WarpKriging clones
    std::vector<WarpKriging> clones;
    clones.reserve(multistart);
    for (int i = 0; i < multistart; ++i)
      clones.push_back(clone_for_thread());

    // Worker function
    auto worker = [&](int task_id) {
      try {
        WarpKriging& wk = clones[task_id];
        // Reset to initial warp params and set starting theta
        if (n_warp > 0)
          wk.unpack_warp_params(wp_starts[task_id]);
        wk.m_theta = theta0_rand.row(task_id).t();
        wk.refresh_cache();

        double neg_ll = optimizer_fn(wk);

        OptimizationResult r = extract_result(wk, neg_ll);
        r.start_theta = theta0_rand.row(task_id).t();
        r.start_warp_params = wp_starts[task_id];
        std::lock_guard<std::mutex> lock(results_mutex);
        results[task_id] = std::move(r);
      } catch (const std::exception& e) {
        if (Optim::log_level > Optim::log_none) {
          std::lock_guard<std::mutex> lock(results_mutex);
          arma::cout << "Warning: WarpKriging multistart " << (task_id + 1) << " failed: " << e.what() << arma::endl;
        }
      }
    };

    // Starts run concurrently in a thread pool (see run_task_pool).
    run_task_pool(multistart, worker, "multistart");

    // Find best result
    int best_idx = -1;
    double best_neg_ll = std::numeric_limits<double>::infinity();
    for (int i = 0; i < multistart; ++i) {
      if (results[i].success && results[i].neg_ll < best_neg_ll) {
        best_neg_ll = results[i].neg_ll;
        best_idx = i;
      }
    }
    if (best_idx >= 0) {
      restore_best(results[best_idx]);
      m_start_theta = results[best_idx].start_theta;
      m_start_warp_params = results[best_idx].start_warp_params;
    }
  };

  if (base_method == "none") {
    // --- Skip optimisation: keep current θ and warp params as-is ---
    return;
  }

  // Starting point of the (single) start; a multistart overwrites it with
  // the starting point of its best start.
  m_start_theta = m_theta;
  m_start_warp_params = pack_warp_params();

  if (base_method == "BFGS" || n_warp == 0) {
    // --- Joint L-BFGS-B (with multistart) ---
    if (multistart <= 1) {
      run_joint_bfgs(*this);
    } else {
      run_parallel_multistart(run_joint_bfgs);
    }

  } else {
    // --- Bi-level Adam+BFGS (with multistart) ---
    if (multistart <= 1) {
      run_adam_bfgs(*this);
    } else {
      run_parallel_multistart(run_adam_bfgs);
    }

    // Joint BFGS polish if method requests it
    if (base_method == "BFGS+Adam+BFGS" && n_warp > 0) {
      run_joint_bfgs(*this);
    }
  }
}

// -------------------------------------------------------------------------
//  fit()  —  map-based overload (tuning knobs via string map)
// -------------------------------------------------------------------------
void WarpKriging::fit(const arma::vec& y,
                      const arma::mat& X,
                      const Trend::RegressionModel& regmodel,
                      bool normalize,
                      const std::string& optim,
                      const std::string& objective,
                      const std::map<std::string, std::string>& parameters,
                      const std::string& method) {
  libKriging::detail::BudgetScope budget_scope;  // BLAS at the thread budget for the call
  fit(y, X, regmodel, normalize, optim, objective, Parameters{}, parameters, method);
}

// -------------------------------------------------------------------------
//  fit()  —  typed-parameters overload (θ / warp-params / noise seeds),
//            optionally combined with string optimiser knobs via `tuning`.
// -------------------------------------------------------------------------
void WarpKriging::fit(const arma::vec& y,
                      const arma::mat& X,
                      const Trend::RegressionModel& regmodel,
                      bool normalize,
                      const std::string& optim,
                      const std::string& objective,
                      const Parameters& parameters,
                      const std::map<std::string, std::string>& tuning,
                      const std::string& method) {
  libKriging::detail::BudgetScope budget_scope;  // BLAS at the thread budget for the call
  if (Kriging::fit_objective_method(objective, method).second != "exact")
    throw std::invalid_argument("WarpKriging::fit: method \"" + method + "\" is not implemented (only \"exact\" is)");
  for (const auto& [key, val] : tuning) {
    if (key == "adam_lr")
      m_adam_lr = std::stod(val);
    else if (key == "max_iter_adam")
      m_max_iter_adam = std::stoul(val);
    else if (key == "max_iter_bfgs")
      m_max_iter_bfgs = std::stoul(val);
    else if (key == "calibration_folds")
      m_calibration_folds = std::stoul(val);
  }

  if (y.n_elem != X.n_rows)
    throw std::invalid_argument("fit: y/X size mismatch");

  if (!m_is_joint && X.n_cols != m_warp_specs.size())
    throw std::invalid_argument("fit: X has " + std::to_string(X.n_cols) + " columns but "
                                + std::to_string(m_warp_specs.size()) + " warp specs were given");

  if (!m_is_joint)
    validate_discrete_columns(X, "fit");

  // the last simulate(..., will_update=true) was built for the previous model
  clear_simulate_state();

  m_y = y;
  m_X_raw = X;
  m_regmodel = regmodel;
  m_normalize = normalize;
  {
    const double kernel_jitter = LinearAlgebra::kernel_chol_jitter(m_covType);
    m_chol_jitter = kernel_jitter > 0.0 ? kernel_jitter : LinearAlgebra::num_nugget;
    m_cache_jitter = m_chol_jitter;  // search; committed after optimise_joint
    m_objective_ll = arma::datum::nan;
  }

  // Per-observation noise (optional). When set, populate_Model carries
  // noise_i/σ² on the diagonal and refresh_cache_theta_only runs an inner
  // 1D MLE for σ² (no closed form with noise).
  if (parameters.noise.has_value()) {
    const arma::vec& nv = *parameters.noise;
    if (nv.n_elem != y.n_elem)
      throw std::invalid_argument("fit: parameters.noise has " + std::to_string(nv.n_elem) + " elements but y has "
                                  + std::to_string(y.n_elem));
    if (arma::any(nv < 0))
      throw std::invalid_argument("fit: parameters.noise contains negative values");
    m_noise = nv;
  } else {
    m_noise.reset();
  }

  // For joint MLP, instantiate the network now that d_in is known.
  if (m_is_joint)
    ensure_joint_warp(X.n_cols);

  normalise_data();
  calibrate_warps();  // tie domain-bounded warps (Knots) to the input range

  // Scale noise to normalized-y space (matches NoiseKriging::fit).
  if (!m_noise.is_empty() && m_normalize) {
    m_noise /= (m_scaleY * m_scaleY);
  }

  // θ seed: use caller-supplied value or default to 1.
  if (parameters.theta.has_value()) {
    const arma::vec& t0 = *parameters.theta;
    if (t0.n_elem != m_feature_dim)
      throw std::invalid_argument("fit: parameters.theta has " + std::to_string(t0.n_elem)
                                  + " elements but feature_dim is " + std::to_string(m_feature_dim));
    m_theta = t0;
  } else {
    m_theta = arma::ones<arma::vec>(m_feature_dim);
  }

  // Warp-params seed: unpack into warp objects (if provided and non-empty).
  if (parameters.warp_params.has_value() && parameters.warp_params->n_elem > 0) {
    const arma::vec& wp = *parameters.warp_params;
    arma::uword nw = total_warp_params();
    if (wp.n_elem != nw)
      throw std::invalid_argument("fit: parameters.warp_params has " + std::to_string(wp.n_elem)
                                  + " elements but total_warp_params is " + std::to_string(nw));
    unpack_warp_params(wp);
  }

  // Set base "estim" flags so KrigingImpl helpers (e.g. update_no_refit_impl)
  // commit β̂/σ² consistently with Warp's semantics:
  //   - β is estimated via GLS unless regmodel=None;
  //   - σ² is estimated by closed-form MLE only when no per-point noise
  //     is supplied (with noise it's set by the inner 1D MLE in
  //     refresh_cache_theta_only and must not be overwritten on extension).
  m_est_beta = (m_regmodel != Trend::RegressionModel::None);
  m_est_theta = true;
  m_est_sigma2 = m_noise.is_empty();

  // σ̂² is concentrated — computed in refresh_cache
  refresh_cache();
  optimise_joint(optim);
  commit_model_chol_jitter();
  m_optim = optim;
  m_var_scale = 1.0;
  if (Optim::parse_method(optim, "BFGS").first != "none")
    calibrate_variance();
  m_objective = "LL";
  m_fitted = true;
}

// -------------------------------------------------------------------------
//  predict()
// -------------------------------------------------------------------------
std::tuple<arma::vec, arma::vec, arma::mat, arma::mat, arma::mat> WarpKriging::predict(const arma::mat& X_n,
                                                                                       bool return_stdev,
                                                                                       bool return_cov,
                                                                                       bool return_deriv) const {
  libKriging::detail::BudgetScope budget_scope;  // BLAS at the thread budget for the call
  if (!m_fitted)
    throw std::runtime_error("predict: model not fitted");
  validate_discrete_columns(X_n, "predict");

  // phi_fn: apply warping to normalized inputs (n × d_input) → (n × d_phi)
  // Base normalization (centerX/scaleX with 0/1 for non-continuous) is equivalent to
  // Warp's per-dim continuous-only normalization — see W-2 key insight in refactor_table.md.
  auto phi_fn = [this](const arma::mat& Xnorm) -> arma::mat { return apply_warping(Xnorm); };

  // jac_fn: Jacobian dΦ/dx at a normalized input column (d_input,) → (d_phi × d_input)
  auto jac_fn = [this](const arma::vec& x_norm_col) -> arma::mat {
    if (m_is_joint && m_joint_warp) {
      // Analytical backpropagation through the joint MLP
      return m_joint_warp->jacobian_input(x_norm_col.t());
    }
    // Analytical per-dimension warp derivatives (block-diagonal Jacobian)
    const arma::uword d_in = x_norm_col.n_elem;
    arma::mat J(m_feature_dim, d_in, arma::fill::zeros);
    arma::uword row = 0;
    for (arma::uword k = 0; k < d_in; k++) {
      arma::vec dw = m_warps[k]->deriv_input(x_norm_col(k));
      arma::uword d_out_k = m_warps[k]->output_dim();
      J.submat(row, k, row + d_out_k - 1, k) = dw;
      row += d_out_k;
    }
    return J;
  };

  return predict_impl(X_n,
                      return_stdev,
                      return_cov,
                      return_deriv,
                      /*R_on_factor=*/1.0,
                      /*R_nn_factor=*/1.0,
                      /*R_nn_diag=*/{},
                      /*var_scale=*/m_sigma2 * m_var_scale,
                      phi_fn,
                      jac_fn);
}

// -------------------------------------------------------------------------
//  simulate()  — consistent with Kriging::simulate
//
//  Works at correlation scale (R, not K = σ²R), applies σ̂ scaling after
//  Cholesky, then denormalizes.  This is more numerically robust than
//  doing Cholesky on the full-scale covariance matrix.
// -------------------------------------------------------------------------
arma::mat WarpKriging::simulate(int nsim,
                                int seed,
                                const arma::mat& X_n,
                                const bool will_update,
                                const arma::vec& with_noise) {
  libKriging::detail::BudgetScope budget_scope;  // BLAS at the thread budget for the call
  if (!m_fitted)
    throw std::runtime_error("simulate: model not fitted");

  const arma::uword n_n = X_n.n_rows;

  if (X_n.n_cols != m_X_raw.n_cols)
    throw std::runtime_error("Simulate locations have wrong dimension: " + std::to_string(X_n.n_cols) + " instead of "
                             + std::to_string(m_X_raw.n_cols));

  if (with_noise.n_elem > 1 && with_noise.n_elem != n_n)
    throw std::runtime_error("Noise vector should have same length as X_n: " + std::to_string(with_noise.n_elem)
                             + " instead of " + std::to_string(n_n));
  check_noise_variances(with_noise, "simulate");

  if (!m_is_joint)
    validate_discrete_columns(X_n, "simulate");

  // Route through KrigingImpl::simulate_impl with the warping as a feature-map callback.
  // The base normalises X_n, applies phi (warp), builds R_nn/R_on in phi-space, draws samples.
  auto phi_fn = [this](const arma::mat& Xn) { return apply_warping(Xn); };
  arma::mat y_n = simulate_impl(nsim,
                                seed,
                                X_n,
                                will_update,
                                /*R_on_factor=*/1.0,
                                /*R_on_coincident_to_one=*/false,
                                /*R_nn_factor=*/1.0,
                                /*R_nn_diag=*/{},
                                /*Sigma_divisor=*/1.0 / m_var_scale,
                                phi_fn);

  if (will_update) {
    lastsimup_noise_u.clear();  // invalidate noise cache so update_simulate recomputes
    lastsim_with_noise = with_noise;
  }

  // Optional observation noise: with_noise holds noise VARIANCES in raw
  // y-space (like fit's / update's noise), N(0, with_noise) added after
  // denormalisation (as Kriging::simulate).
  add_noise_draws(y_n, with_noise, "simulate");
  return y_n;
}

// -------------------------------------------------------------------------
//  update_simulate()  — FOXY algorithm, ported from Kriging::update_simulate
// -------------------------------------------------------------------------
LIBKRIGING_EXPORT arma::mat WarpKriging::update_simulate(const arma::vec& y_u,
                                                         const arma::mat& X_u,
                                                         const arma::vec& noise_u) {
  libKriging::detail::BudgetScope budget_scope;  // BLAS at the thread budget for the call
  if (y_u.n_elem != X_u.n_rows)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (" + std::to_string(X_u.n_rows) + "x"
                             + std::to_string(X_u.n_cols) + "), y: (" + std::to_string(y_u.n_elem) + ")");

  if (X_u.n_cols != m_X_raw.n_cols)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (...x" + std::to_string(m_X_raw.n_cols)
                             + "), new X: (...x" + std::to_string(X_u.n_cols) + ")");

  const bool has_model_noise = !m_noise.is_empty();
  // the noise of the new observations is needed (checked before any state change)
  if (has_model_noise && noise_u.is_empty())
    throw std::invalid_argument("heterogeneous noise model: use update_simulate(y_u, X_u, noise_u)");

  if (lastsim_y_n.is_empty() || lastsim_y_n.n_rows == 0)
    throw std::runtime_error("No previous simulation data available");

  if (has_model_noise) {
    if (noise_u.n_elem != X_u.n_rows)
      throw std::runtime_error("update_simulate: model fit with noise requires noise_u of length "
                               + std::to_string(X_u.n_rows) + " (got " + std::to_string(noise_u.n_elem) + ")");
    if (arma::any(noise_u < 0))
      throw std::runtime_error("update_simulate: noise_u contains negative values");
  } else if (!noise_u.is_empty()) {
    throw std::runtime_error("update_simulate: model fit without noise but noise_u was provided");
  }

  arma::vec noise_u_norm;
  if (has_model_noise)
    noise_u_norm = noise_u / (m_scaleY * m_scaleY);

  const arma::vec diag_uu = has_model_noise ? arma::vec(1.0 + noise_u_norm / m_sigma2) : arma::vec{};

  // Noise must also match for cache reuse; X_u match is checked by base internally.
  const bool noise_ok = (lastsimup_noise_u.n_elem == noise_u_norm.n_elem)
                        && (noise_u_norm.is_empty()
                            || arma::approx_equal(lastsimup_noise_u, noise_u_norm, "absdiff", arma::datum::eps));

  auto phi_fn = [this](const arma::mat& Xn) { return apply_warping(Xn); };
  arma::mat y_n = update_simulate_impl(y_u,
                                       X_u,
                                       noise_ok,
                                       /*R_uu_factor=*/1.0,
                                       diag_uu,
                                       /*R_uo_factor=*/1.0,
                                       /*R_un_factor=*/1.0,
                                       /*R_un_coincident_to_one=*/false,
                                       /*Sigma_divisor=*/1.0 / m_var_scale,
                                       phi_fn);

  lastsimup_noise_u = noise_u_norm;

  // re-add the with_noise (variances) of the last simulate(..., will_update=true)
  if (lastsim_with_noise.n_elem == 1 || lastsim_with_noise.n_elem == y_n.n_rows)
    add_noise_draws(y_n, lastsim_with_noise, "update_simulate");
  return y_n;
}

void WarpKriging::clear_simulate_state() {
  KrigingImpl::clear_simulate_state();
  lastsim_with_noise.reset();
  lastsimup_noise_u.reset();
}

// -------------------------------------------------------------------------
//  update()
// -------------------------------------------------------------------------
void WarpKriging::update(const arma::vec& y_u, const arma::mat& X_u, const bool refit, const arma::vec& noise_u) {
  libKriging::detail::BudgetScope budget_scope;  // BLAS at the thread budget for the call
  if (!m_fitted)
    throw std::runtime_error("update: model not fitted");
  if (y_u.n_elem != X_u.n_rows)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (" + std::to_string(X_u.n_rows) + "x"
                             + std::to_string(X_u.n_cols) + "), y: (" + std::to_string(y_u.n_elem) + ")");
  if (X_u.n_cols != m_X_raw.n_cols)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (...x" + std::to_string(m_X_raw.n_cols)
                             + "), new X: (...x" + std::to_string(X_u.n_cols) + ")");

  validate_discrete_columns(X_u, "update");

  // Noise consistency check
  const bool has_model_noise = !m_noise.is_empty();
  if (has_model_noise && noise_u.is_empty())
    throw std::invalid_argument("heterogeneous noise model: use update(y_u, X_u, refit, noise_u)");
  if (has_model_noise) {
    if (noise_u.n_elem != y_u.n_elem)
      throw std::runtime_error("update: model fit with noise requires noise_u of length " + std::to_string(y_u.n_elem)
                               + " (got " + std::to_string(noise_u.n_elem) + ")");
    if (arma::any(noise_u < 0))
      throw std::runtime_error("update: noise_u contains negative values");
  } else if (!noise_u.is_empty()) {
    throw std::runtime_error("update: model fit without noise but noise_u was provided");
  }

  // input validated: the last simulate(..., will_update=true) was built for
  // the model before this update
  clear_simulate_state();

  if (refit) {
    // De-normalise, append, re-normalise, re-optimise
    arma::vec y_all = arma::join_vert(m_y * m_scaleY + m_centerY, y_u);
    arma::mat X_all = m_X_raw;
    for (arma::uword j = 0; j < X_all.n_cols; ++j) {
      bool is_cont = m_is_joint || (j < m_is_continuous.size() && m_is_continuous[j]);
      if (is_cont)
        X_all.col(j) = X_all.col(j) * m_scaleX(j) + m_centerX(j);
    }
    X_all = arma::join_vert(X_all, X_u);

    // De-normalise noise, append, it will be re-normalised below after stddev recompute.
    arma::vec noise_all;
    if (has_model_noise)
      noise_all = arma::join_vert(m_noise * m_scaleY * m_scaleY, noise_u);

    m_y = y_all;
    m_X_raw = X_all;
    m_noise = noise_all;  // empty if no model noise
    normalise_data();
    calibrate_warps();  // re-tie Knots to the enlarged input range before re-fit
    if (!m_noise.is_empty() && m_normalize)
      m_noise /= (m_scaleY * m_scaleY);
    m_cache_jitter = m_chol_jitter;  // search; committed after optimise_joint
    refresh_cache();

    // The refit is warm-started from the current fit: keep the starting point
    // of the original fit for the variance calibration.
    const arma::vec start_theta = m_start_theta, start_warp_params = m_start_warp_params;
    if (Optim::parse_method(m_optim, "BFGS").first == "BFGS") {
      // Joint L-BFGS-B warm-started from the current fit (single start).
      optimise_joint("BFGS");
    } else {
      arma::uword saved_adam = m_max_iter_adam;
      m_max_iter_adam /= 5;
      optimise_joint("BFGS+Adam");
      m_max_iter_adam = saved_adam;
    }
    commit_model_chol_jitter();
    m_start_theta = start_theta;
    m_start_warp_params = start_warp_params;
    calibrate_variance();
  } else {
    // Fast incremental update: keep all hyperparameters fixed, use Cholesky
    // update via KrigingImpl::update_no_refit_impl. Warp's per-dim
    // normalization (continuous-only) is equivalent to base's rowvec
    // normalization here because normalise_data leaves m_centerX(j)=0,
    // m_scaleX(j)=1 for non-continuous dims, so the rowvec arithmetic is a
    // no-op on those columns.
    //
    // The extend_class_data hook is responsible for:
    //   * extending m_X_raw with the same per-dim-normalized values base
    //     just appended to m_X (kept in sync so apply_warping works);
    //   * extending m_noise (Warp-specific channel);
    //   * overwriting m_X with apply_warping(m_X_raw) — base appended raw
    //     normalized X_u to its m_X, which is the Φ-space slot for Warp,
    //     so we must re-warp before base's compute_dX runs.
    KrigingImpl::update_no_refit_impl(
        y_u,
        X_u,
        /*extend_class_data=*/
        [&]() {
          // Mirror base's normalization to extend m_X_raw (base also did this
          // arithmetic on Xn_u before appending to m_X).
          arma::mat Xn_u = X_u;
          Xn_u.each_row() -= m_centerX;
          Xn_u.each_row() /= m_scaleX;
          m_X_raw = arma::join_cols(m_X_raw, Xn_u);

          if (has_model_noise) {
            arma::vec noise_u_norm = noise_u / (m_scaleY * m_scaleY);
            m_noise = arma::join_cols(m_noise, noise_u_norm);
          }

          // Replace base's (corrupted) Φ slot with the actual warped design.
          m_X = apply_warping(m_X_raw);
        },
        /*build_model=*/
        [&]() { return make_Model(m_theta, m_sigma2); });

    m_logdet = 2.0 * arma::sum(arma::log(m_T.diag()));
    // no search here: logLikelihood() is the committed model's own
    m_objective_ll = arma::datum::nan;
  }
}

//  summary()
// -------------------------------------------------------------------------
std::string WarpKriging::summary() const {
  std::ostringstream oss;
  oss << "* WarpKriging\n";
  if (summary_top(oss, &m_X_raw)) {
    if (m_feature_dim != m_X_raw.n_cols)
      oss << "  * features: " << m_feature_dim << "\n";
    oss << "  * warpings:\n";
    if (m_is_joint) {
      oss << "      joint: \"" << m_warp_specs[0].to_string() << "\"";
      if (m_joint_warp)
        oss << "  →  " << m_joint_warp->describe();
      oss << "\n";
    } else {
      for (arma::uword j = 0; j < m_warps.size(); ++j) {
        oss << "      x" << j << ": \"" << m_warp_specs[j].to_string() << "\"  →  " << m_warps[j]->describe() << "\n";
        if (!m_warp_specs[j].level_names.empty()) {
          oss << "             levels: {";
          for (arma::uword k = 0; k < m_warp_specs[j].level_names.size(); ++k) {
            if (k > 0)
              oss << ", ";
            oss << k << "=\"" << m_warp_specs[j].level_names[k] << "\"";
          }
          oss << "}\n";
        }
      }
    }
    if (m_fitted)
      oss << "  * total warp params: " << total_warp_params() << "\n";
    if (m_fitted && m_var_scale != 1.0)
      oss << "  * predictive variance: sigma2 x " << m_var_scale << " (" << m_calibration_folds
          << "-fold calibration)\n";
    summary_bottom(oss);
  }
  return oss.str();
}

// -------------------------------------------------------------------------
//  save / load helpers (shared with MLPKriging facade)
// -------------------------------------------------------------------------
void WarpKriging::dump_to_json(nlohmann::json& j) const {
  // Warp-specific fields not in base
  std::vector<std::string> warp_strs;
  for (const auto& s : m_warp_specs)
    warp_strs.push_back(s.to_string());
  j["warping"] = warp_strs;
  j["X_raw"] = to_json(m_X_raw);  // base writes m_X (=Phi) as "X"
  j["is_continuous"] = m_is_continuous;
  j["feature_dim"] = m_feature_dim;
  j["max_iter_bfgs"] = m_max_iter_bfgs;
  j["max_iter_adam"] = m_max_iter_adam;
  j["adam_lr"] = m_adam_lr;
  j["calibration_folds"] = m_calibration_folds;
  j["var_scale"] = m_var_scale;
  j["start_theta"] = to_json(m_start_theta);
  j["start_warp_params"] = to_json(m_start_warp_params);
  j["fitted"] = m_fitted;
  j["logdet"] = m_logdet;
  j["warp_params"] = to_json(pack_warp_params());
  // Shared base fields: covType, X (=Phi), X_raw omitted (we write it above),
  // centerX/Y, scaleX/Y, normalize, regmodel, optim, objective,
  // dX, maxdX, F, T, R, M, star, circ, z, Rinv,
  // beta/est_beta, theta/est_theta, sigma2/est_sigma2, noise
  dump_common_to_json(j);
  j["chol_jitter"] = m_chol_jitter;
  j["model_chol_jitter"] = m_model_chol_jitter;
  if (!std::isnan(m_objective_ll))
    j["objective_ll"] = m_objective_ll;
}

void WarpKriging::load_from_json(const nlohmann::json& j) {
  if (j.contains("covType")) {
    // v3: field names match the base schema
    load_common_from_json(j);  // sets m_X (=Phi), m_centerX, m_y, …
    m_X_raw = mat_from_json(j["X_raw"]);
  } else {
    // v2 legacy: translate old field names
    m_X_raw = mat_from_json(j["X"]);
    m_y = colvec_from_json(j["y"]);
    m_normalize = j["normalize"].template get<bool>();
    m_centerX = rowvec_from_json(j["X_mean"]);
    m_scaleX = rowvec_from_json(j["X_std"]);
    m_centerY = j["y_mean"].template get<double>();
    m_scaleY = j["y_std"].template get<double>();
    m_regmodel = Trend::fromString(j["regmodel"].template get<std::string>());
    m_theta = colvec_from_json(j["theta"]);
    m_sigma2 = j["sigma2"].template get<double>();
    if (j.contains("noise"))
      m_noise = colvec_from_json(j["noise"]);
    m_beta = colvec_from_json(j["beta"]);
    m_z = colvec_from_json(j["z"]);
    m_M = mat_from_json(j["M"]);
    m_circ = mat_from_json(j["circ"]);
    m_F = mat_from_json(j["F"]);
    m_R = mat_from_json(j["R"]);
    m_T = mat_from_json(j["C"]);
    m_Rinv = mat_from_json(j["Rinv"]);
    m_X = mat_from_json(j["Phi"]);
    m_dX = mat_from_json(j["dPhi"]);
    m_maxdX = arma::max(arma::abs(m_dX), 1);
    m_is_empty = false;
  }
  // Absent from files saved before the fixed jitter: they used num_nugget
  m_chol_jitter = j.contains("chol_jitter") ? j["chol_jitter"].template get<double>() : LinearAlgebra::num_nugget;
  // Absent from files saved before the exact committed model: one jitter for both
  m_model_chol_jitter = j.contains("model_chol_jitter") ? j["model_chol_jitter"].template get<double>() : m_chol_jitter;
  m_cache_jitter = m_model_chol_jitter;
  m_objective_ll = j.contains("objective_ll") ? j["objective_ll"].template get<double>() : arma::datum::nan;
  // Warp-specific extras (present in both v2 and v3)
  m_logdet = j["logdet"].template get<double>();
  if (j.contains("is_continuous"))
    m_is_continuous = j["is_continuous"].template get<std::vector<bool>>();
  if (j.contains("feature_dim"))
    m_feature_dim = j["feature_dim"].template get<arma::uword>();
  m_max_iter_bfgs = j["max_iter_bfgs"].template get<arma::uword>();
  m_max_iter_adam = j["max_iter_adam"].template get<arma::uword>();
  m_adam_lr = j["adam_lr"].template get<double>();
  // Absent from files saved before #407: no calibration.
  m_calibration_folds = j.contains("calibration_folds") ? j["calibration_folds"].template get<arma::uword>() : 0;
  m_var_scale = j.contains("var_scale") ? j["var_scale"].template get<double>() : 1.0;
  if (j.contains("start_theta"))
    m_start_theta = colvec_from_json(j["start_theta"]);
  if (j.contains("start_warp_params"))
    m_start_warp_params = colvec_from_json(j["start_warp_params"]);
  m_fitted = j["fitted"].template get<bool>();
  if (m_is_joint)
    ensure_joint_warp(m_X_raw.n_cols);
  arma::vec wp = colvec_from_json(j["warp_params"]);
  unpack_warp_params(wp);
  // Domain-bounded warps (Knots) carry no serialised state of their own; the
  // input range is recovered deterministically from the restored m_X_raw.
  calibrate_warps();
}

// -------------------------------------------------------------------------
//  save / load
// -------------------------------------------------------------------------
void WarpKriging::save(const std::string filename) const {
  nlohmann::json j;
  j["version"] = 3;
  j["content"] = "WarpKriging";
  dump_to_json(j);
  std::ofstream f(filename);
  f << std::setw(4) << j;
}

WarpKriging WarpKriging::load(const std::string filename) {
  std::ifstream f(filename);
  nlohmann::json j = nlohmann::json::parse(f);

  uint32_t version = j["version"].template get<uint32_t>();
  if (version < 2 || version > 3)
    throw std::runtime_error(
        asString("Bad version to load from '", filename, "'; found ", version, ", requires 2 or 3"));
  std::string content = j["content"].template get<std::string>();
  if (content != "WarpKriging")
    throw std::runtime_error(
        asString("Bad content to load from '", filename, "'; found '", content, "', requires 'WarpKriging'"));

  auto warping = j["warping"].template get<std::vector<std::string>>();
  // v3 uses "covType" (base name); v2 used "kernel"
  std::string kernel
      = j.contains("covType") ? j["covType"].template get<std::string>() : j["kernel"].template get<std::string>();
  WarpKriging wk(warping, kernel);
  wk.load_from_json(j);
  return wk;
}

}  // namespace libKriging

// -------------------------------------------------------------------------
//  Public warped-kernel covariance (NestedKriging support)
// -------------------------------------------------------------------------

namespace libKriging {

arma::vec WarpKriging::warp_params() const {
  return pack_warp_params();
}

arma::mat WarpKriging::covMat(const arma::mat& X1, const arma::mat& X2) const {
  if (!m_fitted)
    throw std::runtime_error("covMat: model not fitted");
  if (X1.n_cols != m_X_raw.n_cols || X2.n_cols != m_X_raw.n_cols)
    throw std::invalid_argument("covMat: X1/X2 must have the same number of columns as the design");
  validate_discrete_columns(X1, "covMat");
  validate_discrete_columns(X2, "covMat");

  arma::mat Xn1 = X1;
  arma::mat Xn2 = X2;
  Xn1.each_row() -= m_centerX;
  Xn1.each_row() /= m_scaleX;
  Xn2.each_row() -= m_centerX;
  Xn2.each_row() /= m_scaleX;

  return m_sigma2 * build_Rcross(apply_warping(Xn1), apply_warping(Xn2));
}

}  // namespace libKriging
