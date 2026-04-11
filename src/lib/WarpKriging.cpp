/**
 * @file WarpKriging.cpp
 * @brief Per-variable warping Kriging implementation for libKriging.
 * See WarpKriging.hpp for documentation.
 */

#include "libKriging/WarpKriging.hpp"
#include "libKriging/AdamBFGS.hpp"
#include "libKriging/Covariance.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/Optim.hpp"
#include "libKriging/Random.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace libKriging {

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

  if (type_str == "categorical") {
    if (args_str.empty())
      throw std::invalid_argument(
          "WarpSpec::from_string: categorical requires n_levels, e.g. 'categorical(5)' or 'categorical(5,2)'");
    auto parts = split(args_str, ',');
    arma::uword nl = static_cast<arma::uword>(std::stoul(parts[0]));
    arma::uword ed = (parts.size() >= 2) ? static_cast<arma::uword>(std::stoul(parts[1])) : 2;
    return WarpSpec::categorical(nl, ed);
  }

  if (type_str == "ordinal") {
    if (args_str.empty())
      throw std::invalid_argument("WarpSpec::from_string: ordinal requires n_levels, e.g. 'ordinal(4)'");
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
    case WarpType::Embedding:
      return "categorical(" + std::to_string(n_levels) + "," + std::to_string(embed_dim) + ")";
    case WarpType::Ordinal:
      return "ordinal(" + std::to_string(n_levels) + ")";
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
  }
  return "unknown";
}

// *************************************************************************
//  WarpNone
// *************************************************************************

arma::mat WarpNone::forward(const arma::vec& x) const {
  return arma::mat(x);  // (n × 1)
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
//  WarpBoxCox  :  w(x) = (x^λ − 1)/λ   (λ via unconstrained param)
//  If λ ≈ 0 → log(x)
// *************************************************************************

WarpBoxCox::WarpBoxCox() : m_lambda(1.0) {}

arma::vec WarpBoxCox::get_params() const {
  return {m_lambda};
}

void WarpBoxCox::set_params(const arma::vec& p) {
  m_lambda = p(0);
}

arma::mat WarpBoxCox::forward(const arma::vec& x) const {
  arma::vec out(x.n_elem);
  for (arma::uword i = 0; i < x.n_elem; ++i) {
    double xi = std::max(x(i), 1e-10);  // ensure positivity
    if (std::abs(m_lambda) < 1e-6)
      out(i) = std::log(xi);
    else
      out(i) = (std::pow(xi, m_lambda) - 1.0) / m_lambda;
  }
  return arma::mat(out);
}

arma::vec WarpBoxCox::backward(const arma::vec& x, const arma::mat& dL_dPhi) const {
  // dw/dλ = [x^λ (λ ln(x) − 1) + 1] / λ²  (for λ ≠ 0)
  double grad_lambda = 0.0;
  for (arma::uword i = 0; i < x.n_elem; ++i) {
    double xi = std::max(x(i), 1e-10);
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
  s << "BoxCox(lambda=" << m_lambda << ")";
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

arma::mat WarpKumaraswamy::forward(const arma::vec& x) const {
  double a = std::exp(m_log_a);
  double b = std::exp(m_log_b);
  arma::vec out(x.n_elem);
  for (arma::uword i = 0; i < x.n_elem; ++i) {
    double xi = std::clamp(x(i), 1e-10, 1.0 - 1e-10);
    out(i) = 1.0 - std::pow(1.0 - std::pow(xi, a), b);
  }
  return arma::mat(out);
}

arma::vec WarpKumaraswamy::backward(const arma::vec& x, const arma::mat& dL_dPhi) const {
  double a = std::exp(m_log_a);
  double b = std::exp(m_log_b);
  double grad_log_a = 0.0, grad_log_b = 0.0;

  for (arma::uword i = 0; i < x.n_elem; ++i) {
    double xi = std::clamp(x(i), 1e-10, 1.0 - 1e-10);
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
  s << "Kumaraswamy(a=" << std::exp(m_log_a) << ", b=" << std::exp(m_log_b) << ")";
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

arma::mat WarpNeuralMono::forward(const arma::vec& x) const {
  arma::vec W1 = arma::exp(m_raw_W1);  // positive weights
  arma::vec W2 = arma::exp(m_raw_W2);

  arma::uword n = x.n_elem;
  arma::vec out(n);

  for (arma::uword i = 0; i < n; ++i) {
    // hidden = softplus(W1 * x_i + b1)
    arma::vec z = W1 * x(i) + m_b1;
    arma::vec h(m_H);
    for (arma::uword j = 0; j < m_H; ++j)
      h(j) = std::log1p(std::exp(z(j)));  // softplus

    out(i) = arma::dot(W2, h) + m_b2;
  }
  return arma::mat(out);
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
    arma::vec z = W1 * x(i) + m_b1;
    arma::vec h(m_H), sig(m_H);
    for (arma::uword j = 0; j < m_H; ++j) {
      h(j) = std::log1p(std::exp(z(j)));       // softplus
      sig(j) = 1.0 / (1.0 + std::exp(-z(j)));  // sigmoid = d(softplus)/dz
    }

    // d(out)/d(W2) = h  → d/d(raw_W2) = h * W2 (chain via exp)
    g_rW2 += dl * (h % W2);
    g_b2 += dl;

    // d(out)/d(h) = W2
    arma::vec dout_dh = W2;

    // d(h_j)/d(z_j) = sigmoid(z_j)
    arma::vec dh_dz = sig;

    // d(z_j)/d(W1_j) = x_i  → d/d(raw_W1) = x_i * W1 (chain via exp)
    g_rW1 += dl * (dout_dh % dh_dz) * x(i) % W1;
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
  s << "NeuralMono(H=" << m_H << ", " << n_params() << " params)";
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
  // x is (n × 1) scalar input; reshape to matrix for batched matmul
  arma::mat H(x.n_elem, 1);
  H.col(0) = x;

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

// -- backward ---------------------------------------------------------------

arma::vec WarpMLP::backward(const arma::vec& x, const arma::mat& dL_dPhi) const {
  const arma::uword L = m_W.size();

  // Forward pass with caching
  std::vector<arma::mat> Z_cache(L);
  std::vector<arma::mat> H_cache(L + 1);
  H_cache[0] = arma::mat(x.n_elem, 1);
  H_cache[0].col(0) = x;

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
  s << ", " << m_n_params << " params)";
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

arma::mat WarpMLPJoint::forward(const arma::mat& X) const {
  arma::mat H = X;  // (n × d_in)
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

  // Forward with caching
  std::vector<arma::mat> Z_cache(L);
  std::vector<arma::mat> H_cache(L + 1);
  H_cache[0] = X;

  for (arma::uword l = 0; l < L; ++l) {
    arma::mat Z = H_cache[l] * m_W[l];
    Z.each_row() += m_b[l].t();
    Z_cache[l] = Z;
    if (l + 1 < L)
      H_cache[l + 1] = WarpMLP::apply_act(Z, m_act);
    else
      H_cache[l + 1] = Z;
  }

  // Backward
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

std::string WarpMLPJoint::describe() const {
  std::ostringstream s;
  s << "MLPJoint(" << m_d_in;
  for (arma::uword l = 0; l < m_W.size(); ++l)
    s << " -> " << m_W[l].n_cols;
  s << ", " << m_n_params << " params)";
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
//  WarpKriging  —  implementation
// *************************************************************************

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
    case WarpType::MLPJoint:
      // MLPJoint is not a per-variable IWarp — handled separately in build_warps
      throw std::invalid_argument("make_warp: MLPJoint is handled by build_warps, not make_warp");
  }
  throw std::invalid_argument("Unknown WarpType");
}

void WarpKriging::build_warps() {
  m_warps.clear();
  m_joint_warp.reset();
  m_has_joint = false;
  m_feature_dim = 0;
  m_is_continuous.clear();

  // Check for MLPJoint — must be the only entry
  if (m_warp_specs.size() == 1 && m_warp_specs[0].type == WarpType::MLPJoint) {
    m_has_joint = true;
    // d_in will be set at fit() time when we know X.n_cols
    // For now just record the feature dim
    m_feature_dim = m_warp_specs[0].d_out;
    // mark all inputs as continuous for normalisation
    // (actual count set at fit time)
    return;
  }

  // Ensure no MLPJoint mixed with per-variable warps
  for (const auto& spec : m_warp_specs) {
    if (spec.type == WarpType::MLPJoint)
      throw std::invalid_argument("mlp_joint must be the only warping entry (cannot mix with per-variable warps)");
  }

  // Per-variable mode
  for (const auto& spec : m_warp_specs) {
    auto w = make_warp(spec);
    m_feature_dim += w->output_dim();
    m_warps.push_back(std::move(w));
    m_is_continuous.push_back(spec.type != WarpType::Embedding && spec.type != WarpType::Ordinal);
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

  auto cov = Covariance::resolve(s);
  _Cov = std::move(cov.Cov);
  _DlnCovDtheta = std::move(cov.DlnCovDtheta);
  _DlnCovDx = std::move(cov.DlnCovDx);
}

void WarpKriging::init_from_specs(const std::vector<WarpSpec>& specs, const std::string& kernel) {
  m_warp_specs = specs;
  m_kernel_name = kernel;
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
                         const std::string& regmodel,
                         bool normalize,
                         const std::string& optim,
                         const std::string& objective,
                         const std::map<std::string, std::string>& parameters) {
  init_from_specs(parse_warp_strings(warping), kernel);
  fit(y, X, regmodel, normalize, optim, objective, parameters);
}

// -------------------------------------------------------------------------
//  Apply warping:  X → Φ
//  Two modes: per-variable concatenation or joint MLP
// -------------------------------------------------------------------------
arma::mat WarpKriging::apply_warping(const arma::mat& X) const {
  if (m_has_joint) {
    return m_joint_warp->forward(X);  // (n × d_in) → (n × d_out)
  }

  // Per-variable mode
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
//  Trend matrix
// -------------------------------------------------------------------------
arma::mat WarpKriging::build_trend_matrix(const arma::mat& X) const {
  const arma::uword n = X.n_rows;
  if (m_regmodel == "constant")
    return arma::ones<arma::mat>(n, 1);

  // For linear/quadratic, use the warped features rather than raw X
  arma::mat Phi = apply_warping(X);
  arma::uword d = Phi.n_cols;

  if (m_regmodel == "linear") {
    arma::mat F(n, 1 + d);
    F.col(0) = arma::ones<arma::vec>(n);
    F.cols(1, d) = Phi;
    return F;
  }
  if (m_regmodel == "quadratic") {
    arma::uword p = 1 + d + d * (d + 1) / 2;
    arma::mat F(n, p);
    F.col(0) = arma::ones<arma::vec>(n);
    arma::uword c = 1;
    for (arma::uword j = 0; j < d; ++j)
      F.col(c++) = Phi.col(j);
    for (arma::uword j = 0; j < d; ++j)
      for (arma::uword k = j; k < d; ++k)
        F.col(c++) = Phi.col(j) % Phi.col(k);
    return F;
  }
  throw std::invalid_argument("Unknown regmodel: " + m_regmodel);
}

// -------------------------------------------------------------------------
//  Data normalisation  (only continuous variables are normalised)
// -------------------------------------------------------------------------
void WarpKriging::normalise_data() {
  arma::uword d = m_X.n_cols;
  m_X_mean = arma::zeros<arma::rowvec>(d);
  m_X_std = arma::ones<arma::rowvec>(d);

  if (m_normalize) {
    for (arma::uword j = 0; j < d; ++j) {
      bool is_cont = m_has_joint || (j < m_is_continuous.size() && m_is_continuous[j]);
      if (is_cont) {
        m_X_mean(j) = arma::mean(m_X.col(j));
        m_X_std(j) = arma::stddev(m_X.col(j));
        if (m_X_std(j) < 1e-12)
          m_X_std(j) = 1.0;
        m_X.col(j) = (m_X.col(j) - m_X_mean(j)) / m_X_std(j);
      }
    }
    m_y_mean = arma::mean(m_y);
    m_y_std = arma::stddev(m_y);
    if (m_y_std < 1e-12)
      m_y_std = 1.0;
    m_y = (m_y - m_y_mean) / m_y_std;
  } else {
    m_y_mean = 0.0;
    m_y_std = 1.0;
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
// -------------------------------------------------------------------------
//  Precomputed pairwise differences (like Kriging's compute_dX)
// -------------------------------------------------------------------------
void WarpKriging::compute_dPhi() {
  const arma::uword n = m_Phi.n_rows;
  const arma::uword d = m_Phi.n_cols;
  m_dPhi.set_size(d, n * n);
  m_dPhi.zeros();

  const double* Phi_mem = m_Phi.memptr();
  double* dPhi_mem = m_dPhi.memptr();

  for (arma::uword i = 0; i < n; ++i) {
    for (arma::uword j = i + 1; j < n; ++j) {
      arma::uword ij = i * n + j;
      arma::uword ji = j * n + i;
      for (arma::uword k = 0; k < d; ++k) {
        double diff = Phi_mem[i + k * n] - Phi_mem[j + k * n];
        dPhi_mem[k + ij * d] = diff;
        dPhi_mem[k + ji * d] = diff;
      }
    }
  }
}

arma::mat WarpKriging::build_Rcross(const arma::mat& Phi_new, const arma::mat& Phi_train) const {
  const arma::uword m = Phi_new.n_rows;
  const arma::uword n = Phi_train.n_rows;
  arma::mat Rc(m, n);
  // covMat_rect expects transposed inputs (d × n)
  LinearAlgebra::covMat_rect(&Rc, Phi_new.t(), Phi_train.t(), m_theta, _Cov, 1.0);
  return Rc;
}

// -------------------------------------------------------------------------
//  refresh_cache  — rebuild Φ, R, Cholesky, β̂ (GLS), σ̂² (MLE), α
// -------------------------------------------------------------------------
void WarpKriging::refresh_cache() {
  m_Phi = apply_warping(m_X);
  compute_dPhi();
  refresh_cache_theta_only();
}

/// Rebuild R, Cholesky, β̂, σ̂², α from current m_Phi and m_theta.
/// Skips recomputing m_Phi and m_dPhi — use when only θ changed.
void WarpKriging::refresh_cache_theta_only() {
  const arma::uword n = m_y.n_elem;

  m_R.set_size(n, n);
  arma::vec diag_with_nugget(n, arma::fill::value(1.0 + 1e-8));
  m_C = LinearAlgebra::cholCov(&m_R, m_dPhi, m_theta, _Cov, 1, diag_with_nugget);
  m_logdet = 2.0 * arma::sum(arma::log(m_C.diag()));

  m_F = build_trend_matrix(m_X);
  arma::mat Cinv_F = arma::solve(arma::trimatl(m_C), m_F);
  arma::vec Cinv_y = arma::solve(arma::trimatl(m_C), m_y);

  arma::mat FtRinvF = Cinv_F.t() * Cinv_F;
  arma::vec FtRinvy = Cinv_F.t() * Cinv_y;
  m_beta = arma::solve(FtRinvF, FtRinvy);

  arma::vec residual = m_y - m_F * m_beta;
  arma::vec Cinv_res = arma::solve(arma::trimatl(m_C), residual);
  m_sigma2 = std::max(arma::dot(Cinv_res, Cinv_res) / n, 1e-12);

  m_alpha = arma::solve(arma::trimatu(m_C.t()), Cinv_res);
}

// -------------------------------------------------------------------------
//  Concentrated profile log-likelihood
//    LL(θ) = -n/2 [1 + log(2π) + log(σ̂²)] - ½ log|R|
// -------------------------------------------------------------------------
double WarpKriging::concentrated_ll() const {
  const double n = static_cast<double>(m_y.n_elem);
  return -0.5 * n * (1.0 + std::log(2.0 * arma::datum::pi) + std::log(m_sigma2)) - 0.5 * m_logdet;
}

double WarpKriging::logLikelihood() const {
  if (!m_fitted)
    throw std::runtime_error("WarpKriging: model not fitted");
  return concentrated_ll();
}

std::tuple<double, arma::vec, arma::mat> WarpKriging::logLikelihoodFun(const arma::vec& theta_gp,
                                                                       bool withGrad,
                                                                       bool /*withHess*/) const {
  auto* self = const_cast<WarpKriging*>(this);
  arma::vec old_theta = m_theta;
  self->m_theta = theta_gp;
  self->refresh_cache();

  double ll = concentrated_ll();
  arma::vec grad;
  if (withGrad) {
    auto [ll2, g] = concentrated_ll_and_grad_theta();
    grad = g;
    (void)ll2;
  }

  self->m_theta = old_theta;
  self->refresh_cache();
  return {ll, grad, arma::mat()};
}

// -------------------------------------------------------------------------
//  ∂R/∂θ_k  (analytical, n×n matrix)
// -------------------------------------------------------------------------
arma::mat WarpKriging::build_dR_dtheta_k(const arma::mat& /*Phi*/, arma::uword k) const {
  const arma::uword n = m_Phi.n_rows;
  arma::mat dR(n, n, arma::fill::zeros);

  for (arma::uword i = 0; i < n; ++i) {
    for (arma::uword j = i + 1; j < n; ++j) {
      arma::vec dlnCov = _DlnCovDtheta(m_dPhi.col(i * n + j), m_theta);
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

  arma::mat Rinv = arma::solve(arma::trimatu(m_C.t()), arma::solve(arma::trimatl(m_C), arma::eye(n, n)));

  arma::mat dLL_dR = 0.5 * (m_alpha * m_alpha.t() / m_sigma2 - Rinv);

  // Accumulate tr(dLL_dR · dR_k) for all k in a single pass over (i,j)
  arma::vec grad_log_theta(d, arma::fill::zeros);

  for (arma::uword i = 0; i < n; ++i) {
    for (arma::uword j = i + 1; j < n; ++j) {
      // Use _DlnCovDtheta like Kriging: ∂R_ij/∂θ_k = R_ij * DlnCovDtheta_k
      arma::vec dlnCov = _DlnCovDtheta(m_dPhi.col(i * n + j), m_theta);
      double R_ij = m_R(i, j);
      double w = 2.0 * dLL_dR(i, j);  // symmetry: (i,j) + (j,i)

      for (arma::uword k = 0; k < d; ++k) {
        double dR_ij = R_ij * dlnCov(k);
        grad_log_theta(k) += w * dR_ij * m_theta(k);
      }
    }
  }

  return {ll, grad_log_theta};
}

// -------------------------------------------------------------------------
//  Gradient of LL w.r.t. warp params  (backprop through K = σ̂² R)
// -------------------------------------------------------------------------
arma::mat WarpKriging::dK_dPhi(const arma::mat& Phi, const arma::mat& dL_dK) const {
  const arma::uword n = Phi.n_rows;
  const arma::uword d = Phi.n_cols;
  arma::mat dL_dPhi(n, d, arma::fill::zeros);

  // K_ij = σ² · C(dPhi_ij, θ)
  // ∂K_ij/∂Φ_i = K_ij · DlnCovDx(dPhi_ij, θ)   (sign: dPhi = Φ_i - Φ_j, so ∂dPhi/∂Φ_i = +1)
  for (arma::uword i = 0; i < n; ++i) {
    for (arma::uword j = 0; j < n; ++j) {
      if (i == j)
        continue;
      double coeff = dL_dK(i, j);
      if (std::abs(coeff) < 1e-15)
        continue;

      double K_ij = m_sigma2 * m_R(i, j);
      arma::vec dlnCdx = _DlnCovDx(m_dPhi.col(i * n + j), m_theta);
      // ∂K_ij/∂Φ_i_k = K_ij * dlnCdx_k
      dL_dPhi.row(i) += coeff * K_ij * dlnCdx.t();
    }
  }
  return dL_dPhi;
}

arma::vec WarpKriging::warp_gradient() const {
  const arma::uword n = m_y.n_elem;

  arma::mat Kinv
      = (1.0 / m_sigma2) * arma::solve(arma::trimatu(m_C.t()), arma::solve(arma::trimatl(m_C), arma::eye(n, n)));
  arma::mat dLL_dK = 0.5 * (m_alpha * m_alpha.t() - Kinv);

  arma::mat dLL_dPhi = dK_dPhi(m_Phi, dLL_dK);

  arma::uword n_warp = total_warp_params();
  arma::vec grad(n_warp, arma::fill::zeros);

  if (m_has_joint && m_joint_warp) {
    arma::vec gw = m_joint_warp->backward(m_X, dLL_dPhi);
    if (gw.n_elem > 0)
      grad.head(gw.n_elem) = gw;
  } else {
    arma::uword col = 0, idx = 0;
    for (arma::uword j = 0; j < m_warps.size(); ++j) {
      arma::uword dj = m_warps[j]->output_dim();
      arma::uword np = m_warps[j]->n_params();
      if (np > 0) {
        arma::mat dLL_dPhi_j = dLL_dPhi.cols(col, col + dj - 1);
        arma::vec gw = m_warps[j]->backward(m_X.col(j), dLL_dPhi_j);
        grad.subvec(idx, idx + np - 1) = gw;
        idx += np;
      }
      col += dj;
    }
  }
  return grad;
}

// -------------------------------------------------------------------------
//  Warp param packing  (θ is NOT here — optimised separately)
// -------------------------------------------------------------------------
arma::uword WarpKriging::total_warp_params() const {
  if (m_has_joint)
    return m_joint_warp ? m_joint_warp->n_params() : 0;
  arma::uword total = 0;
  for (const auto& w : m_warps)
    total += w->n_params();
  return total;
}

arma::vec WarpKriging::pack_warp_params() const {
  arma::uword n_warp = total_warp_params();
  if (n_warp == 0)
    return {};
  arma::vec wp(n_warp);
  arma::uword idx = 0;
  if (m_has_joint && m_joint_warp) {
    wp = m_joint_warp->get_params();
  } else {
    for (const auto& w : m_warps) {
      arma::uword np = w->n_params();
      if (np > 0) {
        wp.subvec(idx, idx + np - 1) = w->get_params();
        idx += np;
      }
    }
  }
  return wp;
}

void WarpKriging::unpack_warp_params(const arma::vec& wp) {
  arma::uword idx = 0;
  if (m_has_joint && m_joint_warp) {
    if (wp.n_elem > 0)
      m_joint_warp->set_params(wp);
  } else {
    for (auto& w : m_warps) {
      arma::uword np = w->n_params();
      if (np > 0) {
        w->set_params(wp.subvec(idx, idx + np - 1));
        idx += np;
      }
    }
  }
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
  arma::vec log_theta0 = arma::log(m_theta);

  // Compute data-driven θ bounds from feature-space pairwise diffs (like Kriging uses m_maxdX)
  arma::vec maxdPhi = arma::max(arma::abs(m_dPhi), 1);
  arma::vec theta_lower = Optim::theta_lower_factor * maxdPhi;
  arma::vec theta_upper = Optim::theta_upper_factor * maxdPhi;
  // Clamp to avoid degenerate bounds
  theta_lower = arma::clamp(theta_lower, 1e-10, arma::datum::inf);
  theta_upper = arma::max(theta_lower * 2.0, theta_upper);
  arma::vec log_theta_lower = arma::log(theta_lower);
  arma::vec log_theta_upper = arma::log(theta_upper);

  // Parse multistart count from method: "BFGS10" → 10, "BFGS" → 1, "BFGS5+Adam" → 5
  int multistart = 1;
  std::string base_method = method;
  // Extract the BFGS prefix and any trailing number
  std::string bfgs_prefix = "BFGS";
  if (method.rfind(bfgs_prefix, 0) == 0) {
    // Find end of digits after "BFGS"
    size_t pos = bfgs_prefix.size();
    size_t num_end = pos;
    while (num_end < method.size() && std::isdigit(method[num_end]))
      ++num_end;
    if (num_end > pos) {
      multistart = std::stoi(method.substr(pos, num_end - pos));
      // Reconstruct base method: "BFGS" + everything after the digits
      base_method = bfgs_prefix + method.substr(num_end);
    }
  }

  // Helper lambda: run joint L-BFGS-B on all params from current member state.
  // Returns the negative LL at the optimum (lower is better).
  auto run_joint_bfgs = [&]() -> double {
    arma::uword n_total = n_warp + d_theta;
    lbfgsb::Optimizer optimizer{static_cast<unsigned int>(n_total)};
    optimizer.iprint = -1;
    optimizer.max_iter = m_max_iter_bfgs;
    optimizer.pgtol = 1e-6;
    optimizer.factr = 1e7;

    arma::vec x0(n_total);
    if (n_warp > 0)
      x0.head(n_warp) = pack_warp_params();
    x0.tail(d_theta) = arma::log(m_theta);

    arma::vec lb(n_total), ub(n_total);
    arma::ivec btype(n_total);
    if (n_warp > 0) {
      lb.head(n_warp).fill(-1e20);
      ub.head(n_warp).fill(1e20);
      btype.head(n_warp).fill(0);
    }
    lb.tail(d_theta) = log_theta_lower;
    ub.tail(d_theta) = log_theta_upper;
    btype.tail(d_theta).fill(2);

    auto obj_fn = [&](const arma::vec& x, arma::vec& grad) -> double {
      if (n_warp > 0)
        unpack_warp_params(x.head(n_warp));
      m_theta = arma::exp(x.tail(d_theta));
      refresh_cache();

      auto [ll, g_log_theta] = concentrated_ll_and_grad_theta();
      grad.tail(d_theta) = -g_log_theta;
      if (n_warp > 0)
        grad.head(n_warp) = -warp_gradient();
      return -ll;
    };

    auto result = optimizer.minimize(obj_fn, x0, lb.memptr(), ub.memptr(), btype.memptr());

    if (n_warp > 0)
      unpack_warp_params(x0.head(n_warp));
    m_theta = arma::exp(x0.tail(d_theta));
    refresh_cache();
    return result.f_opt;
  };

  if (base_method == "BFGS" || n_warp == 0) {
    // --- Joint L-BFGS-B (with multistart) ---
    if (multistart <= 1) {
      run_joint_bfgs();
    } else {
      // Generate random starting points for θ in [theta_lower, theta_upper]
      arma::mat theta0_rand(multistart, d_theta);
      for (int i = 0; i < multistart; ++i)
        theta0_rand.row(i)
            = arma::trans(theta_lower + arma::randu<arma::vec>(d_theta) % (theta_upper - theta_lower));
      // First start uses the current theta (usually ones)
      theta0_rand.row(0) = m_theta.t();

      // Save initial warp state (for non-none warpings)
      arma::vec wp_init = pack_warp_params();

      double best_neg_ll = std::numeric_limits<double>::infinity();
      arma::vec best_theta;
      arma::vec best_wp;
      double best_sigma2 = 0;
      arma::vec best_beta;
      arma::vec best_alpha;
      arma::mat best_C;
      arma::mat best_R;
      arma::mat best_Phi;
      arma::mat best_dPhi;
      double best_logdet = 0;

      for (int i = 0; i < multistart; ++i) {
        // Reset warp params to initial state
        if (n_warp > 0)
          unpack_warp_params(wp_init);
        m_theta = theta0_rand.row(i).t();
        refresh_cache();

        double neg_ll = run_joint_bfgs();

        if (neg_ll < best_neg_ll) {
          best_neg_ll = neg_ll;
          best_theta = m_theta;
          best_wp = pack_warp_params();
          best_sigma2 = m_sigma2;
          best_beta = m_beta;
          best_alpha = m_alpha;
          best_C = m_C;
          best_R = m_R;
          best_Phi = m_Phi;
          best_dPhi = m_dPhi;
          best_logdet = m_logdet;
        }
      }

      // Restore best result
      if (n_warp > 0)
        unpack_warp_params(best_wp);
      m_theta = best_theta;
      m_sigma2 = best_sigma2;
      m_beta = best_beta;
      m_alpha = best_alpha;
      m_C = best_C;
      m_R = best_R;
      m_Phi = best_Phi;
      m_dPhi = best_dPhi;
      m_logdet = best_logdet;
    }

  } else {
    // --- Bi-level Adam+BFGS (with multistart) ---
    auto run_adam_bfgs = [&]() -> double {
      AdamBFGS opt(n_warp, d_theta);
      opt.max_iter_adam = m_max_iter_adam;
      opt.adam_lr = m_adam_lr;
      opt.max_iter_bfgs = m_max_iter_bfgs;
      opt.bfgs_pgtol = 1e-6;
      opt.bfgs_factr = 1e7;
      opt.maximize = true;

      arma::vec current_wp = pack_warp_params();
      arma::vec current_log_theta = arma::log(m_theta);

      auto obj_fn = [this, &current_wp](const arma::vec& x_outer,
                                        const arma::vec& x_inner,
                                        arma::vec* grad_outer,
                                        arma::vec* grad_inner) -> double {
        bool warp_changed = false;
        if (x_outer.n_elem > 0) {
          if (current_wp.n_elem != x_outer.n_elem || arma::any(current_wp != x_outer)) {
            unpack_warp_params(x_outer);
            current_wp = x_outer;
            warp_changed = true;
          }
        }

        m_theta = arma::exp(x_inner);

        if (warp_changed)
          refresh_cache();
        else
          refresh_cache_theta_only();

        double ll = concentrated_ll();

        if (grad_inner) {
          auto [ll2, g_log_theta] = concentrated_ll_and_grad_theta();
          *grad_inner = g_log_theta;
          (void)ll2;
        }
        if (grad_outer && x_outer.n_elem > 0) {
          *grad_outer = warp_gradient();
        }
        return ll;
      };

      auto result = opt.optimize(current_wp, current_log_theta, log_theta_lower, log_theta_upper, obj_fn);

      if (n_warp > 0)
        unpack_warp_params(result.x_outer);
      m_theta = arma::exp(result.x_inner);
      refresh_cache();
      return -concentrated_ll();  // return neg LL for consistency
    };

    if (multistart <= 1) {
      run_adam_bfgs();
    } else {
      arma::mat theta0_rand(multistart, d_theta);
      for (int i = 0; i < multistart; ++i)
        theta0_rand.row(i)
            = arma::trans(theta_lower + arma::randu<arma::vec>(d_theta) % (theta_upper - theta_lower));
      theta0_rand.row(0) = m_theta.t();

      arma::vec wp_init = pack_warp_params();

      double best_neg_ll = std::numeric_limits<double>::infinity();
      arma::vec best_theta;
      arma::vec best_wp;
      double best_sigma2 = 0;
      arma::vec best_beta;
      arma::vec best_alpha;
      arma::mat best_C;
      arma::mat best_R;
      arma::mat best_Phi;
      arma::mat best_dPhi;
      double best_logdet = 0;

      for (int i = 0; i < multistart; ++i) {
        if (n_warp > 0)
          unpack_warp_params(wp_init);
        m_theta = theta0_rand.row(i).t();
        refresh_cache();

        double neg_ll = run_adam_bfgs();

        if (neg_ll < best_neg_ll) {
          best_neg_ll = neg_ll;
          best_theta = m_theta;
          best_wp = pack_warp_params();
          best_sigma2 = m_sigma2;
          best_beta = m_beta;
          best_alpha = m_alpha;
          best_C = m_C;
          best_R = m_R;
          best_Phi = m_Phi;
          best_dPhi = m_dPhi;
          best_logdet = m_logdet;
        }
      }

      if (n_warp > 0)
        unpack_warp_params(best_wp);
      m_theta = best_theta;
      m_sigma2 = best_sigma2;
      m_beta = best_beta;
      m_alpha = best_alpha;
      m_C = best_C;
      m_R = best_R;
      m_Phi = best_Phi;
      m_dPhi = best_dPhi;
      m_logdet = best_logdet;
    }

    // Joint BFGS polish if method requests it
    if (base_method == "BFGS+Adam+BFGS" && n_warp > 0) {
      run_joint_bfgs();
    }
  }
}

// -------------------------------------------------------------------------
//  fit()
// -------------------------------------------------------------------------
void WarpKriging::fit(const arma::vec& y,
                      const arma::mat& X,
                      const std::string& regmodel,
                      bool normalize,
                      const std::string& optim,
                      const std::string& /*objective*/,
                      const std::map<std::string, std::string>& parameters) {
  if (y.n_elem != X.n_rows)
    throw std::invalid_argument("fit: y/X size mismatch");

  // Column count check: in joint mode, 1 spec covers all columns
  if (!m_has_joint && X.n_cols != m_warp_specs.size())
    throw std::invalid_argument("fit: X has " + std::to_string(X.n_cols) + " columns but "
                                + std::to_string(m_warp_specs.size()) + " warp specs were given");

  m_y = y;
  m_X = X;
  m_regmodel = regmodel;
  m_normalize = normalize;

  for (const auto& [key, val] : parameters) {
    if (key == "adam_lr")
      m_adam_lr = std::stod(val);
    if (key == "max_iter_adam")
      m_max_iter_adam = std::stoul(val);
    if (key == "max_iter_bfgs")
      m_max_iter_bfgs = std::stoul(val);
  }

  normalise_data();

  // In joint mode, instantiate the WarpMLPJoint now that we know d_in
  if (m_has_joint) {
    const auto& spec = m_warp_specs[0];
    auto hdims = spec.hidden_dims;
    if (hdims.empty())
      hdims = {32, 16};
    m_joint_warp = std::make_unique<WarpMLPJoint>(X.n_cols, hdims, spec.d_out, WarpMLP::parse_act(spec.activation), 42);
    m_feature_dim = spec.d_out;
  }

  m_theta = arma::ones<arma::vec>(m_feature_dim);
  // σ̂² is concentrated — computed in refresh_cache

  refresh_cache();
  optimise_joint(optim);
  m_fitted = true;
}

// -------------------------------------------------------------------------
//  predict()
// -------------------------------------------------------------------------
std::tuple<arma::vec, arma::vec, arma::mat, arma::mat, arma::mat> WarpKriging::predict(const arma::mat& x_new,
                                                                                      bool withStd,
                                                                                      bool withCov,
                                                                                      bool withDeriv) const {
  if (!m_fitted)
    throw std::runtime_error("predict: model not fitted");

  const arma::uword d = x_new.n_cols;
  arma::mat x_n = x_new;
  if (m_normalize) {
    for (arma::uword j = 0; j < x_n.n_cols; ++j) {
      bool is_cont = m_has_joint || (j < m_is_continuous.size() && m_is_continuous[j]);
      if (is_cont)
        x_n.col(j) = (x_n.col(j) - m_X_mean(j)) / m_X_std(j);
    }
  }

  const arma::uword n_n = x_n.n_rows;
  const arma::uword n_o = m_Phi.n_rows;
  arma::mat Phi_new = apply_warping(x_n);
  arma::mat F_new = build_trend_matrix(x_n);
  // Use correlation cross-matrix (not covariance K = σ²R), because m_C is
  // the Cholesky of R and m_alpha = R⁻¹(y − Fβ).
  arma::mat Rcross = build_Rcross(Phi_new, m_Phi);

  arma::vec mean = F_new * m_beta + Rcross * m_alpha;
  mean = mean * m_y_std + m_y_mean;

  arma::vec stdev;
  arma::mat cov;
  arma::mat Dyhat_n;
  arma::mat Dystdev_n;

  // Compute v = C⁻¹ Rcross' when needed for stdev or derivatives
  arma::mat v;
  arma::mat Cinv_F;
  arma::mat FtRinvF_inv;
  arma::vec ysd2_n;

  if (withStd || withCov || withDeriv) {
    v = arma::solve(arma::trimatl(m_C), Rcross.t());
    Cinv_F = arma::solve(arma::trimatl(m_C), m_F);
    arma::mat FtRinvF = Cinv_F.t() * Cinv_F;
    FtRinvF_inv = arma::inv_sympd(FtRinvF);

    if (withCov) {
      arma::mat R_new(n_n, n_n);
      LinearAlgebra::covMat_sym_X(&R_new, Phi_new.t(), m_theta, _Cov);
      cov = R_new - v.t() * v;
      arma::mat H = F_new.t() - Cinv_F.t() * v;
      cov += H.t() * FtRinvF_inv * H;
      cov *= (m_sigma2 * m_y_std * m_y_std);
      cov = 0.5 * (cov + cov.t());
      cov.diag() = arma::clamp(cov.diag(), 0.0, arma::datum::inf);
      stdev = arma::sqrt(cov.diag());
      ysd2_n = cov.diag();
    } else {
      ysd2_n.set_size(n_n);
      for (arma::uword i = 0; i < n_n; ++i) {
        ysd2_n(i) = std::max(0.0, 1.0 - arma::dot(v.col(i), v.col(i)));
        arma::vec r_i = F_new.row(i).t() - Cinv_F.t() * v.col(i);
        ysd2_n(i) += arma::dot(r_i, FtRinvF_inv * r_i);
      }
      ysd2_n *= (m_sigma2 * m_y_std * m_y_std);
      ysd2_n = arma::clamp(ysd2_n, 0.0, arma::datum::inf);
      if (withStd)
        stdev = arma::sqrt(ysd2_n);
    }
  }

  if (withDeriv) {
    const double h = 1.0E-5;
    const double sigma2 = m_sigma2;

    Dyhat_n.set_size(n_n, d);
    Dyhat_n.zeros();
    arma::mat Dysd2_n(n_n, d, arma::fill::zeros);

    // Ecirc_n needed for stdev derivative
    arma::mat Ecirc_n;
    arma::mat circ;
    if (withStd) {
      arma::mat FtRinvF = Cinv_F.t() * Cinv_F;
      circ = arma::chol(FtRinvF);  // upper triangular
      arma::mat Fhat_n = v.t() * Cinv_F;
      arma::mat E_n = F_new - Fhat_n;
      Ecirc_n = arma::solve(arma::trimatu(circ), E_n.t()).t();  // E_n * circ⁻¹
    }

    for (arma::uword i = 0; i < n_n; i++) {
      // Build perturbed points for finite-difference Jacobian (2d points)
      arma::mat x_perturbed(2 * d, d);
      for (arma::uword k = 0; k < d; k++) {
        x_perturbed.row(k) = x_n.row(i);
        x_perturbed.row(d + k) = x_n.row(i);
        x_perturbed(k, k) += h;
        x_perturbed(d + k, k) -= h;
      }

      // Warping Jacobian: dΦ/dx  (feature_dim × d)
      arma::mat Phi_perturbed = apply_warping(x_perturbed);
      arma::mat J_warp(m_feature_dim, d);
      for (arma::uword k = 0; k < d; k++) {
        J_warp.col(k) = (Phi_perturbed.row(k) - Phi_perturbed.row(d + k)).t() / (2.0 * h);
      }

      // Trend derivative: dF/dx  (d × p) via finite differences  [same layout as Kriging]
      arma::mat F_perturbed = build_trend_matrix(x_perturbed);
      arma::mat DF_n_i(d, F_new.n_cols);
      for (arma::uword k = 0; k < d; k++) {
        DF_n_i.row(k) = (F_perturbed.row(k) - F_perturbed.row(d + k)) / (2.0 * h);
      }

      // DR_on_i: derivative of cross-correlation w.r.t. x  (n_o × d)
      // dR(Φ_new, Φ_train)/dx = R * DlnCovDx(ΔΦ, θ)ᵀ · J_warp
      arma::mat DR_on_i(n_o, d);
      for (arma::uword j = 0; j < n_o; j++) {
        arma::vec dPhi = Phi_new.row(i).t() - m_Phi.row(j).t();
        arma::vec dlnCovDx = _DlnCovDx(dPhi, m_theta);
        DR_on_i.row(j) = Rcross(i, j) * (dlnCovDx.t() * J_warp);
      }

      // W_i = C⁻¹ DR_on_i  (n_o × d)
      arma::mat W_i = arma::solve(arma::trimatl(m_C), DR_on_i);

      // Mean derivative: dŷ/dx = dF/dx · β + dR/dx · α
      Dyhat_n.row(i) = (DF_n_i * m_beta + DR_on_i.t() * m_alpha).t();

      if (withStd) {
        // dvar/dx = -2 vᵢᵀ Wᵢ + 2 Ecirc_i · circ⁻ᵀ(DF_i - Wᵢᵀ Cinv_F)ᵀ
        arma::mat DEcirc_n_i
            = arma::solve(arma::trimatu(circ).t(), (DF_n_i - W_i.t() * Cinv_F).t());  // circ⁻ᵀ * (...)ᵀ
        Dysd2_n.row(i) = -2.0 * v.col(i).t() * W_i + 2.0 * Ecirc_n.row(i) * DEcirc_n_i;
      }
    }
    Dyhat_n *= m_y_std;
    Dysd2_n *= sigma2 * m_y_std * m_y_std;

    if (withStd) {
      // Dystdev = d(sqrt(var))/dx = dvar/dx / (2·stdev)
      Dystdev_n.set_size(n_n, d);
      for (arma::uword i = 0; i < n_n; i++) {
        double sd_i = std::sqrt(ysd2_n(i));
        if (sd_i > 0)
          Dystdev_n.row(i) = Dysd2_n.row(i) / (2.0 * sd_i);
        else
          Dystdev_n.row(i).zeros();
      }
    }
  }

  return {mean, stdev, cov, Dyhat_n, Dystdev_n};
}

// -------------------------------------------------------------------------
//  simulate()  — consistent with Kriging::simulate
//
//  Works at correlation scale (R, not K = σ²R), applies σ̂ scaling after
//  Cholesky, then denormalizes.  This is more numerically robust than
//  doing Cholesky on the full-scale covariance matrix.
// -------------------------------------------------------------------------
arma::mat WarpKriging::simulate(int nsim, uint64_t seed, const arma::mat& x_new) const {
  if (!m_fitted)
    throw std::runtime_error("simulate: model not fitted");

  const arma::uword n_n = x_new.n_rows;

  // Normalize new inputs
  arma::mat x_n = x_new;
  if (m_normalize) {
    for (arma::uword j = 0; j < x_n.n_cols; ++j) {
      bool is_cont = m_has_joint || (j < m_is_continuous.size() && m_is_continuous[j]);
      if (is_cont)
        x_n.col(j) = (x_n.col(j) - m_X_mean(j)) / m_X_std(j);
    }
  }

  arma::mat Phi_new = apply_warping(x_n);
  arma::mat F_new = build_trend_matrix(x_n);

  // R_nn: correlation between new points (at correlation scale)
  arma::mat R_nn(n_n, n_n);
  LinearAlgebra::covMat_sym_X(&R_nn, Phi_new.t(), m_theta, _Cov);

  // R_on: cross-correlation (n_train × n_new)
  arma::mat Rcross = build_Rcross(Phi_new, m_Phi);  // n_new × n_train

  // v = C⁻¹ Rcross'  (C is lower-Cholesky of R_train)
  arma::mat v = arma::solve(arma::trimatl(m_C), Rcross.t());

  // Kriging mean (normalized scale)
  arma::vec yhat_n = F_new * m_beta + Rcross * m_alpha;

  // Conditional covariance at correlation scale (including trend uncertainty)
  arma::mat Cinv_F = arma::solve(arma::trimatl(m_C), m_F);
  arma::mat FtRinvF = Cinv_F.t() * Cinv_F;
  arma::mat Ecirc_n = F_new - v.t() * Cinv_F;                       // n_n × p
  arma::mat FtRinvF_inv = arma::inv_sympd(FtRinvF);

  arma::mat Sigma_nKo = R_nn - v.t() * v + Ecirc_n * FtRinvF_inv * Ecirc_n.t();

  // Cholesky at correlation scale — more robust than on the full-scale covariance
  arma::mat LSigma = LinearAlgebra::safe_chol_lower(Sigma_nKo);

  // Generate simulations: mean + L * Z * sqrt(σ²), then denormalize
  arma::mat y_n(n_n, nsim);
  y_n.each_col() = yhat_n;
  Random::reset_seed(seed);
  y_n += LSigma * Random::randn_mat(n_n, nsim) * std::sqrt(m_sigma2);

  // Denormalize
  y_n = m_y_mean + m_y_std * y_n;

  return y_n;
}

// -------------------------------------------------------------------------
//  update()
// -------------------------------------------------------------------------
void WarpKriging::update(const arma::vec& y_new, const arma::mat& X_new) {
  if (!m_fitted)
    throw std::runtime_error("update: model not fitted");

  // De-normalise stored data, append, re-normalise
  arma::vec y_all = arma::join_vert(m_y * m_y_std + m_y_mean, y_new);
  arma::mat X_all = m_X;
  for (arma::uword j = 0; j < X_all.n_cols; ++j) {
    bool is_cont = m_has_joint || (j < m_is_continuous.size() && m_is_continuous[j]);
    if (is_cont)
      X_all.col(j) = X_all.col(j) * m_X_std(j) + m_X_mean(j);
  }
  X_all = arma::join_vert(X_all, X_new);

  m_y = y_all;
  m_X = X_all;
  normalise_data();
  refresh_cache();

  arma::uword saved_adam = m_max_iter_adam;
  m_max_iter_adam /= 5;
  optimise_joint("BFGS+Adam");
  m_max_iter_adam = saved_adam;
}

// -------------------------------------------------------------------------
//  summary()
// -------------------------------------------------------------------------
std::string WarpKriging::summary() const {
  std::ostringstream oss;
  oss << "* WarpKriging\n"
      << "  - kernel:      " << m_kernel_name << "\n"
      << "  - regmodel:    " << m_regmodel << "\n"
      << "  - normalize:   " << (m_normalize ? "true" : "false") << "\n"
      << "  - n obs:       " << m_y.n_elem << "\n"
      << "  - d input:     " << m_X.n_cols << "\n"
      << "  - d features:  " << m_feature_dim << "\n";

  if (m_has_joint && m_joint_warp) {
    oss << "  - warping:     \"" << m_warp_specs[0].to_string() << "\"  →  " << m_joint_warp->describe() << "\n";
  } else {
    oss << "  - warpings:\n";
    for (arma::uword j = 0; j < m_warps.size(); ++j) {
      oss << "      x" << j << ": \"" << m_warp_specs[j].to_string() << "\"  →  " << m_warps[j]->describe() << "\n";
    }
  }

  if (m_fitted) {
    oss << "  - sigma2:      " << m_sigma2 << "\n"
        << "  - theta:       " << m_theta.t() << "  - beta:        " << m_beta.t()
        << "  - LL:          " << logLikelihood() << "\n"
        << "  - total warp params: " << total_warp_params() << "\n";
  } else {
    oss << "  ** model not yet fitted **\n";
  }
  return oss.str();
}

}  // namespace libKriging
