/**
 * @file WarpKriging.cpp
 * @brief Per-variable warping Kriging implementation for libKriging.
 * See WarpKriging.hpp for documentation.
 */

#include "libKriging/WarpKriging.hpp"

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
  WarpSpec s; s.type = WarpType::None; return s;
}
WarpSpec WarpSpec::affine() {
  WarpSpec s; s.type = WarpType::Affine; return s;
}
WarpSpec WarpSpec::boxcox() {
  WarpSpec s; s.type = WarpType::BoxCox; return s;
}
WarpSpec WarpSpec::kumaraswamy() {
  WarpSpec s; s.type = WarpType::Kumaraswamy; return s;
}

WarpSpec WarpSpec::neural_mono(arma::uword nh) {
  WarpSpec s; s.type = WarpType::NeuralMono; s.n_hidden = nh; return s;
}

WarpSpec WarpSpec::categorical(arma::uword nlevels, arma::uword edim) {
  WarpSpec s; s.type = WarpType::Embedding;
  s.n_levels = nlevels; s.embed_dim = edim; return s;
}

WarpSpec WarpSpec::ordinal(arma::uword nlevels) {
  WarpSpec s; s.type = WarpType::Ordinal; s.n_levels = nlevels; return s;
}

WarpSpec WarpSpec::mlp(const std::vector<arma::uword>& hdims,
                       arma::uword dout,
                       const std::string& act) {
  WarpSpec s;
  s.type = WarpType::MLP;
  s.hidden_dims = hdims;
  s.d_out = dout;
  s.activation = act;
  return s;
}

// *************************************************************************
//  WarpNone
// *************************************************************************

arma::mat WarpNone::forward(const arma::vec& x) const {
  return arma::mat(x);  // (n × 1)
}

arma::vec WarpNone::backward(const arma::vec& /*x*/,
                             const arma::mat& /*dL_dPhi*/) const {
  return {};  // no params → empty gradient
}

// *************************************************************************
//  WarpAffine  :  w(x) = a·x + b
// *************************************************************************

WarpAffine::WarpAffine() : m_a(1.0), m_b(0.0) {}

arma::vec WarpAffine::get_params() const { return {m_a, m_b}; }

void WarpAffine::set_params(const arma::vec& p) {
  m_a = p(0);
  m_b = p(1);
}

arma::mat WarpAffine::forward(const arma::vec& x) const {
  return arma::mat(m_a * x + m_b);
}

arma::vec WarpAffine::backward(const arma::vec& x,
                               const arma::mat& dL_dPhi) const {
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

arma::vec WarpBoxCox::get_params() const { return {m_lambda}; }

void WarpBoxCox::set_params(const arma::vec& p) { m_lambda = p(0); }

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

arma::vec WarpBoxCox::backward(const arma::vec& x,
                               const arma::mat& dL_dPhi) const {
  // dw/dλ = [x^λ (λ ln(x) − 1) + 1] / λ²  (for λ ≠ 0)
  double grad_lambda = 0.0;
  for (arma::uword i = 0; i < x.n_elem; ++i) {
    double xi = std::max(x(i), 1e-10);
    double dw_dl;
    if (std::abs(m_lambda) < 1e-6) {
      dw_dl = 0.5 * std::log(xi) * std::log(xi);  // Taylor approx
    } else {
      double xp = std::pow(xi, m_lambda);
      dw_dl = (xp * (m_lambda * std::log(xi) - 1.0) + 1.0)
              / (m_lambda * m_lambda);
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

arma::vec WarpKumaraswamy::backward(const arma::vec& x,
                                    const arma::mat& dL_dPhi) const {
  double a = std::exp(m_log_a);
  double b = std::exp(m_log_b);
  double grad_log_a = 0.0, grad_log_b = 0.0;

  for (arma::uword i = 0; i < x.n_elem; ++i) {
    double xi = std::clamp(x(i), 1e-10, 1.0 - 1e-10);
    double xa  = std::pow(xi, a);
    double u   = 1.0 - xa;           // 1 - x^a
    double ub  = std::pow(u, b);     // (1 - x^a)^b

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
  s << "Kumaraswamy(a=" << std::exp(m_log_a)
    << ", b=" << std::exp(m_log_b) << ")";
  return s.str();
}

// *************************************************************************
//  WarpNeuralMono  :  monotone 1-hidden-layer network
//
//  w(x) = |W2|^T softplus(|W1| x + b1) + b2
//
//  Monotonicity is guaranteed by using positive weights (stored as exp(raw)).
// *************************************************************************

WarpNeuralMono::WarpNeuralMono(arma::uword n_hidden, uint64_t seed)
    : m_H(n_hidden) {
  arma::arma_rng::set_seed(seed);
  double scale = std::sqrt(2.0 / 1.0);  // Kaiming init, fan_in = 1
  m_raw_W1 = arma::randn<arma::vec>(m_H) * scale;
  m_b1     = arma::zeros<arma::vec>(m_H);
  m_raw_W2 = arma::randn<arma::vec>(m_H) * scale;
  m_b2     = 0.0;
}

arma::uword WarpNeuralMono::n_params() const {
  return m_H + m_H + m_H + 1;  // W1 + b1 + W2 + b2
}

arma::vec WarpNeuralMono::get_params() const {
  arma::vec p(n_params());
  arma::uword idx = 0;
  p.subvec(idx, idx + m_H - 1) = m_raw_W1; idx += m_H;
  p.subvec(idx, idx + m_H - 1) = m_b1;     idx += m_H;
  p.subvec(idx, idx + m_H - 1) = m_raw_W2; idx += m_H;
  p(idx) = m_b2;
  return p;
}

void WarpNeuralMono::set_params(const arma::vec& p) {
  arma::uword idx = 0;
  m_raw_W1 = p.subvec(idx, idx + m_H - 1); idx += m_H;
  m_b1     = p.subvec(idx, idx + m_H - 1); idx += m_H;
  m_raw_W2 = p.subvec(idx, idx + m_H - 1); idx += m_H;
  m_b2     = p(idx);
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

arma::vec WarpNeuralMono::backward(const arma::vec& x,
                                   const arma::mat& dL_dPhi) const {
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
      h(j)   = std::log1p(std::exp(z(j)));       // softplus
      sig(j) = 1.0 / (1.0 + std::exp(-z(j)));    // sigmoid = d(softplus)/dz
    }

    // d(out)/d(W2) = h  → d/d(raw_W2) = h * W2 (chain via exp)
    g_rW2 += dl * (h % W2);
    g_b2  += dl;

    // d(out)/d(h) = W2
    arma::vec dout_dh = W2;

    // d(h_j)/d(z_j) = sigmoid(z_j)
    arma::vec dh_dz = sig;

    // d(z_j)/d(W1_j) = x_i  → d/d(raw_W1) = x_i * W1 (chain via exp)
    g_rW1 += dl * (dout_dh % dh_dz) * x(i) % W1;
    g_b1  += dl * (dout_dh % dh_dz);
  }

  arma::uword idx = 0;
  grad.subvec(idx, idx + m_H - 1) = g_rW1; idx += m_H;
  grad.subvec(idx, idx + m_H - 1) = g_b1;  idx += m_H;
  grad.subvec(idx, idx + m_H - 1) = g_rW2; idx += m_H;
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
      const double alpha  = 1.6732632423543772;
      const double lambda = 1.0507009873554805;
      arma::mat out = Z;
      out.transform([&](double z) {
        return lambda * (z >= 0.0 ? z : alpha * (std::exp(z) - 1.0));
      });
      return out;
    }
    case Act::Tanh:
      return arma::tanh(Z);
    case Act::Sigmoid:
      return 1.0 / (1.0 + arma::exp(-Z));
    case Act::ELU: {
      arma::mat out = Z;
      out.transform([](double z) {
        return z >= 0.0 ? z : std::exp(z) - 1.0;
      });
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
      const double alpha  = 1.6732632423543772;
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
  if (lower == "relu")    return Act::ReLU;
  if (lower == "selu")    return Act::SELU;
  if (lower == "tanh")    return Act::Tanh;
  if (lower == "sigmoid") return Act::Sigmoid;
  if (lower == "elu")     return Act::ELU;
  throw std::invalid_argument("WarpMLP: unknown activation: " + s);
}

// -- construction -----------------------------------------------------------

WarpMLP::WarpMLP(const std::vector<arma::uword>& hidden_dims,
                 arma::uword d_out, Act activation, uint64_t seed)
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
    std::memcpy(p.memptr() + idx, m_W[l].memptr(),
                m_W[l].n_elem * sizeof(double));
    idx += m_W[l].n_elem;
    std::memcpy(p.memptr() + idx, m_b[l].memptr(),
                m_b[l].n_elem * sizeof(double));
    idx += m_b[l].n_elem;
  }
  return p;
}

void WarpMLP::set_params(const arma::vec& p) {
  arma::uword idx = 0;
  for (arma::uword l = 0; l < m_W.size(); ++l) {
    std::memcpy(m_W[l].memptr(), p.memptr() + idx,
                m_W[l].n_elem * sizeof(double));
    idx += m_W[l].n_elem;
    std::memcpy(m_b[l].memptr(), p.memptr() + idx,
                m_b[l].n_elem * sizeof(double));
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

arma::vec WarpMLP::backward(const arma::vec& x,
                            const arma::mat& dL_dPhi) const {
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
//  WarpEmbedding  :  level l → row l of E ∈ ℝ^{L × q}
// *************************************************************************

WarpEmbedding::WarpEmbedding(arma::uword n_levels, arma::uword embed_dim,
                             uint64_t seed)
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
      throw std::out_of_range("WarpEmbedding: level " +
                              std::to_string(level) +
                              " >= n_levels " +
                              std::to_string(m_n_levels));
    out.row(i) = m_E.row(level);
  }
  return out;
}

arma::vec WarpEmbedding::backward(const arma::vec& x,
                                  const arma::mat& dL_dPhi) const {
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

WarpOrdinal::WarpOrdinal(arma::uword n_levels, uint64_t seed)
    : m_n_levels(n_levels) {
  arma::arma_rng::set_seed(seed);
  m_raw_gaps = arma::zeros<arma::vec>(n_levels - 1);
}

arma::uword WarpOrdinal::n_params() const { return m_n_levels - 1; }

arma::vec WarpOrdinal::get_params() const { return m_raw_gaps; }

void WarpOrdinal::set_params(const arma::vec& p) { m_raw_gaps = p; }

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

arma::vec WarpOrdinal::backward(const arma::vec& x,
                                const arma::mat& dL_dPhi) const {
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
  if (s == "gauss" || s == "rbf")       return WarpBaseKernel::Gauss;
  if (s == "matern3_2" || s == "matern32") return WarpBaseKernel::Matern32;
  if (s == "matern5_2" || s == "matern52") return WarpBaseKernel::Matern52;
  if (s == "exp" || s == "exponential") return WarpBaseKernel::Exp;
  throw std::invalid_argument("Unknown kernel: " + name);
}

// -------------------------------------------------------------------------
std::unique_ptr<IWarp> WarpKriging::make_warp(const WarpSpec& spec) const {
  switch (spec.type) {
    case WarpType::None:        return std::make_unique<WarpNone>();
    case WarpType::Affine:      return std::make_unique<WarpAffine>();
    case WarpType::BoxCox:      return std::make_unique<WarpBoxCox>();
    case WarpType::Kumaraswamy: return std::make_unique<WarpKumaraswamy>();
    case WarpType::NeuralMono:  return std::make_unique<WarpNeuralMono>(spec.n_hidden);
    case WarpType::MLP:
      return std::make_unique<WarpMLP>(
          spec.hidden_dims, spec.d_out,
          WarpMLP::parse_act(spec.activation));
    case WarpType::Embedding:   return std::make_unique<WarpEmbedding>(spec.n_levels, spec.embed_dim);
    case WarpType::Ordinal:     return std::make_unique<WarpOrdinal>(spec.n_levels);
  }
  throw std::invalid_argument("Unknown WarpType");
}

void WarpKriging::build_warps() {
  m_warps.clear();
  m_feature_dim = 0;
  m_is_continuous.clear();
  for (const auto& spec : m_warp_specs) {
    auto w = make_warp(spec);
    m_feature_dim += w->output_dim();
    m_warps.push_back(std::move(w));
    m_is_continuous.push_back(
        spec.type != WarpType::Embedding && spec.type != WarpType::Ordinal);
  }
}

// -------------------------------------------------------------------------
//  Constructors
// -------------------------------------------------------------------------
WarpKriging::WarpKriging(const std::vector<WarpSpec>& warping,
                         const std::string& kernel)
    : m_warp_specs(warping),
      m_kernel_name(kernel),
      m_base_kernel(parse_kernel(kernel)) {
  build_warps();
}

WarpKriging::WarpKriging(
    const arma::vec& y, const arma::mat& X,
    const std::vector<WarpSpec>& warping, const std::string& kernel,
    const std::string& regmodel, bool normalize,
    const std::string& optim, const std::string& objective,
    const std::map<std::string, std::string>& parameters)
    : m_warp_specs(warping),
      m_kernel_name(kernel),
      m_base_kernel(parse_kernel(kernel)) {
  build_warps();
  fit(y, X, regmodel, normalize, optim, objective, parameters);
}

// -------------------------------------------------------------------------
//  Apply all per-variable warpings → concatenated feature matrix
// -------------------------------------------------------------------------
arma::mat WarpKriging::apply_warping(const arma::mat& X) const {
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
    for (arma::uword j = 0; j < d; ++j) F.col(c++) = Phi.col(j);
    for (arma::uword j = 0; j < d; ++j)
      for (arma::uword k = j; k < d; ++k)
        F.col(c++) = Phi.col(j) % Phi.col(k);
    return F;
  }
  throw std::invalid_argument("Unknown regmodel: " + m_regmodel);
}

// -------------------------------------------------------------------------
//  Kernel functions  (identical to NeuralKernelKriging)
// -------------------------------------------------------------------------
double WarpKriging::kernel_scalar(const arma::rowvec& phi_i,
                                  const arma::rowvec& phi_j) const {
  arma::rowvec diff = phi_i - phi_j;
  arma::rowvec scaled = diff / m_theta.t();
  double r2 = arma::dot(scaled, scaled);
  double r  = std::sqrt(r2);

  switch (m_base_kernel) {
    case WarpBaseKernel::Gauss:
      return m_sigma2 * std::exp(-0.5 * r2);
    case WarpBaseKernel::Matern32:
      return m_sigma2 * (1.0 + std::sqrt(3.0) * r) *
             std::exp(-std::sqrt(3.0) * r);
    case WarpBaseKernel::Matern52:
      return m_sigma2 * (1.0 + std::sqrt(5.0) * r + 5.0 / 3.0 * r2) *
             std::exp(-std::sqrt(5.0) * r);
    case WarpBaseKernel::Exp:
      return m_sigma2 * std::exp(-r);
  }
  return 0.0;
}

arma::mat WarpKriging::build_K(const arma::mat& Phi) const {
  const arma::uword n = Phi.n_rows;
  arma::mat K(n, n);
  for (arma::uword i = 0; i < n; ++i) {
    K(i, i) = m_sigma2;
    for (arma::uword j = i + 1; j < n; ++j) {
      double kij = kernel_scalar(Phi.row(i), Phi.row(j));
      K(i, j) = kij;
      K(j, i) = kij;
    }
  }
  return K;
}

arma::mat WarpKriging::build_Kcross(const arma::mat& Phi_new,
                                    const arma::mat& Phi_train) const {
  const arma::uword m = Phi_new.n_rows;
  const arma::uword n = Phi_train.n_rows;
  arma::mat Kc(m, n);
  for (arma::uword i = 0; i < m; ++i)
    for (arma::uword j = 0; j < n; ++j)
      Kc(i, j) = kernel_scalar(Phi_new.row(i), Phi_train.row(j));
  return Kc;
}

// -------------------------------------------------------------------------
//  Internal LL
// -------------------------------------------------------------------------
double WarpKriging::compute_ll_internal(const arma::vec& y,
                                        const arma::mat& F,
                                        const arma::vec& beta,
                                        const arma::vec& alpha,
                                        double logdet) {
  const arma::uword n = y.n_elem;
  arma::vec residual = y - F * beta;
  double quad = arma::dot(residual, alpha);
  return -0.5 * (n * std::log(2.0 * arma::datum::pi) + logdet + quad);
}

double WarpKriging::logLikelihood() const {
  if (!m_fitted)
    throw std::runtime_error("WarpKriging: model not fitted");
  return compute_ll_internal(m_y, m_F, m_beta, m_alpha, m_logdet);
}

std::tuple<double, arma::vec, arma::mat>
WarpKriging::logLikelihoodFun(const arma::vec& theta_gp,
                              bool withGrad, bool /*withHess*/) const {
  auto* self = const_cast<WarpKriging*>(this);
  arma::vec old_theta = m_theta;
  self->m_theta = theta_gp;
  self->refresh_cache();
  double ll = compute_ll_internal(m_y, m_F, m_beta, m_alpha, m_logdet);

  arma::vec grad;
  if (withGrad) {
    const double h = 1e-5;
    grad.set_size(theta_gp.n_elem);
    for (arma::uword k = 0; k < theta_gp.n_elem; ++k) {
      arma::vec tp = theta_gp, tm = theta_gp;
      tp(k) += h; tm(k) -= h;
      self->m_theta = tp; self->refresh_cache();
      double llp = compute_ll_internal(m_y, m_F, m_beta, m_alpha, m_logdet);
      self->m_theta = tm; self->refresh_cache();
      double llm = compute_ll_internal(m_y, m_F, m_beta, m_alpha, m_logdet);
      grad(k) = (llp - llm) / (2.0 * h);
    }
  }
  self->m_theta = old_theta;
  self->refresh_cache();
  return {ll, grad, arma::mat()};
}

// -------------------------------------------------------------------------
//  Data normalisation  (only continuous variables)
// -------------------------------------------------------------------------
void WarpKriging::normalise_data() {
  arma::uword d = m_X.n_cols;
  m_X_mean = arma::zeros<arma::rowvec>(d);
  m_X_std  = arma::ones<arma::rowvec>(d);

  if (m_normalize) {
    for (arma::uword j = 0; j < d; ++j) {
      if (m_is_continuous[j]) {
        m_X_mean(j) = arma::mean(m_X.col(j));
        m_X_std(j)  = arma::stddev(m_X.col(j));
        if (m_X_std(j) < 1e-12) m_X_std(j) = 1.0;
        m_X.col(j) = (m_X.col(j) - m_X_mean(j)) / m_X_std(j);
      }
      // Discrete/ordinal columns are NOT normalised
    }
    m_y_mean = arma::mean(m_y);
    m_y_std  = arma::stddev(m_y);
    if (m_y_std < 1e-12) m_y_std = 1.0;
    m_y = (m_y - m_y_mean) / m_y_std;
  } else {
    m_y_mean = 0.0;
    m_y_std  = 1.0;
  }
}

// -------------------------------------------------------------------------
//  Cache refresh
// -------------------------------------------------------------------------
void WarpKriging::refresh_cache() {
  const arma::uword n = m_y.n_elem;

  m_Phi = apply_warping(m_X);

  arma::mat K = build_K(m_Phi);
  K.diag() += 1e-8 * m_sigma2;

  m_C = arma::chol(K, "lower");
  m_logdet = 2.0 * arma::sum(arma::log(m_C.diag()));

  m_F = build_trend_matrix(m_X);
  arma::mat Cinv_F = arma::solve(arma::trimatl(m_C), m_F);
  arma::vec Cinv_y = arma::solve(arma::trimatl(m_C), m_y);

  arma::mat FtKinvF = Cinv_F.t() * Cinv_F;
  arma::vec FtKinvy = Cinv_F.t() * Cinv_y;
  m_beta = arma::solve(FtKinvF, FtKinvy);

  arma::vec residual = m_y - m_F * m_beta;
  arma::vec Cinv_res = arma::solve(arma::trimatl(m_C), residual);
  m_alpha = arma::solve(arma::trimatu(m_C.t()), Cinv_res);
}

// -------------------------------------------------------------------------
//  Parameter packing:  [ warp_params | log(theta) | log(sigma2) ]
// -------------------------------------------------------------------------
arma::uword WarpKriging::total_warp_params() const {
  arma::uword total = 0;
  for (const auto& w : m_warps) total += w->n_params();
  return total;
}

arma::vec WarpKriging::pack_params() const {
  arma::uword n_warp = total_warp_params();
  arma::uword n_gp   = m_theta.n_elem + 1;
  arma::vec all(n_warp + n_gp);

  arma::uword idx = 0;
  for (const auto& w : m_warps) {
    if (w->n_params() > 0) {
      arma::vec wp = w->get_params();
      all.subvec(idx, idx + wp.n_elem - 1) = wp;
      idx += wp.n_elem;
    }
  }
  all.subvec(idx, idx + m_theta.n_elem - 1) = arma::log(m_theta);
  idx += m_theta.n_elem;
  all(idx) = std::log(m_sigma2);
  return all;
}

void WarpKriging::unpack_params(const arma::vec& all) {
  arma::uword idx = 0;
  for (auto& w : m_warps) {
    arma::uword np = w->n_params();
    if (np > 0) {
      w->set_params(all.subvec(idx, idx + np - 1));
      idx += np;
    }
  }
  m_theta = arma::exp(all.subvec(idx, idx + m_theta.n_elem - 1));
  idx += m_theta.n_elem;
  m_sigma2 = std::exp(all(idx));
}

// -------------------------------------------------------------------------
//  dK/dPhi
// -------------------------------------------------------------------------
arma::mat WarpKriging::dK_dPhi(const arma::mat& Phi,
                               const arma::mat& dL_dK) const {
  const arma::uword n = Phi.n_rows;
  const arma::uword d = Phi.n_cols;
  arma::mat dL_dPhi(n, d, arma::fill::zeros);

  for (arma::uword i = 0; i < n; ++i) {
    for (arma::uword j = 0; j < n; ++j) {
      if (i == j) continue;
      double coeff = dL_dK(i, j);
      if (std::abs(coeff) < 1e-15) continue;

      arma::rowvec diff = Phi.row(i) - Phi.row(j);
      arma::rowvec scaled = diff / m_theta.t();
      double r2 = arma::dot(scaled, scaled);
      double r  = std::sqrt(std::max(r2, 1e-30));

      arma::rowvec dk_dphi_i(d);

      switch (m_base_kernel) {
        case WarpBaseKernel::Gauss: {
          double k_val = m_sigma2 * std::exp(-0.5 * r2);
          dk_dphi_i = -k_val * (diff / (m_theta.t() % m_theta.t()));
          break;
        }
        case WarpBaseKernel::Matern32: {
          double dk_dr = m_sigma2 * (-3.0 * r) *
                         std::exp(-std::sqrt(3.0) * r);
          if (r > 1e-15) {
            arma::rowvec dr_dphi = diff / (m_theta.t() % m_theta.t() * r);
            dk_dphi_i = dk_dr * dr_dphi;
          } else {
            dk_dphi_i.zeros();
          }
          break;
        }
        case WarpBaseKernel::Matern52: {
          double sr5 = std::sqrt(5.0);
          double exp_term = std::exp(-sr5 * r);
          double dk_dr = m_sigma2 * exp_term *
                         (-5.0 / 3.0 * r * (1.0 + sr5 * r));
          if (r > 1e-15) {
            arma::rowvec dr_dphi = diff / (m_theta.t() % m_theta.t() * r);
            dk_dphi_i = dk_dr * dr_dphi;
          } else {
            dk_dphi_i.zeros();
          }
          break;
        }
        case WarpBaseKernel::Exp: {
          double k_val = m_sigma2 * std::exp(-r);
          if (r > 1e-15) {
            arma::rowvec dr_dphi = diff / (m_theta.t() % m_theta.t() * r);
            dk_dphi_i = -k_val * dr_dphi;
          } else {
            dk_dphi_i.zeros();
          }
          break;
        }
      }
      dL_dPhi.row(i) += coeff * dk_dphi_i;
    }
  }
  return dL_dPhi;
}

// -------------------------------------------------------------------------
//  LL + gradient  (all params)
// -------------------------------------------------------------------------
std::pair<double, arma::vec>
WarpKriging::compute_loglik_and_grad(const arma::vec& all_params,
                                     bool need_grad) const {
  auto* self = const_cast<WarpKriging*>(this);
  self->unpack_params(all_params);
  self->refresh_cache();

  double ll = compute_ll_internal(m_y, m_F, m_beta, m_alpha, m_logdet);

  if (!need_grad) return {ll, {}};

  const arma::uword n = m_y.n_elem;
  arma::mat Kinv = arma::solve(arma::trimatu(m_C.t()),
                               arma::solve(arma::trimatl(m_C),
                                           arma::eye(n, n)));
  arma::mat dLL_dK = 0.5 * (m_alpha * m_alpha.t() - Kinv);

  // ∂LL/∂Phi  →  ∂LL/∂(warp_params)  via chain rule per variable
  arma::mat dLL_dPhi = self->dK_dPhi(m_Phi, dLL_dK);

  arma::uword n_warp = total_warp_params();
  arma::uword n_gp   = m_theta.n_elem + 1;
  arma::vec grad(n_warp + n_gp, arma::fill::zeros);

  // Warp parameter gradients
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

  // GP hyper-param gradients (numerical)
  const double h = 1e-5;
  for (arma::uword k = 0; k < n_gp; ++k) {
    arma::vec pp = all_params, pm = all_params;
    arma::uword offset = n_warp + k;
    pp(offset) += h; pm(offset) -= h;
    self->unpack_params(pp); self->refresh_cache();
    double llp = compute_ll_internal(m_y, m_F, m_beta, m_alpha, m_logdet);
    self->unpack_params(pm); self->refresh_cache();
    double llm = compute_ll_internal(m_y, m_F, m_beta, m_alpha, m_logdet);
    grad(n_warp + k) = (llp - llm) / (2.0 * h);
  }

  self->unpack_params(all_params);
  self->refresh_cache();
  return {ll, grad};
}

// -------------------------------------------------------------------------
//  Adam step
// -------------------------------------------------------------------------
void WarpKriging::adam_step(
    arma::vec& params, const arma::vec& grad,
    arma::vec& mm, arma::vec& vm,
    arma::uword t, double lr, double b1, double b2, double eps) const {
  mm = b1 * mm + (1.0 - b1) * grad;
  vm = b2 * vm + (1.0 - b2) * (grad % grad);
  arma::vec mh = mm / (1.0 - std::pow(b1, t));
  arma::vec vh = vm / (1.0 - std::pow(b2, t));
  params += lr * mh / (arma::sqrt(vh) + eps);  // ascent
}

// -------------------------------------------------------------------------
//  Joint optimisation
// -------------------------------------------------------------------------
void WarpKriging::optimise_joint(const std::string& method) {
  arma::vec all = pack_params();
  const arma::uword n_total = all.n_elem;
  const arma::uword n_warp  = total_warp_params();
  const arma::uword n_gp    = n_total - n_warp;

  if (method == "Adam" || method == "adam") {
    arma::vec mm = arma::zeros(n_total);
    arma::vec vm = arma::zeros(n_total);
    double best_ll = -arma::datum::inf;
    arma::vec best_params = all;

    for (arma::uword t = 1; t <= m_max_iter_adam; ++t) {
      auto [ll, grad] = compute_loglik_and_grad(all, true);
      if (ll > best_ll) { best_ll = ll; best_params = all; }
      adam_step(all, grad, mm, vm, t, m_adam_lr, 0.9, 0.999, 1e-8);
    }
    unpack_params(best_params);

  } else {
    // "BFGS+Adam": Adam on warps, gradient ascent on GP hypers
    arma::vec mm_w = arma::zeros(n_warp);
    arma::vec vm_w = arma::zeros(n_warp);
    double best_ll = -arma::datum::inf;
    arma::vec best_params = all;

    for (arma::uword epoch = 0; epoch < 10; ++epoch) {
      arma::uword steps = m_max_iter_adam / 10;
      for (arma::uword t = 1; t <= steps; ++t) {
        all = pack_params();
        auto [ll, grad] = compute_loglik_and_grad(all, true);
        if (ll > best_ll) { best_ll = ll; best_params = all; }

        // Update warp params with Adam
        if (n_warp > 0) {
          arma::vec wp = all.head(n_warp);
          arma::vec wg = grad.head(n_warp);
          adam_step(wp, wg, mm_w, vm_w,
                    epoch * steps + t, m_adam_lr, 0.9, 0.999, 1e-8);
          all.head(n_warp) = wp;
        }

        // Update GP params with gradient ascent
        arma::vec gp_grad = grad.tail(n_gp);
        all.tail(n_gp) += 1e-3 * gp_grad;
        unpack_params(all);
      }
    }
    unpack_params(best_params);
  }
  refresh_cache();
}

// -------------------------------------------------------------------------
//  fit()
// -------------------------------------------------------------------------
void WarpKriging::fit(
    const arma::vec& y, const arma::mat& X,
    const std::string& regmodel, bool normalize,
    const std::string& optim, const std::string& /*objective*/,
    const std::map<std::string, std::string>& parameters) {
  if (y.n_elem != X.n_rows)
    throw std::invalid_argument("fit: y/X size mismatch");
  if (X.n_cols != m_warp_specs.size())
    throw std::invalid_argument(
        "fit: X has " + std::to_string(X.n_cols) + " columns but " +
        std::to_string(m_warp_specs.size()) + " warp specs were given");

  m_y = y;
  m_X = X;
  m_regmodel = regmodel;
  m_normalize = normalize;

  for (const auto& [key, val] : parameters) {
    if (key == "adam_lr")        m_adam_lr = std::stod(val);
    if (key == "max_iter_adam")  m_max_iter_adam = std::stoul(val);
    if (key == "max_iter_bfgs") m_max_iter_bfgs = std::stoul(val);
  }

  normalise_data();

  m_theta  = arma::ones<arma::vec>(m_feature_dim);
  m_sigma2 = arma::var(m_y);
  if (m_sigma2 < 1e-12) m_sigma2 = 1.0;

  refresh_cache();
  optimise_joint(optim);
  m_fitted = true;
}

// -------------------------------------------------------------------------
//  predict()
// -------------------------------------------------------------------------
std::tuple<arma::vec, arma::vec, arma::mat>
WarpKriging::predict(const arma::mat& x_new,
                     bool withStd, bool withCov) const {
  if (!m_fitted)
    throw std::runtime_error("predict: model not fitted");

  arma::mat x_n = x_new;
  if (m_normalize) {
    for (arma::uword j = 0; j < x_n.n_cols; ++j) {
      if (m_is_continuous[j])
        x_n.col(j) = (x_n.col(j) - m_X_mean(j)) / m_X_std(j);
    }
  }

  const arma::uword m = x_n.n_rows;
  arma::mat Phi_new = apply_warping(x_n);
  arma::mat F_new = build_trend_matrix(x_n);
  arma::mat Kcross = build_Kcross(Phi_new, m_Phi);

  arma::vec mean = F_new * m_beta + Kcross * m_alpha;
  mean = mean * m_y_std + m_y_mean;

  arma::vec stdev;
  arma::mat cov;

  if (withStd || withCov) {
    arma::mat v = arma::solve(arma::trimatl(m_C), Kcross.t());
    arma::mat Cinv_F = arma::solve(arma::trimatl(m_C), m_F);
    arma::mat FtKinvF = Cinv_F.t() * Cinv_F;
    arma::mat FtKinvF_inv = arma::inv_sympd(FtKinvF);

    if (withCov) {
      arma::mat K_new = build_K(Phi_new);
      cov = K_new - v.t() * v;
      arma::mat R = F_new.t() - Cinv_F.t() * v;
      cov += R.t() * FtKinvF_inv * R;
      cov *= (m_y_std * m_y_std);
      cov = 0.5 * (cov + cov.t());
      cov.diag() = arma::clamp(cov.diag(), 0.0, arma::datum::inf);
      stdev = arma::sqrt(cov.diag());
    } else {
      arma::vec var_diag(m);
      for (arma::uword i = 0; i < m; ++i) {
        var_diag(i) = std::max(0.0,
                               m_sigma2 - arma::dot(v.col(i), v.col(i)));
        arma::vec r_i = F_new.row(i).t() - Cinv_F.t() * v.col(i);
        var_diag(i) += arma::dot(r_i, FtKinvF_inv * r_i);
      }
      var_diag *= (m_y_std * m_y_std);
      var_diag = arma::clamp(var_diag, 0.0, arma::datum::inf);
      stdev = arma::sqrt(var_diag);
    }
  }
  return {mean, stdev, cov};
}

// -------------------------------------------------------------------------
//  simulate()
// -------------------------------------------------------------------------
arma::mat WarpKriging::simulate(int nsim, uint64_t seed,
                                const arma::mat& x_new) const {
  if (!m_fitted)
    throw std::runtime_error("simulate: model not fitted");

  arma::arma_rng::set_seed(seed);
  auto [mean, stdev, cov] = predict(x_new, false, true);

  const arma::uword m = x_new.n_rows;
  cov.diag() += 1e-8 * m_sigma2 * m_y_std * m_y_std;
  arma::mat L;
  bool ok = arma::chol(L, cov, "lower");
  if (!ok) {
    arma::vec eigval;
    arma::mat eigvec;
    arma::eig_sym(eigval, eigvec, cov);
    eigval = arma::clamp(eigval, 1e-10, arma::datum::inf);
    L = eigvec * arma::diagmat(arma::sqrt(eigval));
  }

  arma::mat Z = arma::randn<arma::mat>(m, nsim);
  arma::mat sims(m, nsim);
  for (int s = 0; s < nsim; ++s)
    sims.col(s) = mean + L * Z.col(s);
  return sims;
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
    if (m_is_continuous[j])
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
  oss << "* WarpKriging (per-variable warped Kriging)\n"
      << "  - kernel:      " << m_kernel_name << "\n"
      << "  - regmodel:    " << m_regmodel << "\n"
      << "  - normalize:   " << (m_normalize ? "true" : "false") << "\n"
      << "  - n obs:       " << m_y.n_elem << "\n"
      << "  - d input:     " << m_warp_specs.size() << "\n"
      << "  - d features:  " << m_feature_dim << "\n";

  oss << "  - warpings:\n";
  for (arma::uword j = 0; j < m_warps.size(); ++j) {
    oss << "      x" << j << ": " << m_warps[j]->describe() << "\n";
  }

  if (m_fitted) {
    oss << "  - sigma2:      " << m_sigma2 << "\n"
        << "  - theta:       " << m_theta.t()
        << "  - beta:        " << m_beta.t()
        << "  - LL:          " << logLikelihood() << "\n"
        << "  - total warp params: " << total_warp_params() << "\n";
  } else {
    oss << "  ** model not yet fitted **\n";
  }
  return oss.str();
}

}  // namespace libKriging
