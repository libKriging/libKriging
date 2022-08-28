#include "Params.hpp"
#include "tools/overload.hpp"

static std::string describe(const Params::SupportedTypes& v) {
  return std::visit(overload{[](const int32_t&) { return "int32"; },
                             [](const uint64_t&) { return "uint64"; },
                             [](const double&) { return "scalar"; },
                             [](const bool&) { return "logical"; },
                             [](const arma::mat&) { return "matrix"; },
                             [](const std::string&) { return "string"; }},
                    v);
}

void Params::display() const {
  for (const auto& kv : m_kv) {
    std::cout << kv.first << " -> " << describe(kv.second) << "\n";
  }
}
