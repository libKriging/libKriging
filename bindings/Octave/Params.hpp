#ifndef LIBKRIGING_BINDINGS_OCTAVE_PARAMS_HPP
#define LIBKRIGING_BINDINGS_OCTAVE_PARAMS_HPP

#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include "libKriging/utils/lk_armadillo.hpp"

class Params {
 public:
  using SupportedTypes = std::variant<int32_t, uint64_t, double, bool, arma::mat, std::string>;

 private:
  std::unordered_map<std::string, SupportedTypes> m_kv;

 public:
  Params() = default;

 public:
  template <typename T>
  void set(const std::string& key, T value) {
    m_kv[key] = SupportedTypes(value);
  }

  template <typename T>
  [[nodiscard]] std::optional<T> get(const std::string& key) const {
    auto finder = m_kv.find(key);
    if (finder == m_kv.end()) {
      return std::nullopt;
    } else {
      const auto& values = finder->second;
      const T* value = std::get_if<T>(&values);
      if (value == nullptr) {
        return std::nullopt;
      } else {
        return std::make_optional(*value);
      }
    }
  }

  void display() const;
};

#endif  // LIBKRIGING_BINDINGS_OCTAVE_PARAMS_HPP
