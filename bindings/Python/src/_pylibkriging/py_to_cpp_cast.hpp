#ifndef LIBKRIGING_BINDINGS_PYTHON_SRC__PYLIBKRIGING_PY_TO_CPP_CAST_HPP
#define LIBKRIGING_BINDINGS_PYTHON_SRC__PYLIBKRIGING_PY_TO_CPP_CAST_HPP

#include "libKriging/utils/lk_armadillo.hpp"  // should always be before any armadillo include

#include <pybind11/pybind11.h>
#include <algorithm>
#include <optional>
#include <stdexcept>

#include "../../../Octave/tools/formatString.hpp"

namespace py = pybind11;

template <typename T>
std::optional<T> to_cpp_cast(py::handle obj);

template <typename T>
struct CastSupportedType {};

template <typename T>
std::optional<T> get_entry(const py::dict& dict, std::string_view key) {
  auto finder = std::find_if(
      dict.begin(), dict.end(), [&key](const auto& kv) { return kv.first.template cast<std::string_view>() == key; });
  if (finder != dict.end()) {
    auto opt_val = to_cpp_cast<T>(finder->second);
    if (opt_val.has_value()) {
      return opt_val;
    } else {
      throw std::invalid_argument(
          formatString("entry '", key, "' exits but cannot satisfy required type '", CastSupportedType<T>::name, "'"));
    }
  } else {
    return std::nullopt;
  }
}

template <>
struct CastSupportedType<bool> {
  static constexpr std::string_view name = "bool";
};

template <>
struct CastSupportedType<int> {
  static constexpr std::string_view name = "int";
};

template <>
struct CastSupportedType<double> {
  static constexpr std::string_view name = "float";
};

template <>
struct CastSupportedType<std::string> {
  static constexpr std::string_view name = "str";
};

template <>
struct CastSupportedType<arma::mat> {
  static constexpr std::string_view name = "numpy matrix";
};

template <>
struct CastSupportedType<arma::colvec> {
  static constexpr std::string_view name = "numpy col vector";
};

template <>
struct CastSupportedType<arma::rowvec> {
  static constexpr std::string_view name = "numpy row vector";
};

#endif  // LIBKRIGING_BINDINGS_PYTHON_SRC__PYLIBKRIGING_PY_TO_CPP_CAST_HPP
