#ifndef LIBKRIGING_BINDINGS_R_RLIBKRIGING_SRC_UTILS_HPP
#define LIBKRIGING_BINDINGS_R_RLIBKRIGING_SRC_UTILS_HPP

#include <optional>

// alternative version of std::make_optional to be compatible with GCC 8.1-8.3
template <typename T, typename X>
std::optional<T> make_optional0(X&& x) {
  T t = x;
  return std::optional<T>(std::in_place, t);
}

#endif  // LIBKRIGING_BINDINGS_R_RLIBKRIGING_SRC_UTILS_HPP
