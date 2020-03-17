#ifndef LIBKRIGING_UTILS_ENUM_UTILS_HPP
#define LIBKRIGING_UTILS_ENUM_UTILS_HPP

#include <string>

#include "details/enum_utils.hpp"

// Returns number of enum values.
template <typename E>
constexpr auto enum_count() noexcept -> detail::enable_if_enum_t<E, std::size_t> {
  // The upper bound of enum size is hardcoded to 128
  return detail::count<E>(std::make_integer_sequence<int, 128>{});
}

// This is the type that will hold all the strings.
// Each enumeration type will declare its own specialization.
// Any enum that does not have a specialization will generate a compiler error
// indicating that there is no definition of this variable (as there should be
// be no definition of a generic version).
template <typename T>
struct enumStrings {
  static const char* data[];
};

template <typename T>
std::string enumToString(T const& e) {
  static_assert(enum_count<T>() == std::end(enumStrings<T>::data) - std::begin(enumStrings<T>::data),
                "enumStrings<T> has an invalid size");
  return enumStrings<T>::data[static_cast<int>(e)];
}

template <typename T>
T enumFromString(const std::string& value) {
  static auto begin = std::begin(enumStrings<T>::data);
  static auto end = std::end(enumStrings<T>::data);

  auto find = std::find(begin, end, value);
  if (find != end) {
    return static_cast<T>(std::distance(begin, find));
  } else {
    // FIXME use std::optional as returned type
    throw std::exception();
  }
}

#endif  // LIBKRIGING_UTILS_ENUM_UTILS_HPP