#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

// This is the type that will hold all the strings.
// Each enumeration type will declare its own specialization.
// Any enum that does not have a specialization will generate a compiler error
// indicating that there is no definition of this variable (as there should be
// be no definition of a generic version).
template <typename T>
struct enumStrings {
  static char const* data[];
};

template <typename T>
std::string enumToString(T const& e) {
  assert(static_cast<std::size_t>(e) < sizeof(enumStrings<T>::data));
  return enumStrings<T>::data[static_cast<int>(e)];
}

template <typename T>
T enumFromString(const std::string& value) {
  //  static auto begin = std::begin(enumStrings<T>::data);
  //  static auto end = std::end(enumStrings<T>::data);

  auto begin = enumStrings<T>::data;
  auto end = begin + 3;  // FIXME: emergency fix with common value

  auto find = std::find(begin, end, value);
  if (find != end) {
    return static_cast<T>(std::distance(begin, find));
  } else {
    // FIXME use std::optional as returned type
    throw std::exception();
  }
}
