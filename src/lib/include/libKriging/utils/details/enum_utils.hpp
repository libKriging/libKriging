#ifndef LIBKRIGING_UTILS_DETAILS_ENUM_UTILS_HPP
#define LIBKRIGING_UTILS_DETAILS_ENUM_UTILS_HPP

#include <algorithm>
#include <limits>
#include <type_traits>
#include <utility>

namespace detail {

template <typename T>
constexpr bool is_enum_v = std::is_enum<T>::value&& std::is_same<T, std::decay_t<T>>::value;

template <typename E, E V>
constexpr auto is_valid_enum_value() noexcept {
  static_assert(is_enum_v<E>, "enum_utils is only for enum");
#if defined(__clang__) || defined(__GNUC__)
  constexpr const char* name = __PRETTY_FUNCTION__;
  constexpr std::size_t size = sizeof(__PRETTY_FUNCTION__) - 2;
#elif defined(_MSC_VER)
  constexpr const char* name = __FUNCSIG__;
  constexpr std::size_t size = sizeof(__FUNCSIG__) - 17;
#endif
  // Try to build a valid name from an internal compiler naming
  for (std::size_t i = size; i > 0; --i) {
    if (!((name[i - 1] >= '0' && name[i - 1] <= '9') || (name[i - 1] >= 'a' && name[i - 1] <= 'z')
          || (name[i - 1] >= 'A' && name[i - 1] <= 'Z') || (name[i - 1] == '_'))) {
      return (size > i
              && ((name[i] >= 'a' && name[i] <= 'z') || (name[i] >= 'A' && name[i] <= 'Z') || (name[i] == '_')));
    }
  }
  return false;
}

template <typename E, int... I>
struct stats {};

template <typename E>
struct stats<E> {
  static const int min_index = std::numeric_limits<int>::max();
  static const int max_index = -1;
  static const int count = 0;
};

template <typename E, int I, int... others>
struct stats<E, I, others...> {
  static const bool is_valid = is_valid_enum_value<E, static_cast<E>(I)>();
  static const int min_index
      = (is_valid) ? std::min(stats<E, others...>::min_index, I) : stats<E, others...>::min_index;
  static const int max_index
      = (is_valid) ? std::max(stats<E, others...>::max_index, I) : stats<E, others...>::max_index;
  static const int count = stats<E, others...>::count + (is_valid);
};

template <typename E, int... I>
constexpr auto count(std::integer_sequence<int, I...>) noexcept {
  static_assert(is_enum_v<E>, "magic_enum::detail::values requires enum type.");
  constexpr int count = stats<E, I...>::count;
  constexpr bool is_dense = stats<E, I...>::min_index == 0 && stats<E, I...>::max_index == count - 1;
  static_assert(is_dense, "enum_utils only supports enum indexed by a contiguous sequence starting from 0");
  return count;
}

template <typename T, typename R>
using enable_if_enum_t = std::enable_if_t<std::is_enum<std::decay_t<T>>::value, R>;

}  // namespace detail

#endif  // LIBKRIGING_UTILS_DETAILS_ENUM_UTILS_HPP
