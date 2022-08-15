#ifndef LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_DETAILS_HPP
#define LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_DETAILS_HPP

#include <functional>
#include <tuple>
#include <utility>

namespace details {

inline std::size_t composeHash(std::size_t acc, std::size_t new_hash) {
  return acc ^ (new_hash + 0x9e3779b9 + (acc << 6) + (acc >> 2));  // should not be symmetric => this hack
  //                                                             // same magic number as in boost::hash
  // https://stackoverflow.com/questions/4948780/magic-number-in-boosthash-combine/4948967#4948967
}

template <typename T>
static std::size_t hashValue(T&& t) {
  return std::hash<std::decay_t<T>>{}(std::forward<T>(t));
}

template <typename Tuple, std::size_t... ids>
static std::size_t _tupleHash(Tuple&& tuple, const std::index_sequence<ids...>&&) {
  if constexpr (sizeof...(ids) == 0) {
    return 1;
  } else if constexpr (sizeof...(ids) == 1) {
    return hashValue(std::get<0>(tuple));
  } else {
    std::size_t result = 0;
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 10
#endif
    for (auto const& hash : {hashValue(std::get<ids>(tuple))...}) {
      result = composeHash(result, hash);
    }
    return result;
  }
}

template <typename Tuple>
static std::size_t tupleHash(Tuple&& tuple) {
  return _tupleHash(std::forward<Tuple>(tuple), std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>{}>{});
}

}  // namespace details

#endif  // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_DETAILS_HPP
