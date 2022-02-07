#ifndef LIBKRIGING_SRC_LIB_CACHE_HPP
#define LIBKRIGING_SRC_LIB_CACHE_HPP

#include <functional>
#include <optional>
#include <tuple>
#include <unordered_map>
#include "libKriging/libKriging_exports.h"

template <typename T>
struct Signature;

template <typename R, typename... Args>
struct Signature<std::function<R(Args...)>> {
  using type = R(Args...);
};

struct CacheStat {
  uint32_t min_hit;
  uint32_t max_hit;
  uint32_t total_hit;
  float mean_hit;
  size_t cache_size;
};

template <typename Callable, typename Signature>
class CacheFunction {};

template <typename Callable, typename R, typename... Args>  // true type for performance and details
class CacheFunction<Callable, Signature<std::function<R(Args...)>>> {
 private:
  using HashKey = std::size_t;

 public:
  LIBKRIGING_EXPORT explicit CacheFunction(const Callable& callable) : m_callable(callable) {}

  LIBKRIGING_EXPORT auto operator()(Args... args) const -> R {
    const auto arg_key = hash_args(args...);
    auto [finder, is_new] = m_cache.insert({arg_key, R{}});
    ++m_cache_hit[arg_key];
    if (is_new) {
      return finder->second = m_callable(std::forward<Args>(args)...);
    } else {
      return finder->second;
    }
  }

 public:
  static auto hash_args(Args... args) -> HashKey {
    if constexpr (sizeof...(Args) == 0) {
      return 1;
    } else if constexpr (sizeof...(Args) == 1) {
      return std::hash<Args...>{}(args...);
    } else {
      return tupleHash(std::forward_as_tuple(args...), std::make_index_sequence<sizeof...(Args)>{});
    }
  }

  auto inspect(Args... args) -> uint32_t {
    const auto arg_key = hash_args(args...);
    const auto finder = m_cache_hit.find(arg_key);
    if (finder == m_cache_hit.end()) {
      return 0;
    } else {
      return finder->second;
    }
  }

  auto stat() -> CacheStat {
    std::vector<uint32_t> hits;
    const auto cache_size = m_cache_hit.size();
    hits.reserve(cache_size);
    for (auto [_, e] : m_cache_hit) {
      hits.push_back(e);
    }
    const auto min_hit = [&] {
      const auto min_element = std::min_element(hits.begin(), hits.end());
      return (min_element == hits.end()) ? 0 : *min_element;
    }();
    const auto max_hit = [&] {
      const auto max_element = std::max_element(hits.begin(), hits.end());
      return (max_element == hits.end()) ? 0 : *max_element;
    }();
    const auto total_hit = std::accumulate(hits.begin(), hits.end(), uint32_t{0});
    return {min_hit, max_hit, total_hit, static_cast<float>(total_hit) / cache_size, cache_size};
  }

 private:
  Callable m_callable;
  mutable std::unordered_map<HashKey, R> m_cache;
  mutable std::unordered_map<HashKey, uint32_t> m_cache_hit;

  template <typename Tuple, std::size_t... ids>
  static std::size_t tupleHash(const Tuple&& tuple, const std::index_sequence<ids...>&&) {
    std::size_t result = 0;
#pragma unroll
    for (auto const& hash : {hashValue(std::get<ids>(tuple))...}) {
      result ^= hash + 0x9e3779b9 + (result << 6) + (result >> 2);  // should not be symmetric => this hack
      //                                                            // same magic number as in boost::hash
      // https://stackoverflow.com/questions/4948780/magic-number-in-boosthash-combine/4948967#4948967
    }
    return result;
  };

  template <typename T>
  static std::size_t hashValue(T&& t) {
    return std::hash<std::decay_t<T>>{}(std::forward<T>(t));
  }
};

template <typename F>
CacheFunction(const F& f) -> CacheFunction<F, Signature<decltype(std::function{f})>>;

#endif  // LIBKRIGING_SRC_LIB_CACHE_HPP
