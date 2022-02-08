#ifndef LIBKRIGING_SRC_LIB_CACHE_HPP
#define LIBKRIGING_SRC_LIB_CACHE_HPP

#include <cassert>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include "libKriging/libKriging_exports.h"
#include "libKriging/utils/cache_details.hpp"

#define LIBKRIGING_CACHE_ANALYSE  // TODO
#define LIBKRIGING_CACHE_VERIFY

/* ----------------------------------------------------------------------------------------------------------------- */

class CacheFunctionCommon {
 public:
  struct CacheStat {
    uint32_t min_hit;
    uint32_t max_hit;
    uint32_t total_hit;
    float mean_hit;
    size_t cache_size;
  };

 protected:
  using HashKey = std::size_t;

 public:
  LIBKRIGING_EXPORT ~CacheFunctionCommon();
  LIBKRIGING_EXPORT auto stat() -> CacheStat;

 protected:
  mutable std::unordered_map<HashKey, uint32_t> m_cache_hit;
};

/* ----------------------------------------------------------------------------------------------------------------- */

std::ostream& operator<<(std::ostream& o, const CacheFunctionCommon::CacheStat&);

/* ----------------------------------------------------------------------------------------------------------------- */

template <typename Callable, typename Signature, typename... Contexts>
class CacheFunction {};

template <typename Callable,
          typename R,
          typename... Args,
          typename... Contexts>  // true type for performance and details
class CacheFunction<Callable, details::Signature<std::function<R(Args...)>>, Contexts...> : public CacheFunctionCommon {
 public:
  using type = typename details::Signature<std::function<R(Args...)>>::type;

 public:
  LIBKRIGING_EXPORT CacheFunction(const Callable& callable, const Contexts&... contexts)
      : m_callable(callable), m_context(contexts...) {}

  LIBKRIGING_EXPORT auto operator()(Args... args) const -> R {
    const auto arg_key = hash_args(args...);
    auto [finder, is_new] = m_cache.insert({arg_key, R{}});
    ++m_cache_hit[arg_key];
    if (is_new) {
      return finder->second = m_callable(std::forward<Args>(args)...);
    } else {
#ifdef LIBKRIGING_CACHE_VERIFY
      auto expected_result = m_callable(std::forward<Args>(args)...);
      assert(expected_result == finder->second);
#endif
      return finder->second;
    }
  }

 public:
  auto hash_args(Args... args) const -> HashKey {
    const auto args_hash = details::tupleHash(std::forward_as_tuple(args...));
    if constexpr (std::tuple_size<decltype(m_context)>{} == 0) {
      return args_hash;
    } else {
      const auto context_hash = details::tupleHash(m_context);
      return details::composeHash(context_hash, args_hash);
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

 private:
  Callable m_callable;
  std::tuple<const Contexts&...> m_context;
  mutable std::unordered_map<HashKey, R> m_cache;
};

/* ----------------------------------------------------------------------------------------------------------------- */

template <typename F, typename... Contexts>
CacheFunction(const F& f, const Contexts&...)
    -> CacheFunction<F, details::Signature<decltype(std::function{f})>, Contexts...>;

#endif  // LIBKRIGING_SRC_LIB_CACHE_HPP
