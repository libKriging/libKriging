#ifndef LIBKRIGING_SRC_LIB_CACHE_HPP
#define LIBKRIGING_SRC_LIB_CACHE_HPP

#include <functional>
#include "libKriging/libKriging_exports.h"

template <typename T>
struct Signature;

template <typename R, typename... Args>
struct Signature<std::function<R(Args...)>> {
  using type = R(Args...);
};

template <typename F, typename Signature>
class CacheFunction {};

template <typename F, typename R, typename... Args>
class CacheFunction<F, Signature<std::function<R(Args...)>>> {
 public:
  LIBKRIGING_EXPORT CacheFunction(const F& f) : m_f(f) {}

  //  template <typename... Args>
  //  LIBKRIGING_EXPORT auto operator()(Args... args) const {
  //    return m_f(args...);
  //  }

 private:
  std::function<R(Args...)> m_f;
};

//template <typename F>
//CacheFunction(const F& f) -> CacheFunction<F, Signature<decltype(std::function{f})>>;

#endif  // LIBKRIGING_SRC_LIB_CACHE_HPP
