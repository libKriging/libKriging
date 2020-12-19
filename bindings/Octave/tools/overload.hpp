#ifndef LIBKRIGING_BINDINGS_OCTAVE_TOOLS_OVERLOAD_HPP
#define LIBKRIGING_BINDINGS_OCTAVE_TOOLS_OVERLOAD_HPP

#if 1  // C++17 version

template <typename... Ts>
struct overload : Ts... {
  // overload(Ts... ts) : Ts(ts)... {} // can be replaced by CTAD
  using Ts::operator()...;
};
// Custom Template Argument Deduction Rules
template <typename... Ts>
overload(Ts...)->overload<Ts...>;

#else  // C++14 version

template <typename T, typename... Ts>
struct Overloader
    : T
    , Overloader<Ts...> {
  using T::operator();
  using Overloader<Ts...>::operator();
  // […]
};

template <typename T>
struct Overloader<T> : T {
  using T::operator();
};

template <typename... T>
constexpr auto overload(T&&... t) {
  return Overloader<T...>{std::forward<T>(t)...};
}

#endif

#endif  // LIBKRIGING_BINDINGS_OCTAVE_TOOLS_OVERLOAD_HPP
