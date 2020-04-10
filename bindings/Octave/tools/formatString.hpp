#ifndef LIBKRIGING_BINDINGS_OCTAVE_TOOLS_FORMATSTRING_HPP
#define LIBKRIGING_BINDINGS_OCTAVE_TOOLS_FORMATSTRING_HPP

#include <sstream>

auto formatString = [](auto&& arg, auto&&... args) {
  std::ostringstream oss;
  oss << std::forward<decltype(arg)>(arg);
  ((oss << std::forward<decltype(args)>(args)), ...);
  return oss.str();
};

#endif  // LIBKRIGING_BINDINGS_OCTAVE_TOOLS_FORMATSTRING_HPP
