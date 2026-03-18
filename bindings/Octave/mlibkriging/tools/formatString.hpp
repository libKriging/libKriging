#ifndef LIBKRIGING_BINDINGS_OCTAVE_TOOLS_FORMATSTRING_HPP
#define LIBKRIGING_BINDINGS_OCTAVE_TOOLS_FORMATSTRING_HPP

#include <sstream>
#include <string>

template <typename Arg, typename... Args>
std::string formatString(Arg&& arg, Args&&... args) {
  std::ostringstream oss;
  oss << std::forward<decltype(arg)>(arg);
  ((oss << std::forward<decltype(args)>(args)), ...);
  return oss.str();
};

#endif  // LIBKRIGING_BINDINGS_OCTAVE_TOOLS_FORMATSTRING_HPP
