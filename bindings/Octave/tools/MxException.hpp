#ifndef LIBKRIGING_BINDINGS_OCTAVE_TOOLS_MXEXCEPTION_HPP
#define LIBKRIGING_BINDINGS_OCTAVE_TOOLS_MXEXCEPTION_HPP

#include <string>

#include "formatString.hpp"

struct MxException : public std::exception {
  template <typename... Args>
  MxException(const char* id_, const char* msg, Args... args) : id(id_), msg(formatString(msg, args...)) {}

  const char* id;
  const std::string msg;
};

#endif  // LIBKRIGING_BINDINGS_OCTAVE_TOOLS_MXEXCEPTION_HPP
