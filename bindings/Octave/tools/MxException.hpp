#ifndef LIBKRIGING_BINDINGS_OCTAVE_TOOLS_MXEXCEPTION_HPP
#define LIBKRIGING_BINDINGS_OCTAVE_TOOLS_MXEXCEPTION_HPP

#include <string>

#include "formatString.hpp"

#ifdef MEX_DEBUG
#define LOCATION() \
  MxException::Location { __FILE__, __LINE__ }
#else
#define LOCATION() \
  MxException::NoLocation {}
#endif

struct MxException : public std::exception {
 public:
  struct Location {
    const char* file;
    const int line;
  };
  struct NoLocation {};

 public:
  //  template <typename... Args>
  //  MxException(const char* id_, const char* msg, Args... args) : id(id_), msg(formatString(msg, args...)) {}

  template <typename Arg, typename... Args>
  MxException(Location&& location, const char* id_, Arg msg, Args... args)
      : id(id_), msg(formatString(msg, args..., " at ", location.file, ':', location.line)) {}

  template <typename Arg, typename... Args>
  MxException(NoLocation&&, const char* id_, Arg msg, Args... args) : id(id_), msg(formatString(msg, args...)) {}

  const char* id;
  const std::string msg;
};

#endif  // LIBKRIGING_BINDINGS_OCTAVE_TOOLS_MXEXCEPTION_HPP
