#ifndef LIBKRIGING_BINDINGS_OCTAVE_TOOLS_REQUIRESARG_HPP
#define LIBKRIGING_BINDINGS_OCTAVE_TOOLS_REQUIRESARG_HPP

#include <string>
#include <variant>

#include "overload.hpp"

namespace RequiresArg {
struct AtLeast {
  unsigned n;
};

struct Exactly {
  unsigned n;
};
struct Range {
  unsigned min, max;
};
struct KVPairs {};
struct Autodetect {};

using Requirement = std::variant<AtLeast, Exactly, Range, KVPairs, Autodetect>;

bool validate(const Requirement& v, unsigned n);
std::string describe(const Requirement& v);
}  // namespace RequiresArg

#endif  // LIBKRIGING_BINDINGS_OCTAVE_TOOLS_REQUIRESARG_HPP
