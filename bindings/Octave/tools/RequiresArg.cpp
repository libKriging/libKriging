#include "RequiresArg.hpp"

#include "formatString.hpp"

namespace RequiresArg {

bool validate(const Requirement& v, const unsigned n) {
  return std::visit(overload{
                        [n](const RequiresArg::AtLeast& x) { return (x.n <= n); },
                        [n](const RequiresArg::Exactly& x) { return (x.n == n); },
                        [n](const RequiresArg::Range& x) { return (x.min <= n && n <= x.max); },
                        [](const RequiresArg::Autodetect& /*unused*/) { return true; }  // default
                    },
                    v);
}

std::string describe(const Requirement& v) {
  return std::visit(
      overload{
          [](const RequiresArg::AtLeast& x) { return formatString("at least ", x.n, " arguments"); },
          [](const RequiresArg::Exactly& x) { return formatString("exactly ", x.n, " arguments"); },
          [](const RequiresArg::Range& x) { return formatString("between ", x.min, " and ", x.max, " arguments"); },
          [](const RequiresArg::Autodetect& /*unused*/) { return formatString("any number of arguments"); }  // default
      },
      v);
}

}  // namespace RequiresArg
