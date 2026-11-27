#ifndef LIBKRIGING_SRC_LIB_ENVSETTINGS_HPP
#define LIBKRIGING_SRC_LIB_ENVSETTINGS_HPP

// Default values of the settings from the LK_* environment variables
// (docs/Settings.md, #468). Read once, when the library is loaded (the static
// initializers of Optim.cpp, Parallel.cpp, LinearAlgebra.cpp, ThreadBudget.cpp);
// a setter call overrides them. An unset or empty variable gives the built-in
// default silently; a value that is not a valid number (the whole string, not
// a prefix: "12abc" and "3.7" for an integer are rejected), not a boolean, or
// out of range gives the built-in default and a [WARNING] line.

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>

#include "libKriging/utils/lk_armadillo.hpp"

namespace libKriging {
namespace detail {

inline void env_setting_warning(const char* name, const char* value, const char* expected, const std::string& def) {
  arma::cout << "[WARNING] " << name << "=\"" << value << "\" ignored (expected " << expected << "), default " << def
             << " used (docs/Settings.md)" << arma::endl;
}

/// Integer setting: the whole value must be an integer in [min_value, INT_MAX].
inline int env_int(const char* name, int def, int min_value = std::numeric_limits<int>::min()) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0')
    return def;
  errno = 0;
  char* end = nullptr;
  const long long n = std::strtoll(v, &end, 10);
  const bool ok = end != v && *end == '\0' && errno == 0 && n >= min_value && n <= std::numeric_limits<int>::max();
  if (!ok) {
    const std::string expected = min_value == std::numeric_limits<int>::min()
                                     ? std::string("an integer")
                                     : "an integer >= " + std::to_string(min_value);
    env_setting_warning(name, v, expected.c_str(), std::to_string(def));
    return def;
  }
  return static_cast<int>(n);
}

/// Real setting: the whole value must be a finite number >= min_value.
inline double env_double(const char* name, double def, double min_value = -std::numeric_limits<double>::infinity()) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0')
    return def;
  errno = 0;
  char* end = nullptr;
  const double x = std::strtod(v, &end);
  const bool ok = end != v && *end == '\0' && errno == 0 && std::isfinite(x) && x >= min_value;
  if (!ok) {
    const std::string expected
        = std::isinf(min_value) ? std::string("a finite number") : "a finite number >= " + std::to_string(min_value);
    std::ostringstream d;
    d << def;
    env_setting_warning(name, v, expected.c_str(), d.str());
    return def;
  }
  return x;
}

/// Boolean setting: 1 / true / yes / on and 0 / false / no / off, any case.
inline bool env_bool(const char* name, bool def) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0')
    return def;
  std::string s(v);
  for (char& c : s)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (s == "1" || s == "true" || s == "yes" || s == "on")
    return true;
  if (s == "0" || s == "false" || s == "no" || s == "off")
    return false;
  env_setting_warning(name, v, "1/true/yes/on or 0/false/no/off", def ? "true" : "false");
  return def;
}

}  // namespace detail
}  // namespace libKriging

#endif  // LIBKRIGING_SRC_LIB_ENVSETTINGS_HPP
