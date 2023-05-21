#ifndef LIBKRIGING_UTILS_HPP
#define LIBKRIGING_UTILS_HPP

#include <sstream>
#include <string>
#include <utility>

template <typename... Args>
std::string asString(Args&&... args) {
  std::ostringstream oss;
  ((oss << std::forward<Args>(args)), ...);
  return oss.str();
}

#endif  // LIBKRIGING_UTILS_HPP
