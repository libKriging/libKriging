#ifndef LIBKRIGING_UTILS_HPP
#define LIBKRIGING_UTILS_HPP

template <typename... Args>
std::string asString(Args&&... args) {
  std::ostringstream oss;
  (void)(int[]){0, (void(oss << std::forward<Args>(args)), 0)...};
  return oss.str();
};

#endif  // LIBKRIGING_UTILS_HPP
