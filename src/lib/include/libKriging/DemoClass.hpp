#ifndef LIBKRIGING_DEMOCLASS_HPP
#define LIBKRIGING_DEMOCLASS_HPP

#include <string>
#include "libKriging_exports.h"

class DemoClass {
 public:
  LIBKRIGING_EXPORT DemoClass();
  LIBKRIGING_EXPORT std::string name();
  LIBKRIGING_EXPORT int f();
};

#endif  // LIBKRIGING_DEMOCLASS_HPP
