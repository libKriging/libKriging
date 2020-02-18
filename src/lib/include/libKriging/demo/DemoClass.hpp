#ifndef LIBKRIGING_DEMOCLASS_HPP
#define LIBKRIGING_DEMOCLASS_HPP

#include <string>

#include "libKriging/libKriging_exports.h"

/** This is a demo class about about class usage
 *  @ingroup Demo Demo group
 */
class DemoClass {
 public:
  /** \deprecated Demo only */
  LIBKRIGING_EXPORT DemoClass();
  /** \deprecated Demo only */
  LIBKRIGING_EXPORT std::string name();
  /** \deprecated Demo only */
  LIBKRIGING_EXPORT int f();
};

#endif  // LIBKRIGING_DEMOCLASS_HPP
