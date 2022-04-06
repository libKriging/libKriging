#ifndef LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_VERSION_H
#define LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_VERSION_H

#include <string>
#include "libKriging/libKriging_exports.h"

namespace libKriging {
LIBKRIGING_EXPORT std::string version();
LIBKRIGING_EXPORT std::string buildTag();
}  // namespace libKriging

#endif  // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_VERSION_H
