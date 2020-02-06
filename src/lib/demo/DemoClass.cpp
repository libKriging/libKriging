#include "libKriging/demo/DemoClass.hpp"

#include <typeinfo>

LIBKRIGING_EXPORT
DemoClass::DemoClass() = default;

LIBKRIGING_EXPORT
std::string DemoClass::name() {
  return typeid(this).name();
}

LIBKRIGING_EXPORT
int DemoClass::f() {  // NOLINT(readability-convert-member-functions-to-static)
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
  return 42;
}