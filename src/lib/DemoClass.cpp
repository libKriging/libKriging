#include "include/libKriging/DemoClass.hpp"

#include <typeinfo>

LIBKRIGING_EXPORT
DemoClass::DemoClass() = default;

LIBKRIGING_EXPORT
std::string DemoClass::name() {
  return typeid(this).name();
}

LIBKRIGING_EXPORT
int DemoClass::f() {
  return 42;
}