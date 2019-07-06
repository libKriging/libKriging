#include "include/libKriging/TestClass.hpp"

#include <typeinfo>

LIBKRIGING_EXPORT
TestClass::TestClass() = default;

LIBKRIGING_EXPORT
std::string TestClass::name() {
  return typeid(this).name();
}

LIBKRIGING_EXPORT
int TestClass::f() {
  return 42;
}