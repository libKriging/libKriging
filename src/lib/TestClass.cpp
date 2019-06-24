#include "include/libKriging/TestClass.h"

#include <typeinfo>

LIBKRIGING_EXPORT
TestClass::
TestClass() = default;

LIBKRIGING_EXPORT
std::string
TestClass::
f() {
    return typeid(this).name();
}