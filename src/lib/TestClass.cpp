#include "include/libKriging/TestClass.h"

#include <typeinfo>

TestClass::
TestClass() = default;

std::string
TestClass::
f() {
    return typeid(this).name();
}