#include "include/libKriging/TestClass.h"

TestClass::
TestClass() = default;

std::string
TestClass::
f() {
    return typeid(this).name();
}