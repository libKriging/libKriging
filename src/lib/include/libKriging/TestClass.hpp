#ifndef LIBKRIGING_TESTCLASS_HPP
#define LIBKRIGING_TESTCLASS_HPP

#include <string>
#include "libKriging_exports.h"

class TestClass {
public:
    LIBKRIGING_EXPORT TestClass();
    LIBKRIGING_EXPORT std::string name();
    LIBKRIGING_EXPORT int f();
};

#endif //LIBKRIGING_TESTCLASS_HPP
