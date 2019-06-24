#include <iostream>

#include "libKriging/TestClass.h"

#include <memory>

int main() {
    std::cout << "libKriging class tests" << std::endl;

    std::unique_ptr<TestClass> x(new TestClass());
    std::cout << x->f() << std::endl;

    return 0;
}