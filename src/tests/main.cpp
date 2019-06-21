#include <iostream>

#include "libKriging/libKriging.h"

#include <memory>

int main() {
    std::cout << "libKriging tests" << std::endl;

    std::unique_ptr<TestClass> x(new TestClass());
    std::cout << x->f() << std::endl;

}