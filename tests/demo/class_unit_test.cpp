#include <iostream>
#include <memory>

#include "libKriging/demo/DemoClass.hpp"

int main() {
  std::cout << "libKriging class tests" << std::endl;

  std::unique_ptr<DemoClass> x(new DemoClass());
  std::cout << x->name() << " " << x->f() << std::endl;

  return 0;
}