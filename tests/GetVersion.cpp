#include <cassert>
#include <iostream>

#include "libKriging/version.hpp"

int main() {
  std::cout << "libKriging version : v" << libKriging::version() << std::endl;
  std::cout << "libKriging build tag : " << libKriging::buildTag() << std::endl;
}
