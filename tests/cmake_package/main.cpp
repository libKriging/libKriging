#include <iostream>

#include "libKriging/version.hpp"

int main() {
  std::cout << "libKriging version : v" << libKriging::version() << std::endl;
  return 0;
}
