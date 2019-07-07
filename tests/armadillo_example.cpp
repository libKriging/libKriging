#include "libKriging/ArmadilloTestClass.hpp"
#include <memory>

int main() {
  std::unique_ptr<ArmadilloTestClass> x(new ArmadilloTestClass());
}
