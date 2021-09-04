#include <memory>

#include <cassert>
#include "libKriging/demo/DemoArmadilloClass.hpp"

int main() {
  const int n = 40;
  arma::rowvec a(n, arma::fill::randn);
  arma::rowvec b(n, arma::fill::randn);
  std::unique_ptr<DemoArmadilloClass> cl(new DemoArmadilloClass(a));
  arma::rowvec result = cl->apply(b);
  arma::rowvec ref = a + b;
  assert(arma::norm(result - ref) < 1e-16);
}
