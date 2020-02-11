#include <memory>

#include "libKriging/demo/DemoArmadilloClass.hpp"

int main() {
  const int n = 40;
  arma::mat X(n, n, arma::fill::randn);
  arma::mat Z = X* X.t();

  std::unique_ptr<DemoArmadilloClass> x(new DemoArmadilloClass("Z", Z));
  x->test();
  arma::vec ev = x->getEigenValues();

  std::cout << "Eigen vectors" << ev << std::endl;
}
