#include <memory>
#include "libKriging/DemoArmadilloClass.hpp"

int main() {
  std::unique_ptr<DemoArmadilloClass> x(new DemoArmadilloClass());
  x->test();

  const int n = 40;
  arma::mat X(n, n, arma::fill::randn);
  arma::mat Z = X* X.t();
  arma::vec ev = x->getEigenValues(Z);
  std::cout << "Eigen vectors" << ev << std::endl;
}
