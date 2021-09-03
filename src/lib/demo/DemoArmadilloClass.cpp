
#include "libKriging/demo/DemoArmadilloClass.hpp"

#include <iostream>

#include "libKriging/utils/lk_armadillo.hpp"

using namespace std;
using namespace arma;

// Armadillo documentation is available at:
// http://arma.sourceforge.net/docs.html

// NOTE:
// * the C++11 "auto" keyword is not recommended for use with Armadillo objects and functions
// * many NOLINT directives for clang-tidy since this a 'demo'

LIBKRIGING_EXPORT
DemoArmadilloClass::DemoArmadilloClass(arma::rowvec a) : m_a(std::move(a)) {
  std::cout << "Building DemoArmadilloClass size=" << m_a.n_elem << "\n";
}

LIBKRIGING_EXPORT
DemoArmadilloClass::~DemoArmadilloClass() {
  std::cout << "Destorying DemoArmadilloClass size=" << m_a.n_elem << "\n";
}

LIBKRIGING_EXPORT
arma::rowvec DemoArmadilloClass::apply(const rowvec& b) const {
  std::cout << "Apply DemoArmadilloClass size=" << m_a.n_elem << "\n";
  return m_a + b;
}
