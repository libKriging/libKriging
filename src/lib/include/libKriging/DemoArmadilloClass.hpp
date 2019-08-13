//
// Created by Pascal Havé on 2019-07-07.
//

#ifndef LIBKRIGING_DEMOARMADILLOCLASS_HPP
#define LIBKRIGING_DEMOARMADILLOCLASS_HPP

#include "libKriging_exports.h"
#include <armadillo>

class DemoArmadilloClass {
 public:
  LIBKRIGING_EXPORT DemoArmadilloClass();

  static arma::vec getEigenValues(arma::mat M) { return arma::eig_sym(M); }
};

#endif  // LIBKRIGING_DEMOARMADILLOCLASS_HPP
