//
// Created by Pascal Havé on 2019-07-07.
//

#ifndef LIBKRIGING_ARMADILLOTESTCLASS_HPP
#define LIBKRIGING_ARMADILLOTESTCLASS_HPP

#include "libKriging_exports.h"
#include <armadillo>

class ArmadilloTestClass {
 public:
  LIBKRIGING_EXPORT ArmadilloTestClass();

  static arma::vec getEigenValues(arma::mat M) { return arma::eig_sym(M); }
};

#endif //LIBKRIGING_ARMADILLOTESTCLASS_HPP
