//
// Created by Pascal Havé on 2019-07-07.
//

#ifndef LIBKRIGING_DEMOARMADILLOCLASS_HPP
#define LIBKRIGING_DEMOARMADILLOCLASS_HPP

#include <armadillo>
#include "libKriging_exports.h"

/** This is a demo class about about armadillo usage
 *  @ingroup Demo Demo group
 */
class DemoArmadilloClass {
 public:
  /** \deprecated Demo only */
  LIBKRIGING_EXPORT DemoArmadilloClass();
  /** \deprecated Demo only */
  static arma::vec getEigenValues(arma::mat M) { return arma::eig_sym(M); }
};

#endif  // LIBKRIGING_DEMOARMADILLOCLASS_HPP
