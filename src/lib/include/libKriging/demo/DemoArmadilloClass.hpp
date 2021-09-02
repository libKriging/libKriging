//
// Created by Pascal Havé on 2019-07-07.
//

#ifndef LIBKRIGING_DEMOARMADILLOCLASS_HPP
#define LIBKRIGING_DEMOARMADILLOCLASS_HPP

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/libKriging_exports.h"

/** This is a demo class about about armadillo usage
 *  @ingroup Demo Demo group
 */
class DemoArmadilloClass {
 public:
  /** \deprecated Demo only */
  LIBKRIGING_EXPORT DemoArmadilloClass(arma::rowvec a);

  /** \deprecated Demo only */
  LIBKRIGING_EXPORT ~DemoArmadilloClass();

  /** \deprecated Demo only */
  LIBKRIGING_EXPORT arma::rowvec apply(const arma::rowvec& b) const;

 public:
  arma::rowvec m_a;
};

#endif  // LIBKRIGING_DEMOARMADILLOCLASS_HPP