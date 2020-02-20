//
// Created by Pascal Havé on 2019-07-07.
//

#ifndef LIBKRIGING_DEMOARMADILLOCLASS_HPP
#define LIBKRIGING_DEMOARMADILLOCLASS_HPP

#include <armadillo>

#include "libKriging/libKriging_exports.h"

/** This is a demo class about about armadillo usage
 *  @ingroup Demo Demo group
 */
class DemoArmadilloClass {
 public:
  /** \deprecated Demo only */
  LIBKRIGING_EXPORT DemoArmadilloClass(std::string id, arma::Mat<double> M);
  /** \deprecated Demo only */
  LIBKRIGING_EXPORT ~DemoArmadilloClass();
  /** \deprecated Demo only */
  LIBKRIGING_EXPORT void test();
  /** \deprecated Demo only */
  LIBKRIGING_EXPORT arma::vec getEigenValues();

 private:
  const std::string m_id;
  const arma::Mat<double> m_m;
};

#endif  // LIBKRIGING_DEMOARMADILLOCLASS_HPP