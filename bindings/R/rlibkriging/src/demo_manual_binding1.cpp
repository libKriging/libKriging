/***************************************************************************
 R callable function
 ***************************************************************************/

#include <R.h>
#include <Rdefines.h>

// FIXME: collision with Rinternals
//#include <iostream>

#include <libKriging/demo/DemoClass.hpp>
#include <memory>

extern "C" SEXP demo_binding1() {
  //    std::cout << "libKriging class tests" << std::endl;
  printf("libKriging class tests\n");

  std::unique_ptr<DemoClass> x(new DemoClass());
  int tmp = [&x]() { return x->f(); }();

  SEXP result;

  PROTECT(result = NEW_INTEGER(2));
  INTEGER(result)[0] = (int)1;
  INTEGER(result)[1] = (int)tmp;
  UNPROTECT(1);

  return (result);
}
