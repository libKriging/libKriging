/***************************************************************************
 R callable function
 ***************************************************************************/

// Required since libKriging.hpp has signatures with armadillo types
#include <RcppArmadillo.h>

#include <R.h>
#include <Rdefines.h>

// FIXME: collision with Rinternals
//#include <iostream>

#include <memory>

#include <libKriging/demo/DemoArmadilloClass.hpp>

extern "C" SEXP demo_binding2() {

    //    std::cout << "libKriging class tests" << std::endl;
    printf("libKriging class tests\n");

    SEXP result;

    PROTECT(result = NEW_INTEGER(2));
    INTEGER(result)[0] = (int) 1;
    INTEGER(result)[1] = (int) 2;
    UNPROTECT(1);

    return (result);
}
